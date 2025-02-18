from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from hybra.utils import condition_number, fir_tightener3000, audfilters, plot_response
from hybra.utils import plot_coefficients as plot_coefficients_

class HybrA(nn.Module):
    def __init__(self,
                 kernel_max:Union[int,None]=128,
                 num_channels:int=40,
                 fc_max:Union[float,int,None]=None,
                 fs:int=16000, 
                 L:int=16000,
                 bwmul:float=1,
                 scale:str='mel',
                 learned_kernel_size:int=16,
                 start_tight:bool=True):
        
        super().__init__()

        [kernels, d, fc, _, _, _, kernel_max, Ls] = audfilters(
            kernel_max=kernel_max,num_channels=num_channels, fc_max=fc_max, fs=fs,L=L,bwmul=bwmul,scale=scale
        )

        self.aud_kernels = kernels
        self.stride = d
        self.fc = fc
        self.kernel_max = kernel_max
        self.Ls = Ls
        self.fs = fs
        self.num_channels = num_channels
        self.learned_kernel_size = learned_kernel_size

        self.aud_kernels_real = kernels.real.to(torch.float32)
        self.aud_kernels_imag = kernels.imag.to(torch.float32)

        self.register_buffer('kernels_real', self.aud_kernels_real)
        self.register_buffer('kernels_imag', self.aud_kernels_imag)
        self.output_real_forward = None
        self.output_imag_forward = None

        # Initialize trainable filters
        k = torch.tensor(self.num_channels / (self.learned_kernel_size * self.num_channels))
        learned_kernels = (-torch.sqrt(k) - torch.sqrt(k)) * torch.rand([self.num_channels, 1, self.learned_kernel_size]) + torch.sqrt(k)

        if start_tight:
            learned_kernels = fir_tightener3000(
                learned_kernels.squeeze(1), self.learned_kernel_size, D=d, eps=1.01, Ls=self.learned_kernel_size * d
            ).to(torch.float32).unsqueeze(1)
            #learned_kernels = learned_kernels / torch.norm(learned_kernels, dim=-1, keepdim=True)
        
        self.learned_kernels_real = nn.Parameter(learned_kernels, requires_grad=True)
        self.learned_kernels_imag = nn.Parameter(learned_kernels, requires_grad=True)

        # compute the initial hybrid filters
        self.hybra_kernels_real = F.conv1d(
            self.aud_kernels_real.squeeze(1),
            self.learned_kernels_real,
            groups=self.num_channels,
            padding="same",
        ).unsqueeze(1)
        self.hybra_kernels_imag = F.conv1d(
            self.aud_kernels_imag.squeeze(1),
            self.learned_kernels_imag,
            groups=self.num_channels,
            padding="same",
        ).unsqueeze(1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Forward pass of the HybridFilterbank.
        
        Parameters:
        -----------
        x (torch.Tensor) - input tensor of shape (batch_size, 1, signal_length)
        
        Returns:
        --------
        x (torch.Tensor) - output tensor of shape (batch_size, num_channels, signal_length//hop_length)
        """

        kernel_real = F.conv1d(
            self.aud_kernels_real.to(x.device).squeeze(1),
            self.learned_kernels_real,
            groups=self.num_channels,
            padding="same",
        ).unsqueeze(1)
        self.hybra_kernels_real = kernel_real.clone().detach()

        kernel_imag = F.conv1d(
            self.aud_kernels_imag.to(x.device).squeeze(1),
            self.learned_kernels_imag,
            groups=self.num_channels,
            padding="same",
        ).unsqueeze(1)
        self.hybra_kernels_imag = kernel_imag.clone().detach()
        
        output_real = F.conv1d(
            F.pad(x.unsqueeze(1), (self.kernel_max//2, self.kernel_max//2), mode='circular'),
            kernel_real,
            stride=self.stride,
        )
        
        output_imag = F.conv1d(
            F.pad(x.unsqueeze(1), (self.kernel_max//2,self.kernel_max//2), mode='circular'),
            kernel_imag,
            stride=self.stride,
        )

        return output_real + 1j*output_imag

    def encoder(self, x:torch.Tensor):
        """For learning use forward method!

        """
        out = F.conv1d(
                    F.pad(x.unsqueeze(1),(self.kernel_max//2, self.kernel_max//2), mode='circular'),
                    self.hybra_kernels_real.to(x.device),
                    stride=self.stride,
                ) + 1j * F.conv1d(
                    F.pad(x.unsqueeze(1),(self.kernel_max//2, self.kernel_max//2), mode='circular'),
                    self.hybra_kernels_imag.to(x.device),
                    stride=self.stride,
                )
                
        return out
    
    def decoder(self, x_real:torch.Tensor, x_imag:torch.Tensor) -> torch.Tensor:
        """Forward pass of the dual HybridFilterbank.

        Parameters:
        -----------
        x (torch.Tensor) - input tensor of shape (batch_size, num_channels, signal_length//hop_length)

        Returns:
        --------
        x (torch.Tensor) - output tensor of shape (batch_size, signal_length)
        """
        L_in = x_real.shape[-1]
        L_out = self.Ls

        padding = self.kernel_max // 2

        # L_out = (L_in -1) * stride - 2 * padding + dialation * (kernel_size - 1) + output_padding + 1 ; dialation = 1
        output_padding = L_out - (L_in - 1) * self.stride + 2 * padding - self.kernel_max
        
        x = (
            F.conv_transpose1d(
                x_real,
                self.hybra_kernels_real.to(x_real.device),
                stride=self.stride,
                padding=padding,
                output_padding=output_padding
            ) + F.conv_transpose1d(
                x_imag,
                self.hybra_kernels_imag.to(x_imag.device),
                stride=self.stride,
                padding=padding,
                output_padding=output_padding
            )
        )

        return x.squeeze(1)

    @property
    def condition_number(self, learnable:bool=False):
        # coefficients = self.hybra_kernels_real.detach().clone().squeeze(1) + 1j* self.hybra_kernels_imag.detach().clone().squeeze(1)
        kernels = (self.hybra_kernels_real + 1j*self.hybra_kernels_imag).squeeze()
        kernels = F.pad(kernels, (0, self.Ls - kernels.shape[-1]), mode='constant', value=0)
        if learnable:
            return condition_number(kernels, self.stride)
        else:
            return condition_number(kernels, self.stride).item()    
    def plot_response(self):
        plot_response((self.hybra_kernels_real + 1j*self.hybra_kernels_imag).squeeze().detach().numpy(), self.fs)
    def plot_decoder_response(self):
        plot_response((self.hybra_kernels_real + 1j*self.hybra_kernels_imag).squeeze().detach().numpy(), self.fs, decoder=True)
    def plot_coefficients(self, x):
        with torch.no_grad():
            coefficients = torch.log10(torch.abs(self.forward(x)[0]**2))
        plot_coefficients_(coefficients, self.fc, self.Ls, self.fs)
