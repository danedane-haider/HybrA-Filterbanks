from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from hybra.utils import audfilters, condition_number
from hybra.utils import plot_response as plot_response_
from hybra.utils import plot_coefficients as plot_coefficients_
from hybra._fit_dual import fit, tight

class ISAC(nn.Module):
    def __init__(self,
                 kernel_size:Union[int,None]=128,
                 num_channels:int=40,
                 fc_max:Union[float,int,None]=None,
                 stride:int=None,
                 fs:int=16000, 
                 L:int=16000,
                 bw_multiplier:float=1,
                 scale:str='erb',
                 tighten=False,
                 is_encoder_learnable=False,
                 use_decoder=False,
                 is_decoder_learnable=False,):
        super().__init__()

        [kernels, d, fc, fc_min, fc_max, kernel_min, kernel_size, Ls] = audfilters(
            kernel_size=kernel_size, num_channels=num_channels, fc_max=fc_max, fs=fs, L=L, bw_multiplier=bw_multiplier, scale=scale
        )

        if stride is not None:
            if stride > d:
                print(f"Using stride {stride} instead of the optimal {d} may affect the condition number 🌪️.")
            d = stride
            Ls = int(torch.ceil(torch.tensor(L / d)) * d)
            print(f"The output length is set to {Ls}.")
        else:
            print(f"The optimal stride is {d} and the output length is set to {Ls}.")
            
        self.kernels = kernels
        self.stride = d
        self.fc = fc
        self.fc_min = fc_min
        self.fc_max = fc_max
        self.kernel_min = kernel_min
        self.kernel_size = kernel_size
        self.Ls = Ls
        self.fs = fs
        self.scale = scale

        k_real = kernels.real.to(torch.float32)
        k_imag = kernels.imag.to(torch.float32)
        
        if tighten:
            max_iter = 1000
            fit_eps = 1.01
            k_real, k_imag, _ = tight(k_real+1j*k_imag, d, Ls, fs, fit_eps, max_iter)

        if is_encoder_learnable:
            self.register_parameter('kernels_real', nn.Parameter(k_real, requires_grad=True))
            self.register_parameter('kernels_imag', nn.Parameter(k_imag, requires_grad=True))
        else:
            self.register_buffer('kernels_real', k_real)
            self.register_buffer('kernels_imag', k_imag)
        
        self.use_decoder = use_decoder
        if use_decoder:
            max_iter = 1000 # TODO: should we do something like that?
            decoder_fit_eps = 1e-6
            decoder_kernels_real, decoder_kernels_imag, _, _ = fit(k_real+1j*k_imag, d, Ls, fs, decoder_fit_eps, max_iter)

            if is_decoder_learnable:
                self.register_parameter('decoder_kernels_real', nn.Parameter(decoder_kernels_real, requires_grad=True))
                self.register_parameter('decoder_kernels_imag', nn.Parameter(decoder_kernels_imag, requires_grad=True))
            else:        	
                self.register_buffer('decoder_kernels_real', decoder_kernels_real)
                self.register_buffer('decoder_kernels_imag', decoder_kernels_imag)

    def forward(self, x):
        x = F.pad(x.unsqueeze(1), (self.kernel_size//2, self.kernel_size//2), mode='circular')

        out_real = F.conv1d(x, self.kernels_real.to(x.device).unsqueeze(1), stride=self.stride)
        out_imag = F.conv1d(x, self.kernels_imag.to(x.device).unsqueeze(1), stride=self.stride)

        return out_real + 1j * out_imag

    def decoder(self, x_real:torch.Tensor, x_imag:torch.Tensor) -> torch.Tensor:
        """Filterbank synthesis.

        Parameters:
        -----------
        x (torch.Tensor) - input tensor of shape (batch_size, num_channels, signal_length//hop_length)

        Returns:
        --------
        x (torch.Tensor) - output tensor of shape (batch_size, signal_length)
        """
        L_in = x_real.shape[-1]
        L_out = self.Ls

        kernel_size = self.kernel_size
        padding = kernel_size // 2

        # L_out = (L_in -1) * stride - 2 * padding + dialation * (kernel_size - 1) + output_padding + 1 ; dialation = 1
        output_padding = L_out - (L_in - 1) * self.stride + 2 * padding - kernel_size
        
        x = (
            F.conv_transpose1d(
                x_real,
                self.decoder_kernels_real.to(x_real.device).unsqueeze(1),
                stride=self.stride,
                padding=padding,
                output_padding=output_padding
            ) + F.conv_transpose1d(
                x_imag,
                self.decoder_kernels_imag.to(x_imag.device).unsqueeze(1),
                stride=self.stride,
                padding=padding,
                output_padding=output_padding
            )
        )

        return x.squeeze(1)

    def plot_response(self):
        plot_response_(g=(self.kernels_real + 1j*self.kernels_imag).cpu().detach().numpy(), fs=self.fs, scale=self.scale, plot_scale=True, fc_min=self.fc_min, fc_max=self.fc_max, kernel_min=self.kernel_min)

    def plot_decoder_response(self):
        if self.use_decoder:
            plot_response_(g=(self.decoder_kernels_real+1j*self.decoder_kernels_imag).detach().cpu().numpy(), fs=self.fs, scale=self.scale, decoder=True)
        else:
            raise NotImplementedError("No decoder configured")

    def plot_coefficients(self, x):
        with torch.no_grad():
            coefficients = torch.log10(torch.abs(self.forward(x)[0]**2))
        plot_coefficients_(coefficients, self.fc, self.Ls, self.fs)

    @property
    def condition_number(self):
        kernels = (self.kernels_real + 1j*self.kernels_imag).squeeze()
        #kernels = F.pad(kernels, (0, self.Ls - kernels.shape[-1]), mode='constant', value=0)
        return condition_number(kernels, int(self.stride), self.Ls)
    
    @property
    def condition_number_decoder(self):
        kernels = (self.decoder_kernels_real + 1j*self.decoder_kernels_imag).squeeze()
        #kernels = F.pad(kernels, (0, self.Ls - kernels.shape[-1]), mode='constant', value=0)
        return condition_number(kernels, int(self.stride), self.Ls)