from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from hybra.utils import condition_number, audfilters, plot_response
from hybra.utils import ISACgram as ISACgram_
from hybra._fit_dual import tight_hybra

class HybrA(nn.Module):
    def __init__(self,
                 kernel_size:int=128,
                 learned_kernel_size:int=23,
                 num_channels:int=40,
                 stride:int=None,
                 fc_max:Union[float,int]=None,
                 fs:int=16000, 
                 L:int=16000,
                 supp_mult:float=1,
                 scale:str='mel',
                 tighten:bool=False,
                 det_init:bool=False,
                 verbose:bool=True):
        """HybrA filterbank.

        Parameters:
        -----------
        kernel_size (int) - size of the kernels of the auditory filterbank
        learned_kernel_size (int) - size of the learned kernels
        num_channels (int) - number of channels
        stride (int) - stride of the auditory filterbank. if 'None', stride is set to yield 25% overlap
        fc_max (float) - maximum frequency on the auditory scale. if 'None', it is set to fs//2.
        fs (int) - sampling frequency
        L (int) - signal length
        supp_mult (float) - support multiplier. 
        scale (str) - auditory scale ('mel', 'erb', 'bark', 'log10', 'elelog'). elelog is a scale adapted to the hearing of elephants
        tighten (bool) - whether to tighten the hybrid filterbank
        det_init (bool) - whether to initialize the learned filters with diracs
        """
        
        super().__init__()

        [aud_kernels, d, fc, _, _, _, kernel_size, Ls] = audfilters(
            kernel_size=kernel_size, num_channels=num_channels, fc_max=fc_max, fs=fs, L=L, supp_mult=supp_mult, scale=scale
        )

        if stride is not None:
            d = stride
            Ls = int(torch.ceil(torch.tensor(L / d)) * d)

        if verbose:
            print(f"Max kernel size: {kernel_size}")
            if stride is not None and stride > 0:
                print(f"Warning: stride {stride} is larger than the optimal stride {d}, may affect condition number 🌪️.\nOutput length: {Ls}")
            else:
                print(f"Optimal stride: {d}\nOutput length: {Ls}")

        self.aud_kernels = aud_kernels
        self.kernel_size = kernel_size
        self.learned_kernel_size = learned_kernel_size
        self.stride = d
        self.num_channels = num_channels
        self.fc = fc
        self.Ls = Ls
        self.fs = fs

        self.aud_kernels_real = aud_kernels.real.to(torch.float32)
        self.aud_kernels_imag = aud_kernels.imag.to(torch.float32)

        self.register_buffer('kernels_real', self.aud_kernels_real)
        self.register_buffer('kernels_imag', self.aud_kernels_imag)
        self.output_real_forward = None
        self.output_imag_forward = None

        # Initialize learned kernels
        if det_init:
            learned_kernels = torch.zeros([self.num_channels, 1, self.learned_kernel_size])
            learned_kernels[:,0,0] = 1.0
        else:
            learned_kernels = torch.randn([self.num_channels, 1, self.learned_kernel_size])/torch.sqrt(torch.tensor(self.learned_kernel_size*self.num_channels))
            learned_kernels = learned_kernels / torch.norm(learned_kernels, p=1, dim=-1, keepdim=True)

        if tighten:
            max_iter = 1000
            fit_eps = 1.01
            learned_kernels_real, learned_kernels_imag, _ = tight_hybra(
                self.aud_kernels_real + 1j*self.aud_kernels_imag, learned_kernels, d, Ls, fs, fit_eps, max_iter)  
            self.learned_kernels_real = nn.Parameter(learned_kernels_real, requires_grad=True)
            self.learned_kernels_imag = nn.Parameter(learned_kernels_imag, requires_grad=True)
        else:
            self.learned_kernels_real = nn.Parameter(learned_kernels, requires_grad=True)
            self.learned_kernels_imag = nn.Parameter(learned_kernels, requires_grad=True)

        # compute the initial hybrid filters
        self.hybra_kernels_real = F.conv1d(
            self.aud_kernels_real.squeeze(1).to(self.learned_kernels_real.device),
            self.learned_kernels_real,
            groups=self.num_channels,
            padding="same",
        ).unsqueeze(1)
        self.hybra_kernels_imag = F.conv1d(
            self.aud_kernels_imag.squeeze(1).to(self.learned_kernels_imag.device),
            self.learned_kernels_imag,
            groups=self.num_channels,
            padding="same",
        ).unsqueeze(1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        kernel_real = F.conv1d(
            self.aud_kernels_real.to(x.device).squeeze(1),
            self.learned_kernels_real.to(x.device),
            groups=self.num_channels,
            padding="same",
        ).unsqueeze(1)
        self.hybra_kernels_real = kernel_real.clone().detach()

        kernel_imag = F.conv1d(
            self.aud_kernels_imag.to(x.device).squeeze(1),
            self.learned_kernels_imag.to(x.device),
            groups=self.num_channels,
            padding="same",
        ).unsqueeze(1)
        self.hybra_kernels_imag = kernel_imag.clone().detach()
        
        output_real = F.conv1d(
            F.pad(x.unsqueeze(1), (self.kernel_size//2, self.kernel_size//2), mode='circular'),
            kernel_real,
            stride=self.stride,
        )
        
        output_imag = F.conv1d(
            F.pad(x.unsqueeze(1), (self.kernel_size//2,self.kernel_size//2), mode='circular'),
            kernel_imag,
            stride=self.stride,
        )

        return output_real + 1j*output_imag

    def encoder(self, x:torch.Tensor):
        """For learning use forward method!

        """
        out = F.conv1d(
                    F.pad(x.unsqueeze(1),(self.kernel_size//2, self.kernel_size//2), mode='circular'),
                    self.hybra_kernels_real.to(x.device),
                    stride=self.stride,
                ) + 1j * F.conv1d(
                    F.pad(x.unsqueeze(1),(self.kernel_size//2, self.kernel_size//2), mode='circular'),
                    self.hybra_kernels_imag.to(x.device),
                    stride=self.stride,
                )
                
        return out
    
    def decoder(self, x_real:torch.Tensor, x_imag:torch.Tensor) -> torch.Tensor:
        L_in = x_real.shape[-1]
        L_out = self.Ls

        padding = self.kernel_size // 2

        # L_out = (L_in -1) * stride - 2 * padding + dialation * (kernel_size - 1) + output_padding + 1 ; dialation = 1
        output_padding = L_out - (L_in - 1) * self.stride + 2 * padding - self.kernel_size
        
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
    
    # plotting methods
    
    def ISACgram(self, x):
        with torch.no_grad():
            coefficients = torch.log10(torch.abs(self.forward(x)[0]**2))
        ISACgram_(coefficients, self.fc, self.Ls, self.fs)

    def plot_response(self):
        plot_response((self.hybra_kernels_real + 1j*self.hybra_kernels_imag).squeeze().cpu().detach().numpy(), self.fs)

    def plot_decoder_response(self):
        plot_response((self.hybra_kernels_real + 1j*self.hybra_kernels_imag).squeeze().cpu().detach().numpy(), self.fs, decoder=True)

    @property
    def condition_number(self, learnable:bool=False):
        kernels = (self.hybra_kernels_real + 1j*self.hybra_kernels_imag).squeeze()
        if learnable:
            return condition_number(kernels, self.stride, self.Ls)
        else:
            return condition_number(kernels, self.stride, self.Ls).item() 


