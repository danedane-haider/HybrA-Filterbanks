from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from hybra.utils import audfilters, condition_number, upsample, circ_conv, circ_conv_transpose
from hybra.utils import plot_response as plot_response_
from hybra.utils import ISACgram as ISACgram_
from hybra._fit_dual import fit, tight

class ISAC(nn.Module):
    def __init__(self,
                 kernel_size:Union[int,None]=128,
                 num_channels:int=40,
                 fc_max:Union[float,int,None]=None,
                 stride:int=None,
                 fs:int=16000, 
                 L:int=16000,
                 supp_mult:float=1,
                 scale:str='mel',
                 tighten=False,
                 is_encoder_learnable=False,
                 use_decoder=False,
                 is_decoder_learnable=False,
                 verbose:bool=True):
        super().__init__()

        [kernels, d, fc, fc_min, fc_max, kernel_min, kernel_size, Ls] = audfilters(
            kernel_size=kernel_size, num_channels=num_channels, fc_max=fc_max, fs=fs, L=L, supp_mult=supp_mult, scale=scale
        )

        if verbose:
            print(f"Max kernel size: {kernel_size}")
        if stride is not None:
            if stride > d:
                if verbose:
                    print(f"Warning: stride {stride} is larger than the optimal stride {d}, may affect condition number 🌪️.")
            d = stride
            Ls = int(torch.ceil(torch.tensor(L / d)) * d)
            if verbose:
                print(f"Output length: {Ls}")
        else:
            if verbose:
                print(f"Optimal stride: {d}\nOutput length: {Ls}")
            
        self.kernels = kernels
        self.kernel_size = kernel_size
        self.kernel_min = kernel_min
        self.fc = fc
        self.fc_min = fc_min
        self.fc_max = fc_max
        self.stride = d
        self.Ls = Ls
        self.fs = fs
        self.scale = scale
        
        if tighten:
            max_iter = 1000
            fit_eps = 1.01
            kernels, _ = tight(kernels, d, Ls, fs, fit_eps, max_iter)

        if is_encoder_learnable:
            self.register_buffer('kernels_complex', nn.Parameter(kernels, requires_grad=True))
        else:
            self.register_buffer('kernels_complex', kernels)
        
        self.use_decoder = use_decoder
        if use_decoder:
            max_iter = 1000 # TODO: should we do something like that?
            decoder_fit_eps = 1e-6
            decoder_kernels, _, _ = fit(kernels.clone(), d, Ls, fs, decoder_fit_eps, max_iter)

            if is_decoder_learnable:
                self.register_buffer('decoder_kernels_complex', nn.Parameter(decoder_kernels, requires_grad=True))
            else:    
                self.register_buffer('decoder_kernels_complex', decoder_kernels)

    def forward(self, x):
        """Filterbank analysis.
        Parameters:
        -----------
        x (torch.Tensor) - input tensor of shape (batch_size, signal_length)
        Returns:
        --------
        x (torch.Tensor) - output tensor of shape (batch_size, num_channels, signal_length//hop_length)
        """
        x = torch.fft.fft(x, self.Ls, dim=-1) * torch.fft.fft(self.kernels_complex, self.Ls, dim=-1).unsqueeze(0)
        x = torch.fft.ifft(x, self.Ls, dim=-1)
        x = x[:, :, ::self.stride]
        #x = torch.roll(input=x, shifts=-self.kernel_size.item() // 2, dim=-1)
        return x

    def decoder(self, x:torch.Tensor) -> torch.Tensor:
        """Filterbank synthesis.

        Parameters:
        -----------
        x (torch.Tensor) - input tensor of shape (batch_size, num_channels, signal_length//hop_length)

        Returns:
        --------
        x (torch.Tensor) - output tensor of shape (batch_size, signal_length)
        """

        x = upsample(x, self.stride)
        if x.shape[-1] != self.Ls:
            raise ValueError(f"Coefficients have the wrong length ({x.shape[-1]} instead of {self.Ls}).")
        x = torch.fft.fft(x, self.Ls, dim=-1) * torch.fft.fft(torch.fliplr(torch.conj(self.decoder_kernels_complex)), self.Ls, dim=-1).unsqueeze(0)
        x = torch.fft.ifft(x, self.Ls, dim=-1)
        x = torch.sum(x, dim=0)
        return x.squeeze(1)

    def plot_response(self):
        plot_response_(g=(self.kernels_complex).cpu().detach().numpy(), fs=self.fs, scale=self.scale, plot_scale=True, fc_min=self.fc_min, fc_max=self.fc_max, kernel_min=self.kernel_min)

    def plot_decoder_response(self):
        if self.use_decoder:
            plot_response_(g=(self.decoder_kernels_complex).detach().cpu().numpy(), fs=self.fs, scale=self.scale, decoder=True)
        else:
            raise NotImplementedError("No decoder configured")

    def ISACgram(self, x):
        with torch.no_grad():
            coefficients = self.forward(x)
        ISACgram_(coefficients, self.fc, self.Ls, self.fs)

    @property
    def condition_number(self):
        kernels = (self.kernels).squeeze()
        #kernels = F.pad(kernels, (0, self.Ls - kernels.shape[-1]), mode='constant', value=0)
        return condition_number(kernels, int(self.stride), self.Ls)
    
    @property
    def condition_number_decoder(self):
        kernels = (self.decoder_kernels_complex).squeeze()
        #kernels = F.pad(kernels, (0, self.Ls - kernels.shape[-1]), mode='constant', value=0)
        return condition_number(kernels, int(self.stride), self.Ls)