from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from hybra.utils import audfilters, condition_number
from hybra.utils import plot_response as plot_response_
from hybra.utils import plot_coefficients as plot_coefficients_
from hybra._fit_dual import fit

class ISAC(nn.Module):
    def __init__(self,
                 kernel_max:Union[int,None]=128,
                 num_channels:int=40,
                 fc_max:Union[float,int,None]=None,
                 fs:int=16000, 
                 L:int=16000,
                 bwmul:float=1,
                 scale:str='mel',
                 is_encoder_learnable=False,
                 use_decoder=False,
                 is_decoder_learnable=False):
        super().__init__()

        [kernels, d, fc, fc_min, fc_max, kernel_min, kernel_max, Ls] = audfilters(
            kernel_max=kernel_max,num_channels=num_channels, fc_max=fc_max, fs=fs,L=L,bwmul=bwmul,scale=scale
        )

        self.filters = kernels
        self.stride = d
        self.fc = fc
        self.fc_min = fc_min
        self.fc_max = fc_max
        self.kernel_min = kernel_min
        self.kernel_max = kernel_max
        self.Ls = Ls
        self.fs = fs
        self.scale = scale

        kernels_real = kernels.real.to(torch.float32)
        kernels_imag = kernels.imag.to(torch.float32)

        if is_encoder_learnable:
            self.register_parameter('kernels_real', nn.Parameter(kernels_real, requires_grad=True))
            self.register_parameter('kernels_imag', nn.Parameter(kernels_imag, requires_grad=True))
        else:
            self.register_buffer('kernels_real', kernels_real)
            self.register_buffer('kernels_imag', kernels_imag)
        
        self.use_decoder = use_decoder
        if use_decoder:
            max_iter = 1000 # TODO: should we do something like that?
            decoder_fit_eps = 1e-6
            decoder_kernels_real, decoder_kernels_imag, _, _ = fit(kernel_max=kernel_max,num_channels=num_channels, fc_max=fc_max, fs=fs,L=L,bwmul=bwmul,scale=scale, decoder_fit_eps=decoder_fit_eps, max_iter=max_iter)

            if is_decoder_learnable:
                self.register_parameter('decoder_kernels_real', nn.Parameter(decoder_kernels_real, requires_grad=True))
                self.register_parameter('decoder_kernels_imag', nn.Parameter(decoder_kernels_imag, requires_grad=True))
            else:        	
                self.register_buffer('decoder_kernels_real', decoder_kernels_real)
                self.register_buffer('decoder_kernels_imag', decoder_kernels_imag)

    def forward(self, x):
        x = F.pad(x.unsqueeze(1), (self.kernel_max//2, self.kernel_max//2), mode='circular')

        out_real = F.conv1d(x, self.kernels_real.to(x.device).unsqueeze(1), stride=self.stride)
        out_imag = F.conv1d(x, self.kernels_imag.to(x.device).unsqueeze(1), stride=self.stride)

        return out_real + 1j * out_imag

    def decoder(self, x_real:torch.Tensor, x_imag:torch.Tensor) -> torch.Tensor:
        """Filterbank synthesis.

        Parameters:
        -----------
        x (torch.Tensor) - input tensor of shape (batch_size, n_filters, signal_length//hop_length)

        Returns:
        --------
        x (torch.Tensor) - output tensor of shape (batch_size, signal_length)
        """
        L_in = x_real.shape[-1]
        L_out = self.Ls

        kernel_size = self.kernel_max
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
        filters = (self.kernels_real + 1j*self.kernels_imag).squeeze()
        filters = F.pad(filters, (0, self.Ls - filters.shape[-1]), mode='constant', value=0)
        return condition_number(filters, int(self.stride))
    
    @property
    def condition_number_decoder(self):
        filters = (self.decoder_kernels_real + 1j*self.decoder_kernels_imag).squeeze()
        filters = F.pad(filters, (0, self.Ls - filters.shape[-1]), mode='constant', value=0)
        return condition_number(filters, int(self.stride))