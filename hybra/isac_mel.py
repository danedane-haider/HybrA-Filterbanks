from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from hybra.utils import audfilters
from hybra.utils import plot_response as plot_response_
from hybra.utils import plot_coefficients as plot_coefficients_

class ISACMelSpectrogram(nn.Module):
    def __init__(self,
                 kernel_max:Union[int,None]=None,
                 num_channels:int=40,
                 fc_max:Union[float,int,None]=None,
                 fs:int=16000, 
                 L:int=16000,
                 bwmul:float=1,
                 scale:str='erb',
                 is_encoder_learnable=False,
                 is_averaging_kernel_learnable=False,
                 is_log=False):
        super().__init__()

        [kernels, d, fc, fc_min, fc_max, kernel_min, kernel_max, Ls] = audfilters(
            kernel_max=kernel_max,num_channels=num_channels, fc_max=fc_max, fs=fs,L=L,bwmul=bwmul,scale=scale
        )

        self.kernels = kernels
        self.stride = d
        self.kernel_max = kernel_max
        self.kernel_min = kernel_min
        self.fs = fs
        self.fc = fc
        self.fc_min = fc_min
        self.fc_max = fc_max
        self.num_channels = num_channels
        self.Ls = Ls

        self.time_avg = self.kernel_max // self.stride
        self.time_avg_stride = self.time_avg // 2

        kernels_real = kernels.real.to(torch.float32)
        kernels_imag = kernels.imag.to(torch.float32)

        self.is_log = is_log

        if is_encoder_learnable:
            self.register_parameter('kernels_real', nn.Parameter(kernels_real, requires_grad=True))
            self.register_parameter('kernels_imag', nn.Parameter(kernels_imag, requires_grad=True))
        else:
            self.register_buffer('kernels_real', kernels_real)
            self.register_buffer('kernels_imag', kernels_imag)

        if is_averaging_kernel_learnable:
            self.register_parameter('averaging_kernel', nn.Parameter(torch.ones([self.num_channels,1,self.time_avg]), requires_grad=True))
        else:
            self.register_buffer('averaging_kernel', torch.ones([self.num_channels,1,self.time_avg]))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = F.conv1d(
            F.pad(x, (self.kernel_max//2, self.kernel_max//2), mode='circular'),
            self.kernels_real.unsqueeze(1),
            stride=self.stride,
        )**2 + F.conv1d(
            F.pad(x, (self.kernel_max//2,self.kernel_max//2), mode='circular'),
            self.kernels_imag.unsqueeze(1),
            stride=self.stride,
        )**2
        output = F.conv1d(
            x,
            self.averaging_kernel.to(x.device),
            groups=self.num_channels,
            stride=self.time_avg_stride
        )

        if self.is_log:
            output = torch.log10(output)

        return output

    def plot_coefficients(self, x):
        with torch.no_grad():
            coefficients = self.forward(x)
        plot_coefficients_(coefficients, self.fc, self.Ls, self.fs)

    def plot_response(self):
        plot_response_(g=(self.kernels_real + 1j*self.kernels_imag).detach().numpy(), fs=self.fs, scale=True, fc_min=self.fc_min, fc_max=self.fc_max, kernel_min=self.kernel_max)
