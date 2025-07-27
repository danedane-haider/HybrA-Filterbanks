from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from hybra.utils import audfilters, circ_conv
from hybra.utils import plot_response as plot_response_
from hybra.utils import ISACgram as ISACgram_

class MelSpec(nn.Module):
    """Constructor for an ISAC Mel spectrogram filterbank.
    Args:
        kernel_size (int) - size of the kernels of the auditory filterbank
        num_channels (int) - number of channels
        fc_max (float) - maximum frequency on the auditory scale. if 'None', it is set to fs//2.
        stride (int) - stride of the auditory filterbank. if 'None', stride is set to yield 25% overlap
        fs (int) - sampling frequency
        L (int) - signal length
        supp_mult (float) - support multiplier.
        scale (str) - auditory scale ('mel', 'erb', 'bark', 'log10', 'elelog'). elelog is a scale adapted to the hearing of elephants
        is_encoder_learnable (bool) - whether the encoder kernels are learnable
        is_averaging_kernel_learnable (bool) - whether the averaging kernel is learnable
        is_log (bool) - whether to apply log10 to the output
        verbose (bool) - whether to print information about the filterbank
    """
    def __init__(self,
                 kernel_size:Union[int,None]=None,
                 num_channels:int=40,
                 fc_max:Union[float,int,None]=None,
                 stride:Union[int,None]=None,
                 fs:int=16000, 
                 L:int=16000,
                 supp_mult:float=1,
                 scale:str='mel',
                 is_encoder_learnable=False,
                 is_averaging_kernel_learnable=False,
                 is_log=False,
                 verbose:bool=True):
        super().__init__()

        [aud_kernels, d, fc, fc_min, fc_max, kernel_min, kernel_size, Ls] = audfilters(
            kernel_size=kernel_size,num_channels=num_channels, fc_max=fc_max, fs=fs,L=L, supp_mult=supp_mult,scale=scale
        )

        if stride is not None:
            d = stride
            Ls = int(torch.ceil(torch.tensor(L / d)) * d)

        if verbose:
            print(f"Max. kernel size: {kernel_size}")
            print(f"Min. kernel size: {kernel_min}")
            print(f"Number of channels: {num_channels}")
            print(f"Stride for min. 25% overlap: {d}")
            print(f"Signal length: {Ls}")

        self.num_channels = num_channels
        self.stride = d
        self.kernel_size = kernel_size
        self.kernel_min = kernel_min
        self.fs = fs
        self.fc = fc
        self.fc_min = fc_min
        self.fc_max = fc_max
        self.Ls = Ls

        self.time_avg = int(self.kernel_size // self.stride)
        self.time_avg_stride = self.time_avg // 2

        self.is_log = is_log

        if is_encoder_learnable:
            self.register_parameter('kernels', nn.Parameter(aud_kernels, requires_grad=True))
        else:
            self.register_buffer('kernels', aud_kernels)

        if is_averaging_kernel_learnable:
            self.register_parameter('averaging_kernel', nn.Parameter(torch.ones([self.num_channels,1,self.time_avg], dtype=aud_kernels.dtype), requires_grad=True))
        else:
            self.register_buffer('averaging_kernel', torch.ones([self.num_channels,1,self.time_avg], dtype=aud_kernels.dtype))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = circ_conv(x.unsqueeze(1), self.kernels, self.stride)
        x = F.conv1d(
            x,
            self.averaging_kernel.to(x.device),
            groups=self.num_channels,
            stride=self.time_avg_stride
        )

        if self.is_log:
            x = torch.log10(x)

        return x

    def ISACgram(self, x):
        with torch.no_grad():
            coefficients = self.forward(x)
        ISACgram_(coefficients, self.fc, self.Ls, self.fs)

    def plot_response(self):
        plot_response_(g=(self.kernels).detach().numpy(), fs=self.fs, scale=True, fc_min=self.fc_min, fc_max=self.fc_max, kernel_min=self.kernel_min)
