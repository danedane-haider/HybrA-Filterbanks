from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from hybra.utils import audfilters, circ_conv
from hybra.utils import plot_response as plot_response_
from hybra.utils import ISACgram as ISACgram_

class ISACSpec(nn.Module):
    """Constructor for an ISAC Mel spectrogram filterbank.
    Args:
        kernel_size (int) - size of the kernels of the auditory filterbank
        num_channels (int) - number of channels
        stride (int) - stride of the auditory filterbank. if 'None', stride is set to yield 25% overlap
        fc_max (float) - maximum frequency on the auditory scale. if 'None', it is set to fs//2.
        fmax (float) - maximum frequency computed.
        fs (int) - sampling frequency
        L (int) - signal length
        supp_mult (float) - support multiplier.
        scale (str) - auditory scale ('mel', 'erb', 'bark', 'log10', 'elelog'). elelog is a scale adapted to the hearing of elephants
        power (float) - power of the ISAC spectrogram
        avg_size (int) - size of the averaging kernel. if 'None', it is set to kernel_size / stride.
        is_log (bool) - whether to apply log to the output
        is_encoder_learnable (bool) - whether the encoder kernels are learnable
        is_avg_learnable (bool) - whether the averaging kernels are learnable
        verbose (bool) - whether to print information about the filterbank
    """
    def __init__(self,
                 kernel_size:Union[int,None]=None,
                 num_channels:int=40,
                 stride:Union[int,None]=None,
                 fc_max:Union[float,int,None]=None,
                 fmax:Union[int,None]=None,
                 fs:int=None, 
                 L:int=None,
                 supp_mult:float=1,
                 scale:str='mel',
                 power:float=2.0,
                 avg_size:int=None,
                 is_log=False,
                 is_encoder_learnable=False,
                 is_avg_learnable=False,
                 verbose:bool=True):
        super().__init__()

        [aud_kernels, d, fc, fc_min, fc_max, kernel_min, kernel_size, Ls, tsupp] = audfilters(
            kernel_size=kernel_size,num_channels=num_channels, fc_max=fc_max, fs=fs,L=L, supp_mult=supp_mult,scale=scale
        )

        if stride is not None:
            d = stride
            Ls = int(torch.ceil(torch.tensor(Ls / d)) * d)

        if verbose:
            print(f"Max. kernel size: {kernel_size}")
            print(f"Min. kernel size: {kernel_min}")
            print(f"Number of channels: {num_channels}")
            print(f"Stride for min. 25% overlap: {d}")
            print(f"Signal length: {Ls}")

        if fmax is not None:
            num_channels = torch.sum(fc <= fmax)
            aud_kernels = aud_kernels[:num_channels, :]

        self.num_channels = num_channels
        self.stride = d
        self.kernel_size = kernel_size
        self.kernel_min = kernel_min
        self.fs = fs
        self.fc = fc
        self.fc_min = fc_min
        self.fc_max = fc_max
        self.Ls = Ls
        self.is_log = is_log
        self.power = power

        if is_encoder_learnable:
            self.register_parameter('kernels', nn.Parameter(aud_kernels, requires_grad=True))
        else:
            self.register_buffer('kernels', aud_kernels)

        if avg_size is None:
            averaging_kernels = torch.ones(self.num_channels, 1, min(1024, self.kernel_size) // self.stride)
            # avg_size = torch.maximum(torch.tensor(2), tsupp // self.stride)
            # self.avg_max = avg_size.max().item()
            # averaging_kernels = torch.ones(self.num_channels, 1, self.avg_max)
            # for i in range(self.num_channels):
            #    averaging_kernels[i, 0, self.avg_max//2:self.avg_max//2+avg_size[i]] = 1.0
        else:
            averaging_kernels = torch.ones([self.num_channels,1,avg_size])

        if is_avg_learnable:
            self.register_parameter('avg_kernels', nn.Parameter(averaging_kernels, requires_grad=True))
        else:
            self.register_buffer('avg_kernels', averaging_kernels)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = circ_conv(x.unsqueeze(1), self.kernels, self.stride).abs()#**self.power
        x = F.conv1d(
            x,
            self.avg_kernels.to(x.device),
            groups=self.num_channels,
            stride=1,
            padding='same',
        )

        if self.is_log:
            x = torch.log(x + 1e-10)
        return x

    def ISACgram(self, x, fmax=None, vmin=None, log_scale=False):
        with torch.no_grad():
            coefficients = self.forward(x).abs()
        ISACgram_(c=coefficients, fc=self.fc, L=self.Ls, fs=self.fs, fmax=fmax, vmin=vmin, log_scale=log_scale)

    def plot_response(self):
        plot_response_(g=(self.kernels).detach().numpy(), fs=self.fs, scale=True, fc_min=self.fc_min, fc_max=self.fc_max, kernel_min=self.kernel_min)
