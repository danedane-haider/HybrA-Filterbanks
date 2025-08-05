from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import create_dct
from torchaudio.transforms import AmplitudeToDB

from hybra.utils import audfilters, circ_conv
from hybra.utils import plot_response as plot_response_
from hybra.utils import ISACgram as ISACgram_
from hybra import ISACSpec



class ISACCC(nn.Module):
    """Constructor for ISAC Cepstrum coefficients.
    Args:
        kernel_size (int) - size of the kernels of the auditory filterbank
        num_channels (int) - number of channels
        stride (int) - stride of the auditory filterbank. if 'None', stride is set to yield 25% overlap
        num_cc (int) - number of cepstrum coefficients
        fc_max (float) - maximum frequency on the auditory scale. if 'None', it is set to fs//2.
        fmax (float) - maximum frequency computed from the ISACSpec. if 'None', it is set to fs//2.
        fs (int) - sampling frequency
        L (int) - signal length
        supp_mult (float) - support multiplier.
        power (float) - power of ISACSpec
        scale (str) - auditory scale ('mel', 'erb', 'bark', 'log10', 'elelog'). elelog is a scale adapted to the hearing of elephants
        is_log (bool) - whether to apply log to the output
        verbose (bool) - whether to print information about the filterbank
    """
    def __init__(self,
                 kernel_size:Union[int,None]=None,
                 num_channels:int=40,
                 stride:Union[int,None]=None,
                 num_cc:int=13,
                 fc_max:Union[float,int,None]=None,
                 fmax:Union[float,int,None]=None,
                 fs:int=16000, 
                 L:int=16000,
                 supp_mult:float=1,
                 power:float=2.0,
                 scale:str='mel',
                 is_log:bool=False,
                 verbose:bool=True):
        super().__init__()

        self.isac = ISACSpec(
            kernel_size=kernel_size,
            num_channels=num_channels,
            stride=stride,
            fc_max=fc_max,
            fs=fs,
            L=L,
            supp_mult=supp_mult,
            power=power,
            scale=scale,
            is_log=False,
            verbose=verbose
        )

        self.fc_min = self.isac.fc_min
        self.fc_max = self.isac.fc_max
        self.kernel_min = self.isac.kernel_min
        self.fs = fs
        self.Ls = self.isac.Ls
        self.num_channels = num_channels
        self.num_cc = num_cc
        self.fmax = fmax
        self.is_log = is_log

        if self.num_cc > num_channels:
            raise ValueError("Cannot select more cepstrum coefficients than # channels")
        
        if self.fmax is not None:
            self.num_channels = torch.sum(self.isac.fc <= self.fmax)

        dct_mat = create_dct(self.num_cc, self.num_channels, norm='ortho').to(self.isac.kernels.device)
        self.register_buffer("dct_mat", dct_mat)

        self.amplitude_to_DB = AmplitudeToDB("power", 80.0)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        coeff = self.isac(x)
        if self.fmax is not None:
            coeff = coeff[:, :self.num_channels, :]
        if self.is_log:
            coeff = torch.log(coeff + 1e-10)
        else:
            coeff = self.amplitude_to_DB(coeff)
        return torch.matmul(coeff.transpose(-1, -2), self.dct_mat).transpose(-1, -2)

    def ISACgram(self, x):
        with torch.no_grad():
            coefficients = self.forward(x)
        ISACgram_(coefficients, None, self.Ls, self.fs)

    def plot_response(self):
        plot_response_(g=(self.isac.kernels[:self.num_channels, :]).detach().numpy(), fs=self.isac.fs, scale=True, fc_min=self.isac.fc_min, fc_max=self.isac.fc_max, kernel_min=self.isac.kernel_min)
