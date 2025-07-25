from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from hybra.utils import audfilters, condition_number, circ_conv, circ_conv_transpose, frame_bounds
from hybra.utils import plot_response as plot_response_
from hybra.utils import ISACgram as ISACgram_
from hybra._fit_dual import fit, tight

class ISAC(nn.Module):
    """Constructor for an ISAC filterbank.

    Args:
        kernel_size (int) - size of the kernels of the auditory filterbank
        num_channels (int) - number of channels
        fc_max (float) - maximum frequency on the auditory scale. if 'None', it is set to fs//2.
        stride (int) - stride of the auditory filterbank. if 'None', stride is set to yield 25% overlap
        fs (int) - sampling frequency
        L (int) - signal length
        supp_mult (float) - support multiplier.
        scale (str) - auditory scale ('mel', 'erb', 'bark', 'log10', 'elelog'). elelog is a scale adapted to the hearing of elephants
        tighten (bool) - whether to further tighten the filterbank
        is_encoder_learnable (bool) - whether the encoder kernels are learnable
        fir_decoder (bool) - computes an approximate perfect reconstruction decoder
        is_decoder_learnable (bool) - whether the decoder kernels are learnable
        verbose (bool) - whether to print information about the filterbank
    """
    def __init__(self,
                 kernel_size:Union[int,None]=128,
                 num_channels:int=40,
                 fc_max:Union[float,int,None]=None,
                 stride:Union[int,None]=None,
                 fs:int=16000, 
                 L:int=16000,
                 supp_mult:float=1,
                 scale:str='mel',
                 tighten=False,
                 is_encoder_learnable=False,
                 fit_decoder=False,
                 is_decoder_learnable=False,
                 verbose:bool=True):
        super().__init__()

        [aud_kernels, d, fc, fc_min, fc_max, kernel_min, kernel_size, Ls] = audfilters(
            kernel_size=kernel_size, num_channels=num_channels, fc_max=fc_max, fs=fs, L=L, supp_mult=supp_mult, scale=scale
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
            
        self.aud_kernels = aud_kernels
        self.kernel_size = kernel_size
        self.kernel_min = kernel_min
        self.fc = fc
        self.fc_min = fc_min
        self.fc_max = fc_max
        self.stride = d
        self.Ls = Ls
        self.fs = fs
        self.scale = scale
        self.fit_decoder = fit_decoder

        # optional preprocessing
        
        if tighten:
            aud_kernels = tight(aud_kernels, d, Ls, fs, fit_eps = 1.0001, max_iter = 1000)
        
        if fit_decoder:
            decoder_kernels = fit(aud_kernels.clone(), d, Ls, fs, decoder_fit_eps = 0.0001, max_iter = 10000)
        else:
            decoder_kernels = aud_kernels.clone()

        # set the parameters for the convolutional layers

        if is_encoder_learnable:
            self.register_buffer('kernels', nn.Parameter(aud_kernels, requires_grad=True))
        else:
            self.register_buffer('kernels', aud_kernels)

        if is_decoder_learnable:
            self.register_buffer('decoder_kernels', nn.Parameter(decoder_kernels, requires_grad=True))
        else:    
            self.register_buffer('decoder_kernels', decoder_kernels)

    def forward(self, x):
        return circ_conv(x.unsqueeze(1), self.kernels, self.stride)

    def decoder(self, x:torch.Tensor) -> torch.Tensor:
        _, B = frame_bounds(self.decoder_kernels, self.stride, self.Ls)
        return circ_conv_transpose(x, self.decoder_kernels / B, self.stride).squeeze(1)
    
    # plotting methods

    def ISACgram(self, x):
        with torch.no_grad():
            coefficients = self.forward(x)
        ISACgram_(coefficients, self.fc, self.Ls, self.fs)

    def plot_response(self):
        plot_response_(g=(self.kernels).cpu().detach().numpy(), fs=self.fs, scale=self.scale, plot_scale=True, fc_min=self.fc_min, fc_max=self.fc_max, kernel_min=self.kernel_min)

    def plot_decoder_response(self):
        plot_response_(g=(self.decoder_kernels).detach().cpu().numpy(), fs=self.fs, scale=self.scale, decoder=True)

    @property
    def condition_number(self):
        kernels = (self.kernels).squeeze()
        return condition_number(kernels, int(self.stride), self.Ls)
    
    @property
    def condition_number_decoder(self):
        kernels = (self.decoder_kernels).squeeze()
        return condition_number(kernels, int(self.stride), self.Ls)
