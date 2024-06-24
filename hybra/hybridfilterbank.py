import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from hybra.utils import audfilters
import os

_current_dir: str = os.path.dirname(os.path.realpath(__file__))

class HybrA(nn.Module):

    def __init__(self, n_filters:int=256, filter_length:int=512, hop_length:int=128, f_scale:str='erb', sr:int=16000, kernel_size:int=11, pretrained:bool=True):
        super().__init__()

        if pretrained:
            with open(_current_dir+"/ressources/auditory_filters_example.npy", "rb") as f:
                self.auditory_filters = torch.tensor(np.load(f), dtype=torch.complex64)
        else:
            self.auditory_filters = audfilters(n_filters, filter_length, hop_length, f_scale, sr)
        
        self.auditory_filters_real = torch.flip(self.auditory_filters.real, dims=(1,))
        self.auditory_filters_imag = torch.flip(self.auditory_filters.imag, dims=(1,))

        self.conv1d_encoder = nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size, stride=1, padding='same', bias=False, groups=n_filters)
        self.conv1d_decoder = nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size, stride=1, padding='same', bias=False, groups=n_filters)

        self.n_filters = n_filters
        self.filter_length = filter_length
        self.hop_length = hop_length
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Forward pass of the HybridFilterbank.
        
        Parameters:
        -----------
        x (torch.Tensor) - input tensor of shape (batch_size, 1, signal_length)
        
        Returns:
        --------
        x (torch.Tensor) - output tensor of shape (batch_size, n_filters, signal_length//hop_length)
        """
        hybrid_encoder_real = self.conv1d_encoder(self.auditory_filters_real).unsqueeze(1)
        hybrid_encoder_imag = self.conv1d_encoder(self.auditory_filters_imag).unsqueeze(1)
        x_real = F.conv1d(x, hybrid_encoder_real, stride=self.hop_length, padding=0)
        x_imag = F.conv1d(x, hybrid_encoder_imag, stride=self.hop_length, padding=0)
        return x_real + 1j * x_imag
    
    def decoder(self, x:torch.Tensor) -> torch.Tensor:
        """Forward pass of the dual HybridFilterbank.

        Parameters:
        -----------
        x (torch.Tensor) - input tensor of shape (batch_size, n_filters, signal_length//hop_length)

        Returns:
        --------
        x (torch.Tensor) - output tensor of shape (batch_size, 1, signal_length)
        """
        hybrid_decoder_real = self.conv1d_decoder(self.auditory_filters_real).unsqueeze(1)
        hybrid_decoder_imag = self.conv1d_decoder(self.auditory_filters_imag).unsqueeze(1)
        x_real = F.conv_transpose1d(x.real, hybrid_decoder_real, stride=self.hop_length, padding=0)
        x_imag = F.conv_transpose1d(x.imag, hybrid_decoder_imag, stride=self.hop_length, padding=0)
        return self.hop_length * (x_real + 1j * x_imag)