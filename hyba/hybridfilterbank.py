import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
__current_dir = os.path.dirname(os.path.realpath(__file__))

class HybridFilterbank(nn.Module):

    def __init__(self, n_filters:int=256, filter_length:int=512, hop_length:int=128, frequency_scale:str='erb', sr:int=16000, kernel_size:int=31, stride:int=1, pretrained:bool=True):
        super().__init__()

        if pretrained:
            with open(f"{__current_dir}/ressources/auditory_filters_example.npy", "rb") as f:
                self.auditory_filters = torch.tensor(np.load(f), dtype=torch.complex64)
                self.auditory_filters = torch.flip(self.auditory_filters, dims=(1,))

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=kernel_size, stride=stride, padding_mode='circular', bias=False)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Forward pass of the HybridFilterbank.
        
        Parameters:
        -----------
        x (torch.Tensor) - input tensor of shape (batch_size, 1, signal_length)
        
        Returns:
        --------
        x (torch.Tensor) - output tensor of shape (batch_size, n_filters, signal_length//hop_length)
        """
        x = F.pad(x, (self.filter_length//2, self.filter_length//2), mode='circular')
        x = F.conv1d(input=x, weight=self.auditory_filters, bias=False, stride=self.hop_length)
        x = self.conv1d(x)
        return x
    
    def decoder(self, x:torch.Tensor) -> torch.Tensor:
        # circular pad
        x = F.pad(x, (self.n_filters//2, self.n_filters//2), mode='circular')
        x = F.conv_transpose1d(input=x, weight=self.auditory_filters, bias=False, stride=self.hop_length)
        return x