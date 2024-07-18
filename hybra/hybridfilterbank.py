import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from hybra.utils import audfilters, calculate_condition_number
import os

_current_dir: str = os.path.dirname(os.path.realpath(__file__))

class HybrA(nn.Module):
    def __init__(self, 
            n_filters:int=256,
            kernel_size:int=23,
            device=torch.device('cpu')):
        super().__init__()

        with open(_current_dir+"/ressources/auditory_filters_speech_256.npy", "rb") as f:
            auditory_filters = torch.tensor(np.load(f), dtype=torch.complex64)
        
        self.auditory_filters_real = auditory_filters.real.to(device).unsqueeze(1)
        self.auditory_filters_imag = auditory_filters.imag.to(device).unsqueeze(1)

        self.encoder = nn.Conv1d(
             in_channels=n_filters,
             out_channels=n_filters,
             kernel_size=kernel_size,
             stride=1, padding='same',
             bias=False,
             groups=n_filters)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Forward pass of the HybridFilterbank.
        
        Parameters:
        -----------
        x (torch.Tensor) - input tensor of shape (batch_size, 1, signal_length)
        
        Returns:
        --------
        x (torch.Tensor) - output tensor of shape (batch_size, n_filters, signal_length//hop_length)
        """
        output_real = F.conv1d(
            x.unsqueeze(1),
            self.auditory_filters_real,
            stride=1,
            padding=0,
        )
        self.output_real_forward = output_real.clone().detach()
        output_imag = F.conv1d(
            x.unsqueeze(1),
            self.auditory_filters_imag,
            stride=1,
            padding=0,
        )
        self.output_imag_forward = output_imag.clone().detach()

        output_real = self.encoder(output_real)
        output_imag = self.encoder(output_imag)

        return torch.log10(
            torch.max(
                output_real**2 + output_imag**2, 1e-8 * torch.ones_like(output_real)
            )
            )
    
    def decoder(self, x:torch.Tensor) -> torch.Tensor:
        """Forward pass of the dual HybridFilterbank.

        Parameters:
        -----------
        x (torch.Tensor) - input tensor of shape (batch_size, n_filters, signal_length//hop_length)

        Returns:
        --------
        x (torch.Tensor) - output tensor of shape (batch_size, signal_length)
        """
        x_real = x * self.output_real_forward
        x_imag = x * self.output_imag_forward
        x = (
            F.conv_transpose1d(
                x_real,
                self.auditory_filters_real,
                stride=1,
                padding=0,
            )
            + F.conv_transpose1d(
                x_imag,
                self.auditory_filters_imag,
                stride=1,
                padding=0,
            )
        )

        return x.squeeze(1), x_real + 1j * x_imag
    @property
    def condition_number(self):
        if self.random_dual_encoder:
            coefficients_real = self.encoder_real.weight.detach().clone().squeeze(1)
            coefficients_imag = self.encoder_imag.weight.detach().clone().squeeze(1)
            return float(calculate_condition_number(coefficients_real)), float(
                calculate_condition_number(coefficients_imag)
            )
        else:
            coefficients = self.encoder.weight.detach().clone().squeeze(1)
            return float(calculate_condition_number(coefficients))