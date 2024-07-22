import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from hybra.utils import calculate_condition_number, fir_tightener3000

class HybrA(nn.Module):
    def __init__(self, path_to_auditory_filter_config, start_tight:bool=True):
        super().__init__()

        config = torch.load(path_to_auditory_filter_config)
        
        self.auditory_filters_real = torch.tensor(config['auditory_filters_real'])
        self.auditory_filters_imag = torch.tensor(config['auditory_filters_imag'])
        self.auditory_filters_stride = config['auditory_filters_stride']
        n_filters = config['n_filters']
        kernel_size = config['kernel_size']

        self.output_real_forward = None
        self.output_imag_forward = None

        self.encoder = nn.Conv1d(
             in_channels=n_filters,
             out_channels=n_filters,
             kernel_size=kernel_size,
             stride=1, padding='same',
             bias=False,
             groups=n_filters)
    
        if start_tight:
            self.encoder.weight = nn.Parameter(torch.tensor(fir_tightener3000(
                self.encoder.weight.squeeze(1).detach().numpy(), kernel_size, eps=1.01
            ),  dtype=torch.float32).unsqueeze(1))
        
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
            self.auditory_filters_real.to(x.device),
            stride=self.auditory_filters_stride,
            padding=0,
        )
        self.output_real_forward = output_real.clone().detach()
        output_imag = F.conv1d(
            x.unsqueeze(1),
            self.auditory_filters_imag.to(x.device),
            stride=self.auditory_filters_stride,
            padding=0,
        )
        self.output_imag_forward = output_imag.clone().detach()

        output_real = self.encoder(output_real)
        output_imag = self.encoder(output_imag)

        return output_real**2 + output_imag**2
    
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
                self.auditory_filters_real.to(x_real.device),
                stride=self.auditory_filters_stride,
                padding=0,
            )
            + F.conv_transpose1d(
                x_imag,
                self.auditory_filters_imag.to(x_imag.device),
                stride=self.auditory_filters_stride,
                padding=0,
            )
        )

        return x.squeeze(1)

    @property
    def condition_number(self):
        coefficients = self.encoder.weight.detach().clone().squeeze(1)
        return float(calculate_condition_number(coefficients))
