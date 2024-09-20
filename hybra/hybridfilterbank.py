import torch
import torch.nn as nn
import torch.nn.functional as F
from hybra.utils import calculate_condition_number, fir_tightener3000, fir_tightener4000, random_filterbank

class HybrA(nn.Module):
    def __init__(self, path_to_auditory_filter_config, start_tight:bool=True):
        super().__init__()

        config = torch.load(path_to_auditory_filter_config, weights_only=False, map_location="cpu")

        self.auditory_filters_real = torch.tensor(config['auditory_filters_real'])
        self.auditory_filters_imag = torch.tensor(config['auditory_filters_imag'])
        self.auditory_filters_stride = 1#config['auditory_filters_stride']
        self.auditory_filter_length = self.auditory_filters_real.shape[-1]
        self.n_filters = config['n_filters']
        self.kernel_size = config['kernel_size']

        encoder_weight = random_filterbank(N=self.auditory_filter_length, J=self.n_filters, T=self.kernel_size)

        if start_tight:
            # TODO: Remove after corrected audlet
            auditory_filterbank = self.auditory_filters_real.squeeze(1)+ 1j*self.auditory_filters_imag.squeeze(1)
            self.auditory_filterbank = fir_tightener3000(auditory_filterbank, self.auditory_filter_length, eps=1.0001).unsqueeze(1)

            encoder_weight = fir_tightener4000(
                    encoder_weight.squeeze(1), self.kernel_size, eps=1.1
                ).unsqueeze(1)

        self.encoder_weight = nn.Parameter(encoder_weight, requires_grad=True)
        self.hybra_filters = torch.empty(self.auditory_filterbank.shape)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Forward pass of the HybridFilterbank.

        Parameters:
        -----------
        x (torch.Tensor) - input tensor of shape (batch_size, 1, signal_length)

        Returns:
        --------
        x (torch.Tensor) - output tensor of shape (batch_size, n_filters, signal_length//hop_length)
        """
        kernel = torch.fft.ifft(
            torch.fft.fft(self.auditory_filterbank.to(x.device).squeeze(1),dim=1) *
            torch.fft.fft(self.encoder_weight.squeeze(1), dim=1),
            dim=1
            ).unsqueeze(1)

        self.hybra_filters = kernel.clone().detach()

        padding_length = self.hybra_filters.shape[-1] - 1

        output_real = F.conv1d(
            F.pad(x.unsqueeze(1), (padding_length, 0), mode='circular'),
            kernel.real,
            stride=self.auditory_filters_stride,
        )

        output_imag = F.conv1d(
            F.pad(x.unsqueeze(1), (padding_length,0), mode='circular'),
            kernel.imag,
            stride=self.auditory_filters_stride,
        )

        out = output_real + 1j* output_imag

        return out

    def encoder(self, x:torch.Tensor):
        """For learning use forward method

        """
        padding_length = self.hybra_filters.shape[-1] - 1

        return F.conv1d(
            F.pad(x.unsqueeze(1), (padding_length, 0), mode='circular'),
            self.hybra_filters.real.to(x.device),
            stride=self.auditory_filters_stride,
        ) + 1j * F.conv1d(
            F.pad(x.unsqueeze(1), (padding_length,0), mode='circular'),
            self.hybra_filters.imag.to(x.device),
            stride=self.auditory_filters_stride,
        )

    def decoder(self, x_real:torch.Tensor, x_imag:torch.Tensor) -> torch.Tensor:
        """Forward pass of the dual HybridFilterbank.

        Parameters:
        -----------
        x (torch.Tensor) - input tensor of shape (batch_size, n_filters, signal_length//hop_length)

        Returns:
        --------
        x (torch.Tensor) - output tensor of shape (batch_size, signal_length)
        """
        padding_length = self.hybra_filters.shape[-1] - 1
        x = (
            F.conv_transpose1d(
                F.pad(x_real, (0,padding_length), mode='circular'),
                self.hybra_filters.real,
                stride=self.auditory_filters_stride,
                padding=padding_length
            )
            + F.conv_transpose1d(
                F.pad(x_imag, (0,padding_length), mode='circular'),
                self.hybra_filters.imag,
                stride=self.auditory_filters_stride,
                padding=padding_length
            )
        )

        return self.auditory_filters_stride * x.squeeze(1)

    @property
    def condition_number(self):
        return float(calculate_condition_number(self.hybra_filters.squeeze(1)))
