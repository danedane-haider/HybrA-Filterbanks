import torch
import torch.nn as nn
import torch.nn.functional as F
from hybra.utils import fir_tightener3000, random_filterbank, kappa_alias

class HybrA(nn.Module):
    def __init__(self, path_to_auditory_filter_config, start_tight=True):
        super().__init__()
        
        config = torch.load(path_to_auditory_filter_config, weights_only=False, map_location="cpu")

        self.signal_length = 0
        self.audlet_real = config['auditory_filters_real'].clone().detach()
        self.audlet_imag = config['auditory_filters_imag'].clone().detach()
        self.audlet = self.audlet_real.squeeze(1) + 1j * self.audlet_imag.squeeze(1)
        self.audlet_stride = config['auditory_filters_stride']
        self.audlet_length = self.audlet_real.shape[-1]
        self.audlet_channels = config['n_filters']
        self.kernel_length = config['kernel_size']
        
        random_kernels = random_filterbank(N=self.audlet_length, J=1, T=self.kernel_length, norm=True, support_only=False)
        
        if start_tight:
#             random_kernels = fir_tightener4000(random_kernels.squeeze(1), self.encoder_length, 1,eps=1.1)
            random_kernels = fir_tightener3000(random_kernels, self.kernel_length, D=1, eps=1.01)
            random_kernels = torch.cat(self.audlet_channels * [random_kernels], dim=0)          

        self.kernels = nn.Parameter(random_kernels, requires_grad=True)
        self.filters = torch.empty(self.audlet.shape)
        self._filters = torch.empty(self.audlet.shape) # this is the one to optimize

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Forward pass of the HybridFilterbank.

        Parameters:
        -----------
        x (torch.Tensor) - input tensor of shape (batch_size, 1, signal_length)

        Returns:
        --------
        x (torch.Tensor) - output tensor of shape (batch_size, num_channels, signal_length//hop_length)
        """
        self.signal_length = x.shape[-1]
        x = F.pad(x, (0, x.shape[-1] % self.audlet_stride), mode='constant', value=0)
        hybra_kernels = torch.fft.ifft(
            torch.fft.fft(self.audlet.to(x.device).squeeze(1), dim=1) *
            torch.fft.fft(self.kernels.squeeze(1), dim=1),
            dim=1
            ).unsqueeze(1)
    
        self._filters = hybra_kernels
        self.filters = hybra_kernels.clone().detach()
        padding_length = self.filters.shape[-1] - 1

        output_real = F.conv1d(
            F.pad(x.unsqueeze(1), (padding_length, 0), mode='circular'),
            torch.fliplr(hybra_kernels.real),
            stride=self.audlet_stride,
        )

        output_imag = F.conv1d(
            F.pad(x.unsqueeze(1), (padding_length, 0), mode='circular'),
            torch.fliplr(hybra_kernels.imag),
            stride=self.audlet_stride,
        )

        return output_real + 1j * output_imag

    def encoder(self, x:torch.Tensor):
        """For learning use forward method

        """
        padding_length = self.filters.shape[-1] - 1

        return F.conv1d(
            F.pad(x.unsqueeze(1), (padding_length, 0), mode='circular'),
            torch.fliplr(self.filters.real.to(x.device)),
            stride = self.audlet_stride,
        ) + 1j * F.conv1d(
            F.pad(x.unsqueeze(1), (padding_length, 0), mode='circular'),
            torch.fliplr(self.filters.imag.to(x.device)),
            stride = self.audlet_stride,
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

        padded_signal_length = x.shape[-1] * self.audlet_stride
        padding_length = self.filters.shape[-1] - 1
        Ls = x.shape[-1] + padding_length
        kernel_size = self.filters.real.shape[-1]
        output_padding_length = int((padded_signal_length - kernel_size - (Ls-1) * self.audlet_stride) / -2) 
        
        x_real = x.real
        x_imag = x.imag

        x = (
            F.conv_transpose1d(
                F.pad(x_real, (0, padding_length), mode='circular'),
                torch.fliplr(self.filters.real),
                stride=self.audlet_stride,
                padding=output_padding_length
            )
            + F.conv_transpose1d(
                F.pad(x_imag, (0, padding_length), mode='circular'),
                torch.fliplr(self.filters.imag),
                stride=self.audlet_stride,
                padding=output_padding_length
            )
        )
        
        return torch.roll(2*self.audlet_stride * x.squeeze(1), output_padding_length - (kernel_size-1))[:, :self.signal_length]

    @property
    def condition_number(self):
        return float(kappa_alias(self.filters.squeeze(1), self.audlet_stride, aliasing=False))
