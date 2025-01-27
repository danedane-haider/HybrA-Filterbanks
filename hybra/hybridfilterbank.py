import torch
import torch.nn as nn
import torch.nn.functional as F
from hybra.utils import calculate_condition_number, fir_tightener3000, audfilters_fir

class HybrA(nn.Module):
    def __init__(self, filterbank_config={'fs':16000, 'Ls':16000, 'fmin':0, 'fmax':None, 'spacing':1/2, 'bwmul':1, 'filter_len':480, 'redmul':1, 'scale':'erb'}, start_tight:bool=True):
        super().__init__()
        [filters,a,M,fc,L,fc_orig,fc_low,fc_high,ind_crit]  = audfilters_fir(**filterbank_config)

        fir_kernels_real = torch.tensor(filters.real, dtype=torch.float32)
        fir_kernels_imag = torch.tensor(filters.imag, dtype=torch.float32)

        self.register_buffer('fir_kernels_real', fir_kernels_real)
        self.register_buffer('fir_kernels_imag', fir_kernels_imag)

        self.n_filters = fir_kernels_real.shape[0]
        self.fir_filter_len = filterbank_config['filter_len']
        self.fir_filter_stride = 1 #TODO: do something with that

        self.output_real_forward = None
        self.output_imag_forward = None

        k = torch.tensor(self.n_filters / (self.kernel_size * self.n_filters))
        encoder_weight = (-torch.sqrt(k) - torch.sqrt(k)) * torch.rand([self.n_filters, 1, self.kernel_size]) + torch.sqrt(k)

        if start_tight:
            encoder_weight = torch.tensor(fir_tightener3000(
                encoder_weight.squeeze(1).numpy(), self.kernel_size, eps=1.01
            ),  dtype=torch.float32).unsqueeze(1)
        
        self.encoder_weight_real = nn.Parameter(encoder_weight, requires_grad=True)
        self.encoder_weight_imag = nn.Parameter(encoder_weight, requires_grad=True)

        self.hybra_filters_real = torch.empty(1)
        self.hybra_filters_imag = torch.empty(1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Forward pass of the HybridFilterbank.
        
        Parameters:
        -----------
        x (torch.Tensor) - input tensor of shape (batch_size, 1, signal_length)
        
        Returns:
        --------
        x (torch.Tensor) - output tensor of shape (batch_size, n_filters, signal_length//hop_length)
        """

        kernel_real = F.conv1d(
            self.fir_kernels_real.to(x.device).squeeze(1),
            self.encoder_weight_real,
            groups=self.n_filters,
            padding="same",
        ).unsqueeze(1)
        self.hybra_filters_real = kernel_real.clone().detach()

        kernel_imag = F.conv1d(
            self.fir_kernels_imag.to(x.device).squeeze(1),
            self.encoder_weight_imag,
            groups=self.n_filters,
            padding="same",
        ).unsqueeze(1)
        self.hybra_filters_imag = kernel_imag.clone().detach()
        
        output_real = F.conv1d(
            F.pad(x.unsqueeze(1), (self.fir_filter_len//2, self.fir_filter_len//2), mode='circular'),
            kernel_real,
            stride=self.fir_fitler_stride,
        )
        
        output_imag = F.conv1d(
            F.pad(x.unsqueeze(1), (self.fir_filter_len//2,self.fir_filter_len//2), mode='circular'),
            kernel_imag,
            stride=self.fir_fitler_stride,
        )

        out = output_real + 1j* output_imag

        return out

    def encoder(self, x:torch.Tensor):
        """For learning use forward method

        """
        out = F.conv1d(
                    F.pad(x.unsqueeze(1),(self.fir_filter_len//2, self.fir_filter_len//2), mode='circular'),
                    self.hybra_filters_real.to(x.device),
                    stride=self.fir_fitler_stride,
                ) + 1j * F.conv1d(
                    F.pad(x.unsqueeze(1),(self.fir_filter_len//2, self.fir_filter_len//2), mode='circular'),
                    self.hybra_filters_imag.to(x.device),
                    stride=self.fir_fitler_stride,
                )
                
        return out
    
    def decoder(self, x_real:torch.Tensor, x_imag:torch.Tensor) -> torch.Tensor:
        """Forward pass of the dual HybridFilterbank.

        Parameters:
        -----------
        x (torch.Tensor) - input tensor of shape (batch_size, n_filters, signal_length//hop_length)

        Returns:
        --------
        x (torch.Tensor) - output tensor of shape (batch_size, signal_length)
        """
        x = (
            F.conv_transpose1d(
                x_real,
                self.hybra_filters_real,
                stride=self.fir_fitler_stride,
                padding=self.fir_filter_len//2,
            )
            + F.conv_transpose1d(
                x_imag,
                self.hybra_filters_imag,
                stride=self.fir_fitler_stride,
                padding=self.fir_filter_len//2,
            )
        )

        return x.squeeze(1)

    @property
    def condition_number(self):
        # coefficients = self.hybra_filters_real.detach().clone().squeeze(1) + 1j* self.hybra_filters_imag.detach().clone().squeeze(1)
        coefficients = self.encoder_weight_real.squeeze(1) + 1j*self.encoder_weight_imag.squeeze(1)
        return float(calculate_condition_number(coefficients))
