import torch
import torch.nn as nn
import torch.nn.functional as F
from hybra.utils import calculate_condition_number, fir_tightener3000, fir_tightener4000, random_filterbank

class HybrA(nn.Module):
    def __init__(self, path_to_auditory_filter_config, sig_len, fs, start_tight:bool=True):
        super().__init__()

        config = torch.load(path_to_auditory_filter_config, weights_only=False, map_location="cpu")



        self.auditory_filters_real = torch.tensor(config['auditory_filters_real'])
        self.auditory_filters_imag = torch.tensor(config['auditory_filters_imag'])
        self.auditory_filters_stride = 1#config['auditory_filters_stride']
        self.auditory_filter_length = self.auditory_filters_real.shape[-1]
        self.n_filters = config['n_filters']
        self.kernel_size = config['kernel_size']

        self.output_real_forward = None
        self.output_imag_forward = None

        encoder_weight_real = random_filterbank(N=sig_len*fs, J=self.n_filters, T=self.kernel_size)
        encoder_weight_imag = random_filterbank(N=sig_len*fs, J=self.n_filters, T=self.kernel_size)
        self.sig_len = sig_len*fs

        if start_tight:
            auditory_filterbank = self.auditory_filters_real.squeeze(1)+ 1j*self.auditory_filters_imag.squeeze(1)
            auditory_filterbank = F.pad(auditory_filterbank, (0,sig_len*fs-self.auditory_filter_length))
            auditory_filterbank = fir_tightener3000(auditory_filterbank, self.auditory_filter_length, eps=1.01)
            self.auditory_filterbank = auditory_filterbank[:,:self.auditory_filter_length].unsqueeze(1)

            encoder_weight = encoder_weight_real + 1j* encoder_weight_imag
            encoder_weight = fir_tightener4000(
                encoder_weight.squeeze(1), self.kernel_size, eps=1.1
            ).unsqueeze(1)
            encoder_weight = encoder_weight[...,:self.kernel_size]
        
        self.encoder_weight = nn.Parameter(encoder_weight, requires_grad=True)
        self.hybra_filters = torch.empty(1, dtype=torch.complex64)
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
        kernel = torch.fft.ifft(
            torch.fft.fft(self.auditory_filterbank.to(x.device).squeeze(1),dim=1) * 
            torch.fft.fft(F.pad(self.encoder_weight.squeeze(1), (0,self.auditory_filter_length-self.kernel_size)), 
            dim=1)
            ).unsqueeze(1)
        self.hybra_filters = kernel.clone().detach()
        kernel_real = kernel.real

        self.hybra_filters_real = kernel_real.clone().detach()

        kernel_imag = kernel.imag

        self.hybra_filters_imag = kernel_imag.clone().detach()
        
        output_real = F.conv1d(
            F.pad(x.unsqueeze(1), (self.auditory_filter_length-1, 0), mode='circular'),
            kernel_real,
            stride=self.auditory_filters_stride,
        )
        
        output_imag = F.conv1d(
            F.pad(x.unsqueeze(1), (self.auditory_filter_length-1,0), mode='circular'),
            kernel_imag,
            stride=self.auditory_filters_stride,
        )

        out = output_real + 1j* output_imag

        return out

    def encoder(self, x:torch.Tensor):
        """For learning use forward method

        """
        out = F.conv1d(
                    F.pad(x.unsqueeze(1),(self.auditory_filter_length//2, self.auditory_filter_length//2), mode='circular'),
                    self.hybra_filters_real.to(x.device),
                    stride=self.auditory_filters_stride,
                ) + 1j * F.conv1d(
                    F.pad(x.unsqueeze(1),(self.auditory_filter_length//2, self.auditory_filter_length//2), mode='circular'),
                    self.hybra_filters_imag.to(x.device),
                    stride=self.auditory_filters_stride,
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
        Lin = x_real.shape[-1]
        kernel_size = self.hybra_filters_real.shape[-1]
        padding_length = kernel_size-1
        x = (
            F.conv_transpose1d(
                F.pad(x_real, (0,kernel_size-1), mode='circular'),
                self.hybra_filters_real,
                stride=self.auditory_filters_stride,
                padding=padding_length
            )
            + F.conv_transpose1d(
                F.pad(x_imag, (0,kernel_size-1), mode='circular'),
                self.hybra_filters_imag,
                stride=self.auditory_filters_stride,
                padding=padding_length
            )
        )

        return x.squeeze(1)

    @property
    def condition_number(self):
        coefficients = self.encoder_weight_real.squeeze(1) + 1j*self.encoder_weight_imag.squeeze(1)
        return float(calculate_condition_number(coefficients))
