import torch
import torch.nn as nn
import torch.nn.functional as F
from hybra.utils import audfilters_fir, fir_tightener3000
from hybra.utils import plot_response as plot_response_
from hybra._fit_neurodual import fit

class AudletFIR(nn.Module):
    def __init__(self, filterbank_config={'filter_len':256,
                                          'num_channels':64,
                                          'fs':16000,
                                          'Ls':16000,
                                          'bwmul':1},
                                          is_encoder_learnable=False,
                                          use_decoder=False,
                                          is_decoder_learnable=False,
                                          decoder_fit_eps=1e-5):
        super().__init__()

        [filters, d, fc, fc_crit, L] = audfilters_fir(**filterbank_config)

        self.filters = filters
        self.stride = d
        self.filter_len = filterbank_config['filter_len'] 
        self.fs = filterbank_config['fs']
        self.fc = fc
        self.fc_crit = fc_crit

        kernels_real = torch.tensor(filters.real, dtype=torch.float32)
        kernels_imag = torch.tensor(filters.imag, dtype=torch.float32)

        if is_encoder_learnable:
            self.register_parameter('kernels_real', nn.Parameter(kernels_real, requires_grad=True))
            self.register_parameter('kernels_imag', nn.Parameter(kernels_imag, requires_grad=True))
        else:
            self.register_buffer('kernels_real', kernels_real)
            self.register_buffer('kernels_imag', kernels_imag)
        
        self.use_decoder = use_decoder
        if use_decoder:
            max_iter = 1000 # TODO: should we do something like that?
            decoder_kernels_real, decoder_kernels_imag, _, _ = fit(filterbank_config, decoder_fit_eps, max_iter)

            if is_decoder_learnable:
                self.register_parameter('decoder_kernels_real', nn.Parameter(decoder_kernels_real, requires_grad=True))
                self.register_parameter('decoder_kernels_imag', nn.Parameter(decoder_kernels_imag, requires_grad=True))
            else:        	
                self.register_buffer('decoder_kernels_real', decoder_kernels_real)
                self.register_buffer('decoder_kernels_imag', decoder_kernels_imag)

    def forward(self, x):
        x = F.pad(x.unsqueeze(1), (self.filter_len//2, self.filter_len//2), mode='circular')

        out_real = F.conv1d(x, self.kernels_real.to(x.device).unsqueeze(1), stride=self.stride)
        out_imag = F.conv1d(x, self.kernels_imag.to(x.device).unsqueeze(1), stride=self.stride)

        return out_real + 1j * out_imag

    def decoder(self, x_real:torch.Tensor, x_imag:torch.Tensor) -> torch.Tensor:
        """Filterbank synthesis.

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
                self.decoder_kernels_real.to(x_real.device).unsqueeze(1),
                stride=self.stride,
                padding=self.filter_len//2,
                output_padding=self.stride - 2
            ) + F.conv_transpose1d(
                x_imag,
                self.decoder_kernels_imag.to(x_imag.device).unsqueeze(1),
                stride=self.stride,
                padding=self.filter_len//2,
                output_padding=self.stride - 2
            )
        )

        return x.squeeze(1)

    def plot_response(self):
        plot_response_(g=(self.kernels_real + 1j*self.kernels_imag).detach().numpy(), fs=self.fs, scale=True, fc_crit=self.fc_crit)

    def plot_decoder_response(self):
        if self.use_decoder:
            plot_response_(g=(self.decoder_kernels_real+1j*self.decoder_kernels_imag).detach().numpy(), fs=self.fs, decoder=True)
        else:
            raise NotImplementedError("No decoder configured")
