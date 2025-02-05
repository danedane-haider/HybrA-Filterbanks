import torch
import torch.nn as nn
import torch.nn.functional as F
from hybra.utils import audfilters_fir, fir_tightener3000
from hybra.utils import plot_response as plot_response_
from hybra._fit_neurodual import fit

class MelFIR(nn.Module):
    def __init__(self, filterbank_config={'filter_len':256,
                                          'num_channels':64,
                                          'fs':16000,
                                          'Ls':16000,
                                          'bwmul':1},
                                          is_encoder_learnable=False):
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

    def forward(self, x):
        x = F.conv1d(
            F.pad(x.unsqueeze(1), (self.filter_len//2, self.filter_len//2), mode='circular'),
            self.kernel_real,
            stride=self.stride,
        )**2 + F.conv1d(
            F.pad(x.unsqueeze(1), (self.filter_len//2,self.filter_len//2), mode='circular'),
            self.kernel_imag,
            stride=self.stride,
        )**2
        
        output = F.conv1d(
            x,
            self.kernels_real[0,:].to(x.device),
            groups=self.num_channels,
            padding="same",
        ).unsqueeze(1)

        return output

    def plot_response(self):
        plot_response_(g=(self.kernels_real + 1j*self.kernels_imag).detach().numpy(), fs=self.fs, scale=True, fc_crit=self.fc_crit)
