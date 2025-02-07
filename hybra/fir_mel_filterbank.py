import torch
import torch.nn as nn
import torch.nn.functional as F
from hybra.utils import audfilters_fir
from hybra.utils import plot_response as plot_response_
from hybra.utils import plot_coefficients as plot_coefficients_

class ISACSpec(nn.Module):
    def __init__(self, filterbank_config={'filter_len':256,
                                          'num_channels':42,
                                          'fs':16000,
                                          'Ls':16000,
                                          'bwmul':1},
                                          is_encoder_learnable=False,
                                          is_averaging_kernel_learnable=False,
                                          is_log=False):
        super().__init__()

        [filters, d, fc, fc_crit, L] = audfilters_fir(**filterbank_config)

        self.filters = filters
        self.stride = d
        self.filter_len = filterbank_config['filter_len'] 
        self.fs = filterbank_config['fs']
        self.fc = fc
        self.fc_crit = fc_crit
        self.num_channels = filterbank_config['num_channels']
        self.Ls = filterbank_config['Ls']

        self.time_avg = self.filter_len // self.stride
        self.time_avg_stride = self.time_avg // 2

        kernels_real = torch.tensor(filters.real, dtype=torch.float32)
        kernels_imag = torch.tensor(filters.imag, dtype=torch.float32)

        self.is_log = is_log

        if is_encoder_learnable:
            self.register_parameter('kernels_real', nn.Parameter(kernels_real, requires_grad=True))
            self.register_parameter('kernels_imag', nn.Parameter(kernels_imag, requires_grad=True))
        else:
            self.register_buffer('kernels_real', kernels_real)
            self.register_buffer('kernels_imag', kernels_imag)

        if is_averaging_kernel_learnable:
            self.register_parameter('averaging_kernel', nn.Parameter(torch.ones([self.num_channels,1,self.time_avg]), requires_grad=True))
        else:
            self.register_buffer('averaging_kernel', torch.ones([self.num_channels,1,self.time_avg]))

    def forward(self, x):
        x = F.conv1d(
            F.pad(x, (self.filter_len//2, self.filter_len//2), mode='circular'),
            self.kernels_real.unsqueeze(1),
            stride=self.stride,
        )**2 + F.conv1d(
            F.pad(x, (self.filter_len//2,self.filter_len//2), mode='circular'),
            self.kernels_imag.unsqueeze(1),
            stride=self.stride,
        )**2
        output = F.conv1d(
            x,
            self.averaging_kernel.to(x.device),
            groups=self.num_channels,
            stride=self.time_avg_stride
        )

        if self.is_log:
            output = torch.log10(output)

        return output

    def plot_coefficients(self, x):
        with torch.no_grad():
            coefficients = self.forward(x)
        plot_coefficients_(coefficients, self.fc, self.Ls, self.fs)

    def plot_response(self):
        plot_response_(g=(self.kernels_real + 1j*self.kernels_imag).detach().numpy(), fs=self.fs, scale=True, fc_crit=self.fc_crit)
