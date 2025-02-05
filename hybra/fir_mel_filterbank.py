import torch
import torch.nn as nn
import torch.nn.functional as F
from hybra.utils import audfilters_fir
from hybra.utils import plot_response as plot_response_

class AudSpec(nn.Module):
    def __init__(self, filterbank_config={'filter_len':256,
                                          'num_channels':42,
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
        self.num_channels = filterbank_config['num_channels']

        self.time_avg = self.filter_len // self.stride
        self.time_avg_stride = self.time_avg // 2

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
            torch.ones([self.num_channels,1,self.time_avg]).to(x.device),
            groups=self.num_channels,
            stride=self.time_avg_stride
        )

        return output

    def plot_response(self):
        plot_response_(g=(self.kernels_real + 1j*self.kernels_imag).detach().numpy(), fs=self.fs, scale=True, fc_crit=self.fc_crit)
