import torch
import torch.nn as nn
import torch.nn.functional as F
<<<<<<< HEAD
from hybra.utils import audfilters_fir
from hybra.utils import plot_response as plot_response_

class AudSpec(nn.Module):
    def __init__(self, filterbank_config={'filter_len':256,
                                          'num_channels':42,
=======
from hybra.utils import audfilters_fir, fir_tightener3000
from hybra.utils import plot_response as plot_response_
from hybra._fit_neurodual import fit

class MelFIR(nn.Module):
    def __init__(self, filterbank_config={'filter_len':256,
                                          'num_channels':64,
>>>>>>> 7745df7ebe5bf8a8cf4f13c888eea09b3592052f
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
<<<<<<< HEAD
        self.num_channels = filterbank_config['num_channels']
=======
>>>>>>> 7745df7ebe5bf8a8cf4f13c888eea09b3592052f

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
<<<<<<< HEAD
            F.pad(x, (self.filter_len//2, self.filter_len//2), mode='circular'),
            self.kernels_real.unsqueeze(1),
            stride=self.stride,
        )**2 + F.conv1d(
            F.pad(x, (self.filter_len//2,self.filter_len//2), mode='circular'),
            self.kernels_imag.unsqueeze(1),
=======
            F.pad(x.unsqueeze(1), (self.filter_len//2, self.filter_len//2), mode='circular'),
            self.kernel_real,
            stride=self.stride,
        )**2 + F.conv1d(
            F.pad(x.unsqueeze(1), (self.filter_len//2,self.filter_len//2), mode='circular'),
            self.kernel_imag,
>>>>>>> 7745df7ebe5bf8a8cf4f13c888eea09b3592052f
            stride=self.stride,
        )**2
        
        output = F.conv1d(
            x,
<<<<<<< HEAD
            self.kernels_real[0,:].repeat(self.num_channels,1).to(x.device).unsqueeze(1),
            groups=self.num_channels
        )
=======
            self.kernels_real[0,:].to(x.device),
            groups=self.num_channels,
            padding="same",
        ).unsqueeze(1)
>>>>>>> 7745df7ebe5bf8a8cf4f13c888eea09b3592052f

        return output

    def plot_response(self):
        plot_response_(g=(self.kernels_real + 1j*self.kernels_imag).detach().numpy(), fs=self.fs, scale=True, fc_crit=self.fc_crit)
