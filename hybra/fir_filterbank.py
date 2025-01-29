import torch
import torch.nn as nn
import torch.nn.functional as F
from hybra.utils import audfilters_fir 

class AudletFIR(nn.Module):
    def __init__(self, filterbank_config={'fs':16000,
                                          'Ls':16000,
                                          'fmin':0,
                                          'fmax':None,
                                          'spacing':1,
                                          'bwmul':1,
                                          'filter_len':120,
                                          'redmul':1,
                                          'scale':'erb'},
                                          learnable=True):
        super().__init__()

        [filters,a,M,fc,L,fc_orig,fc_low,fc_high,ind_crit] = audfilters_fir(**filterbank_config)

        self.filter_len = filterbank_config['filter_len'] 
        
        kernels_real = torch.tensor(filters.real, dtype=torch.float32)
        kernels_imag = torch.tensor(filters.imag, dtype=torch.float32)

        if learnable:
            self.register_parameter('kernels_real', nn.Parameter(kernels_real, requires_grad=True))
            self.register_parameter('kernels_imag', nn.Parameter(kernels_imag, requires_grad=True))
        else:
            self.register_buffer('kernels_real', kernels_real)
            self.register_buffer('kernels_imag', kernels_imag)

    def forward(self, x):
        x = F.pad(x.unsqueeze(1), (self.filter_len//2, self.filter_len//2), mode='circular')

        out_real = F.conv1d(x, self.kernels_real.to(x.device).unsqueeze(1))
        out_imag = F.conv1d(x, self.kernels_imag.to(x.device).unsqueeze(1))

        return out_real + 1j * out_imag

