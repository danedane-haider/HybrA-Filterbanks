import torch
import torch.nn as nn
import torch.nn.functional as F
from hybra.utils import audfilters_fir 

class FIRFilterbank(nn.Module):

    def __init__(self, filterbank_config={'fs':16000, 'Ls':16000, 'fmin':0, 'fmax':None, 'spacing':1/2, 'bwmul':1, 'filter_len':480, 'redmul':1, 'scale':'erb'}, learnable=True):
        super().__init__()

        [filters,a,M,fc,L,fc_orig,fc_low,fc_high,ind_crit]  = audfilters_fir(**filterbank_config)

        self.filter_len = filterbank_config['filter_len'] 
        
        fir_kernels_real = torch.tensor(filters.real, dtype=torch.float32)
        fir_kernels_imag = torch.tensor(filters.imag, dtype=torch.float32)

        if learnable:
            self.register_parameter('fir_kernels_real', nn.Parameter(fir_kernels_real, requires_grad=True))
            self.register_buffer('fir_kernels_imag', nn.Parameter(fir_kernels_imag, requires_grad=True))
        else:
            self.register_buffer('fir_kernels_real', fir_kernels_real)
            self.register_buffer('fir_kernels_imag', fir_kernels_imag)

    def forward(self, x):
        out_real = F.conv1d(
            F.pad(x.unsqueeze(1), (self.filter_len//2, self.filter_len//2), mode='circular'),
            self.fir_kernels_real.to(x.device).unsqueeze(1),
        )
        out_imag = F.conv1d(
            F.pad(x.unsqueeze(1), (self.filter_len//2, self.filter_len//2), mode='circular'),
            self.fir_kernels_imag.to(x.device).unsqueeze(1),
        )

        return out_real + 1j * out_imag

