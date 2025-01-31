import torch
import torch.nn as nn
import torch.nn.functional as F

from hybra import AudletFIR

class NeuroDual(nn.Module):
    def __init__(self, filterbank_config={'filter_len':128,
                                          'num_channels':64,
                                          'fs':16000,
                                          'Ls':16000,
                                          'bwmul':1},
                                          learnable=True):
        super().__init__()

        self.audlet_fir = AudletFIR(filterbank_config=filterbank_config, learnable=False)
        self.stride = self.audlet_fir.stride
        self.filter_len = filterbank_config['filter_len']
        self.num_channels = filterbank_config['num_channels']

        kernel_real = torch.nn.functional.pad(torch.tensor(self.audlet_fir.filters.real, dtype=torch.float32), (0, 0))
        kernel_imag = torch.nn.functional.pad(torch.tensor(self.audlet_fir.filters.imag, dtype=torch.float32), (0, 0))
        
        self.register_parameter('kernels_real', nn.Parameter(kernel_real, requires_grad=True))
        self.register_parameter('kernels_imag', nn.Parameter(kernel_imag, requires_grad=True))

    def forward(self, x):
        x = self.audlet_fir(x)

        x = F.conv_transpose1d(
            x.real,
            self.kernels_real.to(x.real.device).unsqueeze(1),
            stride=self.stride,
            padding=self.filter_len//2,
            output_padding=self.stride-4
            ) + \
                F.conv_transpose1d(
                x.imag,
                self.kernels_imag.to(x.imag.device).unsqueeze(1),
                stride=self.stride,
                padding=self.filter_len//2,
                output_padding=self.stride-4
            )

        return x