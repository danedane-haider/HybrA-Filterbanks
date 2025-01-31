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

        self.kernels_real = self.audlet_fir.kernels_real
        self.kernels_imag = self.audlet_fir.kernels_imag
        self.stride = self.audlet_fir.stride
        self.filter_len = filterbank_config['filter_len']
        self.num_channels = filterbank_config['num_channels']

        #self.linear_real = nn.Linear(self.num_channels, 1, bias=False)
        
        self.conv_real = nn.ConvTranspose1d(in_channels=1,
                                            out_channels=64,
                                            kernel_size=self.filter_len,
                                            stride=self.stride,
                                            padding=self.filter_len//2,
                                            bias=False,
                                            padding_mode='circular')
        
        self.conv_real.weight.data = self.kernels_real.unsqueeze(1).unsqueeze(1)
        
        self.conv_real = nn.ConvTranspose1d(in_channels=1,
                                            out_channels=64,
                                            kernel_size=self.filter_len,
                                            stride=self.stride,
                                            padding=self.filter_len//2,
                                            bias=False,
                                            padding_mode='circular')
        
        self.conv_real.weight.data = self.kernels_imag.unsqueeze(1).unsqueeze(1)

    def forward(self, x):
        x = self.audlet_fir(x)
        x_real = self.conv_real(x.real.unsqueeze(1))

        return x.squeeze(0)