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
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=self.num_channels*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.filter_len, self.filter_len)),
        )

    def forward(self, x):
        x = self.audlet_fir(x)
        x_coeff_real = x.real
        x_coeff_imag = x.imag
        x = torch.stack((x_coeff_real, x_coeff_imag), dim=1)
        x = self.conv(x)
        x_real, x_imag = torch.split(x, self.num_channels, dim=1)

        # Reshape x_real and x_imag to [1, 64, 120, 120] for matrix multiplication
        x_real = x_real.view(1, self.num_channels, self.filter_len, self.filter_len)
        x_imag = x_imag.view(1, self.num_channels, self.filter_len, self.filter_len)

        dual_kernel_real = torch.zeros((1, self.num_channels, self.filter_len))
        dual_kernel_imag = torch.zeros((1, self.num_channels, self.filter_len))

        # Matrix multiplication for each channel
        for i in range(self.num_channels):
            dual_kernel_real[:, i, :] = torch.matmul(x_real[:, i, :, :], self.kernels_real.unsqueeze(0)[:, i, :].T).view(1, self.filter_len)
            dual_kernel_imag[:, i, :] = torch.matmul(x_imag[:, i, :, :], self.kernels_imag.unsqueeze(0)[:, i, :].T).view(1, self.filter_len)

        dual_kernel_real = dual_kernel_real.permute(1, 0, 2)
        dual_kernel_imag = dual_kernel_imag.permute(1, 0, 2)

        x = (
            F.conv_transpose1d(
                x_coeff_real,
                dual_kernel_real,
                stride=self.stride,
                padding=self.filter_len//2-2,
            )
            + F.conv_transpose1d(
                x_coeff_imag,
                dual_kernel_real,
                stride=self.stride,
                padding=self.filter_len//2-2,
            )
        )

        return x.squeeze(0)