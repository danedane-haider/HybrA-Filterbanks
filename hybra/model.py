import torch
import torch.nn as nn
import torch.nn.functional as F

from hybra.firfilterbank import AudletFIR

class NeuroDual(nn.Module):
    def __init__(self, filterbank_config={'fs':16000,
                                          'Ls':16000,
                                          'fmin':0,
                                          'fmax':None,
                                          'spacing':1,
                                          'bwmul':1,
                                          'filter_len':120,
                                          'redmul':1,
                                          'scale':'erb'},
                                          learnable=False):
        super().__init__()

        self.audlet_fir = AudletFIR(filterbank_config=filterbank_config, learnable=False)

        #self.audlet_fir_dual = 

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # linear layer from 32 channels to variable size output
        self.fc1 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.audlet_fir(x)
        x = self.conv1(x)


        return