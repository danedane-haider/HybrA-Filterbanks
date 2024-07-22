import torch
import torch.nn as nn
from hybra import HybrA

class NSNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_before = nn.Linear(256, 400)

        self.gru = nn.GRU(
            input_size=400,
            hidden_size=400,
            num_layers=2,
            batch_first=True,
        )

        self.linear_after = nn.Linear(400, 600)
        self.linear_after2 = nn.Linear(600, 600)
        self.linear_after3 = nn.Linear(600, 256)


    def forward(self, x):

        x = x.permute(0, 2, 1)
        x = torch.relu(self.linear_before(x))
        x, _ = self.gru(x)
        x = torch.relu(self.linear_after(x))
        x = torch.relu(self.linear_after2(x))
        x = torch.sigmoid(self.linear_after3(x))
        x = x.permute(0, 2, 1)

        return x
    
class HybridfilterbankModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.filterbank = HybrA('auditory_filters_speech.pth')
        self.nsnet = NSNet()

    def forward(self, x):
        x = self.filterbank(x)
        x = torch.log10(
            torch.max(x, 1e-8 * torch.ones_like(x)))
        x = self.nsnet(x)
        x_coefficiants = x * self.filterbank.output_real_forward +1j * x * self.filterbank.output_imag_forward
        return self.filterbank.decoder(x), x_coefficiants
