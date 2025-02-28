# plot the scales from hybra.utils. freqtoaud

import numpy as np
import matplotlib.pyplot as plt
from hybra.utils import freqtoaud, fctobw
import torch

fs = 16000
L = 16000
num_channels = 40

# plt.plot(freqtoaud(torch.linspace(0, fs/2, num_channels), scale='erb').numpy() / freqtoaud(torch.tensor(fs//2), scale='erb').numpy(), label='ERB')
# plt.plot(freqtoaud(torch.linspace(0, fs/2, num_channels), scale='mel').numpy() / freqtoaud(torch.tensor(fs//2), scale='mel').numpy(), label='Mel')
# plt.plot(freqtoaud(torch.linspace(0, fs/2, num_channels), scale='bark').numpy() / freqtoaud(torch.tensor(fs//2), scale='bark').numpy(), label='Bark')
# plt.plot(freqtoaud(torch.linspace(0, fs/2, num_channels), scale='log10').numpy() / freqtoaud(torch.tensor(fs//2), scale='log10').numpy(), label='Log')
# plt.legend()
# plt.show()

plt.plot(fctobw(torch.linspace(0, fs/2, num_channels), scale='erb').numpy() / fctobw(torch.tensor(fs//2), scale='erb').numpy(), label='ERB')
plt.plot(fctobw(torch.linspace(0, fs/2, num_channels), scale='mel').numpy() / fctobw(torch.tensor(fs//2), scale='mel').numpy(), label='Mel')
plt.plot(fctobw(torch.linspace(0, fs/2, num_channels), scale='bark').numpy() / fctobw(torch.tensor(fs//2), scale='bark').numpy(), label='Bark')
plt.plot(fctobw(torch.linspace(0, fs/2, num_channels), scale='log10').numpy() / fctobw(torch.tensor(fs//2), scale='log10').numpy(), label='Log')
plt.legend()
plt.show()