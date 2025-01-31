# instantiate the model and train it

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from hybra import AudletFIR
from hybra import NeuroDual

from datasets import random_sweep

# plt.imshow(np.abs(output.squeeze().detach().numpy()), aspect='auto', origin='lower')
# plt.show()

model = NeuroDual(filterbank_config={'filter_len':128,
                                     'num_channels':64,
                                     'fs':16000,
                                     'Ls':16000,
                                     'bwmul':1},
                                     learnable=True)

# test_input = random_sweep(1).unsqueeze(0)

# output = model(test_input)
# print(test_input.shape, output.shape)

# make a training loop

import torch.optim as optim

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for i in range(100):
    optimizer.zero_grad()
    input = random_sweep(1).unsqueeze(0).to(device)
    target = input
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    print(f'Loss: {loss.item()}')