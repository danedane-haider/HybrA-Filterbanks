# instantiate the model and train it

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import soundfile
import torch.optim as optim

from hybra import AudletFIR
from hybra import NeuroDual
from hybra import plot_response
from datasets import random_sweep, noise_uniform
from hybra._fit_neurodual import MSETight


model = NeuroDual(filterbank_config={'filter_len':128,
                                     'num_channels':64,
                                     'fs':16000,
                                     'Ls':32000,
                                     'bwmul':1},
                                     learnable=True)

model2 = NeuroDual(filterbank_config={'filter_len':128,
                                     'num_channels':64,
                                     'fs':16000,
                                     'Ls':16000,
                                     'bwmul':1},
                                     learnable=True)


device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

model.to(device)
model2.to(device)

optimizer = optim.Adam(model.parameters(), lr=5e-4)
optimizer2 = optim.Adam(model2.parameters(), lr=5e-4)
criterion = MSETight(beta=1e-8)
criterion2 = MSETight(beta=0)

losses = []
kappas = []
losses2 = []
kappas2 = []

#plot_response(model.kernels_real.squeeze().detach().cpu().numpy() + 1j * model.kernels_imag.squeeze().detach().cpu().numpy(), 16000)

model.train()

for i in range(500):
    optimizer.zero_grad()
    #input = random_sweep(1).unsqueeze(0).to(device)
    input = noise_uniform(2).unsqueeze(0).to(device)
    #input = torch.rand(1, 16000).to(device)
    target = input
    output = model(input)

    w_real = model.kernels_real.squeeze()
    w_imag = model.kernels_imag.squeeze()

    loss, loss_tight, kappa = criterion(output, target, w_real + 1j*w_imag)
    loss_tight.backward()
    optimizer.step()
    losses.append(loss.item())
    kappas.append(kappa)
    print(f'Loss: {loss.item()}')
    print(f'Kappa: {kappa}')

# model2.train()

# for i in range(500):
#     optimizer2.zero_grad()
#     #input = random_sweep(1).unsqueeze(0).to(device)
#     input = noise_uniform(1).unsqueeze(0).to(device)
#     #input = torch.rand(1, 16000).to(device)
#     target = input
#     output = model2(input)

#     w_real = model2.kernels_real.squeeze()
#     w_imag = model2.kernels_imag.squeeze()

#     loss, _, kappa = criterion2(output, target, w_real + 1j*w_imag)
#     loss.backward()
#     optimizer2.step()
#     losses2.append(loss.item())
#     kappas2.append(kappa)
#     print(f'Loss: {loss.item()}')
#     print(f'Kappa: {kappa}')

# make two subplots

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(losses, label='Loss')
#plt.plot(losses2, label='Loss2')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(kappas, label='Kappa')
#plt.plot(kappas2, label='Kappa2')
plt.legend()
plt.show()

plot_response(model.kernels_real.squeeze().detach().cpu().numpy() + 1j * model.kernels_imag.squeeze().detach().cpu().numpy(), 16000)



# model.eval()

# with torch.no_grad():
#     input, fs = soundfile.read('/Users/dani/Documents/Data/test.wav')
#     input = input[16000:48000, 0]
#     input = torch.tensor(input, dtype=torch.float32).unsqueeze(0).to(device)
#     output = model(input)

# soundfile.write('/Users/dani/Documents/Data/test_out.wav', output.squeeze().detach().cpu().numpy(), fs)