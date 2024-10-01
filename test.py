from hybra import HybrA
import soundfile
import torch
from hybra.utils import kappa_alias
import matplotlib.pyplot as plt

audio, fs = soundfile.read('./audio/signal.wav')
SIGLEN = 5
sig_len = int(SIGLEN*fs)
hybra_fb = HybrA('./filters/audlet_0025.pth')

# make mono
if len(audio.shape) > 1:
    audio = audio[:,0]

print(hybra_fb._filters.shape)

audio =  torch.tensor(audio[:sig_len], dtype=torch.float32)[None,...]
audio_enc = hybra_fb(audio)
audio_dec = hybra_fb.decoder(audio_enc)

k, a = kappa_alias(hybra_fb.filters.squeeze(), D=hybra_fb.audlet_stride)

print('Kappa:', k, "Aliasing:", a )

plt.figure()
plt.plot(audio[0].clone().detach())
plt.plot(audio_dec[0].clone().detach())
plt.show()

plt.figure()
plt.plot(audio[0].clone().detach()-audio_dec[0].clone().detach())
plt.show()