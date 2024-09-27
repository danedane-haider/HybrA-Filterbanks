from hybra import HybrA
import soundfile
import torch
from hybra.utils import calculate_condition_number

audio, fs = soundfile.read('./audio/signal.wav')
SIGLEN = 5
hybra_fb = HybrA('./filters/auditory_filters_speech.pth',SIGLEN,fs)

# make mono
if len(audio.shape) > 1:
    audio = audio[:,0]

audio =  torch.tensor(audio[:SIGLEN*fs], dtype=torch.float32)[None,...]
audio_enc = hybra_fb(audio)

audio_dec = hybra_fb.decoder(audio_enc.real, audio_enc.imag)
print(audio_dec.shape)
#audio_dec = audio_dec[:,:SIGLEN*fs]

import matplotlib.pyplot as plt

plt.figure()
plt.plot(audio[0].clone().detach()-audio_dec[0].clone().detach())
plt.show()