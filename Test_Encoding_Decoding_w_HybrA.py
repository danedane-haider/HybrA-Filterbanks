import os
import torch
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from hybra import HybrA
from hybra.utils import calculate_condition_number 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

AUDIO = './audio/signal.wav'
OUTPATH = './audio'
FILTER_PATH = './filters/cqt.pth'
SIG_LEN = 5#s

audio, fs = sf.read(AUDIO)

# make mono
if len(audio.shape) > 1:
    audio = audio[:,0]

audio =  torch.tensor(audio[:SIG_LEN*fs], dtype=torch.float32)[None,...].to(device)

hybra_fb = HybrA(FILTER_PATH, SIG_LEN, fs)
hybra_fb = hybra_fb.to(device)
audio_enc = hybra_fb(audio)
audio_dec = hybra_fb.decoder(audio_enc.real, audio_enc.imag)


cond_number = calculate_condition_number(hybra_fb.hybra_filters.squeeze(1))
print(f"Condition number Hybra-Filter (Auditory-FB * Conv1D): {cond_number}")
print(f"Condition number Encoder Filters: {calculate_condition_number(hybra_fb.encoder_weight.squeeze(1))}")
print(f"Condition number Auditory Filters: {calculate_condition_number(hybra_fb.auditory_filterbank.squeeze(1))}")

sf.write(os.path.join(OUTPATH, 'reconst_audio.wav'), audio_dec.detach().cpu().numpy().T, fs)

#audio_dec = audio_dec[:,:80000].roll(-4095)

print('Scaling-Factor: '+str(np.mean(audio[0].detach().cpu().numpy().squeeze()/audio_dec.detach().cpu().numpy().squeeze())))
fig,ax = plt.subplots(nrows=2)
ax[0].plot(audio[0,...].detach().cpu().numpy(),label='original audio')
ax[0].plot(audio_dec[0,...].detach().cpu().numpy(),linestyle='--', label='reconstructed audio')
ax[0].legend()
ax[0].set_title('Original and Reconstructed Audio')
ax[1].plot(audio[0,...].detach().cpu().numpy().T-audio_dec[0,...].detach().cpu().numpy().T, label='original - reconstruction')
ax[1].legend()
ax[1].set_title('Amplitude Error')
plt.show()

print("stop")