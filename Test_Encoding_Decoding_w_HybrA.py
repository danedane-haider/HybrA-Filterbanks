import os
import torch
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from hybra_v2 import HybrA
from hybra_v2.utils import calculate_condition_number 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

AUDIO = './filterbank_experiments/test_audio/signal.wav'
OUTPATH = './filterbank_experiments/reconst_audio_out'

hybra_fb = HybrA('./filterbank_experiments/saved_filterbanks/cqt.pth')


auditory_fb = (hybra_fb.auditory_filters_real + 1j*hybra_fb.auditory_filters_imag).squeeze()
nfft = auditory_fb.shape[-1]




hybra_fb = hybra_fb.to(device)

if False:
    learned_weights_real = torch.load("./filterbank_experiments/saved_filterbanks/HybrAModel200.pth", map_location=device)['model_state_dict']['filterbank.encoder_weight_real']
    learned_weights_imag = torch.load("./filterbank_experiments/saved_filterbanks/HybrAModel200.pth", map_location=device)['model_state_dict']['filterbank.encoder_weight_real']

    hybra_fb.encoder_weight_real = torch.nn.Parameter(learned_weights_real)
    hybra_fb.encoder_weight_imag = torch.nn.Parameter(learned_weights_imag)

audio, fs = sf.read(AUDIO)
#audio = torch.from_numpy(audio.astype('float32')).T.to(device)
#audio = torch.from_numpy(audio.astype('float32')).unsqueeze(0).to(device)
if len(audio.shape) == 1:
    audio = audio[...,None]

audio =  torch.tensor(audio[:5*fs,0], dtype=torch.float32)[None,...].to(device)
audio_filt = hybra_fb(audio)

reconst_audio = hybra_fb.decoder(audio_filt.real, audio_filt.imag)
# rms_audio = (audio**2).mean().sqrt()
# rms_reconst_audio = (reconst_audio**2).mean().sqrt()
# fact = rms_audio / rms_reconst_audio
# reconst_audio *= fact
#audio /= audio.abs().max()
#reconst_audio /= reconst_audio.abs().max()

#hybra_filters = hybra_fb.hybra_filters_real + 1j*hybra_fb.hybra_filters_imag

cond_number = calculate_condition_number(hybra_fb.hybra_filters.squeeze(1))
print(f"Condition number Hybra-Filter (Auditory-FB * Conv1D): {cond_number}")


print(f"Condition number Encoder Filters: {calculate_condition_number(hybra_fb.encoder_weight.squeeze(1))}")
#print(f"Condition number Encoder Filters (Imag): {calculate_condition_number(hybra_fb.encoder_weight.imag.squeeze(1))}")

print(f"Condition number Auditory Filters: {calculate_condition_number(hybra_fb.auditory_filterbank.squeeze(1))}")
#print(f"Condition number Auditory Filters (Imag): {calculate_condition_number(hybra_fb.auditory_filterbank.imag.squeeze(1))}")

sf.write(os.path.join(OUTPATH, 'reconst_audio.wav'), reconst_audio.detach().cpu().numpy().T, fs)



print('Scale-Factor:'+str(np.mean(audio[0].detach().cpu().numpy().squeeze()/reconst_audio.detach().cpu().numpy().squeeze())))
plt.figure()
plt.plot(audio[0].detach().cpu().numpy().T)
plt.plot(reconst_audio.detach().cpu().numpy().T)
plt.show()

print("top")