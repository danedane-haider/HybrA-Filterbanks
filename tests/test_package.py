import torch
import soundfile

from hybra import HybrA
from hybra.utils import calculate_condition_number

def test_reconstruction():
    hybra_fb = HybrA('./filters/auditory_filters_speech.pth')
    audio, fs = soundfile.read('./audio/signal.wav')

    # make mono
    if len(audio.shape) > 1:
        audio = audio[:,0]

    audio =  torch.tensor(audio[:5*fs], dtype=torch.float32)[None,...]
    audio_enc = hybra_fb(audio)
    audio_dec = hybra_fb.decoder(audio_enc.real, audio_enc.imag)

    assert torch.allclose(audio[0,...], audio_dec[0,...], atol=0.05)

def test_tightness():
    hybra_fb = HybrA('./filters/auditory_filters_speech.pth')
    audio, fs = soundfile.read('./audio/signal.wav')

    # make mono
    if len(audio.shape) > 1:
        audio = audio[:,0]

    audio =  torch.tensor(audio[:5*fs], dtype=torch.float32)[None,...]
    hybra_fb(audio)

    assert torch.allclose(calculate_condition_number(hybra_fb.hybra_filters.squeeze(1)), torch.tensor(1.), atol=0.1)
