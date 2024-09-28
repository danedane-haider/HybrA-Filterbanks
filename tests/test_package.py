import torch
import soundfile

from hybra import HybrA

def test_reconstruction():
    hybra_fb = HybrA('./filters/auditory_filters_speech.pth')
    audio, fs = soundfile.read('./audio/signal.wav')

    # make mono
    if len(audio.shape) > 1:
        audio = audio[:,0]

    audio =  torch.tensor(audio[:5*fs], dtype=torch.float32)[None,...]
    audio_enc = hybra_fb(audio)
    audio_dec = hybra_fb.decoder(audio_enc)

    assert torch.allclose(audio[0,...], audio_dec[0,...], atol=0.1)

def test_tightness():
    hybra_fb = HybrA('./filters/auditory_filters_speech.pth')
    audio, fs = soundfile.read('./audio/signal.wav')

    # make mono
    if len(audio.shape) > 1:
        audio = audio[:,0]

    audio =  torch.tensor(audio[:5*fs], dtype=torch.float32)[None,...]
    hybra_fb(audio)

    assert torch.allclose(torch.tensor(hybra_fb.condition_number), torch.tensor(1.), atol=0.1)

if __name__ == "__main__":
    test_reconstruction()
    test_tightness()