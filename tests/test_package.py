import torch
import soundfile

from hybra import HybrA
from hybra.utils import calculate_condition_number

def test_reconstruction():
    AUDIO = './audio/signal.wav'
    FILTER_PATH = './filters/cqt.pth'
    SIG_LEN = 5#s

    audio, fs = soundfile.read(AUDIO)

    # make mono
    if len(audio.shape) > 1:
        audio = audio[:,0]

    audio =  torch.tensor(audio[:SIG_LEN*fs], dtype=torch.float32)[None,...]

    hybra_fb = HybrA(FILTER_PATH, SIG_LEN, fs)
    hybra_fb = hybra_fb.to(device)
    audio_enc = hybra_fb(audio)
    audio_dec = hybra_fb.decoder(audio_enc.real, audio_enc.imag)


    cond_number = calculate_condition_number(hybra_fb.hybra_filters.squeeze(1))
    print(f"Condition number Hybra-Filter (Auditory-FB * Conv1D): {cond_number}")
    print(f"Condition number Encoder Filters: {calculate_condition_number(hybra_fb.encoder_weight.squeeze(1))}")
    print(f"Condition number Auditory Filters: {calculate_condition_number(hybra_fb.auditory_filterbank.squeeze(1))}")


    torch.allclose(audio[0,...], audio_dec[0,...], rtol=0.003)

    assert torch.allclose(audio[0,...], audio_dec[0,...], atol=0.005)


if __name__ == "__main__":
    test_reconstruction()