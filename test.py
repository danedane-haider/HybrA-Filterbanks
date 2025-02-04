import torch
import numpy as np
import matplotlib.pyplot as plt
from hybra import AudletFIR, HybrA

filterbank = AudletFIR(use_decoder=True, is_encoder_learnable=True)

filterbank.plot_response()
filterbank.plot_decoder_response()

import soundfile

x, fs = soundfile.read('./audio/noisy_speech.wav')
x = torch.tensor(x[fs:fs*5, 0], dtype=torch.float32).unsqueeze(0)

encoded = filterbank(x)
decoded = filterbank.decoder(encoded.real, encoded.imag)

soundfile.write('./audio/encoded.wav', decoded[0].detach().numpy(), fs)
