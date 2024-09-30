import torch
import numpy as np
from hybra.utils import fir_tightener3000, smooth_fir, kappa_alias

audfreqz = np.load('/Users/Dane/Desktop/audfreqz.npy')
audfreqz = torch.from_numpy(audfreqz)

N = audfreqz.shape[0]
J = audfreqz.shape[1]
D = 128
T = 512

aud = smooth_fir(audfreqz, T)
print(aud.shape)

aud = fir_tightener3000(aud, T, D, 1.0001, Ls=N)

torch.save({'auditory_filters_real': aud.real.to(dtype=torch.float32),
            'auditory_filters_imag': aud.imag.to(dtype=torch.float32),
            'auditory_filters_stride': 128,
            'n_filters': 256,
            'kernel_size': 512}, 'filters/audlet.pth')

print(kappa_alias(aud, D=D))