import torch
import numpy as np
from hybra.utils import fir_tightener3000, kappa_alias, random_filterbank

T = 512
J = 256
D = 64
N = T+D

fb = random_filterbank(T, J, T)

fb_tight = fir_tightener3000(fb, T, D, 1.01, Ls=N)

fb_tight_pad_1 = torch.cat([fb_tight, torch.zeros(J, N-T)], dim=-1)
fb_tight_pad_2 = torch.cat([fb_tight, torch.zeros(J, 100*N-T)], dim=-1)

print(kappa_alias(fb_tight_pad_1, D=D, aliasing=False), kappa_alias(fb_tight_pad_2, D=D, aliasing=False))