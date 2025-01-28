import numpy as np
import matplotlib.pyplot as plt
from hybra import AudletFIR
from hybra.utils import audfilters_fir, response, plot_response

[g, a, M2, fc_new, L, fc, fc_low, fc_high, ind_crit] = audfilters_fir(fs=16000, Ls=4*16000, fmin=0, fmax=None, spacing=1/4, bwmul=1, filter_len=480, redmul=1, scale='erb')
G = response(g,16000, a)
plot_response(g,16000,a,fc,fc_low,fc_high,ind_crit)

print(fc_new, fc)
