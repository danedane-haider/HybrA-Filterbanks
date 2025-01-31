import numpy as np
import matplotlib.pyplot as plt
from hybra import AudletFIR
from hybra.utils import audfilters_fir, response, plot_response

filter_length = 128
num_channels = 40
fs = 16000
Ls = 4*fs
bwmul = 1
scale = 'erb'

[g, d, fc, fc_crit, L] = audfilters_fir(filter_length, num_channels, fs, Ls, bwmul)

plot_response(g, fs, scale=True, fc_crit=fc_crit)

# plt.plot(np.real(g[0,:]))
# plt.plot(np.imag(g[0,:]))
# plt.plot(np.real(g[1,:]))
# plt.plot(np.imag(g[1,:]))
# plt.plot(np.real(g[2,:]))
# plt.plot(np.imag(g[2,:]))
# plt.show()