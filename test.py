import numpy as np
import matplotlib.pyplot as plt
from hybra import AudletFIR
from hybra.utils import audfilters_fir, response, plot_response

filter_length = 50
num_channels = 64
fs = 16000
Ls = 4*fs
bwmul = 1
scale = 'erb'

[g, d, fc, L] = audfilters_fir(filter_length, num_channels, fs, Ls, bwmul, scale)

# G = response(g,fs)
# plot_response(g,fc,fs)

# # plt.plot(np.real(g[0,:]))
# # plt.plot(np.imag(g[0,:]))
# # plt.plot(np.real(g[1,:]))
# # plt.plot(np.imag(g[1,:]))
# # plt.plot(np.real(g[2,:]))
# # plt.plot(np.imag(g[2,:]))
# # plt.show()

# Lg = g.shape[-1]
# M = g.shape[0]
# g_long = np.concatenate([g, np.zeros((M, fs - Lg))], axis=1)
# G = np.abs(np.fft.fft(g_long, axis=1)[:,:g_long.shape[1]//2])**2
# G = np.sum(G, axis=0)

# print(max(G) / min(G))

