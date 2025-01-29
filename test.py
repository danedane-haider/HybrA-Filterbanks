import numpy as np
import matplotlib.pyplot as plt
from hybra import AudletFIR
from hybra.utils import audfilters_fir, response, plot_response

[g, a, M2, fc_new, L, fc, fc_low, fc_high, ind_crit] = audfilters_fir(fs=16000,
                                                                      Ls=4*16000,
                                                                      fmin=0,
                                                                      fmax=None,
                                                                      spacing=1/2,
                                                                      bwmul=1,
                                                                      filter_len=256,
                                                                      redmul=1,
                                                                      scale='erb'
                                                                      )
G = response(g,16000, a)
plot_response(g,16000,a,fc,fc_low,fc_high,ind_crit)

# plt.plot(np.real(g[0,:]))
# plt.plot(np.imag(g[0,:]))
# plt.plot(np.real(g[1,:]))
# plt.plot(np.imag(g[1,:]))
# plt.plot(np.real(g[2,:]))
# plt.plot(np.imag(g[2,:]))
# plt.show()

Lg = g.shape[-1]
M = g.shape[0]
g_long = np.concatenate([g, np.zeros((M, 16000 - Lg))], axis=1)
G = np.abs(np.fft.fft(g_long, axis=1)[:,:g_long.shape[1]//2])**2 / np.sqrt(a)
G = np.sum(G, axis=0)

print(max(G) / min(G))

