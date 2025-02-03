import numpy as np
import matplotlib.pyplot as plt
from hybra import AudletFIR

a = AudletFIR(decoder=True)

a.plot_response()
a.plot_decoder_response()

# plt.plot(np.real(g[0,:]))
# plt.plot(np.imag(g[0,:]))
# plt.plot(np.real(g[1,:]))
# plt.plot(np.imag(g[1,:]))
# plt.plot(np.real(g[2,:]))
# plt.plot(np.imag(g[2,:]))
# plt.show()