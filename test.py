import numpy as np
import matplotlib.pyplot as plt
from hybra import AudletFIR

a = AudletFIR(decoder=True)

a.plot_response()
a.plot_decoder_response()