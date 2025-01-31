import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp

import torch

def random_sweep(dur=2, fs=16000):
    """
    Generate an exponential sine sweep using PyTorch.
    
    Parameters:
    - sr: Sample rate (Hz)
    - min_freq: Minimum frequency (Hz)
    - max_freq: Maximum frequency (Hz)
    - duration: Sweep duration (seconds). If None, randomly selected < 1 sec.
    
    Returns:
    - Tensor of sweep waveform.
    """

    length = dur*fs

    fmin = np.random.randint(1, fs//4, (1,))
    fmax = np.random.randint(fs//4, fs//2, (1,))

    #duration = np.abs(np.random.randn(1).item() * 0.9 * dur + 0.3)
    duration = (np.random.randint(length//10, length, (1,)))[0]
    t = np.linspace(0, duration/fs, duration)  # Time vector

    amplitude = np.random.rand(1).item() * 0.4 + 0.1
    sweep = chirp(t, f0=fmin, f1=fmax, t1=duration/fs, method='log') * amplitude

    start_pad = np.random.randint(0, length-duration, (1,))
    end_pad = length - duration - start_pad
    sweep = torch.tensor(sweep, dtype=torch.float32)

    return torch.nn.functional.pad(sweep, (start_pad[0], end_pad[0]), value=0)