# import torch

# def sweep(dur=1, fs=8000, f_min, f_max, batch_size):
#     """
#     Generate an exponential sine sweep using PyTorch.
    
#     Parameters:
#     - sr: Sample rate (Hz)
#     - min_freq: Minimum frequency (Hz)
#     - max_freq: Maximum frequency (Hz)
#     - duration: Sweep duration (seconds). If None, randomly selected < 1 sec.
    
#     Returns:
#     - Tensor of sweep waveform.
#     """
#     time = torch.arange(dur*fs).reshape(1, -1) / fs
#     fmin = torch.rand(batch_size, 1) * f_min

#     duration = torch.rand(1).item() * 0.99 + 0.1  # Random value between 0.1 and 0.99 sec
#     t = torch.linspace(0, duration, int(fs * duration))  # Time vector
    
#     # Compute instantaneous phase
#     K = t / torch.log(torch.tensor(f_max / f_min))
#     phase = 2 * torch.pi * f_min * K * (torch.exp(t / K) - 1)
#     sweep = torch.sin(phase)
    
#     return sweep