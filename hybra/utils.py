import numpy as np
import torch

def calculate_condition_number(w) -> torch.Tensor:
    """""
    Calculate the condition number of a convolution operator via the Littlewood Payley sum.
    Input: w (torch.tensor) - matrix with filter impulse respones in the columns
    Output: kappa (torch.tensor) - condition number of the convolution operator
    """""
    w_hat: torch.Tensor = torch.sum(torch.abs(torch.fft.fft(w, dim=1)) ** 2, dim=0)
    B: torch.Tensor = torch.max(w_hat, dim=0).values
    A: torch.Tensor = torch.min(w_hat, dim=0).values
    kappa: torch.Tensor = B/A
    return kappa

def audfilters(n_filters, filter_length, hop_length, frequency_scale, sr):
    """""
    constructs a set of filters *g* that are equidistantly spaced on a perceptual frequency scale (see |freqtoaud|) between 0 and the Nyquist frequency.
    The filter bandwidths are proportional to the  critical bandwidth of the auditory filters |audfiltbw|.
    The filters are intended to work with signals with a sampling rate of *fs*.
    The signal length *Ls* is mandatory, since we need to avoid too narrow frequency windows.
    """""
    g = np.zeros((n_filters, filter_length), dtype=np.complex64)
    return g