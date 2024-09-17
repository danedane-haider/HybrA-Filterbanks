import numpy as np
import torch

def random_filterbank(N, J, T, norm=True, support_only=False):
    """""
    Constructs a random filterbank of J filters of support T, padded with zeros to have length N.
    Input: N (int) - signal length
              J (int) - number of filters
                T (int) - support of the filter or length of learned conv1d kernel (default: T=N)
    Output: kappa (torch.tensor) - condition number of the convolution operator
    """""
    if T == None:
        T = N
    if norm:
        w = torch.randn(J, T).div(torch.sqrt(torch.tensor(J*T)))
    else:
        w = torch.randn(J, T)
    if support_only:
        w_cat = w
    else:
        w_cat = torch.cat([w, torch.zeros(J, N-T)], dim=1)
    return w_cat


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
    Constructs a set of filters *g* that are equidistantly spaced on a perceptual frequency scale (see |freqtoaud|) between 0 and the Nyquist frequency.
    The filter bandwidths are proportional to the  critical bandwidth of the auditory filters |audfiltbw|.
    The filters are intended to work with signals with a sampling rate of *fs*.
    The signal length *Ls* is mandatory, since we need to avoid too narrow frequency windows.
    """""
    g = np.zeros((n_filters, filter_length), dtype=np.complex64)
    return g

def smooth_fir(frequency_responses, support):
    """""
    Takes a matrix of frequency responses (as columns) and constructs a smoothed FIR version with support length *support*.
    """""
    g = np.exp(-np.pi * np.arange(-support//2,support//2)**2 / ((support-12)/2)**2)
    supper = g.reshape(1,-1)
    gi = np.fft.ifft(frequency_responses, axis=0)
    gi = np.roll(gi, support//2, axis=0)
    g_re = np.real(gi[:support]).T * supper
    g_im = np.imag(gi[:support]).T * supper
    return g_re + 1j * g_im

def tight(w, ver="S"):
    """
    Construction of the canonical tight filterbank
    :param w: analysis filterbank
    :param ver: version of the tight filterbank: 'S' yields canonical tight filterbank, 'flat_spec' yields tight filterbank with flat spectral response
    :return: canonical tight filterbank
    """
    if ver == "S":
        w_freqz = np.fft.fft(w, axis=1)
        lp = np.sum(np.abs(w_freqz) ** 2, axis=0)
        w_freqz_tight = w_freqz * lp ** (-0.5)
        w_tight = np.fft.ifft(w_freqz_tight, axis=1)
    elif ver == "flat_spec":
        M, N = w.shape
        w_freqz = np.fft.fft(w, axis=1).T
        w_tight = np.zeros((M, N), dtype=np.complex64)
        for k in range(N):
            H = w_freqz[k, :]
            U = H / np.linalg.norm(H)
            w_tight[:, k] = np.conj(U)
        w_tight = np.fft.ifft(w_tight.T, axis=0).T
    else:
        raise NotImplementedError
    return w_tight

def frame_bounds_lp(w, freq=False):
    # if the filters are given already as frequency responses
    if freq:
        w_hat = np.sum(np.abs(w) ** 2, axis=1)
    else:
        w_hat = np.sum(np.abs(np.fft.fft(w, axis=1)) ** 2, axis=0)
    B = np.max(w_hat)
    A = np.min(w_hat)

    return A, B

def can_tight(w):
    """
    Construction of the canonical tight filterbank
    :param w: analysis filterbank
    :return: canonical tight filterbank
    """
    w_freqz = torch.fft.fft(w, dim=1)
    lp = torch.sum(w_freqz.abs() ** 2, dim=0)
    w_freqz_tight = w_freqz * lp ** (-0.5)
    w_tight = torch.fft.ifft(w_freqz_tight, dim=1)
    return w_tight

def frame_bounds(w, frequency_domain=False):
    if frequency_domain:
        w_hat = torch.sum(w.abs() ** 2, dim=1)
    else:
        w_hat = torch.sum(torch.fft.fft(w, dim=1).abs() ** 2, dim=0)
    B = torch.max(w_hat).item()
    A = torch.min(w_hat).item()
    return A, B

def fir_tightener3000(w, supp, eps=1.01):
    """
    Iterative tightening procedure with fixed support for a given filterbank 
    :param w: analysis filterbank
    :param supp: desired support of the tight filterbank
    :param eps: desired precision for kappa = B/A
    :return: approximately tight filterbank
    """
    A, B = frame_bounds(w)
    kappa = B / A
    w_tight = w.clone()
    while kappa > eps:
        w_tight = can_tight(w_tight)
        w_tight[:, supp:] = 0
        #w_tight = torch.real(w_tight)
        A, B = frame_bounds(w_tight)
        kappa = B / A
        #error = np.linalg.norm(w - w_tight)
        #if print_kappa:
        #print("kappa:", "%.4f" % kappa)
    return w_tight


def fir_tightener4000(w, supp, eps=1.01):
    """
    Iterative tightening procedure with fixed support for a given filterbank
    :param w: analysis filterbank
    :param supp: desired support of the tight filterbank
    :param eps: desired precision for kappa = B/A
    :return: approximately tight filterbank, where every filter is additionally a tight filterbank
    """
    for i in range(w.shape[0]):
        print(i)
        filter = w[i,:].reshape(1,-1)
        w[i,:] = fir_tightener3000(filter, supp, eps)
    return w