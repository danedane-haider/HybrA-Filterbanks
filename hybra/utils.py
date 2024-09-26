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

def frame_bounds(w, D):
    """
    in: frequency responses of fb (system length, n_filters), decimation factor
    out: frame bounds
    """
    N = w.shape[0]
    M = w.shape[1]
    assert N % D == 0

    A = torch.tensor([torch.inf])
    B = torch.tensor([0])
    Ha = torch.zeros((D,M))
    Hb = torch.zeros((D,M))

    for j in range(N//D):
        idx_a = np.mod(j - np.arange(D) * (N//D), N).astype(int)
        idx_b = np.mod(np.arange(D) * (N//D) - j, N).astype(int)
        Ha = w[idx_a, :]
        Hb = torch.conj(w[idx_b, :])
        lam = torch.linalg.eigvalsh(Ha @ Ha.H + Hb @ Hb.H).real
        A = torch.min(A, torch.min(lam))
        B = torch.max(B, torch.max(lam))
    return A/D, B/D
    
def calculate_condition_number(w, D) -> torch.Tensor:
    """
    Computes the frame bounds of the filterbank
    :param w: frequency responses of fb (system length, n_filters)
    :param D: decimation factor
    :return: A, B - frame bounds
    """
    A, B = frame_bounds(w, D)
    return B/A

def smooth_fir(frequency_responses, support, time_domain=False):
    """""
    Takes a matrix of frequency responses (as columns) and constructs a smoothed FIR version with support length *support*.
    """""
    g = np.exp(-np.pi * np.arange(-support//2,support//2)**2 / ((support-12)/2)**2)
    supper = g.reshape(1,-1)
    if time_domain:
        imp = frequency_responses.T
    else:
        imp = np.fft.ifft(frequency_responses, axis=0)
        imp = np.roll(imp, support//2, axis=0)
    g_re = np.real(imp[:support]).T * supper
    g_im = np.imag(imp[:support]).T * supper
    return torch.from_numpy(g_re + 1j * g_im)

def can_tight(w, D=1):
    """
    Construction of the canonical tight filterbank
    :param w: analysis filterbank
    :param D: decimation factor. Default is 1
    :return: canonical tight filterbank
    """
    if D == 1:
        w_hat = torch.fft.fft(w, dim=1)
        lp = torch.sum(w.abs() ** 2, dim=0)
        w_tight = w * lp ** (-0.5)
        return torch.fft.ifft(w_tight, dim=1)
    else:
        N = w.shape[0]
        M = w.shape[1]
        w_tight = torch.zeros(M, N, dtype=w.dtype)
        for j in range(N//D):
            idx = np.mod(j - np.arange(D) * (N//D), N).astype(int)
            H = w[idx, :]
            U, S, V = torch.linalg.svd(H, full_matrices=False)
            H = U @ V
            w_tight[:,idx] = H.T
        return torch.fft.ifft(w_tight.T, dim=1) * np.sqrt(D)

def kappa_alias(w, D):
    w_hat = torch.fft.fft(w.T, dim=0)
    kappa = calculate_condition_number(w_hat, D)

    if D == 1:
        return kappa, torch.tensor([0])
    else:
        N = w.shape[1]
        alias = torch.zeros_like(w_hat)
        for j in range(1,D):
            alias += w_hat * torch.conj(w_hat.roll(j * N//D, 0))
        alias = torch.sum(alias, dim=1)
    return kappa, torch.linalg.norm(alias)


def fir_tightener3000(w, supp, D=1, eps=1.01):
    """
    Iterative tightening procedure with fixed support for a given filterbank 
    :param w: analysis filterbank
    :param supp: desired support of the tight filterbank
    :param eps: desired precision for kappa = B/A
    :return: approximately tight filterbank
    """
    kappa = calculate_condition_number(w, D)
    w_tight = w.clone()
    while kappa > eps:
        w_tight = can_tight(w_tight)
        w_tight[:, supp:] = 0
        kappa = calculate_condition_number(w_tight, D)
    return w_tight

def fir_tightener4000(w, supp, D=1, eps=1.01):
    """
    Iterative tightening procedure with fixed support for a given filterbank
    :param w: analysis filterbank
    :param supp: desired support of the tight filterbank
    :param eps: desired precision for kappa = B/A
    :return: approximately tight filterbank, where every filter is additionally a tight filterbank
    """
    for i in range(w.shape[0]):
        filter = w[i,:].reshape(1,-1)
        w[i,:] = fir_tightener3000(filter, supp, D, eps)
    return w
    
def fir_tightener5000(w, supp, D=1, eps=1.01):
    """
    Iterative tightening procedure with fixed support for a given filterbank 
    :param w: analysis filterbank
    :param supp: desired support of the tight filterbank
    :param eps: desired precision for kappa = B/A
    :return: approximately tight filterbank only on support: ready to use for conv1d
    """
    W = torch.fft.fft(w.T, dim=0)
    kappa = calculate_condition_number(W, D)
    w_tight = W.clone()
    while kappa > eps:
        w_tight = can_tight(w_tight, D)
        w_tight = torch.fft.ifft(w_tight.T, dim=1)
        w_tight[:, supp:] = 0
        w_tight = torch.fft.fft(w_tight.T, dim=0)
        kappa = calculate_condition_number(w_tight, D)
        print(kappa)
    return torch.fft.ifft(w_tight.T, dim=1)[:, :supp]
