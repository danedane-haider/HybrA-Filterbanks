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

def calculate_condition_number(w, D=1) -> torch.Tensor:
    """
    Computes the frame bounds of the filterbank
    :param w: analysis filterbank
    :param D: decimation factor
    :return: A, B - frame bounds
    """
    if D == 1:
        w_hat: torch.Tensor = torch.sum(torch.fft.fft(w, dim=1).abs() ** 2, dim=0)
        B: torch.Tensor = torch.max(w_hat, dim=0).values
        A: torch.Tensor = torch.min(w_hat, dim=0).values
    else:
        W: torch.Tensor = fb_analysis(w, D)
        sig: torch.Tensor = torch.svd(W).S
        B: torch.Tensor = torch.max(sig**2).values
        A: torch.Tensor = torch.min(sig**2).values
    kappa: torch.Tensor = B/A
    return kappa

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
        w_freqz = torch.fft.fft(w, dim=1)
        lp = torch.sum(w_freqz.abs() ** 2, dim=0)
        w_freqz_tight = w_freqz * lp ** (-0.5)
        w_tight = torch.fft.ifft(w_freqz_tight, dim=1)
        return w_tight
    else:
        W = fb_analysis(w, D)
        S = W.T @ W
        lam, U = torch.linalg.eig(S)
        lam_square = torch.sqrt(lam.real)**(-1)
        S_inv_sqrt = (U @ torch.diag(lam_square).to(torch.complex64) @ U.T).to(torch.float32)
        return (S_inv_sqrt @ w.T).T

def fb_analysis(w, D):
    """
    Construction of the analysis operator matrix having all shifted copies of the filters as rows
    :param w: analysis filterbank
    :param D: decimation factor
    :return: analysis operator matrix
    """
    N = w.shape[1]
    J = w.shape[0]
    w = w.flip((1,)).roll(1, 1)
    W = torch.cat([w.flip((1,)), torch.narrow(w.flip((1,)), dim=1, start=0, length=N-1)], dim=1)
    W = W.unfold(1, N, 1).flip((-1,)).reshape(J*N, N)
    return W[::D, :]

def kappa_alias(w, D):
    """
    Computes the condition number and aliasing term of the filterbank
    :param w: analysis filterbank
    :param D: decimation factor
    :return: kappa, alias - condition number and aliasing term
    """
    w_hat = torch.fft.fft(w, dim=1)
    diag = torch.sum(w_hat.abs() ** 2, dim=0)
    A = torch.min(diag)
    B = torch.max(diag)
    kappa = B/A
    if D == 1:
        alias = torch.tensor([0])
    else:
        hop = w.shape[1]//D
        alias = torch.zeros(w_hat.shape)
        for j in range(1,D):
            alias += torch.abs(w_hat * torch.conj(torch.roll(w_hat,j*hop,1)))
        alias = torch.sum(alias, dim=0)
    return kappa, alias # minimize the sum of them

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
            #print("kappa:", "%.4f" % kappa, ", error:", "%.4f" % error)
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
        filter = w[i,:].reshape(1,-1)
        w[i,:] = fir_tightener3000(filter, supp, eps)
    return w