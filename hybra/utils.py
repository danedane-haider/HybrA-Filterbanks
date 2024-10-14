import torch

def random_filterbank(N:int, J:int, T:int, norm:bool=True, support_only:bool=False) -> torch.Tensor:
    """
    Constructs a [J, N] tensor with J i.i.d. Gaussian random filters of length N with support T.
    Input:  N: Signal length
            J: Number of filters
            T: Filter support (or length of conv1d kernels). Default: T=N
            norm: If True then the the variance is 1/(TJ)
            support_only: If True then N <- T 
    Output: Impulse responses of the random filterbank (torch.tensor[J, T])
    """
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
    
def kappa_alias(w:torch.Tensor, D:int, aliasing:bool=True) -> torch.Tensor:
    """
    Computes the condition number and the norm of the aliasing term of a filterbank using the polyphase representation.
    Input:  w: Impulse responses of the filterbank as 2-D Tensor torch.tensor[J, T]
            D: Decimation (or downsampling) factor, must divide filter length!
            aliasing: If False, only the condition umber is returned
    Output: Condition number and norm of the aliasing term
    """
    w_hat = torch.fft.fft(w, dim=-1).T
    kappa = condition_number(w_hat, D)

    if aliasing:
        alias = torch.zeros_like(w_hat)
        N = alias.shape[0]
        for j in range(1,D):
            alias += w_hat * torch.conj(w_hat.roll(j * N//D, 0))
        alias = torch.sum(alias, dim=1)
        return kappa, torch.linalg.norm(alias)
    else:
        return kappa

def condition_number(w_hat:torch.Tensor, D:int) -> torch.Tensor:
    """
    Computes the condition number of a filterbank w_hat using the polyphase representation.
    Input:  w: Frequency responses of the filterbank as 2-D Tensor torch.tensor[length, num_channels]
            D: Decimation (or downsampling) factor, must divide filter length!
    Output: Condition number.
    """
    if D == 1:
        lp = torch.sum(w_hat.abs() ** 2, dim=1)
        A = torch.min(lp)
        B = torch.max(lp)
        return B/A
    else:    
        N = w_hat.shape[0]
        J = w_hat.shape[1]
        assert N % D == 0, "Oh no! Decimation factor must divide signal length!"

        A = torch.tensor([torch.inf]).to(w_hat.device)
        B = torch.tensor([0]).to(w_hat.device)
        Ha = torch.zeros((D,J)).to(w_hat.device)
        Hb = torch.zeros((D,J)).to(w_hat.device)

        for j in range(N//D):
            idx_a = (j - torch.arange(D) * (N//D)) % N
            idx_b = (torch.arange(D) * (N//D) - j) % N
            Ha = w_hat[idx_a, :]
            Hb = torch.conj(w_hat[idx_b, :])
            lam = torch.linalg.eigvalsh(Ha @ Ha.H + Hb @ Hb.H).real
            A = torch.min(A, torch.min(lam))
            B = torch.max(B, torch.max(lam))
        return B/A

def can_tight(w:torch.Tensor, D:int) -> torch.Tensor:
    """
    Computes the canonical tight filterbank of w (time domain) using the polyphase representation.
    Input:  w: Impulse responses of the filterbank as 2-D Tensor torch.tensor[num_channels, length]
            D: Decimation (or downsampling) factor, must divide filter length!
    Output: Canonical tight filterbank of W (torch.tensor[num_channels, length])
    """
    w_hat = torch.fft.fft(w.T, dim=0)
    if D == 1:
        lp = torch.sum(w_hat.abs() ** 2, dim=1).reshape(-1,1)
        w_hat_tight = w_hat * (lp ** (-0.5))
        return torch.fft.ifft(w_hat_tight.T, dim=1)
    else:
        N = w_hat.shape[0]
        J = w_hat.shape[1]
        assert N % D == 0, "Oh no! Decimation factor must divide signal length!"

        w_hat_tight = torch.zeros(J, N, dtype=torch.complex64)
        for j in range(N//D):
            idx = (j - torch.arange(D) * (N//D)) % N
            H = w_hat[idx, :]
            U, _, V = torch.linalg.svd(H, full_matrices=False)
            H = U @ V
            w_hat_tight[:,idx] = H.T.to(torch.complex64)
        return torch.fft.ifft(torch.fft.ifft(w_hat_tight.T, dim=1) * D ** 0.5, dim=0).T

def fir_tightener3000(w, supp, D, eps=1.01, Ls=None):
    """
    Iterative tightening procedure with fixed support for a given filterbank w
    Input:  w: Impulse responses of the filterbank as 2-D Tensor torch.tensor[num_channels, length].
            supp: Desired support of the resulting filterbank
            D: Decimation (or downsampling) factor, must divide filter length!
            eps: Desired condition number
            Ls: control syste length
    Output: Filterbank with condition number *kappa* and support length *supp*. If length=supp then the resulting filterbank is the canonical tight filterbank of w.
    """
    if Ls is not None:
        w =  torch.cat([w, torch.zeros(w.shape[0], Ls-w.shape[1])], dim=1)
    w_tight = w.clone()
    kappa = kappa_alias(w, D, aliasing=False)
    while kappa > eps:
        w_tight = can_tight(w_tight, D)
        w_tight[:, supp:] = 0
        kappa = kappa_alias(w_tight, D, aliasing=False)
    if Ls is None:
        return w_tight
    else:
        return w_tight[:,:supp]

def fir_tightener4000(w, supp, eps=1.01):
    """
    Iterative tightening procedure with fixed support for a given filterbank w. Every filter will form an (approximate) tight frame.
    Input:  w: Impulse responses of the filterbank as 2-D Tensor torch.tensor[num_channels, length].
            supp: Desired support of the resulting filterbank
            eps: Desired condition number
    Output: Filterbank with support length *supp* and every filter has condition number *kappa*.
    """
    D=1 # only works without decimation!
    for i in range(w.shape[0]):
        filter = w[i,:].reshape(1,-1)
        w[i,:] = fir_tightener3000(filter, supp, D, eps)
    return w

def smooth_fir(w_hat:torch.Tensor, supp:int, time_domain:bool=False) -> torch.Tensor:
    """
    Takes a filterbank in frequency domain (as columns) and constructs a smoothed FIR version with support length *support*.
    Input:  w: Frequency responses of the filterbank as 2-D Tensor torch.tensor[length, num_channels]
            supp: Desired support in time domain
            time_domain: If True, w is treated as containing impulse responses torch.tensor[num_channels, length]
    Output: Impulse responses
    """
    g = torch.exp(-torch.pi * torch.arange(-supp//2,supp//2)**2 / ((supp-12)/2)**2)
    g = g.reshape(1,-1)
    if time_domain:
        w = w_hat
    else:
        w = torch.fft.ifft(w_hat, dim=0).T
        w = w.roll(supp//2, 0)[:, :supp]
    return w * g