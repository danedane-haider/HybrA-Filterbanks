import numpy as np
import torch
import matplotlib.pyplot as plt

from typing import Union, Tuple

####################################################################################################
##################### Cool routines to study decimated filterbanks #################################
####################################################################################################

def frame_bounds(w:torch.Tensor, D:int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the frame bounds of a filterbank given in impulse responses using the polyphase representation.
    Parameters:
        w: Impulse responses of the filterbank as 2-D Tensor torch.tensor[num_channels, length]
        D: Decimation (or downsampling) factor, must divide filter length!
    Returns:
        tuple:
            A, B: Frame bounds
    """
    w_hat = torch.fft.fft(w, dim=-1).T
    if D == 1:
        lp = torch.sum(w_hat.abs() ** 2, dim=1)
        A = torch.min(lp)
        B = torch.max(lp)
        return A, B
    else:
        N = w_hat.shape[0]
        M = w_hat.shape[1]
        assert N % D == 0, "Oh no! Decimation factor must divide signal length!"

        if w_hat.device.type == "mps":
            temp_device = torch.device("cpu")
        else:
            temp_device = w_hat.device
        
        w_hat_cpu = w_hat.to(temp_device)
        A = torch.tensor([torch.inf]).to(temp_device)
        B = torch.tensor([0]).to(temp_device)
        Ha = torch.zeros((D,M)).to(temp_device)
        Hb = torch.zeros((D,M)).to(temp_device)

        for j in range(N//D):
            idx_a = (j - torch.arange(D) * (N//D)) % N
            idx_b = (torch.arange(D) * (N//D) - j) % N
            Ha = w_hat_cpu[idx_a, :]
            Hb = torch.conj(w_hat_cpu[idx_b, :])
            lam = torch.linalg.eigvalsh(Ha @ Ha.H + Hb @ Hb.H).real
            A = torch.min(A, torch.min(lam))
            B = torch.max(B, torch.max(lam))
        return A/D, B/D

def condition_number(w:torch.Tensor, D:int) -> torch.Tensor:
    """
    Computes the condition number of a filterbank.
    Parameters:
        w: Impulse responses of the filterbank as 2-D Tensor torch.tensor[num_channels, signal_length]
        D: Decimation factor (stride), must divide signal_length!
    Returns:
        kappa: Condition number
    """
    A, B = frame_bounds(w, D)
    return B / A

def can_tight(w:torch.Tensor, D:int) -> torch.Tensor:
    """
    Computes the canonical tight filterbank of w (time domain) using the polyphase representation.
    Parameters:
        w: Impulse responses of the filterbank as 2-D Tensor torch.tensor[num_channels, signal_length]
        D: Decimation factor, must divide signal_length!
    Returns:
        W: Canonical tight filterbank of W (torch.tensor[num_channels, signal_length])
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

def frequency_correlation(w:torch.Tensor, D:int, padto:Union[int,None]=None, diag_only:bool=False) -> torch.Tensor:
    """
    Computes the frequency correlation functions.
    Parameters:
        w: Impulse responses of the filterbank as 2-D Tensor torch.tensor[num_channels, sig_length]
        D (int): Decimation factor, must divide filter length!
    Output:
        G: (length x D) matrix with aliasing terms as columns
    """
    if padto is not None:
        w = torch.cat([w, torch.zeros(w.shape[0], padto-w.shape[1]).to(w.device)], dim=1)
    w_hat = torch.fft.fft(w, dim=-1).T
    N = w_hat.shape[0]
    assert N % D == 0, "Oh no! Decimation factor must divide signal length!"
    G = torch.zeros(N, D)
    G[:,0] = torch.sum(torch.abs(w_hat)**2, dim=1)
    if diag_only:
        return G[:,0].T
    else:
        for j in range(1,D):
            G[:,j] = torch.sum(w_hat * torch.conj(w_hat.roll(j * N//D, 0)), dim=1)
        return G.T

def alias(w:torch.Tensor, D:int, padto:Union[int,None]=None, diag_only:bool=False) -> torch.Tensor:
    """
    Computes the norm of the aliasing terms.
    Parameters:
        w: Impulse responses of the filterbank as 2-D Tensor torch.tensor[num_channels, sig_length]
        D: Decimation factor, must divide filter length!
    Output:
        A: Energy of the aliasing terms
    """
    G = frequency_correlation(w=w, D=D, padto=padto, diag_only=diag_only)
    if diag_only:
        return torch.max(G).div(torch.min(G))
    else:
        return torch.max(G[0,:]).div(torch.min(G[0,:])) + torch.norm(G[1::,:], dim=0)**2

def fir_tightener3000(w:torch.Tensor, supp:int, D:int, eps:float=1.01, Ls:Union[int,None]=None):
    """
    Iterative tightening procedure with fixed support for a given filterbank w
    Parameters:
        w: Impulse responses of the filterbank as 2-D Tensor torch.tensor[num_channels, signal_length].
        supp: Desired support of the resulting filterbank
        D: Decimation factor, must divide filter length!
        eps: Desired precision for the condition number
        Ls: System length (if not already given by w). If set, the resulting filterbank is padded with zeros to length Ls.
    Returns:
        Filterbank with condition number *eps* and support length *supp*. If length=supp then the resulting filterbank is the canonical tight filterbank of w.
    """
    print('Hold on, the kernels are tightening')
    if Ls is not None:
        w =  torch.cat([w, torch.zeros(w.shape[0], Ls-w.shape[1])], dim=1)
    w_tight = w.clone()
    kappa = condition_number(w, D).item()
    while kappa > eps:
        w_tight = can_tight(w_tight, D)
        w_tight[:, supp:] = 0
        kappa = condition_number(w_tight, D).item()
    if Ls is None:
        return w_tight
    else:
        return w_tight[:,:supp]
    

####################################################################################################
################### Routines for constructing auditory filterbanks #################################
####################################################################################################

def freqtoaud(freq:Union[float,int,torch.Tensor], scale:str="erb"):
    """
    Converts frequencies (Hz) to auditory scale units.

    Parameters:
        freq (float or ndarray): Frequency value(s) in Hz.
        scale (str): Auditory scale. Supported values are:
                    - 'erb' (default)
                    - 'mel'
                    - 'bark'
                    - 'log10'
    Returns:
        float or ndarray: Corresponding auditory scale units.
    """

    scale = scale.lower()
    
    if isinstance(freq, (int, float)):
        freq = torch.tensor(freq) 

    if scale == "erb":
        # Glasberg and Moore's ERB scale
        return 9.2645 * torch.sign(freq) * torch.log(1 + torch.abs(freq) * 0.00437)

    elif scale == "mel":
        # MEL scale
        return 1000 / torch.log(torch.tensor(17 / 7)) * torch.sign(freq) * torch.log(1 + torch.abs(freq) / 700)

    elif scale == "bark":
        # Bark scale from Traunmuller (1990)
        return torch.sign(freq) * ((26.81 / (1 + 1960 / torch.abs(freq))) - 0.53)

    elif scale == "log10":
        # Logarithmic scale
        return torch.log10(freq)

    else:
        raise ValueError(f"Unsupported scale: '{scale}'. Available options are: 'mel', 'erb', 'bark', 'log10'.")

def audtofreq(aud:Union[float,int,torch.Tensor], scale:str="erb"):
    """
    Converts auditory units to frequency (Hz).
    Parameters:
        aud (float or numpy array): Auditory scale value(s) to convert.
        scale (str): Auditory scale. Supported values are:
                    - 'erb' (default)
                    - 'mel'
                    - 'bark'
                    - 'log10'
    Returns:
        float or numpy array: Frequency value(s) in Hz.
    """
    if scale == "erb":
        return (1 / 0.00437) * (torch.exp(aud / 9.2645) - 1)

    elif scale == "mel":
        return 700 * torch.sign(aud) * (torch.exp(torch.abs(aud) * torch.log(torch.tensor(17 / 7)) / 1000) - 1)
    
    elif scale == "bark":
        return torch.sign(aud) * 1960 / (26.81 / (torch.abs(aud) + 0.53) - 1)
    
    elif scale == "log10":
        return 10 ** aud

    else:
        raise ValueError(f"Unsupported scale: '{scale}'. Available options are: 'mel', 'erb', 'bark', 'log10'.")


def audspace(fmin:Union[float,int,torch.Tensor], fmax:Union[float,int,torch.Tensor], num_channels:int, scale:str="erb"):
    """
    Computes a vector of values equidistantly spaced on the selected auditory scale.

    Parameters:
        fmin (float): Minimum frequency in Hz.
        fmax (float): Maximum frequency in Hz.
        num_channels (int): Number of points in the output vector.
        audscale (str): Auditory scale (default is 'erb').
    Returns:
        tuple:
            y (ndarray): Array of frequencies equidistantly scaled on the auditory scale.
    """
    
    if num_channels <= 0:
        raise ValueError("n must be a positive integer scalar.")
    
    if fmin > fmax:
        raise ValueError("fmin must be less than or equal to fmax.")

    # Convert [fmin, fmax] to auditory scale
    audlimits = freqtoaud(torch.tensor([fmin, fmax]), scale)

    # Generate frequencies spaced evenly on the auditory scale
    aud_space = torch.linspace(audlimits[0], audlimits[1], num_channels)
    y = audtofreq(aud_space, scale)

    # Ensure exact endpoints
    y[0] = fmin
    y[-1] = fmax

    return y

def freqtoaud_mod(freq:Union[float,int,torch.Tensor], fc_low:Union[float,int,torch.Tensor], fc_high:Union[float,int,torch.Tensor], scale="erb"):
    """
    Modified auditory scale function with linear region below fc_crit.
    
    Parameters:
        freq (ndarray): Frequency values in Hz.
        fc_low (float): Lower transition frequency in Hz.
        fc_high (float): Upper transition frequency in Hz.
    Returns:
        ndarray:
            Values on the modified auditory scale.
    """
    aud_crit_low = freqtoaud(fc_low, scale)
    aud_crit_high = freqtoaud(fc_high, scale)
    slope_low = (freqtoaud(fc_low * 1.01, scale) - aud_crit_low) / (fc_low * 0.01)
    slope_high = (freqtoaud(fc_high * 1.01, scale) - aud_crit_high) / (fc_high * 0.01)

    linear_low = freq < fc_low
    linear_high = freq > fc_high
    auditory = [not x for x in (linear_low + linear_high)]

    aud = torch.zeros_like(freq, dtype=torch.float32)

    aud[linear_low] = slope_low * (freq[linear_low] - fc_low) + aud_crit_low
    aud[auditory] = freqtoaud(freq[auditory], scale)
    aud[linear_high] = slope_high * (freq[linear_high] - fc_high) + aud_crit_high

    return aud

def audtofreq_mod(aud:Union[float,int,torch.Tensor], fc_low:Union[float,int,torch.Tensor], fc_high:Union[float,int,torch.Tensor], scale="erb"):
    """
    Inverse of freqtoaud_mod to map auditory scale back to frequency.
    
    Parameters:
        aud (ndarray): Auditory scale values.
        fc_low (float): Lower transition frequency in Hz.
        fc_high (float): Upper transition frequency in Hz.
    Returns:
        ndarray:
            Frequency values in Hz
    """
    aud_crit_low = freqtoaud(fc_low, scale)
    aud_crit_high = freqtoaud(fc_high, scale)
    slope_low = (freqtoaud(fc_low * 1.01, scale) - aud_crit_low) / (fc_low * 0.01)
    slope_high = (freqtoaud(fc_high * 1.01, scale) - aud_crit_high) / (fc_high * 0.01)

    linear_low = aud < aud_crit_low
    linear_high = aud > aud_crit_high
    auditory_part = [not x for x in (linear_low + linear_high)]

    freq = torch.zeros_like(aud, dtype=torch.float32)

    freq[linear_low] = (aud[linear_low] - aud_crit_low) / slope_low + fc_low
    freq[auditory_part] = audtofreq(aud[auditory_part], scale)
    freq[linear_high] = (aud[linear_high] - aud_crit_high) / slope_high + fc_high

    return freq

def audspace_mod(fc_low:Union[float,int,torch.Tensor], fc_high:Union[float,int,torch.Tensor], fs:int, num_channels:int, scale:str="erb"):
    """Generate M frequency samples that are equidistant in the modified auditory scale.
    
    Parameters:
        fc_crit (float): Critical frequency in Hz.
        fs (int): Sampling rate in Hz.
        M (int): Number of filters/channels.

    Returns:
        ndarray:
            Frequency values in Hz and in the auditory scale.
    """
    if fc_low > fc_high:
        raise ValueError("fc_low must be less than fc_high.")
    elif fc_low == fc_high:
        # equidistant samples form 0 to fs/2
        if scale == "log10":
            fc = torch.linspace(1, fs//2, num_channels)
        else:
            fc = torch.linspace(0, fs//2, num_channels)
        return fc, freqtoaud_mod(fc, fs//2, fs//2, scale)
    elif fc_low < fc_high:
        # Convert [0, fs//2] to modified auditory scale
        if scale == "log10":
            aud_min = freqtoaud_mod(torch.tensor([1]), fc_low, fc_high, scale)[0]
        else:
            aud_min = freqtoaud_mod(torch.tensor([0]), fc_low, fc_high, scale)[0]
        aud_max = freqtoaud_mod(torch.tensor([fs//2]), fc_low, fc_high, scale)[0]

        # Generate frequencies spaced evenly on the modified auditory scale
        fc_aud = torch.linspace(aud_min, aud_max, num_channels)

        # Convert back to frequency scale
        fc = audtofreq_mod(fc_aud, fc_low, fc_high, scale)

        # Ensure exact endpoints
        fc[0] = 0
        fc[-1] = fs//2

        return fc, fc_aud
    else:
        raise ValueError("There is something wrong with fc_low and fc_high.")

def fctobw(fc:Union[float,int,torch.Tensor], scale="erb"):
    """
    Computes the critical bandwidth of a filter at a given center frequency.

    Parameters:
        fc (float or ndarray): Center frequency in Hz. Must be non-negative.
        audscale (str): Auditory scale. Supported values are:
                    - 'erb': Equivalent Rectangular Bandwidth (default)
                    - 'bark': Bark scale
                    - 'mel': Mel scale
                    - 'log10': Logarithmic scale

    Returns:
        ndarray or float:
            Critical bandwidth at each center frequency.
    """
    if isinstance(fc, (list, tuple, int, float)):
        fc = torch.tensor(fc)
    if not (isinstance(fc, (float, int, torch.Tensor)) and torch.all(fc >= 0)):
        raise ValueError("fc must be a non-negative scalar or array.")

    # Compute bandwidth based on the auditory scale
    if scale == "erb":
        bw = 24.7 + fc / 9.265
    elif scale == "bark":
        bw = 25 + 75 * (1 + 1.4e-6 * fc**2)**0.69
    elif scale == "mel":
        bw = torch.log10(torch.tensor(17 / 7)) * (700 + fc) / 1000
    elif scale in ["log10"]:
        bw = fc
    else:
        raise ValueError(f"Unsupported auditory scale: {scale}")

    return bw

def bwtofc(bw:Union[float,int,torch.Tensor], scale="erb"):
    """
    Computes the center frequency corresponding to a given critical bandwidth.

    Parameters:
        bw (float or ndarray): Critical bandwidth. Must be non-negative.
        scale (str): Auditory scale. Supported values are:
                 - 'erb': Equivalent Rectangular Bandwidth
                 - 'bark': Bark scale
                 - 'mel': Mel scale
                 - 'log10': Logarithmic scale

    Returns:
        ndarray or float:
            Center frequency corresponding to the given bandwidth.
    """
    if isinstance(bw, (list, tuple)):
        bw = torch.tensor(bw)
    if not (isinstance(bw, (float, int, torch.Tensor)) and torch.all(bw >= 0)):
        raise ValueError("bw must be a non-negative scalar or array.")

    # Compute center frequency based on the auditory scale
    if scale == "erb":
        fc = (bw - 24.7) * 9.265
    elif scale == "bark":
        fc = torch.sqrt(((bw - 25) / 75)**(1 / 0.69) / 1.4e-6)
    elif scale == "mel":
        fc = 1000 * (bw / torch.log10(torch.tensor(17 / 7))) - 700
    elif scale in ["log10"]:
        fc = bw
    else:
        raise ValueError(f"Unsupported auditory scale: {scale}")

    return fc

def firwin(kernel_size:int, padto:int=None):
    """
    FIR window generation in Python.
    
    Parameters:
        kernel_size (int): Length of the window.
        padto (int): Length to which it should be padded.
        name (str): Name of the window.
        
    Returns:
        g (ndarray): FIR window.
    """
    g = torch.hann_window(kernel_size, periodic=False)
    g /= torch.sum(torch.abs(g))

    if padto is None or padto == kernel_size:
        return g
    elif padto > kernel_size:
        g_padded = torch.concatenate([g, torch.zeros(padto - len(g))])
        g_centered = torch.roll(g_padded, int((padto - len(g))//2))
        return g_centered
    else:
        raise ValueError("padto must be larger than kernel_size.")


def modulate(g:torch.Tensor, fc:Union[float,int,torch.Tensor], fs:int):
    """Modulate a filters.
    
    Args:
        g (list of torch.Tensor): Filters.
        fc (list): Center frequencies.
        fs (int): Sampling rate.
    
    Returns:
        g_mod (list of torch.Tensor): Modulated filters.
    """
    Lg = len(g)
    g_mod = g * torch.exp(2 * torch.pi * 1j * fc * torch.arange(Lg) / fs)
    return g_mod


####################################################################################################
########################################### ISAC ###################################################
####################################################################################################


def audfilters(kernel_max:Union[int,None]=None, num_channels:int=96, fc_max:Union[float,int,None]=None, fs:int=16000, L:int=16000, bwmul:float=1, scale:str='erb') -> tuple[torch.Tensor, int, int, Union[int,float], Union[int,float], int, int, int]:
    """
    Generate FIR filter kernels with length *kernel_max* equidistantly spaced on auditory frequency scales.
    
    Parameters:
        kernel_max (int): Size of the filter kernels (equals maximum window length).
        num_channels (int): Number of channels.
        fc_max (int): Maximum frequency (in Hz) that should lie on the aud scale.
        fs (int): Sampling rate.
        L (int): Signal length.
        bwmul (float): Bandwidth multiplier.
        scale (str): Auditory scale.
    
    Returns:
        tuple:
            kernels (torch.Tensor): Generated kernels.
            d (int): Downsampling rates.
            fc (list): Center frequencies.
            fc_min (int, float): First transition frequency.
            fc_max (int, float): Second transition frequency.
            kernel_min (int): Minimum kernel size.
            kernel_max (int): Maximum kernel size.
            L (int): Admissible signal length.
    """

    ####################################################################################################
    # Bandwidth conversion
    ####################################################################################################

    probeLs = 10000
    probeLg = 1000
    g_probe = firwin(probeLg, probeLs)
    
    # peak normalize
    gf_probe = torch.fft.fft(g_probe) / torch.max(torch.abs(torch.fft.fft(g_probe)))

    # compute ERB-type bandwidth of the prototype
    bw_conversion = torch.norm(gf_probe)**2 * probeLg / probeLs / 4
    if scale in ['erb','bark']:
        bw_factor = fs * 10.64
    elif scale == 'mel':
        bw_factor = fs / 12
    elif scale == 'log10':
        bw_factor = fs * 0.001
    
    
    ####################################################################################################
    # Center frequencies
    ####################################################################################################

    # default values
    if kernel_max is None:
        fc_full = audspace(0, fs//2, num_channels, scale)
        fsupp_min = fctobw(fc_full[1], scale) / bw_conversion * bwmul
        kernel_max = int(torch.round(bw_conversion / fsupp_min * bw_factor))
    
    if fc_max is None:
        fc_max = fs // 2

    # get the bandwidth for the maximum kernel size and the associated center frequency
    fsupp_low = bw_conversion / kernel_max * bw_factor
    fc_min = bwtofc(fsupp_low / bwmul * bw_conversion, scale)

    # get the bandwidth for the maximum center frequency and the associated kernel size
    fsupp_high = fctobw(fc_max, scale) / bw_conversion * bwmul
    kernel_min = int(torch.round(bw_conversion / fsupp_high * bw_factor))

    if fc_min >= fc_max:
        fc_max = fc_min
        kernel_min = kernel_max
        Warning(f"fc_max was increased to {fc_min} to enable the kernel size of {kernel_max}.")

    # get center frequencies
    [fc, _] = audspace_mod(fc_min, fc_max, fs, num_channels, scale)

    num_low = torch.where(fc < fc_min)[0].shape[0]
    num_high = torch.where(fc > fc_max)[0].shape[0]
    num_aud = num_channels - num_low - num_high

    ####################################################################################################
    # Frequency and time supports
    ####################################################################################################

    # get time supports
    tsupp_low = (torch.ones(num_low) * kernel_max).int()
    tsupp_high = torch.ones(num_high) * kernel_min
    if num_low + num_high == num_channels:
        fsupp = fctobw(fc_max, scale) / bw_conversion * bwmul
        tsupp = tsupp_low
    else:
        fsupp = fctobw(fc[num_low:num_low+num_aud], scale) / bw_conversion * bwmul
        tsupp_aud = torch.round(bw_conversion / fsupp * bw_factor)
        tsupp = torch.concatenate([tsupp_low, tsupp_aud, tsupp_high]).int()

    # Decimation factor (stride) to get a nice frame and according signal length (lcm of d and Ls)
    d = torch.floor(torch.min(fs / fsupp))
    #d = kernel_min // 2
    Ls = int(torch.ceil(L / d) * d)

    ####################################################################################################
    # Generate filters
    ####################################################################################################

    g = torch.zeros((num_channels, kernel_max), dtype=torch.complex128)

    g[0,:] = torch.sqrt(d) * firwin(kernel_max) / torch.sqrt(torch.tensor(2))
    g[-1,:] = torch.sqrt(d) * modulate(firwin(tsupp[-1], kernel_max), fs//2, fs) / torch.sqrt(torch.tensor(2))

    for m in range(1, num_channels - 1):
        g[m,:] = torch.sqrt(d) * modulate(firwin(tsupp[m], kernel_max), fc[m], fs)

    return g, int(d), fc, fc_min, fc_max, kernel_min, kernel_max, Ls

####################################################################################################
####################################################################################################
####################################################################################################

def response(g, fs):
    """Frequency response of the filters (Total power spectral density).
    
    Args:
        g (numpy.Array): Filter kernels.
        fs (int): Sampling rate for plotting Hz.
    """
    Lg = g.shape[-1]
    num_channels = g.shape[0]
    g_long = np.concatenate([g, np.zeros((num_channels, int(fs) - Lg))], axis=1)
    g_neg = np.conj(g_long)
    g_full = np.concatenate([g_long, g_neg], axis=0)
    G = np.abs(np.fft.fft(g_full, axis=1)[:,:fs//2])**2

    return G

def plot_response(g, fs, scale='erb', plot_scale=False, fc_min=None, fc_max=None, kernel_min=None, decoder=False):
    """Plotting routine for the frequencs scale and the frequency responses of the filters.
    
    Args:
        g (numpy.Array): Filters.
        fs (int): Sampling rate for plotting Hz.
        scale (str): Auditory scale.
        plot_scale (bool): Plot the scale or not.
        fc_min (float): Lower transition frequency in Hz.
        fc_max (float): Upper transition frequency in Hz.
        kernel_min (int): Minimum kernel size.
        decoder (bool): Plot for the synthesis fb.
    """
    num_channels = g.shape[0]
    kernel_max = g.shape[1]

    g_hat = response(g, fs)
    g_hat_pos = g_hat[:num_channels,:]
    psd = np.sum(g_hat, axis=0)

    if plot_scale:
        plt.figure(figsize=(8, 2))
        freq_samples, _ = audspace_mod(fc_min, fc_max, fs, num_channels, scale)
        freqs = torch.linspace(0, fs//2, fs//2)

        auds = freqtoaud_mod(freqs, fc_min, fc_max, scale).numpy()

        plt.scatter(freq_samples.numpy(), freqtoaud_mod(freq_samples, fc_min, fc_max, scale).numpy(), color="black", label="Center frequencies", linewidths = 0.05)
        plt.plot(freqs, auds, color='black')

        if fc_min is not None:
            plt.axvline(fc_min, color='black', linestyle='--', label="Transition: lin - aud", alpha=0.5)
            plt.fill_betweenx(y=[auds[0]-1, auds[-1]*1.1], x1=0, x2=fc_min, color='gray', alpha=0.25)
            plt.fill_betweenx(y=[auds[0]-1, auds[-1]*1.1], x1=fc_min, x2=fs//2, color='gray', alpha=0.1)

        if fc_max is not None:
            plt.axvline(fc_max, color='black', linestyle='--', label="Transition: aud - lin", alpha=0.5)
            plt.fill_betweenx(y=[auds[0]-1, auds[-1]*1.1], x1=0, x2=fc_max, color='gray', alpha=0.25)
            plt.fill_betweenx(y=[auds[0]-1, auds[-1]*1.1], x1=fc_max, x2=fs//2, color='gray', alpha=0.1)

        plt.xlim([0, fs//2])
        plt.ylim([auds[0]-1, auds[-1]*1.1])
        plt.xlabel("Frequency (Hz)")
        # text_x = fc_min / 2
        # text_y = auds[-1] 
        # plt.text(text_x, text_y, 'linear', color='black', ha='center', va='center', fontsize=12, alpha=0.75)
        # plt.text(text_x + fc_min - 1, text_y, 'ERB', color='black', ha='center', va='center', fontsize=12, alpha=0.75)
        plt.title(f"ISAC Scale for {num_channels} channels. Max kernel size: {kernel_max}, Min kernel size: {kernel_min}")
        plt.ylabel("Auditory Units")
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()

    fig, ax = plt.subplots(2, 1, figsize=(6, 3), sharex=True)

    fr_id = 0
    psd_id = 1
    
    f_range = np.linspace(0, fs//2, fs//2)
    ax[fr_id].set_xlim([0, fs//2])
    ax[fr_id].set_ylim([0, np.max(g_hat_pos)*1.1])
    ax[fr_id].plot(f_range, g_hat_pos.T)
    if decoder:
        ax[fr_id].set_title('PSDs of the synthesis filters')
    if not decoder:
        ax[fr_id].set_title('PSDs of the analysis filters')
    #ax[fr_id].set_xlabel('Frequency [Hz]')
    ax[fr_id].set_ylabel('Magnitude')

    ax[psd_id].plot(f_range, psd)
    ax[psd_id].set_xlim([0, fs//2])
    ax[psd_id].set_ylim([0, np.max(psd)*1.1])
    ax[psd_id].set_title('Total PSD')
    ax[psd_id].set_xlabel('Frequency [Hz]')
    ax[psd_id].set_ylabel('Magnitude')

    if fc_min is not None:
        ax[fr_id].fill_betweenx(y=[0, np.max(g_hat)*1.1], x1=0, x2=fc_min, color='gray', alpha=0.25)
        ax[fr_id].fill_betweenx(y=[0, np.max(g_hat)*1.1], x1=fc_min, x2=fs//2, color='gray', alpha=0.1)
        ax[psd_id].fill_betweenx(y=[0, np.max(psd)*1.1], x1=0, x2=fc_min, color='gray', alpha=0.25)
        ax[psd_id].fill_betweenx(y=[0, np.max(psd)*1.1], x1=fc_min, x2=fs//2, color='gray', alpha=0.1)
    
    if fc_max is not None:
        ax[fr_id].fill_betweenx(y=[0, np.max(g_hat)*1.1], x1=0, x2=fc_max, color='gray', alpha=0.25)
        ax[fr_id].fill_betweenx(y=[0, np.max(g_hat)*1.1], x1=fc_max, x2=fs//2, color='gray', alpha=0.1)
        ax[psd_id].fill_betweenx(y=[0, np.max(psd)*1.1], x1=0, x2=fc_max, color='gray', alpha=0.25)
        ax[psd_id].fill_betweenx(y=[0, np.max(psd)*1.1], x1=fc_max, x2=fs//2, color='gray', alpha=0.1)

    plt.tight_layout()
    plt.show()

def plot_coefficients(coefficients, fc, L, fs):
    """Plot the ISAC coefficients.

    Args:
        coefficients (numpy.Array): Filterbank coefficients.
        fc (numpy.Array): Center frequencies.
        L (int): Signal length.
        fs (int): Sampling rate.
    """
    fig, ax = plt.subplots()
    ax.pcolor(coefficients.cpu().numpy())#, origin='lower', aspect='auto')

    locs, labels = plt.yticks()
    ax.set_yticks(locs[1:-1], [int(np.round(y, 0)) for y in fc[[int(x) for x in locs[1:-1]]]])

    locs, labels = plt.yticks()
    ax.set_xticks(np.linspace(0, coefficients.shape[-1], len(locs)-2))
    ax.set_xticklabels([np.round(x, 1) for x in np.linspace(0, L/fs, len(locs)-2)])

    ax.set_title('Filterbank coefficients')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [s]')
    plt.tight_layout()
    plt.show()
