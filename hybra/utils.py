import numpy as np
import torch
import matplotlib.pyplot as plt

####################################################################################################
##################### Cool routines to study decimated filterbanks #################################
####################################################################################################

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
        assert N % int(D) == 0, "Oh no! Decimation factor must divide signal length!"

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

def calculate_condition_number(w:torch.Tensor, D:int):
    """
    in: frequency responses of fb (n_filters, system length), decimation factor
    out: condition number
    """
    w_hat = torch.fft.fft(w, dim=-1).T
    return condition_number(w_hat, D)

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
        assert N % int(D) == 0, "Oh no! Decimation factor must divide signal length!"

        w_hat_tight = torch.zeros(J, N, dtype=torch.complex64)
        for j in range(N//D):
            idx = (j - torch.arange(D) * (N//D)) % N
            H = w_hat[idx, :]
            U, _, V = torch.linalg.svd(H, full_matrices=False)
            H = U @ V
            w_hat_tight[:,idx] = H.T.to(torch.complex64)
        return torch.fft.ifft(torch.fft.ifft(w_hat_tight.T, dim=1) * D ** 0.5, dim=0).T

def frame_bounds(w, D):
    """
    in: frequency responses of fb (system length, n_filters), decimation factor
    out: frame bounds
    """
    N = w.shape[0]
    M = w.shape[1]
    assert N % D == 0

    A = torch.inf
    B = 0
    Ha = torch.zeros((D,M))
    Hb = torch.zeros((D,M))

    for j in range(N//D):
        idx_a = np.mod(j - np.arange(D) * (N//D), N).astype(int)
        idx_b = np.mod(np.arange(D) * (N//D) - j, N).astype(int)
        Ha = w[idx_a, :]
        Hb = torch.conj(w[idx_b, :])
        lam = torch.linalg.eigvalsh(Ha @ Ha.H + Hb @ Hb.H).real
        A = np.min([A, torch.min(lam).item()]).item()
        B = np.max([B, torch.max(lam).item()]).item()
    return A/D, B/D

def kappa_alias(w:torch.Tensor, D:int) -> torch.Tensor:
    """
    Computes the frequency correlation functions.
    Input:  w: Impulse responses of the filterbank as 2-D Tensor torch.tensor[num_channels, sig_length]
            D: Decimation (or downsampling) factor, must divide filter length!
            aliasing: If False, only the condition umber is returned
    Output: Condition number and norm of the aliasing term
    """
    w_hat = torch.fft.fft(w, dim=-1).T
    N = w_hat.shape[0]
    G = torch.zeros(N, D)
    G[:,0] = torch.sum(torch.abs(w_hat)**2, dim=1)
    for j in range(1,D):
        G[:,j] = torch.sum(w_hat * torch.conj(w_hat.roll(j * N//D, 0)), dim=1)
    return G

def alias_conditioner(w:torch.Tensor, D:int, aliasing:bool=True) -> torch.Tensor:
    G = kappa_alias(w, D)
    return torch.max(G[:,0]).div(torch.min(G[:,0])) + torch.norm(G[:,1::], dim=1)**2 - 1

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
    kappa = calculate_condition_number(w, D).item()
    while kappa > eps:
        w_tight = can_tight(w_tight, D)
        w_tight[:, supp:] = 0
        kappa = calculate_condition_number(w_tight, D).item()
    if Ls is None:
        return w_tight
    else:
        return w_tight[:,:supp]
    

####################################################################################################
################### Routines for constructing auditory filterbanks #################################
####################################################################################################

def freqtoaud(freq, scale="erb"):
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
    
    if scale == "mel":
        # MEL scale
        return 1000 / np.log(17 / 7) * np.sign(freq) * np.log(1 + np.abs(freq) / 700)

    elif scale == "erb":
        # Glasberg and Moore's ERB scale
        return 9.2645 * np.sign(freq) * np.log(1 + np.abs(freq) * 0.00437)

    elif scale == "bark":
        # Bark scale from Traunmuller (1990)
        return np.sign(freq) * ((26.81 / (1 + 1960 / np.abs(freq))) - 0.53)

    elif scale in ["log10", "semitone"]:
        # Logarithmic scale
        return np.log10(freq)

    else:
        raise ValueError(f"Unsupported scale: '{scale}'. Available options are: 'mel', 'erb', 'bark', 'log10'.")

def audtofreq(aud, scale="erb"):
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
    if scale == "mel":
        return 700 * np.sign(aud) * (np.exp(np.abs(aud) * np.log(17 / 7) / 1000) - 1)
    elif scale == "erb":
        return (1 / 0.00437) * (np.exp(aud / 9.2645) - 1)
    elif scale == "bark":
        return np.sign(aud) * 1960 / (26.81 / (np.abs(aud) + 0.53) - 1)
    elif scale in ["log10", "semitone"]:
        return 10 ** aud
    else:
        raise ValueError(f"Unsupported scale: '{scale}'. Available options are: 'mel', 'erb', 'bark', 'log10'.")


def audspace(fmin, fmax, num_channels, scale="erb"):
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
        bw (float): Bandwidth between each sample on the auditory scale.
    """
    if not (isinstance(fmin, (int, float)) and np.isscalar(fmin)):
        raise ValueError("fmin must be a scalar.")
    
    if not (isinstance(fmax, (int, float)) and np.isscalar(fmax)):
        raise ValueError("fmax must be a scalar.")
    
    if not (isinstance(num_channels, int) and num_channels > 0):
        raise ValueError("n must be a positive integer scalar.")
    
    if fmin > fmax:
        raise ValueError("fmin must be less than or equal to fmax.")

    # Convert [fmin, fmax] to auditory scale
    audlimits = freqtoaud(np.array([fmin, fmax]), scale)

    # Generate frequencies spaced evenly on the auditory scale
    aud_space = np.linspace(audlimits[0], audlimits[1], num_channels)
    y = audtofreq(aud_space, scale)

    # Ensure exact endpoints
    y[0] = fmin
    y[-1] = fmax

    return y

def freqtoaud_mod(freq, fc_crit):
    """
    Modified auditory scale function with linear region below fc_crit.
    
    Parameters:
    freq (ndarray): Frequency values in Hz.
    fc_crit (float): Critical frequency in Hz.

    Returns:
    ndarray: Values on the modified auditory scale.
    """
    aud_crit = freqtoaud(fc_crit)
    slope = (freqtoaud(fc_crit * 1.01) - aud_crit) / (fc_crit * 0.01)

    aud = np.zeros_like(freq, dtype=np.float32)
    linear_part = freq < fc_crit
    auditory_part = freq >= fc_crit

    aud[linear_part] = slope * (freq[linear_part] - fc_crit) + aud_crit
    aud[auditory_part] = freqtoaud(freq[auditory_part])

    return aud

def audtofreq_mod(aud, fc_crit):
    """
    Inverse of freqtoaud_mod to map auditory scale back to frequency.
    
    Parameters:
    aud (ndarray): Auditory scale values.
    fc_crit (float): Critical frequency in Hz.

    Returns:
    ndarray: Frequency values in Hz
    """
    aud_crit = freqtoaud(fc_crit)
    slope = (freqtoaud(fc_crit * 1.01) - aud_crit) / (fc_crit * 0.01)

    freq = np.zeros_like(aud, dtype=np.float32)
    linear_part = aud < aud_crit
    auditory_part = aud >= aud_crit

    freq[linear_part] = (aud[linear_part] - aud_crit) / slope + fc_crit
    freq[auditory_part] = audtofreq(aud[auditory_part])

    return freq

def audspace_mod(fc_crit, fs, num_channels):
    """Generate M frequency samples that are equidistant in the modified auditory scale.
    
    Parameters:
    fc_crit (float): Critical frequency in Hz.
    fs (int): Sampling rate in Hz.
    M (int): Number of filters/channels.

    Returns:
    ndarray: Frequency values in Hz and in the auditory scale.
    """

    # Convert [0, fs//2] to modified auditory scale
    aud_min = freqtoaud_mod(np.array([0]), fc_crit)[0]
    aud_max = freqtoaud_mod(np.array([fs//2]), fc_crit)[0]

    # Generate frequencies spaced evenly on the modified auditory scale
    fc_aud = np.linspace(aud_min, aud_max, num_channels)

    # Convert back to frequency scale
    fc = audtofreq_mod(fc_aud, fc_crit)

    # Ensure exact endpoints
    fc[0] = 0
    fc[-1] = fs//2

    return fc, fc_aud

def fctobw(fc, scale="erb"):
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
    ndarray or float: Critical bandwidth at each center frequency.
    """
    if isinstance(fc, (list, tuple)):
        fc = np.array(fc)
    if not (isinstance(fc, (float, int, np.ndarray)) and np.all(fc >= 0)):
        raise ValueError("fc must be a non-negative scalar or array.")

    # Compute bandwidth based on the auditory scale
    if scale == "erb":
        bw = 24.7 + fc / 9.265
    elif scale == "bark":
        bw = 25 + 75 * (1 + 1.4e-6 * fc**2)**0.69
    elif scale == "mel":
        bw = np.log(17 / 7) * (700 + fc) / 1000
    elif scale in ["log10"]:
        bw = fc
    else:
        raise ValueError(f"Unsupported auditory scale: {scale}")

    return bw

def bwtofc(bw, scale="erb"):
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
    ndarray or float: Center frequency corresponding to the given bandwidth.
    """
    if isinstance(bw, (list, tuple)):
        bw = np.array(bw)
    if not (isinstance(bw, (float, int, np.ndarray)) and np.all(bw >= 0)):
        raise ValueError("bw must be a non-negative scalar or array.")

    # Compute center frequency based on the auditory scale
    if scale == "erb":
        fc = (bw - 24.7) * 9.265
    elif scale == "bark":
        fc = np.sqrt(((bw - 25) / 75)**(1 / 0.69) / 1.4e-6)
    elif scale == "mel":
        fc = 1000 * (bw / np.log(17 / 7)) - 700
    elif scale in ["log10"]:
        fc = bw
    else:
        raise ValueError(f"Unsupported auditory scale: {scale}")

    return fc

def firwin(window_length, padding_length=None):
    """
    FIR window generation in Python.
    
    Parameters:
        window_length (int): Length of the window.
        padding_length (int): Length to which it should be padded.
        name (str): Name of the window.
        
    Returns:
        g (ndarray): FIR window.
    """

    if window_length % 2 == 0:
        x = np.concatenate([np.linspace(0, 0.5 - 1/window_length, window_length//2), np.linspace(-0.5, -1/window_length, window_length//2)])
    else:
        x = np.concatenate([np.linspace(0, 0.5 - 0.5/window_length, window_length//2), np.linspace(-0.5 + 0.5/window_length, -0.5/window_length, window_length//2)])
    x += window_length//2 / window_length

    # Hann window
    g = 0.5 + 0.5 * np.cos(2 * np.pi * x)
    
    # L1 Normalization
    g /= np.sum(np.abs(g))
    #g /= np.max(np.abs(g))

    if padding_length is None:
        if window_length % 2 == 0:
            return g
        else:
            return np.concatenate([g, np.zeros(1)])
    elif padding_length == window_length:
        return g
    elif padding_length > window_length:
        g_padded = np.concatenate([g, np.zeros(padding_length - len(g))])
        g_centered = np.roll(g_padded, (padding_length - len(g))//2)
        return g_centered
    else:
        raise ValueError("padding_length must be larger than window_length.")


def modulate(g, fc, fs):
    """Modulate a filters.
    
    Args:
        g (list of torch.Tensor): Filters.
        fc (list): Center frequencies.
        fs (int): Sampling rate.
    
    Returns:
        g_mod (list of torch.Tensor): Modulated filters.
    """
    Lg = len(g)
    g_mod = g * np.exp(2 * np.pi * 1j * fc * np.arange(Lg) / fs)
    return g_mod


####################################################################################################
########################################## the #####################################################
################################## filterbank generator ############################################
####################################################################################################


def audfilters_fir(filter_len, num_channels, fs, Ls, bwmul=1, scale='erb'):
    """
    Generate FIR filter kernel with length *filter_len* equidistantly spaced on auditory frequency scales.
    
    Parameters:
        filter_len (int): Length of the FIR filter.
        num_channels (int): Number of channels.
        fs (int): Sampling rate.
        Ls (int): Signal length.
        bwmul (float): Bandwidth multiplier.
        scale (str): Auditory scale.
    
    Returns:
        filters (list of torch.Tensor): Generated filters.
        a (list): Downsampling rates.
        fc (list): Center frequencies.
        L (int): Admissible signal length.
    """

    ####################################################################################################
    # Bandwidth conversion
    ####################################################################################################

    probeLs = 10000
    probeLg = 1000
    g_probe = firwin(probeLg, probeLs)
    
    # peak normalize
    gf_probe = np.fft.fft(g_probe) / np.max(np.abs(np.fft.fft(g_probe)))

    # compute ERB-type bandwidth of the prototype
    bw_conversion = np.linalg.norm(gf_probe)**2 * probeLg / probeLs / 4
    weird_factor = fs * 10.64
    
    ####################################################################################################
    # Center frequencies
    ####################################################################################################

    # get the bandwidth for the maximum admissible filter length and the associated center frequency
    fsupp_crit = bw_conversion / filter_len * weird_factor
    fc_crit = bwtofc(fsupp_crit / bwmul * bw_conversion)
    fc_crit_aud = audtofreq(fc_crit)

    [fc, fc_aud] = audspace_mod(fc_crit, fs, num_channels)
    num_lin = np.where(fc < fc_crit)[0].shape[0]

    ####################################################################################################
    # Frequency and time supports
    ####################################################################################################

    # frequency support for the auditory part
    fsupp = fctobw(fc[num_lin:]) / bw_conversion * bwmul

    # time support for the auditory part
    tsupp_lin = (np.ones(num_lin) * filter_len).astype(int)
    tsupp_aud = (np.round(bw_conversion / fsupp * weird_factor)).astype(int)
    tsupp = np.concatenate([tsupp_lin, tsupp_aud])

    # Maximal decimation factor (stride) to get a nice frame and accoring signal length
    d = np.floor(np.min(fs / fsupp)).astype(int)
    L = int(np.ceil(Ls / d) * d)

    ####################################################################################################
    # Generate filters
    ####################################################################################################

    g = np.zeros((num_channels, filter_len), dtype=np.complex128)

    g[0,:] = np.sqrt(d) * firwin(filter_len) #/ np.sqrt(2)
    g[-1,:] = np.sqrt(d) * modulate(firwin(tsupp[-1], filter_len), fs//2, fs) #/ np.sqrt(2)

    for m in range(1, num_channels - 1):
        g[m,:] = np.sqrt(d) * modulate(firwin(tsupp[m], filter_len), fc[m], fs)

    return g, d, fc, fc_crit, L

####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################

def response(g, fs):
    """Frequency response of the filters.
    
    Args:
        g (numpy.Array): Filters.
        fs (int): Sampling rate for plotting Hz.
        a (int): Downsampling rate.
    """
    Lg = g.shape[-1]
    num_channels = g.shape[0]
    g_long = np.concatenate([g, np.zeros((num_channels, int(fs) - Lg))], axis=1)
    G = np.abs(np.fft.fft(g_long, axis=1)[:,:fs//2])**2

    return G

def plot_response(g, fs, scale=False, fc_crit=None, decoder=False):
    """Frequency response of the filters.
    
    Args:
        g (numpy.Array): Filters.
        a (int): Downsampling rate.
        fs (int): Sampling rate for plotting Hz.
        fc_orig (numpy.Array): Original center frequencies.
        fc_low (numpy.Array): Center frequencies of the low-pass filters.
        fc_high (numpy.Array): Center frequencies of the high-pass filters.
        ind_crit (int): Index of the critical filter.
    """
    g_hat = response(g, fs)
    psd = np.sum(g_hat, axis=0)

    if scale:
        fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        scale_id = 0
        fr_id = 1
        psd_id = 2

        num_channels = g.shape[0]
        filter_length = g.shape[1]
        freq_samples, aud_samples = audspace_mod(fc_crit, fs, num_channels)
        freqs = np.linspace(0, fs//2, fs//2)

        auds = freqtoaud_mod(freqs, fc_crit)

        ax[scale_id].scatter(freq_samples, freqtoaud_mod(freq_samples, fc_crit), color="black", label="Center frequencies", linewidths = 0.05)
        ax[scale_id].plot(freqs, auds, color='black')

        if fc_crit is not None:
            ax[scale_id].axvline(fc_crit, color='salmon', linestyle='--', label="Transition frequency", alpha=0.5)
            ax[scale_id].fill_betweenx(y=[auds[0]-1, auds[-1]*1.1], x1=0, x2=fc_crit, color='gray', alpha=0.25)
            ax[scale_id].fill_betweenx(y=[auds[0]-1, auds[-1]*1.1], x1=fc_crit, x2=fs//2, color='gray', alpha=0.1)

            ax[fr_id].fill_betweenx(y=[0, np.max(g_hat)*1.1], x1=0, x2=fc_crit, color='gray', alpha=0.25)
            ax[fr_id].fill_betweenx(y=[0, np.max(g_hat)*1.1], x1=fc_crit, x2=fs//2, color='gray', alpha=0.1)
            ax[psd_id].fill_betweenx(y=[0, np.max(psd)*1.1], x1=0, x2=fc_crit, color='gray', alpha=0.25)
            ax[psd_id].fill_betweenx(y=[0, np.max(psd)*1.1], x1=fc_crit, x2=fs//2, color='gray', alpha=0.1)

        ax[scale_id].set_xlim([0, fs//2])
        ax[scale_id].set_ylim([auds[0]-1, auds[-1]*1.1])
        #ax[scale_id].set_xlabel("Frequency (Hz)")
        text_x = fc_crit / 2
        text_y = auds[-1] - 2
        ax[scale_id].text(text_x, text_y, 'linear', color='black', ha='center', va='center', fontsize=12, alpha=0.75)
        ax[scale_id].text(text_x + fc_crit - 1, text_y, 'ERB', color='black', ha='center', va='center', fontsize=12, alpha=0.75)
        ax[scale_id].set_title(f"Modified Auditory Scale for {num_channels} filters of length {filter_length}")
        ax[scale_id].set_ylabel("Auditory Units")
        ax[scale_id].legend(loc='lower right')

    else:
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        fr_id = 0
        psd_id = 1
    
    f_range = np.linspace(0, fs//2, fs//2)
    ax[fr_id].set_xlim([0, fs//2])
    ax[fr_id].set_ylim([0, np.max(g_hat)*1.1])
    ax[fr_id].plot(f_range, g_hat.T)
    if decoder:
        ax[fr_id].set_title('Frequency responses of the synthesis filters')
    if not decoder:
        ax[fr_id].set_title('Frequency responses of the analysis filters')
    #ax[fr_id].set_xlabel('Frequency [Hz]')
    ax[fr_id].set_ylabel('Magnitude')

    ax[psd_id].plot(f_range, psd)
    ax[psd_id].set_xlim([0, fs//2])
    ax[psd_id].set_ylim([0, np.max(psd)*1.1])
    ax[psd_id].set_title('Power spectral density')
    ax[psd_id].set_xlabel('Frequency [Hz]')
    ax[psd_id].set_ylabel('Magnitude')

    plt.show()