import numpy as np
import torch
import matplotlib.pyplot as plt

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

def fir_tightener3000(w, supp, eps=1.1, print_kappa=False):
    """
    Iterative construction of a tight filterbank with a given support
    :param w: analysis filterbank
    :param supp: desired support of the tight filterbank
    :param eps: desired precision for kappa = B/A
    :return: tight filterbank
    """
    A, B = frame_bounds_lp(w)
    w_tight = w.copy()
    while B / A > eps:
        w_tight = tight(w_tight)
        w_tight[:, supp:] = 0
        w_tight = np.real(w_tight)
        A, B = frame_bounds_lp(w_tight)
        kappa = B / A
        error = np.linalg.norm(w - w_tight)
        if print_kappa:
            print("kappa:", "%.4f" % kappa, ", error:", "%.4f" % error)
    return w_tight

def audtofreq(aud, scale="erb"):
    """
    Converts auditory units to frequency (Hz). Note that fs = 2.

    Parameters:
    aud (float or numpy array): Auditory scale value(s) to convert.
    scale (str): The auditory scale. Options are:
        - "mel": Mel scale
        - "erb": Equivalent Rectangular Bandwidth scale (default)
        - "bark": Bark scale
        - "log10": Base-10 logarithmic scale
        - "semitone": Semitone logarithmic scale

    Returns:
    float or numpy array: Frequency value(s) in Hz.
    """
    if scale == "mel":
        return 700 * np.sign(aud) * (np.exp(np.abs(aud) * np.log(17 / 7) / 1000) - 1)
    elif scale == "erb":
        return (1 / 0.00437) * np.sign(aud) * (np.exp(np.abs(aud) / 9.2645) - 1)
    elif scale == "bark":
        return np.sign(aud) * 1960 / (26.81 / (np.abs(aud) + 0.53) - 1)
    elif scale in ["log10", "semitone"]:
        return 10 ** aud
    else:
        raise ValueError(f"Unsupported scale: '{scale}'. Available options are: 'mel', 'erb', 'bark', 'log10', 'semitone'.")

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
        raise ValueError(f"Unsupported scale: '{scale}'. Available options are: 'mel', 'erb', 'bark', 'log10', 'semitone'.")

def audspace(fmin, fmax, n, scale="erb"):
    """
    Computes a vector of values equidistantly spaced on the selected auditory scale.

    Parameters:
    fmin (float): Minimum frequency in Hz.
    fmax (float): Maximum frequency in Hz.
    n (int): Number of points in the output vector.
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
    
    if not (isinstance(n, int) and n > 0):
        raise ValueError("n must be a positive integer scalar.")
    
    if fmin > fmax:
        raise ValueError("fmin must be less than or equal to fmax.")

    # Convert [fmin, fmax] to auditory scale
    audlimits = freqtoaud(np.array([fmin, fmax]), scale)

    # Generate frequencies spaced evenly on the auditory scale
    aud_space = np.linspace(audlimits[0], audlimits[1], n)
    y = audtofreq(aud_space, scale)

    # Calculate the bandwidth
    #bw = (audlimits[1] - audlimits[0]) / (n - 1)

    # Set exact endpoints
    y[0] = fmin
    y[-1] = fmax

    return y

def fctobw(fc, scale="erb"):
    """
    Computes the critical bandwidth of the auditory filter at a given center frequency.

    Parameters:
    fc (float or ndarray): Center frequency in Hz. Must be non-negative.
    audscale (str): Auditory scale. Supported values are:
                    - 'erb': Equivalent Rectangular Bandwidth (default)
                    - 'bark': Bark scale
                    - 'mel': Mel scale
                    - 'log10': Logarithmic scale
                    - 'semitone': Semitone scale

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
    elif scale in ["log10", "semitone"]:
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
                 - 'semitone': Semitone scale

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
    elif scale in ["log10", "semitone"]:
        fc = bw
    else:
        raise ValueError(f"Unsupported auditory scale: {scale}")

    return fc

def firwin(window_length, padding_length=None, name='hann'):
    """
    FIR window generation in Python.
    
    Parameters:
        window_length (int): Length of the window.
        padding_length (int): Length of the padding.
        name (str): Name of the window.
        
    Returns:
        g (ndarray): FIR window.
        info (dict): Metadata about the window.
    """
    if window_length % 2 == 0:
        x = np.concatenate([np.linspace(0, 0.5 - 1/window_length, window_length//2), np.linspace(-0.5, -1/window_length, window_length//2)])
    else:
        x = np.concatenate([np.linspace(0, 0.5 - 0.5/window_length, window_length//2), np.linspace(-0.5 + 0.5/window_length, -0.5/window_length, window_length//2)])
    
    x += window_length//2 / window_length

    g = 0.5 + 0.5 * np.cos(2 * np.pi * x)
    
    # L1 Normalization
    g /= np.sum(np.abs(g))
    #g /= np.max(np.abs(g))

    if padding_length is None:
        if window_length % 2 == 0:
            return g
        else:
            return np.concatenate([g, np.zeros(1)])
    
    if padding_length > window_length:
        g_padded = np.concatenate([g, np.zeros(padding_length - len(g))])
        g_centered = np.roll(g_padded, len(g)//2)
        return g_centered

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
    g_mod = g * np.exp(2*np.pi*1j*fc*np.arange(Lg)/fs)
    return g_mod

def audfilters_fir(filter_length, num_channels, fs, Ls, bwmul=1, scale='erb'):
    """
    Generate FIR filter kernel with length *filter_length* equidistantly spaced on auditory frequency scales.
    
    Parameters:
        filter_length (int): Length of the FIR filter.
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
    bw_conversion = np.linalg.norm(gf_probe)**2 * probeLg / probeLs / 2
    
    ####################################################################################################
    # Center frequencies
    ####################################################################################################

    # get the bandwidth for the maximum admissible filter length and the associated center frequency
    fsupp_crit = bw_conversion / filter_length * fs
    fc_crit = bwtofc(fsupp_crit, scale) # in Hz
    #fc_crit_aud = freqtoaud(fc_crit_hz) # in auditory units

    aud = audspace(0, fs//2, fs//2)
    intersection = int(np.abs(aud - fc_crit).argmin())
    hz = np.linspace(0, fs//2, fs//2) * fc_crit / intersection

    fc_full = np.maximum(hz, aud)
    ind = np.linspace(0, len(fc_full) - 1, num_channels, dtype=int)
    fc = fc_full[ind]

    # channels that go to the linear part
    num_lin = len(fc[fc < fc_crit])

    ####################################################################################################
    # Frequency and time supports
    ####################################################################################################

    # frequency support for the auditory part
    fsupp = fctobw(fc[num_lin+1:]) / bw_conversion * bwmul

    # time support for the auditory part
    tsupp = (np.round(bw_conversion / fsupp * fs * 10.64)).astype(int)

    # Maximal decimation factor (stride) to get a nice frame and accoring signal length
    d = np.floor(np.min(fs / fsupp)).astype(int)
    L = int(np.ceil(Ls / d) * d)

    ####################################################################################################
    # Generate filters
    ####################################################################################################

    g = np.zeros((num_channels, filter_length), dtype=np.complex128)

    g[0,:] = np.sqrt(d) * firwin(filter_length) / np.sqrt(2)
    g[-1,:] = np.sqrt(d) * modulate(firwin(tsupp[-1], filter_length), fs//2, fs) / np.sqrt(2)

    # linear part
    for m in range(1,num_lin):
        g[m,:] = np.sqrt(d) * modulate(firwin(filter_length), fc[m], fs)
    
    # auditory part
    for m in range(num_channels - num_lin - 1):
        g[m,:] = np.sqrt(d) * modulate(firwin(tsupp[m], filter_length), fc[m], fs)

    return g, d, fc, L

def response(g, fs):
    """Frequency response of the filters.
    
    Args:
        g (numpy.Array): Filters.
        fs (int): Sampling rate for plotting Hz.
        a (int): Downsampling rate.
    """
    Lg = g.shape[-1]
    num_channels = g.shape[0]
    g_long = np.concatenate([g, np.zeros((num_channels, fs - Lg))], axis=1)
    G = np.abs(np.fft.fft(g_long, axis=1))**2

    return G

def plot_response(g, fc, fs):
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
    G = response(g, fs)

    f_range = np.linspace(-fs//2, fs//2, fs)
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))
    ax[0].plot(f_range, G.T)
    ax[0].set_title('Frequency responses of the filters')
    ax[0].set_xlabel('Frequency [Hz]')
    ax[0].set_ylabel('Magnitude')
    ax[0].set_yscale('log')

    ax[1].plot(f_range, np.sum(G, axis=0))
    ax[1].set_title('Power spectral density')
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_ylabel('Magnitude')

    f = np.linspace(0, fs // 2, len(fc))
    ax[2].plot(fc, f, label='center frequencies', color='blue', linewidth=1)
    ax[2].plot(freqtoaud(f), f, label='auditory scale', color='black', linewidth=1)

    # fc_low2 = np.linspace(fc_orig[0], fc_low[-1], len(ind_crit))
    # fc_new2 = np.concatenate((fc_low2, fc_high))
    # ax[2].plot(fc_orig, np.linspace(0, fs // 2, len(fc_orig)), color='black', linewidth=1)
    # ax[2].plot(fc_new2, np.linspace(0, fs // 2, len(fc_new2)), linestyle='--', label='kernel length = ', color='red', linewidth=1)
    plt.legend()

    plt.show()