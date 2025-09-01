from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

####################################################################################################
##################### Cool routines to study decimated filterbanks #################################
####################################################################################################


def frame_bounds(
    w: torch.Tensor, d: int, Ls: Union[int, None] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute frame bounds of a filterbank using polyphase representation.

    Frame bounds characterize the numerical stability and invertibility of the
    filterbank transform. Tight frames (A ≈ B) provide optimal stability.

    Args:
        w (torch.Tensor): Impulse responses of shape (num_channels, length)
        d (int): Decimation (stride) factor
        Ls (int, optional): Signal length. If None, computed automatically. Default: None

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Lower and upper frame bounds (A, B)

    Note:
        For d=1, reduces to computing min/max of power spectral density.
        For d>1, uses polyphase analysis to compute worst-case eigenvalues.

    Example:
        >>> w = torch.randn(40, 128)
        >>> A, B = frame_bounds(w, d=4)
        >>> condition_number = B / A
    """
    if Ls is None:
        Ls = int(torch.ceil(torch.tensor(w.shape[-1] * 2 / d)) * d)
    w_full = torch.cat([w, torch.conj(w)], dim=0)
    w_hat = torch.fft.fft(w_full, Ls, dim=-1).T
    if d == 1:
        psd = torch.sum(w_hat.abs() ** 2, dim=-1)
        A = torch.min(psd)
        B = torch.max(psd)
        return A, B
    else:
        N = w_hat.shape[0]
        M = w_hat.shape[1]
        assert N % d == 0, "Oh no! Decimation factor must divide signal length!"

        if w_hat.device.type == "mps":
            temp_device = torch.device("cpu")
        else:
            temp_device = w_hat.device

        w_hat_cpu = w_hat.to(temp_device)
        A = torch.tensor([torch.inf]).to(temp_device)
        B = torch.tensor([0]).to(temp_device)
        Ha = torch.zeros((d, M)).to(temp_device)
        Hb = torch.zeros((d, M)).to(temp_device)

        for j in range(N // d):
            idx_a = (j - torch.arange(d) * (N // d)) % N
            idx_b = (torch.arange(d) * (N // d) - j) % N
            Ha = w_hat_cpu[idx_a, :]
            Hb = torch.conj(w_hat_cpu[idx_b, :])
            lam = torch.linalg.eigvalsh(Ha @ Ha.H + Hb @ Hb.H).real
            A = torch.min(A, torch.min(lam))
            B = torch.max(B, torch.max(lam))
        return (A / d).to(w_hat.device), (B / d).to(w_hat.device)


def condition_number(
    w: torch.Tensor, d: int, Ls: Union[int, None] = None
) -> torch.Tensor:
    """Compute condition number of a filterbank frame operator.

    The condition number κ = B/A quantifies numerical stability, where A and B
    are the lower and upper frame bounds. Lower values indicate better stability.

    Args:
        w (torch.Tensor): Impulse responses of shape (num_channels, signal_length)
        d (int): Decimation factor (stride)
        Ls (int, optional): Signal length. If None, computed automatically. Default: None

    Returns:
        torch.Tensor: Condition number κ = B/A

    Note:
        κ = 1 indicates a tight frame (optimal stability).
        κ >> 1 suggests potential numerical instability.

    Example:
        >>> w = torch.randn(40, 128)
        >>> kappa = condition_number(w, d=4)
        >>> print(f"Condition number: {kappa.item():.2f}")
    """
    A, B = frame_bounds(w, d, Ls)
    A = torch.max(
        A, torch.tensor(1e-6, dtype=A.dtype, device=A.device)
    )  # Avoid division by zero
    return B / A


def frequency_correlation(
    w: torch.Tensor, d: int, Ls: Union[int, None] = None, diag_only: bool = False
) -> torch.Tensor:
    """
    Computes the frequency correlation functions (vectorized version).
    Parameters:
        w: (J, K) - Impulse responses
        d: Decimation factor
        Ls: FFT length (default: nearest multiple of d ≥ 2K-1)
        diag_only: If True, only return diagonal (i.e., PSD)
    Returns:
        G: (d, Ls) complex tensor with frequency correlations
    """
    K = w.shape[-1]
    if Ls is None:
        Ls = int(torch.ceil(torch.tensor((2 * K - 1) / d)) * d)
    w_full = torch.cat([w, torch.conj(w)], dim=0)
    w_hat = torch.fft.fft(w_full, Ls, dim=-1)  # shape: [J, Ls]
    N = Ls
    assert N % d == 0, "Decimation factor must divide FFT length"

    # Diagonal: sum_j |w_hat_j|^2
    diag = torch.sum(w_hat.abs() ** 2, dim=0)  # shape: [Ls]

    if diag_only:
        return torch.real(diag)

    G = [diag]  # G[0] = diagonal

    for j in range(1, d):
        rolled = torch.roll(w_hat, shifts=j * (N // d), dims=-1)
        val = torch.sum(w_hat * torch.conj(rolled), dim=0)
        G.append(val)

    G = torch.stack(G, dim=0)  # shape: [d, Ls]
    return G


def alias(
    w: torch.Tensor, d: int, Ls: Union[int, None] = None, diag_only: bool = False
) -> torch.Tensor:
    """
    Computes the norm of the aliasing terms.
    Parameters:
        w: Impulse responses of the filterbank as 2-D Tensor torch.tensor[num_channels, sig_length]
        d: Decimation factor, must divide filter length!
    Output:
        A: Energy of the aliasing terms
    """
    G = frequency_correlation(w=w, d=d, Ls=Ls, diag_only=diag_only)
    if diag_only:
        return torch.max(G).div(torch.min(G))
    else:
        # return torch.max(torch.real(G[0,:])).div(torch.min(torch.real(G[0,:]))) + torch.sum(torch.norm(G[1::,:], p=2, dim=-1), dim=-1) - 1
        return torch.norm(
            torch.real(G[0, :]) - torch.ones_like(G[0, :]), p=2
        ) + torch.sum(torch.norm(G[1::, :], p=2, dim=-1), dim=-1)


def can_tight(w: torch.Tensor, d: int, Ls: int) -> torch.Tensor:
    """
    Computes the canonical tight filterbank of w (time domain) using the polyphase representation.
    Parameters:
        w: Impulse responses of the filterbank as 2-d Tensor torch.tensor[num_channels, signal_length]
        d: Decimation factor, must divide signal_length!
    Returns:
        W: Canonical tight filterbank of W (torch.tensor[num_channels, signal_length])
    """
    w_hat = torch.fft.fft(w.T, Ls, dim=0)
    if d == 1:
        lp = torch.sum(w_hat.abs() ** 2, dim=1).reshape(-1, 1)
        w_hat_tight = w_hat * (lp ** (-0.5))
        return torch.fft.ifft(w_hat_tight.T, dim=1)
    else:
        N = w_hat.shape[0]
        J = w_hat.shape[1]
        assert N % d == 0, "Oh no! Decimation factor must divide signal length!"

        w_hat_tight = torch.zeros(J, N, dtype=torch.complex64)
        for j in range(N // d):
            idx = (j - torch.arange(d) * (N // d)) % N
            H = w_hat[idx, :]
            U, _, V = torch.linalg.svd(H, full_matrices=False)
            H = U @ V
            w_hat_tight[:, idx] = H.T.to(torch.complex64)
        return torch.fft.ifft(torch.fft.ifft(w_hat_tight.T, dim=1) * d**0.5, dim=0).T


def fir_tightener3000(
    w: torch.Tensor, supp: int, d: int, eps: float = 1.01, Ls: Union[int, None] = None
):
    """
    Iterative tightening procedure with fixed support for a given filterbank w
    Parameters:
        w: Impulse responses of the filterbank as 2-D Tensor torch.tensor[num_channels, signal_length].
        supp: Desired support of the resulting filterbank
        d: Decimation factor, must divide filter length!
        eps: Desired precision for the condition number
        Ls: System length (if not already given by w). If set, the resulting filterbank is padded with zeros to length Ls.
    Returns:
        Filterbank with condition number *eps* and support length *supp*. If length=supp then the resulting filterbank is the canonical tight filterbank of w.
    """
    print("Hold on, the kernels are tightening")
    if Ls is not None:
        w = torch.cat([w, torch.zeros(w.shape[0], Ls - w.shape[1])], dim=1)
    w_tight = w.clone()
    kappa = condition_number(w, d).item()
    while kappa > eps:
        w_tight = can_tight(w_tight, d)
        w_tight[:, supp:] = 0
        kappa = condition_number(w_tight, d).item()
    if Ls is None:
        return w_tight
    else:
        return w_tight[:, :supp]


def upsample(x: torch.Tensor, d: int) -> torch.Tensor:
    N = x.shape[-1] * d
    x_up = F.pad(torch.zeros_like(x), (0, N - x.shape[-1]))
    x_up[:, :, ::d] = x
    return x_up


def circ_conv(x: torch.Tensor, kernels: torch.Tensor, d: int = 1) -> torch.Tensor:
    """Circular convolution with optional downsampling.

    Performs efficient circular convolution using FFT, followed by downsampling.
    The kernels are automatically centered for proper phase alignment.

    Args:
        x (torch.Tensor): Input signal of shape (..., signal_length)
        kernels (torch.Tensor): Filter kernels of shape (num_channels, 1, kernel_length)
            or (num_channels, kernel_length)
        d (int): Downsampling factor (stride). Default: 1

    Returns:
        torch.Tensor: Convolved and downsampled output of shape (..., num_channels, output_length)

    Note:
        Uses circular convolution which assumes periodic boundary conditions.
        Kernels are automatically zero-padded and centered.

    Example:
        >>> x = torch.randn(1, 1000)
        >>> kernels = torch.randn(40, 128)
        >>> y = circ_conv(x, kernels, d=4)
    """
    L = x.shape[-1]
    x = x.to(kernels.dtype)

    kernels_long = F.pad(kernels, (0, L - kernels.shape[-1]), mode="constant", value=0)
    kernels_centered = torch.roll(kernels_long, shifts=-kernels.shape[-1] // 2, dims=-1)

    x_fft = torch.fft.fft(x, n=L, dim=-1)
    k_fft = torch.fft.fft(kernels_centered, n=L, dim=-1)
    y_fft = x_fft * k_fft
    y = torch.fft.ifft(y_fft)

    return y[:, :, ::d]


def circ_conv_transpose(
    y: torch.Tensor, kernels: torch.Tensor, d: int = 1
) -> torch.Tensor:
    """Transpose (adjoint) of circular convolution with upsampling.

    Implements the adjoint operation of circ_conv for signal reconstruction.
    Used in synthesis/decoder operations of filterbanks.

    Args:
        y (torch.Tensor): Input coefficients of shape (..., num_channels, num_frames)
        kernels (torch.Tensor): Filter kernels of shape (num_channels, 1, kernel_length)
            or (num_channels, kernel_length)
        d (int): Upsampling factor (stride). Default: 1

    Returns:
        torch.Tensor: Reconstructed signal of shape (..., 1, signal_length)

    Note:
        This is the mathematical adjoint, not the true inverse. For perfect reconstruction,
        appropriate dual frame filters should be used.

    Example:
        >>> coeffs = torch.randn(1, 40, 250)
        >>> kernels = torch.randn(40, 128)
        >>> x_recon = circ_conv_transpose(coeffs, kernels, d=4)
    """
    L = y.shape[-1] * d
    y_up = upsample(y, d)

    kernels_long = F.pad(kernels, (0, L - kernels.shape[-1]), mode="constant", value=0)
    kernels_centered = torch.roll(kernels_long, shifts=-kernels.shape[-1] // 2, dims=-1)
    kernels_synth = torch.flip(torch.conj(kernels_centered), dims=(1,))

    y_fft = torch.fft.fft(y_up, n=L, dim=-1)
    k_fft = torch.fft.fft(kernels_synth, n=L, dim=-1)
    x_fft = y_fft * k_fft
    x = torch.fft.ifft(x_fft, dim=-1)
    x = torch.sum(x, dim=-2, keepdim=True)

    return torch.roll(x, 1, -1)


####################################################################################################
################### Routines for constructing auditory filterbanks #################################
####################################################################################################


def freqtoaud(
    freq: Union[float, int, torch.Tensor],
    scale: str = "erb",
    fs: Union[int, None] = None,
) -> torch.Tensor:
    """Convert frequencies from Hz to auditory scale units.

    Transforms linear frequency values to perceptually-motivated auditory scales
    that better reflect human frequency discrimination.

    Args:
        freq (Union[float, int, torch.Tensor]): Frequency value(s) in Hz
        scale (str): Auditory scale type. One of {'erb', 'mel', 'log10', 'elelog'}. Default: 'erb'
        fs (int, optional): Sampling frequency (required for 'elelog' scale). Default: None

    Returns:
        torch.Tensor: Corresponding auditory scale units

    Raises:
        ValueError: If unsupported scale is specified or fs is missing for 'elelog'

    Note:
        - ERB: Equivalent Rectangular Bandwidth (Glasberg & Moore)
        - MEL: Mel scale (perceptually uniform pitch)
        - Bark: Bark scale (critical band rate)
        - elelog: Logarithmic scale adapted for elephant hearing

    Example:
        >>> freq_hz = torch.tensor([100, 1000, 8000])
        >>> mel_units = freqtoaud(freq_hz, scale='mel')
    """

    scale = scale.lower()

    if isinstance(freq, (int, float)):
        freq = torch.tensor(freq)

    if scale == "erb":
        # Glasberg and Moore's ERB scale
        return 9.2645 * torch.sign(freq) * torch.log(1 + torch.abs(freq) * 0.00437)

    elif scale == "mel":
        # MEL scale
        return (
            1000
            / torch.log(torch.tensor(17 / 7))
            * torch.sign(freq)
            * torch.log(1 + torch.abs(freq) / 700)
        )

    # elif scale == "bark":
    #     # Bark scale from Traunmuller (1990)
    #     return torch.sign(freq) * ((26.81 / (1 + 1960 / torch.abs(freq))) - 0.53)

    elif scale == "log10":
        # Logarithmic scale
        return torch.log10(torch.maximum(torch.ones(1), freq))

    elif scale == "elelog":
        if fs is None:
            raise ValueError(
                "Sampling frequency fs must be provided for 'elelog' scale."
            )
        fmin = 1
        fmax = fs // 2
        k = 0.88
        A = fmin / (1 - k)
        alpha = np.log10(fmax / A + k)
        return np.log((freq / A + k) / alpha)  # - np.log((fmin / A + k) / alpha)

    else:
        raise ValueError(
            f"Unsupported scale: '{scale}'. Available options are: 'mel', 'erb', 'log10', 'elelog'."
        )


def audtofreq(
    aud: Union[float, int, torch.Tensor],
    scale: str = "erb",
    fs: Union[int, None] = None,
) -> torch.Tensor:
    """Convert auditory scale units back to frequencies in Hz.

    Args:
        aud (Union[float, int, torch.Tensor]): Auditory scale values
        scale (str): Auditory scale type. One of {'erb', 'mel', 'log10', 'elelog'}. Default: 'erb'
        fs (int, optional): Sampling frequency (required for 'elelog' scale). Default: None

    Returns:
        torch.Tensor: Corresponding frequencies in Hz

    Example:
        >>> mel_units = torch.tensor([100, 1000, 2000])
        >>> freq_hz = audtofreq(mel_units, scale='mel')
    """
    if scale == "erb":
        return (1 / 0.00437) * (torch.exp(aud / 9.2645) - 1)

    elif scale == "mel":
        return (
            700
            * torch.sign(aud)
            * (torch.exp(torch.abs(aud) * torch.log(torch.tensor(17 / 7)) / 1000) - 1)
        )

    # elif scale == "bark":
    #     return torch.sign(aud) * 1960 / (26.81 / (torch.abs(aud) + 0.53) - 1)

    elif scale == "log10":
        return 10**aud

    elif scale == "elelog":
        if fs is None:
            raise ValueError(
                "Sampling frequency fs must be provided for 'elelog' scale."
            )
        fmin = 1
        fmax = fs // 2
        k = 0.88
        A = fmin / (1 - k)
        alpha = np.log10(fmax / A + k)
        return A * (np.exp(aud) * alpha - k)

    else:
        raise ValueError(
            f"Unsupported scale: '{scale}'. Available options are: 'mel', 'erb', 'log10', 'elelog'."
        )


def audspace(
    fmin: Union[float, int, torch.Tensor],
    fmax: Union[float, int, torch.Tensor],
    num_channels: int,
    scale: str = "erb",
):
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
    if scale == "log10" or scale == "elelog":
        fmin = torch.maximum(torch.tensor(fmin), torch.ones(1))
    audlimits = freqtoaud(torch.tensor([fmin, fmax]), scale)

    # Generate frequencies spaced evenly on the auditory scale
    aud_space = torch.linspace(audlimits[0], audlimits[1], num_channels)
    y = audtofreq(aud_space, scale)

    # Ensure exact endpoints
    y[0] = fmin
    y[-1] = fmax

    return y


def freqtoaud_mod(
    freq: Union[float, int, torch.Tensor],
    fc_low: Union[float, int, torch.Tensor],
    fc_high: Union[float, int, torch.Tensor],
    scale="erb",
    fs=None,
):
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
    aud_crit_low = freqtoaud(fc_low, scale, fs)
    aud_crit_high = freqtoaud(fc_high, scale, fs)
    slope_low = (freqtoaud(fc_low * 1.01, scale, fs) - aud_crit_low) / (fc_low * 0.01)
    slope_high = (freqtoaud(fc_high * 1.01, scale, fs) - aud_crit_high) / (
        fc_high * 0.01
    )

    linear_low = freq < fc_low
    linear_high = freq > fc_high
    auditory = [not x for x in (linear_low + linear_high)]

    aud = torch.zeros_like(freq, dtype=torch.float32)

    aud[linear_low] = slope_low * (freq[linear_low] - fc_low) + aud_crit_low
    aud[auditory] = freqtoaud(freq[auditory], scale, fs)
    aud[linear_high] = slope_high * (freq[linear_high] - fc_high) + aud_crit_high

    return aud


def audtofreq_mod(
    aud: Union[float, int, torch.Tensor],
    fc_low: Union[float, int, torch.Tensor],
    fc_high: Union[float, int, torch.Tensor],
    scale="erb",
    fs=None,
):
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
    aud_crit_low = freqtoaud(fc_low, scale, fs)
    aud_crit_high = freqtoaud(fc_high, scale, fs)
    slope_low = (freqtoaud(fc_low * 1.01, scale, fs) - aud_crit_low) / (fc_low * 0.01)
    slope_high = (freqtoaud(fc_high * 1.01, scale, fs) - aud_crit_high) / (fc_high * 0.01)

    linear_low = aud < aud_crit_low
    linear_high = aud > aud_crit_high
    auditory_part = [not x for x in (linear_low + linear_high)]

    freq = torch.zeros_like(aud, dtype=torch.float32)

    freq[linear_low] = (aud[linear_low] - aud_crit_low) / slope_low + fc_low
    freq[auditory_part] = audtofreq(aud[auditory_part], scale, fs)
    freq[linear_high] = (aud[linear_high] - aud_crit_high) / slope_high + fc_high

    return freq


def audspace_mod(
    fc_low: Union[float, int, torch.Tensor],
    fc_high: Union[float, int, torch.Tensor],
    fs: int,
    num_channels: int,
    scale: str = "erb",
):
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
        fc = torch.linspace(0, fs // 2, num_channels)
        return fc, freqtoaud_mod(fc, fs // 2, fs // 2, scale, fs)
    elif fc_low < fc_high:
        # Convert [0, fs//2] to modified auditory scale
        aud_min = freqtoaud_mod(torch.tensor([0]), fc_low, fc_high, scale, fs)[0]
        aud_max = freqtoaud_mod(torch.tensor([fs // 2]), fc_low, fc_high, scale, fs)[0]

        # Generate frequencies spaced evenly on the modified auditory scale
        fc_aud = torch.linspace(aud_min, aud_max, num_channels)

        # Convert back to frequency scale
        fc = audtofreq_mod(fc_aud, fc_low, fc_high, scale, fs)

        # Ensure exact endpoints
        fc[0] = 0
        fc[-1] = fs // 2

        return fc, fc_aud
    else:
        raise ValueError("There is something wrong with fc_low and fc_high.")


def fctobw(fc: Union[float, int, torch.Tensor], scale="mel"):
    """
    Computes the critical bandwidth of a filter at a given center frequency.

    Parameters:
        fc (float or ndarray): Center frequency in Hz. Must be non-negative.
        audscale (str): Auditory scale. Supported values are:
                    - 'mel': Mel scale (default)
                    - 'erb': Equivalent Rectangular Bandwidth
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
    # elif scale == "bark":
    #     bw = 25 + 75 * (1 + 1.4e-6 * fc**2) ** 0.69
    elif scale == "mel":
        bw = torch.log10(torch.tensor(17 / 7)) * (700 + fc) / 1000
    elif scale == "log10":
        bw = fc
    else:
        raise ValueError(f"Unsupported auditory scale: {scale}")

    return bw


def bwtofc(bw: Union[float, int, torch.Tensor], scale="mel"):
    """
    Computes the center frequency corresponding to a given critical bandwidth.

    Parameters:
        bw (float or ndarray): Critical bandwidth. Must be non-negative.
        scale (str): Auditory scale. Supported values are:
                 - 'mel': Mel scale
                 - 'erb': Equivalent Rectangular Bandwidth
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
    # elif scale == "bark":
    #     fc = torch.sqrt(((bw - 25) / 75) ** (1 / 0.69) / 1.4e-6)
    elif scale == "mel":
        fc = 1000 * (bw / torch.log10(torch.tensor(17 / 7))) - 700
    elif scale == "log10":
        fc = bw
    else:
        raise ValueError(f"Unsupported auditory scale: {scale}")

    return fc


def firwin(kernel_size: int, padto: int = None):
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
        g_centered = torch.roll(g_padded, int((padto - len(g)) // 2))
        return g_centered
    else:
        raise ValueError("padto must be larger than kernel_size.")


def modulate(g: torch.Tensor, fc: Union[float, int, torch.Tensor], fs: int):
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


def audfilters(
    kernel_size: Union[int, None] = None,
    num_channels: int = 96,
    fc_max: Union[float, int, None] = None,
    fs: int = None,
    L: int = None,
    supp_mult: float = 1,
    scale: str = "mel",
) -> Tuple[
    torch.Tensor,
    int,
    torch.Tensor,
    Union[int, float],
    Union[int, float],
    int,
    int,
    int,
    torch.Tensor,
]:
    """Generate auditory-inspired FIR filterbank kernels.

    Creates a bank of bandpass filters with center frequencies distributed according
    to perceptual auditory scales (mel, erb, etc.). Filters are designed with
    variable bandwidths matching critical bands of human auditory perception.

    Args:
        kernel_size (int, optional): Maximum filter kernel size. If None, computed automatically. Default: None
        num_channels (int): Number of frequency channels. Default: 96
        fc_max (float, optional): Maximum center frequency in Hz. If None, uses fs//2. Default: None
        fs (int): Sampling frequency in Hz. Default: None (required)
        L (int): Signal length in samples. If None, uses fs. Default: None
        supp_mult (float): Support multiplier for kernel sizing. Default: 1.0
        scale (str): Auditory scale. One of {'mel', 'erb', 'log10', 'elelog'}. Default: 'mel'

    Returns:
        Tuple containing:
            - kernels (torch.Tensor): Filter kernels of shape (num_channels, kernel_size)
            - d (int): Recommended stride for 50% overlap
            - fc (torch.Tensor): Center frequencies in Hz
            - fc_min (Union[int, float]): Minimum center frequency
            - fc_max (Union[int, float]): Maximum center frequency
            - kernel_min (int): Minimum kernel size used
            - kernel_size (int): Maximum kernel size used
            - Ls (int): Adjusted signal length
            - tsupp (torch.Tensor): Time support for each filter

    Raises:
        ValueError: If parameters are invalid (negative values, unsupported scale, etc.)

    Note:
        The filterbank construction follows auditory modeling principles where:
        - Low frequencies use longer filters (better frequency resolution)
        - High frequencies use shorter filters (better time resolution)
        - Bandwidth scales according to critical band theory

    Example:
        >>> kernels, stride, fc, _, _, _, _, Ls, _ = audfilters(
        ...     kernel_size=128, num_channels=40, fs=16000, scale='mel'
        ... )
        >>> print(f"Generated {kernels.shape[0]} filters with stride {stride}")
    """

    # check if all inputs are valid
    if kernel_size is not None and kernel_size <= 0:
        raise ValueError("kernel_size must be a positive integer.")
    if num_channels <= 0:
        raise ValueError("num_channels must be a positive integer.")
    # check if fs is a positive integer
    if fs is None:
        raise ValueError("sampling rate must be set.")
    if not isinstance(fs, int) or fs <= 0:
        raise ValueError("fs must be a positive integer.")
    if L is None:
        L = fs
    if not isinstance(L, int) or L <= 0:
        raise ValueError("L must be a positive integer.")
    if supp_mult < 0:
        raise ValueError("supp_mult must be a non-negative float.")
    if scale not in ["mel", "erb", "log10", "elelog"]:
        raise ValueError(
            "scale must be one of 'mel', 'erb', 'log10', or 'elelog'."
        )
    if fc_max is not None and (fc_max <= 0 or fc_max >= fs // 2):
        raise ValueError("fc_max must be a positive integer less than fs/2.")

    ####################################################################################################
    # Bandwidth conversion
    ####################################################################################################

    probeLs = 10000
    probeLg = 1000
    g_probe = firwin(probeLg, probeLs)

    # peak normalize
    gf_probe = torch.real(
        torch.fft.fft(g_probe) / torch.max(torch.abs(torch.fft.fft(g_probe)))
    )
    bw_probe = torch.norm(gf_probe) ** 2 * probeLg / probeLs / 2

    # preset bandwidth factors to get a good condition number
    if scale == "erb":
        bw_factor = 0.608
    elif scale == "mel":
        bw_factor = 111.33
    elif scale == "log10":
        bw_factor = 0.2
    # elif scale == "bark":
    #     bw_factor = 0.5
    elif scale == "elelog":
        bw_factor = 1

    bw_conversion = bw_probe / bw_factor  # * num_channels / 40

    ####################################################################################################
    # Center frequencies
    ####################################################################################################

    # checking the maximum kernel size
    if scale == "elelog":
        cycles = 10
        kernel_max = fs // 10 * cycles  # capture frequencies of 10Hz for 10 cycles

        if kernel_size is None:
            kernel_size = kernel_max

        fc_min = 10

        if fc_max is None:
            fc_max = fs // 2

        kernel_min = int(fs / fc_max * cycles)
    else:
        fsupp_min = fctobw(0, scale)

        # if not specified, set the kernel size equal to the sampling frequency fs
        if kernel_size is None:
            kernel_size = int(torch.minimum(torch.round(bw_conversion / fsupp_min * fs), torch.tensor(fs)))

        # get the bandwidth for the kernel size and the associated center frequency
        fsupp_low = bw_conversion / kernel_size * fs
        fc_min = bwtofc(fsupp_low, scale)

        if fc_max is None:
            fc_max = fs // 2

        # get the bandwidth for the maximum center frequency and the associated kernel size
        fsupp_high = fctobw(fc_max, scale)
        kernel_min = int(torch.round(bw_conversion / fsupp_high * fs))

        if fc_min >= fc_max:
            fc_max = fc_min
            kernel_min = kernel_size
            Warning(
                f"fc_max was increased to {fc_min} to enable the kernel size of {kernel_size}."
            )

    # get center frequencies
    [fc, _] = audspace_mod(fc_min, fc_max, fs, num_channels, scale)

    num_low = torch.where(fc < fc_min)[0].shape[0]
    num_high = torch.where(fc > fc_max)[0].shape[0]
    num_aud = num_channels - num_low - num_high

    ####################################################################################################
    # Frequency and time supports
    ####################################################################################################

    # get time supports
    tsupp_low = (torch.ones(num_low) * kernel_size).int()
    tsupp_high = (torch.ones(num_high) * kernel_min).int()
    if scale == "elelog":
        tsupp_aud = (
            torch.minimum(
                torch.tensor(kernel_size),
                torch.round(fs / fc[num_low : num_low + num_aud] * cycles),
            )
        ).int()
        tsupp = torch.concatenate([tsupp_low, tsupp_aud, tsupp_high]).int()
    else:
        if num_low + num_high == num_channels:
            fsupp = fctobw(fc_max, scale)
            tsupp = tsupp_low
        else:
            fsupp = fctobw(fc[num_low : num_low + num_aud], scale)
            tsupp_aud = torch.round(bw_conversion / fsupp * fs)
            tsupp = torch.concatenate([tsupp_low, tsupp_aud, tsupp_high]).int()

    if supp_mult < 1:
        tsupp = torch.max(
            torch.round(tsupp * supp_mult), torch.ones_like(tsupp) * 8
        ).int()
    else:
        tsupp = torch.min(
            torch.round(tsupp * supp_mult), torch.ones_like(tsupp) * L
        ).int()

    kernel_min = tsupp.min()
    kernel_size = tsupp.max()

    # Decimation factor (stride) for 50% overlap
    d = torch.maximum(kernel_min // 2, torch.tensor(1))
    Ls = int(torch.ceil(L / d) * d)

    ####################################################################################################
    # Generate filters
    ####################################################################################################

    g = torch.zeros((num_channels, kernel_size), dtype=torch.cfloat)

    g[0, :] = torch.sqrt(d) * firwin(kernel_size) / torch.sqrt(torch.tensor(2))
    g[-1, :] = (
        torch.sqrt(d)
        * modulate(firwin(tsupp[-1], kernel_size), fs // 2, fs)
        / torch.sqrt(torch.tensor(2))
    )

    for m in range(1, num_channels - 1):
        g[m, :] = torch.sqrt(d) * modulate(firwin(tsupp[m], kernel_size), fc[m], fs)

    # _, B = frame_bounds(g, d, Ls)
    # g = g / B**0.5

    return g, int(d), fc, fc_min, fc_max, kernel_min, kernel_size, Ls, tsupp


####################################################################################################
####################################################################################################
####################################################################################################


def response(g: np.ndarray, fs: int) -> np.ndarray:
    """Compute frequency responses of filter kernels.

    Args:
        g (np.ndarray): Filter kernels of shape (num_channels, kernel_size)
        fs (int): Sampling frequency for frequency axis scaling

    Returns:
        np.ndarray: Magnitude-squared frequency responses of shape (2*num_channels, fs//2)

    Note:
        Computes responses for both analysis and conjugate filters.
    """
    g_full = np.concatenate([g, np.conj(g)], axis=0)
    G = np.abs(np.fft.fft(g_full, fs, axis=1)[:, : fs // 2]) ** 2

    return G


def plot_response(
    g: np.ndarray,
    fs: int,
    scale: str = "mel",
    plot_scale: bool = False,
    fc_min: Union[float, None] = None,
    fc_max: Union[float, None] = None,
    decoder: bool = False,
) -> None:
    """Plot frequency responses and auditory scale visualization of filters.

    Creates comprehensive visualization showing individual filter responses,
    total power spectral density, and optional auditory scale mapping.

    Args:
        g (np.ndarray): Filter kernels of shape (num_channels, kernel_size)
        fs (int): Sampling frequency in Hz for frequency axis scaling
        scale (str): Auditory scale name for scale plotting. Default: 'mel'
        plot_scale (bool): Whether to plot the auditory scale mapping. Default: False
        fc_min (float, optional): Lower transition frequency for scale visualization. Default: None
        fc_max (float, optional): Upper transition frequency for scale visualization. Default: None
        decoder (bool): Whether filters are for synthesis (affects plot titles). Default: False

    Note:
        This function displays plots and does not return values.
        Creates 2-3 subplots depending on plot_scale parameter.

    Example:
        >>> filters = np.random.randn(40, 128)
        >>> plot_response(filters, fs=16000, scale='mel', plot_scale=True)
    """
    num_channels = g.shape[0]

    g_hat = response(g, fs)
    g_hat_pos = g_hat[:num_channels, :]
    g_hat_pos[np.isnan(g_hat_pos)] = 0
    psd = np.sum(g_hat, axis=0)
    psd[np.isnan(psd)] = 0

    if plot_scale:
        plt.figure(figsize=(8, 2))
        freq_samples, _ = audspace_mod(fc_min, fc_max, fs, num_channels, scale)
        freqs = torch.linspace(0, fs // 2, fs // 2)

        auds = freqtoaud_mod(freqs, fc_min, fc_max, scale, fs).numpy()
        auds_orig = freqtoaud(freqs, scale, fs).numpy()

        plt.scatter(
            freq_samples.numpy(),
            freqtoaud_mod(freq_samples, fc_min, fc_max, scale, fs).numpy(),
            color="black",
            label="Center frequencies",
            linewidths=0.04,
        )
        plt.plot(freqs, auds, color="black", label=f"ISAC {scale}-scale")
        plt.plot(
            freqs,
            auds_orig,
            color="black",
            linestyle="--",
            alpha=0.5,
            label=f"Original {scale}-scale",
        )

        if fc_min is not None:
            plt.axvline(fc_min, color="black", alpha=0.25)
            plt.fill_betweenx(
                y=[auds[0] - 1, auds[-1] * 1.1],
                x1=0,
                x2=fc_min,
                color="gray",
                alpha=0.25,
            )
            plt.fill_betweenx(
                y=[auds[0] - 1, auds[-1] * 1.1],
                x1=fc_min,
                x2=fs // 2,
                color="gray",
                alpha=0.1,
            )

        if fc_max is not None:
            plt.axvline(fc_max, color="black", alpha=0.25)
            plt.fill_betweenx(
                y=[auds[0] - 1, auds[-1] * 1.1],
                x1=0,
                x2=fc_max,
                color="gray",
                alpha=0.25,
            )
            plt.fill_betweenx(
                y=[auds[0] - 1, auds[-1] * 1.1],
                x1=fc_max,
                x2=fs // 2,
                color="gray",
                alpha=0.1,
            )

        plt.xlim([0, fs // 2])
        plt.ylim([auds[0] - 1, auds[-1] * 1.1])
        plt.xlabel("Frequency (Hz)")
        # text_x = fc_min / 2
        # text_y = auds[-1]
        # plt.text(text_x, text_y, 'linear', color='black', ha='center', va='center', fontsize=12, alpha=0.75)
        # plt.text(text_x + fc_min - 1, text_y, 'ERB', color='black', ha='center', va='center', fontsize=12, alpha=0.75)
        # plt.title(f"ISAC {scale}-scale")

        plt.ylabel("Auditory Units")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

    fig, ax = plt.subplots(2, 1, figsize=(6, 3), sharex=True)

    fr_id = 0
    psd_id = 1

    f_range = np.linspace(0, fs // 2, fs // 2)
    ax[fr_id].set_xlim([0, fs // 2])
    ax[fr_id].set_ylim([0, np.max(g_hat_pos) * 1.1])
    ax[fr_id].plot(f_range, g_hat_pos.T)
    if decoder:
        ax[fr_id].set_title("PSDs of the synthesis filters")
    if not decoder:
        ax[fr_id].set_title("PSDs of the analysis filters")
    # ax[fr_id].set_xlabel('Frequency [Hz]')
    ax[fr_id].set_ylabel("Magnitude")

    ax[psd_id].plot(f_range, psd)
    ax[psd_id].set_xlim([0, fs // 2])
    ax[psd_id].set_ylim([0, np.max(psd) * 1.1])
    ax[psd_id].set_title("Total PSD")
    ax[psd_id].set_xlabel("Frequency [Hz]")
    ax[psd_id].set_ylabel("Magnitude")

    if fc_min is not None:
        ax[fr_id].fill_betweenx(
            y=[0, np.max(g_hat) * 1.1], x1=0, x2=fc_min, color="gray", alpha=0.25
        )
        ax[fr_id].fill_betweenx(
            y=[0, np.max(g_hat) * 1.1], x1=fc_min, x2=fs // 2, color="gray", alpha=0.1
        )
        ax[psd_id].fill_betweenx(
            y=[0, np.max(psd) * 1.1], x1=0, x2=fc_min, color="gray", alpha=0.25
        )
        ax[psd_id].fill_betweenx(
            y=[0, np.max(psd) * 1.1], x1=fc_min, x2=fs // 2, color="gray", alpha=0.1
        )

    if fc_max is not None:
        ax[fr_id].fill_betweenx(
            y=[0, np.max(g_hat) * 1.1], x1=0, x2=fc_max, color="gray", alpha=0.25
        )
        ax[fr_id].fill_betweenx(
            y=[0, np.max(g_hat) * 1.1], x1=fc_max, x2=fs // 2, color="gray", alpha=0.1
        )
        ax[psd_id].fill_betweenx(
            y=[0, np.max(psd) * 1.1], x1=0, x2=fc_max, color="gray", alpha=0.25
        )
        ax[psd_id].fill_betweenx(
            y=[0, np.max(psd) * 1.1], x1=fc_max, x2=fs // 2, color="gray", alpha=0.1
        )

    plt.tight_layout()
    plt.show()


def ISACgram(
    c: torch.Tensor,
    fc: Union[torch.Tensor, None] = None,
    L: Union[int, None] = None,
    fs: Union[int, None] = None,
    fmax: Union[float, None] = None,
    log_scale: bool = False,
    vmin: Union[float, None] = None,
    cmap: str = "inferno",
) -> None:
    """Plot time-frequency representation of filterbank coefficients.

    Creates a spectrogram-like visualization with frequency on y-axis and time on x-axis.
    Supports logarithmic scaling and frequency range limitation for better visualization.

    Args:
        c (torch.Tensor): Filterbank coefficients of shape (batch_size, num_channels, num_frames)
        fc (torch.Tensor, optional): Center frequencies in Hz for y-axis labeling. Default: None
        L (int, optional): Original signal length for time axis scaling. Default: None
        fs (int, optional): Sampling frequency for time axis scaling. Default: None
        fmax (float, optional): Maximum frequency to display in Hz. Default: None
        log_scale (bool): Whether to apply log10 scaling to coefficients. Default: False
        vmin (float, optional): Minimum value for dynamic range clipping. Default: None
        cmap (str): Matplotlib colormap name. Default: 'inferno'

    Note:
        This function displays a plot and does not return values.
        Only processes the first batch element if batch_size > 1.

    Example:
        >>> coeffs = torch.randn(1, 40, 250)
        >>> fc = torch.linspace(100, 8000, 40)
        >>> ISACgram(coeffs, fc=fc, L=16000, fs=16000, log_scale=True)
    """
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    c = c[0].detach().cpu().numpy()

    if log_scale:
        c = np.log10(np.abs(c) + 1e-10)

    if fc is not None and fmax is not None:
        c = c[: np.argmax(fc > fmax), :]

    if vmin is not None:
        mesh = ax.pcolor(c, cmap=cmap, vmin=np.min(c) * vmin)
    else:
        mesh = ax.pcolor(c, cmap=cmap)

    # Add colorbar
    plt.colorbar(mesh, ax=ax)

    # Axis labeling
    if fc is not None:
        locs = np.linspace(0, c.shape[0] - 1, min(len(fc), 10)).astype(int)
        ax.set_yticks(locs)
        ax.set_yticklabels([int(np.round(fc[i])) for i in locs])

        # X-axis: time
        num_time_labels = 10
        xticks = np.linspace(0, c.shape[1] - 1, num_time_labels)
        ax.set_xticks(xticks)
        ax.set_xticklabels(
            [np.round(x, 1) for x in np.linspace(0, L // fs, num_time_labels)]
        )

        ax.set_ylabel("Frequency [Hz]")
        ax.set_xlabel("Time [s]")
    else:
        ax.set_ylabel("Frequency index")
        ax.set_xlabel("Time samples")

    plt.tight_layout()
    #plt.savefig('/Users/dani/Library/Mobile Documents/com~apple~CloudDocs/Documents/PhD/ELECOM/IBAC/rumble_avg.png', dpi=600)
    plt.show()
