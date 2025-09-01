from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from hybra._fit_dual import tight_hybra
from hybra.utils import ISACgram as ISACgram_
from hybra.utils import (
    audfilters,
    circ_conv,
    circ_conv_transpose,
    condition_number,
    frame_bounds,
    plot_response,
)


class HybrA(nn.Module):
    """Hybrid Auditory filterbank combining fixed and learnable components.

    HybrA (Hybrid Auditory) filterbanks extend ISAC by combining fixed auditory-inspired
    filters with learnable filters through channel-wise convolution. This hybrid approach
    enables data-driven adaptation while maintaining perceptual auditory characteristics
    and frame-theoretic stability guarantees.

    Args:
        kernel_size (int): Kernel size of the auditory filterbank. Default: 128
        learned_kernel_size (int): Kernel size of the learned filterbank. Default: 23
        num_channels (int): Number of frequency channels. Default: 40
        stride (int, optional): Stride of the auditory filterbank. If None, uses 50% overlap. Default: None
        fc_max (float, optional): Maximum frequency on the auditory scale in Hz. If None, uses fs//2. Default: None
        fs (int): Sampling frequency in Hz. Default: None (required)
        L (int): Signal length in samples. Default: None (required)
        supp_mult (float): Support multiplier for kernel sizing. Default: 1.0
        scale (str): Auditory scale type. One of {'mel', 'erb', 'log10', 'elelog'}.
            'elelog' is adapted for elephant hearing. Default: 'mel'
        tighten (bool): Whether to apply tightening to improve frame bounds. Default: False
        det_init (bool): Whether to initialize learned filters as diracs (True) or randomly (False). Default: False
        verbose (bool): Whether to print filterbank information during initialization. Default: False

    Note:
        The hybrid construction h_m = g_m ⊛ ℓ_m combines ISAC auditory filters (g_m)
        with compact learnable filters (ℓ_m) through convolution. This maintains the
        perceptual benefits of auditory scales while enabling data-driven optimization
        and preserving perfect reconstruction properties.

    Example:
        >>> filterbank = HybrA(kernel_size=128, num_channels=40, fs=16000, L=16000)
        >>> x = torch.randn(1, 16000)
        >>> coeffs = filterbank(x)
        >>> reconstructed = filterbank.decoder(coeffs)
    """

    def __init__(
        self,
        kernel_size: int = 128,
        learned_kernel_size: int = 23,
        num_channels: int = 40,
        stride: Union[int, None] = None,
        fc_max: Union[float, int, None] = None,
        fs: int = None,
        L: int = None,
        supp_mult: float = 1,
        scale: str = "mel",
        tighten: bool = False,
        det_init: bool = False,
        verbose: bool = False,
    ):
        super().__init__()

        [aud_kernels, d_50, fc, _, fc_max, kernel_min, kernel_size, Ls] = audfilters(
            kernel_size=kernel_size,
            num_channels=num_channels,
            fc_max=fc_max,
            fs=fs,
            L=L,
            supp_mult=supp_mult,
            scale=scale,
        )

        if stride is not None:
            d = stride
            Ls = int(torch.ceil(torch.tensor(Ls / d)) * d)
        else:
            d = d_50

        if verbose:
            print(f"Max. kernel size: {kernel_size}")
            print(f"Min. kernel size: {kernel_min}")
            print(f"Number of channels: {num_channels}")
            print(f"Stride for min. 50% overlap: {d_50}")
            print(f"Signal length: {Ls}")

        self.register_buffer("kernels", aud_kernels)
        self.kernel_size = kernel_size
        self.learned_kernel_size = learned_kernel_size
        self.stride = d
        self.num_channels = num_channels
        self.fc = fc
        self.Ls = Ls
        self.fs = fs

        if det_init:
            learned_kernels = torch.zeros(
                [self.num_channels, 1, self.learned_kernel_size]
            )
            learned_kernels[:, 0, 0] = 1.0
        else:
            learned_kernels = torch.randn(
                [self.num_channels, 1, self.learned_kernel_size]
            ) / torch.sqrt(torch.tensor(self.learned_kernel_size * self.num_channels))
            learned_kernels = learned_kernels / torch.norm(
                learned_kernels, p=1, dim=-1, keepdim=True
            )

        learned_kernels = learned_kernels.to(self.kernels.dtype)

        if tighten:
            learned_kernels = tight_hybra(
                self.kernels, learned_kernels, d, Ls, fs, fit_eps=1.0001, max_iter=1000
            )

        self.learned_kernels = nn.Parameter(learned_kernels, requires_grad=True)

        self.hybra_kernels = F.conv1d(
            self.kernels.squeeze(1),
            self.learned_kernels,
            groups=self.num_channels,
            padding="same",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the HybrA filterbank.

        Args:
            x (torch.Tensor): Input signal of shape (batch_size, signal_length) or (signal_length,)

        Returns:
            torch.Tensor: Filterbank coefficients of shape (batch_size, num_channels, num_frames)
        """
        hybra_kernels = F.conv1d(
            self.kernels.squeeze(1),
            self.learned_kernels,
            groups=self.num_channels,
            padding="same",
        )
        self.hybra_kernels = hybra_kernels.clone().detach()

        return circ_conv(x.unsqueeze(1), hybra_kernels, self.stride)

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        """Encode signal using fixed hybrid kernels (no gradient computation).

        Args:
            x (torch.Tensor): Input signal of shape (batch_size, signal_length) or (signal_length,)

        Returns:
            torch.Tensor: Filterbank coefficients of shape (batch_size, num_channels, num_frames)

        Note:
            Use forward() method during training to enable gradient computation.
            This method uses pre-computed kernels for inference.
        """
        return circ_conv(x.unsqueeze(1), self.hybra_kernels, self.stride)

    def decoder(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct signal from filterbank coefficients.

        Args:
            x (torch.Tensor): Filterbank coefficients of shape (batch_size, num_channels, num_frames)

        Returns:
            torch.Tensor: Reconstructed signal of shape (batch_size, signal_length)

        Note:
            Uses frame bounds normalization for approximate perfect reconstruction.
        """
        _, B = frame_bounds(self.hybra_kernels.squeeze(1), self.stride, None)
        return circ_conv_transpose(x, self.hybra_kernels / B, self.stride).squeeze(1)

    # plotting methods

    def ISACgram(self, x: torch.Tensor, fmax: Union[float, None] = None) -> None:
        """Plot time-frequency representation of the signal.

        Args:
            x (torch.Tensor): Input signal to visualize
            fmax (float, optional): Maximum frequency to display in Hz. Default: None

        Note:
            This method displays a plot and does not return values.
        """
        with torch.no_grad():
            coefficients = self.forward(x).abs()
        ISACgram_(c=coefficients, fc=self.fc, L=self.Ls, fs=self.fs, fmax=fmax)

    def plot_response(self) -> None:
        """Plot frequency response of the analysis filters.

        Note:
            This method displays a plot and does not return values.
        """
        plot_response((self.hybra_kernels).squeeze().cpu().detach().numpy(), self.fs)

    def plot_decoder_response(self) -> None:
        """Plot frequency response of the synthesis (decoder) filters.

        Note:
            This method displays a plot and does not return values.
        """
        plot_response(
            (self.hybra_kernels).squeeze().cpu().detach().numpy(), self.fs, decoder=True
        )

    @property
    def condition_number(self, learnable: bool = False) -> Union[torch.Tensor, float]:
        """Compute condition number of the filterbank.

        Args:
            learnable (bool): If True, returns tensor for gradient computation.
                If False, returns scalar value. Default: False

        Returns:
            Union[torch.Tensor, float]: Condition number of the frame operator

        Note:
            Lower condition numbers indicate better numerical stability.
            Values close to 1.0 indicate tight frames.
        """
        kernels = (self.hybra_kernels).squeeze()
        if learnable:
            return condition_number(kernels, self.stride, self.Ls)
        else:
            return condition_number(kernels, self.stride, self.Ls).item()
