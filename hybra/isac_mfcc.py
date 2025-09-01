from typing import Union

import torch
import torch.nn as nn
from torchaudio.functional import create_dct
from torchaudio.transforms import AmplitudeToDB

from hybra import ISACSpec
from hybra.utils import ISACgram as ISACgram_
from hybra.utils import plot_response as plot_response_


class ISACCC(nn.Module):
    """ISAC Cepstral Coefficients (ISACCC) extractor for speech features.

    ISACCC computes cepstral coefficients from ISAC (Invertible and Stable Auditory
    filterbank with Customizable kernels) spectrograms using the Discrete Cosine
    Transform (DCT). This provides compact features suitable for speech recognition
    and audio classification tasks.

    Args:
        kernel_size (int, optional): Size of the filter kernels. If None, computed automatically. Default: None
        num_channels (int): Number of frequency channels. Default: 40
        stride (int, optional): Stride of the filterbank. If None, uses 50% overlap. Default: None
        num_cc (int): Number of cepstral coefficients to extract. Default: 13
        fc_max (float, optional): Maximum frequency on the auditory scale in Hz.
            If None, uses fs//2. Default: None
        fmax (float, optional): Maximum frequency for ISACSpec computation in Hz. Default: None
        fs (int): Sampling frequency in Hz. Default: 16000
        L (int): Signal length in samples. Default: 16000
        supp_mult (float): Support multiplier for kernel sizing. Default: 1.0
        power (float): Power applied to ISACSpec coefficients. Default: 2.0
        scale (str): Auditory scale type. One of {'mel', 'erb', 'log10', 'elelog'}.
            'elelog' is adapted for elephant hearing. Default: 'mel'
        is_log (bool): Whether to apply log instead of dB conversion. Default: False
        verbose (bool): Whether to print filterbank information during initialization. Default: False

    Raises:
        ValueError: If num_cc > num_channels

    Note:
        The DCT is applied with orthonormal basis functions for energy preservation.
        The number of cepstral coefficients should typically be much smaller than
        the number of frequency channels for dimensionality reduction.

    Example:
        >>> mfcc_extractor = ISACCC(num_channels=40, num_cc=13, fs=16000, L=16000)
        >>> x = torch.randn(1, 16000)
        >>> cepstral_coeffs = mfcc_extractor(x)
    """

    def __init__(
        self,
        kernel_size: Union[int, None] = None,
        num_channels: int = 40,
        stride: Union[int, None] = None,
        num_cc: int = 13,
        fc_max: Union[float, int, None] = None,
        fmax: Union[float, int, None] = None,
        fs: int = 16000,
        L: int = 16000,
        supp_mult: float = 1,
        power: float = 2.0,
        scale: str = "mel",
        is_log: bool = False,
        verbose: bool = False,
    ):
        super().__init__()

        self.isac = ISACSpec(
            kernel_size=kernel_size,
            num_channels=num_channels,
            stride=stride,
            fc_max=fc_max,
            fs=fs,
            L=L,
            supp_mult=supp_mult,
            power=power,
            scale=scale,
            is_log=False,
            verbose=verbose,
        )

        self.fc_min = self.isac.fc_min
        self.fc_max = self.isac.fc_max
        self.kernel_min = self.isac.kernel_min
        self.fs = fs
        self.Ls = self.isac.Ls
        self.num_channels = num_channels
        self.num_cc = num_cc
        self.fmax = fmax
        self.is_log = is_log

        if self.num_cc > num_channels:
            raise ValueError("Cannot select more cepstrum coefficients than # channels")

        if self.fmax is not None:
            self.num_channels = torch.sum(self.isac.fc <= self.fmax)

        dct_mat = create_dct(self.num_cc, self.num_channels, norm="ortho").to(
            self.isac.kernels.device
        )
        self.register_buffer("dct_mat", dct_mat)

        self.amplitude_to_DB = AmplitudeToDB("power", 80.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute ISAC cepstral coefficients.

        Args:
            x (torch.Tensor): Input signal of shape (batch_size, signal_length) or (signal_length,)

        Returns:
            torch.Tensor: Cepstral coefficients of shape (batch_size, num_cc, num_frames)

        Note:
            The process involves: ISAC spectrogram -> log/dB conversion -> DCT transform.
        """
        coeff = self.isac(x)
        if self.fmax is not None:
            coeff = coeff[:, : self.num_channels, :]
        if self.is_log:
            coeff = torch.log(coeff + 1e-10)
        else:
            coeff = self.amplitude_to_DB(coeff)
        return torch.matmul(coeff.transpose(-1, -2), self.dct_mat).transpose(-1, -2)

    def ISACgram(self, x: torch.Tensor) -> None:
        """Plot cepstral coefficients representation.

        Args:
            x (torch.Tensor): Input signal to visualize

        Note:
            This method displays a plot of the cepstral coefficients and does not return values.
        """
        with torch.no_grad():
            coefficients = self.forward(x)
        ISACgram_(coefficients, None, self.Ls, self.fs)

    def plot_response(self) -> None:
        """Plot frequency response of the underlying ISAC filters.

        Note:
            This method displays a plot and does not return values.
        """
        plot_response_(
            g=(self.isac.kernels[: self.num_channels, :]).detach().numpy(),
            fs=self.isac.fs,
            scale=True,
            fc_min=self.isac.fc_min,
            fc_max=self.isac.fc_max,
        )
