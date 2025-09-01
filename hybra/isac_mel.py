from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from hybra.utils import ISACgram as ISACgram_
from hybra.utils import audfilters, circ_conv
from hybra.utils import plot_response as plot_response_


class ISACSpec(nn.Module):
    """ISAC spectrogram filterbank for time-frequency analysis.

    ISACSpec combines ISAC (Invertible and Stable Auditory filterbank with Customizable kernels)
    with temporal averaging to produce spectrogram-like representations. The filterbank applies
    auditory-inspired filters followed by temporal smoothing for robust feature extraction.

    Args:
        kernel_size (int, optional): Size of the filter kernels. If None, computed automatically. Default: None
        num_channels (int): Number of frequency channels. Default: 40
        stride (int, optional): Stride of the filterbank. If None, uses 50% overlap. Default: None
        fc_max (float, optional): Maximum frequency on the auditory scale in Hz.
            If None, uses fs//2. Default: None
        fmax (float, optional): Maximum frequency for output truncation in Hz. Default: None
        fs (int): Sampling frequency in Hz. Default: None (required)
        L (int): Signal length in samples. Default: None (required)
        supp_mult (float): Support multiplier for kernel sizing. Default: 1.0
        scale (str): Auditory scale type. One of {'mel', 'erb', 'log10', 'elelog'}.
            'elelog' is adapted for elephant hearing. Default: 'mel'
        power (float): Power applied to coefficients before averaging. Default: 2.0
        avg_size (int, optional): Size of the temporal averaging kernel.
            If None, computed automatically. Default: None
        is_log (bool): Whether to apply logarithm to the output. Default: False
        is_encoder_learnable (bool): Whether encoder kernels are learnable parameters. Default: False
        is_avg_learnable (bool): Whether averaging kernels are learnable parameters. Default: False
        verbose (bool): Whether to print filterbank information during initialization. Default: False

    Note:
        The temporal averaging provides robustness to time variations while preserving
        spectral characteristics. The power parameter controls the nonlinearity applied
        before averaging.

    Example:
        >>> spectrogram = ISACSpec(num_channels=40, fs=16000, L=16000, power=2.0)
        >>> x = torch.randn(1, 16000)
        >>> spec = spectrogram(x)
    """

    def __init__(
        self,
        kernel_size: Union[int, None] = None,
        num_channels: int = 40,
        stride: Union[int, None] = None,
        fc_max: Union[float, int, None] = None,
        fmax: Union[int, None] = None,
        fs: int = None,
        L: int = None,
        supp_mult: float = 1,
        scale: str = "mel",
        power: float = 2.0,
        avg_size: int = None,
        is_log=False,
        is_encoder_learnable=False,
        is_avg_learnable=False,
        verbose: bool = False,
    ):
        super().__init__()

        [aud_kernels, d_50, fc, fc_min, fc_max, kernel_min, kernel_size, Ls, tsupp] = (
            audfilters(
                kernel_size=kernel_size,
                num_channels=num_channels,
                fc_max=fc_max,
                fs=fs,
                L=L,
                supp_mult=supp_mult,
                scale=scale,
            )
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

        if fmax is not None:
            num_channels = torch.sum(fc <= fmax)
            aud_kernels = aud_kernels[:num_channels, :]

        self.num_channels = num_channels
        self.stride = d
        self.kernel_size = kernel_size
        self.kernel_min = kernel_min
        self.fs = fs
        self.fc = fc
        self.fc_min = fc_min
        self.fc_max = fc_max
        self.Ls = Ls
        self.is_log = is_log
        self.power = power

        if is_encoder_learnable:
            self.register_parameter(
                "kernels", nn.Parameter(aud_kernels, requires_grad=True)
            )
        else:
            self.register_buffer("kernels", aud_kernels)

        if avg_size is None:
            averaging_kernels = torch.ones(
                self.num_channels,
                1,
                min(1024, self.kernel_size) // self.stride // 2 * 2 + 1,
            )
            # avg_size = torch.maximum(torch.tensor(2), tsupp // self.stride)
            # self.avg_max = avg_size.max().item()
            # averaging_kernels = torch.ones(self.num_channels, 1, self.avg_max)
            # for i in range(self.num_channels):
            #    averaging_kernels[i, 0, self.avg_max//2:self.avg_max//2+avg_size[i]] = 1.0
        else:
            averaging_kernels = torch.ones([self.num_channels, 1, avg_size])

        if is_avg_learnable:
            self.register_parameter(
                "avg_kernels", nn.Parameter(averaging_kernels, requires_grad=True)
            )
        else:
            self.register_buffer("avg_kernels", averaging_kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ISACSpec filterbank.

        Args:
            x (torch.Tensor): Input signal of shape (batch_size, signal_length) or (signal_length,)

        Returns:
            torch.Tensor: Spectrogram coefficients of shape (batch_size, num_channels, num_frames)

        Note:
            The output is temporally averaged and optionally log-scaled for robustness.
        """
        x = circ_conv(x.unsqueeze(1), self.kernels, self.stride).abs()  # **self.power
        x = F.conv1d(
            x,
            self.avg_kernels.to(x.device),
            groups=self.num_channels,
            stride=1,
            padding="same",
        )

        if self.is_log:
            x = torch.log(x + 1e-10)
        return x

    def ISACgram(
        self,
        x: torch.Tensor,
        fmax: Union[float, None] = None,
        vmin: Union[float, None] = None,
        log_scale: bool = False,
    ) -> None:
        """Plot time-frequency spectrogram representation of the signal.

        Args:
            x (torch.Tensor): Input signal to visualize
            fmax (float, optional): Maximum frequency to display in Hz. Default: None
            vmin (float, optional): Minimum value for dynamic range clipping. Default: None
            log_scale (bool): Whether to apply log scaling to coefficients. Default: False

        Note:
            This method displays a plot and does not return values.
        """
        with torch.no_grad():
            coefficients = self.forward(x).abs()
        ISACgram_(
            c=coefficients,
            fc=self.fc,
            L=self.Ls,
            fs=self.fs,
            fmax=fmax,
            vmin=vmin,
            log_scale=log_scale,
        )

    def plot_response(self) -> None:
        """Plot frequency response of the analysis filters.

        Note:
            This method displays a plot and does not return values.
        """
        plot_response_(
            g=(self.kernels).detach().numpy(),
            fs=self.fs,
            scale=True,
            fc_min=self.fc_min,
            fc_max=self.fc_max,
        )
