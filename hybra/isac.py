from typing import Union

import torch
import torch.nn as nn

from hybra._fit_dual import fit, tight
from hybra.utils import ISACgram as ISACgram_
from hybra.utils import (
    audfilters,
    circ_conv,
    circ_conv_transpose,
    condition_number,
    frame_bounds,
)
from hybra.utils import plot_response as plot_response_


class ISAC(nn.Module):
    """ISAC (Invertible and Stable Auditory filterbank with Customizable kernels) filterbank.

    ISAC filterbanks are invertible and stable, perceptually-motivated filterbanks specifically
    designed for machine learning integration. They provide perfect reconstruction properties
    with customizable kernel sizes and auditory-inspired frequency decomposition.

    Args:
        kernel_size (int): Size of the filter kernels. Default: 128
        num_channels (int): Number of frequency channels. Default: 40
        fc_max (float, optional): Maximum frequency on the auditory scale in Hz.
            If None, uses fs//2. Default: None
        stride (int, optional): Stride of the filterbank. If None, uses 50% overlap. Default: None
        fs (int): Sampling frequency in Hz. Default: None (required)
        L (int): Signal length in samples. Default: None (required)
        supp_mult (float): Support multiplier for kernel sizing. Default: 1.0
        scale (str): Auditory scale type. One of {'mel', 'erb', 'log10', 'elelog'}.
            'elelog' is adapted for elephant hearing. Default: 'mel'
        tighten (bool): Whether to apply tightening for better frame bounds. Default: False
        is_encoder_learnable (bool): Whether encoder kernels are learnable parameters. Default: False
        fit_decoder (bool): Whether to compute approximate perfect reconstruction decoder. Default: False
        is_decoder_learnable (bool): Whether decoder kernels are learnable parameters. Default: False
        verbose (bool): Whether to print filterbank information during initialization. Default: False

    Note:
        ISAC filterbanks provide invertible and stable transforms with perfect reconstruction.
        The filters have user-defined maximum temporal support and can serve as learnable
        convolutional kernels. The frame bounds can be controlled through the `tighten`
        parameter for numerical stability.

    Example:
        >>> filterbank = ISAC(kernel_size=128, num_channels=40, fs=16000, L=16000)
        >>> x = torch.randn(1, 16000)
        >>> coeffs = filterbank(x)
        >>> reconstructed = filterbank.decoder(coeffs)
    """

    def __init__(
        self,
        kernel_size: Union[int, None] = 128,
        num_channels: int = 40,
        fc_max: Union[float, int, None] = None,
        stride: Union[int, None] = None,
        fs: int = None,
        L: int = None,
        supp_mult: float = 1,
        scale: str = "mel",
        tighten=False,
        is_encoder_learnable=False,
        fit_decoder=False,
        is_decoder_learnable=False,
        verbose: bool = False,
    ):
        super().__init__()

        [aud_kernels, d_50, fc, fc_min, fc_max, kernel_min, kernel_size, Ls, _] = (
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

        self.aud_kernels = aud_kernels
        self.kernel_size = kernel_size
        self.kernel_min = kernel_min
        self.fc = fc
        self.fc_min = fc_min
        self.fc_max = fc_max
        self.stride = d
        self.Ls = Ls
        self.fs = fs
        self.scale = scale
        self.fit_decoder = fit_decoder

        # optional preprocessing

        if tighten:
            aud_kernels = tight(aud_kernels, d, Ls, fs, fit_eps=1.0001, max_iter=1000)

        if fit_decoder:
            decoder_kernels = fit(
                aud_kernels.clone(), d, Ls, fs, decoder_fit_eps=0.0001, max_iter=10000
            )
        else:
            decoder_kernels = aud_kernels.clone()

        # set the parameters for the convolutional layers

        if is_encoder_learnable:
            self.register_buffer(
                "kernels", nn.Parameter(aud_kernels, requires_grad=True)
            )
        else:
            self.register_buffer("kernels", aud_kernels)

        if is_decoder_learnable:
            self.register_buffer(
                "decoder_kernels", nn.Parameter(decoder_kernels, requires_grad=True)
            )
        else:
            self.register_buffer("decoder_kernels", decoder_kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ISAC filterbank.

        Args:
            x (torch.Tensor): Input signal of shape (batch_size, signal_length) or (signal_length,)

        Returns:
            torch.Tensor: Filterbank coefficients of shape (batch_size, num_channels, num_frames)
        """
        return circ_conv(x.unsqueeze(1), self.kernels, self.stride)

    def decoder(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct signal from ISAC coefficients.

        Args:
            x (torch.Tensor): Filterbank coefficients of shape (batch_size, num_channels, num_frames)

        Returns:
            torch.Tensor: Reconstructed signal of shape (batch_size, signal_length)

        Note:
            Uses frame bounds normalization for approximate perfect reconstruction.
        """
        _, B = frame_bounds(self.decoder_kernels, self.stride, self.Ls)
        return circ_conv_transpose(x, self.decoder_kernels / B, self.stride).squeeze(1)

    # plotting methods

    def ISACgram(
        self,
        x: torch.Tensor,
        fmax: Union[float, None] = None,
        vmin: Union[float, None] = None,
        log_scale: bool = False,
    ) -> None:
        """Plot time-frequency representation of the signal.

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
            g=(self.kernels).cpu().detach().numpy(),
            fs=self.fs,
            scale=self.scale,
            plot_scale=True,
            fc_min=self.fc_min,
            fc_max=self.fc_max,
        )

    def plot_decoder_response(self) -> None:
        """Plot frequency response of the synthesis (decoder) filters.

        Note:
            This method displays a plot and does not return values.
        """
        plot_response_(
            g=(self.decoder_kernels).detach().cpu().numpy(),
            fs=self.fs,
            scale=self.scale,
            decoder=True,
        )

    @property
    def condition_number(self) -> torch.Tensor:
        """Compute condition number of the analysis filterbank.

        Returns:
            torch.Tensor: Condition number of the frame operator

        Note:
            Lower condition numbers indicate better numerical stability.
            Values close to 1.0 indicate tight frames.
        """
        kernels = (self.kernels).squeeze()
        return condition_number(kernels, int(self.stride), self.Ls)

    @property
    def condition_number_decoder(self) -> torch.Tensor:
        """Compute condition number of the synthesis filterbank.

        Returns:
            torch.Tensor: Condition number of the decoder frame operator

        Note:
            Lower condition numbers indicate better numerical stability for reconstruction.
        """
        kernels = (self.decoder_kernels).squeeze()
        return condition_number(kernels, int(self.stride), self.Ls)
