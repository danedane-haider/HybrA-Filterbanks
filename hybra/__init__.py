"""HybrA-Filterbanks: Auditory-inspired filterbanks for deep learning.

This package provides PyTorch implementations of auditory-inspired filterbanks
including ISAC, HybrA, and variants for spectrograms and cepstral coefficients.

Classes:
    ISAC: Invertible and Stable Auditory filterbank with Customizable kernels
    HybrA: Hybrid Auditory filterbank combining fixed and learnable filters
    ISACSpec: ISAC spectrogram with temporal averaging
    ISACCC: ISAC Cepstral Coefficients extractor

Example:
    >>> import hybra
    >>> filterbank = hybra.ISAC(kernel_size=128, num_channels=40, fs=16000, L=16000)
    >>> x = torch.randn(1, 16000)
    >>> coeffs = filterbank(x)
"""

from .hybridfilterbank import HybrA
from .isac import ISAC
from .isac_mel import ISACSpec
from .isac_mfcc import ISACCC

__all__ = ["ISAC", "HybrA", "ISACSpec", "ISACCC"]
