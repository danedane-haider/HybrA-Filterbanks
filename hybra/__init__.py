from .hybridfilterbank import HybrA
from utils import audfilters_fir

def fir_filter_designer(fs, Ls, fmin=0, fmax=None, spacing=1/2, bwmul=1, max_supp=480, redmul=1, scale='erb'):
    """
    Generate FIR filters equidistantly spaced on auditory frequency scales.
    
    Parameters:
        fs (int): Sampling rate.
        Ls (int): Signal length.
        fmin (int): Minimum frequency (Hz).
        fmax (int): Maximum frequency (Hz).
        spacing (float): Spacing between filters (scale units).
        bwmul (float): Bandwidth multiplier.
        max_supp (int): Maximum window length (samples).
        scale (str): Frequency scale ('erb', 'bark', 'mel', etc.).
    
    Returns:
        filters (list of torch.Tensor): Generated filters.
        a (list): Downsampling rates.
        fc (list): Center frequencies.
        L (int): Admissible signal length.
    """

    [filters,a,M,fc,L,fc_orig,fc_low,fc_high,ind_crit] = audfilters_fir(fs, Ls, fmin, fmax, spacing, bwmul, max_supp, redmul, scale)

    return filters, a, fc, L

