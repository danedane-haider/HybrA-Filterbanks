import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from hybra.utils import audfilters_fir
from hybra.utils import plot_response as plot_response_
from hybra.utils import plot_coefficients as plot_coefficients_
from hybra.utils import calculate_condition_number, firwin
from hybra._fit_neurodual import fit


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
    g_mod = g * torch.exp(2 * torch.pi * 1j * fc * torch.arange(Lg) / fs)
    return g_mod

def fctobw(fc):
    return 25 + 75 * (1 + 1.4e-6 * fc**2)**0.69

def freqtoaud(freq):
    return 9.2645 * torch.sign(freq) * torch.log(1 + torch.abs(freq) * 0.00437)

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

    aud = torch.zeros_like(freq, dtype=torch.float32)
    linear_part = freq < fc_crit
    auditory_part = freq >= fc_crit

    aud[linear_part] = slope * (freq[linear_part] - fc_crit) + aud_crit
    aud[auditory_part] = freqtoaud(freq[auditory_part])

    return aud

def bwtofc(bw):
    return (bw - 24.7) * 9.265

def audtofreq(aud):
    return (1 / 0.00437) * (torch.exp(aud / 9.2645) - 1)

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

    freq = torch.zeros_like(aud, dtype=torch.float32)
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
    aud_min = freqtoaud_mod(torch.tensor([0]), fc_crit)[0]
    aud_max = freqtoaud_mod(torch.tensor([fs//2]), fc_crit)[0]

    # Generate frequencies spaced evenly on the modified auditory scale
    fc_aud = Variable(torch.linspace(aud_min, aud_max, num_channels))

    # Convert back to frequency scale
    fc = audtofreq_mod(fc_aud, fc_crit)

    # Ensure exact endpoints
    fc[0] = 0
    fc[-1] = fs//2

    return fc, fc_aud

class ModAudletFIR(nn.Module):
    def __init__(self, filterbank_config={'filter_len':256,
                                          'num_channels':64,
                                          'fs':16000,
                                          'Ls':16000,
                                          'bwmul':1}):
        super().__init__()

        [filters, d, fc, fc_crit, L] = audfilters_fir(**filterbank_config)

        self.filters = filters
        self.stride = d
        self.filter_len = filterbank_config['filter_len'] 
        self.fs = filterbank_config['fs']
        self.fc = fc
        self.fc_crit = fc_crit
        self.Ls = L
        self.bwmul = filterbank_config['bwmul']
        self.num_channels = filterbank_config['num_channels']

        self.kernels_real = torch.tensor(filters.real, dtype=torch.float32)
        self.kernels_imag = torch.tensor(filters.imag, dtype=torch.float32)

    def forward(self, x):
        x = F.pad(x.unsqueeze(1), (self.filter_len//2, self.filter_len//2), mode='circular')

        ####################################################################################################
        # Bandwidth conversion
        ####################################################################################################

        probeLs = 10000
        probeLg = 1000
        g_probe = torch.tensor(firwin(probeLg, probeLs))
        
        # peak normalize
        gf_probe = torch.fft.fft(g_probe) / torch.max(torch.abs(torch.fft.fft(g_probe)))

        # compute ERB-type bandwidth of the prototype
        bw_conversion = torch.linalg.norm(gf_probe)**2 * probeLg / probeLs / 4
        weird_factor = self.fs * 10.64
        
        ####################################################################################################
        # Center frequencies
        ####################################################################################################

        # get the bandwidth for the maximum admissible filter length and the associated center frequency
        fsupp_crit = bw_conversion / self.filter_len * weird_factor
        fc_crit = bwtofc(fsupp_crit / self.bwmul * bw_conversion)

        [fc, _] = audspace_mod(fc_crit, self.fs, self.num_channels)
        self.fc = fc
        num_lin = torch.where(fc < fc_crit)[0].shape[0]

        ####################################################################################################
        # Frequency and time supports
        ####################################################################################################

        # time support for the auditory part
        tsupp_lin = (torch.ones(num_lin) * self.filter_len).astype(int)
        # frequency support for the auditory part
        if num_lin == self.num_channels:
            fsupp = fctobw(self.fs//2) / bw_conversion * self.bwmul
            tsupp = tsupp_lin
        else:
            fsupp = fctobw(fc[num_lin:]) / bw_conversion * self.bwmul
            tsupp_aud = (torch.round(bw_conversion / fsupp * weird_factor)).astype(int)
            tsupp = torch.concatenate([tsupp_lin, tsupp_aud])

        # Maximal decimation factor (stride) to get a nice frame and accoring signal length
        d = torch.floor(torch.min(self.fs / fsupp)).astype(int)

        ####################################################################################################
        # Generate filters
        ####################################################################################################

        g = torch.zeros((self.num_channels, self.filter_len), dtype=torch.complex128)

        g[0,:] = torch.sqrt(d) * firwin(self.filter_len) / torch.sqrt(2)
        g[-1,:] = torch.sqrt(d) * modulate(firwin(tsupp[-1], self.filter_len), self.fs//2, self.fs) / self.sqrt(2)

        for m in range(1, self.num_channels - 1):
            g[m,:] = torch.sqrt(d) * modulate(firwin(tsupp[m], self.filter_len), fc[m], self.fs)

        return out_real + 1j * out_imag

    def plot_response(self):
        plot_response_(g=(self.kernels_real + 1j*self.kernels_imag).detach().numpy(), fs=self.fs, scale=True, fc_crit=self.fc_crit)

    def plot_coefficients(self, x):
        with torch.no_grad():
            coefficients = torch.log10(torch.abs(self.forward(x)[0]**2))
        plot_coefficients_(coefficients, self.fc, self.Ls, self.fs)

    @property
    def condition_number(self):
        filters = (self.kernels_real + 1j*self.kernels_imag).squeeze()
        # pad with zeros to have length Ls
        filters = F.pad(filters, (0, self.Ls - filters.shape[-1]), mode='constant', value=0)
        return calculate_condition_number(filters, int(self.stride))