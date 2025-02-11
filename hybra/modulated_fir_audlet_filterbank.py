import torch
import torch.nn as nn
import torch.nn.functional as F
from hybra.utils import audfilters_fir
from hybra.utils import plot_response as plot_response_
from hybra.utils import plot_coefficients as plot_coefficients_
from hybra.utils import calculate_condition_number, firwin, fir_tightener3000

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

class ModAudletFIR(nn.Module):
    def __init__(self, filterbank_config={'filter_len':128,
                                          'num_channels':40,
                                          'fs':16000,
                                          'Ls':16000,
                                          'bwmul':1},
                                          is_hybra=False,
                                          hybra_kernel_length=24):
        super().__init__()

        [filters, d, fc, fc_crit, L] = audfilters_fir(**filterbank_config)

        self.filters = filters
        self.stride = d
        self.filter_len = filterbank_config['filter_len'] 
        self.fs = filterbank_config['fs']
        self.fc = torch.nn.Parameter(torch.tensor(fc, dtype=torch.float32), requires_grad=True)
        self.fc_crit = fc_crit
        self.Ls = L
        self.bwmul = filterbank_config['bwmul']
        self.num_channels = filterbank_config['num_channels']
        
        self.is_hybra = is_hybra
        self.kernel_len = hybra_kernel_length
        if self.is_hybra:        # Initialize trainable filters
            k = torch.tensor(self.num_channels / (self.kernel_len * self.num_channels))
            encoder_weight = (-torch.sqrt(k) - torch.sqrt(k)) * torch.rand([self.num_channels, 1, self.kernel_len]) + torch.sqrt(k)

            encoder_weight = torch.tensor(fir_tightener3000(
                encoder_weight.squeeze(1), self.kernel_len, D=d, eps=1.01
            ),  dtype=torch.float32).unsqueeze(1)
            encoder_weight = encoder_weight / torch.norm(encoder_weight, dim=-1, keepdim=True)
            self.register_parameter('hybra_kernels_real', nn.Parameter(encoder_weight, requires_grad=True))
            self.register_parameter('hybra_kernels_imag', nn.Parameter(encoder_weight, requires_grad=True))

    def forward(self, x):

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
        self.fc_crit = bwtofc(fsupp_crit / self.bwmul * bw_conversion)
        num_lin = torch.where(self.fc < self.fc_crit)[0].shape[0]

        ####################################################################################################
        # Frequency and time supports
        ####################################################################################################

        # time support for the auditory part
        tsupp_lin = (torch.ones(num_lin) * self.filter_len).int()
        # frequency support for the auditory part
        if num_lin == self.num_channels:
            fsupp = fctobw(self.fs//2) / bw_conversion * self.bwmul
            tsupp = tsupp_lin
        else:
            fsupp = fctobw(self.fc[num_lin:]) / bw_conversion * self.bwmul
            tsupp_aud = (torch.round(bw_conversion / fsupp * weird_factor)).int()
            tsupp = torch.concatenate([tsupp_lin, tsupp_aud])

        # Maximal decimation factor (stride) to get a nice frame and accoring signal length
        d = torch.floor(torch.min(self.fs / fsupp)).int()

        ####################################################################################################
        # Generate filters
        ####################################################################################################

        g = torch.zeros((self.num_channels, self.filter_len), dtype=torch.complex128)

        g[0,:] = torch.sqrt(d) * firwin(self.filter_len) / torch.sqrt(torch.tensor(2))
        g[-1,:] = torch.sqrt(d) * modulate(torch.tensor(firwin(tsupp[-1].item(), self.filter_len),dtype=torch.float32), self.fs//2, self.fs) / torch.sqrt(torch.tensor(2))

        for m in range(1, self.num_channels - 1):
            g[m,:] = torch.sqrt(d) * modulate(torch.tensor(firwin(tsupp[m].item(), self.filter_len), dtype=torch.float32), self.fc[m], self.fs)

        if self.is_hybra:
            self.kernels_real = F.conv1d(
                g.real.float().to(x.device).squeeze(1),
                self.hybra_kernels_real,
                groups=self.num_channels,
                padding="same",
            )

            self.kernels_imag  = F.conv1d(
                g.imag.float().to(x.device).squeeze(1),
                self.hybra_kernels_imag,
                groups=self.num_channels,
                padding="same",
            )

        else:
            self.kernels_real = g.real.float()
            self.kernels_imag = g.imag.float()


        x = F.pad(x.unsqueeze(1), (self.filter_len//2, self.filter_len//2), mode='circular')

        out_real = F.conv1d(x, self.kernels_real.to(x.device).unsqueeze(1), stride=d.item())
        out_imag = F.conv1d(x, self.kernels_imag.to(x.device).unsqueeze(1), stride=d.item())



        return out_real + 1j * out_imag

    def plot_response(self):
        plot_response_(g=(self.kernels_real + 1j*self.kernels_imag).detach().numpy(), fs=self.fs, scale=True, fc_crit=self.fc_crit.numpy())

    def plot_coefficients(self, x):
        with torch.no_grad():
            coefficients = torch.log10(torch.abs(self.forward(x)[0]**2))
        plot_coefficients_(coefficients.detach().numpy(), self.fc.detach().numpy(), self.Ls, self.fs)

    @property
    def condition_number(self):
        filters = (self.kernels_real + 1j*self.kernels_imag).squeeze()
        # pad with zeros to have length Ls
        filters = F.pad(filters, (0, self.Ls - filters.shape[-1]), mode='constant', value=0)
        return calculate_condition_number(filters, int(self.stride))