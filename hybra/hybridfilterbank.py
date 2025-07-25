import torch
import torch.nn as nn
import torch.nn.functional as F
from hybra.utils import condition_number, audfilters, plot_response, circ_conv, circ_conv_transpose, frame_bounds
from hybra.utils import ISACgram as ISACgram_
from hybra._fit_dual import tight_hybra
from typing import Union

class HybrA(nn.Module):
    def __init__(self,
                 kernel_size:int=128,
                 learned_kernel_size:int=23,
                 num_channels:int=40,
                 stride:Union[int,None]=None,
                 fc_max:Union[float,int,None]=None,
                 fs:int=16000, 
                 L:int=16000,
                 supp_mult:float=1,
                 scale:str='mel',
                 tighten:bool=False,
                 det_init:bool=False,
                 verbose:bool=True):
        """Constructor for a HybrA filterbank.

        Args:
            kernel_size (int) - kernel size of the auditory filterbank
            learned_kernel_size (int) - kernel size of the learned filterbank
            num_channels (int) - number of channels
            stride (int) - stride of the auditory filterbank. if 'None': 25% overlap
            fc_max (float) - maximum frequency on the auditory scale. if 'None': fs//2.
            fs (int) - sampling frequency
            L (int) - signal length
            supp_mult (float) - support multiplier. 
            scale (str) - auditory scale ('mel', 'erb', 'bark', 'log10', 'elelog'). elelog is a scale adapted to the hearing of elephants. Default: 'mel'.
            tighten (bool) - whether to tighten the hybrid filterbank. Default: False.
            det_init (bool) - whether to initialize the learned filters with diracs or randomly. Default: False.
        """
        
        super().__init__()

        [aud_kernels, d, fc, _, fc_max, kernel_min, kernel_size, Ls] = audfilters(
            kernel_size=kernel_size, num_channels=num_channels, fc_max=fc_max, fs=fs, L=L, supp_mult=supp_mult, scale=scale
        )

        if stride is not None:
            d = stride
            Ls = int(torch.ceil(torch.tensor(L / d)) * d)

        if verbose:
            print(f"Max. kernel size: {kernel_size}")
            print(f"Min. kernel size: {kernel_min}")
            print(f"Number of channels: {num_channels}")
            print(f"Stride for min. 25% overlap: {d}")
            print(f"Signal length: {Ls}")

        self.register_buffer('kernels', aud_kernels)
        self.kernel_size = kernel_size
        self.learned_kernel_size = learned_kernel_size
        self.stride = d
        self.num_channels = num_channels
        self.fc = fc
        self.Ls = Ls
        self.fs = fs

        if det_init:
            learned_kernels = torch.zeros([self.num_channels, 1, self.learned_kernel_size])
            learned_kernels[:,0,0] = 1.0
        else:
            learned_kernels = torch.randn([self.num_channels, 1, self.learned_kernel_size])/torch.sqrt(torch.tensor(self.learned_kernel_size*self.num_channels))
            learned_kernels = learned_kernels / torch.norm(learned_kernels, p=1, dim=-1, keepdim=True)
        
        learned_kernels = learned_kernels.to(self.kernels.dtype)

        if tighten:
            learned_kernels = tight_hybra(self.kernels, learned_kernels, d, Ls, fs, fit_eps = 1.0001, max_iter = 1000)  

        self.learned_kernels = nn.Parameter(learned_kernels, requires_grad=True)

        self.hybra_kernels = F.conv1d(
            self.kernels.squeeze(1),
            self.learned_kernels,
            groups=self.num_channels,
            padding="same",
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        hybra_kernels = F.conv1d(
            self.kernels.squeeze(1),
            self.learned_kernels,
            groups=self.num_channels,
            padding="same",
        )
        self.hybra_kernels = hybra_kernels.clone().detach()

        return circ_conv(x.unsqueeze(1), hybra_kernels, self.stride)

    def encoder(self, x:torch.Tensor):
        """
        For learning use forward method!
        """
        return circ_conv(x.unsqueeze(1), self.hybra_kernels, self.stride)
    
    def decoder(self, x:torch.Tensor) -> torch.Tensor:
        _, B = frame_bounds(self.hybra_kernels.squeeze(1), self.stride, None)
        return circ_conv_transpose(x, self.hybra_kernels / B, self.stride).squeeze(1)
    
    # plotting methods
    
    def ISACgram(self, x):
        with torch.no_grad():
            coefficients = self.forward(x)
        ISACgram_(coefficients, self.fc, self.Ls, self.fs)

    def plot_response(self):
        plot_response((self.hybra_kernels).squeeze().cpu().detach().numpy(), self.fs)

    def plot_decoder_response(self):
        plot_response((self.hybra_kernels).squeeze().cpu().detach().numpy(), self.fs, decoder=True)

    @property
    def condition_number(self, learnable:bool=False):
        kernels = (self.hybra_kernels).squeeze()
        if learnable:
            return condition_number(kernels, self.stride, self.Ls)
        else:
            return condition_number(kernels, self.stride, self.Ls).item() 


