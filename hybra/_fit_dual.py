import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings

from hybra.utils import audfilters, condition_number, alias

class MSETight(nn.Module):
    def __init__(self, beta:float=0.0, fs:int=16000, diag_only:bool=False):
        super().__init__()
        self.beta = beta
        self.loss = nn.MSELoss()
        self.fs = fs
        self.diag_only = diag_only

    def forward(self, preds=None, target=None, kernels=None, d=None, Ls=None):
        if kernels is not None:
            kappa = alias(kernels, d, diag_only=self.diag_only).to(kernels.device)
            if preds is not None:
                loss = self.loss(preds, target)
                return loss, loss + self.beta * (kappa - 1), kappa.item()
            else:
                return self.beta * (kappa - 1), kappa.item()
        else:
            loss = self.loss(preds, target)
            return loss

def noise_uniform(Ls):
    Ls = int(Ls)
    X = torch.rand(Ls // 2 + 1) * 2 - 1

    X_full = torch.zeros(Ls, dtype=torch.cfloat)
    X_full[0:Ls//2+1] = X
    if Ls % 2 == 0:
        X_full[Ls//2+1:] = torch.conj(X[1:Ls//2].flip(0))
    else:
        X_full[Ls//2+1:] = torch.conj(X[1:Ls//2+1].flip(0))

    x = torch.fft.ifft(X_full).real
    x = x / torch.max(torch.abs(x))

    return x.unsqueeze(0)

############################################################################################################
# Compute ISAC dual
############################################################################################################

class ISACDual(nn.Module):
    def __init__(self, kernels, d, Ls):
        super().__init__()
        
        #[kernels, d, _, _, _, _, kernel_size, Ls] = audfilters(kernel_size=kernel_size, num_channels=num_channels, fc_max=fc_max, fs=fs, L=L, bw_multiplier=bw_multiplier, scale=scale)
        self.stride = d
        self.kernel_size = kernels.shape[-1]
        self.Ls = Ls
        
        self.register_buffer('kernels_real', torch.real(kernels).to(torch.float32))
        self.register_buffer('kernels_imag', torch.imag(kernels).to(torch.float32))

        self.register_parameter('decoder_kernels_real', nn.Parameter(torch.real(kernels).to(torch.float32), requires_grad=True))
        self.register_parameter('decoder_kernels_imag', nn.Parameter(torch.imag(kernels).to(torch.float32), requires_grad=True))


    def forward(self, x):
        x = F.pad(x.unsqueeze(1), (self.kernel_size//2, self.kernel_size//2), mode='circular')
        
        x_real = F.conv1d(x, self.kernels_real.to(x.device).unsqueeze(1), stride=self.stride)
        x_imag = F.conv1d(x, self.kernels_imag.to(x.device).unsqueeze(1), stride=self.stride)
        
        L_in = x_real.shape[-1]
        L_out = self.Ls

        kernel_size = self.kernel_size
        padding = kernel_size // 2

        # L_out = (L_in -1) * stride - 2 * padding + dialation * (kernel_size - 1) + output_padding + 1 ; dialation = 1
        output_padding = L_out - (L_in -1) * self.stride + 2 * padding - kernel_size

        x = F.conv_transpose1d(
            x_real,
            self.decoder_kernels_real.unsqueeze(1),
            stride=self.stride,
            padding=padding,
            output_padding=output_padding
            ) + F.conv_transpose1d(
                x_imag,
                self.decoder_kernels_imag.unsqueeze(1),
                stride=self.stride,
                padding=padding,
                output_padding=output_padding
            )
        
        return x.squeeze(1)

def fit(kernels, d, Ls, fs, decoder_fit_eps, max_iter):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device set to {device}")

    model = ISACDual(kernels, d, Ls).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = MSETight(beta=1e-8, fs=fs, diag_only=True).to(device)

    losses = []
    kappas = []	

    loss_item = float('inf')
    i = 0
    print("Computing synthesis kernels for ISAC. This might take a bit ⛷️")
    while loss_item >= decoder_fit_eps:
        optimizer.zero_grad()
        x_in = noise_uniform(model.Ls).to(device)
        x_out = model(x_in)
        
        w_real = model.decoder_kernels_real.squeeze()
        w_imag = model.decoder_kernels_imag.squeeze()
        
        loss, loss_tight, kappa = criterion(x_out, x_in, w_real + 1j*w_imag, d=d, Ls=None)
        loss_tight.backward()
        optimizer.step()
        losses.append(loss.item())
        kappas.append(kappa)

        if i > max_iter:
            print(f"Max. iteration of {max_iter} reached.")
            break
        i += 1

    print(f"Final Stats:\n\tFinal PSD ratio: {kappas[-1]}\n\tBest MSE loss: {losses[-1]}")
    
    return model.decoder_kernels_real.detach(), model.decoder_kernels_imag.detach(), losses, kappas

############################################################################################################
# Tightening ISAC
############################################################################################################

class ISACTight(nn.Module):
    def __init__(self, kernels, d, Ls):
        super().__init__()
        
        #[kernels, d, _, _, _, _, kernel_size, Ls] = audfilters(kernel_size=kernel_size, num_channels=num_channels, fc_max=fc_max, fs=fs, L=L, bw_multiplier=bw_multiplier, scale=scale)
        #self.kernels = kernels
        self.stride = d
        self.kernel_size = kernels.shape[-1]
        self.Ls = Ls

        self.register_parameter('kernels_real', nn.Parameter(torch.real(kernels).to(torch.float32), requires_grad=True))
        self.register_parameter('kernels_imag', nn.Parameter(torch.imag(kernels).to(torch.float32), requires_grad=True))

    def forward(self):        
        return self.kernels_real + 1j*self.kernels_imag
    
    @property
    def condition_number(self):
        kernels = (self.kernels_real + 1j*self.kernels_imag).squeeze()
        #kernels = F.pad(kernels, (0, self.Ls - kernels.shape[-1]), mode='constant', value=0)
        return condition_number(kernels, int(self.stride), self.Ls)


def tight(kernels, d, Ls, fs, fit_eps, max_iter):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device set to {device}")

    model = ISACTight(kernels, d, Ls).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    criterion = MSETight(beta=1, fs=fs).to(device)

    print(f"Init Condition number:\n\t{model.condition_number.item()}")

    kappas = []	

    loss_item = float('inf')
    i = 0
    print("Tightening ISAC. This might take a bit 🏂")
    while loss_item >= fit_eps:
        optimizer.zero_grad()
        model()
        kernels_real = model.kernels_real.squeeze()
        kernels_imag = model.kernels_imag.squeeze()
        
        kappa, kappa_item = criterion(preds=None, target=None, kernels=kernels_real + 1j*kernels_imag, d=d, Ls=None)
        kappa.backward()
        optimizer.step()
        kappas.append(kappa_item)

        if i > max_iter:
            print(f"Max. iteration of {max_iter} reached.")
            break
        i += 1

    print(f"Final Condition number:\n\t{model.condition_number.item()}")
    
    return model.kernels_real.detach(), model.kernels_imag.detach(), kappas

############################################################################################################
# Tightening HybrA
############################################################################################################

class HybrATight(nn.Module):
    def __init__(self, aud_kernels, learned_kernels, d, Ls):
        super().__init__()
        
        self.stride = d
        self.kernel_size = aud_kernels.shape[-1]
        self.num_channels = aud_kernels.shape[0]
        self.Ls = Ls

        self.register_buffer('aud_kernels_real', torch.real(aud_kernels).to(torch.float32))
        self.register_buffer('aud_kernels_imag', torch.imag(aud_kernels).to(torch.float32))

        self.register_parameter('learned_kernels_real', nn.Parameter(learned_kernels.to(torch.float32), requires_grad=True))
        self.register_parameter('learned_kernels_imag', nn.Parameter(learned_kernels.to(torch.float32), requires_grad=True))

        self.hybra_kernels_real = F.conv1d(
            self.aud_kernels_real.squeeze(1),
            self.learned_kernels_real,
            groups=self.num_channels,
            padding="same",
        ).unsqueeze(1)
        self.hybra_kernels_imag = F.conv1d(
            self.aud_kernels_imag.squeeze(1),
            self.learned_kernels_imag,
            groups=self.num_channels,
            padding="same",
        ).unsqueeze(1)

    def forward(self):

        self.hybra_kernels_real = F.conv1d(
            self.aud_kernels_real.squeeze(1),
            self.learned_kernels_real,
            groups=self.num_channels,
            padding="same",
        ).unsqueeze(1)
        self.hybra_kernels_imag = F.conv1d(
            self.aud_kernels_imag.squeeze(1),
            self.learned_kernels_imag,
            groups=self.num_channels,
            padding="same",
        ).unsqueeze(1)
        
        return self.hybra_kernels_real + 1j*self.hybra_kernels_imag
    
    @property
    def condition_number(self):
        kernels = (self.hybra_kernels_real + 1j*self.hybra_kernels_imag).squeeze()
        #kernels = F.pad(kernels, (0, self.Ls - kernels.shape[-1]), mode='constant', value=0)
        return condition_number(kernels, int(self.stride), self.Ls)
    
def tight_hybra(aud_kernels, learned_kernels, d, Ls, fs, fit_eps, max_iter):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device set to {device}")

    model = HybrATight(aud_kernels, learned_kernels, d, Ls).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = MSETight(beta=1, fs=fs).to(device)

    print(f"Init Condition number:\n\t{model.condition_number.item()}")

    kappas = []	

    loss_item = float('inf')
    i = 0
    print("Tightening HybrA. This might take a bit 🏄")
    while loss_item >= fit_eps:
        optimizer.zero_grad()
        model()
        kernels_real = model.hybra_kernels_real.squeeze()
        kernels_imag = model.hybra_kernels_imag.squeeze()
        
        kappa, kappa_item = criterion(preds=None, target=None, kernels=kernels_real + 1j*kernels_imag, d=d, Ls=None)
        kappa.backward()
        optimizer.step()
        kappas.append(kappa_item)

        if i > max_iter:
            print(f"Max. iteration of {max_iter} reached.")
            break
        i += 1

    print(f"Final Condition number:\n\t{model.condition_number.item()}")
    
    return model.learned_kernels_real.detach(), model.learned_kernels_imag.detach(), kappas