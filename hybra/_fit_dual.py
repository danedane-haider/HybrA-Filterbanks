import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings

from hybra.utils import condition_number, alias, upsample, circ_conv, circ_conv_transpose, frame_bounds

class MSETight(nn.Module):
    def __init__(self, beta:float=0.0, fs:int=16000, diag_only:bool=False):
        super().__init__()
        self.beta = beta
        self.loss = nn.MSELoss()
        self.fs = fs
        self.diag_only = diag_only

    def forward(self, preds=None, target=None, kernels=None, d=None, Ls=None):
        # usual L2 loss
        if kernels is None:
            loss = self.loss(preds, target)
            return loss
        else:
            r = alias(kernels, d, Ls, diag_only=self.diag_only)
            # use it for tightening only
            if preds is None:
                return self.beta * r, r.item()
            # use it for regularization
            else:
                loss = self.loss(preds, target)
                return loss, loss + self.beta * r, r.item()


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
# Fit ISAC dual
############################################################################################################

class ISACDual(nn.Module):
    def __init__(self, kernels, d, Ls):
        super().__init__()
        
        self.stride = d
        self.kernel_size = kernels.shape[-1]
        self.Ls = Ls
        
        self.register_buffer('kernels', kernels)
        self.register_parameter('decoder_kernels', nn.Parameter(kernels, requires_grad=True))

        _, B = frame_bounds(kernels, d, Ls)
        self.B = B


    def forward(self, x):
        _, B = frame_bounds(self.decoder_kernels, self.stride, self.Ls)
        x = circ_conv(x, self.kernels, self.stride)
        x = circ_conv_transpose(x, self.decoder_kernels / self.B, self.stride)
        return x.real

def fit(kernels, d, Ls, fs, decoder_fit_eps, max_iter):

    model = ISACDual(kernels, d, Ls)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    criterion = MSETight(beta=1e-12, fs=fs, diag_only=True)

    losses = []
    kappas = []	

    loss_item = float('inf')
    i = 0
    print("Computing synthesis kernels for ISAC ⛷️")
    while loss_item >= decoder_fit_eps:
        optimizer.zero_grad()
        x_in = noise_uniform(model.Ls)
        x_out = model(x_in)
        
        loss, loss_tight, kappa = criterion(x_out, x_in, model.decoder_kernels.squeeze(), d=d, Ls=None)
        loss_tight.backward()
        optimizer.step()
        losses.append(loss.item())
        kappas.append(kappa)

        error = (kappas[-1] - 1.0)**0.01
        criterion.beta *= error

        if i > max_iter:
            print(f"Max. iteration of {max_iter} reached.")
            break
        i += 1

    print(f"Final Stats:\n\tFinal PSD ratio: {kappas[-1]}\n\tBest MSE loss: {losses[-1]}")
    
    return model.decoder_kernels.detach()

############################################################################################################
# Tightening ISAC
############################################################################################################

class ISACTight(nn.Module):
    def __init__(self, kernels, d, Ls):
        super().__init__()
        
        self.stride = d
        self.kernel_size = kernels.shape[-1]
        self.Ls = Ls

        self.register_parameter('kernels', nn.Parameter(kernels, requires_grad=True))

    def forward(self):        
        return self.kernels
    
    @property
    def condition_number(self):
        kernels = (self.kernels).squeeze()
        return condition_number(kernels, int(self.stride), self.Ls)


def tight(kernels, d, Ls, fs, fit_eps, max_iter):

    model = ISACTight(kernels, d, Ls)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    criterion = MSETight(beta=1, fs=fs, diag_only=False)

    print(f"Init Condition number:\n\t{model.condition_number.item()}")

    loss_item = float('inf')
    i = 0
    print("Tightening ISAC 🏂")
    while loss_item >= fit_eps:
        optimizer.zero_grad()
        model()
        kernels = model.kernels.squeeze()
        
        kappa, _ = criterion(preds=None, target=None, kernels=kernels, d=d, Ls=None)
        kappa.backward()
        optimizer.step()

        k = condition_number(kernels, d, Ls).item()
        error = (k - 1.0)**0.01

        for param_group in optimizer.param_groups:
            param_group['lr'] *= error

        if i > max_iter:
            print(f"Max. iteration of {max_iter} reached.")
            break
        i += 1

    print(f"Final Condition number:\n\t{model.condition_number.item()}")
    
    return model.kernels.detach()

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

    model = HybrATight(aud_kernels, learned_kernels, d, Ls)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = MSETight(beta=1, fs=fs)

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