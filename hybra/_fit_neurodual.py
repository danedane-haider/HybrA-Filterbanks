import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from hybra.utils import audfilters_fir

class MSETight(torch.nn.Module):

    def __init__(self, beta: float = 0.0):
        super().__init__()
        self.beta = beta
        self.loss = torch.nn.MSELoss()

    def forward(self, preds, target, w=None):
        loss = self.loss(preds, target)
        w_hat = torch.sum(torch.abs(torch.fft.fft(w, dim=1))**2, dim=0)

        kappa = w_hat.max() / w_hat.min()

        return loss, loss + self.beta * (kappa - 1), kappa.item()

def noise_uniform(dur=1, fs=16000):
    N = int(dur * fs)
    X = torch.rand(N // 2 + 1) * 2 - 1
    
    X_full = torch.zeros(N, dtype=torch.cfloat)
    X_full[0:N//2+1] = X
    X_full[N//2+1:] = torch.conj(X[1:N//2].flip(0))
    
    x = torch.fft.ifft(X_full).real
    x = x / torch.max(torch.abs(x))
    
    return x.unsqueeze(0)

class NeuroDual(nn.Module):
	def __init__(self, filterbank_config):
		super().__init__()
		
		[filters, d, fc, fc_crit, _] = audfilters_fir(**filterbank_config)
		self.filters = filters
		self.stride = d
		self.filter_len = filterbank_config['filter_len'] 
		self.fs = filterbank_config['fs']
		self.fc = fc
		self.fc_crit = fc_crit
		
		kernels_real = torch.tensor(filters.real, dtype=torch.float32)
		kernels_imag = torch.tensor(filters.imag, dtype=torch.float32)
		self.register_buffer('kernels_real', kernels_real)
		self.register_buffer('kernels_imag', kernels_imag)
		
		kernel_decoder_real = torch.nn.functional.pad(torch.tensor(kernels_real, dtype=torch.float32), (0, 0))
		kernel_decoder_imag = torch.nn.functional.pad(torch.tensor(kernels_imag, dtype=torch.float32), (0, 0))
		
		self.register_parameter('kernels_decoder_real', nn.Parameter(kernel_decoder_real, requires_grad=True))
		self.register_parameter('kernels_decoder_imag', nn.Parameter(kernel_decoder_imag, requires_grad=True))

	def forward(self, x):
		x = F.pad(x.unsqueeze(1), (self.filter_len//2, self.filter_len//2), mode='circular')
		
		x_real = F.conv1d(x, self.kernels_real.to(x.device).unsqueeze(1), stride=self.stride)
		x_imag = F.conv1d(x, self.kernels_imag.to(x.device).unsqueeze(1), stride=self.stride)
		
		x = F.conv_transpose1d(
			x_real,
			self.kernels_decoder_real.unsqueeze(1),
			stride=self.stride,
			padding=self.filter_len//2,
			output_padding=self.stride-2
			) + F.conv_transpose1d(
				x_imag,
				self.kernels_decoder_imag.unsqueeze(1),
				stride=self.stride,
				padding=self.filter_len//2,
				output_padding=self.stride-2
			)
		
		return x

def fit(filterbank_config, eps_kappa):
	model = NeuroDual(filterbank_config=filterbank_config)
	optimizer = optim.Adam(model.parameters(), lr=5e-4)
	criterion = MSETight(beta=1e-8)

	losses = []
	kappas = []	

	kappa = float('inf')
	while kappa >= eps_kappa:
		optimizer.zero_grad()
# 		x = noise_uniform(filterbank_config['Ls']/filterbank_config['fs'],filterbank_config['fs'])

		x = noise_uniform(1)
		
		output = model(x)
		
		w_real = model.kernels_decoder_real.squeeze()
		w_imag = model.kernels_decoder_imag.squeeze()
		
		loss, loss_tight, kappa = criterion(output, x, w_real + 1j*w_imag)
		loss_tight.backward()
		optimizer.step()
		losses.append(loss.item())
		kappas.append(kappa)
		print(f'Loss: {loss.item()}')
		print(f'Kappa: {kappa}')
	
	return model.kernels_decoder_real.detach(), model.kernels_decoder_imag.detach(), losses, kappas
