import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from hybra.utils import audfilters_fir
import warnings

class MSETight(torch.nn.Module):
    def __init__(self, beta: float = 0.0, fs: int = 16000):
        super().__init__()
        self.beta = beta
        self.loss = torch.nn.MSELoss()
        self.fs = fs

    def forward(self, preds, target, w=None):
        loss = self.loss(preds, target)
        Lg = w.shape[-1]
        num_channels = w.shape[0]
        w_long = torch.concatenate([w, torch.zeros((num_channels, self.fs - Lg))], axis=1)
        w_hat = torch.sum(torch.abs(torch.fft.fft(w_long, dim=1)[:, :self.fs//2])**2, dim=0)
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
		
		self.register_buffer('kernels_real', torch.tensor(filters.real, dtype=torch.float32))
		self.register_buffer('kernels_imag', torch.tensor(filters.imag, dtype=torch.float32))

		self.register_parameter('kernels_decoder_real', nn.Parameter(torch.tensor(filters.real, dtype=torch.float32), requires_grad=True))
		self.register_parameter('kernels_decoder_imag', nn.Parameter(torch.tensor(filters.imag, dtype=torch.float32), requires_grad=True))

	def forward(self, x):
		L_out = x.shape[-1]
		
		x = F.pad(x.unsqueeze(1), (self.filter_len//2, self.filter_len//2), mode='circular')
		x_real = F.conv1d(x, self.kernels_real.to(x.device).unsqueeze(1), stride=self.stride)
		x_imag = F.conv1d(x, self.kernels_imag.to(x.device).unsqueeze(1), stride=self.stride)
		
		L_in = x_real.shape[-1]
		kernel_size = self.kernels_decoder_real.shape[-1]
		padding = self.filter_len // 2

		# L_out = (L_in -1) * stride - 2 * padding + dialation * (kernel_size - 1) + output_padding + 1 ; dialation = 1
		output_padding=L_out - (L_in -1)*self.stride + 2*padding - kernel_size

		x = F.conv_transpose1d(
			x_real,
			self.kernels_decoder_real.unsqueeze(1),
			stride=self.stride,
			padding=padding,
			output_padding=output_padding
			) + F.conv_transpose1d(
				x_imag,
				self.kernels_decoder_imag.unsqueeze(1),
				stride=self.stride,
				padding=padding,
				output_padding=output_padding
			)
		
		return x

def fit(filterbank_config, eps_loss, max_iter):
	model = NeuroDual(filterbank_config=filterbank_config)
	optimizer = optim.Adam(model.parameters(), lr=5e-4)
	criterion = MSETight(beta=1e-8)

	losses = []
	kappas = []	

	loss_item = float('inf')
	i = 0
	print("Computing the synthesis filterbank. This might take a while :)")
	while loss_item >= eps_loss:
		optimizer.zero_grad()
		x = noise_uniform(filterbank_config['Ls']/filterbank_config['fs'],filterbank_config['fs'])
		output = model(x)
		
		w_real = model.kernels_decoder_real.squeeze()
		w_imag = model.kernels_decoder_imag.squeeze()
		
		loss, loss_tight, kappa = criterion(output, x.unsqueeze(0), w_real + 1j*w_imag)
		loss_item = loss.item()
		loss_tight.backward()
		optimizer.step()
		losses.append(loss.item())
		kappas.append(kappa)

		if i > max_iter:
			warnings.warn(f"Did not converge after {max_iter} iterations.")
			break
		i += 1

	print(f"Final Fit:\n\tkappa: {kappas[-1]}\n\tloss: {losses[-1]}")
	
	return model.kernels_decoder_real.detach(), model.kernels_decoder_imag.detach(), losses, kappas
