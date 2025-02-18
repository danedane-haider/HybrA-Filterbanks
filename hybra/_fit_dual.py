import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings

from hybra.utils import audfilters

class MSETight(torch.nn.Module):
	def __init__(self, beta:float=0.0, fs:int=16000):
		super().__init__()
		self.beta = beta
		self.loss = torch.nn.MSELoss()
		self.fs = fs

	def forward(self, preds, target, w=None):
		loss = self.loss(preds, target)
		Lg = w.shape[-1]
		num_channels = w.shape[0]
		w_long = torch.concatenate([w, torch.zeros((num_channels, self.fs - Lg)).to(preds.device)], axis=1)
		w_neg = torch.conj(w_long)
		w_full = torch.concatenate([w_long, w_neg], dim=0)
		w_hat = torch.sum(torch.abs(torch.fft.fft(w_full, dim=1)[:, :self.fs//2])**2, dim=0)
		kappa = w_hat.max() / w_hat.min()

		return loss, loss + self.beta * (kappa - 1), kappa.item()

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

class ISACDual(nn.Module):
	def __init__(self, kernel_max, num_channels, fc_max, fs, L, bwmul, scale):
		super().__init__()
		
		[kernels, d, _, _, _, _, kernel_max, Ls] = audfilters(kernel_max=kernel_max,num_channels=num_channels, fc_max=fc_max, fs=fs,L=L,bwmul=bwmul,scale=scale)
		self.kernels = kernels
		self.stride = d
		self.kernel_max = kernel_max
		self.Ls = Ls
		
		self.register_buffer('kernels_real', torch.real(kernels).to(torch.float32))
		self.register_buffer('kernels_imag', torch.imag(kernels).to(torch.float32))

		self.register_parameter('decoder_kernels_real', nn.Parameter(torch.real(kernels).to(torch.float32), requires_grad=True))
		self.register_parameter('decoder_kernels_imag', nn.Parameter(torch.imag(kernels).to(torch.float32), requires_grad=True))

	def forward(self, x):
		x = F.pad(x.unsqueeze(1), (self.kernel_max//2, self.kernel_max//2), mode='circular')
		
		x_real = F.conv1d(x, self.kernels_real.to(x.device).unsqueeze(1), stride=self.stride)
		x_imag = F.conv1d(x, self.kernels_imag.to(x.device).unsqueeze(1), stride=self.stride)
		
		L_in = x_real.shape[-1]
		L_out = self.Ls

		kernel_size = self.kernel_max
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

def fit(kernel_max, num_channels, fc_max, fs, L, bwmul, scale, decoder_fit_eps, max_iter):
	model = ISACDual(kernel_max, num_channels, fc_max, fs, L, bwmul, scale)
	optimizer = optim.Adam(model.parameters(), lr=5e-4)
	criterion = MSETight(beta=1e-7, fs=fs)

	losses = []
	kappas = []	

	loss_item = float('inf')
	i = 0
	print("Computing the synthesis filterbank. This might take a while ⛷️")
	while loss_item >= decoder_fit_eps:
		optimizer.zero_grad()
		x = noise_uniform(model.Ls)
		output = model(x)
		
		w_real = model.decoder_kernels_real.squeeze()
		w_imag = model.decoder_kernels_imag.squeeze()
		
		loss, loss_tight, kappa = criterion(output, x, w_real + 1j*w_imag)
		loss_item = loss.item()
		loss_tight.backward()
		optimizer.step()
		losses.append(loss.item())
		kappas.append(kappa)

		if i > max_iter:
			warnings.warn(f"Did not converge after {max_iter} iterations.")
			break
		i += 1

	print(f"Final Stats:\n\tFinal PSD ratio: {kappas[-1]}\n\tBest MSE loss: {losses[-1]}")
	
	return model.decoder_kernels_real.detach(), model.decoder_kernels_imag.detach(), losses, kappas
