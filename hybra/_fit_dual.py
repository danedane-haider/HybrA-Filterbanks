import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings

from hybra.utils import audfilters, condition_number, alias

class MSETight(nn.Module):
	def __init__(self, beta:float=0.0, fs:int=16000):
		super().__init__()
		self.beta = beta
		self.loss = nn.MSELoss()
		self.fs = fs

	def forward(self, preds=None, target=None, kernels=None, d=None, Ls=None):
		if kernels is not None:
			Lg = kernels.shape[-1]
			num_channels = kernels.shape[0]
			kernels_long = torch.concatenate([kernels, torch.zeros((num_channels, self.fs - Lg)).to(kernels.device)], axis=1)
			kernels_neg = torch.conj(kernels_long)
			kernels_full = torch.concatenate([kernels_long, kernels_neg], dim=0)
			kernels_hat = torch.sum(torch.abs(torch.fft.fft(kernels_full, dim=1)[:, :self.fs//2])**2, dim=0)
			kappa = kernels_hat.max() / kernels_hat.min()
			# padto = int(torch.ceil(torch.tensor(self.fs / d)) * d)
			#kernels = F.pad(kernels, (0, Ls - kernels.shape[-1]), mode='constant', value=0)
			#kappa = condition_number(kernels, int(d))
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

class ISACDual(nn.Module):
	def __init__(self, kernels, d, Ls):
		super().__init__()
		
		#[kernels, d, _, _, _, _, kernel_max, Ls] = audfilters(kernel_max=kernel_max, num_channels=num_channels, fc_max=fc_max, fs=fs, L=L, bwmul=bwmul, scale=scale)
		self.stride = d
		self.kernel_max = kernels.shape[-1]
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

def fit(kernels, d, Ls, fs, decoder_fit_eps, max_iter):
	model = ISACDual(kernels, d, Ls)
	optimizer = optim.Adam(model.parameters(), lr=0.00001)
	criterion = MSETight(beta=1e-5, fs=fs)

	losses = []
	kappas = []	

	loss_item = float('inf')
	i = 0
	print("Computing synthesis kernels for ISAC. This might take a while ⛷️")
	while loss_item >= decoder_fit_eps:
		optimizer.zero_grad()
		x_in = noise_uniform(model.Ls)
		x_out = model(x_in)
		
		w_real = model.decoder_kernels_real.squeeze()
		w_imag = model.decoder_kernels_imag.squeeze()
		
		loss, loss_tight, kappa = criterion(x_out, x_in, w_real + 1j*w_imag)
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


class ISACTight(nn.Module):
	def __init__(self, kernels, d, Ls):
		super().__init__()
		
		#[kernels, d, _, _, _, _, kernel_max, Ls] = audfilters(kernel_max=kernel_max, num_channels=num_channels, fc_max=fc_max, fs=fs, L=L, bwmul=bwmul, scale=scale)
		#self.kernels = kernels
		self.stride = d
		self.kernel_max = kernels.shape[-1]
		self.Ls = Ls

		self.register_parameter('kernels_real', nn.Parameter(torch.real(kernels).to(torch.float32), requires_grad=True))
		self.register_parameter('kernels_imag', nn.Parameter(torch.imag(kernels).to(torch.float32), requires_grad=True))

	def forward(self):
		k_real = self.kernels_real
		k_imag = self.kernels_imag
		
		return k_real + 1j*k_imag
	
	@property
	def condition_number(self):
		filters = (self.kernels_real + 1j*self.kernels_imag).squeeze()
		filters = F.pad(filters, (0, self.Ls - filters.shape[-1]), mode='constant', value=0)
		return condition_number(filters, int(self.stride))


def tight(kernels, d, Ls, fs, fit_eps, max_iter):
	model = ISACTight(kernels, d, Ls)
	optimizer = optim.Adam(model.parameters(), lr=0.00001)
	criterion = MSETight(beta=1, fs=fs)

	print(f"Init Condition number:\n\t{model.condition_number.item()}")

	kappas = []	

	loss_item = float('inf')
	i = 0
	print("Tightening ISAC. This might take a while 🏂")
	while loss_item >= fit_eps:
		optimizer.zero_grad()
		
		kernels_real = model.kernels_real.squeeze()
		kernels_imag = model.kernels_imag.squeeze()
		
		kappa, kappa_item = criterion(preds=None, target=None, kernels=kernels_real + 1j*kernels_imag, d=d, Ls=Ls)
		kappa.backward()
		optimizer.step()
		kappas.append(kappa_item)

		if i > max_iter:
			warnings.warn(f"Did not converge after {max_iter} iterations.")
			break
		i += 1

	print(f"Init Condition number:\n\t{model.condition_number.item()}\nand PSD ratio\n\t{kappas[-1]}")
	
	return model.kernels_real.detach(), model.kernels_imag.detach(), kappas
