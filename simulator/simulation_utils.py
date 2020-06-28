import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import math
import torch

class Simulator:
	def __init__(self, shape, delta_x, delta_y, w):
		self.shape = shape
		x = np.arange(1, shape[0] + 1)
		y = np.arange(1, shape[1] + 1)
		self.xv, self.yv = np.meshgrid(x, y)
		self.delta_x = delta_x
		self.delta_y = delta_y
		self.w = w
		self.deltaf_x = (1 / shape[0]) / delta_x
		self.deltaf_y = (1 / shape[1]) / delta_y
		self.delta = math.sqrt(self.delta_x**2 + self.delta_y**2)
		self.shifts = []

	def propagator(self, z):
		phase_term = 1 / (self.w * self.w) \
					 - np.power((self.xv - self.shape[0] / 2 - 1) * self.deltaf_x, 2) \
					 - np.power((self.yv - self.shape[1] / 2 - 1) * self.deltaf_y, 2) \
					 + 0j
		return np.exp(2j * np.pi * z * np.sqrt(phase_term))

	def reconstruct(self, initializer, z):
		prop = self.propagator(z)

		init_hologram = ifftshift(fft2(fftshift(initializer)))# * self.delta**2
		prop_hologram = init_hologram * prop
		image = fftshift(ifft2(ifftshift(prop_hologram)))# / self.delta**2

		return image

	def construct(self, amplitude, phase):
		return amplitude * np.exp(1j * phase)



class Simulator_torch:
	def __init__(self, shape, delta_x, delta_y, w, dtype=torch.FloatTensor):
		self.shape = shape
		x = np.arange(1, shape[2] + 1)
		y = np.arange(1, shape[3] + 1)
		self.xv, self.yv = np.meshgrid(x, y)
		self.delta_x = delta_x
		self.delta_y = delta_y
		self.w = w
		self.deltaf_x = (1 / shape[2]) / delta_x
		self.deltaf_y = (1 / shape[3]) / delta_y
		self.delta = math.sqrt(self.delta_x ** 2 + self.delta_y ** 2)
		self.dtype = dtype

	def fftshift(self, x, axes=None):
		if axes is None:
			axes = tuple(range(x.ndim))
			shift = [dim // 2 for dim in x.shape]
		else:
			shift = [x.shape[ax] // 2 for ax in axes]
		return torch.roll(x, shift, axes)

	def ifftshift(self, x, axes=None):
		if axes is None:
			axes = tuple(range(x.ndim))
			shift = [-(dim // 2) for dim in x.shape]
		else:
			shift = [-(x.shape[ax] // 2) for ax in axes]
		return torch.roll(x, shift, axes)

	def mul(self, x, y):
		real = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
		imag = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
		return torch.stack([real, imag], dim=-1)

	def abs(self, x):
		real, imag = torch.unbind(x, -1)
		return torch.sqrt(torch.pow(real, 2) + torch.pow(imag, 2))

	def angle(self, x):
		real, imag = torch.unbind(x, -1)
		return torch.atan2(imag, real)

	def propagator(self, z):
		phase_term = 1 / (self.w * self.w) \
					 - np.power((self.xv - self.shape[2] / 2 - 1) * self.deltaf_x, 2) \
					 - np.power((self.yv - self.shape[3] / 2 - 1) * self.deltaf_y, 2)
		#phase_term = 2j * np.pi * z * np.sqrt(phase_term)
		phase_term = torch.from_numpy(np.pi * z * np.sqrt(phase_term)).unsqueeze(0).type(self.dtype)
		phase_term = torch.stack([torch.cos(2 * phase_term), torch.sin(2 * phase_term)], dim=-1)
		phase_term = torch.stack([phase_term] * self.shape[0], dim=0)
		return phase_term

	def reconstruct(self, obj, z):
		prop = self.propagator(z)

		axes = (2, 3)
		init_hologram = self.ifftshift(torch.fft(self.fftshift(obj, axes), signal_ndim=2), axes)  # * self.delta**2
		prop_hologram = self.mul(x=init_hologram, y=prop)
		image = self.fftshift(torch.ifft(self.ifftshift(prop_hologram, axes), signal_ndim=2), axes)  # / self.delta**2

		return image

	def make_complex(self, amplitude, phase):
		real = amplitude * torch.cos(phase)
		imag = amplitude * torch.sin(phase)
		return torch.stack([real, imag], dim=4)