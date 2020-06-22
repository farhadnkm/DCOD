import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import math

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