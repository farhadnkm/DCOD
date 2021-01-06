import torch
import math
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import tensorflow as tf
from scipy import interpolate


def fftshift_t(x, axes=None):
	if axes is None:
		axes = tuple(range(x.ndim))
		shift = [dim // 2 for dim in x.shape]
	else:
		shift = [x.shape[ax] // 2 for ax in axes]
	return torch.roll(x, shift, axes)


def ifftshift_t(x, axes=None):
	if axes is None:
		axes = tuple(range(x.ndim))
		shift = [-(dim // 2) for dim in x.shape]
	else:
		shift = [-(x.shape[ax] // 2) for ax in axes]
	return torch.roll(x, shift, axes)


def mul(x, y):
	return torch.stack(
		[x.select(-1, 0) * y.select(-1, 0) - x.select(-1, 1) * y.select(-1, 1),
		 x.select(-1, 0) * y.select(-1, 1) + x.select(-1, 1) * y.select(-1, 0)], dim=-1)


def abs(x, squared=False):
	if squared:
		return torch.pow(x.select(-1, 0), 2) + torch.pow(x.select(-1, 1), 2)
	else:
		return torch.sqrt(torch.pow(x.select(-1, 0), 2) + torch.pow(x.select(-1, 1), 2))


def angle(x):
	return torch.atan2(x.select(-1, 1), x.select(-1, 0))


def complex_tensor(amplitude, phase):
	return torch.stack(
		[amplitude * torch.cos(phase),
		 amplitude * torch.sin(phase)], dim=-1)


class Simulator:
	def __init__(self, shape, delta_x, delta_y, w):
		self.shape = shape
		self.delta_x = delta_x
		self.delta_y = delta_y
		self.k = 2 * np.pi / w
		self.k2 = self.k * self.k
		kx = np.fft.fftfreq(shape[0], delta_x / (2 * np.pi))
		ky = np.fft.fftfreq(shape[1], delta_y / (2 * np.pi))
		kx, ky = np.meshgrid(kx, ky, indexing='ij', sparse=True)
		self.kz2 = kx * kx + ky * ky
		self.valid_mask = self.k2 > self.kz2

	def propagator(self, z):
		p = np.zeros(self.shape, dtype=np.complex_)
		p[self.valid_mask] = np.exp(1j * np.sqrt(self.k2 - self.kz2[self.valid_mask]) * z)
		return p

	def reconstruct(self, initializer, z):
		prop = self.propagator(z)
		return ifft2(prop * fft2(initializer))


class Simulator_Torch:
	def __init__(self, shape, delta_x, delta_y, w, dtype=torch.FloatTensor):
		self.shape = shape
		x = np.arange(1, shape[2] + 1)
		y = np.arange(1, shape[3] + 1)
		self.xv, self.yv = np.meshgrid(x, y)
		self.xv = torch.from_numpy(self.xv).type(dtype)
		self.yv = torch.from_numpy(self.yv).type(dtype)
		self.delta_x = delta_x
		self.delta_y = delta_y
		self.w = w
		self.deltaf_x = (1 / shape[2]) / delta_x
		self.deltaf_y = (1 / shape[3]) / delta_y
		self.delta = math.sqrt(self.delta_x ** 2 + self.delta_y ** 2)
		self.dtype = dtype

	def propagator(self, z):
		phase_term = torch.ones_like(self.xv) * 1 / (self.w * self.w) \
					 - torch.pow((self.xv - self.shape[2] / 2 - 1) * self.deltaf_x, 2) \
					 - torch.pow((self.yv - self.shape[3] / 2 - 1) * self.deltaf_y, 2)

		phase_term = (np.pi * z * torch.sqrt(phase_term)).unsqueeze(0).type(self.dtype)
		phase_term = complex_tensor(amplitude=torch.ones_like(phase_term), phase=2 * phase_term)
		phase_term = torch.stack([phase_term] * self.shape[0], dim=0)
		return phase_term

	def reconstruct(self, obj, z, axes=(2, 3)):
		prop = self.propagator(z)

		obj_f = ifftshift_t(torch.fft(fftshift_t(obj, axes), signal_ndim=len(axes)), axes)# * self.delta**2
		obj_f_p = mul(x=obj_f, y=prop)
		obj_rec = fftshift_t(torch.ifft(ifftshift_t(obj_f_p, axes), signal_ndim=len(axes)), axes)# / self.delta**2
		return obj_rec




class Simulator_TF:
	def __init__(self, shape, delta_x, delta_y, wl, upsample_ratio=4, origin=(0.5, 0.5), dtype_c=tf.complex64, dtype_f=tf.float32):
		'''self.shape = shape
		self.x = np.arange(-shape[0]//2, shape[0]//2)
		self.y = np.arange(-shape[0]//2, shape[0]//2)
		self.xm, self.ym = np.meshgrid(self.x, self.y)
		self.xm = tf.convert_to_tensor(self.xm, dtype=dtype_f)
		self.ym = tf.convert_to_tensor(self.ym, dtype=dtype_f)
		self.delta_x = delta_x
		self.delta_y = delta_y
		self.wl = wl
		self.deltaf_x = (1 / shape[0]) / delta_x
		self.deltaf_y = (1 / shape[1]) / delta_y
		u = self.xm * self.deltaf_x
		v = self.ym * self.deltaf_y
		self.w = tf.math.sqrt(1 / (self.wl * self.wl) - tf.math.pow(u, 2) - tf.math.pow(v, 2))

		self.origin = origin
		ox = origin[0]
		oy = origin[1]
		self.upsample_ratio = upsample_ratio
		self.ups_shape = [upsample_ratio * i for i in self.shape]
		self.u = np.arange(-np.floor(self.shape[0]*ox), np.floor(self.shape[0]*(1-ox))).astype('float64') * self.deltaf_x
		self.v = np.arange(-np.floor(self.shape[1]*oy), np.floor(self.shape[1]*(1-oy))).astype('float64') * self.deltaf_y
		self.u_ups = np.arange(-np.floor(self.ups_shape[0]*ox), np.floor(self.ups_shape[0]*(1-ox))).astype('float64') * self.deltaf_x / self.upsample_ratio
		self.v_ups = np.arange(-np.floor(self.ups_shape[1]*oy), np.floor(self.ups_shape[1]*(1-oy))).astype('float64') * self.deltaf_y / self.upsample_ratio
		self.uv_ups = np.array(np.meshgrid(self.u_ups, self.v_ups, indexing='ij')).reshape((2, self.ups_shape[0] * self.ups_shape[1]))

		self.w_ups = np.sqrt(1 / (self.wl ** 2) - self.uv_ups[0] ** 2 - self.uv_ups[1] ** 2)[np.newaxis, ...]
		self.uvw_ups = np.concatenate((self.uv_ups, self.w_ups), axis=0)'''
		self.shape = shape
		self.delta_x = delta_x
		self.delta_y = delta_y
		self.k = 2 * np.pi / wl
		self.k2 = self.k * self.k
		kx = np.fft.fftfreq(shape[0], delta_x / (2 * np.pi))
		ky = np.fft.fftfreq(shape[1], delta_y / (2 * np.pi))
		kx, ky = np.meshgrid(kx, ky, indexing='ij', sparse=True)
		self.kz2 = tf.convert_to_tensor(kx * kx + ky * ky, dtype=dtype_f)
		self.valid_mask = self.k2 > self.kz2

		self.dtype_f = dtype_f
		self.dtype_c = dtype_c

	def propagator(self, z):
		p = tf.zeros(self.shape, dtype=self.dtype_f)
		p = tf.complex(real=p, imag=tf.math.sqrt(self.k2 - self.kz2) * z)
		return tf.math.exp(p)

	'''
	def tilt(self, obj, theta=0, phi=0):
		trans_x = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
		trans_y = np.array([[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]])
		trans = np.transpose(np.matmul(trans_x, trans_y))

		obj_abs = np.abs(obj)
		obj_angle = np.angle(obj)
		upsample_abs = interpolate.interp2d(self.u, self.v, obj_abs, kind='linear')
		upsample_angle = interpolate.interp2d(self.u, self.v, obj_angle, kind='linear')
		obj_ups_abs = upsample_abs(self.u_ups, self.v_ups)
		obj_ups_angle = upsample_angle(self.u_ups, self.v_ups)
		obj_ups = obj_ups_abs * np.exp(1j * obj_ups_angle)

		new_uvw = np.dot(trans, self.uvw_ups)
		new_u = new_uvw[0].reshape((self.ups_shape[0], self.ups_shape[1]), order='F')
		new_v = new_uvw[1].reshape((self.ups_shape[0], self.ups_shape[1]), order='F')
		new_w = new_uvw[2].reshape((self.ups_shape[0], self.ups_shape[1]), order='F')

		new_uvw[0] = new_uvw[0] * self.upsample_ratio / self.deltaf_x + np.floor(self.ups_shape[0] * self.origin[0])
		new_uvw[1] = new_uvw[1] * self.upsample_ratio / self.deltaf_y + np.floor(self.ups_shape[1] * self.origin[1])
		new_uvw = new_uvw[0:2].round().astype(int)

		um, vm = new_uvw.reshape((2, self.ups_shape[0], self.ups_shape[1]), order='F')
		indices = um + self.ups_shape[1] * vm

		#jacobian = np.abs(
		#	(trans[1, 0] * trans[2, 1] - trans[2, 0] * trans[1, 1]) * new_u / new_w +
		#	(trans[2, 0] * trans[0, 1] - trans[0, 0] * trans[2, 1]) * new_v / new_w +
		#	(trans[0, 0] * trans[1, 1] - trans[1, 0] * trans[0, 1]))

		obj_f = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(obj_ups)))
		obj_trans = np.take(obj_f, indices, mode='clip')# / jacobian
		obj_t = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(obj_trans)))

		return obj_t
	'''
	def reconstruct(self, obj, z):
		prop = self.propagator(z)
		return tf.signal.ifft2d(prop * tf.signal.fft2d(obj))


'''
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import gdal
from PIL import Image
from skimage.restoration import unwrap_phase

img_path = 'D:/GoogleDrive_/Colab/Dataset/Custom/cheek/2.tif'
img = gdal.Open(img_path).ReadAsArray().astype('float32')
img = img[1000:1000+512, 850:850+512]
img /= 2**16
bg_path = 'D:/GoogleDrive_/Colab/Dataset/Custom/background1.tif'
bg = gdal.Open(bg_path).ReadAsArray().astype('float32')
bg = bg[1000:1000+512, 850:850+512]
bg /= 2**16

img /= bg
minh = np.min(img)
img -= minh
img /= 1 - minh

obj = img + 0j
z = -238
s = Simulator(shape=obj.shape, delta_x=1.12, delta_y=1.12, w=532e-3)

res_amp = np.abs(s.reconstruct(obj, z))
res_amp /= 1.4
res_amp = np.clip(res_amp, 0, 1)
plt.imshow(res_amp, cmap='gray', vmin=0, vmax=1)
plt.show()

export_path_amp = 'D:/Research data/results/images/cheek cells/backprop/abs.png'
out_amp = np.uint8(res_amp * 255)
out_amp = Image.fromarray(out_amp)
out_amp = out_amp.convert('L')
out_amp.save(export_path_amp)

res_ph = unwrap_phase(np.angle(s.reconstruct(obj, z)))
res_ph += np.pi
res_ph /= 2 * np.pi
res_ph += 0.3
res_ph = np.clip(res_ph, 0, 1)
plt.imshow(res_ph, cmap='gray', vmin=0, vmax=1)
plt.show()

export_path_ph = 'D:/Research data/results/images/cheek cells/backprop/phase.png'
out_ph = np.uint8(res_ph * 255)
out_ph = Image.fromarray(out_ph)
out_ph = out_ph.convert('L')
out_ph.save(export_path_ph)

'''