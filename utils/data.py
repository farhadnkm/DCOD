#from __future__ import print_function, division
import os
import torch
from skimage import io
import skimage.color as skcolor
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import cv2
import tifffile as tiff
import simulator.simulation_utils as utils



class HologramDataset(Dataset):
	def __init__(self, amp_dir, phase_dir,
				 amp_dir_gt, phase_dir_gt,  transform=None, color_format='gray'):
		self.amp_dir = amp_dir
		self.phase_dir = phase_dir
		self.amp_dir_gt = amp_dir_gt
		self.phase_dir_gt = phase_dir_gt
		self.amps = self._sorted_alphanumeric(os.listdir(amp_dir))
		self.phases = self._sorted_alphanumeric(os.listdir(phase_dir))
		self.amps_gt = self._sorted_alphanumeric(os.listdir(amp_dir_gt))
		self.phases_gt = self._sorted_alphanumeric(os.listdir(phase_dir_gt))
		self.transform = transform

		self.color_format = color_format.lower()
		assert self.color_format in {'rgb', 'gray'}

	def __len__(self):
		return len(self.amps)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		intensity_path = os.path.join(self.amp_dir, self.amps[idx])
		intensity = io.imread(intensity_path, as_gray=True).astype('float32')
		intensity = self._convert_colormap(intensity)
		intensity /= 255
		intensity *= 2
		intensity -= 1

		phase_path = os.path.join(self.phase_dir, self.phases[idx])
		phase = io.imread(phase_path, as_gray=True).astype('float32')
		phase = self._convert_colormap(phase)
		phase /= 255
		phase *= 2
		phase -= 1
		#phase *= 2 * np.pi
		#phase -= np.pi

		intensity_gt_path = os.path.join(self.amp_dir_gt, self.amps_gt[idx])
		intensity_gt = io.imread(intensity_gt_path, as_gray=True).astype('float32')
		intensity_gt = self._convert_colormap(intensity_gt)
		intensity_gt /= 255
		intensity_gt *= 2
		intensity_gt -= 1

		phase_gt_path = os.path.join(self.phase_dir_gt, self.phases_gt[idx])
		phase_gt = io.imread(phase_gt_path, as_gray=True).astype('float32')
		phase_gt = self._convert_colormap(phase_gt)
		phase_gt /= 255
		phase_gt *= 2
		phase_gt -= 1
		#phase_gt *= 2 * np.pi
		#phase_gt -= np.pi

		#input = np.stack([intensity, phase], axis=0)
		#input = np.array(phase[np.newaxis, ...])
		#output = np.stack([intensity_gt, phase_gt], axis=0)
		#output = np.array(phase_gt[np.newaxis, ...])

		input = np.moveaxis(phase, -1, 0)
		output = np.moveaxis(phase_gt, -1, 0)
		sample = {'input': input, 'output': output}

		if self.transform:
			sample = self.transform(sample)

		return sample

	def _getkey(self, fname):
		return int(fname.split('.')[0])

	def _sorted_alphanumeric(self, flist):
		return sorted(flist, key=self._getkey)

	def _convert_colormap(self, img):
		try:
			c = img.shape[2]
		except:
			c = 1

		if self.color_format == 'gray':
			return skcolor.rgb2gray(img) if c >= 2 else img
		elif self.color_format == 'rgb':
			return skcolor.gray2rgb(img) if c == 1 else img


class AverageMeter(object):
	"""
	Keeps track of most recent, average, sum, and count of a metric.
	"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
	"""
	Clips gradients computed during backpropagation to avoid explosion of gradients.
	:param optimizer: optimizer with the gradients to be clipped
	:param grad_clip: clip value
	"""
	for group in optimizer.param_groups:
		for param in group['params']:
			if param.grad is not None:
				param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(state, filename):
	"""
	Save model checkpoint.
	:param state: checkpoint contents
	"""

	torch.save(state, filename)


def adjust_learning_rate(optimizer, shrink_factor):
	"""
	Shrinks learning rate by a specified factor.
	:param optimizer: optimizer whose learning rate must be shrunk.
	:param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
	"""

	print("\nDECAYING learning rate.")
	for param_group in optimizer.param_groups:
		param_group['lr'] = param_group['lr'] * shrink_factor
	print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


class DDN_Dataset():
	def __init__(self, input_feature_maps, amp_dir_gt, ph_dir_gt, bg_dir, z,
				 down_sampling_scale=0.5, input_bit_depth=8, wavelength=532.2e-3):

		self.input_feature_maps = input_feature_maps
		self.init_z = z

		amp = np.asarray(Image.open(amp_dir_gt)).astype('float32')
		bg = np.asarray(Image.open(bg_dir)).astype('float32')
		ph = tiff.imread(ph_dir_gt).astype('float32')
		x, y = round(amp.shape[0] * down_sampling_scale), round(amp.shape[1] * down_sampling_scale)

		#amp = cv2.resize(amp, dsize=(x, y))
		#bg = cv2.resize(bg, dsize=(x, y))
		#amp -= bg
		#amp *= -1
		#amp /= np.max(amp)

		amp = np.ones((x, y))

		ph = cv2.resize(ph, dsize=(x, y))
		ph /= 2200
		ph /= np.pi/2
		self.object_func = {'amp': amp, 'ph': ph}
		object_func_c = self.object_func['amp'] * np.exp(1j * self.object_func['ph'])  #amp * np.exp(1j * ph)

		delta_x = 0.5 / down_sampling_scale
		delta_y = 0.5 / down_sampling_scale
		w = wavelength
		shape = object_func_c.shape
		self.simulator = utils.Simulator(shape=shape, delta_x=delta_x, delta_y=delta_y, w=w)

		# Amplitude simulation
		amp_sim = np.abs(self.simulator.reconstruct(initializer=object_func_c, z=2)) ** 2
		self.object_func['amp'] = 2-amp_sim
		object_func_c = self.object_func['amp'] * np.exp(1j * self.object_func['ph'])

		res = self.simulator.reconstruct(initializer=object_func_c, z=z)
		self.hologram_amp = (np.abs(res) ** 2).astype('float32')
		self.hologram_amp /= np.max(self.hologram_amp)
		#self.hologram_amp *= 2 ** input_bit_depth - 1
		#self.hologram_amp = np.floor(self.hologram_amp).astype('float32')

		#self.hologram_amp.astype('complex128')
		#self.hologram_amp = self.object_func['ph']


	@staticmethod
	def convert_colormap(img, color_format='gray'):
		try:
			c = img.shape[2]
		except:
			c = 1

		if color_format == 'gray':
			return skcolor.rgb2gray(img) if c >= 2 else img
		elif color_format == 'rgb':
			return skcolor.gray2rgb(img) if c == 1 else img

