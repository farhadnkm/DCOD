#from __future__ import print_function, division
import os
import torch
from skimage import io
import numpy as np
from torch.utils.data import Dataset


class Config:
	def __init__(self, dataset, data_loader, model, optimizer, loss_criterion,
				 iterations, start_epoch, batch_size, device, log_dir,
				 checkpoint_name='checkpoint.tar', log_name='log.csv'):
		self.dataset = dataset
		self.data_loader = data_loader
		self.model = model
		self.optimizer = optimizer
		self.loss_criterion = loss_criterion
		self.iterations = iterations
		self.start_epoch = start_epoch
		self.batch_size = batch_size
		self.device = device
		self.log_dir = log_dir
		self.checkpoint_name = checkpoint_name
		self.log_name = log_name


class HologramDataset(Dataset):
	def __init__(self, amp_dir, phase_dir,
				 amp_dir_gt, phase_dir_gt,  transform=None):
		self.amp_dir = amp_dir
		self.phase_dir = phase_dir
		self.amp_dir_gt = amp_dir_gt
		self.phase_dir_gt = phase_dir_gt
		self.amps = self._sorted_alphanumeric(os.listdir(amp_dir))
		self.phases = self._sorted_alphanumeric(os.listdir(phase_dir))
		self.amps_gt = self._sorted_alphanumeric(os.listdir(amp_dir_gt))
		self.phases_gt = self._sorted_alphanumeric(os.listdir(phase_dir_gt))
		self.transform = transform

	def __len__(self):
		return len(self.amps)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		intensity_path = os.path.join(self.amp_dir, self.amps[idx])
		intensity = io.imread(intensity_path, as_gray=True).astype('float32')
		intensity /= 255
		intensity *= 2
		intensity -= 1

		phase_path = os.path.join(self.phase_dir, self.phases[idx])
		phase = io.imread(phase_path, as_gray=True).astype('float32')
		phase /= 255
		phase *= 2
		phase -= 1
		#phase *= 2 * np.pi
		#phase -= np.pi

		intensity_gt_path = os.path.join(self.amp_dir_gt, self.amps_gt[idx])
		intensity_gt = io.imread(intensity_gt_path, as_gray=True).astype('float32')
		intensity_gt /= 255
		intensity_gt *= 2
		intensity_gt -= 1

		phase_gt_path = os.path.join(self.phase_dir_gt, self.phases_gt[idx])
		phase_gt = io.imread(phase_gt_path, as_gray=True).astype('float32')
		phase_gt /= 255
		phase_gt *= 2
		phase_gt -= 1
		#phase_gt *= 2 * np.pi
		#phase_gt -= np.pi

		#input = np.stack([intensity, phase], axis=0)
		input = np.array(phase[np.newaxis, ...])
		#output = np.stack([intensity_gt, phase_gt], axis=0)
		output = np.array(phase_gt[np.newaxis, ...])
		sample = {'input': input, 'output': output}

		if self.transform:
			sample = self.transform(sample)

		return sample

	def _getkey(self, fname):
		return int(fname.split('.')[0])

	def _sorted_alphanumeric(self, flist):
		return sorted(flist, key=self._getkey)


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