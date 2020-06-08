#from __future__ import print_function, division
import os
import torch
import random
import numbers
from skimage import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


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


class Config:
	def __init__(self, dataset, data_loader, iterations, batch_size, device,
				 log_root='./', log_folder='log', checkpoint_name='checkpoint.tar'):
		self.dataset = dataset
		self.data_loader = data_loader
		self.iterations = iterations
		self.batch_size = batch_size
		self.device = device
		self.log_root = log_root
		self.log_folder = log_folder
		self.checkpoint_name = checkpoint_name
		self.log_file_name = 'log.csv'
		self.summary_file_name = 'summary.txt'

class GaussianNoise(object):
	def __init__(self, mean=0., std=1., key='input'):
		self.std = std
		self.mean = mean
		self.key = key

	def __call__(self, img):
		shape = img[self.key].shape
		noise = np.random.randn(shape[1], shape[2]) * self.std + self.mean
		for i in range(shape[0]):
			img[self.key][i] += noise
		return img

	def __repr__(self):
		return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class RandomCrop(object):
	def __init__(self, size, keys=('input', 'output')):
		if isinstance(size, numbers.Number):
			self.size = (int(size), int(size))
		else:
			self.size = size
		self.keys = keys

	def __call__(self, img):
		input = img[self.keys[0]]
		output = img[self.keys[1]]
		_, w, h = input.shape
		th, tw = self.size
		if w == tw and h == th:
			i, j = 0, 0
		else:
			i = random.randint(0, h - th)
			j = random.randint(0, w - tw)

		return {self.keys[0]: input[:, i: i + th, j: j + tw],
				self.keys[1]: output[:, i: i + th, j: j + tw]}

class ToTensor(object):
	def __init__(self, keys=('input', 'output')):
		self.keys = keys
	def __call__(self, sample):
		input, output = sample[self.keys[0]], sample[self.keys[1]]

		return {self.keys[0]: torch.from_numpy(input),
				self.keys[1]: torch.from_numpy(output)}


def show_batch(dataloader, batch_index, sample_index):
	fig, axes = plt.subplots(1, 4)

	for i, sample_batched in enumerate(dataloader):
		if i == batch_index:
			print(i, sample_batched['input'].size(),
				  sample_batched['output'].size())
			axes[0].imshow(sample_batched['input'][sample_index, 0, :, :], cmap='gray')
			axes[1].imshow(sample_batched['input'][sample_index, 1, :, :], cmap='gray')
			axes[2].imshow(sample_batched['output'][sample_index, 0, :, :], cmap='gray')
			axes[3].imshow(sample_batched['output'][sample_index, 1, :, :], cmap='gray')
			break

	plt.show()
def show_batch_single_channel(dataloader, batch_index, sample_index):
	fig, axes = plt.subplots(1, 2)

	for i, sample_batched in enumerate(dataloader):
		if i == batch_index:
			print(i, sample_batched['input'].size(),
				  sample_batched['output'].size())
			axes[0].imshow(sample_batched['input'][sample_index, 0, :, :], cmap='gray')
			axes[1].imshow(sample_batched['output'][sample_index, 0, :, :], cmap='gray')
			break

	plt.show()

AMP_DIR = 'D:\Research_data\pap_smear_dataset\Dataset\Amplitude'
PH_DIR = 'D:\Research_data\pap_smear_dataset\Dataset\Phase'
AMP_DIR_GT = 'D:\Research_data\pap_smear_dataset\Dataset\Amplitude_GT'
PH_DIR_GT = 'D:\Research_data\pap_smear_dataset\Dataset\Phase_GT'

if __name__ == '__main__':  # <== this line is necessary
	transforms_train = transforms.Compose([
		RandomCrop(size=128),
		ToTensor()
	])
	dataset = HologramDataset(amp_dir=AMP_DIR, phase_dir=PH_DIR, amp_dir_gt=AMP_DIR_GT,
							  phase_dir_gt=PH_DIR_GT, transform=transforms_train)
	dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
	#show_batch(dataloader, batch_index=0, sample_index=0)
	show_batch_single_channel(dataloader, batch_index=0, sample_index=0)