import torch
import random
import numbers
import numpy as np

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

		c, w, h = input.shape
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
