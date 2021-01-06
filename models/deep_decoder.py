import math
import torch
import torch.nn as nn
from torchsummary import summary


def add_module(self, module):
	self.add_module(str(len(self) + 1), module)


torch.nn.Module.add = add_module


class Conv2d_pad(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad_mode='zero'):
		super(Conv2d_pad, self).__init__()

		padder = None
		to_pad = int((kernel_size - 1) / 2)
		if pad_mode == 'reflection':
			padder = nn.ReflectionPad2d(to_pad)
			to_pad = 0

		convolver = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=to_pad, bias=False)

		layers = filter(lambda x: x is not None, [padder, convolver])
		self.conv = nn.Sequential(*layers)

	def forward(self, data):
		return self.conv(data)


class DeepDecoder(nn.Module):
	def __init__(self, layers_channels, out_channels, kernel_sizes, pad_mode='reflection', upsample_mode='bilinear',
				 activation_func=nn.ReLU(), out_activation=nn.Sigmoid(), bn_affine=True):
		super(DeepDecoder, self).__init__()

		layers = []
		for i in range(len(layers_channels)):
			try:
				out_layer_channels = layers_channels[i + 1]
			except IndexError:
				out_layer_channels = layers_channels[-1]

			layers.append(Conv2d_pad(layers_channels[i], out_layer_channels, kernel_sizes[i], 1, pad_mode=pad_mode))
			layers.append(nn.Upsample(scale_factor=2, mode=upsample_mode))
			layers.append(activation_func)
			layers.append(nn.BatchNorm2d(out_layer_channels, affine=bn_affine))

		layers.append(Conv2d_pad(layers_channels[-1], layers_channels[-1], kernel_sizes[-1], 1, pad_mode=pad_mode))
		layers.append(activation_func)
		layers.append(nn.BatchNorm2d(layers_channels[-1], affine=bn_affine))

		layers.append(Conv2d_pad(layers_channels[-1], out_channels, kernel_sizes[-1], 1, pad_mode=pad_mode))
		if out_activation is not None:
			layers.append(out_activation)

		self.deep_decoder = nn.Sequential(*layers)

	def forward(self, data):
		return self.deep_decoder(data)

class DeepEncoder(nn.Module):
	def __init__(self, layers_channels, kernel_sizes, pad_mode='reflection', downsample_mode='bilinear',
				 activation_func=nn.ReLU(), bn_affine=True):
		super(DeepEncoder, self).__init__()

		layers = []
		for i in range(len(layers_channels)):
			try:
				out_layer_channels = layers_channels[i + 1]
			except IndexError:
				out_layer_channels = layers_channels[-1]

			layers.append(Conv2d_pad(layers_channels[i], out_layer_channels, kernel_sizes[i], 1, pad_mode=pad_mode))
			layers.append(nn.Upsample(scale_factor=0.5, mode=downsample_mode))
			layers.append(activation_func)
			layers.append(nn.BatchNorm2d(out_layer_channels, affine=bn_affine))

		self.deep_decoder = nn.Sequential(*layers)

	def forward(self, data):
		return self.deep_decoder(data)