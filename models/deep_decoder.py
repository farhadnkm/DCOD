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


class DecoderBlock(nn.Module):
	def __init__(self, n_channels, upsample_mode='bilinear'):
		super(DecoderBlock, self).__init__()

		self.block = nn.Sequential(
			nn.Conv2d(n_channels, n_channels, (1, 1), stride=1, padding=0, bias=False),
			nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False),
			nn.ReLU(),
			nn.BatchNorm2d(n_channels, affine=True),  # affine true learns beta and gama otherwise beta=1, gamma=0
			)

	def forward(self, data):	#data: (N, c, h, w)
		return self.block(data)

class DeepDecoder(nn.Module):
	def __init__(self, in_channels=64, levels=4):
		super(DeepDecoder, self).__init__()

		layers = [DecoderBlock(in_channels)]*levels
		layers.append(nn.Conv2d(in_channels, 1, (1, 1), stride=1, padding=1, bias=False))
		layers.append(nn.Sigmoid())
		self.deep_decoder = nn.Sequential(*layers)

	def forward(self, data):
		output = self.deep_decoder(data)  # * 2 * math.pi
		return output


class DeepDecoder3(nn.Module):
	def __init__(self, layers_channels, out_channels, kernel_sizes, pad_mode='reflection', upsample_mode='bilinear',
				 activation_func=nn.ReLU(), bn_affine=True):
		super(DeepDecoder3, self).__init__()

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
		layers.append(nn.Sigmoid())

		self.deep_decoder = nn.Sequential(*layers)

	def forward(self, data):
		return self.deep_decoder(data)



class DeepDecoder2(nn.Module):
	def __init__(self, layers_channels, out_channels=1, kernel_size=1, need_sigmoid=True,
				 pad_mode='reflection', upsample_mode='bilinear',
				 act_fun=nn.ReLU(),  # nn.LeakyReLU(0.2, inplace=True)
				 bn_before_act=False, bn_affine=True, upsample_first=True):
		super(DeepDecoder2, self).__init__()

		layers_channels = layers_channels + [layers_channels[-1], layers_channels[-1]]
		print(layers_channels)
		n_scales = len(layers_channels)

		if not (isinstance(kernel_size, list) or isinstance(kernel_size, tuple)):
			kernel_size = [kernel_size] * n_scales

		layers = []

		for i in range(len(layers_channels) - 1):

			if upsample_first:
				layers.append(Conv2d_pad(layers_channels[i], layers_channels[i + 1], kernel_size[i], 1, pad_mode=pad_mode))
				if upsample_mode != 'none' and i != len(layers_channels) - 2:
					layers.append(nn.Upsample(scale_factor=2, mode=upsample_mode))
			else:
				if upsample_mode != 'none' and i != 0:
					layers.append(nn.Upsample(scale_factor=2, mode=upsample_mode))
				layers.append(Conv2d_pad(layers_channels[i], layers_channels[i + 1], kernel_size[i], 1, pad_mode=pad_mode))

			if i != len(layers_channels) - 1:
				if bn_before_act:
					layers.append(nn.BatchNorm2d(layers_channels[i + 1], affine=bn_affine))
				layers.append(act_fun)
				if not bn_before_act:
					layers.append(nn.BatchNorm2d(layers_channels[i + 1], affine=bn_affine))

		layers.append(Conv2d_pad(layers_channels[-1], out_channels, 1, pad_mode=pad_mode))

		if need_sigmoid:
			layers.append(nn.Sigmoid())

		self.model = nn.Sequential(*layers)

	def forward(self, data):
		return self.model(data)# * 2 * math.pi


model = DeepDecoder3(layers_channels=[32]*5, out_channels=1, kernel_sizes=[1]*5)
#model = DeepDecoder2(layers_channels=[32]*5, out_channels=1, kernel_size=1)
#model = DeepDecoder(in_channels=32, levels=5)
#for param in model.parameters():
#	print(param.shape)

#model_summary = summary(model, input_size=(32, 16, 16), batch_size=1, device='cpu')
# with open(os.path.join(log_dir, SUMMARY_NAME), 'w') as text_file:
#	text_file.write(model_summary)

dummy = torch.ones((1, 32, 16, 16))  # ((Batch size, channels, h, w))
