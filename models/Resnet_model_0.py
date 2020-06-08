import torch.nn as nn


class Conv2dAuto(nn.Conv2d):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


class ConvTranspose2dAuto(nn.ConvTranspose2d):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


def Conv_bn(in_channels, out_channels, conv, *args, **kwargs):
	return nn.Sequential(
		conv(in_channels=in_channels, out_channels=out_channels, *args, **kwargs),
		nn.BatchNorm2d(num_features=out_channels)
	)


class ResidualSubBlock(nn.Module):
	def __init__(self, in_channels, out_channels, activation=nn.ReLU):
		super().__init__()
		self.subBlock = nn.Sequential(
			Conv_bn(in_channels=in_channels, out_channels=out_channels, conv=Conv2dAuto, kernel_size=(3, 3), bias=False, stride=1),
			activation(),
			Conv_bn(in_channels=out_channels, out_channels=out_channels, conv=Conv2dAuto, kernel_size=(3, 3), bias=False, stride=1),
		)
		self.shortcut = nn.Sequential(
			Conv_bn(in_channels=in_channels, out_channels=out_channels, conv=Conv2dAuto, kernel_size=(1, 1), bias=False, stride=1),
			activation(),
		)

	def forward(self, x):
		shortcut = self.shortcut(x)
		x = self.subBlock(x)
		x += shortcut
		return x

class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, activation=nn.ReLU):
		super().__init__()
		self.block = nn.Sequential(
			ResidualSubBlock(in_channels, out_channels, activation),
			activation(),
			ResidualSubBlock(out_channels, out_channels, activation),
			activation(),
			ResidualSubBlock(out_channels, out_channels, activation),
			activation(),
			ResidualSubBlock(out_channels, out_channels, activation),
		)

	def forward(self, x):
		return self.block(x)


class Model(nn.Module):
	def __init__(self):
		super().__init__()
		self.layers = nn.Sequential(
			Conv_bn(1, 16, conv=Conv2dAuto, kernel_size=(5, 5), bias=True, stride=1),
			nn.ReLU(),
			nn.Dropout(0.9),
			nn.MaxPool2d(kernel_size=(2, 2), stride=2),
			ResidualBlock(16, 32, nn.ReLU),
			nn.ReLU(),
			nn.Dropout(0.9),
			nn.MaxPool2d(kernel_size=(2, 2), stride=2),
			ResidualBlock(32, 64, nn.ReLU),
			nn.ReLU(),
			nn.Dropout(0.9),
			nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=2, padding=1, output_padding=1, bias=False),
			nn.ReLU(),
			nn.Dropout(0.9),
			nn.ConvTranspose2d(32, 1, kernel_size=(3, 3), stride=2, padding=1, output_padding=1, bias=False),
			#nn.ReLU(),
			#nn.Dropout(0.9),
			#nn.Linear(256, 256),
			#nn.Tanh()
		)

	def forward(self, x):
		return self.layers(x)

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)

# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# discriminator section:
# it says it's better to use strided convolution layers rather than using pooling layer.
# because the network learns its own way of downsampling