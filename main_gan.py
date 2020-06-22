import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import train
from utils.data import HologramDataset
from utils.transforms import RandomCrop, ToTensor
from utils.torchsummary import summary
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models.Resnet_model_1 import Generator, Discriminator, TruncatedVGG19, ResNet

class Config_GAN:
	def __init__(self, dataset_path, log_path, log_folder,
				 resnet_checkpoint_name='checkpoint_resnet.pt.tar',
				 device='cpu', batch_size=1, iterations=1e4):

		self.device = device
		self.iterations = iterations

		amp_dir = os.path.join(dataset_path, 'Amplitude')
		ph_dir = os.path.join(dataset_path, 'Phase')
		amp_dir_gt = os.path.join(dataset_path, 'Amplitude_GT')
		ph_dir_gt = os.path.join(dataset_path, 'Phase_GT')

		log_name = 'log_gan.csv'
		summary_generator = 'summary_generator.txt'
		summary_discriminator = 'summary_discriminator.txt'
		checkpoint_name = 'checkpoint_gan.pt.tar'

		self.log_root = os.path.join(log_path, log_folder)
		if not os.path.exists(self.log_root):
			os.mkdir(self.log_root)

		self.log_dir = os.path.join(self.log_root, log_name)
		self.checkpoint_dir = os.path.join(self.log_root, checkpoint_name)

		self.learning_rate = 1e-4
		vgg19_i = 5
		vgg19_j = 4
		self.beta = 1e-3
		self.grad_clip = None

		transforms_train = transforms.Compose([
			RandomCrop(size=128),
			ToTensor()
		])

		self.dataset = HologramDataset(amp_dir=amp_dir, phase_dir=ph_dir, amp_dir_gt=amp_dir_gt,
									   phase_dir_gt=ph_dir_gt, transform=transforms_train, color_format='rgb')

		self.data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=4)

		# model_summary = summary(model, input_size=(1, 128, 128),
		#						batch_size=BATCHSIZE, device=DEVICE)
		# if model_summary is not None:
		#	print(model_summary)
		# else:
		#	print('its None')
		# with open(os.path.join(log_dir, SUMMARY_NAME), 'w') as text_file:
		#	text_file.write(model_summary)

		self.start_epoch = 0
		if not os.path.exists(self.checkpoint_dir):
			self.generator = Generator(large_kernel_size=9, in_channels=3, n_channels=64, n_blocks=16)

			self.generator.initialize_with_resnet(resnet_checkpoint=os.path.join(os.path.join(self.log_root, resnet_checkpoint_name)))
			# generator.train()

			self.optimizer_g = optim.Adam(params=filter(lambda p: p.requires_grad, self.generator.parameters()),
										  lr=self.learning_rate)

			self.discriminator = Discriminator(kernel_size=3, in_channel=3, n_channels=64, n_blocks=8, fc_size=1024)

			# optimizer = optim.SGD(model.parameters(), lr=0.0002, momentum=0.9)
			self.optimizer_d = optim.Adam(params=filter(lambda p: p.requires_grad, self.discriminator.parameters()),
										  lr=self.learning_rate)

		# dummy = torch.ones((1, 1, 256, 256))  # ((Batch size, channels, h, w))
		# print(model)
		# print(model(dummy).shape)
		else:
			checkpoint = torch.load(self.checkpoint_dir)
			self.start_epoch = checkpoint['epoch'] + 1
			self.generator = checkpoint['generator']
			self.discriminator = checkpoint['discriminator']
			self.optimizer_g = checkpoint['optimizer_g']
			self.optimizer_d = checkpoint['optimizer_d']
			print("\nLoaded checkpoint from epoch %d.\n" % (checkpoint['epoch'] + 1))

		# Truncated VGG19 network to be used in the loss calculation
		self.truncated_vgg19 = TruncatedVGG19(i=vgg19_i, j=vgg19_j)
		self.truncated_vgg19.eval()

		self.resent_loss_criterion = nn.MSELoss()
		self.adversarial_loss_criterion = nn.BCEWithLogitsLoss()
		self.content_loss_criterion = nn.MSELoss()

		self.generator = self.generator.to(device)
		self.discriminator = self.discriminator.to(device)
		self.truncated_vgg19 = self.truncated_vgg19.to(device)
		self.content_loss_criterion = self.content_loss_criterion.to(device)
		self.adversarial_loss_criterion = self.adversarial_loss_criterion.to(device)


class Config_Resnet:
	def __init__(self, dataset_path, log_path, log_folder, device='cpu', batch_size=1, iterations=1e4):
		self.device = device
		self.batch_size = batch_size
		self.iterations = iterations

		amp_dir = os.path.join(dataset_path, 'Amplitude')
		ph_dir = os.path.join(dataset_path, 'Phase')
		amp_dir_gt = os.path.join(dataset_path, 'Amplitude_GT')
		ph_dir_gt = os.path.join(dataset_path, 'Phase_GT')

		checkpoint_name = 'checkpoint_resnet.pt.tar'
		log_name = 'log_resnet.csv'
		summary_name = 'summary_resnet.txt'

		self.log_root = os.path.join(log_path, log_folder)
		if not os.path.exists(self.log_root):
			os.mkdir(self.log_root)

		self.log_dir = os.path.join(self.log_root, log_name)
		self.checkpoint_dir = os.path.join(self.log_root, checkpoint_name)


		transforms_train = transforms.Compose([
			RandomCrop(size=128),
			ToTensor()
		])

		self.dataset = HologramDataset(amp_dir=amp_dir, phase_dir=ph_dir, amp_dir_gt=amp_dir_gt,
								  phase_dir_gt=ph_dir_gt, transform=transforms_train, color_format='rgb')

		self.data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=4)

		self.resnet = ResNet(large_kernel_size=9, in_channels=3, n_channels=64, n_blocks=16)
		# dummy = torch.ones((1, 1, 256, 256))  # ((Batch size, channels, h, w))
		# print(model)
		# print(model(dummy).shape)


		self.loss_criterion = nn.MSELoss()

		# optimizer = optim.SGD(model.parameters(), lr=0.0002, momentum=0.9)
		self.optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, self.resnet.parameters()), lr=1e-4, betas=(0.5, 0.999))

		#model_summary = summary(model, input_size=(1, 128, 128),
		#						batch_size=BATCHSIZE, device=DEVICE)
		#if model_summary is not None:
		#	print(model_summary)
		#else:
		#	print('its None')
		#with open(os.path.join(log_dir, SUMMARY_NAME), 'w') as text_file:
		#	text_file.write(model_summary)

		self.start_epoch = 0
		if os.path.exists(self.checkpoint_dir):
			checkpoint = torch.load(self.checkpoint_dir)
			self.start_epoch = checkpoint['epoch'] + 1
			self.resnet = checkpoint['resnet']
			self.optimizer = checkpoint['optimizer']
			print("\nLoaded checkpoint from epoch %d.\n" % (checkpoint['epoch'] + 1))


if __name__ == '__main__':
	# random_seed = random.randint(1, 10000)
	random_seed = 999
	print("Random Seed: ", random_seed)
	random.seed(random_seed)
	torch.manual_seed(random_seed)

	dataset_path = 'D:/Research_data/pap_smear_dataset/Dataset'
	log_path = 'D:/Research_data/train_logs'
	log_folder = 'log_04'
	device = 'cpu'
	batch_size = 1
	iterations = 1e4
	'''
	c = Config_Resnet(dataset_path=dataset_path,
					  log_path=log_path,
					  log_folder=log_folder,
					  batch_size=batch_size,
					  iterations=iterations)
	train.run_resnet(c)
	'''
	c = Config_GAN(dataset_path=dataset_path,
				   log_path=log_path,
				   log_folder=log_folder,
				   batch_size=batch_size,
				   iterations=iterations)
	train.run_gan(c)
