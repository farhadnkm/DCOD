import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import train
from utils.data import Config, HologramDataset
from utils.transforms import RandomCrop, ToTensor
from utils.torchsummary import summary
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models.Resnet_model_0 import Model, weights_init
from models import Resnet_model_1 as resnet1


if __name__ == '__main__':

	dataset_path = 'D:/Research_data/pap_smear_dataset/Dataset'
	log_path = 'D:/Research_data/train_logs'

	AMP_DIR = os.path.join(dataset_path, 'Amplitude')
	PH_DIR = os.path.join(dataset_path, 'Phase')
	AMP_DIR_GT = os.path.join(dataset_path, 'Amplitude_GT')
	PH_DIR_GT = os.path.join(dataset_path, 'Phase_GT')

	LOG_ROOT = log_path
	LOG_FOLDER = 'log_03'
	CHECKPOINT_NAME = 'checkpoint.pt.tar'
	LOG_NAME = 'log.csv'
	SUMMARY_NAME = 'summary.txt'

	ITERATIONS = 1e4
	# DEVICE = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
	DEVICE = 'cpu'
	BATCHSIZE = 1



	# random_seed = random.randint(1, 10000)
	random_seed = 999
	print("Random Seed: ", random_seed)
	random.seed(random_seed)
	torch.manual_seed(random_seed)

	transforms_train = transforms.Compose([
		RandomCrop(size=128),
		ToTensor()
	])

	dataset = HologramDataset(amp_dir=AMP_DIR, phase_dir=PH_DIR, amp_dir_gt=AMP_DIR_GT,
							  phase_dir_gt=PH_DIR_GT, transform=transforms_train)

	data_loader = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=4)

	#model = Model().to(DEVICE)
	#model.apply(weights_init)
	model = resnet1.ResNet(large_kernel_size=9, in_channels=1, n_channels=64, n_blocks=16)
	model.train()
	# dummy = torch.ones((1, 1, 256, 256))  # ((Batch size, channels, h, w))
	# print(model)
	# print(model(dummy).shape)


	loss_criterion = nn.MSELoss()

	# optimizer = optim.SGD(model.parameters(), lr=0.0002, momentum=0.9)
	optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, betas=(0.5, 0.999))



	log_dir = os.path.join(LOG_ROOT, LOG_FOLDER)
	if not os.path.exists(log_dir):
		os.mkdir(log_dir)

	#model_summary = summary(model, input_size=(1, 128, 128),
	#						batch_size=BATCHSIZE, device=DEVICE)
	#if model_summary is not None:
	#	print(model_summary)
	#else:
	#	print('its None')
	#with open(os.path.join(log_dir, SUMMARY_NAME), 'w') as text_file:
	#	text_file.write(model_summary)

	start_epoch = 0
	if os.path.exists(os.path.join(log_dir, CHECKPOINT_NAME)):
		checkpoint = torch.load(os.path.join(log_dir, CHECKPOINT_NAME))
		start_epoch = checkpoint['epoch'] + 1
		model = checkpoint['model']
		optimizer = checkpoint['optimizer']
		print("\nLoaded checkpoint from epoch %d.\n" % (checkpoint['epoch'] + 1))

	config = Config(dataset=dataset,
					data_loader=data_loader,
					model=model,
					optimizer=optimizer,
					loss_criterion=loss_criterion,
					iterations=ITERATIONS,
					start_epoch=start_epoch,
					batch_size=BATCHSIZE,
					device=DEVICE,
					log_dir=log_dir,
					checkpoint_name=CHECKPOINT_NAME,
					log_name=LOG_NAME)

	train.run(config)