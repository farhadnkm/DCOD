import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from models.deep_decoder import DeepDecoder
from utils.data import DDN_Dataset
import train_dnn
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class Config:
	def __init__(self, dataset, iterations, device, log_path, log_folder):
		self.dataset = dataset
		self.device = device
		self.iterations = iterations
		self.lr = 0.01

		checkpoint_name = 'checkpoint.pt.tar'
		log_name = 'log.csv'
		summary_name = 'summary.txt'

		self.log_root = os.path.join(log_path, log_folder)
		if not os.path.exists(self.log_root):
			os.mkdir(self.log_root)

		self.log_dir = os.path.join(self.log_root, log_name)
		self.checkpoint_dir = os.path.join(self.log_root, checkpoint_name)

		input_shape = self.dataset.input_feature_maps.shape
		self.decoder = DeepDecoder(in_channels=input_shape[1], levels=5)

		self.loss_criterion = nn.MSELoss()

		#self.optimizer = optim.SGD(params=filter(lambda p: p.requires_grad, self.decoder.parameters()),
		#						   lr=self.lr, momentum=0.01)
		self.optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, self.decoder.parameters()), lr=self.lr)
		#							betas=(0.5, 0.999))
		# model_summary = summary(model, input_size=(1, 128, 128),
		#						batch_size=BATCHSIZE, device=DEVICE)
		# with open(os.path.join(log_dir, SUMMARY_NAME), 'w') as text_file:
		#	text_file.write(model_summary)

		self.start_iter = 0
		if os.path.exists(self.checkpoint_dir):
			checkpoint = torch.load(self.checkpoint_dir)
			self.start_iter = checkpoint['iteration'] + 1
			self.decoder = checkpoint['decoder']
			self.optimizer = checkpoint['optimizer']
			print("\nLoaded checkpoint from iteration %d.\n" % (checkpoint['iteration'] + 1))


if __name__ == '__main__':
	random_seed = 999
	print("Random Seed: ", random_seed)
	random.seed(random_seed)
	torch.manual_seed(random_seed)

	log_path = 'D:/Research_data/train_logs'
	log_folder = 'log_05'
	device = 'cpu'
	iterations = int(1e4)

	bg_dir = 'D:/Research_data/pap_smear_dataset/Complementary/background.png'
	amp_dir = 'D:/Research_data/pap_smear_dataset/Intensity/77_intensity.png'
	ph_dir = 'D:/Research_data/pap_smear_dataset/Phase/77_phase.tif'

	input_feature_maps = torch.randn([1, 32, 16, 16])  # the output shape: (1, c, n, n)
	input_feature_maps *= 0.2
	input_feature_maps += 0.5
	print(torch.min(input_feature_maps))
	dataset = DDN_Dataset(input_feature_maps=input_feature_maps,
						  amp_dir_gt=amp_dir,
						  ph_dir_gt=ph_dir,
						  bg_dir=bg_dir,
						  z=150,
						  down_sampling_scale=0.5,
						  input_bit_depth=8)

	#plt.imshow(dataset.hologram_amp, cmap='gray')
	#plt.show()

	config = Config(dataset=dataset,
					iterations=iterations,
					device=device,
					log_path=log_path,
					log_folder=log_folder)

	train_dnn.run(config)