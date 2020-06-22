import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from models.deep_decoder import DeepDecoder3

if __name__ == '__main__':
	random_seed = 999
	print("Random Seed: ", random_seed)
	random.seed(random_seed)
	torch.manual_seed(random_seed)

	log_path = 'D:/Research_data/train_logs'
	log_folder = 'log_05'
	checkpoint_name = 'checkpoint.pt.tar'
	log_name = 'log.csv'
	summary_name = 'summary.txt'
	device = 'cpu'
	iterations = 20000

	log_root = os.path.join(log_path, log_folder)
	if not os.path.exists(log_root):
		os.mkdir(log_root)

	opt_input = False
	lr_decay_epoch = 0
	reg_noise_std = 0.005
	reg_noise_decay_every = 500

	path = 'C:/Users/e-min\Desktop/supplement_deep_decoder-master/test_data/astronaut.png'

	image = np.asarray(Image.open(path)).astype('float32') / 255.
	image = Variable(torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)).to(device)

	net_input = Variable(torch.zeros([1, 32, 16, 16]))
	net_input.data.uniform_()
	net_input.data *= 1. / 10

	net = DeepDecoder3(layers_channels=[32]*5, out_channels=3, kernel_sizes=[1]*5)

	net_input_saved = net_input.data.clone()
	noise = net_input.data.clone()
	params = [x for x in net.parameters()]


	if opt_input == True:  # optimizer over the input as well
		net_input.requires_grad = True
		params += [net_input]

	optimizer = optim.Adam(params, lr=0.005, weight_decay=0)

	log_dir = os.path.join(log_root, log_name)
	if os.path.exists(log_dir):
		log = pd.read_csv(log_dir, index_col=0)
	else:
		log = pd.DataFrame(columns=['iteration', 'loss_val', 'loss_avg'])

	checkpoint_dir = os.path.join(log_root, checkpoint_name)
	start_i = 0
	if os.path.exists(checkpoint_dir):
		checkpoint = torch.load(checkpoint_dir)
		start_i = checkpoint['iteration'] + 1
		net = checkpoint['net']
		optimizer = checkpoint['optimizer']
		print("\nLoaded checkpoint from iteration %d.\n" % (checkpoint['iteration'] + 1))

	mse = torch.nn.MSELoss()  # .type(dtype)
	loss_cache = []
	for i in range(start_i, iterations):
		if lr_decay_epoch is not 0:
			#optimizer = exp_lr_scheduler(optimizer, i, init_lr=LR, lr_decay_epoch=lr_decay_epoch)
			pass
		if reg_noise_std > 0:
			if i % reg_noise_decay_every == 0:
				reg_noise_std *= 0.7
			net_input = Variable(net_input_saved + (noise.normal_() * reg_noise_std))

		optimizer.zero_grad()
		out = net(net_input.to(device))

		# training loss
		loss = mse(out, image)
		loss.backward()

		loss_cache.append(loss.data.item())
		if len(loss_cache) > 50:
			loss_cache.pop(0)

		loss_avg = sum(loss_cache)/len(loss_cache)

		log = pd.concat([log, pd.DataFrame({
			'iteration': i,
			'loss_val': loss.data.item(),
			'loss_avg': loss_avg}, index=[0])], sort=False, axis=0, ignore_index=True)

		if i % 10 == 0:
			out2 = net(Variable(net_input_saved).to(device))
			loss2 = mse(out2, image)

			report = ['iteration: ' + str(i) + '/' + str(iterations - 1),
					  ' ===> loss: {loss:.5f}'.format(loss=loss.data.item()),
					  '({loss:.5f} average)'.format(loss=loss_avg),
					  '- actual: {loss:.5f}'.format(loss=loss2.data.item())]

			if loss.data.item() < loss_avg - 0.00001:
				torch.save({'iteration': i,
							'net': net,
							'optimizer': optimizer},
						   checkpoint_dir)
				report.append('- Loss reduced to ' + '{loss:.5f}'.format(loss=loss.data.item()) + ', model is saved.')

			print(' '.join(report))

			log.to_csv(log_dir, header=True,
					   columns=['iteration', 'loss_val', 'loss_avg'])

			plt.imshow(out.detach().permute(0, 2, 3, 1).numpy()[0])
			plt.show()
		else:
			print('iteration: ' + str(i) + '/' + str(iterations - 1),
				  ' ===> loss: {loss:.5f}'.format(loss=loss.data.item()),
				  '({loss:.5f} average)'.format(loss=loss_avg))

		optimizer.step()