import numpy as np
import os
import pandas as pd
import time
import torch
from utils.data import AverageMeter, adjust_learning_rate, clip_gradient
#import simulator.simulation_utils as utils
import math
import matplotlib.pyplot as plt

def train(config, iter, log):
	torch.autograd.set_detect_anomaly(True)
	config.decoder.train()

	iter_time = AverageMeter()  # forward prop. + back prop. time
	losses = AverageMeter()  # loss

	start = time.time()

	input_feature_maps = config.dataset.input_feature_maps.to(config.device)
	result = config.decoder(input_feature_maps)

	if math.remainder(iter, 20) == 0 and iter != 0:
		plt.imshow(result.detach().numpy()[0, 0], cmap='gray')
		plt.show()
	'''
	image_ph = np.exp(1j * result.detach().numpy())
	amp = np.abs(config.dataset.simulator.reconstruct(initializer=image_ph, z=2)) ** 2
	amp = 2 - amp
	image = amp * image_ph
	est_hol_amp = np.abs(config.dataset.simulator.reconstruct(initializer=image, z=-1*config.dataset.init_z)) ** 2
	est_hol_amp /= np.max(est_hol_amp)

	result.data = torch.from_numpy(est_hol_amp.astype('float32')).to(config.device)'''
	target = torch.from_numpy(config.dataset.hologram_amp).unsqueeze(0).unsqueeze(0).to(config.device)

	loss = config.loss_criterion(result, target)
	config.optimizer.zero_grad()
	loss.backward()
	config.optimizer.step()

	losses.update(loss.item())
	iter_time.update(time.time() - start)

	print('iteration: ' + str(iter) + '/' + str(config.iterations - 1),
		  ' ===> loss:' + str(losses.val),
		  '({loss.avg:.4f} average)'.format(loss=losses))

	del input_feature_maps, result#, image_ph, amp, image, target, est_hol_amp

	return pd.concat([log, pd.DataFrame({
		'iteration': iter,
		'loss_val': losses.val,
		'loss_average': losses.avg,
		'iter_time': iter_time.val,
		'iter_time_avg': iter_time.avg}, index=[0])], sort=False, axis=0, ignore_index=True)


def run(config):
	global log

	if os.path.exists(config.log_dir):
		log = pd.read_csv(config.log_dir, index_col=0)
	else:
		log = pd.DataFrame(columns=['iteration', 'loss_val', 'loss_average',
									'iter_time', 'iter_time_avg'])

	print("Starting Training Loop...")
	for iter in range(config.start_iter, config.iterations):

		log = train(
			config=config,
			iter=iter,
			log=log)

		if math.remainder(iter, 50) == 0 and iter != 0:
			torch.save({'iteration': iter,
						'decoder': config.decoder,
						'optimizer': config.optimizer},
					   config.checkpoint_dir)

			# data logging
			log.to_csv(config.log_dir, header=True,
					   columns=['iteration', 'loss_val',
							 'loss_average', 'iter_time', 'iter_time_avg'])
