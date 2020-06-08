import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from train_utils import HologramDataset, AverageMeter, adjust_learning_rate
from Resnet_model import Model, weights_init
from torchsummary import summary
import numpy as np
import random
import os
import pandas as pd
import time
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def train(config, model, loss_criterion, optimizer, epoch, log):
	batch_time = AverageMeter()  # forward prop. + back prop. time
	data_time = AverageMeter()  # data loading time
	losses = AverageMeter()  # loss

	start = time.time()
	new_log = log
	print('Starting epoch ' + str(epoch))
	for i, sample in enumerate(config.data_loader, 0):
		data_time.update(time.time() - start)

		batch = sample['input'].to(config.device)
		output = model(batch)

		target = sample['output'].to(config.device)
		loss = loss_criterion(output, target)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		losses.update(loss.item(), config.batch_size)
		batch_time.update(time.time() - start)
		start = time.time()
		new_log = pd.concat([new_log, pd.DataFrame({
			'epoch': epoch,
			'iteration': i,
			'loss_val': losses.val,
			'loss_average': losses.avg,
			'batch_time': batch_time.val,
			'batch_time_avg': batch_time.avg,
			'data_time': data_time.val,
			'data_time_avg': data_time.avg}, index=[0])], sort=False, axis=0, ignore_index=True)

		epochs = int(config.iterations // len(config.data_loader) + 1)
		print('epoch: ' + str(epoch) + '/' + str(epochs - 1),
			  '_ iteration: ' + str(i) + '/' + str(len(config.data_loader)),
			  ' ===> loss:' + str(losses.val),
			  '({loss.avg:.4f} average)'.format(loss=losses))

	return new_log

def main(config):
	epochs = int(config.iterations // len(config.data_loader) + 1)

	log_dir = os.path.join(config.log_root, config.log_folder)
	if not os.path.exists(log_dir):
		os.mkdir(log_dir)

	# dummy = torch.ones((1, 1, 256, 256))  # ((Batch size, channels, h, w))
	# print(model)
	# print(model(dummy).shape)

	loss_criterion = nn.MSELoss()

	if os.path.exists(os.path.join(log_dir, config.checkpoint_name)):
		checkpoint = torch.load(os.path.join(log_dir, config.checkpoint_name))
		start_epoch = checkpoint['epoch'] + 1
		model = checkpoint['model']
		optimizer = checkpoint['optimizer']
		print("\nLoaded checkpoint from epoch %d.\n" % (checkpoint['epoch'] + 1))
	else:
		start_epoch = 0
		model = Model().to(config.device)

		model_summary = summary(model, input_size=(1, 128, 128),
								batch_size=config.batch_size, device=config.device)
		with open(os.path.join(log_dir, config.summary_file_name), 'w') as text_file:
			text_file.write(model_summary)

		model.apply(weights_init)
		model.train()
		# optimizer = optim.SGD(model.parameters(), lr=0.0002, momentum=0.9)
		optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

	if os.path.exists(os.path.join(log_dir, config.log_file_name)):
		log = pd.read_csv(os.path.join(log_dir, config.log_file_name), index_col=0)
	else:
		log = pd.DataFrame(columns=['epoch', 'iteration', 'loss_val',
									'loss_average', 'batch_time', 'batch_time_avg',
									'data_time', 'data_time_avg'])

	'''
	for i, sample in enumerate(config.data_loader, 0):
		out = model(sample['input'].to(config.device))
		#out = sample['input'].to(DEVICE)
		plt.imshow(out[0, 0, :, :].detach().numpy())
		plt.show()
		if i==0:
			break
	'''

	print("Starting Training Loop...")
	for epoch in range(start_epoch, epochs):

		if epoch == int((config.iterations / 2) // len(config.data_loader) + 1):
			adjust_learning_rate(optimizer, 0.1)

		log = train(
			config=config,
			model=model,
			loss_criterion=loss_criterion,
			optimizer=optimizer,
			epoch=epoch,
			log=log)

		torch.save({'epoch': epoch,
					'model': model,
					'optimizer': optimizer},
				   os.path.join(log_dir, config.checkpoint_name))

		# data logging
		log.to_csv(os.path.join(log_dir, config.log_file_name), header=True,
				   columns=['epoch', 'iteration', 'loss_val',
						 'loss_average', 'batch_time', 'batch_time_avg',
						 'data_time', 'data_time_avg'])
