from utils.data import AverageMeter, adjust_learning_rate
import os
import pandas as pd
import time
import torch


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
			  '_ iteration: ' + str(i) + '/' + str(len(config.data_loader)-1),
			  ' ===> loss:' + str(losses.val),
			  '({loss.avg:.4f} average)'.format(loss=losses))

	return new_log

def run(config):
	epochs = int(config.iterations // len(config.data_loader) + 1)

	if os.path.exists(os.path.join(config.log_dir, config.log_name)):
		log = pd.read_csv(os.path.join(config.log_dir, config.log_name), index_col=0)
	else:
		log = pd.DataFrame(columns=['epoch', 'iteration', 'loss_val',
									'loss_average', 'batch_time', 'batch_time_avg',
									'data_time', 'data_time_avg'])

	print("Starting Training Loop...")
	for epoch in range(config.start_epoch, epochs):

		if epoch == int((config.iterations / 2) // len(config.data_loader) + 1):
			adjust_learning_rate(config.optimizer, 0.1)

		log = train(
			config=config,
			model=config.model,
			loss_criterion=config.loss_criterion,
			optimizer=config.optimizer,
			epoch=epoch,
			log=log)

		torch.save({'epoch': epoch,
					'model': config.model,
					'optimizer': config.optimizer},
				   os.path.join(config.log_dir, config.checkpoint_name))

		# data logging
		log.to_csv(os.path.join(config.log_dir, config.log_name), header=True,
				   columns=['epoch', 'iteration', 'loss_val',
						 'loss_average', 'batch_time', 'batch_time_avg',
						 'data_time', 'data_time_avg'])
