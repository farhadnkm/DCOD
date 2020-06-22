from utils.data import AverageMeter, adjust_learning_rate, clip_gradient
import os
import pandas as pd
import time
import torch


def train_resnet(config, epoch, log):

	config.resnet.train()

	batch_time = AverageMeter()  # forward prop. + back prop. time
	data_time = AverageMeter()  # data loading time
	losses = AverageMeter()  # loss

	start = time.time()
	new_log = log
	print('Starting epoch ' + str(epoch))
	for i, sample in enumerate(config.data_loader, 0):
		data_time.update(time.time() - start)

		input_batch = sample['input'].to(config.device)
		recovered = config.resnet(input_batch)

		target = sample['output'].to(config.device)
		loss = config.loss_criterion(recovered, target)
		config.optimizer.zero_grad()
		loss.backward()
		config.optimizer.step()

		losses.update(loss.item(), config.batch_size)
		batch_time.update(time.time() - start)

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

	del input_batch, target, recovered
	return new_log


def train_gan(config, epoch, log):

	config.generator.train()
	config.discriminator.train()

	batch_time = AverageMeter()  # forward prop. + back prop. time
	data_time = AverageMeter()  # data loading time
	losses_c = AverageMeter()  # content loss
	losses_a_g = AverageMeter()  # adversarial loss in the generator
	losses_a_d = AverageMeter()  # adversarial loss in the discriminator

	start = time.time()
	new_log = log
	for i, sample in enumerate(config.data_loader, 0):
		data_time.update(time.time() - start)
		# Move to default device
		input_batch = sample['input'].to(config.device)

		#sr_imgs = convert_image(sr_imgs, source='[-1, 1]', target='imagenet-norm')  # (N, 3, 96, 96), imagenet-normed

		target = sample['output'].to(config.device)

		# GENERATOR UPDATE

		recovered = config.generator(input_batch)
		recovered_3c = recovered[:, 0, :, :]
		recovered_in_vgg_space = config.truncated_vgg19(recovered)
		target_in_vgg_space = config.truncated_vgg19(target).detach() # detached because they're constant, targets

		recovered_discriminated = config.discriminator(recovered)

		content_loss = config.content_loss_criterion(recovered_in_vgg_space, target_in_vgg_space)
		adversarial_loss = config.adversarial_loss_criterion(recovered_discriminated, torch.ones_like(recovered_discriminated))

		perceptual_loss = content_loss + config.beta * adversarial_loss

		config.optimizer_g.zero_grad()
		perceptual_loss.backward()

		if config.grad_clip is not None:
			clip_gradient(config.optimizer_g, config.grad_clip)

		config.optimizer_g.step()

		losses_c.update(content_loss.item(), input_batch.size(0))
		losses_a_g.update(adversarial_loss.item(), input_batch.size(0))

		# DISCRIMINATOR UPDATE

		target_discriminated = config.discriminator(target)
		recovered_discriminated = config.discriminator(recovered.detach())
		# But didn't we already discriminate the SR images earlier, before updating the generator (G)? Why not just use that here?
		# Because, if we used that, we'd be back-propagating (finding gradients) over the G too when backward() is called
		# It's actually faster to detach the SR images from the G and forward-prop again, than to back-prop. over the G unnecessarily
		# See FAQ section in the tutorial

		adversarial_loss = config.adversarial_loss_criterion(recovered_discriminated, torch.zeros_like(recovered_discriminated)) + \
							config.adversarial_loss_criterion(target_discriminated, torch.ones_like(target_discriminated))

		config.optimizer_d.zero_grad()
		adversarial_loss.backward()

		if config.grad_clip is not None:
			clip_gradient(config.optimizer_d, config.grad_clip)

		config.optimizer_d.step()

		losses_a_d.update(adversarial_loss.item(), target.size(0))
		batch_time.update(time.time() - start)

		new_log = pd.concat([new_log, pd.DataFrame({
			'epoch': epoch,
			'iteration': i,
			'loss_g_val': losses_a_g.val,
			'loss_g_average': losses_a_g.avg,
			'loss_d_val': losses_a_d.val,
			'loss_d_average': losses_a_d.avg,
			'loss_c_val': losses_c.val,
			'loss_c_average': losses_c.avg,
			'batch_time': batch_time.val,
			'batch_time_avg': batch_time.avg,
			'data_time': data_time.val,
			'data_time_avg': data_time.avg
		}, index=[0])], sort=False, axis=0, ignore_index=True)

		epochs = int(config.iterations // len(config.data_loader) + 1)

		print('epoch: ' + str(epoch) + '/' + str(epochs - 1),
			  '_ iteration: ' + str(i) + '/' + str(len(config.data_loader)-1),
			  ' ===> losses: G: {losses_a_g.val:.4f} ({losses_a_g.avg:.4f} avg), '.format(losses_a_g=losses_a_g),
			  'D: {losses_a_d.val:.4f} ({losses_a_d.avg:.4f} avg), '.format(losses_a_d=losses_a_d),
			  'C: {losses_c.val:.4f} ({losses_c.avg:.4f} avg), '.format(losses_c=losses_c))

	del input_batch, target, recovered, target_in_vgg_space, recovered_in_vgg_space, target_discriminated, recovered_discriminated  # free some memory since their histories may be stored
	return new_log


def run_resnet(config):
	global log

	epochs = int(config.iterations // len(config.data_loader) + 1)

	if os.path.exists(config.checkpoint_dir):
		log = pd.read_csv(config.checkpoint_dir, index_col=0)
	else:
		log = pd.DataFrame(columns=['epoch', 'iteration', 'loss_val',
									'loss_average', 'batch_time', 'batch_time_avg',
									'data_time', 'data_time_avg'])

	print("Starting Training Loop...")
	for epoch in range(config.start_epoch, epochs):

		if epoch == int((config.iterations / 2) // len(config.data_loader) + 1):
			adjust_learning_rate(config.optimizer, 0.1)

		log = train_resnet(
			config=config,
			epoch=epoch,
			log=log)

		torch.save({'epoch': epoch,
					'resnet': config.resnet,
					'optimizer': config.optimizer},
				   config.checkpoint_dir)

		# data logging
		log.to_csv(config.log_dir, header=True,
				   columns=['epoch', 'iteration', 'loss_val',
						 'loss_average', 'batch_time', 'batch_time_avg',
						 'data_time', 'data_time_avg'])


def run_gan(config):
	global log

	# Total number of epochs to train for
	epochs = int(config.iterations // len(config.data_loader) + 1)

	if os.path.exists(config.log_dir):
		log = pd.read_csv(config.log_dir, index_col=0)
	else:
		log = pd.DataFrame(columns=['epoch', 'iteration', 'loss_g_val',
									'loss_g_average', 'loss_d_val', 'loss_d_average',
									'loss_c_val', 'loss_c_average',
									'batch_time', 'batch_time_avg',
									'data_time', 'data_time_avg'])

	print("Starting Training Loop...")
	for epoch in range(config.start_epoch, epochs):

		# At the halfway point, reduce learning rate to a tenth
		if epoch == int((config.iterations / 2) // len(config.data_loader) + 1):
			adjust_learning_rate(config.optimizer_g, 0.1)
			adjust_learning_rate(config.optimizer_d, 0.1)

		# One epoch's training
		log = train_gan(
			config=config,
			epoch=epoch,
			log=log
		)

		# Save checkpoint
		torch.save({'epoch': epoch,
					'generator': config.generator,
					'discriminator': config.discriminator,
					'optimizer_g': config.optimizer_g,
					'optimizer_d': config.optimizer_d},
				   config.gan_checkpoint_name)

		# data logging
		log.to_csv(config.gan_log_dir, header=True,
				   columns=['epoch', 'iteration', 'loss_g_val',
							'loss_g_average', 'loss_d_val', 'loss_d_average',
							'loss_c_val', 'loss_c_average',
							'batch_time', 'batch_time_avg',
							'data_time', 'data_time_avg'])
