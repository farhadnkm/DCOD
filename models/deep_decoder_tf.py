import tensorflow as tf
from tensorflow.keras import Model, layers as ls, activations as acts, regularizers as regs
import math

'''
##################### MHE regularization ########################
### https://github.com/wy1iu/MHE/blob/master/code/architecture.py
def get_conv_filter(shape, reg, stddev):
	init = tf.random_normal_initializer(stddev=stddev)
	if reg:
		regu = regs.l2(0.002)
		filt = tf.compat.v1.get_variable('filter', shape, initializer=init, regularizer=regu)
	else:
		filt = tf.compat.v1.get_variable('filter', shape, initializer=init)
	return filt


def _add_thomson_constraint(filt, n_filt, model, power):
	filt = tf.reshape(filt, [-1, n_filt])
	if model == 'half_mhe':
		filt_neg = filt * -1
		filt = tf.concat((filt, filt_neg), axis=1)
		n_filt *= 2
	filt_norm = tf.sqrt(tf.reduce_sum(filt * filt, [0], keep_dims=True) + 1e-4)
	norm_mat = tf.matmul(tf.transpose(filt_norm), filt_norm)
	inner_pro = tf.matmul(tf.transpose(filt), filt)
	inner_pro /= norm_mat

	if power == '0':
		cross_terms = 2.0 - 2.0 * inner_pro
		final = -tf.math.log(cross_terms + tf.compat.v1.diag([1.0] * n_filt))
		final -= tf.compat.v1.matrix_band_part(final, -1, 0)
		cnt = n_filt * (n_filt - 1) / 2.0
		loss = 1 * tf.reduce_sum(final) / cnt
	elif power == '1':
		cross_terms = (2.0 - 2.0 * inner_pro + tf.compat.v1.diag([1.0] * n_filt))
		final = tf.pow(cross_terms, tf.ones_like(cross_terms) * (-0.5))
		final -= tf.compat.v1.matrix_band_part(final, -1, 0)
		cnt = n_filt * (n_filt - 1) / 2.0
		loss = 1 * tf.reduce_sum(final) / cnt
	elif power == '2':
		cross_terms = (2.0 - 2.0 * inner_pro + tf.compat.v1.diag([1.0] * n_filt))
		final = tf.pow(cross_terms, tf.ones_like(cross_terms) * (-1))
		final -= tf.compat.v1.matrix_band_part(final, -1, 0)
		cnt = n_filt * (n_filt - 1) / 2.0
		loss = 1 * tf.reduce_sum(final) / cnt
	elif power == 'a0':
		acos = tf.acos(inner_pro) / math.pi
		acos += 1e-4
		final = -tf.math.log(acos)
		final -= tf.compat.v1.matrix_band_part(final, -1, 0)
		cnt = n_filt * (n_filt - 1) / 2.0
		loss = 1 * tf.reduce_sum(final) / cnt
	elif power == 'a1':
		acos = tf.acos(inner_pro) / math.pi
		acos += 1e-4
		final = tf.pow(acos, tf.ones_like(acos) * (-1))
		final -= tf.compat.v1.matrix_band_part(final, -1, 0)
		cnt = n_filt * (n_filt - 1) / 2.0
		loss = 1e-1 * tf.reduce_sum(final) / cnt
	elif power == 'a2':
		acos = tf.acos(inner_pro) / math.pi
		acos += 1e-4
		final = tf.pow(acos, tf.ones_like(acos) * (-2))
		final -= tf.compat.v1.matrix_band_part(final, -1, 0)
		cnt = n_filt * (n_filt - 1) / 2.0
		loss = 1e-1 * tf.reduce_sum(final) / cnt

	tf.compat.v1.add_to_collection('thomson_loss', loss)


def _add_thomson_constraint_final(filt, n_filt, power):
	filt = tf.reshape(filt, [-1, n_filt])
	filt_norm = tf.sqrt(tf.reduce_sum(filt * filt, [0], keep_dims=True) + 1e-4)
	norm_mat = tf.matmul(tf.transpose(filt_norm), filt_norm)
	inner_pro = tf.matmul(tf.transpose(filt), filt)
	inner_pro /= norm_mat

	if power == '0':
		cross_terms = 2.0 - 2.0 * inner_pro
		final = -tf.math.log(cross_terms + tf.compat.v1.diag([1.0] * n_filt))
		final -= tf.compat.v1.matrix_band_part(final, -1, 0)
		cnt = n_filt * (n_filt - 1) / 2.0
		loss = 10 * tf.reduce_sum(final) / cnt
	elif power == '1':
		cross_terms = (2.0 - 2.0 * inner_pro + tf.compat.v1.diag([1.0] * n_filt))
		final = tf.pow(cross_terms, tf.ones_like(cross_terms) * (-0.5))
		final -= tf.compat.v1.matrix_band_part(final, -1, 0)
		cnt = n_filt * (n_filt - 1) / 2.0
		loss = 10 * tf.reduce_sum(final) / cnt
	elif power == '2':
		cross_terms = (2.0 - 2.0 * inner_pro + tf.compat.v1.diag([1.0] * n_filt))
		final = tf.pow(cross_terms, tf.ones_like(cross_terms) * (-1))
		final -= tf.compat.v1.matrix_band_part(final, -1, 0)
		cnt = n_filt * (n_filt - 1) / 2.0
		loss = 10 * tf.reduce_sum(final) / cnt
	elif power == 'a0':
		acos = tf.acos(inner_pro) / math.pi
		acos += 1e-4
		final = -tf.math.log(acos)
		final -= tf.compat.v1.matrix_band_part(final, -1, 0)
		cnt = n_filt * (n_filt - 1) / 2.0
		loss = 10 * tf.reduce_sum(final) / cnt
	elif power == 'a1':
		acos = tf.acos(inner_pro) / math.pi
		acos += 1e-4
		final = tf.pow(acos, tf.ones_like(acos) * (-1))
		final -= tf.compat.v1.matrix_band_part(final, -1, 0)
		cnt = n_filt * (n_filt - 1) / 2.0
		loss = 1 * tf.reduce_sum(final) / cnt
	elif power == 'a2':
		acos = tf.acos(inner_pro) / math.pi
		acos += 1e-4
		final = tf.pow(acos, tf.ones_like(acos) * (-2))
		final -= tf.compat.v1.matrix_band_part(final, -1, 0)
		cnt = n_filt * (n_filt - 1) / 2.0
		loss = 1 * tf.reduce_sum(final) / cnt

	tf.compat.v1.add_to_collection('thomson_final', loss)
'''

def deep_decoder(input_shape, layers_channels, out_channels, kernel_sizes, upsample_mode='bilinear',
			 activation_func=ls.ReLU(), out_activation=acts.sigmoid, bn_affine=True):
	layers = []
	input = ls.Input(shape=input_shape, batch_size=1)
	for i in range(len(layers_channels)):
		try:
			out_layer_channels = layers_channels[i + 1]
		except IndexError:
			out_layer_channels = layers_channels[-1]

		layers.append(ls.Conv2D(out_layer_channels, kernel_sizes[i], strides=1, padding='same'))

		layers.append(ls.UpSampling2D(interpolation=upsample_mode))
		if activation_func is not None:
			layers.append(activation_func)
		layers.append(ls.BatchNormalization(trainable=bn_affine))

	layers.append(ls.Conv2D(layers_channels[-1], kernel_sizes[-1], strides=1, padding='same'))
	if activation_func is not None:
		layers.append(activation_func)
	layers.append(ls.BatchNormalization(trainable=bn_affine))

	layers.append(ls.Conv2D(out_channels, kernel_sizes[-1], strides=1, padding='same'))
	if out_activation is not None:
		layers.append(out_activation)

	out = input
	for layer in layers:
		out = layer(out)
	return Model(input, out, name="DeepDecoder")

'''
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#model.build(input_shape=dummy.shape)
#model.summary()
random_seed = 999
input_t = tf.random.normal([1, 16, 16, 64], mean=0, stddev=0.1, seed=random_seed)
model = deep_decoder(input_shape=input_t[0].shape,
					 layers_channels=[64, 32, 16, 8, 4],
					 kernel_sizes=[1]*5,
					 out_channels=3,
					 upsample_mode='bilinear',
					 activation_func=ls.ReLU(),
					 out_activation=acts.sigmoid,
					 bn_affine=True)


path = 'C:/Users/e-min\Desktop/supplement_deep_decoder-master/test_data/astronaut.png'
image = np.asarray(Image.open(path)).astype('float32') / 255.
target = tf.convert_to_tensor(image, dtype=tf.float32)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
mse = tf.keras.losses.MeanSquaredError()

num_epochs = 2000
for epoch in range(num_epochs):
	epoch_loss_avg = tf.keras.metrics.Mean()
	epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

	# Optimize the model
	with tf.GradientTape() as tape:
		out = model(input_t, training=True)
		loss_value = mse(out, target)
	grads = tape.gradient(loss_value, model.trainable_variables)
	optimizer.apply_gradients(zip(grads, model.trainable_variables))

	if epoch % 5 == 0:
		print("Epoch {:03d}: Loss: {:.5f}".format(epoch, loss_value))
	if epoch % 20 == 0:
		plt.imshow(out.numpy()[0])
		plt.show()
'''