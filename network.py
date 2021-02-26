from tensorflow.keras import Model, layers as ls, activations as acts


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