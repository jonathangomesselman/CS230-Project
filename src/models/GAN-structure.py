import tensorflow as tf


def lrelu(inputs, alpha=0.2):
  return tf.maximum(alpha * inputs, inputs)


def apply_phaseshuffle(x, rad, pad_type='reflect'):
  b, x_len, nch = x.get_shape().as_list()

  phase = tf.random_uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
  pad_l = tf.maximum(phase, 0)
  pad_r = tf.maximum(-phase, 0)
  phase_start = pad_r
  x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

  x = x[:, phase_start:phase_start+x_len]
  x.set_shape([b, x_len, nch])

  return x


'''
Here we define the architecture for the descriminator. We are using the
Descriminator architecture from the WaveGan paper. 
'''
"""
  We likely need to change this???
  Input: [None, 16384, 1]
  # Likely want this to be 16000!!!!!!!!!!!!
  Output: [None] (linear output)
"""
def Discriminator(x, kernel_len=25, dim=64, use_batchnorm=False, phaseshuffle_rad=0):
	batch_size = tf.shape(x)[0]

	if use_batchnorm:
		batchnorm = lambda x: tf.layers.batch_normalization(x, training=True)
	else:
		batchnorm = lambda x: x

	if phaseshuffle_rad > 0:
		phaseshuffle = lambda x: apply_phaseshuffle(x, phaseshuffle_rad)
	else:
		phaseshuffle = lambda x: x

	# Layer 0
  	# [16384, 1] -> [4096, 64]
  	output = x
  	with tf.variable_scope('downconv_0'):
    	output = tf.layers.conv1d(output, dim, kernel_len, 4, padding='SAME')
  	output = lrelu(output)
  	output = phaseshuffle(output)

  	# Layer 1
  	# [4096, 64] -> [1024, 128]
  	with tf.variable_scope('downconv_1'):
    	output = tf.layers.conv1d(output, dim * 2, kernel_len, 4, padding='SAME')
    	output = batchnorm(output)
  	output = lrelu(output)
  	output = phaseshuffle(output)

  	# Layer 2
  	# [1024, 128] -> [256, 256]
  	with tf.variable_scope('downconv_2'):
    	output = tf.layers.conv1d(output, dim * 4, kernel_len, 4, padding='SAME')
    	output = batchnorm(output)
  	output = lrelu(output)
  	output = phaseshuffle(output)

  	# Layer 3
  	# [256, 256] -> [64, 512]
  	with tf.variable_scope('downconv_3'):
    	output = tf.layers.conv1d(output, dim * 8, kernel_len, 4, padding='SAME')
    	output = batchnorm(output)
  	output = lrelu(output)
  	output = phaseshuffle(output)

  	# Layer 4
  	# [64, 512] -> [16, 1024]
  	with tf.variable_scope('downconv_4'):
    	output = tf.layers.conv1d(output, dim * 16, kernel_len, 4, padding='SAME')
    	output = batchnorm(output)
  	output = lrelu(output)

  	# Flatten
  	output = tf.reshape(output, [batch_size, 4 * 4 * dim * 16])

  	# Connect to single logit
  	with tf.variable_scope('output'):
    	output = tf.layers.dense(output, 1)[:, 0]

  	# Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training

  	return output