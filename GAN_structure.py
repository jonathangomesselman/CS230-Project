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
    Input: [None, 16384, 1] - Old
    Input-New: [None, 8192, 1]
    Output: [None] (linear output)
"""
def Discriminator(x, kernel_len=25, dim=64, use_batchnorm=False, phaseshuffle_rad=0):
    batch_size = tf.shape(x)[0]
    # Avoid batchnorm
    #batch_size = 1

    if use_batchnorm:
	   batchnorm = lambda x: tf.layers.batch_normalization(x, training=True)
    else:
	   batchnorm = lambda x: x

    if phaseshuffle_rad > 0:
	   phaseshuffle = lambda x: apply_phaseshuffle(x, phaseshuffle_rad)
    else:
	   phaseshuffle = lambda x: x

    # Layer 0
    # [16384, 1] -> [4096, 64] - Old
    # [8192, 1] -> [2048, 64] - New
    output = x
    print output 
    with tf.variable_scope('downconv_0'):
        #output = tf.layers.conv1d(output, dim, kernel_len, 4, padding='SAME')
        filter = tf.get_variable('Fl', shape=[kernel_len, 1, dim],
                          initializer=tf.random_normal_initializer(stddev=1e-3))
        #filter = tf.zeros([3, 16, 16])
        output = tf.nn.conv1d(output, filters=filter, stride=4, padding='SAME')
    output = lrelu(output)
    output = phaseshuffle(output)

    # Layer 1
    # [4096, 64] -> [1024, 128] - Old
    # [2048, 64] -> [512, 128] - New
    with tf.variable_scope('downconv_1'):
        filter = tf.get_variable('Fl', shape=[kernel_len, dim, dim * 2],
                          initializer=tf.random_normal_initializer(stddev=1e-3))
        output = tf.nn.conv1d(output, filters=filter, stride=4, padding='SAME')
        output = batchnorm(output)
    output = lrelu(output)
    output = phaseshuffle(output)

    # Layer 2
    # [1024, 128] -> [256, 256] - Old
    # [512, 128] -> [128, 256] - New
    with tf.variable_scope('downconv_2'):
        filter = tf.get_variable('Fl', shape=[kernel_len, dim * 2, dim * 4],
                          initializer=tf.random_normal_initializer(stddev=1e-3))
        output = tf.nn.conv1d(output, filters=filter, stride=4, padding='SAME')
        output = batchnorm(output)
    output = lrelu(output)
    output = phaseshuffle(output)

    # Layer 3
    # [256, 256] -> [64, 512] - Old
    # [128, 256] -> [32, 512] - New
    with tf.variable_scope('downconv_3'):
        filter = tf.get_variable('Fl', shape=[kernel_len, dim * 4, dim * 8],
                          initializer=tf.random_normal_initializer(stddev=1e-3))
    	output = tf.nn.conv1d(output, filters=filter, stride=4, padding='SAME')
    	output = batchnorm(output)
    output = lrelu(output)
    output = phaseshuffle(output)

  	# Layer 4
  	# [64, 512] -> [16, 1024]
    # [32, 512] -> [8, 1024]
    with tf.variable_scope('downconv_4'):
        filter = tf.get_variable('Fl', shape=[kernel_len, dim * 8, dim * 16],
                          initializer=tf.random_normal_initializer(stddev=1e-3))
    	output = tf.nn.conv1d(output, filters=filter, stride=4, padding='SAME')
    	output = batchnorm(output)
    output = lrelu(output)

    # Flatten
    # We now have to flatten it to a different size [8192]
    #output = tf.reshape(output, [batch_size, 4 * 4 * dim * 16])
    output = tf.reshape(output, [batch_size, 8192])

  	# Connect to single logit
    with tf.variable_scope('output'):
        #W = tf.get_variable('W', shape=[4 * 4 * dim * 16, 1], initializer=tf.random_normal_initializer(stddev=1e-3))
        # New shape 8192
        W = tf.get_variable('W', shape=[8192, 1], initializer=tf.random_normal_initializer(stddev=1e-3))
        b = tf.zeros([batch_size, 1])
        output = tf.matmul(output, W) + b
        output = output[:, 0]
        #output = tf.layers.dense(output, 1)[:, 0]

  	# Don't need to aggregate batchnorm update ops like we do for the generator because we only use the discriminator for training

  	return output