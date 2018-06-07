import os
import time

import numpy as np
import tensorflow as tf
import keras
import librosa
from scipy import interpolate
from scipy.signal import decimate
from h5Converter import load_h5

r = 4
layers = 4

def spline_up(x_lr, r):
  x_lr = x_lr.flatten()
  x_hr_len = len(x_lr) * r
  x_sp = np.zeros(x_hr_len)
  
  i_lr = np.arange(x_hr_len, step=r)
  i_hr = np.arange(x_hr_len)
  
  f = interpolate.splrep(i_lr, x_lr)

  x_sp = interpolate.splev(i_hr, f)
  #print('x_sp', x_sp.shape)

  return x_sp

def load_batch(batch, inputs, alpha=1, train=True):
    X_in, Y_in, alpha_in = inputs
    X, Y = batch
    
    if Y is not None:
      feed_dict = {X_in : X, Y_in : Y, alpha_in : alpha}
    else:
      feed_dict = {X_in : X, alpha_in : alpha}

    # this is ugly, but only way I found to get this var after model reload
    g = tf.get_default_graph()
    k_tensors = [n for n in g.as_graph_def().node if 'keras_learning_phase' in n.name]
    #assert len(k_tensors) <= 1
    if k_tensors: 
      k_learning_phase = g.get_tensor_by_name(k_tensors[1].name + ':0')
      feed_dict[k_learning_phase] = train

    return feed_dict

def returnFeedDict(X, inputs, predictions, sess):
    '''assert len(X) == 1
    x_sp = spline_up(X, r)
    x_sp = x_sp[:len(x_sp) - (len(x_sp) % (2**(layers+1)))]
    X = x_sp.reshape((1,len(x_sp),1))'''
    feed_dict = load_batch((X,X), inputs, train=True)
    return feed_dict
    #return sess.run(predictions, feed_dict=feed_dict)

class AudioSRGanModel:
    model_name = 'AudioSRGAN'

    # Note used to have a config variable here
    # This will eventually be very useful I think for loading in the settings
    def __init__(self, config, ckpt, batch_size=1, H_R=8192, L_R=8192, sess=None):
        self.H_R = H_R
        self.L_R = L_R
        self.batch_size = batch_size
        self.ckpt = ckpt

        self.config = config
        self.sess = sess

    def generator(self):
        # get checkpoint name
        if os.path.isdir(self.ckpt): checkpoint = tf.train.latest_checkpoint(self.ckpt)
        else: checkpoint = self.ckpt

        meta = checkpoint + '.meta'
        print checkpoint

        # load graph
        saver = tf.train.import_meta_graph(meta)
        g = tf.get_default_graph()

        # load weights
        saver.restore(self.sess, checkpoint)
        # Get the input tensor for the generator
        self.input_generator, Y, alpha = tf.get_collection('inputs')
        print self.input_generator.shape
        #self.input_generator = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminatorGan')
        # Get graph tensors
        predictions = tf.get_collection('preds')[0]
        return predictions


    def lrelu(self, inputs, alpha=0.2):
        return tf.maximum(alpha * inputs, inputs)


    def apply_phaseshuffle(self, x, rad, pad_type='reflect'):
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
    def Discriminator(self, x, kernel_len=25, dim=64, use_batchnorm=False, phaseshuffle_rad=0):
        batch_size = self.batch_size
        # Avoid batchnorm
        #batch_size = 1
        with tf.variable_scope('discriminator') as scope:

            if use_batchnorm:
                batchnorm = lambda x: tf.layers.batch_normalization(x, training=True)
            else:
                batchnorm = lambda x: x

            if phaseshuffle_rad > 0:
                phaseshuffle = lambda x: self.apply_phaseshuffle(x, phaseshuffle_rad)
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
            output = self.lrelu(output)
            output = phaseshuffle(output)

            # Layer 1
            # [4096, 64] -> [1024, 128] - Old
            # [2048, 64] -> [512, 128] - New
            with tf.variable_scope('downconv_1'):
                filter = tf.get_variable('Fl', shape=[kernel_len, dim, dim * 2],
                                  initializer=tf.random_normal_initializer(stddev=1e-3))
                output = tf.nn.conv1d(output, filters=filter, stride=4, padding='SAME')
                output = batchnorm(output)
            output = self.lrelu(output)
            output = phaseshuffle(output)

            # Layer 2
            # [1024, 128] -> [256, 256] - Old
            # [512, 128] -> [128, 256] - New
            with tf.variable_scope('downconv_2'):
                filter = tf.get_variable('Fl', shape=[kernel_len, dim * 2, dim * 4],
                                  initializer=tf.random_normal_initializer(stddev=1e-3))
                output = tf.nn.conv1d(output, filters=filter, stride=4, padding='SAME')
                output = batchnorm(output)
            output = self.lrelu(output)
            output = phaseshuffle(output)

            # Layer 3
            # [256, 256] -> [64, 512] - Old
            # [128, 256] -> [32, 512] - New
            with tf.variable_scope('downconv_3'):
                filter = tf.get_variable('Fl', shape=[kernel_len, dim * 4, dim * 8],
                                  initializer=tf.random_normal_initializer(stddev=1e-3))
                output = tf.nn.conv1d(output, filters=filter, stride=4, padding='SAME')
                output = batchnorm(output)
            output = self.lrelu(output)
            output = phaseshuffle(output)

            # Layer 4
            # [64, 512] -> [16, 1024]
            # [32, 512] -> [8, 1024]
            with tf.variable_scope('downconv_4'):
                filter = tf.get_variable('Fl', shape=[kernel_len, dim * 8, dim * 16],
                                  initializer=tf.random_normal_initializer(stddev=1e-3))
                output = tf.nn.conv1d(output, filters=filter, stride=4, padding='SAME')
                output = batchnorm(output)
            output = self.lrelu(output)

            # Flatten
            # We now have to flatten it to a different size [8192]
            #output = tf.reshape(output, [batch_size, 4 * 4 * dim * 16])
            output = tf.reshape(output, [batch_size, 8192])

            # Connect to single logit
            with tf.variable_scope('output'):
                #W = tf.get_variable('W', shape=[4 * 4 * dim * 16, 1], initializer=tf.random_normal_initializer(stddev=1e-3))
                # New shape 8192
                #W = tf.get_variable('W', shape=[8192, 1], initializer=tf.random_normal_initializer(stddev=1e-3))
                #b = tf.zeros([batch_size, 1])
                #output = tf.matmul(output, W) + b
                #output = output[:, 0]
                output = tf.layers.dense(output, 1)[:, 0]

            #self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')

        # Return logit and a fake sigmoid 
        return output, tf.nn.sigmoid(output)


    def build_model(self):
        self.input_target = tf.placeholder(tf.float32, [self.batch_size, self.H_R, 1], name='input_target')
        #self.input_source = tf.placeholder(tf.float32, [self.batch_size, self.L_R, 1], name='input_source')

        #self.input_source = down_sample_layer(self.input_target)

        self.real = self.input_target
        # Define the generator and get the variables with generator scope
        #with tf.variable_scope('G-Gan'):
        self.fake = self.generator()
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        # Get the loss functions
        self.d_loss, self.g_loss, self.content_loss = self.inference_loss(self.real, self.fake)
        # Using SRGan model
        if self.config.gan == 'SRGan':
            self.d_optim = tf.train.AdamOptimizer(learning_rate=self.config.lr, beta1=self.config.beta1, beta2=self.config.beta2, name='AdamDiscrim').minimize(self.d_loss, var_list=self.d_vars)
            self.g_optim = tf.train.AdamOptimizer(learning_rate=self.config.lr, beta1=self.config.beta1, beta2=self.config.beta2, name='AdamGen').minimize(self.g_loss, var_list=self.g_vars)
        elif self.config.gan == 'WGan':
            doptim = tf.train.RMSPropOptimizer(learning_rate=self.config.lr, name='RMSDiscrim')
            self.g_optim = tf.train.RMSPropOptimizer(learning_rate=self.config.lr, name='RMSGen').minimize(self.g_loss, var_list=self.g_vars)
            #gv_d = doptim.compute_gradients(self.d_loss, var_list=self.d_vars)
            gv_d = tf.gradients(self.d_loss, self.d_vars)
            
            #clip_bounds = [-.01, .01]
            capped_grads, _ = (tf.clip_by_global_norm(gv_d, 1.0))
            self.d_optim = doptim.apply_gradients(zip(capped_grads, self.d_vars))

        # May not need this stuff
        # This is just for logs
        self.d_loss_summary = tf.summary.scalar('d_loss', self.d_loss)
        self.g_loss_summary = tf.summary.scalar('g_loss', self.g_loss)
        self.content_loss_summary = tf.summary.scalar('content_loss', self.content_loss)
        self.summaries = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('./AudioSRGANLogs', self.sess.graph) 
        self.saver = tf.train.Saver()
        print('building model...') 


    # Real represents the HR audio
    # Fake represents the generated output
    def inference_loss(self, real, fake):
        # MSE content loss
        def inference_mse_content_loss(real, fake):
            return tf.reduce_mean(tf.square(real-fake))
            
        def inference_adversarial_loss(x, y, w=1, type_='SRgan'):
            # Use binary cross entropy for now not wesserstein gans
            if type_=='SRgan':
                try:
                    return w*tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
                except:
                    return w*tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            elif self.config.gan == 'WGan': #Wessertein gan
                # Simply return the logits
                return x

        
        content_loss = inference_mse_content_loss(real, fake)
        with tf.name_scope('D_real'), tf.variable_scope('D'):
            d_real_logits, d_real_sigmoid = self.Discriminator(real)
        # Get variables
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')
        #print(self.d_vars)

        # Create the fake discriminator
        with tf.name_scope('D_fake'), tf.variable_scope('D', reuse=True):
            d_fake_logits, d_fake_sigmoid = self.Discriminator(fake)

        #d_real_logits = tf.Print(d_real_logits, [d_real_logits], message='d_real_logits')
        #d_fake_logits = tf.Print(d_fake_logits, [d_fake_logits], message='d_fake_logits')
        # I believe that the logits should be switched here but in the end it does not actually matter
        # i.e. d_fake uses fake logits and d_real uses real logits
        d_real_loss = tf.reduce_mean(inference_adversarial_loss(d_real_logits, tf.ones_like(d_real_sigmoid)))
        d_fake_loss = tf.reduce_mean(inference_adversarial_loss(d_fake_logits, tf.zeros_like(d_fake_sigmoid)))
        g_fake_loss = tf.reduce_mean(inference_adversarial_loss(d_fake_logits, tf.ones_like(d_fake_sigmoid)))
        
        
        #d_fake_loss = tf.Print(d_fake_loss, [d_fake_loss], message='d_fake_loss:')
        #d_real_loss = tf.Print(d_real_loss, [d_real_loss], message='d_real_loss')
        # self.config.lambd is a paramter that we can set as our configuraion but 
        # they define it as 0.001
        d_loss = 0
        if self.config.gan == 'SRGan':
            d_loss =  self.config.lambd*(d_fake_loss+d_real_loss)
        elif self.config.gan == 'WGan':
            # NOTE!!! THis appears flipped because I believe that the loss calculations are backwards
            d_loss = tf.reduce_mean(d_real_logits) -  tf.reduce_mean(d_fake_logits)

        # Set up the weight weight clipping if we are using wesserstein gans
        with tf.name_scope('D_clip_weights'):
            clip_ops = []
            for var in self.d_vars:
                clip_bounds = [-.01, .01]
                clip_ops.append(
                tf.assign(
                    var,
                    tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
                    )
                )
            self.D_clip_weights = tf.group(*clip_ops)

        g_loss = 0
        if self.config.gan == 'SRGan':
            g_loss = content_loss + self.config.lambd*g_fake_loss
        elif self.config.gan =='WGan':
            # May want a constant
            g_loss = content_loss - tf.reduce_mean(d_fake_logits)
        
        return d_loss, g_loss, content_loss

    def train(self):
        # Not totally sure what this is
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        
        # Get the data!!!
        # These data sets should line up
        path = '../data/vctk/speaker1/vctk-speaker1-train.4.16000.8192.4096.h5'
        data_HR, data_LR = load_h5(path)
        print data_HR.shape
        print data_LR.shape

        batch_idxs = int(len(data_HR)/self.batch_size)
        counter = 1

        # If we want to later load a model
        '''
        bool_check, counter = self.load_model(self.config.checkpoint_dir)
        if bool_check:
            print('[!!!] load model successfully')
            counter = counter + 1
        else:
            print('[***] fail to load model')
            counter = 1
        '''
        
        print('total steps:{}'.format(self.config.epoches*batch_idxs))

        start_time = time.time()
        for epoch in range(self.config.epoches):
            # Shuffle data
            permutation = np.random.permutation(data_HR.shape[0])
            data_HR = data_HR[permutation]
            data_LR = data_LR[permutation]
            for idx in range(batch_idxs):
                # I am assuming that this is to start one training example
                # that is a HR example
                batch_HR = data_HR[idx*self.batch_size:(idx+1)*self.batch_size, :, :]

                # I am assuming that this is to start one training example
                # that is a LR example
                batch_LR = data_LR[idx*self.batch_size:(idx+1)*self.batch_size, :, :]
                #print batch_LR.shape
                feed_dict = returnFeedDict(batch_LR, tf.get_collection('inputs'), self.fake, self.sess)
                feed_dict[self.input_target] = batch_HR
                #print(batch_HR)
                #print(batch_LR)
                #print(feed_dict)


                # Here we will train the discriminator more!
                # Train suggested 5 times discriminator per generator
                for i in xrange(self.config.d_updates):
                    if self.config.gan == 'SRGan':
                        #_, d_loss, summaries = self.sess.run([self.d_optim, self.d_loss, self.summaries], feed_dict={self.input_target:batch_HR, self.input_generator: batch_LR})
                        _, d_loss, summaries = self.sess.run([self.d_optim, self.d_loss, self.summaries], feed_dict=feed_dict)
                    elif self.config.gan == 'WGan':
                        #_, d_loss, summaries, _ = self.sess.run([self.d_optim, self.d_loss, self.summaries, self.D_clip_weights], feed_dict={self.input_target:batch_HR, self.input_generator: batch_LR})
                        self.sess.run(self.D_clip_weights)
                        _, d_loss, summaries = self.sess.run([self.d_optim, self.d_loss, self.summaries], feed_dict=feed_dict)

                    #print 'here'

                # Train the generator
                _, g_loss, summaries= self.sess.run([self.g_optim, self.g_loss, self.summaries], feed_dict=feed_dict)
                #g_loss, summaries= self.sess.run([self.g_loss, self.summaries], feed_dict=feed_dict)
                end_time = time.time()
                print('epoch{}[{}/{}]:total_time:{:.4f},d_loss:{:.12f},g_loss:{:12f}'.format(epoch, idx, batch_idxs, end_time-start_time, d_loss, g_loss))

                # We will definitely want this later --- Woddy knows best about saving checkpoints!
                self.summary_writer.add_summary(summaries, global_step=counter)
                self.summary_writer.flush()
                '''
                if np.mod(counter, 100)==0:
                    self.sample(epoch, idx)
                if np.mod(counter, 500)==0:
                    self.save_model(self.config.checkpoint_dir, counter)
                '''
                counter = counter+1
            if ((epoch / 1) == epoch):
                self.saver.save(self.sess, './GANWass', global_step = epoch + 1)