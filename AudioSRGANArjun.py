import os
import time

import numpy as np
import tensorflow as tf
import keras
import librosa
from scipy import interpolate
from scipy.signal import decimate
from GAN_structure import Discriminator
#from keras import backend as K
# tensorflow backend
#from dataset import DataSet

#default params
r = 4 #sampling rate
layers = 4
sr = 16000
out_label = 'singlespeaker-out'

def load_batch(batch, inputs, alpha=1, train=True,):
    X_in, Y_in, alpha_in = inputs
    X, Y = batch
    
    if Y is not None:
      feed_dict = {X_in : X, Y_in : Y, alpha_in : alpha}
    else:
      feed_dict = {X_in : X, alpha_in : alpha}

    # this is ugly, but only way I found to get this var after model reload
    g = tf.get_default_graph()
    k_tensors = [n for n in g.as_graph_def().node if 'keras_learning_phase' in n.name]
    assert len(k_tensors) <= 1
    if k_tensors: 
      k_learning_phase = g.get_tensor_by_name(k_tensors[0].name + ':0')
      feed_dict[k_learning_phase] = train

    return feed_dict

def spline_up(x_lr, r):
  x_lr = x_lr.flatten()
  x_hr_len = len(x_lr) * r
  x_sp = np.zeros(x_hr_len)
  
  i_lr = np.arange(x_hr_len, step=r)
  i_hr = np.arange(x_hr_len)
  
  f = interpolate.splrep(i_lr, x_lr)

  x_sp = interpolate.splev(i_hr, f)

  return x_sp

def predict(X, inputs, predictions, sess):
    assert len(X) == 1
    x_sp = spline_up(X, r)
    x_sp = x_sp[:len(x_sp) - (len(x_sp) % (2**(layers+1)))]
    X = x_sp.reshape((1,len(x_sp),1))
    feed_dict = load_batch((X,X), inputs, train=False)
    return sess.run(predictions, feed_dict=feed_dict)

def upsample_wav(wav, inputs, predictions, sess):
  # load signal
  x_hr, fs = librosa.load(wav, sr)

  # downscale signal
  # x_lr = np.array(x_hr[0::args.r])
  x_lr = decimate(x_hr, r)
  # x_lr = decimate(x_hr, args.r, ftype='fir', zero_phase=True)
  # x_lr = downsample_bt(x_hr, args.r)

  # upscale the low-res version
  P = predict(x_lr.reshape((1,len(x_lr),1)), inputs, predictions, sess)
  x_pr = P.flatten()

  # crop so that it works with scaling ratio
  x_hr = x_hr[:len(x_pr)]
  x_lr = x_lr[:len(x_pr)]

  # save the file
  outname = wav + '.' + out_label
  librosa.output.write_wav(outname + '.hr.wav', x_hr, fs)  
  librosa.output.write_wav(outname + '.lr.wav', x_lr, fs / r)  
  librosa.output.write_wav(outname + '.pr.wav', x_pr, fs)  

  # save the spectrum
  '''S = get_spectrum(x_pr, n_fft=2048)
  save_spectrum(S, outfile=outname + '.pr.png')
  S = get_spectrum(x_hr, n_fft=2048)
  save_spectrum(S, outfile=outname + '.hr.png')
  S = get_spectrum(x_lr, n_fft=2048/args.r)
  save_spectrum(S, outfile=outname + '.lr.png')'''

def load(ckpt, sess):
    # get checkpoint name
    if os.path.isdir(ckpt): checkpoint = tf.train.latest_checkpoint(ckpt)
    else: checkpoint = ckpt
    meta = checkpoint + '.meta'
    print checkpoint
    #saver = tf.train.Saver()

    # load graph
    saver = tf.train.import_meta_graph(meta)
    g = tf.get_default_graph()

    # with tf.name_scope('hidden') as scope:
    #     a = tf.constant(5, name='alpha')
    #     W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name='weights')
    #     b = tf.Variable(tf.zeros([1]), name='biases')

    # load weights
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
    #sess = tf.Session()
    #with tf.Session() as sess:
    saver.restore(sess, checkpoint)
    # get graph tensors
    X, Y, alpha = tf.get_collection('inputs')
    # save tensors as instance variables
    inputs = X, Y, alpha
    predictions = tf.get_collection('preds')[0]
    return predictions
    #print(predictions)
    #return inputs, predictions
    #upsample_wav('p225/p225_001.wav', inputs, predictions, sess)
    #print(predictions.eval(session=sess))
    #keras.callbacks.TensorBoard(log_dir='./keras_logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0)
    #tf.summary.FileWriter('./singlespeaker.lr0.000300.1.g4.b64', graph=sess.graph)
    #tf.train.SummaryWriter('./graph_logs', graph=sess.graph)


    # load existing loss, or erase it, if creating new one
    '''g.clear_collection('losses')

    # create a new training op
    self.train_op = self.create_train_op(X, Y, alpha)
    g.clear_collection('train_op')
    tf.add_to_collection('train_op', self.train_op)'''

def buildGanTrainOps(sess):
    # Note one thing that we need is the input data!!

    # Get the generator
    # We may have to do it is with variable scoe generator?
    with tf.variable_scope('G'):
        generator = load('./singlespeaker.lr0.000300.1.g4.b64/model.ckpt-53', sess)
    # This should work
    G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    #G_vars = tf.get_collection('preds')

    # We may want to print the summary of the graph later

    # Create a real and fake discriminator

    # Real discriminator takes as input the real high quality audio
    # This is where we would upload our actual data
    X_Disc = tf.placeholder(tf.float32, shape=(None, 8192, 1), name='X')
    with tf.name_scope('D_real'), tf.variable_scope('D'):
        D_r = Discriminator(X_Disc)
    # Get the variable scops
    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')

    # Create the fake discriminator
    with tf.name_scope('D_fake'), tf.variable_scope('D', reuse=True):
        D_f = Discriminator(generator)


    # Create the loss functions
    G_loss = -tf.reduce_mean(generator)
    D_loss = tf.reduce_mean(D_f) - tf.reduce_mean(D_r)

    # Enforce Lipschitz constraint for WGAN
    with tf.name_scope('D_clip_weights'):
        clip_ops = []
        for var in D_vars:
            clip_bounds = [-.01, .01]
            clip_ops.append(
                tf.assign(
                    var,
                    tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
                )
            )
        D_clip_weights = tf.group(*clip_ops)

    # Creat the optimizer
    # Just using what wgna
    G_opt = tf.train.RMSPropOptimizer(
        learning_rate=5e-5)
    D_opt = tf.train.RMSPropOptimizer(
        learning_rate=5e-5)

    # Create training ops
    #G_train_op = G_opt.minimize(G_loss, var_list=G_vars,
    #    global_step=tf.train.get_or_create_global_step())
    # Removed globel_step
    #print G_vars
    G_train_op = G_opt.minimize(G_loss, var_list=G_vars)
    D_train_op = D_opt.minimize(D_loss, var_list=D_vars)

    #return G_train_op, D_train_op


#def train(G_train_op, D_train_op):


def main():

    sess = tf.Session()

    #G_train_op, D_train_op = buildGanTrainOps(sess)
    buildGanTrainOps(sess)
    w = tf.summary.FileWriter('graph_logs')

    w.add_graph(tf.get_default_graph())
    w.flush()
    w.close()
    sess.close()
    #load('./singlespeaker.lr0.000300.1.g4.b64/model.ckpt-53')
    #inputs, predictions = load('./singlespeaker.lr0.000300.1.g4.b64/model.ckpt-53')
    #upsample_wav('p225/p225_001.wav', inputs, predictions)


if __name__ == '__main__':
  main()