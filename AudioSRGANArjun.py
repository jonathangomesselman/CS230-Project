import os
import time

import numpy as np
import tensorflow as tf
import keras
import librosa
from scipy import interpolate
from scipy.signal import decimate
import matplotlib.pyplot as plt
import h5py
from h5Converter import load_h5
from scipy import ndimage
#from keras import backend as K
# tensorflow backend
#from dataset import DataSet

#default params
r = 4 #sampling rate
layers = 4
sr = 16000
out_label = 'singlespeaker-out'

def eval_err(X, Y, inputs, sess, n_batch=128):
    batch_iterator = iterate_minibatches(X, Y, n_batch, shuffle=True)
    l2_loss_op, l2_snr_op = tf.get_collection('losses')

    l2_loss, snr = 0, 0
    tot_l2_loss, tot_snr = 0, 0
    for bn, batch in enumerate(batch_iterator):
        feed_dict = load_batch(batch, inputs, train=False)
        l2_loss, l2_snr = sess.run([l2_loss_op, l2_snr_op], feed_dict=feed_dict)
        tot_l2_loss += l2_loss
        tot_snr += l2_snr
        print('curr loss: ', tot_l2_loss)
        print('curr snr: ', tot_snr)
    return tot_l2_loss / (bn+1), tot_snr / (bn+1)

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
  assert len(inputs) == len(targets)
  if shuffle:
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
  for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
    if shuffle:
        excerpt = indices[start_idx:start_idx + batchsize]
    else:
        excerpt = slice(start_idx, start_idx + batchsize)
    yield inputs[excerpt], targets[excerpt]

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
    #assert len(k_tensors) <= 1
    if k_tensors: 
      k_learning_phase = g.get_tensor_by_name(k_tensors[1].name + ':0')
      feed_dict[k_learning_phase] = train

    return feed_dict

def spline_up(x_lr, r):
  print('x_lr', x_lr.shape)
  x_lr = x_lr.flatten()
  x_hr_len = len(x_lr) * r
  x_sp = np.zeros(x_hr_len)
  
  i_lr = np.arange(x_hr_len, step=r)
  i_hr = np.arange(x_hr_len)
  
  f = interpolate.splrep(i_lr, x_lr)

  x_sp = interpolate.splev(i_hr, f)
  print('x_sp', x_sp.shape)

  return x_sp

def predict(X, inputs, predictions, sess):
    assert len(X) == 1
    x_sp = spline_up(X, r)
    x_sp = x_sp[:len(x_sp) - (len(x_sp) % (2**(layers+1)))]
    X = x_sp.reshape((1,len(x_sp),1))
    feed_dict = load_batch((X,X), inputs, train=False)
    return sess.run(predictions, feed_dict=feed_dict)

def upsample_wav(wav, inputs, predictions, sess):

  #changes to path to make code work
  initpath = wav
  wav = "p225/" + wav
  #wav = 'TestGeneratorInput/' + wav
  print(wav)

  # load signal
  x_hr, fs = librosa.load(wav, sr)

  #trainlr = 'TestGeneratorInput/' + '3test.singlespeakertest.lr.wav'
  #x_lr, fs = librosa.load(trainlr, sr)
  '''print('fs', fs)
  print(x_hr)
  print(x_hr.shape)'''

  # downscale signal
  #x_lr = np.array(x_hr[0::args.r])

  # For now, instead of decimating, take from X_train matrix
  x_lr = decimate(x_hr, r)
  '''print(x_lr)
  print(x_lr.shape)'''
  # x_lr = decimate(x_hr, args.r, ftype='fir', zero_phase=True)
  # x_lr = downsample_bt(x_hr, args.r)
  x_sp = spline_up(x_lr, r)
  x_sp = x_sp[:len(x_sp) - (len(x_sp) % (2**(layers+1)))]

  # upscale the low-res version
  P = predict(x_lr.reshape((1,len(x_lr),1)), inputs, predictions, sess)
  x_pr = P.flatten()

  # crop so that it works with scaling ratio
  x_hr = x_hr[:len(x_pr)]
  x_lr = x_lr[:len(x_pr)]
  lsd = compute_log_distortion(x_hr, x_pr)
  print(lsd)
  segsnr = compute_segsnr(x_hr, x_pr)
  print(segsnr)
  '''print(x_pr.shape)
  print(x_hr.shape)
  print('cropped x_lr: ', x_lr)'''
  print(x_lr.shape)

  # save the file
  #outname = wav + '.' + out_label
  outname = "TestGeneratorInput/" + initpath + '.' + out_label
  #utname = "TestGeneratorOutput/" + initpath + '.' + out_label
  librosa.output.write_wav(outname + '.hr.wav', x_hr, fs)  
  librosa.output.write_wav(outname + '.sp.wav', x_sp, fs) 
  #librosa.output.write_wav(outname + '.lr.wav', x_lr, fs) 
  librosa.output.write_wav(outname + '.lr.wav', x_lr, fs / r)  
  librosa.output.write_wav(outname + '.pr.wav', x_pr, fs)  
  #librosa.output.write_wav(outname + '.pr.wav', x_pr, fs * r)  

  # save the spectrum
  S = get_spectrum(x_pr, n_fft=2048)
  save_spectrum(S, outfile=outname + '.pr.png')
  S = get_spectrum(x_hr, n_fft=2048)
  save_spectrum(S, outfile=outname + '.hr.png')
  S = get_spectrum(x_lr, n_fft=2048/r)
  save_spectrum(S, outfile=outname + '.lr.png')
  S = get_spectrum(x_sp, n_fft=2048)
  save_spectrum(S, outfile=outname + '.sp.png')

def load(ckpt):
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
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        # get graph tensors
        X, Y, alpha = tf.get_collection('inputs')
        # save tensors as instance variables
        inputs = X, Y, alpha
        predictions = tf.get_collection('preds')[0]
        print(predictions)
        #return inputs, predictions
        #upsample_wav('p225_002.wav', inputs, predictions, sess)
        upsample_wav('p225_002.wav', inputs, predictions, sess)
        #upsample_wav('3test.singlespeakertest.hr.wav', inputs, predictions, sess)
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

def get_spectrum(x, n_fft=2048):
  S = librosa.stft(x, n_fft)
  p = np.angle(S)
  S = np.log1p(np.abs(S))
  print(S.shape)
  return S

def save_spectrum(S, lim=800, outfile='spectrogram.png'):
  plt.imshow(S.T, aspect=10)
  #plt.xlim([0,lim])
  plt.xlim(0, lim)
  plt.xlabel('Frequency')
  plt.ylabel('Frame')
  #plt.title('ASRWGAN Reconstruction')
  plt.title('High-Res Audio Signal')
  plt.tight_layout()
  plt.savefig(outfile)  

def compute_segsnr(x_hr, x_pr):
  n_windows = len(x_hr) // WIN_SIZE
  x_hr_wins = [x_hr[i*WIN_SIZE:(i+1)*WIN_SIZE] for i in range(n_windows)]
  x_pr_wins = [x_pr[i*WIN_SIZE:(i+1)*WIN_SIZE] for i in range(n_windows)]

  x_wins = [ (x_hr_win, x_pr_win) for x_hr_win, x_pr_win in zip(x_hr_wins, x_pr_wins) ]

  psnrs = [compute_psnr(x_hr_win, x_pr_win) for (x_hr_win, x_pr_win) in x_wins]

  # for psnr, (x_hr_win, x_pr_win) in zip(psnrs, x_wins):
  #   max_hr = np.sqrt(np.mean(x_hr_win**2))
  #   loss = np.mean((x_pr_win - x_hr_win)**2)
  #   psnr = 20 * np.log10(max_hr / np.sqrt(loss) + 1e-8)
  #   print psnr, loss, np.mean((x_hr_win - x_pr_win)**2), max_hr, max_hr / np.sqrt(loss), np.mean(np.abs(x_hr_win)), np.mean(np.abs(x_pr_win))
  # # print [(psnr, np.mean(np.abs(x_hr_win))) for psnr, (x_hr_win, _) in zip(psnrs, x_wins)]
  return np.mean(np.array(psnrs))

def compute_psnr(x_hr, x_pr):
  # max_hr = np.max(np.abs(x_hr))
  max_hr = np.sqrt(np.mean(x_hr**2))
  # max_pr = np.max(np.abs(x_pr))
  # x_pr = x_pr / max_pr * max_hr
  loss = np.mean((x_pr - x_hr)**2)
  psnr = 20 * np.log10(max_hr / np.sqrt(loss) + 1e-8)
  # if len(x_hr) == 8000:
  #   print np.sqrt(loss), max_hr, np.max(x_pr)
  #   print ' '.join(['%.3f' % f for f in x_hr[5000:5050]])
  #   print ' '.join(['%.3f' % f for f in x_pr[5000:5050]])
  #   # print ' '.join(['%.3f' % f for f in x_hr[40000:40050]])
  #   # print ' '.join(['%.3f' % f for f in x_pr[40000:40050]])

  return psnr

# Only use to test format of input data
'''def load_h5(h5_path):
  # load training data
  with h5py.File(h5_path, 'r') as hf:
    print 'List of arrays in input file:', hf.keys()
    X = np.array(hf.get('data'))
    Y = np.array(hf.get('label'))
    print 'Shape of X:', X.shape
    print 'Shape of Y:', Y.shape

  return X, Y'''
def test(ckpt): 
    # get checkpoint name
    if os.path.isdir(ckpt): checkpoint = tf.train.latest_checkpoint(ckpt)
    else: checkpoint = ckpt
    meta = checkpoint + '.meta'
    print checkpoint
    #saver = tf.train.Saver()

    # load graph
    saver = tf.train.import_meta_graph(meta)
    g = tf.get_default_graph()

    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        # get graph tensors
        X, Y, alpha = tf.get_collection('inputs')
        # save tensors as instance variables
        inputs = X, Y, alpha
        predictions = tf.get_collection('preds')[0]
        print(predictions)
        #path = '../data/vctk/speaker1/vctk-speaker1-train.4.16000.8192.4096.h5'
        path = '../data/vctk/speaker1/vctk-speaker1-val.4.16000.8192.4096.h5'
        data_HR, data_LR = load_h5(path)
        # Use the below line when using load_h5 from io.py
        #data_LR, data_HR = load_h5(path)
        #print(data_LR[0])
        #print(data_LR[1])
        avgl2loss, avgl2snr = eval_err(data_LR, data_HR, inputs, sess, n_batch=32)
        print('avgl2loss: ', avgl2loss, ' avgl2snr: ', avgl2snr)

N_FFT = 2048
WIN_SIZE=2048

def get_power(x, n_fft=N_FFT):    
  S = librosa.stft(x, n_fft)
  p = np.angle(S)
  
  S = np.log(np.abs(S)**2 + 1e-8)  
  return S  

def compute_log_distortion(x_hr, x_pr):
  S1 = get_power(x_hr) # (n_frames, n_freq)
  S2 = get_power(x_pr)

  lsd = np.mean(np.sqrt(np.mean((S1 - S2)**2 + 1e-8, axis=1)), axis=0)
  return min(lsd, 10.)

def main():
    #load('./AWSFinalGenWeights/model.ckpt-6241')
    #load('./WassGAN17EPoch/GANWass-17')
    load('./GANWass-47')
    #load('./newWassGAN9Epoch/GANWass-9')
    #inputs, predictions = load('./singlespeaker.lr0.000300.1.g4.b64/model.ckpt-53')
    #upsample_wav('p225/p225_001.wav', inputs, predictions)

    #test('./newWassGAN9Epoch/GANWass-9')
    #test('./AWSFinalGenWeights/model.ckpt-6241')

if __name__ == '__main__':
  main()