import os
import time

import numpy as np
import tensorflow as tf

import librosa
from keras import backend as K
#from dataset import DataSet


def load(ckpt):
    # get checkpoint name
    if os.path.isdir(ckpt): checkpoint = tf.train.latest_checkpoint(ckpt)
    else: checkpoint = ckpt
    meta = checkpoint + '.meta'
    print checkpoint

    #saver = tf.train.Saver()

    # load graph
    sess = tf.Session()
    saver = tf.train.import_meta_graph(meta)
    g = tf.get_default_graph()

    # load weights
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
    saver.restore(sess, checkpoint)

    # get graph tensors
    X, Y, alpha = tf.get_collection('inputs')

    # save tensors as instance variables
    inputs = X, Y, alpha
    predictions = tf.get_collection('preds')[0]
    print(predictions)

    w = tf.summary.FileWriter('graph_logs')

    w.add_graph(tf.get_default_graph())
    w.flush()
    w.close()
    #writer.close()
    sess.close()

    # load existing loss, or erase it, if creating new one
    '''g.clear_collection('losses')

    # create a new training op
    self.train_op = self.create_train_op(X, Y, alpha)
    g.clear_collection('train_op')
    tf.add_to_collection('train_op', self.train_op)'''

def main():
	load('./singlespeaker.lr0.000300.1.g4.b64/model.ckpt-53')
    #load('./model.ckpt-53')

if __name__ == '__main__':
  main()