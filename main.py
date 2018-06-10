import tensorflow as tf
import os
from AudioSRGanModel import *

flags = tf.app.flags
#flags.DEFINE_float('lr', 0.0001, 'learning rate')
flags.DEFINE_float('lr', 5e-5, 'learning rate')
flags.DEFINE_float('beta1', 0.9, 'beta1')
flags.DEFINE_float('beta2', 0.999, 'beta2')
#flags.DEFINE_float('lambd', 0.001, 'coeff for adversarial loss')
flags.DEFINE_float('lambd', 0.1, 'coeff for adversarial loss')
flags.DEFINE_string('logs_dir', 'logs', 'log directory')
flags.DEFINE_integer('epoches', 200, 'training epoches')
#flags.DEFINE_string('gan', 'SRGan', 'Defines the architecture we based our model off of')
flags.DEFINE_string('gan', 'WGan', 'Defines the architecture we based our model off of')
flags.DEFINE_integer('d_updates', 5, 'Number of updates of discriminator vs. generator')
# We have to change these!
flags.DEFINE_string('train_set', 'ImageNet', 'train phase')
flags.DEFINE_string('val_set', 'Set5', 'val phase')
flags.DEFINE_string('test_set', 'Set14', 'test phase')
flags.DEFINE_bool('is_testing', False, 'training or testing')
flags.DEFINE_bool('is_training', True, 'training or testing')
FLAGS = flags.FLAGS

checkpointed_generator = './AWSFinalGenWeights/model.ckpt-6241'

def main():
    # Not sure about these options for now
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

    with tf.Session(config=config) as sess:
    #with tf.Session() as sess:
        audiosrgan = AudioSRGanModel(FLAGS, checkpointed_generator, batch_size=64, H_R=8192, L_R = 8192, sess=sess)
        audiosrgan.build_model()
        if FLAGS.is_training:
            audiosrgan.train()
        if FLAGS.is_testing:
            raise NotImplementedError()
            audiosrgan.test()

if __name__=='__main__':
    # ???
    #with tf.device('/gpu:0'):
        #tf.app.run()

    main()