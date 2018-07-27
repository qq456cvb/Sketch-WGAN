import tensorflow as tf
import glob
import os
from data import load_dataset
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack import *
from tensorpack.tfutils.gradproc import MapGradient, SummaryGradient
from tensorpack.tfutils import (
    get_current_tower_context, optimizer)
from tensorpack.utils.gpu import get_nr_gpu
import tensorflow.contrib.slim as slim
import tensorflow.contrib.rnn as rnn
import multiprocessing
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from wgan import WGANTrainer

MAX_SEQ_LEN = 250
BATCH_SIZE = 100
LSTM_F_SIZE = 256
LSTM_G_SIZE = 512
Z_DIM = 128
GMM_COMPONENTS = 20


class MyDataflow(RNGDataFlow):
    def __init__(self, dataset):
        self.data_set = dataset

    def get_data(self):
        b = self.data_set.random_batch(self.rng)
        yield [b[2], b[1]]


class WGAN(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.int32, [None], 'seq_len'),
                tf.placeholder(tf.float32, [None, MAX_SEQ_LEN + 1, 5], 'strokes5')]

    # B * MAX_LEN * 5
    @auto_reuse_variable_scope
    def f(self, seq_len, strokes):
        with tf.variable_scope('f'):
            cell_fw = rnn.BasicLSTMCell(LSTM_F_SIZE)
            cell_bw = rnn.BasicLSTMCell(LSTM_F_SIZE)
            init_state_fw = cell_fw.zero_state(BATCH_SIZE, tf.float32)
            init_state_bw = cell_bw.zero_state(BATCH_SIZE, tf.float32)
            _, (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                                 strokes, seq_len, init_state_fw,
                                                                                 init_state_bw)
            # print(output_state_fw[1].shape)

            x = tf.concat([output_state_fw[1], output_state_bw[1]], axis=1)
            units = [128, 64, 32, 16]
            for i in range(4):
                x = slim.fully_connected(x, units[i], activation_fn=tf.nn.relu)
            out = tf.squeeze(slim.fully_connected(x, 1, activation_fn=None), axis=1)
            return out

    @auto_reuse_variable_scope
    def g(self, z, strokes):
        with tf.variable_scope('g'):
            cell = rnn.BasicLSTMCell(LSTM_G_SIZE)
            init_state = rnn.LSTMStateTuple(tf.placeholder_with_default(tf.zeros([BATCH_SIZE, LSTM_G_SIZE], name='cz'),
                                                     shape=[None, LSTM_G_SIZE], name='c'),
                                            tf.placeholder_with_default(
                                                tf.zeros([BATCH_SIZE, LSTM_G_SIZE], name='hz'),
                                                shape=[None, LSTM_G_SIZE], name='h'))
            seq_input = tf.concat([tf.tile(tf.expand_dims(z, 1), [1, MAX_SEQ_LEN, 1]), strokes], 2)
            seq_output, _ = tf.nn.static_rnn(cell, tf.unstack(seq_input, axis=1), init_state)
            seq_output = tf.stack(seq_output, axis=1)
            return seq_output

    def sample_GMM(self, rnn_output):
        shape = rnn_output.shape
        rnn_output_reshaped = tf.reshape(rnn_output, [-1, shape[-1]])
        x = slim.fully_connected(rnn_output_reshaped, GMM_COMPONENTS * 6 + 3, activation_fn=None)
        q = tf.nn.softmax(x[:, -3:])
        pi, ux, uy, sigmax, sigmay, rhoxy = tf.split(tf.reshape(x[:, :-3], [-1, GMM_COMPONENTS, 6]), 6, axis=2)
        pi = tf.nn.softmax(tf.squeeze(pi, 2))
        ux = tf.squeeze(ux, 2)
        uy = tf.squeeze(uy, 2)
        sigmax = tf.exp(tf.squeeze(sigmax, 2))
        sigmay = tf.exp(tf.squeeze(sigmay, 2))
        rhoxy = tf.nn.tanh(tf.squeeze(rhoxy, 2))
        sigmaxy = sigmax * sigmay

        # reparameterization trick
        z = tf.random_normal([x.shape.as_list()[0], GMM_COMPONENTS, 2, 1])
        tau = sigmax * sigmax + sigmay * sigmay
        s = sigmaxy * tf.sqrt(1 - rhoxy * rhoxy)
        t = tf.sqrt(tau + 2 * s)
        sigma = tf.reshape(tf.expand_dims(1 / t, -1) * tf.stack([sigmax * sigmax + s, rhoxy * sigmaxy, rhoxy * sigmaxy, sigmay * sigmay + s], -1), [t.shape.as_list()[0], t.shape.as_list()[1], 2, 2])
        z_sample = tf.squeeze(sigma @ z, -1) + tf.stack([ux, uy], -1)
        xy_gmm = tf.reduce_sum(tf.expand_dims(pi, -1) * z_sample, 1)

        return tf.reshape(tf.concat([xy_gmm, q], -1), [shape[0], shape[1], 5])

    def build_graph(self, seq_len, strokes):
        z_f = tf.random_normal([BATCH_SIZE, Z_DIM])
        diff = self.f(seq_len, strokes[:, 1:]) - self.f(seq_len, self.sample_GMM(self.g(z_f, strokes[:, :-1])))
        self.f_loss = -tf.reduce_mean(diff, name='f_loss')

        z = tf.placeholder_with_default(tf.random_normal([BATCH_SIZE, Z_DIM]), [BATCH_SIZE, Z_DIM], name='z')
        self.g_loss = -tf.reduce_mean(self.f(seq_len, self.sample_GMM(self.g(z, strokes[:, :-1]))), name='g_loss')

        add_moving_summary(self.f_loss, self.g_loss)

        self.f_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'f')
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'g')

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-3, trainable=False)
        opt = tf.train.AdamOptimizer(lr)
        gradprocs = [MapGradient(lambda grad: (tf.clip_by_average_norm(grad, 0.1) if grad is not None else grad))]
        # SummaryGradient()]
        opt = optimizer.apply_grad_processors(opt, gradprocs)
        return opt


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    data_folder = '.\\datasets'
    npz = glob.glob(os.path.join(data_folder, 'airplane.npz'))
    train_set, valid_set, test_set = load_dataset('.', npz)
    df = MyDataflow(train_set)
    # if os.name == 'nt':
    #     dataflow = PrefetchData(df, nr_proc=multiprocessing.cpu_count() // 2,
    #                             nr_prefetch=multiprocessing.cpu_count() // 2)
    # else:
    #     dataflow = PrefetchDataZMQ(df, nr_proc=multiprocessing.cpu_count() // 2)

    trainer = WGANTrainer(QueueInput(df), WGAN())
    trainer.train_with_defaults(
        callbacks=[
            #         ModelSaver(),
            EstimatedTimeLeft(),
            # ScheduledHyperParamSetter('learning_rate', [(20, 0.0003), (120, 0.0001)]),
            # ScheduledHyperParamSetter('entropy_beta', [(80, 0.005)]),
            # HumanHyperParamSetter('learning_rate'),
        ],
        steps_per_epoch=100,
        max_epoch=100)