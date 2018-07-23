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

MAX_SEQ_LEN = 250
BATCH_SIZE = 100
LSTM_F_SIZE = 256
LSTM_G_SIZE = 512
Z_DIM = 128


class MyDataflow(RNGDataFlow):
    def __init__(self, dataset):
        self.data_set = dataset

    def get_data(self):
        yield self.data_set.random_batch(self.rng)[:2]


class WGAN(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.int32, [None], 'seq_len'),
                tf.placeholder(tf.float32, [None, MAX_SEQ_LEN, 5], 'strokes5')]

    # B * MAX_LEN * 5
    @auto_reuse_variable_scope
    def f(self, seq_len, strokes):
        with tf.variable_scope('f'):
            cell_fw = rnn.BasicLSTMCell(LSTM_F_SIZE)
            cell_bw = rnn.BasicLSTMCell(LSTM_F_SIZE)
            init_state_fw = cell_fw.zero_state(BATCH_SIZE, tf.float32)
            init_state_bw = cell_bw.zero_state(BATCH_SIZE, tf.float32)
            _, output_state_fw, output_state_bw = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                                 tf.unstack(strokes, axis=1), seq_len, init_state_fw,
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
            seq_output = tf.nn.static_rnn(cell, seq_input, init_state)
            return seq_output

    def build_graph(self, seq_len, strokes):
        z_f = tf.random_normal([BATCH_SIZE, Z_DIM])
        diff = self.f(seq_len, strokes) - self.f(seq_len, self.g(z_f, strokes))
        self.f_loss = -tf.reduce_mean(diff, name='f_loss')

        z = tf.placeholder_with_default(tf.random_normal([BATCH_SIZE, Z_DIM]), [BATCH_SIZE, Z_DIM], name='z')
        self.g_loss = -tf.reduce_mean(self.f(seq_len, self.g(z, strokes)), name='g_loss')

        add_moving_summary(self.f_loss, self.g_loss)

        self.f_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'f')
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'g')

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-3, trainable=False)
        opt = tf.train.AdamOptimizer(lr)
        gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.1))]
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

    config = TrainConfig(
        model=WGAN(),
        dataflow=df,
        callbacks=[
            #         ModelSaver(),
            EstimatedTimeLeft(),
            # ScheduledHyperParamSetter('learning_rate', [(20, 0.0003), (120, 0.0001)]),
            # ScheduledHyperParamSetter('entropy_beta', [(80, 0.005)]),
            # HumanHyperParamSetter('learning_rate'),
        ],
        #     session_init=get_model_loader('train_log/auto_encoder/checkpoint'),
        steps_per_epoch=100,
        max_epoch=100,
    )
    trainer = SimpleTrainer()
    launch_train_with_config(config, trainer)