
import tensorflow as tf
import numpy as np
from tensorpack import (TowerTrainer,
                        ModelDescBase, DataFlow, StagingInput)
from tensorpack.tfutils.tower import TowerContext, TowerFuncWrapper
from tensorpack.graph_builder import DataParallelBuilder, LeastLoadedDeviceSetter
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils.argtools import memoized


class WGANTrainer(TowerTrainer):
    """ A GAN trainer which runs two optimization ops with a certain ratio."""
    def __init__(self, input, model, f_period=5, g_period=1):
        """
        Args:
            d_period(int): period of each d_opt run
            g_period(int): period of each g_opt run
        """
        super(WGANTrainer, self).__init__()
        self._f_period = int(f_period)
        self._g_period = int(g_period)
        assert min(f_period, g_period) == 1

        # Setup input
        cbs = input.setup(model.get_inputs_desc())
        self.register_callback(cbs)

        # Build the graph
        self.tower_func = TowerFuncWrapper(model.build_graph, model.get_inputs_desc())
        with TowerContext('', is_training=True):
            self.tower_func(*input.get_input_tensors())

        opt = model.get_optimizer()
        with tf.name_scope('optimize'):
            self.f_min = opt.minimize(
                model.f_loss, var_list=model.f_vars, name='f_min')
            self.g_min = opt.minimize(
                model.g_loss, var_list=model.g_vars, name='g_min')

    def run_step(self):
        # Define the training iteration
        if self.global_step % (self._f_period) == 0:
            self.hooked_sess.run(self.f_min)
        if self.global_step % (self._g_period) == 0:
            self.hooked_sess.run(self.g_min)