# coding utf-8

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops

from tensorflow.python.ops.rnn_cell_impl import RNNCell


# https://github.com/gyyang/multitask/blob/master/network.py
class LeakyRNNCell(RNNCell):
    def __init__(self,
                 num_units,
                 alpha,
                 recurrent_kernel_initializer,
                 input_kernel_initializer,
                 recurrent_bias_initializer=None,
                 sigma_rec=0,
                 activation='softplus',
                 reuse=None,
                 name=None):
        super(LeakyRNNCell, self).__init__(_reuse=reuse, name=name)

        self._num_units = num_units
        self._alpha = alpha
        self._recurrent_initializer = recurrent_kernel_initializer
        self._input_initializer = input_kernel_initializer
        self._bias_initializer = recurrent_bias_initializer
        self._sigma = np.sqrt(2 / alpha) * sigma_rec

        if activation == 'softplus':
            self._activation = tf.nn.softplus
        elif activation == 'tanh':
            self._activation = tf.tanh
        elif activation == 'relu':
            self._activation = tf.nn.relu
        elif activation == 'power':
            self._activation = lambda x: tf.square(tf.nn.relu(x))
        elif activation == 'retanh':
            self._activation = lambda x: tf.tanh(tf.nn.relu(x))
        else:
            raise ValueError('Unknown activation')

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value
        self._input_kernel = self.add_variable(
            "input_kernel",
            shape=[input_depth, self._num_units],
            initializer=self._input_initializer
        )

        self._recurrent_kernel = self.add_variable(
            "recurrent_kernel",
            shape=[self._num_units, self._num_units],
            initializer=self._recurrent_initializer
        )

        if self._bias_initializer is None:
            self._bias_initializer = init_ops.zeros_initializer()
        self._bias = self.add_variable(
            "recurrent_bias",
            shape=[self._num_units],
            initializer=self._bias_initializer
        )

        self.built = True

    def call(self, inputs, state):
        """Most basic RNN: output = new_state = act(W * input + U * state + B)."""

        gate_inputs = math_ops.matmul(inputs, self._input_kernel)
        recurrent_update = math_ops.matmul(state, self._recurrent_kernel)
        gate_inputs = math_ops.add(gate_inputs, recurrent_update)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        noise = tf.random_normal(tf.shape(state), mean=0, stddev=self._sigma)
        gate_inputs = gate_inputs + noise

        output = self._activation(gate_inputs)

        output = (1 - self._alpha) * state + self._alpha * output

        return output, output
