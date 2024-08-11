# coding utf-8
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops

from tensorflow.python.ops.rnn_cell_impl import RNNCell


class GRUCell(RNNCell):
    def __init__(self,
                 num_units,
                 Wx_initializer,
                 Wr_initializer,
                 Wz_initializer,
                 Wh_initializer,
                 br_initializer=None,
                 bz_initializer=None,
                 reuse=None,
                 name=None):
        super(GRUCell, self).__init__(_reuse=reuse, name=name)

        self._num_units = num_units
        self._Wx_initializer = Wx_initializer
        self._Wr_initializer = Wr_initializer
        self._br_initializer = br_initializer
        self._Wz_initializer = Wz_initializer
        self._bz_initializer = bz_initializer
        self._Wh_initializer = Wh_initializer

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
        self.Wx = self.add_variable('Wx', shape=[input_depth, self._num_units], initializer=self._Wx_initializer)

        self.Wr = self.add_variable('Wr', shape=[input_depth, self._num_units], initializer=self._Wr_initializer)
        if self._br_initializer is None:
            self._br_initializer = init_ops.truncated_normal_initializer(mean=1)
        self.br = self.add_variable('br', shape=[self._num_units], initializer=self._br_initializer)

        self.Wz = self.add_variable('Wz', shape=[input_depth, self._num_units], initializer=self._Wz_initializer)
        if self._bz_initializer is None:
            self._bz_initializer = init_ops.truncated_normal_initializer(mean=1)
        self.bz = self.add_variable('bz', shape=[self._num_units], initializer=self._bz_initializer)

        self.Wh = self.add_variable('Wh', shape=[self._num_units, self._num_units], initializer=self._Wh_initializer)

        self.built = True

    def call(self, inputs, state):
        r = nn_ops.bias_add(math_ops.matmul(inputs, self.Wr), self.br)
        r = math_ops.sigmoid(r)

        z = nn_ops.bias_add(math_ops.matmul(inputs, self.Wz), self.bz)
        z = math_ops.sigmoid(z)

        h_ = math_ops.add(
            math_ops.matmul(inputs, self.Wx),
            math_ops.matmul(state, self.Wh) * r
        )
        h_ = math_ops.tanh(h_)

        h_new = math_ops.add(math_ops.multiply((1-z), h_), math_ops.multiply(state, z))
        return h_new, h_new
