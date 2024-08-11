# coding utf-8

import os
import numpy as np

import tensorflow as tf
from tensorflow.python.ops import rnn
from .RNNCell import LeakyRNNCell

from utils import tools


def tf_popvec(y):
    """Population vector read-out in tensorflow."""

    num_units = y.get_shape().as_list()[-1]
    pref = np.arange(0, 2 * np.pi, 2 * np.pi / num_units)  # preferences
    cos_pref = np.cos(pref)
    sin_pref = np.sin(pref)
    temp_sum = tf.reduce_sum(y, axis=-1)
    temp_cos = tf.reduce_sum(y * cos_pref, axis=-1) / temp_sum
    temp_sin = tf.reduce_sum(y * sin_pref, axis=-1) / temp_sum
    loc = tf.atan2(temp_sin, temp_cos)
    return tf.mod(loc, 2*np.pi)


class Model:
    def __init__(self,
                 model_dir,
                 input_kernel_initializer,
                 recurrent_kernel_initializer,
                 recurrent_bias_initializer=None,
                 output_kernel_initializer=None,
                 output_bias_initializer=None,
                 hp=None
                 ):

        # Reset Tensorflow graphs
        tf.reset_default_graph()  # must be in the beginning

        if hp is None:
            hp = tools.load_hp(model_dir)
            if hp is None:
                raise ValueError(
                    'No hp found for model_dir {:s}'.format(model_dir))

        tf.set_random_seed(hp['seed'])

        self.model_dir = model_dir
        self.hp = hp

        n_input = hp['n_input']
        n_rnn = hp['n_rnn']
        n_output = hp['n_output']

        # Input, target output, and cost mask
        # Shape: [Time, Batch, Num_units]
        self.x = tf.placeholder("float", [None, None, n_input])
        self.y = tf.placeholder("float", [None, None, n_output])
        self.c_mask = tf.placeholder("float", [None, n_output])

        self.rec_kernel_ini = recurrent_kernel_initializer
        self.input_kernel_ini = input_kernel_initializer
        self.rec_bias_ini = recurrent_bias_initializer
        self.output_kernel_ini = output_kernel_initializer
        self.output_bias_ini = output_bias_initializer

        cell = LeakyRNNCell(n_rnn,
                            hp['alpha'],
                            sigma_rec=hp['sigma_rec'],
                            recurrent_kernel_initializer=self.rec_kernel_ini,
                            input_kernel_initializer=self.input_kernel_ini,
                            recurrent_bias_initializer=self.rec_bias_ini,
                            activation=hp['activation'])

        # Dynamic rnn with time major
        self.h, states = rnn.dynamic_rnn(
            cell, self.x, dtype=tf.float32, time_major=True)

        # Output
        with tf.variable_scope("output"):
            if self.output_kernel_ini is None:
                self.output_kernel_ini = tf.glorot_uniform_initializer(dtype=tf.float32)
            w_out = tf.get_variable(
                'output_kernel',
                [n_rnn, n_output],
                dtype=tf.float32,
                initializer=self.output_kernel_ini
            )
            if self.output_bias_ini is None:
                self.output_bias_ini = tf.constant_initializer(0.0, dtype=tf.float32)
            b_out = tf.get_variable(
                'output_bias',
                [n_output],
                dtype=tf.float32,
                initializer=self.output_bias_ini
            )

            h_shaped = tf.reshape(self.h, (-1, n_rnn))
            y_shaped = tf.reshape(self.y, (-1, n_output))
            y_hat_ = tf.matmul(h_shaped, w_out) + b_out
            y_hat = tf.sigmoid(y_hat_)
            self.cost_lsq = tf.reduce_mean(tf.square((y_shaped - y_hat) * self.c_mask))

            self.y_hat = tf.reshape(y_hat, (-1, tf.shape(self.h)[1], n_output))
            y_hat_fix, y_hat_ring = tf.split(self.y_hat, [1, n_output - 1], axis=-1)
            self.y_hat_loc = tf_popvec(y_hat_ring)

        self.var_list = tf.trainable_variables()
        self.recurrent_weight = [v for v in self.var_list if 'recurrent_kernel' in v.name]

        for v in self.var_list:
            if 'input_kernel' in v.name:
                self.w_in = v
            elif 'recurrent_kernel' in v.name:
                self.w_rec = v
            elif 'recurrent_bias' in v.name:
                self.b_rec = v
            elif 'output_kernel' in v.name:
                self.w_out = v
            elif 'output_bias' in v.name:
                self.b_out = v

        # Regularization terms
        self.cost_reg = tf.constant(0.)
        if hp['l1_weight'] > 0:
            if 'distance_matrix' in hp:
                Distance = hp['distance_matrix']
                self.cost_reg += hp['l1_weight'] * tf.reduce_sum(tf.abs(self.recurrent_weight) * Distance)
            else:
                self.cost_reg += hp['l1_weight'] * tf.reduce_sum(tf.abs(self.recurrent_weight))
        if hp['l2_weight'] > 0:
            if 'distance_matrix' in hp:
                Distance = hp['distance_matrix']
                self.cost_reg += hp['l2_weight'] * 0.5 * tf.reduce_sum(tf.abs(self.recurrent_weight) * tf.abs(self.recurrent_weight) * Distance)
            else:
                self.cost_reg += hp['l2_weight'] * tf.nn.l2_loss(self.recurrent_weight)

        # Create an optimizer.
        if 'optimizer' not in hp or hp['optimizer'] == 'adam':
            self.opt = tf.train.AdamOptimizer(
                learning_rate=hp['learning_rate'])
        elif hp['optimizer'] == 'sgd':
            self.opt = tf.train.GradientDescentOptimizer(
                learning_rate=hp['learning_rate'])

        # Set cost
        self.set_optimizer()

        # Variable saver
        self.saver = tf.train.Saver()

    def set_optimizer(self, alpha=0.1):

        cost = (1 - alpha) * self.cost_lsq + alpha * self.cost_reg

        self.grads_and_vars = self.opt.compute_gradients(cost, self.var_list)
        # Apply any applicable weights masks to the gradient and clip
        capped_gvs = []
        for grad, var in self.grads_and_vars:
            if 'input_kernel' in var.op.name:
                if 'w_in_mask' in self.hp:
                    grad *= self.hp['w_in_mask']
            elif 'recurrent_kernel' in var.op.name:
                if 'w_rec_mask' in self.hp:
                    grad *= self.hp['w_rec_mask']
            elif 'output_kernel' in var.op.name:
                if 'w_out_mask' in self.hp:
                    grad *= self.hp['w_out_mask']
            capped_gvs.append((tf.clip_by_value(grad, -1., 1.), var))
        self.train_step = self.opt.apply_gradients(capped_gvs)

    def initialize(self):
        """Initialize the model for training."""
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())

    def restore(self, load_dir=None):
        """restore the model"""
        sess = tf.get_default_session()
        if load_dir is None:
            load_dir = self.model_dir
        save_path = os.path.join(load_dir, 'model.ckpt')
        try:
            self.saver.restore(sess, save_path)
        except:
            # Some earlier checkpoints only stored trainable variables
            self.saver = tf.train.Saver(self.var_list)
            self.saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

    def save(self):
        """Save the model."""
        sess = tf.get_default_session()
        save_path = os.path.join(self.model_dir, 'model.ckpt')
        self.saver.save(sess, save_path)
        print("Model saved in file: %s" % save_path)





