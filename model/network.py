# coding utf-8

import os
# import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import sys
sys.path.append('../')
from model.GRUCell import GRUCell
from utils.tools import load_hp, print_variables
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class Model:
    def __init__(self,
                 model_dir,
                 Wx_initializer,
                 Wr_initializer,
                 Wz_initializer,
                 Wh_initializer,
                 br_initializer=None,
                 bz_initializer=None,
                 output_kernel_initializer=None,
                 output_bias_initializer=None,
                 hp=None
                 ):

        # Reset Tensorflow before running anything
        tf.reset_default_graph()

        if hp is None:
            hp = load_hp(model_dir)
            if hp is None:
                raise ValueError(
                    'No hp found for model_dir {:s}'.format(model_dir))

        tf.set_random_seed(hp['seed'])

        self.model_dir = model_dir
        self.hp = hp

        n_input = hp['n_input']
        n_classes = hp['n_classes']
        n_hidden = hp['n_hidden']

        self.x = tf.placeholder(tf.float32, [None, None, n_input])
        self.y = tf.placeholder(tf.int32, [None, None])

        self.Wx_ini = Wx_initializer
        self.Wr_ini = Wr_initializer
        self.Wz_ini = Wz_initializer
        self.Wh_ini = Wh_initializer
        self.br_ini = br_initializer
        self.bz_ini = bz_initializer
        self.output_kernel_ini = output_kernel_initializer
        self.output_bias_ini = output_bias_initializer

        head = self.x
        self.cell = GRUCell(n_hidden,
                            Wx_initializer=self.Wx_ini,
                            Wr_initializer=self.Wr_ini,
                            Wz_initializer=self.Wz_ini,
                            Wh_initializer=self.Wh_ini,
                            br_initializer=self.br_ini,
                            bz_initializer=self.bz_ini)

        head, _ = tf.nn.dynamic_rnn(self.cell, head, dtype=tf.float32, time_major=True)

        if self.output_kernel_ini is None:
            self.output_kernel_ini = tf.random_normal_initializer(dtype=tf.float32)
        if self.output_bias_ini is None:
            self.output_bias_ini = tf.constant_initializer(0.0, dtype=tf.float32)

        self.y_hat = tf.layers.Dense(units=n_classes,
                                     activation=None,
                                     kernel_initializer=self.output_kernel_ini,
                                     bias_initializer=self.output_bias_ini)(head)
        self.cost = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=self.y_hat))
        model_prediction = tf.argmax(input=self.y_hat, axis=2)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(model_prediction, tf.cast(self.y, tf.int64)), tf.float32))

        self.var_list = tf.trainable_variables()
        self.recurrent_weight = [v for v in self.var_list if 'Wh' in v.name]

        for v in self.var_list:
            if 'Wx' in v.name:
                self.Wx = v
            elif 'Wr' in v.name:
                self.Wr = v
            elif 'Wz' in v.name:
                self.Wz = v
            elif 'Wh' in v.name:
                self.Wh = v
            elif 'br' in v.name:
                self.br = v
            elif 'bz' in v.name:
                self.bz = v
            elif 'dense/kernel' in v.name:
                self.w_out = v
            elif 'dense/bias' in v.name:
                self.b_out = v

        # connection cost terms
        self.cost_wiring = tf.constant(0.)
        if ('l1' in hp and hp['l1'] > 0):
            if 'distance_matrix' in hp:
                Distance = hp['distance_matrix']
                # self.cost_wiring += hp['l1'] * tf.reduce_mean(tf.abs(self.recurrent_weight) * Distance)
                self.cost_wiring += hp['l1'] * tf.reduce_sum(tf.abs(self.recurrent_weight) * Distance)
            else:
                # self.cost_wiring += hp['l1'] * tf.reduce_mean(tf.abs(self.recurrent_weight))
                self.cost_wiring += hp['l1'] * tf.reduce_sum(tf.abs(self.recurrent_weight))


        # Create an optimizer
        self.opt = tf.train.AdamOptimizer(learning_rate=hp['learning_rate'])

        # set cost
        self.set_optimizer()

        # Variable saver
        self.saver = tf.train.Saver()

    def set_optimizer(self):

        # Print Trainable Variables
        # print_variables()

        cost = self.cost + self.cost_wiring
        self.grads_and_vars = self.opt.compute_gradients(cost, self.var_list)
        # Apply any applicable weights masks to the gradient and clip
        capped_gvs = []
        for grad, var in self.grads_and_vars:
            # if 'Wx' in var.op.name:
            #     if 'w_in_mask' in self.hp:
            #         grad *= self.hp['w_in_mask']
            # elif 'Wr' in var.op.name:
            #     if 'w_in_mask' in self.hp:
            #         grad *= self.hp['w_in_mask']
            # elif 'Wz' in var.op.name:
            #     if 'w_in_mask' in self.hp:
            #         grad *= self.hp['w_in_mask']

            if 'Wh' in var.op.name:
                if 'w_rec_mask' in self.hp:
                    grad *= self.hp['w_rec_mask']
            elif 'dense/kernel' in var.op.name:
                if 'w_out_mask' in self.hp:
                    grad *= self.hp['w_out_mask']
            capped_gvs.append((tf.clip_by_value(grad, -1., 1.), var))
        self.train_step = self.opt.apply_gradients(capped_gvs)

    # https://github.com/gyyang/multitask/blob/master/network.py
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

    # https://github.com/gyyang/multitask/blob/master/network.py
    def save(self):
        """Save the model."""
        sess = tf.get_default_session()
        save_path = os.path.join(self.model_dir, 'model.ckpt')
        self.saver.save(sess, save_path)
        print("Model saved in file: %s" % save_path)

    # https://github.com/gyyang/multitask/blob/master/network.py
    def lesion_units(self, sess, units):
        import numpy as np

        # Convert to numpy array
        if units is None:
            return
        elif not hasattr(units, '__iter__'):
            units = np.array([units])
        else:
            units = np.array(units)

        for v in self.var_list:
            v_val = sess.run(v)
            if 'dense/kernel' in v.name:
                # output weights
                v_val[units, :] = 0
            elif 'Wh' in v.name:
                # recurrent weights
                v_val[units, :] = 0
            sess.run(v.assign(v_val))
