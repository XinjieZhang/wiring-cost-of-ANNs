# coding utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import csv
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import sys
sys.path.append('../')

from utils import tools
from utils import task
from model.network import Model
from utils.task import generate_trials
from utils.evaluation import do_eval
from model.structure_evolution import weight_sampler_strict_number, rewireMask
from utils.parameters import get_default_hp, update_initializer


def create_model(model_dir, hp, w_in, w_rec, b_rec=None, w_out=None, b_out=None):

    input_kernel_initializer = tf.constant_initializer(w_in, dtype=tf.float32)
    recurrent_kernel_initializer = tf.constant_initializer(w_rec, dtype=tf.float32)

    recurrent_bias_initializer = tf.constant_initializer(b_rec, dtype=tf.float32) if (b_rec is not None) else b_rec
    output_kernel_initializer = tf.constant_initializer(w_out, dtype=tf.float32) if (w_out is not None) else w_out
    output_bias_initializer = tf.constant_initializer(b_out, dtype=tf.float32) if (b_out is not None) else b_out

    model = Model(model_dir,
                  input_kernel_initializer=input_kernel_initializer,
                  recurrent_kernel_initializer=recurrent_kernel_initializer,
                  recurrent_bias_initializer=recurrent_bias_initializer,
                  output_kernel_initializer=output_kernel_initializer,
                  output_bias_initializer=output_bias_initializer,
                  hp=hp)

    return model


def train(model_dir,
          hp=None,
          maxepoches=40,
          display_step=250,
          ruleset='dm',
          rule_trains=None,
          rule_prob_map=None,
          seed=0,
          load_dir=None):

    tools.mkdir_p(model_dir)

    # Network parameters
    default_hp = get_default_hp(ruleset)
    if hp is not None:
        default_hp.update(hp)
    hp = default_hp
    hp['seed'] = seed
    hp['rng'] = np.random.RandomState(seed)  # Pseudorandom number generator

    # Rules to train and test. Rules in a set are trained together
    if rule_trains is None:
        # By default, training all rules available to this ruleset
        hp['rule_trains'] = task.rules_dict[ruleset]
    else:
        hp['rule_trains'] = rule_trains
    hp['rules'] = hp['rule_trains']

    # Assign probabilities for rule_trains
    if rule_prob_map is None:
        rule_prob_map = dict()

    # Turn into rule_trains format
    hp['rule_probs'] = None
    if hasattr(hp['rule_trains'], '__iter__'):
        # Set default as 1.
        rule_prob = np.array(
                [rule_prob_map.get(r, 1.) for r in hp['rule_trains']])
        hp['rule_probs'] = list(rule_prob/np.sum(rule_prob))

    update_initializer(hp)
    tools.save_hp(hp, model_dir)

    # Display hp
    for key, val in hp.items():
        print('{:20s} = '.format(key) + str(val))

    # initialize weights
    n_input = hp['n_input']
    n_rnn = hp['n_rnn']
    _w_in_start = hp['w_in_start']
    _w_rec_start = hp['w_rec_start']
    _w_rec_init = hp['w_rec_init']
    rng = np.random.RandomState(hp['seed'])

    w_in0 = (rng.randn(n_input, n_rnn) / np.sqrt(n_input) * _w_in_start)

    if _w_rec_init == 'diag':
        w_rec0 = _w_rec_start * np.eye(n_rnn)
    elif _w_rec_init == 'randortho':
        w_rec0 = _w_rec_start * tools.gen_ortho_matrix(n_rnn, rng=rng)
    elif _w_rec_init == 'randgauss':
        w_rec0 = (_w_rec_start *
                  rng.randn(n_rnn, n_rnn) / np.sqrt(n_rnn))

    if (hp['epsilon_rec'] is not None and
            hp['epsilon_rec'] <= 1.0):
        nb_non_zero = hp['num_edges']
        w_rec0, w_rec_mask = weight_sampler_strict_number(w_rec0, n_rnn, n_rnn, nb_non_zero)
        hp['w_rec_mask'] = w_rec_mask.tolist()
        tools.save_hp(hp, model_dir)
    [w_in, w_rec, b_rec, w_out, b_out] = [w_in0, w_rec0, None, None, None]

    # Store results
    log = defaultdict(list)
    log['model_dir'] = model_dir

    # Record time
    t_start = time.time()

    for epoch in range(maxepoches):
        # build the model
        model = create_model(model_dir, hp, w_in, w_rec, b_rec, w_out, b_out)

        with tf.Session() as sess:
            if load_dir is not None:
                model.restore(load_dir)  # complete restore
            else:
                # Assume everything is restored
                sess.run(tf.global_variables_initializer())

            for step in range(1, 1+display_step):
                # Training
                rule_train_now = hp['rng'].choice(hp['rule_trains'], p=hp['rule_probs'])
                # Generate a random batch of trials.
                # Each batch has the same trial length
                trial = generate_trials(
                    rule_train_now, hp, 'random',
                    batch_size=hp['batch_size_train'])

                # Generating feed_dict.
                feed_dict = tools.gen_feed_dict(model, trial, hp)
                sess.run(model.train_step, feed_dict=feed_dict)

            # Validation
            log['trials'].append(display_step * (epoch + 1))
            log['epochs'].append(epoch + 1)
            log['times'].append(time.time() - t_start)
            log = do_eval(sess, model, log, hp['rule_trains'])

            # ------------- weights evolution ---------------
            # Get weight list
            w_in = sess.run(model.w_in)
            w_rec = sess.run(model.w_rec)
            w_out = sess.run(model.w_out)
            b_rec = sess.run(model.b_rec)
            b_out = sess.run(model.b_out)

            # SET
            # It removes the weights closest to zero
            w_rec_mask, w_rec_core = rewireMask(weights=w_rec, noWeights=nb_non_zero, zeta=hp['zeta'])
            # w_rec_mask, w_rec_core = rewireMask_improved(weights=w_rec,
            #                                              noWeights=nb_non_zero,
            #                                              zeta=hp['zeta'],
            #                                              distance_mat=distance_matrix)
            w_rec *= w_rec_core
            hp['w_rec_mask'] = w_rec_mask.tolist()
            tools.save_hp(hp, model_dir)

            # learning rate decay
            if (epoch + 1) % round(maxepoches / 3) == 0:
                lr = hp['learning_rate'] * 0.5
                hp['learning_rate'] = lr
                tools.save_hp(hp, model_dir)

        # print("optimization finished!")


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--modeldir', type=str, default='experiments/DMTask/rewiring_SET/SET_with_l2')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    hp = {'activation': 'softplus',
          'w_rec_init': 'randortho',
          'n_rnn': 256,
          'learning_rate': 0.01,
          'epsilon_rec': 1,
          'num_edges': 10000,
          'l1_weight': 0,  # L1 Regularization coefficient
          'l2_weight': 3e-4,  # L2 Regularization coefficient
          'zeta': 0.2
          }
    train(args.modeldir,
          ruleset='dm',
          seed=0,
          hp=hp)


