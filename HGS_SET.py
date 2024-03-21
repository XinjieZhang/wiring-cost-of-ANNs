# coding utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import csv
import time
from collections import defaultdict

import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import sys
sys.path.append('../')
from model.network import Model
from datasets.Gesture_preprocess import GestureData
from utils.tools import mkdir_p, save_hp


def get_default_hp():
    hp = {
        'n_input': 32,
        'n_hidden': 131,
        'n_classes': 5,
        'learning_rate': 0.005,
        'batch_size': 32,
        'training_iters': 301,
        'log_period': 1,
        'sparsity': None,
    }
    return hp


def weight_sampler_strict_number(w_0, n_in, n_out, nb_non_zero):
    # Generate the random mask
    is_con_0 = np.zeros((n_in, n_out), dtype=bool)
    ind_in = np.random.choice(np.arange(n_in), size=nb_non_zero)
    ind_out = np.random.choice(np.arange(n_out), size=nb_non_zero)
    is_con_0[ind_in, ind_out] = True

    nb_reconnect = nb_non_zero - np.sum(is_con_0)

    reconnect_candidate_coord = np.where(np.logical_not(is_con_0))
    n_candidates = np.shape(reconnect_candidate_coord)[1]
    reconnect_sample_id = np.random.permutation(n_candidates)[:nb_reconnect]

    for i in reconnect_sample_id:
        s = reconnect_candidate_coord[0][i]
        t = reconnect_candidate_coord[1][i]
        is_con_0[s, t] = True

    is_connected = is_con_0.astype(int)
    w = np.where(is_connected, w_0, np.zeros((n_in, n_out)))

    return w, is_connected


def find_first_pos(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def find_last_pos(array, value):
    idx = (np.abs(array - value))[::-1].argmin()
    return array.shape[0] - idx


# https://github.com/dcmocanu/sparse-evolutionary-artificial-neural-networks
def rewireMask(weights, noWeights, zeta):
    # rewire weight matrix

    # remove zeta largest negative and smallest positive weights
    values = np.sort(weights.ravel())
    firstZeroPos = find_first_pos(values, 0)
    lastZeroPos = find_last_pos(values, 0)

    largestNegative = values[int((1-zeta) * firstZeroPos)]
    smallestPositive = values[int(min(values.shape[0] - 1, lastZeroPos +
                                      zeta * (values.shape[0] - lastZeroPos)))]

    rewiredWeights = weights.copy()
    rewiredWeights[rewiredWeights > smallestPositive] = 1
    rewiredWeights[rewiredWeights < largestNegative] = 1
    rewiredWeights[rewiredWeights != 1] = 0
    weightMaskCore = rewiredWeights.copy()

    # add zeta random weights
    nrAdd = 0
    noRewires = noWeights - np.sum(rewiredWeights)
    while (nrAdd < noRewires):
        i = np.random.randint(0, rewiredWeights.shape[0])
        j = np.random.randint(0, rewiredWeights.shape[1])
        if (rewiredWeights[i, j] == 0):
            rewiredWeights[i, j] = 1
            nrAdd += 1

    return [rewiredWeights, weightMaskCore]


def rewireMask_improved(weights, noWeights, zeta, distance_mat):
    # rewire weight matrix

    # remove zeta largest negative and smallest positive weights
    values = np.sort(weights.ravel())
    firstZeroPos = find_first_pos(values, 0)
    lastZeroPos = find_last_pos(values, 0)

    largestNegative = values[int((1-zeta) * firstZeroPos)]
    smallestPositive = values[int(min(values.shape[0] - 1, lastZeroPos +
                                      zeta * (values.shape[0] - lastZeroPos)))]

    is_connect = weights.copy()
    is_connect[is_connect > smallestPositive] = 1
    is_connect[is_connect < largestNegative] = 1
    is_connect[is_connect != 1] = 0
    weightMaskCore = is_connect.copy()

    noRewires = noWeights - np.sum(is_connect)
    reconnect_candidate_coord = np.where(np.logical_not(is_connect))
    n_candidates = np.shape(reconnect_candidate_coord)[1]

    p = list()
    for i in range(n_candidates):
        s = reconnect_candidate_coord[0][i]
        t = reconnect_candidate_coord[1][i]
        if s == t:
            p.append(0)
        else:
            p.append((1-distance_mat)[s][t])
    p = p / np.sum(p)

    reconnect_sample_id = np.random.choice(range(int(n_candidates)), size=int(noRewires), replace=False, p=p)

    for i in reconnect_sample_id:
        s = reconnect_candidate_coord[0][i]
        t = reconnect_candidate_coord[1][i]
        is_connect[s, t] = 1

    return [is_connect, weightMaskCore]


def create_model(model_dir, hp, Wx, Wr, Wz, Wh, br=None, bz=None, w_out=None, b_out=None):

    Wx_initializer = tf.constant_initializer(Wx, dtype=tf.float32)
    Wr_initializer = tf.constant_initializer(Wr, dtype=tf.float32)
    Wz_initializer = tf.constant_initializer(Wz, dtype=tf.float32)
    Wh_initializer = tf.constant_initializer(Wh, dtype=tf.float32)
    br_initializer = tf.constant_initializer(br, dtype=tf.float32) if (br is not None) else br
    bz_initializer = tf.constant_initializer(bz, dtype=tf.float32) if (bz is not None) else bz

    output_kernel_initializer = tf.constant_initializer(w_out, dtype=tf.float32) if (w_out is not None) else w_out
    output_bias_initializer = tf.constant_initializer(b_out, dtype=tf.float32) if (b_out is not None) else b_out

    model = Model(model_dir,
                  Wx_initializer=Wx_initializer,
                  Wr_initializer=Wr_initializer,
                  Wz_initializer=Wz_initializer,
                  Wh_initializer=Wh_initializer,
                  br_initializer=br_initializer,
                  bz_initializer=bz_initializer,
                  output_kernel_initializer=output_kernel_initializer,
                  output_bias_initializer=output_bias_initializer,
                  hp=hp)

    return model


def train(model_dir,
          hp=None,
          seed=0,
          load_dir=None):

    mkdir_p(model_dir)

    gesture_data = GestureData()

    # Network parameters
    default_hp = get_default_hp()
    if hp is not None:
        default_hp.update(hp)
    hp = default_hp
    hp['seed'] = seed
    hp['rng'] = np.random.RandomState(seed)

    # Display hp
    for key, val in hp.items():
        print('{:20s} = '.format(key) + str(val))

    # initializer weights
    n_input = hp['n_input']
    n_hidden = hp['n_hidden']
    rng = np.random.RandomState()

    Wx_0 = rng.randn(n_input, n_hidden)
    Wr_0 = rng.randn(n_input, n_hidden)
    Wz_0 = rng.randn(n_input, n_hidden)
    Wh_0 = rng.randn(n_hidden, n_hidden) / np.sqrt(n_hidden)
    if (hp['sparsity'] is not None and
            hp['sparsity'] <= 1.0):
        # nb_non_zero = int(n_hidden * n_hidden * hp['sparsity'])
        nb_non_zero = hp['num_edges']
        Wh_0, w_rec_mask = weight_sampler_strict_number(Wh_0, n_hidden, n_hidden, nb_non_zero)
        hp['w_rec_mask'] = w_rec_mask.tolist()
        save_hp(hp, model_dir)
    [Wx, Wr, Wz, Wh, br, bz, w_out, b_out] = [Wx_0, Wr_0, Wz_0, Wh_0, None, None, None, None]

    # Store results
    log = defaultdict(list)
    log['model_dir'] = model_dir

    # Record time
    t_start = time.time()

    train_accuracy = []
    valid_accuracy = []
    test_accuracy = []
    best_valid_accuracy = 0
    best_valid_stats = (0, 0, 0, 0, 0, 0, 0)
    for epoch in range(hp['training_iters']):
        # Build the model
        model = create_model(model_dir, hp, Wx, Wr, Wz, Wh, br, bz, w_out, b_out)

        with tf.Session() as sess:
            if load_dir is not None:
                model.restore(load_dir)  # complete restore
            else:
                # Assume everything is restored
                sess.run(tf.global_variables_initializer())

            losses = []
            accs = []
            for batch_x, batch_y in gesture_data.iterate_train(batch_size=hp['batch_size']):
                _, acc, loss = sess.run([model.train_step, model.accuracy, model.cost],
                                        feed_dict={model.x: batch_x, model.y: batch_y})

                losses.append(loss)
                accs.append(acc)
            train_accuracy.append(np.mean(accs))

            # Validation
            if (epoch + 1) % hp['log_period'] == 0:
                test_acc, test_loss = sess.run([model.accuracy, model.cost],
                                               feed_dict={model.x: gesture_data.test_x, model.y: gesture_data.test_y})
                valid_acc, valid_loss = sess.run([model.accuracy, model.cost],
                                                 feed_dict={model.x: gesture_data.valid_x, model.y: gesture_data.valid_y})
                test_accuracy.append(test_acc)
                valid_accuracy.append(valid_acc)
                print(
                    "Epochs {:03d}, train loss: {:0.4f}, train accuracy: {:0.4f}%, valid loss: {:0.4f}, "
                    "valid accuracy: {:0.4f}%, test loss: {:0.4f}, test accuracy: {:0.4f}%, Time: {:0.6f}".format(
                        epoch + 1,
                        np.mean(losses), np.mean(accs) * 100,
                        valid_loss, valid_acc * 100,
                        test_loss, test_acc * 100,
                        time.time() - t_start))
                # Accuracy metric -> higher is better
                if (valid_acc > best_valid_accuracy and epoch > 0):
                    best_valid_accuracy = valid_acc
                    best_valid_stats = (
                        epoch + 1,
                        np.mean(losses), np.mean(acc) * 100,
                        valid_loss, valid_acc * 100,
                        test_loss, test_acc * 100
                    )
                # save the model
                model.save()

            # ------------- weights evolution ---------------
            # Get weight list
            Wx = sess.run(model.Wx)
            Wr = sess.run(model.Wr)
            Wz = sess.run(model.Wz)
            Wh = sess.run(model.Wh)
            br = sess.run(model.br)
            bz = sess.run(model.bz)
            w_out = sess.run(model.w_out)
            b_out = sess.run(model.b_out)

            # save the recurrent network model
            if epoch % 20 == 0:
                fname = open(os.path.join(model_dir, 'edge_list_weighted_' + str(epoch) + '.csv'), 'w', newline='')
                csv.writer(fname).writerow(('Id', 'Source', 'Target', 'Weight'))
                x, y = np.where(Wh)
                for index in range(len(x)):
                    s = x[index]
                    t = y[index]
                    csv.writer(fname).writerow((index, s, t, Wh[s, t]))
                fname.close()

            # SET
            # It removes the weights closest to zero
            w_rec_mask, w_rec_core = rewireMask(weights=Wh, noWeights=nb_non_zero, zeta=hp['zeta'])
            Wh *= w_rec_core
            hp['w_rec_mask'] = w_rec_mask.tolist()
            save_hp(hp, model_dir)

            # learning rate decay
            if (epoch + 1) % round(hp['training_iters'] / 3) == 0:
                lr = hp['learning_rate'] * 0.5
                hp['learning_rate'] = lr
                save_hp(hp, model_dir)

        sess.close()

    print("optimization finished!")
    best_epoch, train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc = best_valid_stats
    print("Best epoch {:03d}, train loss: {:0.6f}, train accuracy: {:0.6f}, valid loss: {:0.6f}, "
          "valid accuracy: {:0.6f}, test loss: {:0.6f}, test accuracy: {:0.6f}".format(
          best_epoch, train_loss, train_acc, valid_loss, valid_acc, test_loss, test_acc
          ))

    # save train loss and train accuracy over all epochs
    np.savetxt(os.path.join(model_dir, 'train_accuracy.txt'), np.asarray(train_accuracy))
    np.savetxt(os.path.join(model_dir, 'valid_accuracy.txt'), np.asarray(valid_accuracy))
    np.savetxt(os.path.join(model_dir, 'test_accuracy.txt'), np.asarray(test_accuracy))

    # plot accuracy over time
    epoch_seq = np.arange(1, hp['training_iters'] + 1)
    plt.plot(epoch_seq, train_accuracy)
    plt.title('train accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--modeldir', type=str, default='results/Gesture/rewiring_SET/SET_baseline')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    hp = {
          'learning_rate': 0.01,
          'num_edges': 764,
          'sparsity': 1,
          'zeta': 0.1
          }
    train(args.modeldir,
          seed=1,
          hp=hp)