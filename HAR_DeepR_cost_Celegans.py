# coding utf-8
# rewiring with C. elegans Connectome

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import csv
import time
from collections import defaultdict

import numpy as np
import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import sys
sys.path.append('../')
from model.network1 import Model
from datasets.Person_preprocess import PersonData
from utils.tools import mkdir_p, save_hp


def get_default_hp():
    hp = {
        'n_input': 3+4,
        'n_hidden': 279,
        'num_edges': 2194,
        'n_classes': 7,
        'learning_rate': 0.005,
        'batch_size': 64,
        'training_iters': 201,
        'log_period': 1,
        'l1': 0
    }
    return hp


def get_input_or_output_nodes():
    input_nodes = []
    with open(os.path.join('datasets', 'C. elegans', 'Sensory.csv'), 'r', encoding='utf-8') as ff:
        reader = csv.reader(ff)
        header = next(reader)
        for row in reader:
            if row[2] not in input_nodes:
                input_nodes.append(row[2])

    output_nodes = []
    with open(os.path.join('datasets', 'C. elegans', 'Neurons_to_Muscles.csv')) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row[1] not in output_nodes:
                output_nodes.append(row[1])

    return input_nodes, output_nodes


def largest_connected_component_of_Celegans():

    G = nx.DiGraph()
    with open(os.path.join('datasets', 'C. elegans', 'Connectome.csv')) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            G.add_edge(row[1], row[2])

    largest_cc = list(max(nx.connected_components(nx.Graph(G)), key=len))

    return largest_cc


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


# https://github.com/guillaumeBellec/deep_rewiring
def assert_connection_number(theta, targeted_number):
    '''
    Function to check during the tensorflow simulation if the number of
    connection in well defined after each simulation
    '''

    is_con = np.greater(theta, 0)
    nb_is_con = np.sum(is_con.astype(int))
    assert np.equal(nb_is_con, targeted_number), "the number of connection has changed"


# https://github.com/guillaumeBellec/deep_rewiring
def rewiring(theta, weights, target_nb_connection, sign_0, epsilon=1e-12):
    '''
    The rewiring operation to use after each iteration.
    :param theta:
    :param target_nb_connection:
    :return:
    '''

    is_con = np.greater(theta, 0).astype(int)
    w = weights * is_con

    n_connected = np.sum(is_con)
    nb_reconnect = target_nb_connection - n_connected
    nb_reconnect = np.max(nb_reconnect, 0)

    reconnect_candidate_coord = np.where(np.logical_not(is_con))

    n_candidates = np.shape(reconnect_candidate_coord)[1]
    reconnect_sample_id = np.random.permutation(n_candidates)[:nb_reconnect]

    for i in reconnect_sample_id:
        s = reconnect_candidate_coord[0][i]
        t = reconnect_candidate_coord[1][i]
        sign = sign_0[s, t]
        w[s, t] = sign * epsilon

    w_mask = np.greater(abs(w), 0).astype(int)

    return w, w_mask, nb_reconnect


def rewiring_improved(theta, weights, target_nb_connection, sign_0, distance_mat, epsilon=1e-12):
    '''
    The rewiring operation to use after each iteration.
    :param theta:
    :param target_nb_connection:
    :return:
    '''

    is_con = np.greater(theta, 0).astype(int)
    w = weights * is_con

    n_connected = np.sum(is_con)
    nb_reconnect = target_nb_connection - n_connected
    nb_reconnect = np.max(nb_reconnect, 0)

    reconnect_candidate_coord = np.where(np.logical_not(is_con))

    n_candidates = np.shape(reconnect_candidate_coord)[1]

    p = list()
    distance_mat = distance_mat / np.max(distance_mat)
    for i in range(n_candidates):
        s = reconnect_candidate_coord[0][i]
        t = reconnect_candidate_coord[1][i]
        if s == t:
            p.append(0)
        else:
            p.append((1 - distance_mat)[s][t])
    p = p / np.sum(p)

    reconnect_sample_id = np.random.choice(range(int(n_candidates)), size=int(nb_reconnect), replace=False, p=p)

    for i in reconnect_sample_id:
        s = reconnect_candidate_coord[0][i]
        t = reconnect_candidate_coord[1][i]
        sign = sign_0[s, t]
        w[s, t] = sign * epsilon

    w_mask = np.greater(abs(w), 0).astype(int)

    return w, w_mask, nb_reconnect


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

    person_data = PersonData()

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
    n_classes = hp['n_classes']

    # load diatance matrix
    largest_cc = largest_connected_component_of_Celegans()

    meta = []  # [node_id, posx, posy, posz]
    with open(os.path.join('datasets', 'C. elegans', 'spatialpositions', 'distances.csv')) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if row[0] in largest_cc:
                meta.append((str(row[0]), float(row[1]), float(row[2]), float(row[3])))

    meta_1 = np.random.permutation(meta)
    fname = open(os.path.join(model_dir, 'C-elegans-connectom.csv'), 'w', newline='')
    csv.writer(fname).writerow(('Id', 'name', 'posx', 'posy', 'posz'))
    for index in range(len(meta_1)):
        csv.writer(fname).writerow((index, meta_1[index][0], meta_1[index][1], meta_1[index][2], meta_1[index][3]))
    fname.close()

    distance_matrix = np.zeros((n_hidden, n_hidden))
    for i in range(n_hidden):
        i_x = float(meta_1[i][1])
        i_y = float(meta_1[i][2])
        i_z = float(meta_1[i][3])
        for j in range(i + 1, n_hidden):
            j_x = float(meta_1[j][1])
            j_y = float(meta_1[j][2])
            j_z = float(meta_1[j][3])
            distance_matrix[i][j] = np.sqrt((i_x - j_x) ** 2 + (i_y - j_y) ** 2 + (i_z - j_z)**2)
            distance_matrix[j][i] = distance_matrix[i][j]
    hp['distance_matrix'] = distance_matrix.tolist()

    normalized = True
    if normalized:
        sigma_D = np.sqrt(np.sum(distance_matrix * distance_matrix) / n_hidden / (n_hidden - 1))
        hp['l1'] = hp['l1'] / sigma_D
        save_hp(hp, model_dir)

    input_nodes, output_nodes = get_input_or_output_nodes()
    w_in_mask = np.zeros((n_input, n_hidden))
    w_out_mask = np.zeros((n_hidden, n_classes))

    randomization = False
    if randomization:
        for index in np.random.choice(n_hidden, len(input_nodes), replace=False):
            w_in_mask[:, index] = 1

        for index in np.random.choice(n_hidden, len(output_nodes), replace=False):
            w_out_mask[index, :] = 1
    else:
        for index in range(len(meta_1)):
            if meta_1[index][0] in input_nodes:
                w_in_mask[:, index] = 1
            if meta_1[index][0] in output_nodes:
                w_out_mask[index, :] = 1

    hp['Wx_mask'] = w_in_mask.tolist()
    hp['Wr_mask'] = w_in_mask.tolist()
    hp['Wz_mask'] = w_in_mask.tolist()
    hp['w_out_mask'] = w_out_mask.tolist()

    rng = np.random.RandomState()
    Wx_0 = rng.randn(n_input, n_hidden)
    Wx_0 = Wx_0 * w_in_mask
    Wr_0 = rng.randn(n_input, n_hidden)
    Wr_0 = Wr_0 * w_in_mask
    Wz_0 = rng.randn(n_input, n_hidden)
    Wz_0 = Wz_0 * w_in_mask

    W_out0 = rng.randn(n_hidden, n_classes)
    W_out0 = W_out0 * w_out_mask

    Wh_0 = rng.randn(n_hidden, n_hidden) / np.sqrt(n_hidden)
    Wh_sign_0 = np.sign(Wh_0)

    nb_non_zero = hp['num_edges']
    Wh_0, w_rec_mask = weight_sampler_strict_number(Wh_0, n_hidden, n_hidden, nb_non_zero)
    hp['w_rec_mask'] = w_rec_mask.tolist()
    save_hp(hp, model_dir)
    [Wx, Wr, Wz, Wh, br, bz, w_out, b_out] = [Wx_0, Wr_0, Wz_0, Wh_0, None, None, W_out0, None]

    # Store results
    log = defaultdict(list)
    log['model_dir'] = model_dir

    # Record time
    t_start = time.time()

    # Reset Tensorflow before running anything
    tf.reset_default_graph()

    train_accuracy = []
    valid_accuracy = []
    test_accuracy = []
    best_valid_accuracy = 0
    best_valid_stats = (0, 0, 0, 0, 0, 0, 0)
    for epoch in range(hp['training_iters']):

        session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        with tf.Session(config=session_conf) as sess:

            # Build the model
            model = create_model(model_dir, hp, Wx, Wr, Wz, Wh, br, bz, w_out, b_out)

            if load_dir is not None:
                sess.run(tf.global_variables_initializer())
                model.restore(load_dir)  # restore trainable variables
            else:
                # Assume everything is restored
                sess.run(tf.global_variables_initializer())

            Wh_0 = sess.run(model.Wh)

            losses = []
            accs = []
            for batch_x, batch_y in person_data.iterate_train(batch_size=hp['batch_size']):
                _, acc, loss = sess.run([model.train_step, model.accuracy, model.cost],
                                        feed_dict={model.x: batch_x, model.y: batch_y})

                losses.append(loss)
                accs.append(acc)
            train_accuracy.append(np.mean(accs))

            # Validation
            if (epoch + 1) % hp['log_period'] == 0:
                test_acc, test_loss = sess.run([model.accuracy, model.cost],
                                               feed_dict={model.x: person_data.test_x, model.y: person_data.test_y})
                valid_acc, valid_loss = sess.run([model.accuracy, model.cost],
                                                 feed_dict={model.x: person_data.valid_x, model.y: person_data.valid_y})
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

            # ------------- weights evolution ---------------
            # Get weight list
            Wh = sess.run(model.Wh)

            # save the recurrent network model
            if epoch % 10 == 0:
                fname = open(os.path.join(model_dir, 'edge_list_weighted_' + str(epoch) + '.csv'), 'w', newline='')
                csv.writer(fname).writerow(('Id', 'Source', 'Target', 'Weight'))
                x, y = np.where(Wh)
                for index in range(len(x)):
                    s = x[index]
                    t = y[index]
                    csv.writer(fname).writerow((index, s, t, Wh[s, t]))
                fname.close()

            # deep rewiring
            # Guillaume Bellec et al. (2017) DEEP REWIRING: TRAINING VERY SPARSE DEEP NETWORKS WORKS
            # arXiv:1711.05136v1
            mask_connected = lambda th: (np.greater(th, 0)).astype(int)
            noise_update = lambda th: np.random.normal(scale=1e-5, size=th.shape)

            l1 = 1e-4  # regulation coefficient
            add_gradient_op = Wh + mask_connected(abs(Wh)) * noise_update(Wh)
            apply_l1_reg = - mask_connected(abs(Wh)) * np.sign(Wh) * l1
            Wh_1 = add_gradient_op + apply_l1_reg

            Wh, w_rec_mask, nb_reconnect = rewiring(Wh_0 * Wh_1, Wh_1, nb_non_zero, Wh_sign_0)
            # Wh, w_rec_mask, nb_reconnect = rewiring_improved(theta=Wh_0 * Wh_1,
            #                                                  weights=Wh_1,
            #                                                  target_nb_connection=nb_non_zero,
            #                                                  sign_0=Wh_sign_0,
            #                                                  distance_mat=distance_matrix)
            assert_connection_number(abs(Wh), nb_non_zero)
            hp['w_rec_mask'] = w_rec_mask.tolist()
            save_hp(hp, model_dir)

            model.weight_update(sess, Wh, 'Wh')
            model.set_optimizer()

            model.save()  # save the model
            load_dir = model_dir

            # learning rate decay
            if (epoch + 1) % round(hp['training_iters'] / 3) == 0:
                lr = hp['learning_rate'] * 0.5
                hp['learning_rate'] = lr
                save_hp(hp, model_dir)

        tf.reset_default_graph()
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

    parser.add_argument('--modeldir', type=str, default='results/Person/C_elegans_connectom/test')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    hp = {
          'training_iters': 151,
          'learning_rate': 0.008,
          'num_edges': 2194,
          'l1': 1e-4,
          'sparsity': 1,
          'sparsity_input': 1,
          'sparsity_output': 1
          }
    train(args.modeldir,
          seed=0,
          hp=hp)