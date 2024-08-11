# coding utf-8
import numpy as np


def createWeightMask(epsilon, noRows, noCols, hp):
    # generate an Erdos Renyi sparse weights mask
    rng = np.random.RandomState(hp['seed'] + 1000)
    mask_weights = (rng.rand(noRows, noCols) < epsilon).astype(int)
    noParameters = np.sum(mask_weights)
    return [noParameters, mask_weights]


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
    distance_mat = distance_mat / np.max(distance_mat)
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


def prune_threshold(weights, thresholds):
    is_con = np.greater(abs(weights), 0).astype(int)
    noWeights = np.sum(is_con)

    rewiredWeights = abs(weights.copy())
    rewiredWeights[rewiredWeights > thresholds] = 1
    rewiredWeights[rewiredWeights != 1] = 0
    weightMaskCore = rewiredWeights.copy()

    noSurviving = np.sum(rewiredWeights)
    noRemoved = noWeights - noSurviving

    return weightMaskCore, noRemoved, noSurviving


def grow_random(noRewires, rewiredWeights):
    is_con = np.greater(abs(rewiredWeights), 0).astype(int)
    reconnect_candidate_coord = np.where(np.logical_not(is_con))

    n_candidates = np.shape(reconnect_candidate_coord)[1]
    reconnect_sample_id = np.random.permutation(n_candidates)[:int(noRewires)]

    for i in reconnect_sample_id:
        s = reconnect_candidate_coord[0][i]
        t = reconnect_candidate_coord[1][i]
        is_con[s, t] = 1

    return is_con.astype(int)


def grow_cost(noRewires, rewiredWeights, distance_mat):
    is_con = np.greater(abs(rewiredWeights), 0).astype(int)
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
            p.append((1-distance_mat)[s][t])
    p = p / np.sum(p)

    reconnect_sample_id = np.random.choice(range(int(n_candidates)), size=int(noRewires), replace=False, p=p)

    for i in reconnect_sample_id:
        s = reconnect_candidate_coord[0][i]
        t = reconnect_candidate_coord[1][i]
        is_con[s, t] = 1

    return is_con.astype(int)


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




