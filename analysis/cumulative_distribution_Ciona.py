# coding utf-8

import os
import csv
import numpy as np
import math
import copy
import random
from scipy.stats import ks_2samp
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def Euclidean_distance(node1, node2):
    i_x = node1[0]
    i_y = node1[1]
    i_z = node1[2]

    j_x = node2[0]
    j_y = node2[1]
    j_z = node2[2]

    d = np.sqrt((i_x - j_x) ** 2 + (i_y - j_y) ** 2 + (i_z - j_z) ** 2)
    return d


def vertex_swapping(meta, nswap=1500):

    swapcount = 0
    while swapcount < nswap:

        i, j = np.random.choice(len(meta), 2, replace=False)

        temp = meta[list(meta.keys())[i]]
        meta[list(meta.keys())[i]] = meta[list(meta.keys())[j]]
        meta[list(meta.keys())[j]] = temp

        swapcount += 1

    return meta


def edge_exchange(G0, nswap=1000):

    G = copy.deepcopy(G0)
    swapcount = 0
    while swapcount < nswap:
        ((u, v), (x, y)) = random.sample(G.edges(), 2)
        if (u == x) or (v == y):
            continue
        if ((u, y) not in G.edges()) and ((x, v) not in G.edges()):
            G.remove_edge(u, v)
            G.remove_edge(x, y)
            G.add_edge(u, y)
            G.add_edge(x, v)
            swapcount += 1

    return G


def cumulative_distribution(G, meta, d_max=None):

    edge_length = []
    for s, t in G.edges:
        # w = G.edges[s, t]['weight']
        edge_length.append(Euclidean_distance(meta[s], meta[t]))
    edge_length = np.array(edge_length)

    x_min = 0
    x_max = math.ceil(np.max(edge_length))

    x = np.linspace(x_min, x_max, x_max + 1)
    y = [0] * len(x)

    for l in edge_length:
        y[int(math.ceil(l))] += 1
    y = y / np.sum(y)

    """
    plt.bar(x, y)
    plt.xlabel('weighted length')
    plt.ylabel('$p$')
    plt.title('distribution')
    plt.show()

    plt.plot(x, y, 'o')
    plt.xlabel('weighted length')
    plt.ylabel('$p$')
    plt.title('log-log distribution')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    """

    # Cumulative distribution
    dp = [0] * len(y)
    dp[-1] = y[-1]
    for i in range(len(y) - 2, -1, -1):
        dp[i] = dp[i + 1] + y[i]

    if d_max is None:
        d_max = np.max(x)

    # plt.plot(x / d_max, dp)
    # plt.yscale('log')
    # plt.show()

    return x / d_max, dp


if __name__ == '__main__':

    # get real biological neural networks
    G = nx.DiGraph()
    with open(os.path.join('../datasets', 'Ciona', 'Synaptic_network.csv')) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            G.add_edge(str(row[1]), str(row[2]))

    meta = dict()
    with open(os.path.join('../datasets', 'Ciona', 'Ciona_CNS_neurons.csv')) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            meta[str(row[0])] = (float(row[2]), float(row[3]), float(row[4]))

    edge_list_1 = []
    for s, t in G.edges:
        edge_list_1.append(Euclidean_distance(meta[s], meta[t]))
    edge_list_1 = np.array(edge_list_1)

    distance = []
    for i in G.nodes():
        for j in G.nodes():
            distance.append(Euclidean_distance(meta[i], meta[j]))
    D_max = np.max(distance)

    x1, dp1 = cumulative_distribution(G, meta, d_max=D_max)

    # get bio-instantiated GRUs
    x = []
    y = []
    x_random = []
    y_random = []

    DATAPATH = os.path.join(os.getcwd(), '../results', 'Person', 'Ciona')
    model_dir = os.path.join(DATAPATH, 'test')

    frontal_meta = dict()  # [node_id, posx, posy, posz]
    node = []
    with open(os.path.join(DATAPATH, 'Ciona_CNS_neurons.csv')) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            node.append(row[1])
            frontal_meta[row[1]] = (float(row[2]), float(row[3]), float(row[4]))

    csv_file = 'edge_list_weighted_200.csv'
    G0 = nx.DiGraph()
    with open(os.path.join(model_dir, csv_file)) as ff:
        reader = csv.reader(ff)
        header = next(reader)
        for row in reader:
            w = abs(float(row[3]))
            G0.add_edge(node[int(row[1])], node[int(row[2])], weight=w)

    edge_list = []
    for s, t in G0.edges:
        edge_list.append(Euclidean_distance(meta[s], meta[t]))
    edge_list = np.array(edge_list)

    d, dp = cumulative_distribution(G0, frontal_meta, d_max=D_max)
    x += list(d)
    y += list(dp)

    # --------------- calculate k-S distance ------------------
    G_random = edge_exchange(G0)
    edge_list_2 = []
    for s, t in G_random.edges:
        edge_list_2.append(Euclidean_distance(meta[s], meta[t]))
    edge_list_2 = np.array(edge_list_2)

    print("K-S distance is: ", np.round(ks_2samp(edge_list_1, edge_list)[0], 4))

    # ---------------- network randomization ---------------------
    # G1 = edge_exchange(G0)
    # d2, dp2 = cumulative_distribution(G=G1, meta=frontal_meta, d_max=D_max)
    frontal_meta1 = vertex_swapping(frontal_meta)
    d2, dp2 = cumulative_distribution(G0, meta=frontal_meta1, d_max=D_max)
    x_random += list(d2)
    y_random += list(dp2)

    # ------------------- visualization ------------------------
    fig = plt.figure(figsize=(3, 2))
    fig.add_axes([0.2, 0.2, 0.75, 0.7])

    plt.plot(x1, dp1, '-', label='CNS of $Ciona$ intestinalis')

    ax = sns.lineplot(x, y, label='Trained GRU')
    ax1 = sns.lineplot(x_random, y_random, label="Randomization", color='grey')
    plt.xlim(0, 1.05)
    plt.ylim(1e-3, 1)
    plt.yscale('log')
    plt.xlabel('$d$', fontsize=8)
    plt.ylabel('$P^{cum}(d)$', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend(loc=1, fontsize=5)
    plt.show()
