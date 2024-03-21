# coding utf-8

import os
import csv
import copy
import random
import numpy as np
import networkx as nx

import itertools
from community import community_louvain
import networkx.algorithms.community as nx_comm
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


"""
def random_0k_weights(edgelist):
    weights = (np.array(edgelist))[:, -1]
    np.random.shuffle(weights)

    new_edges = list()
    for i in range(len(edgelist)):
        s = edgelist[i][1]
        t = edgelist[i][2]
        new_edges.append([str(i), s, t, weights[i]])

    return new_edges
"""


def random_0k(G0, nswap=1000):

    G = copy.deepcopy(G0)
    swapcount = 0
    while swapcount < nswap:
        ((u, v), (x, y)) = random.sample(G.edges(), 2)
        w1 = G.edges[u, v]['weight']
        w2 = G.edges[x, y]['weight']
        G.add_edge(u, v, weight=w2)
        G.add_edge(x, y, weight=w1)
        swapcount += 1

    return G


def random_1k(G0, nswap=1000, max_tries=1000):
    if nswap < max_tries:
        raise nx.NetworkXError("Number of swaps < number of tries allowed.")

    G = copy.deepcopy(G0)
    swapcount = 0
    while swapcount < nswap:
        ((u, v), (x, y)) = random.sample(G.edges(), 2)
        if (u == x) or (v == y):
            continue
        if ((u, y) not in G.edges()) and ((x, v) not in G.edges()):
            w1 = G.edges[u, v]['weight']
            w2 = G.edges[x, y]['weight']
            G.remove_edge(u, v)
            G.remove_edge(x, y)
            G.add_edge(u, y, weight=w1)
            G.add_edge(x, v, weight=w2)
            swapcount += 1

    return G


class clustering(object):

    def __init__(self, edgelist, weighted=False):
        self.edgelist = edgelist

        self.G = nx.Graph()
        if weighted:
            for [_, u, v, w] in self.edgelist:
                # self.G.add_edge(int(u), int(v))
                w = abs(float(w))
                if (int(u), int(v)) in self.G.edges():
                    edge_weight = nx.get_edge_attributes(self.G, 'weight')
                    if (int(u), int(v)) in edge_weight:
                        w = (w + edge_weight[(int(u), int(v))]) / 2
                    else:
                        w = (w + edge_weight[(int(v), int(u))]) / 2
                self.G.add_edge(int(u), int(v), weight=w)
        else:
            for [_, u, v, _] in self.edgelist:
                self.G.add_edge(int(u), int(v))

        # remove isolate from the dataframe
        self.G.remove_nodes_from(nx.isolates(self.G))
        # remove pendants from the dataframe
        # remove = [node for node, degree in self.G.degree() if degree == 1]
        # self.G.remove_nodes_from(remove)
        # remove self loop
        # self.G.remove_edges_from(nx.selfloop_edges(self.G))

    def louvain_method(self):
        # Louvain algorithm
        partition = community_louvain.best_partition(graph=self.G,
                                                     weight='weight',
                                                     random_state=np.random.RandomState())

        print('modularity: ', community_louvain.modularity(partition, self.G, weight='weight'))

        # G0 = random_1k(self.G)
        # partition0 = community_louvain.best_partition(G0)
        # print(community_louvain.modularity(partition0, G0))


        # visualization
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        pos = nx.spring_layout(self.G)
        cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
        nx.draw_networkx_nodes(self.G, pos, partition.keys(), node_size=40,
                               cmap=cmap, node_color=list(partition.values()))
        nx.draw_networkx_edges(self.G, pos, alpha=0.5)
        plt.show()

        return community_louvain.modularity(partition, self.G, weight='weight')

    # https://github.com/zhiyzuo/python-modularity-maximization/blob/master/modularity_maximization/utils.py
    def get_modularity(self, network, partition):
        '''
        Calculate the modularity. Edge weights are ignored.
        Undirected:
        .. math:: Q = \frac{1}{2m}\sum_{i,j} \(A_ij - \frac{k_i k_j}{2m}\) * \detal_(c_i, c_j)
        Directed:
        .. math:: Q = \frac{1}{m}\sum_{i,j} \(A_ij - \frac{k_i^{in} k_j^{out}}{m}\) * \detal_{c_i, c_j}
        Parameters
        ----------
        network : nx.Graph or nx.DiGraph
            The network of interest
        community_dict : dict
            A dictionary to store the membership of each node
            Key is node and value is community index
        Returns
        -------
        float
            The modularity of `network` given `community_dict`
        '''

        G = network.copy()
        nx.set_edge_attributes(G, {e: 1 for e in G.edges}, 'weight')
        A = nx.to_scipy_sparse_matrix(G).astype(float)

        if type(G) == nx.Graph:
            # for undirected graphs, in and out treated as the same thing
            out_degree = in_degree = dict(nx.degree(G))
            M = 2. * (G.number_of_edges())
            print("Calculating modularity for undirected graph")
        elif type(G) == nx.DiGraph:
            in_degree = dict(G.in_degree())
            out_degree = dict(G.out_degree())
            M = 1. * G.number_of_edges()
            print("Calculating modularity for directed graph")
        else:
            print('Invalid graph type')
            raise TypeError

        nodes = list(G)
        Q = np.sum([A[i, j] - in_degree[nodes[i]] *
                    out_degree[nodes[j]] / M
                    for i, j in itertools.product(range(len(nodes)),
                                                  range(len(nodes)))
                    if partition[nodes[i]] == partition[nodes[j]]])

        return Q / M


if __name__ == '__main__':

    DATAPATH = os.path.join(os.getcwd(), '../results', 'Person', 'rewiring_DeepR')

    model_dir = os.path.join(DATAPATH, 'rewiring_DeepR_with_cost')
    csv_file = 'edge_list_weighted_200.csv'

    path = os.path.join(model_dir, csv_file)
    with open(path, newline='') as f:
        edgelist = []
        reader = csv.reader(f)
        edgelist = list(reader)

        # remove the header
        edgelist = edgelist[1:]

        clustering(edgelist, weighted=False).louvain_method()