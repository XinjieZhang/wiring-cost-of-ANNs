# coding utf-8

import os
import networkx as nx
import csv
import numpy as np

import sys
sys.path.append('../')


# calculating the average
def returnAverage(myDict):
    sum = 0
    for i in myDict:
        sum = sum + myDict[i]
    ave = sum / len(myDict)
    return ave


# calculate the network metrics in a given network (main function)
def calculate_network_metrics(G_new):
    # remove isolate and pendants from the dataframe
    G_new.remove_nodes_from(nx.isolates(G_new))
    remove = [node for node, degree in G_new.degree() if degree == 1]
    G_new.remove_nodes_from(remove)
    # remove self loop
    G_new.remove_edges_from(nx.selfloop_edges(G_new))

    print('global_clusterng_coefficient: ', nx.transitivity(G_new))
    print('average_degree_centrality: ', returnAverage(nx.degree_centrality(G_new)))
    print('density: ', nx.density(G_new))
    print('average_clustering_coefficient: ', nx.average_clustering(G_new))
    # print('triadic_census: ', triadic_census(G_new))
    print('global_efficiency: ', efficiency(G_new))


def digraph_average_shortest_length(G, weighted=False):
    n = len(G.nodes)  # number of nodes

    if weighted:
        shortest_path = dict(nx.shortest_path_length(G, weight='weight'))
    else:
        shortest_path = dict(nx.shortest_path_length(G))

    avg_length = 0
    for i in G.nodes:
        v = shortest_path[i].values()
        inverse_d = 0
        for d in v:
            if not d == 0:
                inverse_d += 1 / d

        avg_length += inverse_d / n / (n - 1)

    return 1 / avg_length


def efficiency(G):
    n = nx.number_of_nodes(G)
    shortest_path = dict(nx.shortest_path_length(G))
    E_global = 0
    for i in G.nodes:
        v = shortest_path[i].values()
        for d in v:
            if not d == 0:
                E_global += 1 / d

    return E_global / n / (n - 1)


def wiring_cost(edges, D):
    m = len(edges)
    n = len(D)
    cost = 0
    for (s, t, _) in edges:
        cost += D[int(s), int(t)] / m

    sigma_D = np.sqrt(np.sum(D * D) / n / (n - 1))
    cost = cost / sigma_D

    print('wiring cost: ', cost)

    return cost


def wiring_cost_weighted(edges, D):
    m = len(edges)
    n = len(D)

    cost = 0
    sigma_W = 0
    for (s, t, w) in edges:
        w = float(w)
        cost += float(D[int(s), int(t)]) * w / m
        sigma_W += w * w / m

    sigma_W = np.sqrt(sigma_W)
    sigma_D = np.sqrt(np.sum(D * D) / n / (n - 1))
    cost = cost / sigma_W / sigma_D

    print('wiring cost (weighted): ', cost)

    return cost


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

    G = nx.DiGraph()
    for [id, u, v, w] in edgelist:
        G.add_edge(u, v, weight=np.sign(float(w)))

    calculate_network_metrics(nx.Graph(G))


    # wiring cost

    # load diatance matrix
    frontal_meta = []  # [node_id, posx, posy]
    with open(os.path.join('../datasets', 'random_network', 'node_coordinate.csv')) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            frontal_meta.append((int(row[0]), float(row[1]), float(row[2])))

    n_nodes = len(frontal_meta)
    distance_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        i_x = frontal_meta[i][1]
        i_y = frontal_meta[i][2]
        for j in range(i + 1, n_nodes):
            j_x = frontal_meta[j][1]
            j_y = frontal_meta[j][2]
            distance_matrix[i][j] = np.sqrt((i_x - j_x) ** 2 + (i_y - j_y) ** 2)
            distance_matrix[j][i] = distance_matrix[i][j]

    edges = [edgelist[i][1:] for i in range(len(edgelist))]
    cost = wiring_cost(edges, distance_matrix)
    weighted_cost = wiring_cost_weighted(edges, distance_matrix)
