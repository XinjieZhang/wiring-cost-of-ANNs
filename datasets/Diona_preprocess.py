# coding utf-8

import os
import pandas as pd
import networkx as nx
from networkx import *
import csv

import sys
sys.path.append('../')


# calculating the average
def returnAverage(myDict):
    sum = 0
    for i in myDict:
        sum = sum + myDict[i]
    ave = sum / len(myDict)
    return ave


def save_CNS_neurons_position(Cell):

    fname = open(os.path.join('..\datasets', 'Ciona', 'Ciona_CNS_neurons.csv'), 'w', newline='')
    csv.writer(fname).writerow(('Id', 'type', 'posx', 'posy', 'posz'))
    for index in range(len(Cell)):
        csv.writer(fname).writerow(
            (Cell[index][0],
             Cell[index][1],
             Cell[index][2],
             Cell[index][3],
             Cell[index][4]))
    fname.close()


def save_synaptic_network(CNS_neurons_edgelist):

    fname = open(os.path.join('..\datasets', 'Ciona', 'Synaptic_network.csv'), 'w', newline='')
    csv.writer(fname).writerow(('Id', 'Source', 'Target', 'Weight'))
    for index, line in enumerate(CNS_neurons_edgelist):
        csv.writer(fname).writerow((index, line[0], line[1], line[2]))
    fname.close()


if __name__ == '__main__':

    cell_infor = pd.read_excel(io=os.path.join('..\datasets', 'Ciona', 'elife-16962-fig3-all-neurons.xlsx'))
    Cell_ID = []
    for index in cell_infor.index.values[1:-4]:
        Cell_ID.append(list(cell_infor.iloc[index, [0, 1, 5, 6, 7]]))
    Cell = []
    CNS_neurons = []
    for i in range(len(Cell_ID)):
        if Cell_ID[i][2] != '-':
            Cell.append(Cell_ID[i])
            CNS_neurons.append(str(Cell_ID[i][0]))

    # save positions of CNS neurons
    save_CNS_neurons_position(Cell)

    # ---------------------- get the synaptic network ------------------------
    file_name = os.path.join('..\datasets', 'Ciona', 'elife-16962-fig16-data1-v1.xlsx')
    data = pd.read_excel(io=file_name)

    target_neurons = list(data.columns.values[1:-1])

    CNS_neurons_edgelist = []

    G = nx.DiGraph()
    G.add_nodes_from(CNS_neurons)
    for index in data.index.values[:-1]:
        data_value = data.iloc[index].values

        for i, x in enumerate(data_value[1:-1]):
            if not pd.isnull(x):
                s = str(data_value[0])
                t = str(target_neurons[i])

                if s in CNS_neurons and t in CNS_neurons:
                    CNS_neurons_edgelist.append([s, t, x])
                    G.add_edge(s, t)

    # save synaptic network
    save_synaptic_network(CNS_neurons_edgelist)

    # -------------------------- network analysis ------------------------------
    largest_cc = list(max(nx.connected_components(nx.Graph(G)), key=len))

    # remove isolate from the dataframe
    G.remove_nodes_from(list(nx.isolates(G)))
    G.remove_edges_from(list(nx.selfloop_edges(G)))

    print('global_clusterng_coefficient: ', transitivity(G))
    print('average_degree_centrality: ', returnAverage(nx.degree_centrality(G)))
    print('density: ', nx.density(G))
    print('average_clustering_coefficient: ', returnAverage(nx.clustering(nx.Graph(G))))

    '''
    for calculating shortest path, we first need to check if the Graph is weakly connected. 
    If not, we first get the largest component and calculate the shortest path only 
    for the largest component of the graph.
    '''
    if nx.number_weakly_connected_components(G) == 1:
        print('shortest_path: ', nx.average_shortest_path_length(G))

    else:
        print('NOTE: Shortest path cannot be calculated. Graph is not connected.')
