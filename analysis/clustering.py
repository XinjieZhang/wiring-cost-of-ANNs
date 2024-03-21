# coding utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.append('../')

import csv
import numpy as np
import networkx as nx
# import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from model.network import Model
from utils.tools import load_hp, print_variables
from datasets.Person_preprocess import PersonData
from datasets.Gesture_preprocess import GestureData
import matplotlib.pyplot as plt


class Analysis(object):
    def __init__(self,
                 model_dir,
                 method,
                 dataset,
                 n_clusters=None):

        self.dataset = dataset
        if self.dataset == 'Gesture':
            self.data = GestureData()
        elif self.dataset == 'Person':
            self.data = PersonData()

        self.hp = load_hp(model_dir)
        self.model_dir = model_dir
        self.method = method
        self.n_clusters = n_clusters

        n_input = self.hp['n_input']
        n_hidden = self.hp['n_hidden']
        rng = np.random.RandomState(self.hp['seed'])

        Wx_0 = rng.randn(n_input, n_hidden)
        Wr_0 = rng.randn(n_input, n_hidden)
        Wz_0 = rng.randn(n_input, n_hidden)
        Wh_0 = rng.randn(n_hidden, n_hidden) / np.sqrt(n_hidden)

        Wx_initializer = tf.constant_initializer(Wx_0, dtype=tf.float32)
        Wr_initializer = tf.constant_initializer(Wr_0, dtype=tf.float32)
        Wz_initializer = tf.constant_initializer(Wz_0, dtype=tf.float32)
        Wh_initializer = tf.constant_initializer(Wh_0, dtype=tf.float32)

        # Build the model
        self.model = Model(model_dir,
                           Wx_initializer,
                           Wr_initializer,
                           Wz_initializer,
                           Wh_initializer)
        self.build_graph()

    def build_graph(self):
        with tf.Session() as sess:
            model = self.model
            model.restore()
            w_rec = sess.run(model.Wh)

        self.G = nx.Graph()
        x, y = np.where(w_rec)
        for index in range(len(x)):
            s = x[index]
            t = y[index]
            self.G.add_edge(s, t)

        # remove isolate
        self.G.remove_nodes_from(nx.isolates(self.G))
        # remove pendants from the dataframe
        # remove = [node for node, degree in self.G.degree() if degree == 1]
        # self.G.remove_nodes_from(remove)
        # remove self loop
        # self.G.remove_edges_from(nx.selfloop_edges(self.G))

    def accuracy_per_class(self, y_label, y_pred, label):
        orientation_candidate_coord = np.where(y_label == label)
        n_candidates = np.shape(orientation_candidate_coord)[1]

        y = np.ones(n_candidates) * label
        y_hat = []
        for i in range(n_candidates):
            j = orientation_candidate_coord[0][i]
            k = orientation_candidate_coord[1][i]
            y_hat.append(y_pred[j][k])

        return np.mean(np.equal(y, y_hat))

    # structural clustering
    def girvan_newman_moduality_curve(self):
        from networkx.algorithms import community
        import itertools

        G = self.G
        n_clusters = range(3, 13)
        modularities = list()
        for n in n_clusters:
            comp = community.girvan_newman(G)
            for communities in itertools.islice(comp, n - 1):
                tuple(sorted(c) for c in communities)
            module = tuple(sorted(c) for c in communities)
            modularity = community.modularity(G, module)
            modularities.append(modularity)
        idx = np.argmax(modularities)
        num_cluster = n_clusters[idx]

        return num_cluster

    def given_newman(self):
        from networkx.algorithms import community
        import itertools

        n_cluster = self.n_clusters
        if n_cluster is None:
            n_cluster = self.girvan_newman_moduality_curve()

        G = self.G
        comp = community.girvan_newman(G)
        for communities in itertools.islice(comp, n_cluster - 1):
            tuple(sorted(c) for c in communities)
        clusters = tuple(sorted(c) for c in communities)
        print('Grivan Newman -- modularity is :', community.modularity(G, clusters))

        return clusters

    def greedy_modularity_communities(self):
        from networkx.algorithms.community import greedy_modularity_communities
        from networkx.algorithms import community

        G = self.G
        clusters = list(greedy_modularity_communities(G))
        print('Greedy modularity communities -- number of clusters:', len(clusters))
        print('Greedy modularity communities -- modularity is :', community.modularity(G, clusters))

        return clusters

    def louvain_method(self):
        # Louvain algorithm
        from community import community_louvain

        G = self.G
        partition = community_louvain.best_partition(G)
        print('Louvain method -- modularity is:',
              community_louvain.modularity(partition, G))

        values = list(partition.values())
        indexs = np.arange(np.max(values) + 1)
        clusters = []
        for i in indexs:
            node_list = []
            for key, value in partition.items():
                if int(value) == int(i):
                    node_list.append(int(key))
            clusters.append(node_list)

        # save clusters
        fname = open(os.path.join(self.model_dir, 'module', 'clusters.csv'), 'w', newline='')
        csv.writer(fname).writerow(('Id', 'Is'))
        for i in range(len(clusters)):
            for j in clusters[i]:
                csv.writer(fname).writerow((int(j), int(i + 1)))
        fname.close()

        return clusters

    def lesions(self, clusters):

        if clusters is not None:
            self.clusters = clusters
        else:
            # Community detection method
            method = self.method

            if method == 'Greedy':
                self.clusters = self.greedy_modularity_communities()
            elif method == 'Newman':
                self.clusters = self.given_newman()
            elif method == 'Louvain':
                self.clusters = self.louvain_method()

        # The first will be the intact network
        lesion_units_list = [None]
        for i in range(len(self.clusters)):
            lesion_units = []
            for j in sorted(self.clusters[i]):
                lesion_units.append(j)
            lesion_units_list += [lesion_units]

        perfs_store_list = list()
        perfs_changes = list()

        for i, lesion_units in enumerate(lesion_units_list):
            model = self.model
            with tf.Session() as sess:
                model.restore()
                model.lesion_units(sess, lesion_units)

                feed_dict = {model.x: self.data.test_x,
                             model.y: self.data.test_y}
                y_hat_test, cost = sess.run([model.y_hat, model.cost], feed_dict=feed_dict)
                y_pred = sess.run(tf.argmax(input=y_hat_test, axis=2))

            perfs_store = list()
            if self.dataset == 'Gesture':
                self.rules = ['Rest', 'Preparation', 'Stroke', 'Hold', 'Retraction']
            elif self.dataset == 'Person':
                self.rules = ['lying', 'sitting', 'standing', 'walking', 'falling', 'all', 'ground']
            # self.rules = [0, 1, 2, 3, 4, 5, 6]  # generic representation
            for il, rule in enumerate(self.rules):
                perf = self.accuracy_per_class(y_label=self.data.test_y, y_pred=y_pred, label=il)
                perfs_store.append(perf)

            perfs_store = np.array(perfs_store)
            perfs_store_list.append(perfs_store)

            if i > 0:
                # perfs_changes.append(perfs_store - perfs_store_list[0])
                perfs_changes.append((perfs_store - perfs_store_list[0]) / perfs_store_list[0] * 100)

        perfs_changes = np.array(perfs_changes)

        np.savetxt(os.path.join(self.model_dir, 'module', 'perfs_store_list.txt'), np.asarray(perfs_store_list))

        return perfs_store_list, perfs_changes

    def plot_lesions(self):
        """Lesion individual cluster and show performance."""

        # Colors used for clusters
        kelly_colors = ['#4d4d4d', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
                        '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']

        _, perfs_changes = self.lesions()

        cb_labels = 'Performances change'

        v = np.max(np.abs(perfs_changes)) + 0.1
        v = round(v, 1)
        vmins = -v
        vmaxs = +v
        ticks = [vmins, vmaxs]

        fs = 7
        figsize = (3.5, 2.5)
        rect = [0.22, 0.2, 0.58, 0.7]
        rect_cb = [0.84, 0.2, 0.03, 0.7]
        rect_color = [0.22, 0.15, 0.58, 0.05]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(rect)
        im = ax.imshow(perfs_changes.T, cmap='coolwarm', aspect='auto',
                       interpolation='nearest', vmin=vmins, vmax=vmaxs)

        tick_names = [r for r in self.rules]
        _ = plt.yticks(range(len(tick_names)), tick_names,
                       rotation=0, va='center', fontsize=fs)
        plt.xticks([])
        plt.xlabel('Clusters', fontsize=7, labelpad=13)
        ax.tick_params('both', length=0)
        for loc in ['bottom', 'top', 'left', 'right']:
            ax.spines[loc].set_visible(False)

        ax = fig.add_axes(rect_cb)
        cb = plt.colorbar(im, cax=ax, ticks=ticks)
        cb.outline.set_linewidth(0.5)
        cb.set_label(cb_labels, fontsize=7, labelpad=-8)
        plt.tick_params(axis='both', which='major', labelsize=7)

        ax = fig.add_axes(rect_color)
        for il in range(len(self.clusters)):
            ax.plot([il, il + 1], [0, 0], linewidth=4, solid_capstyle='butt',
                    color=kelly_colors[il+1])
            ax.text(np.mean(il + 0.5), -0.5, str(il + 1), fontsize=6,
                    ha='center', va='top', color=kelly_colors[il+1])
        ax.set_xlim([0, len(self.clusters)])
        ax.set_ylim([-1, 1])
        ax.axis('off')

        plt.show()

    def random_lession(self, clusters, cluster_index, n=100):

        self.clusters = clusters

        # The first will be the intact network
        output_nodes_list = [None]

        output_nodes = [j for j in sorted(self.clusters[cluster_index - 1])]
        output_nodes_list += [output_nodes]

        n_output = len(self.clusters[cluster_index - 1])  # number of output nodes
        n_hidden = self.hp['n_hidden']
        for i in range(n):
            output_nodes_random = np.random.choice(n_hidden, n_output, replace=False)
            output_nodes_list += [output_nodes_random]

        perfs_store_list = list()
        perfs_changes = list()

        for i, output_nodes in enumerate(output_nodes_list):
            model = self.model
            with tf.Session() as sess:
                model.restore()
                model.lesion_units(sess, output_nodes)

                feed_dict = {model.x: self.data.test_x,
                             model.y: self.data.test_y}
                y_hat_test, cost = sess.run([model.y_hat, model.cost], feed_dict=feed_dict)
                y_pred = sess.run(tf.argmax(input=y_hat_test, axis=2))

            perfs_store = list()
            if self.dataset == 'Gesture':
                self.rules = ['Rest', 'Preparation', 'Stroke', 'Hold', 'Retraction']
            elif self.dataset == 'Person':
                self.rules = ['lying', 'sitting', 'standing', 'walking', 'falling', 'all', 'ground']
            # self.rules = [0, 1, 2, 3, 4, 5, 6]  # generic representation
            for il, rule in enumerate(self.rules):
                perf = self.accuracy_per_class(y_label=self.data.test_y, y_pred=y_pred, label=il)
                perfs_store.append(perf)

            perfs_store = np.array(perfs_store)
            perfs_store_list.append(perfs_store)

            if i > 0:
                perfs_changes.append(perfs_store - perfs_store_list[1])

        perfs_changes = np.array(perfs_changes) > 0
        p_value = np.sum(perfs_changes, axis=0) / n
        print('p values is :')
        print(p_value)

        return np.round(np.array(perfs_store_list), 6)

    def plot_output(self, clusters, cluster_index):
        import pandas as pd

        perfs_store_list = self.random_lession(clusters=clusters, cluster_index=cluster_index)

        name = 'Cluster_' + str(cluster_index)
        data_1 = {'Intack': np.array(perfs_store_list)[0, 0],
                  str(name): np.array(perfs_store_list)[1, 0],
                  'Rand': np.array(perfs_store_list)[2:, 0]}
        data_2 = {'Intack': np.array(perfs_store_list)[0, 1],
                  str(name): np.array(perfs_store_list)[1, 1],
                  'Rand': np.array(perfs_store_list)[2:, 1]}
        data_3 = {'Intack': np.array(perfs_store_list)[0, 2],
                  str(name): np.array(perfs_store_list)[1, 2],
                  'Rand': np.array(perfs_store_list)[2:, 2]}
        data_4 = {'intack': np.array(perfs_store_list)[0, 3],
                  str(name): np.array(perfs_store_list)[1, 3],
                  'Rand': np.array(perfs_store_list)[2:, 3]}
        data_5 = {'intack': np.array(perfs_store_list)[0, 4],
                  str(name): np.array(perfs_store_list)[1, 4],
                  'Rand': np.array(perfs_store_list)[2:, 4]}
        data_6 = {'intack': np.array(perfs_store_list)[0, 5],
                  str(name): np.array(perfs_store_list)[1, 5],
                  'Rand': np.array(perfs_store_list)[2:, 5]}
        data_7 = {'intack': np.array(perfs_store_list)[0, 6],
                  str(name): np.array(perfs_store_list)[1, 6],
                  'Rand': np.array(perfs_store_list)[2:, 6]}

        fs = 9
        figsize = (9, 5)
        y_lim = [-0.05, 1.0]
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(top=0.9)

        ax1 = fig.add_axes([0.12, 0.55, 0.16, 0.34])
        ax1.set_title('lying', fontsize=fs)
        ax1.set_ylabel('Performance')
        ax1.set_ylim(y_lim)
        df = pd.DataFrame(data_1)
        df.boxplot(ax=ax1, fontsize=fs, grid=False)

        ax2 = fig.add_axes([0.32, 0.55, 0.16, 0.34])
        ax2.set_title('sitting', fontsize=fs)
        ax2.set_ylim(y_lim)
        df = pd.DataFrame(data_2)
        df.boxplot(ax=ax2, fontsize=fs, grid=False)

        ax3 = fig.add_axes([0.52, 0.55, 0.16, 0.34])
        ax3.set_title('standing', fontsize=fs)
        ax3.set_ylim(y_lim)
        df = pd.DataFrame(data_3)
        df.boxplot(ax=ax3, fontsize=fs, grid=False)

        ax4 = fig.add_axes([0.72, 0.55, 0.16, 0.34])
        ax4.set_title('walking', fontsize=fs)
        ax4.set_ylim(y_lim)
        df = pd.DataFrame(data_4)
        df.boxplot(ax=ax4, fontsize=fs, grid=False)

        ax5 = fig.add_axes([0.12, 0.1, 0.16, 0.34])
        ax5.set_title('falling', fontsize=fs)
        ax5.set_ylim(y_lim)
        ax5.set_ylabel('Performance')
        df = pd.DataFrame(data_5)
        df.boxplot(ax=ax5, fontsize=fs, grid=False)

        ax6 = fig.add_axes([0.32, 0.1, 0.16, 0.34])
        ax6.set_title('all', fontsize=fs)
        ax6.set_ylim(y_lim)
        df = pd.DataFrame(data_6)
        df.boxplot(ax=ax6, fontsize=fs, grid=False)

        ax7 = fig.add_axes([0.52, 0.1, 0.16, 0.34])
        ax7.set_title('ground', fontsize=fs)
        ax7.set_ylim(y_lim)
        df = pd.DataFrame(data_7)
        df.boxplot(ax=ax7, fontsize=fs, grid=False)
        plt.show()


if __name__ == '__main__':

    dataset = 'Person'
    methods = ['Louvain', 'Greedy', 'Newman']  # community detection method

    DATAPATH = os.path.join(os.getcwd(), '../results', 'Person', 'rewiring_DeepR')
    model_dir = os.path.join(DATAPATH, 'rewiring_DeepR_with_cost')

    if not os.path.exists(os.path.join(model_dir, 'module')):
        os.makedirs(os.path.join(model_dir, 'module'))

    """
    with open(os.path.join(model_dir, 'module', 'clusters.csv'), newline='') as f:
        nodelist = []
        reader = csv.reader(f)
        nodelist = list(reader)

    # remove the header
    nodelist = nodelist[1:]

    indexs = np.arange(int(nodelist[-1][1]))
    clusters = []
    for i in indexs:
        node_list = []
        for value, key in nodelist:
            if int(key) == int(i+1):
                node_list.append(int(value))
        clusters.append(node_list)

    """

    # Analysis(model_dir, method=methods[0]).plot_lesions()
    perfs_store_list, perfs_changes = Analysis(model_dir, method=methods[0], dataset=dataset).lesions(clusters=None)

    print(np.round(perfs_changes, 6))

    size = len(perfs_changes)
    x = np.arange(size)

    if dataset == 'Gesture':
        n = 5
    elif dataset == 'Person':
        n = 7

    total_width = 0.8
    width = total_width / n
    x = x - (total_width - width) / 2

    for i in range(n):
        plt.bar(x + i * width, np.asarray(perfs_changes.T[i]), width=width, label=str(i + 1))
    plt.legend()

    plt.xlabel('Cluster index')
    plt.ylabel('Relative ACC')
    plt.show()

    # Analysis(model_dir, method=methods[0], dataset=dataset).plot_output(clusters=clusters, cluster_index=4)

    # perf_list = Analysis(model_dir, method=methods[0], dataset=dataset).random_lession(clusters=clusters, cluster_index=1)
    # for i in range(len(perf_list)):
    #     print(perf_list[i])

