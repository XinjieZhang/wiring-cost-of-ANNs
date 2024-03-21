# coding utf-8

# Before calculating the task variance and module variance of the network,
# run clustering.py to perform community detection on the network


import os
import numpy as np
import matplotlib.pyplot as plt


DATAPATH = os.path.join(os.getcwd(), '../results', 'Person', 'rewiring_DeepR')
model_dir = os.path.join(DATAPATH, 'rewiring_DeepR_with_cost')

txt_file = os.path.join(model_dir, 'module', 'perfs_store_list.txt')

data = np.loadtxt(txt_file)

perfs_list = np.array(data)
relative_perfs_changes = list()
for i in range(len(perfs_list)-1):
    relative_perfs_changes.append((perfs_list[i+1] - perfs_list[0]) / perfs_list[0])

relative_perfs_changes = np.array(relative_perfs_changes)

# task z score
perfs_mean = np.mean(relative_perfs_changes, axis=0)
perfs_std = np.std(relative_perfs_changes, axis=0)

print('task variance: ', np.mean(perfs_std))

# module variance
relative_perfs_changes = relative_perfs_changes.T
relative_perfs_mean = np.mean(relative_perfs_changes, axis=0)
relative_perfs_std = np.std(relative_perfs_changes, axis=0)

print('module variance: ', np.mean(relative_perfs_std))