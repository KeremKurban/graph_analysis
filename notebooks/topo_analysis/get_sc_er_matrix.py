import bluepy
import netsci.visualization as nsv
import numpy as np
import scipy
import netsci.metrics.motifs as nsm
import networkx
import matplotlib.pyplot as plt
import netsci.models.random as nsr
import glob, os,sys
from pathlib import Path
import pandas as pd
import seaborn as sns
import logging

num_repetitions = int(sys.argv[1]) # 10-20GB per motif with parti
target = sys.argv[2]

np.random.seed(num_repetitions)

data_dict = {}
cur_path = os.getcwd()
for i in os.listdir('cylinders_analysis/'):
    if i == 'mosaic':
        continue
    if i != target:
        continue
    cur_conn_path = os.path.join(cur_path,'cylinders_analysis',i,'data/connectivity.npz')
    adj_matrix = np.array(1*scipy.sparse.load_npz(cur_conn_path).todense())
    num_cells = adj_matrix.shape[-1] # -1 because if SC input exists, it will still take postsyn cells
    reciprocity = (adj_matrix+adj_matrix.T == 2).sum() / float(num_cells*(num_cells-1))
    data_dict[i] = adj_matrix

#num_repetitions = 100 #without participation
motifs = []

# Reload
num_SC_nodes_to_cylinder = 267238
#num_SC_edges_to_cylinder = 119561308
#num_cylinder_nodes = 6213 
sc_conn_prob = 0.0704

num_ca1_target_cells = data_dict[target].shape[0] # num of cells in cylinder300 = 6213


print(f'SC fibers n:{num_SC_nodes_to_cylinder}, p:{sc_conn_prob}, num_target_cells:{num_ca1_target_cells}')

for rep in range(num_repetitions):
    print(f'{rep+1}/{num_repetitions}')
    sc_to_target = np.random.binomial(1, sc_conn_prob, size=(num_SC_nodes_to_cylinder, num_ca1_target_cells))
    np.save(f'SC_{target}_ER_matrix_{rep}',sc_to_target)        
