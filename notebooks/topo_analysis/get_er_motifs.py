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

part = int(sys.argv[1])
target = sys.argv[2]

np.random.seed(part)

data_dict = {}
r_dict = {}
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
    r_dict[i] = reciprocity

#num_repetitions = 100 #without participation
num_repetitions = 2 # 10-20GB per motif with parti
motifs = []
n = data_dict[target].shape[0] # num of cells in cylinder300 = 6213
p = len(np.where(data_dict[target] == 1)[0]) / n**2 # p = 0.032
#r = 0.0034 #prob of reciprocal connections
r = r_dict[target]

print(f'n:{n}, p:{p}, r:{r}')
for rep in range(num_repetitions):
    cur_erdos = nsr.erdos_renyi_reciprocal(n, p, r)
    cur_motifs = nsm.motifs(cur_erdos, algorithm='louzoun',participation=True)
    motifs.append(cur_motifs)
np.save(f'{target}_rER_motifs_wparticipation_{part}',motifs)