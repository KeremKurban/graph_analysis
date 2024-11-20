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
graph_type = sys.argv[3] # ER or rER

np.random.seed(num_repetitions)

data_dict = {}
r_dict = {}
cur_path = os.getcwd()
for i in os.listdir('cylinders_analysis/'):
    #if i == 'mosaic':
    #    continue
    if i != target:
        continue
    cur_conn_path = os.path.join(cur_path,'cylinders_analysis',i,'data/connectivity.npz')
    adj_matrix = np.array(1*scipy.sparse.load_npz(cur_conn_path).todense())
    num_cells = adj_matrix.shape[-1] # -1 because if SC input exists, it will still take postsyn cells
    reciprocity = (adj_matrix+adj_matrix.T == 2).sum() / float(num_cells*(num_cells-1))
    data_dict[i] = adj_matrix
    r_dict[i] = reciprocity

#num_repetitions = 100 #without participation
motifs = []
n = data_dict[target].shape[0] # num of cells in cylinder300 = 6213
p = len(np.where(data_dict[target] == 1)[0]) / n**2 # p = 0.032
#r = 0.0034 #prob of reciprocal connections
r = r_dict[target]

if graph_type == 'rER':
    print(f'n:{n}, p:{p}, r:{r}')
elif graph_type == 'ER':
    print(f'n:{n}, p:{p}')

for rep in range(num_repetitions):
    print(f'{rep}/{num_repetitions}')
    if graph_type == 'rER':
        cur_erdos = nsr.erdos_renyi_reciprocal(n, p, r)
        np.save(f'{target}_rER_matrix_{rep}',cur_erdos)        
    elif graph_type == 'ER':
        cur_erdos = nsr.erdos_renyi(n, p)
        np.save(f'{target}_ER_matrix_{rep}',cur_erdos)        

    #cur_motifs = nsm.motifs(cur_erdos, algorithm='louzoun',participation=True)
    #motifs.append(cur_motifs)

