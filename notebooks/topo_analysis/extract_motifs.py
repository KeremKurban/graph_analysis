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

target = sys.argv[1]
np.random.seed(12)

data_dict = {}
cur_path = os.getcwd()


cur_conn_path = os.path.join(cur_path,'cylinders_analysis',target,'data/connectivity.npz')
data_dict[target] = np.array(1*scipy.sparse.load_npz(cur_conn_path).todense())

# real graph
print('Extracting motifs')
motifs =  nsm.motifs(data_dict[target], algorithm='louzoun',participation=True)
np.save(f'{target}_motifs_and_participants',motifs)
