import numpy as np
from scipy import sparse
import networkx as nx
import matplotlib.pyplot as plt
import igraph
from logging import getLogger
logger = getLogger(__name__)

data_dir = '/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/data/'
filename = 'ca1_ca3_to_cylinder300_synaptome.npz'

# log the data
logger.info(f'Loading data from {data_dir}/{filename}')
sparse_matrix = sparse.load_npz(f'{data_dir}/{filename}')
sparse_matrix[sparse_matrix>1] = 1 # convert into connectome

# log the process
logger.info('Calculating the modularity of the connectome')

### CONNECTOME LEVEL ANALYSIS

# TARGET ADJ EXTRACTION
#TODO: Extract subgraphs from the main connectome i.e. dorsal, ventral, medial
#TODO: Optional : Find the volume with the most curvature and compare the subgraphs with it 
#TODO: Extract a target with EM scale (250x140x90 um3) like in Turner et al 2022 for visual cortex (target_name: EMcube_dorsal/ventral/intermediate)
#TODO: Extract adj from SSCx circuit: L23 same as EMcube
#TODO: Extract adjacencies from size EMcube to cylinder300 to dorsal1/3rd
#TODO: Extract adjacencies that divides CA1 into 5 dorsoventrally 

#TODO: Compare EMcubes in dorsal, medial, ventral fashion
#TODO: Compare EMcubes with turner data same size in visual cortex

### STATISTICAL ANALYSIS

#TODO: Write classes to accept replication n=1000 times for significance comparison. Save extracted features in a csv file
#TODO: Apply Benforini correction for multiple comparisons based on sample 

