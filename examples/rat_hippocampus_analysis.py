from coordinate_query import CoordinateQuery, enriched_cells_positions, query_enriched_positions, LON, TRA, RAD
import os
import logging
from tqdm import tqdm
import logging
from voxcell import VoxelData
from voxcell.nexus.voxelbrain import Atlas
import numpy as np
from sys import argv
import igraph
from scipy import sparse
from bluepysnap import Circuit
import matplotlib.pyplot as plt

from ..graph_analysis.downsample import RatCA1SubmatrixGenerator
from ..graph_analysis.graph_analysis import GraphAnalysis
from ..graph_analysis.information_flow import InformationFlow

logging.basicConfig(level=logging.INFO)

ATLAS_DIR = "/gpfs/bbp.cscs.ch/project/proj112/entities/atlas/20211004_BioM/"
COORDINATES_DIR = f'{ATLAS_DIR}/coordinates.nrrd'
# save_dir = "/gpfs/bbp.cscs.ch/project/proj112/home/kurban/topology_paper/volumetric_slices"
save_dir = f"{os.getcwd()}/volumetric_slices"

# Generate submatrices

q = CoordinateQuery(COORDINATES_DIR)
submatrix_generator = RatCA1SubmatrixGenerator(q, ATLAS_DIR)
force_compute = False
## generate lon slices
slice_thickness = 400
submatrix_generator.generate_lon_slices(slice_thickness,save_dir,force_overwrite=force_compute)
## generate tra slices
tra_dt = 0.1
submatrix_generator.generate_tra_slices(tra_dt,save_dir,force_overwrite=force_compute)
## generate intersection slices
submatrix_generator.generate_lon_tra_intersection_slices(save_dir,force_overwrite=force_compute)



#logging.info('Testing Bandwith per group')
adj_path = '/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/data/ca1_synaptome.npz'
synaptome = sparse.load_npz(adj_path)
analysis = GraphAnalysis(synaptome,'out')
CIRCUIT_DIR =  '/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/sonata/circuit_config.json'
c = Circuit(CIRCUIT_DIR)
mtypes_by_gid = c.nodes['hippocampus_neurons'].get().mtype.values
#high,low = analysis.bandwidth_groups(2, mtypes_by_gid)


logging.info('Analyzing Information Flow')
intersection_masks_dir = 'volumetric_slices/lon_tra_intersection'
lon_file_names = os.listdir(f'volumetric_slices/lon')
num_lon_slices = len([i for i in lon_file_names if i.endswith('.nrrd')])
for lon_i in range(1, num_lon_slices + 1):
    info_flow = InformationFlow(adj_path, longit_slice=lon_i, intersection_masks_dir=intersection_masks_dir,
                                circuit_dir=CIRCUIT_DIR)
    info_flow.run(save_parent_dir='data/information_flow')
    plt.close('all')


logging.info('Analyzing degree distribution')
analysis.plot_degree_distribution()
analysis.plot_log_binned_degree_distribution(cumulative=True,fit_min_degree=50)
from scipy.stats import norm, lognorm, expon, gamma
# Define the list of distributions to test
distributions = [norm, lognorm, expon, gamma]
best_fit, best_params, best_sse = analysis.check_best_fit(distributions)
print("Best fit for out-degree distribution:", best_fit.name)
print("Best parameters for out-degree distribution:", best_params)
print("SSE for out-degree distribution:", best_sse)


logging.info('Analyzing clustering coefficient')


logging.info('Analyzing average path lengths')


logging.info('Analyzing small worldness')


logging.info('Analyzing rich club coefficient')


logging.info('Analyzing common neighbors')


logging.info('Analyzing triplet distribution')



