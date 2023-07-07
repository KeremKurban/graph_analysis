# from coordinate_query import CoordinateQuery, enriched_cells_positions, query_enriched_positions, LON, TRA, RAD
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
from scipy.stats import norm, lognorm, expon, gamma, weibull_min, chisquare, powerlaw, beta, pareto, rayleigh, cauchy, pareto, gumbel_r, exponweib, burr, genextreme, triang, johnsonsb, johnsonsu
from graph_analysis.downsample import RatCA1SubmatrixGenerator
from graph_analysis.master import GraphAnalysis
from graph_analysis.information_flow import InformationFlow
from graph_analysis.coordinate_query import CoordinateQuery, enriched_cells_positions, query_enriched_positions, LON, TRA, RAD
import pandas as pd

logging.basicConfig(level=logging.INFO)


results_save_dir = f"{os.getcwd()}/output"
os.makedirs(results_save_dir, exist_ok=True)


# logging.info('Generating submatrices')
# ATLAS_DIR = "/gpfs/bbp.cscs.ch/project/proj112/entities/atlas/20211004_BioM/"
# COORDINATES_DIR = f'{ATLAS_DIR}/coordinates.nrrd'
# volume_save_dir = f"{os.getcwd()}/volumetric_slices"
# q = CoordinateQuery(COORDINATES_DIR)
# submatrix_generator = RatCA1SubmatrixGenerator(q, ATLAS_DIR)
# force_compute = False
# ## generate lon slices
# slice_thickness = 400
# submatrix_generator.generate_lon_slices(slice_thickness,volume_save_dir,force_overwrite=force_compute)
# ## generate tra slices
# tra_dt = 0.1
# submatrix_generator.generate_tra_slices(tra_dt,volume_save_dir,force_overwrite=force_compute)
# ## generate intersection slices
# submatrix_generator.generate_lon_tra_intersection_slices(volume_save_dir,force_overwrite=force_compute)



#logging.info('Testing Bandwith per group')
adj_path = '/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/data/ca1_synaptome.npz'
synaptome = sparse.load_npz(adj_path)
CIRCUIT_DIR =  '/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/sonata/circuit_config.json'
c = Circuit(CIRCUIT_DIR)
mtypes_by_gid = c.nodes['hippocampus_neurons'].get().mtype.values
#high,low = analysis.bandwidth_groups(2, mtypes_by_gid)


# logging.info('Analyzing Information Flow')
# intersection_masks_dir = '/gpfs/bbp.cscs.ch/project/proj112/home/kurban/topology_paper/volumetric_slices/lon_tra_intersection'
# lon_file_names = os.listdir(f'/gpfs/bbp.cscs.ch/project/proj112/home/kurban/topology_paper/volumetric_slices/lon')
# num_lon_slices = len([i for i in lon_file_names if i.endswith('.nrrd')])
# for lon_i in range(1, num_lon_slices + 1):
#     info_flow = InformationFlow(adj_path, longit_slice=lon_i, intersection_masks_dir=intersection_masks_dir,
#                                 circuit_dir=CIRCUIT_DIR)
#     info_flow.run(save_parent_dir=results_save_dir)
#     plt.close('all')


logging.info('Analyzing Degree Distribution')
# distributions = [norm, lognorm, expon, gamma]
distributions = [
    norm, lognorm, expon, gamma, weibull_min, powerlaw,
    beta, pareto, rayleigh, cauchy, pareto, gumbel_r,
    exponweib, burr, genextreme, triang, johnsonsb, johnsonsu
]


def filter_valid_distributions(dists,functions=['fit','pdf','cdf']):
    for dist in dists:
        for func in functions:
            if func not in dist.__dir__():
                try:
                    dists.remove(dist)
                except:
                    print(dist)
    return dists

distributions = filter_valid_distributions(distributions)
logging.info(f'Valid distributions: {[dist.name for dist in distributions]}')

df = pd.DataFrame(columns=['Degree Type', 'Distribution', 'Parameters', 'SSE'])

my_dict = {}

for degree_type in ['in','out','total']:
    analysis = GraphAnalysis(synaptome,degree_type)
    logging.info(f'Analyzing {degree_type}-degree distribution')
    analysis.plot_degree_distribution()
    analysis.plot_log_binned_degree_distribution(cumulative=True,fit_min_degree=50)
    # Define the list of distributions to test
    best_fit, best_params, best_sse = analysis.check_best_fit(distributions)
    # logging.info("Best fit for out-degree distribution: {best_fit.name}")
    # logging.info("Best parameters for out-degree distribution: {best_params}")
    # logging.info("SSE for out-degree distribution: {best_sse}")

    # save the results into a csv file independently
    df = df.append({'Degree Type': degree_type,
                    'Distribution': best_fit.name,
                    'Parameters': best_params,
                    'SSE': best_sse},
                    ignore_index=True)

    analysis.plot_log_binned_degree_distribution(cumulative=True, fit_min_degree=50)

df.to_csv(f'{results_save_dir}/degree_distribution_fits.csv', index=False)


# logging.info('Analyzing clustering coefficient')
# analysis.igraph.

# logging.info('Analyzing average path lengths')


# logging.info('Analyzing small worldness')


# logging.info('Analyzing rich club coefficient')


# logging.info('Analyzing common neighbors')


# logging.info('Analyzing triplet distribution')



