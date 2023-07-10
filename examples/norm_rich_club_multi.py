from scipy import sparse
from bluepysnap import Circuit
from graph_analysis.randomize import WeightPermutedRandomModel
from graph_analysis.rich_cub import RichClubAnalysis
from dask.distributed import Client, LocalCluster
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import ttest_ind
from dask_jobqueue import SLURMCluster
import logging
import pickle
from sys import argv

logging.basicConfig(level=logging.INFO)
logging.getLogger('distributed.core').setLevel(logging.WARNING)

if __name__ == '__main__':
    part = argv[1]
    logging.info(f'Starting part {part}')
    logging.info('Loading graph')
    # Load the graph
    adj_path = '/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/data/ca1_synaptome.npz'
    synaptome = sparse.load_npz(adj_path)

    logging.info('Loading circuit')
    target = 'slice10'
    CIRCUIT_DIR =  '/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/sonata/circuit_config.json'
    circuit = Circuit(CIRCUIT_DIR)
    nodes = circuit.nodes["hippocampus_neurons"]
    target_indices = nodes.ids(target)
    target_synaptome = synaptome[target_indices,:][:,target_indices]
    del synaptome, nodes, circuit

    logging.info('Creating client')
    # Create a Dask client
    client = Client(n_workers=4, memory_limit='80GB')
    # Initialize lists to store results over trials
    norm_rc_trials = []
    # Perform trials
    # rca_syn = RichClubAnalysis(target_synaptome)
    # rc_coeffs_model = rca_syn.calculate_rich_club_coefficients(weighted=True, method='strongest-links-in-network', step=100)
    model_dp = WeightPermutedRandomModel(target_synaptome)
    n_trials = 20

    logging.info('Performing trials for randomized graphs')
    random_rc_coeffs = []
    for i in tqdm(range(n_trials), desc='Performing trials'):
        # Generate randomized graph
        randomized_graph = model_dp.generate(seed=i)
        rca_dp = RichClubAnalysis(randomized_graph)
        k_dict_parallel_dp = rca_dp.calculate_rich_club_coefficients(weighted=True, method='strongest-links-in-network', step=100)
        random_rc_coeffs.append(k_dict_parallel_dp)

    logging.info('Saving results')
    save_dir = '/gpfs/bbp.cscs.ch/project/proj112/home/kurban/topology_paper/toolbox/graph_analysis/examples/output/'
    # np.save(f'{save_dir}/20_random_rc_coeffs_{np.random.choice(1233)}.npy', random_rc_coeffs)
    with open(f'{save_dir}/20_random_rc_coeffs_{str(part)}.pkl', 'wb') as f:
        pickle.dump(random_rc_coeffs, f)
    client.shutdown()    # Shutdown the Dask client

    logging.info('Finished')

# Calculate average over trials
# random_rc_coeffs_avg = np.mean(random_rc_coeffs, axis=0)

# Calculate normalized rich club coefficients
# norm_rc = rc_coeffs_model[:random_rc_coeffs_avg.shape[0]] / random_rc_coeffs_avg

# Statistical test
# _, p_value = ttest_ind(norm_rc_trials, axis=0)


# Plot the results
# f, ax = plt.subplots(figsize=(10,8))
# plt.plot(k_dict_parallel_dp.keys(), norm_rc_avg)

# ax.fill_between(k_dict_parallel_dp.keys(), norm_rc_avg, 1, where=(np.array(norm_rc_avg) >= 1), color='lightblue', alpha=0.4)
# ax.fill_between(k_dict_parallel_dp.keys(), norm_rc_avg, 1, where=(np.array(norm_rc_avg) < 1), color='lightyellow', alpha=0.4)

# ax.axhline(y=1, color='k', linestyle='--', label='y=1')

# plt.title('Average Normalized Rich Club Coefficients for slice10 synaptome (total-weights)')
# plt.xlabel('k (synapses)')
# plt.ylabel('Norm. Rich Club Coefficient')
# plt.legend(['norm_RC_coeff (CA1/weight_permuted)'])
# plt.savefig(f"../output/rich_club_directed_totdegree_weighted.png", dpi=dpi, bbox_inches='tight')
