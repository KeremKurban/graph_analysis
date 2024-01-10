'''
Extract motifs from a graph and calculate their frequencies

Author: Kerem Kurban, Blue Brain Project, EPFL
Date: 2024/01/10

Arguments:
    -n, --node_set: Name of the nodeset. So far only used for logging purposes. Future to be implemented node_set extractor
    -f, --adjacency_matrix: Path to the target graph
    -t, --randomization_type: Type of randomization
    -r, --repetition: Number of repetitions to randomize the circuit
    -s, --seed: Seed for randomization
    -w, --num_workers: Number of workers in parallel
    -o, --output_dir: Output path for the results

Example:
    python3 motifs.py -n cylinder300 
                      -f /gpfs/bbp.cscs.ch/project/proj142/home/kurban/GNNplayground/notebooks/cylinder300_synaptome.npz 
                      -t configurational 
                      -r 2
                      -s 42
                      -w 1
                      -o ../output/motifs
'''

from graph_analysis import triplets, randomize
import networkx as nx
from dotmotif import Motif, GrandIsoExecutor
from scipy import sparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
import dask
from dask.distributed import Client
from logging import getLogger
import psutil

def calculate_motif(motif_name,with_participation=False):
    CM = motif_reader.name_to_matrix(motif_name)
    cur_triplet_motif = reader.matrix_to_dotmotif(CM)
    if cur_triplet_motif == '':
        return None
    else:
        logger.info(f"\n{CM} \n\n dotmotif:\n{cur_triplet_motif}\n\n")
        cur_triplet_motif = Motif(cur_triplet_motif)

    cur_results = Executor.find(cur_triplet_motif)
    total_occurences = len(cur_results)

    if with_participation:
        return (motif_name, cur_results, total_occurences)
    else:    
        return (motif_name, total_occurences)


logger = getLogger(__name__)
logger.setLevel("INFO")

parser = argparse.ArgumentParser(description='Motif analysis')
parser.add_argument('-n','--node_set', help='Name of the node set in sonata', required=True)
parser.add_argument('-f','--adjacency_matrix', help='Path to the target graph in npz format (binary)', required=True)
parser.add_argument('-t','--randomization_type', help='Type of randomization', required=True)
parser.add_argument('-r','--repetition', help='Number of repetitions to randomize the circuit', required=True)
parser.add_argument('-s','--seed', help='Seed for randomization', required=True)
parser.add_argument('-w','--num_workers', help='Number of workers in parallel', required=False, default=1)
parser.add_argument('-o','--output_dir', help='Output path for the results', required=True)
args = vars(parser.parse_args())

target = args['node_set']
file_path = args['adjacency_matrix']
randomization_type = args['randomization_type']
repetition = int(args['repetition'])
seed = int(args['seed'])
num_workers = int(args['num_workers'])
output_dir = args['output_dir']

if num_workers > 1:
    # Initialize dask client with max mem each
    # calculate memory limit per worker using available memory and number of workers
    
    #total memory
    total_mem = psutil.virtual_memory().total
    # total memory in GB
    total_mem = total_mem / 1e9
    # memory per worker to be given to memory_limit in dask client
    mem_per_worker = total_mem / num_workers
    # convert to string and add GB
    mem_per_worker = str(mem_per_worker) + 'GB'

    client = Client(n_workers=num_workers, threads_per_worker=1, memory_limit=mem_per_worker)
    logger.info(f"Running with {num_workers} workers")

check_mtypes = ['Mosaic','Mosaic','Mosaic'] #TODO: should be given in the command line
all_motifs = ['-C','-B','-A','A','B','C','D','E','F','G','H','I','J','K','L','M']
target_adj = sparse.load_npz(file_path)
# target_graph = nx.from_scipy_sparse_array(target_adj,create_using=nx.DiGraph())

# Generate Randomized Model
if randomization_type == 'configurational':
    generator = randomize.ConfigurationModel(target_adj)
    randomized_adjacency = generator.generate(seed=seed)
    target_graph = nx.from_scipy_sparse_array(randomized_adjacency,create_using=nx.DiGraph())
else:
    raise NotImplementedError(f"Randomization type {randomization_type} is not implemented")

# Motif Functions and Converters
complex_motif = Motif("""
# One-direction edge
uniedge(n, m) {
    n -> m
    m !> n
}

# One-direction triangle
unitriangle(x, y, z) {
    uniedge(x, y)
    uniedge(y, z)
    uniedge(z, x)
}

unitriangle(A, B, C)
""") # specific case not like motif2 where edges can also be bi-edge/uniedge
reader = triplets.DotMotifReader()
motif_reader = triplets.MotifReader()


# Motif Calculation
Executor = GrandIsoExecutor(graph=target_graph)

if num_workers>1:
    tasks = [dask.delayed(calculate_motif)(motif_name) for motif_name in all_motifs]
    results = dask.compute(*tasks)
    frequencies = {result[0]: result[1] for result in results if result is not None}
else:
    frequencies = {}
    for motif_name in tqdm(all_motifs):
        print(f"Running motif calculation for {target} of motif {motif_name} of {check_mtypes}...")
        CM = motif_reader.name_to_matrix(motif_name)
        cur_triplet_motif = reader.matrix_to_dotmotif(CM)
        if cur_triplet_motif == '':
            continue
        else:
            print(f"\n{CM} \n\n dotmotif:\n{cur_triplet_motif}\n\n")
            cur_triplet_motif = Motif(cur_triplet_motif)

        cur_results = Executor.find(cur_triplet_motif)
        frequencies[motif_name] = cur_results


# Save the results
df = pd.DataFrame.from_dict(my_dict,orient='index',columns=['Occurences'])
df.to_csv(f'{output_dir}/{target}_{randomization_type}_motif_frequencies.csv')
