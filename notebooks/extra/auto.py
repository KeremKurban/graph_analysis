'''
Author:Kerem Kurban

This code is deprecated and kept for provenance purposes. See information_flow.py for the latest version.

'''
import imp
import os
from voxcell import VoxelData
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from scipy import sparse
from coordinate_query import CoordinateQuery, enriched_cells_positions, query_enriched_positions, LON, TRA, RAD
from bluepy import Circuit
from voxcell.nexus.voxelbrain import Atlas
from bluepy.geometry.roi import Cube
import pandas as pd
from log_progress import log_progress

def load_circuit(circuit_path):
    return Circuit(circuit_path)

def load_atlas(atlas_path):
    return Atlas.open(atlas_path)

def load_orientation(atlas):
    return atlas.load_data("orientation")

def load_coordinate_query(file_path):
    return CoordinateQuery(file_path)

def load_adjacency_matrix(adj_path):
    return sparse.load_npz(adj_path)

def enriched_xyz_ltr(circuit, coordinate_query):
    return enriched_cells_positions(circuit, coordinate_query)

def update_user_target(filename, gids, target_name, notes):
    with open(filename, 'a') as f:
        f.write(notes)
        f.write('\nTarget Cell %s\n{' % target_name)
        f.write(' '.join(['a%d' % gid for gid in gids]))
        f.write('}\n\n')

def get_gids_in_voxel(mask, circuit):
    in_target = mask.lookup(circuit.cells.get(properties=list("xyz")).values)
    gids_in_target = circuit.cells.get().index.values[np.where(in_target==1)]
    return gids_in_target

def generate_tra_slices(q, tra_dt, tra_slices_dir, num_slices):
    tra_slices = np.arange(0,1,tra_dt)
    for idx, tra_min in enumerate(log_progress(tra_slices, every=1)):
        tra_max = tra_min + tra_dt
        cur_slice_mask = q._get_range_mask(min_value=tra_min, max_value=tra_max, axis=TRA)
        orientation.with_data(np.asarray(cur_slice_mask, dtype=np.uint8)).save_nrrd(f"{tra_slices_dir}/slice_tra__{idx+1}__{num_slices}.nrrd")

def generate_slices_across_axis(q, axis, dt, slices_dir, num_slices):
    '''
    Parameters  
    ----------
    q : CoordinateQuery
        CoordinateQuery object
    axis : int
        Axis along which the slices are to be generated
    dt : float
        Slice thickness
    slices_dir : str
        Directory where the slices are to be saved
    num_slices : int
        Number of slices to be generated

    Returns
    -------
    None
    '''
    slices = np.arange(0,1,dt)
    for idx, min_value in enumerate(log_progress(slices, every=1)):
        max_value = min_value + dt
        cur_slice_mask = q._get_range_mask(min_value=min_value, max_value=max_value, axis=axis)
        orientation.with_data(np.asarray(cur_slice_mask, dtype=np.uint8)).save_nrrd(f"{slices_dir}/slice_{axis}__{idx+1}__{num_slices}.nrrd")

def num_lon_slices(volumetric_slices_LON):
    return len(os.listdir(volumetric_slices_LON))


# intersect LON slices with TRA slice
def get_intersection_slice(tra_slice_idx, lon_slice_idx, lon_dir, tra_dir, save_dir, orientation):
    '''
    Parameters
    ----------
    tra_slice_idx : int
        Index of the TRA slice
    lon_slice_idx : int
        Index of the LON slice
    lon_dir : str
        Directory where the LON slices are saved
    tra_dir : str
        Directory where the TRA slices are saved
    save_dir : str
        Directory where the intersection slices are to be saved
    orientation : VoxelData
        Orientation data. Used to export nrrd files

    Returns
    -------
    None
    '''
    tra_slice = VoxelData.load_nrrd(f'{tra_dir}/slice_tra__{tra_slice_idx}__10.nrrd')
    lon_slice = VoxelData.load_nrrd(f'{lon_dir}/slice400_{lon_slice_idx}.nrrd')
    intersection_slice = tra_slice.raw & lon_slice.raw
    save_path = f"{save_dir}/slice_LON_{lon_slice_idx}_TRA_{tra_slice_idx}.nrrd"
    orientation.with_data(np.asarray(intersection_slice, dtype=np.uint8)).save_nrrd(save_path)

def get_synapse_counts(pre_indices, post_indices, adj):
    temp = adj[pre_indices]
    temp = temp[:,post_indices]
    return int(temp.sum())

def visualize_synapse_distribution(tra_df, target_slices):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')

    xpos, ypos = np.meshgrid(np.arange(target_slices) + 0.25, np.arange(target_slices) + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    dx = dy = 0.5 * np.ones_like(zpos)
    dz = tra_df.values.ravel()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    ax.view_init(elev=60, azim=120)
    ax.set_xlabel('Presynaptic Slice')
    ax.set_ylabel('Postsynaptic Slice')
    ax.set_zlabel('Num Synapses')
    ax.zaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}k'))

    plt.title('Synapses across Transverse axis (slice10)')
    plt.show()


# This way you can use the functions independently and also use them in a larger pipeline as needed
# Here is how you could use the above functions:

circuit_path = "/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/CircuitConfig"
atlas_path = "/gpfs/bbp.cscs.ch/project/proj112/entities/atlas/20211004_BioM/"
coordinate_file_path = '/gpfs/bbp.cscs.ch/project/proj112/entities/atlas/20211004_BioM/coordinates.nrrd'
adj_path = '/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/data/ca1_synaptome.npz'
volumetric_slices_LON = '/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/scripts/5_slice_generation/nrrd_masks_serially_generated_but_not_used/'
tra_slices_dir = "tra_masks"
num_slices_TRA = 10

# logging 
import logging
logging.basicConfig(level=logging.INFO)

logging.info('Loading Circuit')
circuit = load_circuit(circuit_path)

logging.info('Loading Atlas')
atlas = load_atlas(atlas_path)
orientation = load_orientation(atlas)
coordinate_query = load_coordinate_query(coordinate_file_path)
xyz_ltr = enriched_xyz_ltr(circuit, coordinate_query)

logging.info('Loading Adjacency Matrix')
adjacency_matrix = load_adjacency_matrix(adj_path)

logging.info('Generating Volumetric Transverse Slices')
volumetric_slices_TRA = './data/volumetric_slices_TRA'
generate_slices_across_axis(coordinate_query, TRA, 0.1, volumetric_slices_TRA, tra_slices_dir)
# generate_tra_slices(coordinate_query, 0.1, tra_slices_dir, tra_slices_dir)

number_of_lon_slices = num_lon_slices(volumetric_slices_LON)

# Re-run the calculations with refactored code
lon_tra_intersection_dir = './data/lon_tra_intersection'
for lon_slice_idx in np.arange(1, number_of_lon_slices+1):
    for tra_slice_idx in np.arange(1, tra_slices_dir+1):
        get_intersection_slice(tra_slice_idx, lon_slice_idx, volumetric_slices_LON, tra_slices_dir, lon_tra_intersection_dir, orientation)

# Find the synapses count
target_slices = range(1, tra_slices_dir+1)
lon_slice = 10

synapse_df = pd.DataFrame().rename_axis('Transverse_Slice_Pre')
for tra_idx_pre in target_slices:
    pre_indices = get_exc_indices(lon_slice, tra_idx_pre, exc_dict)
    for tra_idx_post in target_slices:
        post_indices = get_exc_indices(lon_slice, tra_idx_post, exc_dict)
        num_synapses = get_synapse_counts(pre_indices, post_indices, adjacency_matrix)
        synapse_df.loc[tra_idx_pre, tra_idx_post] = num_synapses

# Visualize the synapse distribution
visualize_synapse_distribution(synapse_df, target_slices)

