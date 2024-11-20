import voxcell
from voxcell import CellCollection, VoxelData, RegionMap
import numpy as np
import requests
from matplotlib import cm
import matplotlib
import pandas as pd
from bluepy import Circuit
from bluepy.geometry.roi import Cube
from coordinate_query import CoordinateQuery, query_enriched_positions, LON, TRA, RAD
import logging
from tqdm import tqdm
from pathlib import Path
from voxcell.nexus.voxelbrain import Atlas
from bluepy import Circuit
from coordinate_query import CoordinateQuery, enriched_cells_positions, query_enriched_positions, LON, TRA, RAD
from log_progress import log_progress
import os
from voxcell import VoxelData
import matplotlib.pyplot as plt
from scipy import sparse
import seaborn as sns

def get_gids_in_voxel(mask,circuit):
    in_target = mask.lookup(circuit.cells.get(properties=list("xyz")).values)
    gids_in_target = circuit.cells.get().index.values[np.where(in_target==1)]
    return gids_in_target

class InformationFlow:
    def __init__(self,adj_path:str,longit_slice:int,intersection_masks_dir:str,circuit_dir:str):
        self.adj_path = adj_path
        self.adj = sparse.load_npz(adj_path)
        self.c = Circuit(f'{circuit_dir}/CircuitConfig')
        self.tra_dt = 0.1 # TODO: this value should be inherited from downsample.py, otherwise it is not consistent
        self.num_TRA_slices = int(1/self.tra_dt)
        self.lon_i = longit_slice
        self.exc_dict = {}
        self.target_slices = range(1,self.num_TRA_slices+1)
        self.tra_df = pd.DataFrame()
        self.tra_df = self.tra_df.rename_axis('Transverse_Slice_Pre')
        self.intersection_masks_dir = intersection_masks_dir

    def get_gids_in_voxel(self,mask,circuit):
        in_target = mask.lookup(circuit.cells.get(properties=list("xyz")).values)
        gids_in_target = circuit.cells.get().index.values[np.where(in_target==1)]
        return gids_in_target

    def get_exc_dict(self):
        for tra_i in np.arange(1,self.num_TRA_slices+1,1):
            int_mask = Atlas.open(self.intersection_masks_dir).load_data(f'slice_LON_{self.lon_i}_TRA_{tra_i}')
            mask_gids = self.get_gids_in_voxel(int_mask,self.c)
            exc_mask_gids = np.intersect1d(mask_gids,self.c.cells.ids('SP_PC'))
            logging.info(f'There are {len(exc_mask_gids)}/{len(mask_gids)} SP_PC cells in slice {self.lon_i}.{tra_i}')
            self.exc_dict[f'{self.lon_i}.{tra_i}'] = exc_mask_gids

    def get_tra_df(self,**kwargs):
        for tra_idx_pre in self.target_slices:
            for tra_idx_post in self.target_slices:
                pre_indices = self.exc_dict[f'{self.lon_i}.{tra_idx_pre}']-1
                post_indices = self.exc_dict[f'{self.lon_i}.{tra_idx_post}']-1
                temp = self.adj[pre_indices]
                temp = temp[:,post_indices]
                num_syns = int(temp.sum())
                self.tra_df.loc[tra_idx_pre,tra_idx_post] = num_syns
        #check if kwargs has save_parent_dir
        save_parent_dir = kwargs.get('save_parent_dir',None)
        if save_parent_dir:
            subdir = f'{save_parent_dir}/raw'
            os.makedirs(subdir,exist_ok=True)
            self.tra_df.to_csv(f'{subdir}/synapse_flow_tra_{self.lon_i}.csv')

    def plot_tra_df(self,save_parent_dir=None):
        sns.heatmap(self.tra_df)
        plt.show()
        if save_parent_dir:
            os.makedirs(save_parent_dir,exist_ok=True)
            plt.savefig(f'{save_parent_dir}/tra_{self.lon_i}.png')

    def run(self,**kwargs):
        self.get_exc_dict()
        self.get_tra_df(**kwargs)
        self.plot_tra_df(**kwargs)

if __name__ == '__main__':
    #set to debug
    logging.basicConfig(level=logging.INFO)

    CIRCUIT_DIR = Path('/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/')
    adj_path = '/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/data/ca1_synaptome.npz'
    intersection_masks_dir = 'volumetric_slices/lon_tra_intersection'

    lon_file_names = os.listdir(f'volumetric_slices/lon')
    num_lon_slices = len([i for i in lon_file_names if i.endswith('.nrrd')])

    for lon_i in range(1,num_lon_slices+1):
        info_flow = InformationFlow(adj_path,
                                    longit_slice=lon_i,
                                    intersection_masks_dir=intersection_masks_dir,
                                    circuit_dir=CIRCUIT_DIR)
        info_flow.run(save_parent_dir='data/information_flow')
        plt.close('all')