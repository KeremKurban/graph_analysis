'''
Analysis of synaptic connections across hippocampal axis

Get masks along LTR axis
Filter GIDs within the masks
Get number of synapses from volumes across the coordinate axis

'''

import logging
import pathlib
from re import S
import voxcell
from voxcell import CellCollection, VoxelData, RegionMap
# from brainbuilder.utils import bbp
import numpy as np
import requests
from matplotlib import cm
import matplotlib
import pandas as pd
import numpy.linalg as la
from bluepy.enums import Cell
from bluepy import Circuit
from bluepy.geometry.roi import Cube
from coordinate_query import CoordinateQuery, query_enriched_positions, LON, TRA, RAD
import math
from log_progress import log_progress
from pathlib import Path
from voxcell.nexus.voxelbrain import Atlas
from bluepy import Circuit
from coordinate_query import CoordinateQuery, enriched_cells_positions, query_enriched_positions, LON, TRA, RAD
from log_progress import log_progress
import os
from voxcell import VoxelData
import matplotlib.pyplot as plt
from scipy import sparse


# class for storing atlas and circuit data
class CA1NetworkProperties:
    def __init__(self):
        self.CIRCUIT_PATH = Path('/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/CircuitConfig')
        self.ATLAS_DIR = Path("/gpfs/bbp.cscs.ch/project/proj112/entities/atlas/20211004_BioM/")
        self.ADJ_PATH = Path('/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/data/ca1_synaptome.npz')
        self.volumentric_slice_paths = {
            LON: '/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/scripts/5_slice_generation/nrrd_masks_serially_generated_but_not_used/',
            TRA: None,
            RAD: None
        }


        self.circuit = Circuit(self.CIRCUIT_PATH)
        self.atlas = Atlas.open(self.ATLAS_DIR.as_posix())
        self.orientation = atlas.load_data("orientation")
        coords_dir = self.ATLAS_DIR/'coordinates.nrrd'
        q = CoordinateQuery(coords_dir.as_posix())
        # Enriched your ltr positions in the positions df from bluepy
        self.xyz_ltr = enriched_cells_positions(self.circuit, self.q)

    def _get_gids_in_voxel(mask,circuit):
        in_target = mask.lookup(circuit.cells.get(properties=list("xyz")).values)
        gids_in_target = circuit.cells.get().index.values[np.where(in_target==1)]
        return gids_in_target

    def extract_volumentric_slices_along_axis(self,t_min,t_max,axis,save_dir,dt=0.1,):
        
        tra_slice_mask = self.q._get_range_mask(min_value=t_min, max_value=t_max, axis=axis)

        self.tra_slices = np.arange(0,1,dt)
        for idx, tra_min in enumerate(log_progress(self.tra_slices, every=1)):
            tra_max = tra_min + dt
            cur_slice_mask = self.q._get_range_mask(min_value=tra_min, max_value=tra_max, axis=axis)
            is_global = os.path.isabs(save_dir)
            if is_global:
                logging.info(f"Saving volumetric slices to global path: {save_dir}")
                self.volumentric_slice_paths[TRA] = f"{save_dir}"
                self.orientation.with_data(np.asarray(cur_slice_mask, dtype=np.uint8)).save_nrrd(f"{save_dir}/slice_tra__{idx+1}__{len(tra_slices)}.nrrd")
            else:
                # get current dir
                logging.info(f"Saving volumetric slices to local path: {cur_dir}/{save_dir}")
                cur_dir = os.getcwd()
                self.volumentric_slice_paths[TRA] = f"{cur_dir}/{save_dir}"
                self.orientation.with_data(np.asarray(cur_slice_mask, dtype=np.uint8)).save_nrrd(f"{cur_dir}/{save_dir}/slice_tra__{idx+1}__{len(tra_slices)}.nrrd")

    def intersect_slices(self, LON_slice_numbers:list):
        num_longit_slices = len(os.listdir(self.volumentric_slice_paths[LON]))

        # intersect LON slices with TRA slice
        for lon_slice_idx in np.arange(1,num_longit_slices+1):
            if lon_slice_idx not in LON_slice_numbers:
                continue 
            for tra_slice_idx in np.arange(1,len(self.tra_slices)+1):
                tra_slice = VoxelData.load_nrrd(f'tra_masks/slice_tra__{tra_slice_idx}__10.nrrd')
                lon_slice = VoxelData.load_nrrd(f'{self.volumentric_slice_paths[LON]}/slice400_{lon_slice_idx}.nrrd')
                print(f'tra_slice: {tra_slice_idx} | lon_slice: {lon_slice_idx}')
                
                interseciton_slice = tra_slice.raw & lon_slice.raw
                print(f"lon_tra_intersection_masks/slice_LON_{lon_slice_idx}_TRA_{tra_slice_idx}.nrrd")
                self.orientation.with_data(np.asarray(interseciton_slice, dtype=np.uint8)).save_nrrd(f"lon_tra_intersection_masks/slice_LON_{lon_slice_idx}_TRA_{tra_slice_idx}.nrrd")

    
    # get synapses from slice idx to idx+1

    def extract_gids_from_voxel_mask(self, voxel_mask_path:str, mtype:str):
        path = Path(voxel_mask_path)
        int_mask = Atlas.open(path.parent.as_posix()).load_data(path.name)
        mask_gids = self._get_gids_in_voxel(int_mask,self.circuit)
        mask_gids = np.intersect1d(mask_gids,self.circuit.cells.ids(mtype))
        return mask_gids

    def get_information_flow_across_transverse_axis(self,lon_slice=10,num_tra_slices=10):
  
        exc_dict = {}
        for tra_i in np.arange(1,num_tra_slices+1,1):
            int_mask = Atlas.open('lon_tra_intersection_masks').load_data(f'slice_LON_{lon_slice}_TRA_{tra_i}')
            mask_gids = self._get_gids_in_voxel(int_mask,self.circuit)
            exc_mask_gids = np.intersect1d(mask_gids,self.circuit.cells.ids('SP_PC'))
            print(f'There are {len(exc_mask_gids)}/{len(mask_gids)} SP_PC cells in slice {lon_slice}.{tra_i}')
            exc_dict[f'{lon_slice}.{tra_i}'] = exc_mask_gids

            
        adj = sparse.load_npz(self.ADJ_PATH.as_posix())
        #synapse counts from proximal to distal as a measure of information flow
        transverse_slices = range(1,num_tra_slices+1)
        tra_df = pd.DataFrame()
        tra_df = tra_df.rename_axis('Transverse_Slice_Pre')
        for tra_idx_pre in transverse_slices:
            for tra_idx_post in transverse_slices:
                pre_indices = exc_dict[f'{lon_slice}.{tra_idx_pre}']-1
                post_indices = exc_dict[f'{lon_slice}.{tra_idx_post}']-1
                temp = adj[pre_indices]
                temp = temp[:,post_indices]
                num_syns = int(temp.sum())
                tra_df.loc[tra_idx_pre,tra_idx_post] = num_syns
        
        return tra_df
    


if __name__ == "__main__":
    
    var = CA1NetworkProperties()
    
    # longit_slice_name = 'slice10'
    # num_divisions_across_LON = 50
    # num_divisions_across_TRA = 10

    #extract volumetric slices
    # savedir = './data/volumetric_slices'
    # lon_slices = var.extract_volumetric_slices(LON,num_divisions_across_LON,savedir=savedir)
    # tra_slices = var.extract_volumetric_slices(TRA,num_divisions_across_TRA,savedir=savedir)

    # intesection = var.intersect_slices(longit_slice_name,tra_slices)
    # intersection_gids = var.extract_gids_from_voxel_mask(intesection,'SP_PC')

    tra_df = var.get_information_flow_across_transverse_axis('slice10',divide_into=10)

    import seaborn as sns
    sns.heatmap(tra_df)