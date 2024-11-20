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
import seaborn as sns
from tqdm import tqdm
import logging
import shutil 

# Part 1: Extract submatrices along LTR

# circ_path = '/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/CircuitConfig'
# # if generated already, use the generated masks
# volumetric_slices_LON = '/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/scripts/5_slice_generation/nrrd_masks_serially_generated_but_not_used/'

# slice_thickness = 400 # um
# usertarget_path = f'slices{slice_thickness}.target'
# circuit = Circuit(circ_path)
# ATLAS_DIR = Path("/gpfs/bbp.cscs.ch/project/proj112/entities/atlas/20211004_BioM/")
# CIRCUIT_DIR = Path('/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/')
# c = Circuit((CIRCUIT_DIR / 'CircuitConfig').as_posix())
# atlas = Atlas.open(ATLAS_DIR.as_posix())
# orientation = atlas.load_data("orientation")

# # load the atlas in the coordinatequery
# file_path = '/gpfs/bbp.cscs.ch/project/proj112/entities/atlas/20211004_BioM/coordinates.nrrd'

# q = CoordinateQuery(file_path)
# # Enriched your ltr positions in the positions df from bluepy
# # xyz_ltr = enriched_cells_positions(circuit, q)


# # generate TRA slices across all CA1
# tra_dt = 0.1
# num_TRA_slices = int(1/tra_dt)

# tra_slices = np.arange(0,1,tra_dt)
# for idx, tra_min in enumerate(log_progress(tra_slices, every=1)):
#     tra_max = tra_min + tra_dt
#     cur_slice_mask = q._get_range_mask(min_value=tra_min, max_value=tra_max, axis=TRA)
#     orientation.with_data(np.asarray(cur_slice_mask, dtype=np.uint8)).save_nrrd(f"tra_masks/slice_tra__{idx+1}__{len(tra_slices)}.nrrd")

# num_lon_slices = len(os.listdir(volumetric_slices_LON))

# # intersect LON slices with TRA slice and save each submask
# for lon_slice_idx in np.arange(1,num_lon_slices+1):
#     # if lon_slice_idx not in [10]:
#     #     continue # skip for now except for 3
#     for tra_slice_idx in np.arange(1,len(tra_slices)+1):
#         tra_slice = VoxelData.load_nrrd(f'tra_masks/slice_tra__{tra_slice_idx}__10.nrrd')
#         lon_slice = VoxelData.load_nrrd(f'{volumetric_slices_LON}/slice400_{lon_slice_idx}.nrrd')
#         print(f'tra_slice: {tra_slice_idx} | lon_slice: {lon_slice_idx}')
        
#         interseciton_slice = tra_slice.raw & lon_slice.raw
#         print(f"lon_tra_intersection_masks/slice_LON_{lon_slice_idx}_TRA_{tra_slice_idx}.nrrd")
#         orientation.with_data(np.asarray(interseciton_slice, dtype=np.uint8)).save_nrrd(f"lon_tra_intersection_masks/slice_LON_{lon_slice_idx}_TRA_{tra_slice_idx}.nrrd")


# make class out of the above code but dont import volumetric_slices_LON. instead generate them with  q._get_range_mask.

class SubmatrixGenerator:
    def __init__(self, coordinate_query: CoordinateQuery, ATLAS_DIR: str):
        self.q = coordinate_query
        self.atlas = Atlas.open(ATLAS_DIR)
        self.orientation = self.atlas.load_data("orientation")

    # geneate a method to check if the output directory exists already
    def _check_output_dir(self, output_dir,force_overwrite):
        if os.path.exists(output_dir):
            if force_overwrite:
                logging.warning(f"Output directory {output_dir} already exists. Overwriting.")
                shutil.rmtree(output_dir)
                os.makedirs(output_dir, exist_ok=True)
                return False
            else:
                logging.warning(f"Output directory {output_dir} already exists. Please check its contents.")
                return True
        else:
            os.makedirs(output_dir, exist_ok=True)
            return False
        
    def generate_lon_slices(self,lon_slice_thickness,save_dir,force_overwrite=False):
        logging.info(f"Generating slices with thickness: {slice_thickness}")
        subdirectory = f"{save_dir}/lon"
        # check if the output files exist already
        dir_exists = self._check_output_dir(subdirectory,force_overwrite)
        if dir_exists:
            return
    
        slice_id = 1
        max_lon_distance = int(q.get_long_center_distance(q.min_max[LON][0], q.min_max[LON][1]))
        slices = np.arange(0, max_lon_distance, lon_slice_thickness)
        self.num_lon_slices = len(slices)
        for l in tqdm(slices, desc="LON slices"):
            l_min, l_max = q.long_micro_meter_slice(l, lon_slice_thickness) # relative coordinate
            percentage = (l_min + l_max)/2
            # logging.info(f"l:{l}, l_min:{l_min}, l_max:{l_max}, percentage={percentage}")
            cur_slice_mask = q._get_range_mask(min_value=l_min, max_value=l_max, axis=LON)

            self.orientation.with_data(np.asarray(cur_slice_mask, dtype=np.uint8)).save_nrrd(f"{subdirectory}/slice_lon_{slice_thickness}_{slice_id}.nrrd")
            slice_id += 1

    def generate_tra_slices(self,tra_dt,save_dir,force_overwrite=False):
        self.tra_dt = tra_dt
        self.num_TRA_slices = int(1/tra_dt)
        logging.info(f"Generating {self.num_TRA_slices} across transverse axis")

        subdirectory = f"{save_dir}/tra"
        dir_exists = self._check_output_dir(subdirectory,force_overwrite)
        if dir_exists:
            return

        tra_slices = np.arange(0,1,tra_dt)
        for idx, tra_min in tqdm(enumerate(tra_slices), desc="TRA slices"):
            tra_max = tra_min + tra_dt
            cur_slice_mask = q._get_range_mask(min_value=tra_min, max_value=tra_max, axis=TRA)
            self.orientation.with_data(np.asarray(cur_slice_mask, dtype=np.uint8)).save_nrrd(f"{subdirectory}/slice_tra_{idx+1}__{len(tra_slices)}.nrrd")

    def generate_lon_tra_intersection_slices(self,save_dir,force_overwrite=False):
        logging.info(f"Generating LON-TRA intersection slices")
        tra_slices = np.arange(0,1,self.tra_dt)
        # count files under savedir/lon that start with slice_lon using regex        
        file_names = os.listdir(f'{save_dir}/lon')
        num_lon_slices = len([i for i in file_names if i.endswith('.nrrd')])

        subdirectory = f"{save_dir}/lon_tra_intersection"
        dir_exists = self._check_output_dir(subdirectory,force_overwrite)
        if dir_exists:
            return

        for lon_slice_idx in tqdm(np.arange(1,num_lon_slices+1), desc="LON-TRA intersections"): # we saved lon slices from slice1
            for tra_slice_idx in np.arange(1,len(tra_slices)+1):
                tra_slice = VoxelData.load_nrrd(f'{save_dir}/tra/slice_tra_{tra_slice_idx}__{len(tra_slices)}.nrrd')
                lon_slice = VoxelData.load_nrrd(f'{save_dir}/lon/slice_lon_{slice_thickness}_{lon_slice_idx}.nrrd')
                # logging.info(f'tra_slice: {tra_slice_idx} | lon_slice: {lon_slice_idx}')
                
                interseciton_slice = tra_slice.raw & lon_slice.raw
                self.orientation.with_data(np.asarray(interseciton_slice, dtype=np.uint8)).save_nrrd(f"{subdirectory}/slice_LON_{lon_slice_idx}_TRA_{tra_slice_idx}.nrrd")
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ATLAS_DIR = "/gpfs/bbp.cscs.ch/project/proj112/entities/atlas/20211004_BioM/"
    COORDINATES_DIR = f'{ATLAS_DIR}/coordinates.nrrd'
    save_dir = "/gpfs/bbp.cscs.ch/project/proj112/home/kurban/topology_paper/volumetric_slices"

    q = CoordinateQuery(COORDINATES_DIR)
    submatrix_generator = SubmatrixGenerator(q, ATLAS_DIR)
    force_compute = True
    # generate lon slices
    slice_thickness = 400
    submatrix_generator.generate_lon_slices(slice_thickness,save_dir,)

    # generate tra slices
    tra_dt = 0.1
    submatrix_generator.generate_tra_slices(tra_dt,save_dir)

    # generate intersection slices
    submatrix_generator.generate_lon_tra_intersection_slices(save_dir,force_compute)