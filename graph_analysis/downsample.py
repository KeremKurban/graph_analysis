import os
import shutil
import logging
from abc import ABC, abstractmethod
from tqdm import tqdm
import logging
from voxcell import VoxelData
from voxcell.nexus.voxelbrain import Atlas
import numpy as np
import bluepysnap 

class CoordinateQuery:
    '''
    Class to query the coordinates of the atlas. Has conversion functions across local coordinate frames.
    '''
    pass


class VolumetricSubmatrixGenerator(ABC):
    def __init__(self, coordinate_query, ATLAS_DIR):
        self.q = coordinate_query
        self.atlas = Atlas.open(ATLAS_DIR)
        self.orientation = self.atlas.load_data("orientation")

    def _check_output_dir(self, output_dir, force_overwrite):
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

    @abstractmethod
    def generate_slices(self, slice_thickness, save_dir, force_overwrite=False):
        pass

class RatCA1SubmatrixGenerator(VolumetricSubmatrixGenerator):
    
    def generate_slices(self, slice_thickness, save_dir, force_overwrite=False):
        logging.info(f"Generating slices with thickness: {slice_thickness}")
        subdirectory = f"{save_dir}/lon"
        dir_exists = self._check_output_dir(subdirectory, force_overwrite)
        if dir_exists:
            return

        # Rest of the implementation specific to rat CA1 region

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

class NodeSetExtractor:
    def __init__(self,circuit:bluepysnap.Circuit) -> None:
        self.circuit = circuit
        self.nodes = None
        
if __name__ == "__main__":
    from coordinate_query import CoordinateQuery, enriched_cells_positions, query_enriched_positions, LON, TRA, RAD

    logging.basicConfig(level=logging.INFO)
    ATLAS_DIR = "/gpfs/bbp.cscs.ch/project/proj112/entities/atlas/20211004_BioM/"
    COORDINATES_DIR = f'{ATLAS_DIR}/coordinates.nrrd'
    # save_dir = "/gpfs/bbp.cscs.ch/project/proj112/home/kurban/topology_paper/volumetric_slices"
    save_dir = f"{os.getcwd()}/volumetric_slices"
    q = CoordinateQuery(COORDINATES_DIR)
    submatrix_generator = RatCA1SubmatrixGenerator(q, ATLAS_DIR)
    force_compute = False
    # generate lon slices
    slice_thickness = 400
    submatrix_generator.generate_lon_slices(slice_thickness,save_dir,force_overwrite=force_compute)

    # generate tra slices
    tra_dt = 0.1
    submatrix_generator.generate_tra_slices(tra_dt,save_dir,force_overwrite=force_compute)

    # generate intersection slices
    submatrix_generator.generate_lon_tra_intersection_slices(save_dir,force_overwrite=force_compute)