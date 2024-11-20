import numpy as np
import pandas as pd 
import bluepy
from pathlib import Path
import logging
from tqdm import tqdm
from voxcell import VoxelData
from voxcell.nexus.voxelbrain import Atlas
from shutil import copyfile

import voxcell
from voxcell import CellCollection, VoxelData, RegionMap
from brainbuilder.utils import bbp
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


def get_voxel_positions(atlas_dir):
    '''Loads an nrrd file and returns a pandas df with indices as voxel coordinates and positions as values'''
    atlas = Atlas.open(atlas_dir)
    orientation = atlas.load_data("orientation")
    ndindex = list(np.ndindex(orientation.shape))
    pos_per_voxel = orientation.indices_to_positions(ndindex)
    return pos_per_voxel

def mask_voxels_in_cylinder(of_radius, in_atlas, centered_at, with_axis_along):
    """..."""
    voxel_positions = get_voxel_positions(in_atlas)
    c = np.cross(voxel_positions - centered_at, with_axis_along)
    distances = np.linalg.norm(c,axis=1) / np.linalg.norm(with_axis_along)
    return distances <= of_radius

def get_gids_from_mask(mask_name,circuit_path):
    c = bluepy.Circuit(circuit_path + '/CircuitConfig_struct')
    atlas = Atlas.open('.')
    mask = atlas.load_data(mask_name)
    in_target = mask.lookup(c.cells.get(properties=list("xyz")).values)
    gids_in_target = c.cells.get().index.values[np.where(in_target==1)] #assuming mask is 0/1
    return gids_in_target

def update_user_target(filename, gids, target_name, notes=''):
    with open(filename, 'a') as f:
        f.write(notes)
        f.write('\n')
        f.write('Target Cell %s\n{\n' % target_name)
        f.write(' '.join(['a%d' % gid for gid in gids]))
        f.write('\n}\n\n')


def get_gids_in_voxel(mask,circuit):
    in_target = mask.lookup(circuit.cells.get(properties=list("xyz")).values)
    gids_in_target = circuit.cells.get().index.values[np.where(in_target==1)]
    return gids_in_target


