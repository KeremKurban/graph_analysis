""" Workaround script used to switch from the ltr coordinate system to xyz for the hippocampus

This was used by armando and andras on 15-12-2018.
It is a custom script for hippocampus analysis only.
"""
import pandas as pd
import numpy as np

from bluepy.geometry.roi import Cube
import voxcell


LON = 0
TRA = 1
RAD = 2
AXES = [LON, TRA, RAD]


def _list_to_double_list(values):
    """ Return a list of values if values is not a list or tuple or array already """
    if len(values) and not isinstance(values[0], (list, tuple, np.ndarray)):
        return [values]
    return values


def _conversion_function(func):
    """ Ensures that to_convert is a double list and return [] if to_convert is empty """
    def wrapper(obj, to_convert, **kwargs):
        # obj is self for CoordinateQuery and to_convert is coords or positions or indices
        double_list = _list_to_double_list(to_convert)
        if len(double_list):
            return func(obj, double_list, **kwargs)
        else:
            return []
    return wrapper


class CoordinateQuery(object):
    def __init__(self, file_path):
        self.coords = CoordinateQuery._import_coordinate_file(file_path)
        self.raw = self.coords.raw
        self.min_max = [self.get_coord_min_max(axis) for axis in AXES]

    @staticmethod
    def _import_coordinate_file(file_path):
        """ Import the voxcell_data containing the voxels and the corresponding ltr values"""
        coord = voxcell.VoxelData.load_nrrd(file_path)
        new_raw = np.copy(coord.raw)
        mask = (new_raw[:, :, :, :] < 0) | (new_raw[:, :, :, :] > 1)
        new_raw[mask] = np.nan
        return coord.with_data(new_raw)

    def _verify_query_inputs(self, value_min, value_max, axis, strict=False):
        """ Check the query inputs for ranges query

        Args :
            value_min: the minimum value for query (float)
            value_max: the maximum value for query (float)
            axis: the axis corresponding to the query (int)
            strict: boolean value to define if value_min/max need to be inside the axis range
        """
        if value_min >= value_max:
            raise ValueError('min >= max for axis {}'.format(axis))
        if strict:
            if not self.min_max[axis][0] <= value_min < self.min_max[axis][1]:
                raise ValueError('value {} out of range for axis {}'.format(value_min, axis))
            if not self.min_max[axis][0] < value_max <= self.min_max[axis][1]:
                raise ValueError('value {} out of range for axis {}'.format(value_min, axis))

    def get_coord_min_max(self, axis):
        """ Return the min and max value for a given axis"""
        values = self.coords.raw[:, :, :, axis]
        return np.nanmin(values), np.nanmax(values)

    @_conversion_function
    def idx_to_ltr(self, indices):
        """ Convert voxel idx to ltr value using the voxcell_data provided in constructor

        Args:
            indices: list or numpy array of indices ([[idx1, idx2, idx3], [idx4, idx5, idx6], ...]))
        """
        return self.coords.raw[indices[:, LON], indices[:, TRA], indices[:, RAD]]

    @_conversion_function
    def idx_to_xyz(self, indices):
        """ Convert voxel idx to xyz value using the voxcell_data provided in constructor

        Args:
            indices: list or numpy array of indices ([[idx1, idx2, idx3], [idx4, idx5, idx6], ...]))

        Returns:
            the positions of all selected voxel in the xyz coordinate system
        """
        return self.coords.indices_to_positions(indices)

    @_conversion_function
    def xyz_to_ltr(self, positions):
        """ Convert xyz to ltr value using the voxcell_data provided in constructor

        Args:
            positions: list or numpy array of indices ([[x1, y1, z1], [x2, y2, z2], ...]))
        """
        indices = self.coords.positions_to_indices(positions)
        return self.idx_to_ltr(indices)

    @_conversion_function
    def xyz_to_idx(self, positions):
        """ Convert xyz to idx value using the voxcell_data provided in constructor

        Args:
            positions: list or numpy array of indices ([[x1, y1, z1], [x2, y2, z2], ...]))

        Returns:
            the indices of all selected voxel
        """
        return self.coords.positions_to_indices(positions)

    @_conversion_function
    def ltr_to_idx(self, coords, tol_increase=0.01):
        """ Convert ltr to idx value using the voxcell_data provided in constructor

        Args:
            coords: list or numpy array of ltr ([[l1, t1, r1], [l2, t2, r2], ...]))
            tol_increase: a value to perform researches of the closest possible value

        Returns:
            the indices of all selected voxel

        Notes:
            This one is a complex process. We do not have a bijection to go from ltr to xyz
            (implicitly to idx) and we just have a mapping of discrete xyz (or idx) values to ltr.
            So we need to perform a search for the closest value of ltr in the array and then
            return the corresponding voxel idx. This process is very inefficient.
        """
        indices = list()
        for coord in coords:
            c_tol = 0
            tmp_idx = np.empty((0, 0))
            while len(tmp_idx) == 0:
                c_tol += tol_increase
                tmp_idx = self.ltr_range_to_idx(coord[LON] - c_tol, coord[LON] + c_tol,
                                                coord[TRA] - c_tol, coord[TRA] + c_tol,
                                                coord[RAD] - c_tol, coord[RAD] + c_tol)
            ltr_values = self.coords.raw[tuple(tmp_idx.T)]
            ok_idx = tmp_idx[np.argmin(np.linalg.norm(ltr_values - np.array(coord), ord=1, axis=1))]
            indices.append(list(ok_idx))
        return indices

    @_conversion_function
    def ltr_to_xyz(self, coords, tol_increase=0.01):
        """ Convert ltr to idx value using the voxcell_data provided in constructor

        Args:
            coords: list or numpy array of ltr ([[l1, t1, r1], [l2, t2, r2], ...]))
            tol_increase: a value to perform researches of the closest possible value

         Returns:
            the positions of all selected voxel in the xyz coordinate system

        Notes:
            This one is a complex process. We do not have a bijection to go from ltr to xyz
            (implicitly to idx) and we just have a mapping of discrete xyz (or idx) values to ltr.
            So we need to perform a search for the closest value of ltr in the array and then
            return the corresponding voxel idx. This process is very inefficient.
        """
        idx = self.ltr_to_idx(coords, tol_increase=tol_increase)
        return self.idx_to_xyz(idx)

    def _get_range_mask(self, min_value, max_value, axis):
        """ Return a mask corresponding to the values inside min_value and max_value in axis """
        return (self.raw[:, :, :, axis] <= max_value) & (self.raw[:, :, :, axis] >= min_value)

    def ltr_range_to_idx(self, lon_min, lon_max, tra_min, tra_max, rad_min, rad_max, strict=False):
        """ Return all idx inside a ltr range

        Args:
            lon_min: the longitude min
            lon_max: the longitude max
            tra_min: the transverse min
            tra_max: the transverse max
            rad_min: the radial min
            rad_max: the radial max

        Returns:
            the indices of all selected voxel in the ltr coordinate system
        """
        self._verify_query_inputs(lon_min, lon_max, LON, strict=strict)
        self._verify_query_inputs(tra_min, tra_max, TRA, strict=strict)
        self._verify_query_inputs(rad_min, rad_max, RAD, strict=strict)
        lon_mask = self._get_range_mask(lon_min, lon_max, LON)
        tra_mask = self._get_range_mask(tra_min, tra_max, TRA)
        rad_mask = self._get_range_mask(rad_min, rad_max, RAD)
        idx = np.array(np.where(lon_mask & tra_mask & rad_mask)).T
        return idx

    def ltr_range_to_xyz(self, lon_min, lon_max, tra_min, tra_max, rad_min, rad_max):
        """ Return all xyz corresponding to a ltr range """
        idx = self.ltr_range_to_idx(lon_min, lon_max, tra_min, tra_max, rad_min, rad_max)
        return self.idx_to_xyz(idx)

    def ltr_iso_to_idx(self, center_value, tol, axis):
        """ Return all idx corresponding to an iso-value +/- a tolerance for a given axis

        Args:
            center_value: the center value in the ltr coordinate system
            tol: the thickness around the isovalue

        Returns:
            the indices of all selected voxel
        """
        up = center_value + tol
        down = center_value - tol
        mask = ((self.raw[:, :, :, axis] < up) & (self.raw[:, :, :, axis] > down))
        return np.array(np.where(mask)).T

    def ltr_iso_to_xyz(self, center_value, tol, axis):
        """ Return all xyz corresponding to an iso-value +/- a tolerance for a given axis

        Args:
             center_value: the center value for the in the ltr coordinate system
             tol: the thickness around the isovalue

        Returns:
            the positions of all selected voxel in the xyz coordinate system
        """
        return self.idx_to_xyz(self.ltr_iso_to_idx(center_value, tol, axis))

    def ltr_slices_to_xyz(self, gap_between_iso, tol, axis, min_value=0, max_value=1):
        """ Return all xyz corresponding to multiple iso-value +/- a tolerance for a given axis

         Args:
             gap_between_iso: distance between 2 slices in ltr coords
             tol: the thickness around the isovalue
             axis: the considered axis
             min_value: first value along axis for the slices
             max_value: last value along axis for the slices

        Returns:
            the positions of all selected voxel  in the xyz coordinate system
         """
        nb_iso = (max_value - min_value) / gap_between_iso
        values = np.linspace(min_value, max_value, nb_iso, endpoint=True)
        xyzs = [self.ltr_iso_to_xyz(value, tol, axis) for value in values]
        return np.concatenate(xyzs, axis=0)

    def convert_xyz_to_bounding_boxes(self, positions):
        """ Convert positions to bluepy bounding boxes """
        return [Cube(position, self.coords.voxel_dimensions[0]) for position in positions]

    def get_long_center_distance(self, lon_min, lon_max):
        """" Heuristic to compute the center longitudinal distance between two longitudes"""
        center = 0.5
        center_down = center - 0.01
        center_up = center + 0.01
        idx = self.ltr_range_to_idx(lon_min, lon_max,
                                    center_down, center_up,
                                    center_down, center_up)
        sel_idx = np.argsort(self.raw[tuple(idx.T)], axis=0)[:, 0]
        sorted_idx = idx[sel_idx]
        xyz = self.coords.indices_to_positions(sorted_idx)[::10, :]
        dist_tot = np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))
        return dist_tot

    def long_micro_meter_slice(self, long_micro, thickness_micro):
        """ Create a slice along the longitudinal axis with thickness defined in micro meter

        Args:
            long_micro: longitudinal value center of the slice defined in micro meter
            thickness_micro: thickness in micro meter

        Returns:
            min and max value which define the slice along the longitudinal axis
        """
        lon_min = self.min_max[LON][0]
        lon_max = self.min_max[LON][1]
        dist_tot = self.get_long_center_distance(lon_min, lon_max)
        thickness = (lon_max - lon_min) * ((thickness_micro * 0.5) / dist_tot)
        long = lon_min + (lon_max - lon_min) * (long_micro / dist_tot)
        return long - thickness, long + thickness


def _between_mask(series, min_value, max_value):
    """ a min < value < max mask for pandas DataFrame """
    return (series > min_value) & (series < max_value)


def enriched_cells_positions(circuit, coord_query):
    """ Function that enriched a position DataFrame from bluepy with ltr coordinates

    Args:
        circuit: the bluepy circuit you want to query
        coord_query: the CoordinateQuery object containing the VoxcellData with ltr coordinates

    Returns:
        A data frame with x, y, z, t, l, r coordinates for all cells in circuit
    """
    xyz = circuit.cells.positions()
    cols = coord_query.xyz_to_ltr(xyz.values)
    xyz['l'] = pd.Series(cols[:, LON], index=xyz.index)
    xyz['t'] = pd.Series(cols[:, TRA], index=xyz.index)
    xyz['r'] = pd.Series(cols[:, RAD], index=xyz.index)
    return xyz


def query_enriched_positions(xyz_ltr, lon_min, lon_max, tra_min, tra_max, rad_min, rad_max):
    """ Query the enriched positions

    Args:
        xyz_ltr: the enriched positions (created with the enriched_cells_positions function)
        lon_min: the longitude min
        lon_max: the longitude max
        tra_min: the transverse min
        tra_max: the transverse max
        rad_min: the radial min
        rad_max: the radial max

    Returns:
        the ids of objects which match the query
    """
    if {'l', 't', 'r'} - set(xyz_ltr):
        raise KeyError('The panda DataFrame needs to have "l", "t" and "r" columns')
    l_mask = _between_mask(xyz_ltr['l'], lon_min, lon_max)
    t_mask = _between_mask(xyz_ltr['t'], tra_min, tra_max)
    r_mask = _between_mask(xyz_ltr['r'], rad_min, rad_max)
    ids = xyz_ltr.index.values[l_mask & t_mask & r_mask]
    return ids
