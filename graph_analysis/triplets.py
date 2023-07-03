from igraph import *
from bluepy.enums import Cell
from scipy import sparse
import bluepy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)


class CA1MotifCalculator:
    '''
    Motif calculator class for calculating positional motifs in a circuit.

    Parameters
    ----------
    target : str
        Circut target name
    adj_file : str
        Path to the adjacency matrix file
    check_mtypes : list
        List of mtypes to check for positional motifs
        e.g. ['SP_PC','SP_CCKBC','SP_Ivy']
    CM : np.array
        Connectivity matrix for FFI motif given in check_mtypes 
        e.g. np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    exc_mtypes : list
        List of excitatory mtypes
        e.g. ['SP_PC']
    
    '''

    def __init__(self, target, adj_file, check_mtypes, CM, exc_mtypes=['SP_PC']):
        self.target = target
        self.adj_file = adj_file
        self.CM = CM  # Connectivity matrix for FFI motif [SC,INT_mtype,SP_PC] np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
        self.exc_mtypes = exc_mtypes
        self.check_mtypes = check_mtypes # i.e. ['SP_PC','SP_CCKBC','SP_Ivy']
        # order exc first!!
        self.mtypes_order = ['SP_PC', 'SLM_PPA', 'SO_BP', 'SO_BS', 'SO_OLM', 'SO_Tri', 'SP_AA', 'SP_BS', 'SP_CCKBC', 'SP_Ivy', 'SP_PVBC', 'SR_SCA'] # TODO: fetch from extraction
        if not os.path.isfile(adj_file):
            raise FileNotFoundError(f"The adjacency matrix file {adj_file} does not exist.")
        
        self.circuit_path = '/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/CircuitConfig'
        if not os.path.isfile(self.circuit_path):
            raise FileNotFoundError(f"The circuit file {self.circuit_path} does not exist.")

        #TODO: SONATA
        self.c = bluepy.Circuit(self.circuit_path) 
        self.mtypes_by_gid = np.array(self.c.cells.get(self.target).mtype.values)
        self.mtypes = sorted(list(self.c.cells.mtypes))
        self.inh_mtypes = self.mtypes.copy()
        self.inh_mtypes.remove(self.exc_mtypes[0])  # TODO: needs to be tuned for multiple exc circuits

    def evaluate(self, x, boolean_value, to_dense=True):
        '''
        If the boolean value is true or 1 , returns the matrix as it is . If 0 or false, returns 'logical not' of the matrix
        '''
        if to_dense:
            return np.where(boolean_value, x.todense(), x.todense() == 0)
        else:
            return False  # not implemented

    def load_adjacency_matrix(self,return_num_projections=True):
        '''
        Loads and returns squarized adjacency matrix from given scipy sparse matrix path.
        '''
        all2ca1_adj = sparse.load_npz(self.adj_file)
        all2SC = sparse.csr_matrix((all2ca1_adj.shape[0], all2ca1_adj.shape[0] - all2ca1_adj.shape[1]))
        all2all_adj = sparse.hstack([all2ca1_adj, all2SC])
        logging.debug(f"all2all_adj shape: {all2all_adj.shape}")
        self.num_external_fibers = all2ca1_adj.shape[0] - all2ca1_adj.shape[1]
        return all2all_adj

    def get_number_of_cells(self, square_matrix,num_projection_samples=False):
        #TODO: SONATA
        # assert matrix is square
        np.testing.assert_equal(square_matrix.shape[0], square_matrix.shape[1])

        num_target_gids = len(self.c.cells.ids(self.target))
        num_exc_cells = len(self.c.cells.ids(group={Cell.MTYPE: "SP_PC", '$target': self.target}))
        num_inh_cells = num_target_gids - num_exc_cells
        if num_projection_samples:
            num_SC_sample = num_projection_samples
        else:
            num_SC_sample = self.num_external_fibers  # could be a downsample i.e. first 100

        logging.debug(f"num SC sample {num_SC_sample}")
        return num_target_gids, num_exc_cells, num_inh_cells, num_SC_sample

    def get_mtypes_start_index(self):
        # mtypes_order = ['SP_PC', 'SLM_PPA', 'SO_BP', 'SO_BS', 'SO_OLM', 'SO_Tri', 'SP_AA', 'SP_BS',
        #         'SP_CCKBC', 'SP_Ivy', 'SP_PVBC', 'SR_SCA']
        target_data = self.c.cells.get(self.target)
        target_mtype_counts = pd.DataFrame(np.unique(target_data.mtype, return_counts=True)[1],
                                        index=np.unique(target_data.mtype, return_counts=True)[0], columns=['count'])
        target_mtype_counts = target_mtype_counts.reindex(index=self.mtypes_order)
        cumsum_mtypes_indices = np.hstack([0, np.cumsum(target_mtype_counts.values)])
        return cumsum_mtypes_indices

    def motif_calculation(self, all2all_adj, num_target_gids, num_exc_cells, num_SC_sample):
        # sc_to_pc = all2all_adj[num_target_gids:num_target_gids + num_SC_sample, :num_exc_cells]  # sc x ca1_exc
        # pc_to_sc = all2all_adj[:num_exc_cells, num_target_gids:num_target_gids + num_SC_sample]  # sc x ca1_exc

        mt_index_dict = {}
        assert np.all(np.isin(self.check_mtypes, [*self.mtypes,'SC','INT','INH']))

        cumsum_mtypes_indices = self.get_mtypes_start_index()
        logging.debug(f"cumsum_mtypes_indices: {cumsum_mtypes_indices}")

        for i in self.check_mtypes:
            if len(np.intersect1d(['INH', 'INT'], [i])) > 0:
                st_index, end_index = cumsum_mtypes_indices[1], cumsum_mtypes_indices[-1] # TODO: fix. hardwired for CA1 with one exc mtype at the 0th index
            elif i == 'SC':
                st_index, end_index = cumsum_mtypes_indices[-1], cumsum_mtypes_indices[-1] + num_SC_sample
            else:
                mt_index_in_ordering = np.where(np.isin(self.mtypes_order, i))[0][0]
                st_index, end_index = cumsum_mtypes_indices[mt_index_in_ordering], cumsum_mtypes_indices[mt_index_in_ordering + 1]
            
            logging.debug(f"{i} st_index: {st_index}, end_index: {end_index}")
            mt_index_dict[i] = [st_index, end_index]

        # print the following only in the debug mode
        mt1_to_mt2 = all2all_adj[mt_index_dict[check_mtypes[0]][0]:mt_index_dict[check_mtypes[0]][1], 
                                mt_index_dict[check_mtypes[1]][0]:mt_index_dict[check_mtypes[1]][1]] 
        mt2_to_mt1 = all2all_adj[mt_index_dict[check_mtypes[1]][0]:mt_index_dict[check_mtypes[1]][1],
                                mt_index_dict[check_mtypes[0]][0]:mt_index_dict[check_mtypes[0]][1]] 

        mt1_to_mt3 = all2all_adj[mt_index_dict[check_mtypes[0]][0]:mt_index_dict[check_mtypes[0]][1], 
                                mt_index_dict[check_mtypes[2]][0]:mt_index_dict[check_mtypes[2]][1]] 
        mt3_to_mt1 = all2all_adj[mt_index_dict[check_mtypes[2]][0]:mt_index_dict[check_mtypes[2]][1],
                                mt_index_dict[check_mtypes[0]][0]:mt_index_dict[check_mtypes[0]][1]] 

        mt2_to_mt3 = all2all_adj[mt_index_dict[check_mtypes[1]][0]:mt_index_dict[check_mtypes[1]][1], 
                                mt_index_dict[check_mtypes[2]][0]:mt_index_dict[check_mtypes[2]][1]] 
        mt3_to_mt2 = all2all_adj[mt_index_dict[check_mtypes[2]][0]:mt_index_dict[check_mtypes[2]][1],
                                mt_index_dict[check_mtypes[1]][0]:mt_index_dict[check_mtypes[1]][1]] 


        A2 = self.evaluate(mt1_to_mt3, CM[0, 2]) * self.evaluate(mt3_to_mt1.transpose(), CM[2, 0])
        B2 = self.evaluate(mt1_to_mt2, CM[0, 1]) * self.evaluate(mt2_to_mt1.transpose(), CM[1, 0])
        C2 = self.evaluate(mt2_to_mt3, CM[1, 2]) * self.evaluate(mt3_to_mt2.transpose(), CM[2, 1])

        M = np.dot(B2.T, A2) * C2

        logging.info(f'Total instances of given motif ({MotifReader().matrix_to_name(CM)}) within the given adj matrix: {int(np.sum(M))}')
        M_sparse = sparse.csc_matrix(M)
        # save = input('Do you want to save the motif matrix? (y/n)')
        # if save == 'y':
        #     sparse.save_npz(f'{target}_triplets_w_{check_mtypes[0]}-{check_mtypes[1]}-{check_mtypes[2]}.npz',M_sparse)
        return int(np.sum(M))

    def get_positions(self, gids):
        gid_positions = self.c.cells.positions(gids)

    def count_motifs(self,**kwargs):
        all2all_adj = self.load_adjacency_matrix()
        num_target_gids, num_exc_cells, num_inh_cells, num_SC_sample = self.get_number_of_cells(all2all_adj, **kwargs)
        num_motifs = self.motif_calculation(all2all_adj, num_target_gids, num_exc_cells, num_SC_sample)
        return num_motifs
    
class MotifReader:
    def __init__(self):

        self.motifs = [
            {'name': '-C','gal_index':0, 'matrix': np.array([[0,0,0], [0,0,0], [0,0,0]])},
            {'name': '-B','gal_index':1, 'matrix': np.array([[0,1,0], [0,0,0], [0,0,0]])},
            {'name': '-A','gal_index':2, 'matrix': np.array([[0,1,0], [1,0,0], [0,0,0]])},
            {'name': 'A', 'gal_index':4, 'matrix': np.array([[0,0,0], [1,0,0], [1,0,0]])},
            {'name': 'B', 'gal_index':3, 'matrix': np.array([[0,0,0], [1,0,0], [0,1,0]])},
            {'name': 'C', 'gal_index':5, 'matrix': np.array([[0,0,0], [1,0,1], [1,0,0]])},
            {'name': 'D', 'gal_index':7, 'matrix': np.array([[0,1,0], [1,0,0], [1,0,0]])},
            {'name': 'E', 'gal_index':8, 'matrix': np.array([[0,0,0], [1,0,0], [1,1,0]])},
            {'name': 'F', 'gal_index':6, 'matrix': np.array([[0,1,1], [1,0,0], [0,0,0]])},
            {'name': 'G', 'gal_index':9, 'matrix': np.array([[0,0,1], [1,0,0], [0,1,0]])},
            {'name': 'H', 'gal_index':11, 'matrix': np.array([[0,1,0], [1,0,0], [1,1,0]])},
            {'name': 'I', 'gal_index':12, 'matrix': np.array([[0,1,1], [1,0,0], [1,0,0]])},
            {'name': 'J', 'gal_index':13, 'matrix': np.array([[0,1,0], [1,0,1], [1,0,0]])},
            {'name': 'K', 'gal_index':10, 'matrix': np.array([[0,0,0], [1,0,1], [1,1,0]])},
            {'name': 'L', 'gal_index':14, 'matrix': np.array([[0,1,1], [1,0,0], [1,1,0]])},
            {'name': 'M', 'gal_index':15, 'matrix': np.array([[0,1,1], [1,0,1], [1,1,0]])}
        ]

    def matrix_to_name(self, motif_matrix):
        for motif in self.motifs:
            for i in range(4):  # Rotate 0, 90, 180, 270 degrees
                if np.array_equal(motif_matrix, np.rot90(motif['matrix'], i)):
                    return motif['name']
        return None  # Return None if no matching pattern is found

    def name_to_matrix(self, name):
        for motif in self.motifs:
            if motif['name'] == name:
                return motif['matrix']
        return None  # Return None if no matching name is found

    def index_to_name(self, index):
        for motif in self.motifs:
            if motif['gal_index'] == index:
                return motif['name']
        return None  # Return None if no matching index is found

    def name_to_index(self, name):
        for motif in self.motifs:
            if motif['name'] == name:
                return motif['gal_index']
        return None  # Return None if no matching name is found


def test_motif_reader():
    import numpy.testing as npt
    motif_reader = MotifReader()

    # Test: matrix_to_name
    motif_matrix_A = np.array([[0,0,0], [1,0,0], [1,0,0]])
    assert motif_reader.matrix_to_name(motif_matrix_A) == 'A'

    # Test: name_to_matrix
    npt.assert_array_equal(motif_reader.name_to_matrix('A'), motif_matrix_A)

    # Test: index_to_name
    assert motif_reader.index_to_name(3) == 'B'

    # Test: name_to_index
    assert motif_reader.name_to_index('B') == 3

    # Test: rotation equivalence
    motif_matrix_A_rotated = np.rot90(motif_matrix_A)
    assert motif_reader.matrix_to_name(motif_matrix_A_rotated) == 'A'

    # Test: invalid input
    assert motif_reader.matrix_to_name(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])) is None
    assert motif_reader.name_to_matrix('Z') is None
    assert motif_reader.index_to_name(100) is None
    assert motif_reader.name_to_index('Z') is None


if __name__ == "__main__":
    # if len(sys.argv) < 3:
    #     raise ValueError("Please provide a target and an adjacency matrix file as command-line arguments.")
    adj_file = '/gpfs/bbp.cscs.ch/project/proj112/home/kurban/hippdiss-422/data/ca3_ca1_block_matrix.npz' #bool matrix

    # logging.info("Testing motif calculation...")
    test_motif_reader()


    logging.info("Testing motif calculation with SC (sampled)...")
    target = 'cylinder300'
    check_mtypes = ['SC','SP_Ivy','SP_PC']
    for motif_name in ['-C','-B','-A']:
        logging.info(f"Running motif calculation for {target} of motif {motif_name} of {check_mtypes}...")
        motif_reader = MotifReader()
        CM = motif_reader.name_to_matrix(motif_name)
        # logging.info(f"Motif matrix:\n{CM}")
        calculator = CA1MotifCalculator(target, adj_file,check_mtypes, CM)
        num_motifs = calculator.count_motifs(num_projection_samples=1000)  

        if motif_name == '-A' and  check_mtypes[0] == 'SC':
            assert num_motifs == 0 , f"Motif {motif_name} should not exist for {check_mtypes}: No cells should not project to CA3"

        
        if CM.sum(axis=0)[2] > 0 and check_mtypes[0] == 'SC': # no one should project to SC
            assert num_motifs == 0, "No cells should not project to CA3"
    


    target = 'cylinder300'
    CM = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    check_mtypes = ['SP_PC','SP_CCKBC','SP_Ivy']
    motif_reader = MotifReader()
    motif_name = motif_reader.matrix_to_name(CM)    
    logging.info(f"Running motif calculation for {target} of motif {motif_name} of {check_mtypes}...")
    calculator = CA1MotifCalculator(target, adj_file,check_mtypes, CM)
    num_motifs = calculator.count_motifs()



    target = 'slice23'
    motif_name = 'H'
    check_mtypes = ['SP_PC','SP_Ivy','SC']
    logging.info(f"Running motif calculation for {target} of motif {motif_name} of {check_mtypes}...")
    motif_reader = MotifReader()
    CM = motif_reader.name_to_matrix(motif_name)
    logging.info(f"Motif matrix:\n{CM}")
    calculator = CA1MotifCalculator(target, adj_file,check_mtypes, CM)
    num_motifs = calculator.count_motifs(num_projection_samples=1000) # BUS ERROR if all SC


    #TODO: boundary condition fixed
    #TODO: sample should be randomized not first k indices
    #TODO: bus error when all SC used with big sized motifs. perhaps divide matrix into smaller pieces.