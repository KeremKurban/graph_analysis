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
import scipy

logging.basicConfig(level=logging.INFO)

class MotifCalculator:
    '''
    Base Class for motif calculation.
    Works with adjacency matrix and motif matrix / motif name
    '''
    def __init__(self,adj:scipy.sparse.csr_matrix):
        self.adj = adj
    
    def count_triplet_motif(self,motif_name, **kwargs):
        '''
        Counts the number of motifs in the given adjacency matrix
        '''
        motif_reader = MotifReader()
        motif_matrix = motif_reader.name_to_matrix(motif_name)

        return self._triplet_calculation(motif_matrix)
    
    def evaluate(self, x, boolean_value, to_dense=True):
        '''
        If the boolean value is true or 1 , returns the matrix as it is . If 0 or false, returns 'logical not' of the matrix
        '''
        if to_dense:
            return np.where(boolean_value, x.todense(), x.todense() == 0)
        else:
            raise NotImplementedError

    def _triplet_calculation(self, CM, **kwargs):

        A2 = self.evaluate(self.adj, CM[0, 2]) * self.evaluate(self.adj.transpose(), CM[2, 0])
        B2 = self.evaluate(self.adj, CM[0, 1]) * self.evaluate(self.adj.transpose(), CM[1, 0])
        C2 = self.evaluate(self.adj, CM[1, 2]) * self.evaluate(self.adj.transpose(), CM[2, 1])
        M = np.dot(B2.T, A2) * C2
        M_sparse = sparse.csc_matrix(M)
        return int(np.sum(M))

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

    def __init__(self, target, check_mtypes,adj_file=None):
        self.target = target
        self.adj_file = f'/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/data/ca1_ca3_to_{target}_synaptome.npz' #FIXME: This needs to be fetched from the circuit if it doesnt exist
        self.check_mtypes = check_mtypes # i.e. ['SP_PC','SP_CCKBC','SP_Ivy']
        # order exc first!!
        self.mtypes_order = ['SP_PC', 'SLM_PPA', 'SO_BP', 'SO_BS', 'SO_OLM', 'SO_Tri', 'SP_AA', 'SP_BS', 'SP_CCKBC', 'SP_Ivy', 'SP_PVBC', 'SR_SCA'] # TODO: fetch from extraction #FIXME: Doesnt apply for other circuit
        if not os.path.isfile(self.adj_file):
            raise FileNotFoundError(f"The adjacency matrix file {self.adj_file} does not exist.")
        
        self.circuit_path = '/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/CircuitConfig' #TODO: Need to be adaptable for other circuits
        if not os.path.isfile(self.circuit_path):
            raise FileNotFoundError(f"The circuit file {self.circuit_path} does not exist.")

        #TODO: SONATA
        self.c = bluepy.Circuit(self.circuit_path) 
        self.mtypes_by_gid = np.array(self.c.cells.get(self.target).mtype.values)
        self.mtypes = sorted(list(self.c.cells.mtypes))
        self.exc_mtypes = np.unique(self.c.cells.get()[self.c.cells.get().morph_class != 'INT'].mtype)
        self.inh_mtypes =  np.unique(self.c.cells.get()[self.c.cells.get().morph_class == 'INT'].mtype)
        # for exc_mtype in self.exc_mtypes:
        #     breakpoint()
        #     self.inh_mtypes.remove(exc_mtype)

        #FIXME: supports only below targets. Will need to fetch from big adjacency matrix or bluepysnap iter_connections.
        self.supported_targets = ['cylinder300','slice10','slice15','slice20']
        if self.target not in self.supported_targets:
            raise ValueError(f"Target {self.target} is not supported. Supported targets are {self.supported_targets}")

        assert os.path.exists(self.adj_file), f"Adjacency matrix for {target} does not exist. Use ca1_synaptome.npz for local and ca3_synaptome.npz for projections"

        # ATLAS related
        self.q = None

    def evaluate(self, x, boolean_value, to_dense=True):
        '''
        If the boolean value is true or 1 , returns the matrix as it is . If 0 or false, returns 'logical not' of the matrix
        '''
        if to_dense:
            return np.where(boolean_value, x.todense(), x.todense() == 0)
        else:
            raise NotImplementedError

    def load_adjacency_matrix(self,return_num_projections=True):
        '''
        Loads and returns squarized adjacency matrix from given scipy sparse matrix path.
        '''
        all2ca1_adj = sparse.load_npz(self.adj_file)
        np.testing.assert_equal(all2ca1_adj.shape[1], len(self.c.cells.ids(self.target)))

        all2SC = sparse.csr_matrix((all2ca1_adj.shape[0], all2ca1_adj.shape[0] - all2ca1_adj.shape[1]))
        all2all_adj = sparse.hstack([all2ca1_adj, all2SC])
        logging.debug(f"all2all_adj shape: {all2all_adj.shape}")
        self.num_external_fibers = all2ca1_adj.shape[0] - all2ca1_adj.shape[1]
        return all2all_adj.tocsr()

    def get_number_of_cells(self, square_matrix,num_projection_samples=False,**kwargs):
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

    def get_mtypes_start_index(self) -> np.array:
        '''
        Queries bluepy circuit for start and end indices of each mtype in the target and returns
        a numpy array of start indices starting with 0 and ending with the total number of cells in the target.
        '''
        target_data = self.c.cells.get(self.target)
        target_mtype_counts = pd.DataFrame(np.unique(target_data.mtype, return_counts=True)[1],
                                        index=np.unique(target_data.mtype, return_counts=True)[0], columns=['count'])
        target_mtype_counts = target_mtype_counts.reindex(index=self.mtypes_order)
        cumsum_mtypes_indices = np.hstack([0, np.cumsum(target_mtype_counts.values)])
        return cumsum_mtypes_indices

    def get_mtype_ranges(self):
        mt_spans = {}
        indices = self.get_mtypes_start_index()
        for idx,mtype in enumerate(self.mtypes):
            mt_spans[mtype] = [indices[idx],indices[idx+1]]
        return mt_spans

    def motif_calculation_from_list(self, all2all_adj, num_SC_sample):
        raise NotImplementedError
        mtype_ranges = self.get_mtype_ranges()

        for i in self.check_mtypes:
            if isinstance(i, list):
                # Initialize empty lists to store the start and end indices
                indices_to_get = []
                # Fetch the start and end indices for the desired keys
                for key in i:
                    if key in mtype_ranges:
                        indices = mtype_ranges[key]
                        indices_to_get.extend(range(indices[0],indices[1]))
                    else:
                        raise ValueError(f"Key {key} not found in mtypes")

                # Get the sampled submatrix
                sampled_matrix = all2all_adj[indices_to_get]
                
                
                # return self.motif_calculation_from_list(i)
            elif len(np.intersect1d(['INH', 'INT'], [i])) > 0:
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

    def motif_calculation(self, all2all_adj, CM, num_SC_sample):
        mt_index_dict = {}

        # check if mtypes/node_sets in the given queries exist in the circuit targets (in future, this should be the only case)
        if not np.all(np.isin(self.check_mtypes, [*self.mtypes,'SC','INT','INH'])): #TODO: Should fetch this from node_sets
            logging.warning(f"check_mtypes: {self.check_mtypes} not found in mtypes: {self.mtypes}. Fetching from circuit targets")
            target_indices = []
            
            for i in self.check_mtypes:
                if i not in self.c.cells.targets:
                    raise ValueError(f"Key {i} not found in circuit targets")
                target_indices_all = self.c.cells.ids(self.target)-1
                inds_i = self.c.cells.ids(i)-1
                global_intersect_inds = np.intersect1d(target_indices_all, inds_i)
                local_intersect_inds = np.where(np.isin(target_indices_all,global_intersect_inds))[0]
                target_indices.append(local_intersect_inds) 
            
            mt1_to_mt2 = all2all_adj[target_indices[0],:][:,target_indices[1]]
            mt2_to_mt1 = all2all_adj[target_indices[1],:][:,target_indices[0]]

            mt1_to_mt3 = all2all_adj[target_indices[0],:][:,target_indices[2]]
            mt3_to_mt1 = all2all_adj[target_indices[2],:][:,target_indices[0]]

            mt2_to_mt3 = all2all_adj[target_indices[1],:][:,target_indices[2]]
            mt3_to_mt2 = all2all_adj[target_indices[2],:][:,target_indices[1]]

        else: # if there is not circuit target in the check_mtypes
            cumsum_mtypes_indices = self.get_mtypes_start_index()
            logging.debug(f"cumsum_mtypes_indices: {cumsum_mtypes_indices}")

            for i in self.check_mtypes:
                if isinstance(i, list):
                    raise NotImplementedError
                if len(np.intersect1d(['INH', 'INT'], [i])) > 0:
                    st_index, end_index = cumsum_mtypes_indices[1], cumsum_mtypes_indices[-1] # TODO: fix. hardwired for CA1 with one exc mtype at the 0th index
                    logging.info(f"st_index: {st_index}, end_index: {end_index}")
                elif i == 'SC':
                    st_index, end_index = cumsum_mtypes_indices[-1], cumsum_mtypes_indices[-1] + num_SC_sample
                else:
                    mt_index_in_ordering = np.where(np.isin(self.mtypes_order, i))[0][0]
                    st_index, end_index = cumsum_mtypes_indices[mt_index_in_ordering], cumsum_mtypes_indices[mt_index_in_ordering + 1]
                
                logging.debug(f"{i} st_index: {st_index}, end_index: {end_index}")
                mt_index_dict[i] = [st_index, end_index]

            # print the following only in the debug mode
            mt1_to_mt2 = all2all_adj[mt_index_dict[self.check_mtypes[0]][0]:mt_index_dict[self.check_mtypes[0]][1], 
                                    mt_index_dict[self.check_mtypes[1]][0]:mt_index_dict[self.check_mtypes[1]][1]] 
            mt2_to_mt1 = all2all_adj[mt_index_dict[self.check_mtypes[1]][0]:mt_index_dict[self.check_mtypes[1]][1],
                                    mt_index_dict[self.check_mtypes[0]][0]:mt_index_dict[self.check_mtypes[0]][1]] 

            mt1_to_mt3 = all2all_adj[mt_index_dict[self.check_mtypes[0]][0]:mt_index_dict[self.check_mtypes[0]][1], 
                                    mt_index_dict[self.check_mtypes[2]][0]:mt_index_dict[self.check_mtypes[2]][1]] 
            mt3_to_mt1 = all2all_adj[mt_index_dict[self.check_mtypes[2]][0]:mt_index_dict[self.check_mtypes[2]][1],
                                    mt_index_dict[self.check_mtypes[0]][0]:mt_index_dict[self.check_mtypes[0]][1]] 

            mt2_to_mt3 = all2all_adj[mt_index_dict[self.check_mtypes[1]][0]:mt_index_dict[self.check_mtypes[1]][1], 
                                    mt_index_dict[self.check_mtypes[2]][0]:mt_index_dict[self.check_mtypes[2]][1]] 
            mt3_to_mt2 = all2all_adj[mt_index_dict[self.check_mtypes[2]][0]:mt_index_dict[self.check_mtypes[2]][1],
                                    mt_index_dict[self.check_mtypes[1]][0]:mt_index_dict[self.check_mtypes[1]][1]] 

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

    def get_positions(self, target):
        return self.c.cells.positions(target)
    
    def get_ltr_positions(self, target):
        if self.q is None:
            from graph_analysis.coordinate_query import CoordinateQuery
            ATLAS_DIR = "/gpfs/bbp.cscs.ch/project/proj112/entities/atlas/20211004_BioM/"
            COORDINATES_DIR = f'{ATLAS_DIR}/coordinates.nrrd'
            self.q = CoordinateQuery(COORDINATES_DIR)

        return self.q.xyz_to_ltr(self.get_positions(target).values)


    def count_motifs(self,**kwargs):
        all2all_adj = self.load_adjacency_matrix()
        # target_adj_with_proj = self.load_adjacency_matrix()
        num_target_gids, num_exc_cells, num_inh_cells, num_SC_sample = self.get_number_of_cells(all2all_adj, **kwargs)
        num_motifs = self.motif_calculation(all2all_adj, num_SC_sample)
        return num_motifs
    
class MotifReader:
    def __init__(self):
        self.motifs = [
            {'name': '-C','gal_index':0, 'matrix': np.array([[0,0,0], [0,0,0], [0,0,0]])},
            {'name': '-B','gal_index':1, 'matrix': np.array([[0,1,0], [0,0,0], [0,0,0]])},
            {'name': '-A','gal_index':2, 'matrix': np.array([[0,1,0], [1,0,0], [0,0,0]])},
            {'name': 'A', 'gal_index':4, 'matrix': np.array([[0,0,0], [1,0,0], [1,0,0]])},
            {'name': 'B', 'gal_index':3, 'matrix': np.array([[0,0,0], [1,0,0], [0,1,0]])},
            {'name': 'C', 'gal_index':5, 'matrix': np.array([[0,0,0], [1,0,1], [0,0,0]])},
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
        import igraph
        g = igraph.Graph.Adjacency(motif_matrix)

        for motif in self.motifs:
            g_canonical = igraph.Graph.Adjacency(motif['matrix'])
            if g.isomorphic(g_canonical):
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

    def convert_name_to_gal_index(self,freqs:dict):
        '''
        Converts motif names to gal index
        '''
        reordered = np.zeros(len(freqs.values()))
        for key,value in freqs.items():
            corresponding_idx = self.name_to_index(key)
            reordered[corresponding_idx] = value
        return reordered

class NodeSetExtractor:
    def __init__(self,circuit,adjacency_matrix,population='hippocampus_neurons'):
        self.adjacency_matrix = adjacency_matrix
        self.nodes = circuit.nodes[population]

    def extract_from_local(self,node_set):
        '''
        Parameters
        ----------
        full_adjacency : scipy.sparse.csr_matrix
            Full adjacency matrix
        node_set : list
            node set name for the target adj to be extracted
        '''
        node_indices = self.nodes.ids(node_set)
        return self.adjacency_matrix[node_indices,:][:,node_indices]
    
    def extract_with_projections(self,node_set,projection_matrix_path:str):
        projection_matrix = sparse.load_npz(projection_matrix_path)
        node_indices = self.nodes.ids(node_set)
        local_target_adj = self.adjacency_matrix[node_indices,:][:,node_indices] # 6k x 6k for cylinder 300
        return np.concatenate([local_target_adj,projection_matrix[:,node_indices]],axis=0) # 6k+N x 6k for cylinder 300
        

# Defining the ExtendedMotifReader class without the dependency on igraph
class DotMotifReader:

    def __init__(self):
        self.node_map = {'A': 0, 'B': 1, 'C': 2}

    def matrix_to_dotmotif(self, motif_matrix):
        """
        Converts a motif matrix to a DotMotif textual representation.
        """
        import itertools
        node_map = {0: 'A', 1: 'B', 2: 'C'}
        dotmotif_str = ""
        for i, j in itertools.product(range(3), repeat=2):
            if motif_matrix[i, j]:
                dotmotif_str += f"{node_map[i]} -> {node_map[j]}\n"
        return dotmotif_str.strip()

    def dotmotif_to_matrix(self, dotmotif_str):
        """
        Converts a DotMotif textual representation to a motif matrix.
        """
        node_map = {'A': 0, 'B': 1, 'C': 2}
        matrix = np.zeros((3, 3), dtype=int)
        for line in dotmotif_str.split('\n'):
            if '->' in line:
                src, dst = line.split('->')
                src = src.strip()
                dst = dst.strip()
                matrix[node_map[src], node_map[dst]] = 1
        return matrix

class DotMotifAnalyzer(DotMotifReader):
    def __init__(self, result_dict, nodes, nodeset_ids):
        self.result_dict = result_dict
        self.nodes = nodes
        self.nodeset_ids = nodeset_ids

    def node_participation(self, motif_name, node_feature,visualize=False):

        analysis_dict = {}

        for node_key in self.node_map.keys():
            try:
                indices = [d[node_key] for d in self.result_dict[motif_name]] # find the indices of the nodes in the motif
                values = self.nodes.get(self.nodeset_ids[indices]).iloc[indices][node_feature] # get the values of the node feature
                unique_values, counts = np.unique(values, return_counts=True) # count the unique values
                analysis_dict[node_key] = dict(zip(unique_values, counts)) # store the counts in a dictionary
            except KeyError:
                analysis_dict[node_key] = None

        if visualize: # plot motif and node feature as bar plots
            #TODO: add graph visualization
                        
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, len(self.node_map.keys()), figsize=(15, 5))
            for i, node_key in enumerate(self.node_map):
                if analysis_dict[node_key] is not None:
                    ax[i].bar(analysis_dict[node_key].keys(), analysis_dict[node_key].values())
                    ax[i].set_title(node_key)
            plt.show()

        return analysis_dict


def test_dotmotif_reader():
    # Example usage
    reader = DotMotifReader()

    # Convert matrix to DotMotif string
    motif_matrix = np.array([[0, 1, 0], [1, 0, 0], [0, 1, 1]])  # This is a simple A -> B, B -> A motif
    dotmotif_str = reader.matrix_to_dotmotif(motif_matrix)
    print("DotMotif String:\n", dotmotif_str)

    # Convert DotMotif string to matrix
    new_matrix = reader.dotmotif_to_matrix(dotmotif_str)
    print("Reconstructed Matrix:\n", new_matrix)

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

    # Test: rotation equivalence (function in networkx)
    A_isomorph = np.array([[0,1,0], [0,0,0], [0,1,0]])
    A_isomorph2 = np.array([[0,0,1], [0,0,1], [0,0,0]])
    assert motif_reader.matrix_to_name(A_isomorph) == 'A'
    assert motif_reader.matrix_to_name(A_isomorph2) == 'A'

    matrix_H = np.array([[0,1,0], [1,0,0], [1,1,0]])
    assert motif_reader.matrix_to_name(matrix_H) == 'H'
    H_isomorph = np.array([[0,0,1], [1,0,1], [1,0,0]])
    assert motif_reader.matrix_to_name(H_isomorph) == 'H'
    H_isomorph2 = np.array([[0,1,1], [0,0,1], [0,1,0]])
    assert motif_reader.matrix_to_name(H_isomorph2) == 'H'

    # Test: invalid input
    assert motif_reader.matrix_to_name(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])) is None
    assert motif_reader.name_to_matrix('Z') is None
    assert motif_reader.index_to_name(100) is None
    assert motif_reader.name_to_index('Z') is None



if __name__ == "__main__":
    # if len(sys.argv) < 3:
    #     raise ValueError("Please provide a target and an adjacency matrix file as command-line arguments.")
    # adj_file = '/gpfs/bbp.cscs.ch/project/proj112/home/kurban/hippdiss-422/data/ca3_ca1_block_matrix.npz' #bool matrix

    # logging.info("Testing motif calculation...")
    test_dotmotif_reader()
    breakpoint()
    test_motif_reader()

    supported_targets = ['cylinder300','slice10','slice15','slice20']
    target = 'slice10'
    target_adj_path = f'/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/data/ca1_ca3_to_{target}_synaptome.npz'
    assert os.path.exists(target_adj_path), f"Adjacency matrix for {target} does not exist. Use ca1_synaptome.npz for local and ca3_synaptome.npz for projections"

    # logging.info("Testing motif calculation with SC (sampled)...")
    # check_mtypes = ['SC','SP_Ivy','SP_PC']
    # for motif_name in ['-C','-B','-A']:
    #     logging.info(f"Running motif calculation for {target} of motif {motif_name} of {check_mtypes}...")
    #     motif_reader = MotifReader()
    #     CM = motif_reader.name_to_matrix(motif_name)
    #     # logging.info(f"Motif matrix:\n{CM}")
    #     calculator = CA1MotifCalculator(target,check_mtypes, CM)
    #     num_motifs = calculator.count_motifs(num_projection_samples=1000)  

    #     if motif_name == '-A' and  check_mtypes[0] == 'SC':
    #         assert num_motifs == 0 , f"Motif {motif_name} should not exist for {check_mtypes}: No cells should not project to CA3"

        
    #     if CM.sum(axis=0)[2] > 0 and check_mtypes[0] == 'SC': # no one should project to SC
    #         assert num_motifs == 0, "No cells should not project to CA3"
    
    from tqdm import tqdm
    logging.info("Testing motif calculation internally")
    check_mtypes = ['Mosaic','Mosaic','Mosaic']
    frequencies = {}
    motif_reader = MotifReader()
    for motif_name in tqdm(['-C','-B','-A','A','B','C','D','E','F','G','H','I','J','K','L','M']):
        logging.info(f"Running motif calculation for {target} of motif {motif_name} of {check_mtypes}...")
        CM = motif_reader.name_to_matrix(motif_name)
        calculator = CA1MotifCalculator(target,check_mtypes, CM)
        num_motifs = calculator.count_motifs()  
        frequencies[motif_name] = num_motifs

    breakpoint()

    # CM = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    # check_mtypes = ['SP_PC','SP_CCKBC','SP_Ivy']
    # motif_reader = MotifReader()
    # motif_name = motif_reader.matrix_to_name(CM)    
    # logging.info(f"Running motif calculation for {target} of motif {motif_name} of {check_mtypes}...")
    # calculator = CA1MotifCalculator(target, target_adj_path,check_mtypes, CM)
    # num_motifs = calculator.count_motifs()

    # CM = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    # check_mtypes = ['SP_PC','SP_CCKBC','SP_Ivy']
    # motif_reader = MotifReader()
    # motif_name = motif_reader.matrix_to_name(CM)    
    # logging.info(f"Running motif calculation for {target} of motif {motif_name} of {check_mtypes}...")

    # calculator = CA1MotifCalculator(target, target_adj_path,check_mtypes, CM)
    # num_motifs = calculator.count_motifs()


    # motif_name = 'M'
    # check_mtypes = ['PeriSomatic_INH','PeriSomatic_INH','PeriSomatic_INH']
    # logging.info(f"Running motif calculation for {target} of motif {motif_name} of {check_mtypes}...")
    # motif_reader = MotifReader()
    # CM = motif_reader.name_to_matrix(motif_name)
    # logging.info(f"Motif matrix:\n{CM}")
    # calculator = CA1MotifCalculator(target,check_mtypes, CM)
    # num_motifs = calculator.count_motifs(num_projection_samples=1000) # BUS ERROR if all SC


    # logging.info('Positional information on triplets')
    # motif_name = 'B'
    # check_mtypes = ['SP_PC','SP_PC','SP_PC']
    # logging.info(f"Running motif calculation for {target} of motif {motif_name} of {check_mtypes}...")
    # motif_reader = MotifReader()
    # CM = motif_reader.name_to_matrix(motif_name)
    # logging.info(f"Motif matrix:\n{CM}")
    # calculator = CA1MotifCalculator(target,check_mtypes, CM)
    # calculator.analyze_positions()
    # # num_motifs = calculator.count_motifs(num_projection_samples=1000) # BUS ERROR if all SC
    # breakpoint()

    #TODO: boundary condition fixed
    #TODO: sample should be randomized not first k indices
    #TODO: bus error when all SC used with big sized motifs. Fperhaps divide matrix into smaller pieces (dask).