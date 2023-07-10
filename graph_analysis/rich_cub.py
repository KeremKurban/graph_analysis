import dask
import dask.bag as db
from dask.distributed import Client, LocalCluster
import numpy as np
# import dask.delayed as delayed
from dask import compute, delayed
import logging 

class RichClubAnalysis:
    """
    Class for performing rich club analysis on a square sparse CSC matrix.
    """
    
    def __init__(self, adjacency_matrix):
        """
        Initialize the RichClubAnalysis object.
        
        Parameters:
            adjacency_matrix (scipy.sparse.csc_matrix): Square sparse CSC matrix representing the graph adjacency.
        """
        self.adjacency_matrix = adjacency_matrix
        self._is_MultiGraph = None
        self._is_Directed = None
        # FIXME: if weights are in range 0-1, not 1...n, then this needs to be changed
        self._total_degree = None

    @property
    def is_MultiGraph(self):
        if self._is_MultiGraph is None:
            # Calculate the parameter value
            self._is_MultiGraph = np.any(self.adjacency_matrix.A > 1)
        return self._is_MultiGraph

    @property
    def is_Directed(self):
        if self._is_Directed is None:
            # Calculate the parameter value
            self._is_Directed = not np.array_equal(adjacency_matrix, adjacency_matrix.T)
        return self._is_Directed

    @property
    def total_degree(self):
        if self._total_degree is None:
            # Calculate the parameter value
            self._total_degree =  np.array(self.adjacency_matrix.sum(axis=0)).flatten() + np.array(self.adjacency_matrix.sum(axis=1)).flatten() # out + indegrees per node
        return self._total_degree     

    def undirected_rich_club_coefficient(self, k):
        """
        Calculate the rich club coefficient for a given degree threshold.
        
        Parameters:
            k (int): Degree threshold.
            
        Returns:
            float: Rich club coefficient.
        """
        if self.is_Directed:
            raise ValueError("Error: The graph is directed!") 

        degrees = np.array(self.adjacency_matrix.sum(axis=0)).flatten()
        k_degrees = degrees[degrees >= k]
        kc = self.adjacency_matrix[degrees >= k, :][:, degrees >= k]
        kc_indices = kc.nonzero()
        kc_count = len(kc_indices[0])
        kc_possible = len(k_degrees) * (len(k_degrees) - 1)
        if kc_possible == 0:
            return np.nan
        return (2 * kc_count) / kc_possible
    
    def weighted_rich_club_coefficient(self, k, method='strongest-links-in-network'):
        '''
        Alstott et al 2014, doi: : 10.1038/srep07258
        Fig2e: strongest-links-in-network
        Fig2f: strongest-links-in-richclub
        Fig2i: all-links-connected-to-rich-nodes
        (rest to be implemented if necessary)
        '''

        assert len(np.unique(self.adjacency_matrix.A)) > 2, "Error: The graph is not weighted!"
        # find total weights per node
        weights = np.array(self.adjacency_matrix.sum(axis=0)).flatten() + np.array(self.adjacency_matrix.sum(axis=1)).flatten()
        k_weights = weights[weights >= k] # filter out weights above threshold
        
        if method == 'strongest-links-in-network':
            kc = self.adjacency_matrix[weights >= k, :][:, weights >= k] # get adj of subgraph > k
            kc_indices = kc.nonzero() # get indices of edges
            kc_count = len(kc_indices[0]) # count edges in that subgraph
            kc_sum = np.sum(kc[kc_indices]) # sum of weights in that subgraph
             # strongest-links-in-network: among the all weights get top kc_count weights
            kc_possible = np.nansum(sorted(weights, reverse=True)[:kc_count]) # sum of top kc_count weights in the network

        elif method == 'strongest-links-around-richclub':
            kc_post = self.adjacency_matrix[weights >= k, :] # get adj of subgraph > k
            kc_pre = self.adjacency_matrix[:, weights >= k] # get adj of subgraph > k
            
            kc_indices = kc.nonzero() # get indices of edges
            kc_count = len(kc_indices[0]) # count edges in that subgraph
            kc_sum = np.sum(kc[kc_indices]) # sum of weights in that subgraph
             # strongest-links-in-richclub: among the weights in the subgraph get top kc_count weights
            kc_possible = np.nansum(sorted(np.concatenate([kc_post.data,kc_pre.data]),reversed=True)[:kc_count])

        elif method == 'all-links-connected-to-rich-nodes':
            kc_post = self.adjacency_matrix[weights >= k, :] # get adj of subgraph > k
            kc_pre = self.adjacency_matrix[:, weights >= k] # get adj of subgraph > k
            
            kc_indices = kc.nonzero() # get indices of edges
            kc_count = len(kc_indices[0]) # count edges in that subgraph
            kc_sum = np.sum(kc[kc_indices]) # sum of weights in that subgraph
            kc_possible = np.nansum(sorted(np.concatenate([kc_post.data,kc_pre.data]),reversed=True))

        if kc_possible == 0:
            logging.warning(f"Warning: maximum possible weights in the network is 0 for k={k}!")
            logging.warning(f"kc_count={kc_count}, kc_sum={kc_sum}, kc_possible={kc_possible}")
            return np.nan
        return (kc_sum) / kc_possible

    def directed_rich_club_coefficient(self, k):
        """
        Calculate the rich club coefficient for a given degree threshold.
        
        Parameters:
            k (int): Degree threshold.
            
        Returns:
            float: Rich club coefficient.
        """
        if not self.is_Directed:
            raise ValueError("Error: The graph is undirected! Use undirected_rich_club_coefficient") 
        
        degrees = np.array(self.adjacency_matrix.sum(axis=0)).flatten() + np.array(self.adjacency_matrix.sum(axis=1)).flatten() # out + indegrees per node
        k_degrees = degrees[degrees >= k]
        kc = self.adjacency_matrix[degrees >= k, :][:, degrees >= k]
        kc_indices = kc.nonzero()
        kc_count = len(kc_indices[0])
        kc_possible = len(k_degrees) * (len(k_degrees) - 1)

        if kc_possible == 0:
            return np.nan
        return (kc_count) / kc_possible

    def plot_rich_club(self):
        """
        Plot the rich club coefficient for different degree thresholds.
        """
        degrees = np.array(self.adjacency_matrix.sum(axis=0)).flatten()
        k_values = np.unique(degrees)
        rc_coefficients = [self.rich_club_coefficient(k) for k in k_values]
        
        plt.plot(k_values, rc_coefficients, 'o-')
        plt.xlabel('Degree Threshold (k)')
        plt.ylabel('Rich Club Coefficient')
        plt.title('Rich Club Analysis')
        plt.show()

    def calculate_rc_coefficient(self,k):
        rc_coefficient = self.directed_rich_club_coefficient(k)
        return k, rc_coefficient

    def calculate_weighted_rc_coefficient(self,k,**kwargs):
        rc_coefficient = self.weighted_rich_club_coefficient(k,**kwargs)
        return k, rc_coefficient

    def calculate_rich_club_coefficients(self, weighted=False, **kwargs):
        k_values = np.arange(1, np.max(self.total_degree))
        k_dict = {}

        # Parallelize the calculation of rc_coefficient for each k
        if weighted:
            # Define a wrapper function to pass additional parameters
            @delayed
            def wrapper(k):
                return self.calculate_weighted_rc_coefficient(k, method=kwargs.get('method', 'strongest-links-in-network'))
            
            k_values = np.arange(1, np.max(self.total_degree), kwargs.get('step', 1))

            # Apply the wrapper function to each k value
            results = compute(*[wrapper(k) for k in k_values], scheduler=kwargs.get('scheduler', None))
        else:
            results = compute(*[delayed(self.calculate_rc_coefficient)(k) for k in k_values], scheduler=kwargs.get('scheduler', None))

        # Collect the results into k_dict
        for k, rc_coefficient in results:
            k_dict[k] = rc_coefficient

        return k_dict



if __name__ == "__main__":
    from scipy import sparse
    from bluepysnap import Circuit

    adj_path = '/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/data/ca1_synaptome.npz'
    synaptome = sparse.load_npz(adj_path)
    connectome = synaptome.copy()
    connectome[connectome > 0] = 1
    # del synaptome

    target = 'slice10'
    CIRCUIT_DIR =  '/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/sonata/circuit_config.json'
    circuit = Circuit(CIRCUIT_DIR)
    nodes = circuit.nodes["hippocampus_neurons"]
    edges = circuit.edges["hippocampus_neurons__hippocampus_neurons__chemical_synapse"]
    target_indices = nodes.ids(target)
    target_synaptome = synaptome[target_indices,:][:,target_indices]
    target_connectome = connectome[target_indices,:][:,target_indices]

    # tot_degrees_conn = np.array(target_connectome.sum(axis=0)).flatten() + np.array(target_connectome.sum(axis=1)).flatten() # out + indegrees per node
    # rca = RichClubAnalysis(target_connectome)
    # client = Client()    # Start Dask client
    # k_dict_parallel_model = rca.calculate_rich_club_coefficients(weighted=False)
    # client.shutdown()    # Shutdown the Dask client
    # breakpoint()

    rca_syn = RichClubAnalysis(target_synaptome)
    # max_workers = 8
    # cluster = LocalCluster(n_workers=max_workers)

    client = Client()    # Start Dask client
    k_dict_parallel_model = rca_syn.calculate_rich_club_coefficients(weighted=True,
                                                                method='strongest-links-in-network',
                                                                scheduler=None,
                                                                step=100)
    client.shutdown()    # Shutdown the Dask client
    breakpoint()