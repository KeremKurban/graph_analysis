import dask
import dask.bag as db
from dask.distributed import Client
import numpy as np

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
        self.is_MultiGraph = np.any(self.adjacency_matrix.A > 1)
        self.is_Directed = not np.array_equal(adjacency_matrix, adjacency_matrix.T)

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

    def calculate_rich_club_coefficients(self,degree):
        '''
        degree: either in or outdegree, given from outside class
        '''
        k_values = np.arange(1, np.max(degree))
        k_dict = {}

        # Create a Dask bag from the k_values
        k_bag = db.from_sequence(k_values)

        # Parallelize the calculation of rc_coefficient for each k
        results = k_bag.map(self.calculate_rc_coefficient).compute()

        # Collect the results into k_dict
        for k, rc_coefficient in results:
            k_dict[k] = rc_coefficient

        return k_dict