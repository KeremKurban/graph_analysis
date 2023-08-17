import unittest
import numpy as np
import networkx as nx
from graph_analysis.triplets import CA1MotifCalculator, MotifReader
from tqdm import tqdm
from scipy import sparse

# class TestCA1MotifCalculator(unittest.TestCase):
#     def setUp(self):
#         super().setUp()  # It's a good practice to call the super's setUp
#         # Create a directed graph
#         self.G = nx.DiGraph()

#         # Add edges to the graph
#         self.G.add_edge('A', 'B')
#         self.G.add_edge('B', 'C')
#         self.G.add_edge('A', 'D')
#         self.G.add_edge('D', 'E')
#         self.G.add_edge('B', 'E')
#         self.G.add_edge('E', 'F')

#     def test_motif_calculation(self):
#         # Set up the target motif and the adjacency matrix
#         CM = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
#         A = nx.to_numpy_array(self.G, nodelist=sorted(self.G.nodes))  # Here, use self.G

#         # Mock the required arguments
#         target = "mock_target"
#         check_mtypes = ["mock_mtype"]

#         # Calculate the motif counts
#         calculator = CA1MotifCalculator(target, check_mtypes, CM)
#         result = calculator.motif_calculation(A)

#         # There should be 2 instances of the target motif in the graph
#         self.assertEqual(result, 2)


# # class for triplet calculation for different motifs
# class TestTriplets(unittest.TestCase):
#     def setUp(self):
#         super().setUp()


class TestCA1MotifCalculator(unittest.TestCase):

    def setUp(self):
        # Create an instance of the class with necessary parameters
        # self.calculator = CA1MotifCalculator(target="cylinder300", check_mtypes=["SP_PC", "SP_CCKBC", "SP_Ivy"],
        #                                      CM=[[0, 1, 1], [0, 0, 1], [0, 0, 0]])
        self.test_matrix = np.array(
                                    [[0,0,0,0,0,0,0],
                                    [1,0,1,0,0,0,0],
                                    [0,0,0,1,0,0,0],
                                    [0,0,0,0,0,0,0],
                                    [0,0,0,0,0,1,1],
                                    [0,0,0,0,0,0,1],
                                    [0,0,0,0,0,0,0]])
        self.test_matrix_sparse = sparse.csr_matrix(self.test_matrix)


        self.test_matrix2 = np.array(
                                    [[0,1,1,0,0,0,0],
                                    [0,0,1,0,0,0,0],
                                    [0,0,0,0,0,0,0],
                                    [0,0,0,0,0,0,0],
                                    [0,0,0,0,0,1,1],
                                    [0,0,0,0,0,0,1],
                                    [0,0,0,0,0,0,0]])
        self.test_matrix_sparse2 = sparse.csr_matrix(self.test_matrix)

    # def test_sample_adjacency_matrix(self):
    #     print('Running netsci algorithm')
    #     import netsci.metrics.motifs as nsm
    #     frequency,participation = nsm.motifs(self.test_matrix, algorithm='louzoun',participation=True)
    #     np.testing.assert_array_equal(frequency, np.array([-1, -1, -1,  1,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0]))
        
    #     print('Running graph-analysis algorithm')
    #     from graph_analysis.triplets import MotifCalculator
    #     calculator = MotifCalculator(adj=self.test_matrix_sparse)
    #     frequency_tbt = {}
    #     for motif_name in tqdm(['-C','-B','-A','A','B','C','D','E','F','G','H','I','J','K','L','M']):
    #         frequency_tbt[motif_name] = calculator.count_triplet_motif(motif_name)
    #     print(frequency_tbt)

    #     motif_reader = MotifReader()
    #     reordered_freqs_tbt = motif_reader.convert_name_to_gal_index(frequency_tbt)
    #     print(reordered_freqs_tbt)

    #     np.testing.assert_array_equal(reordered_freqs_tbt, frequency)

    def test_sample_adjacency_matrix2(self):
        print('Running netsci algorithm')
        import netsci.metrics.motifs as nsm
        frequency,participation = nsm.motifs(self.test_matrix2, algorithm='louzoun',participation=True)
        np.testing.assert_array_equal(frequency, np.array([-1, -1, -1,  0,  0,  0,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0]))
        
        print('Running graph-analysis algorithm')
        from graph_analysis.triplets import MotifCalculator
        calculator = MotifCalculator(adj=self.test_matrix_sparse2)
        frequency_tbt = {}
        for motif_name in tqdm(['-C','-B','-A','A','B','C','D','E','F','G','H','I','J','K','L','M']):
            frequency_tbt[motif_name] = calculator.count_triplet_motif(motif_name)
        print(frequency_tbt)

        motif_reader = MotifReader()
        reordered_freqs_tbt = motif_reader.convert_name_to_gal_index(frequency_tbt)
        print(reordered_freqs_tbt)

        np.testing.assert_array_equal(reordered_freqs_tbt, frequency)

    # def test_load_adjacency_matrix(self):
    #     # Test loading adjacency matrix
    #     adj_matrix = self.calculator.load_adjacency_matrix()
    #     self.assertIsNotNone(adj_matrix)

    # def test_get_number_of_cells(self):
    #     square_matrix = self.calculator.load_adjacency_matrix()
    #     num_gids, num_exc, num_inh, num_SC = self.calculator.get_number_of_cells(square_matrix)
    #     self.assertEqual(num_gids, 100)  # Replace with actual expected values

    # def test_get_mtypes_start_index(self):
    #     start_indices = self.calculator.get_mtypes_start_index()
    #     self.assertEqual(len(start_indices), len(self.calculator.mtypes_order) + 1) # +1 for the last index in cumulative

    # Add more test methods for other functionalities

if __name__ == '__main__':
    unittest.main()
