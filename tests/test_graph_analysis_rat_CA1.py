import unittest
from graph_analysis import GraphAnalysis
from bluepysnap import Circuit
from scipy import sparse
import numpy as np

class TestGraphAnalysis(unittest.TestCase):
    def setUp(self):
        # Load the adjacency matrix
        adj_path = '/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/data/ca1_synaptome.npz'
        self.adjacency_matrix = sparse.load_npz(adj_path)

        # Initialize a Circuit object
        circuit_dir = '/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/sonata'
        self.circuit = Circuit(circuit_dir)

    def test_degree_distribution(self):
        # Create a GraphAnalysis instance
        analysis = GraphAnalysis(self.graph, 'in')

        # Perform degree distribution analysis
        analysis.plot_degree_distribution()
        analysis.plot_log_binned_degree_distribution(cumulative=True, fit_min_degree=50)

        # Perform statistical tests
        from scipy import stats
        distributions = [stats.norm, stats.lognorm, stats.expon, stats.gamma]
        best_fit, best_params, best_sse = analysis.check_best_fit(distributions)

        # Assertions
        self.assertIsNotNone(best_fit)  # Ensure a best fit distribution is obtained
        self.assertIsNotNone(best_params)  # Ensure best fit parameters are obtained
        self.assertIsInstance(best_sse, float)  # Ensure SSE is a float value

    def test_hop_distance(self):
        # Create a GraphAnalysis instance
        analysis = GraphAnalysis(self.graph, 'out')

        # Compute the 3-hop distance matrix
        k = 3
        distance_matrix = analysis.k_hop_distance(k)

        # Assertions
        self.assertEqual(distance_matrix.shape, (self.graph.number_of_nodes(), self.graph.number_of_nodes()))  # Ensure the size of the distance matrix
        self.assertTrue(np.all(distance_matrix >= 0))  # Ensure all distances are non-negative

    def test_k_hop_distance(self):
        # Create a simple graph with 4 nodes and an adjacency matrix
        adjacency_matrix = sparse.csr_matrix([[0, 1, 1, 0],
                                            [1, 0, 0, 1],
                                            [1, 0, 0, 1],
                                            [0, 1, 1, 0]])
        graph_analysis = GraphAnalysis(adjacency_matrix, 'out')

        # Test k-hop distance for k = 1
        k = 1
        expected_result = sparse.csr_matrix([[0, 1, 1, 0],
                                            [1, 0, 0, 1],
                                            [1, 0, 0, 1],
                                            [0, 1, 1, 0]])
        result = graph_analysis.k_hop_distance(k)
        np.testing.assert_equal(result.toarray(), expected_result.toarray())

        # Test k-hop distance for k = 2
        k = 2
        expected_result = sparse.csr_matrix([[0, 1, 1, 1],
                                            [1, 0, 0, 1],
                                            [1, 0, 0, 1],
                                            [1, 1, 1, 0]])
        result = graph_analysis.k_hop_distance(k)
        np.testing.assert_equal(result.toarray(), expected_result.toarray())

        # Test k-hop distance for k = 3
        k = 3
        expected_result = sparse.csr_matrix([[0, 1, 1, 2],
                                            [1, 0, 0, 1],
                                            [1, 0, 0, 1],
                                            [2, 1, 1, 0]])
        result = graph_analysis.k_hop_distance(k)
        np.testing.assert_equal(result.toarray(), expected_result.toarray())

    def test_bandwidth_groups(self):
        # Create a GraphAnalysis instance
        analysis = GraphAnalysis(self.graph, 'in')

        # Generate random group assignments
        num_nodes = self.graph.number_of_nodes()
        groups = np.random.randint(0, 2, size=num_nodes)

        # Perform bandwidth groups analysis
        high_bandwidth_groups, low_bandwidth_groups = analysis.bandwidth_groups(2, groups)

        # Assertions
        self.assertGreater(len(high_bandwidth_groups), 0)  # Ensure at least one high-bandwidth group is identified
        self.assertGreater(len(low_bandwidth_groups), 0)  # Ensure at least one low-bandwidth group is identified

if __name__ == '__main__':
    unittest.main()
