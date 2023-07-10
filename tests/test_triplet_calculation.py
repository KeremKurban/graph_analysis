import unittest
import numpy as np
import networkx as nx
from graph_analysis.triplets import CA1MotifCalculator

class TestCA1MotifCalculator(unittest.TestCase):
    def setUp(self):
        super().setUp()  # It's a good practice to call the super's setUp
        # Create a directed graph
        self.G = nx.DiGraph()

        # Add edges to the graph
        self.G.add_edge('A', 'B')
        self.G.add_edge('B', 'C')
        self.G.add_edge('A', 'D')
        self.G.add_edge('D', 'E')
        self.G.add_edge('B', 'E')
        self.G.add_edge('E', 'F')

    def test_motif_calculation(self):
        # Set up the target motif and the adjacency matrix
        CM = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        A = nx.to_numpy_array(self.G, nodelist=sorted(self.G.nodes))  # Here, use self.G

        # Mock the required arguments
        target = "mock_target"
        check_mtypes = ["mock_mtype"]

        # Calculate the motif counts
        calculator = CA1MotifCalculator(target, check_mtypes, CM)
        result = calculator.motif_calculation(A)

        # There should be 2 instances of the target motif in the graph
        self.assertEqual(result, 2)

if __name__ == '__main__':
    unittest.main()
