import unittest
import numpy as np
from graph_analysis.rich_club import RichClubAnalysis
from scipy.sparse import csc_matrix

class TestRichClubAnalysis(unittest.TestCase):

    def create_test_matrix(self, matrix_type):
    
        if matrix_type == "undirected_unweighted":
            matrix_data = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 0]])
        elif matrix_type == "undirected_weighted":
            matrix_data = np.array([[0, 0, 2], [0, 0, 1.1], [0.6, 5, 0]])
        elif matrix_type == "directed_unweighted":
            matrix_data = np.array([[0, 0, 1], [0, 0, 1], [1, 0, 0]])
        elif matrix_type == "directed_weighted":
            matrix_data = np.array([[0, 0, 0.3], [0, 0, 2], [1, 0, 0]])

        return csc_matrix(matrix_data)

    def test_directed(self):
        matrix_types = [
            "undirected_unweighted", "undirected_weighted", 
            "directed_unweighted", "directed_weighted"
        ]
        expected_results = {
            "undirected_unweighted": False, 
            "undirected_weighted": False, 
            "directed_unweighted": True, 
            "directed_weighted": True
        }

        for matrix_type in matrix_types:
            with self.subTest(matrix_type=matrix_type):
                test_matrix = self.create_test_matrix(matrix_type)
                rca = RichClubAnalysis(test_matrix)
                self.assertEqual(rca.is_Directed, expected_results[matrix_type])
                print(matrix_type)
                                

if __name__ == '__main__':
    unittest.main()
