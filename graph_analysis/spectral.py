import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix
from pygsp import graphs, filters
from sklearn.cluster import KMeans
# from .graph_analysis import GraphAnalysis

class SpectralAnalyzer:
    def __init__(self, adjacency_matrix):
        self.graph = graphs.Graph(adjacency_matrix)
        self.graph.compute_laplacian('combinatorial')
        self.graph.compute_fourier_basis()

    def get_laplacian(self):
        return self.graph.L.todense()

    def get_eigenvalues(self):
        return self.graph.e

    def get_eigenvectors(self):
        return self.graph.U

    def spectral_clustering(self, n_clusters):
        # Use the eigenvector corresponding to the second smallest eigenvalue
        fiedler_vector = self.graph.U[:, 1]
        fiedler_vector = fiedler_vector.reshape(-1, 1)

        # Perform k-means clustering on the Fiedler vector
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(fiedler_vector)

        return labels

    def visualize_graph(self, eigenvectors_indices):
        coords = self.graph.U[:, eigenvectors_indices]

        if len(eigenvectors_indices) == 2:
            plt.scatter(*coords.T, c=self.spectral_clustering(2))
            plt.show()
        elif len(eigenvectors_indices) == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(*coords.T, c=self.spectral_clustering(2))
            plt.show()
        else:
            raise ValueError("Only 2D and 3D visualizations are supported.")

if __name__ == "__main__":
    # Create a random graph
    A = np.random.randint(0, 2, size=(100, 100))
    A = np.triu(A, 1)
    A = A + A.T
    A = csr_matrix(A)

    # Analyze the graph
    analyzer = SpectralAnalyzer(A)
    # analyzer.visualize_graph([1, 2])

    eigs = analyzer.get_eigenvalues()
    breakpoint()