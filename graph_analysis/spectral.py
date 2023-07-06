import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix
from pygsp import graphs, filters
from sklearn.cluster import KMeans
from scipy import sparse
import logging,os
# from .graph_analysis import GraphAnalysis

class SpectralAnalyzer:
    def __init__(self, adjacency_matrix):
        self.graph = graphs.Graph(adjacency_matrix)
        self.graph.compute_laplacian('combinatorial')
        self.graph.compute_fourier_basis()

    def save(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)
    
    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as file:
            return pickle.load(file)

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
            ax.scatter(*coords.T, c=self.spectral_clustering(3))
            plt.show()
        else:
            raise ValueError("Only 2D and 3D visualizations are supported.")

if __name__ == "__main__":
    # Create a random graph
    # A = np.random.randint(0, 2, size=(100, 100))
    # A = np.triu(A, 1)
    # A = A + A.T
    # A = csr_matrix(A)

    logging.info('Loading circuit')
    from bluepysnap import Circuit
    CIRCUIT_DIR = '/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/'
    c = Circuit(f'{CIRCUIT_DIR}/sonata/circuit_config.json')
    nodes = c.nodes['hippocampus_neurons']
    target = 'slice20'
    target_indices = nodes.ids(target)
    target_mtypes = nodes.get(target).mtype.values
    target_mtype_dict = {i:value for i,value in enumerate(target_mtypes)}
    
    # subsample_surround = False
    # surround_indices = nodes.ids('surround')

    adj_path = '/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/data/ca1_synaptome.npz'
    adj = sparse.load_npz(adj_path)
    adj_target = adj[target_indices,:]
    adj_target = adj_target[:,target_indices]

    #normalize adjacency matrix
    # adj_target = adj_target / adj_target.max()

    # # Analyze the graph
    # analyzer = SpectralAnalyzer(adj_target)
    # # analyzer.visualize_graph([1, 2])

    # eigs = analyzer.get_eigenvalues()
    
    # output_dir = '../output/spectral_analysis_test/'
    # os.makedirs(output_dir, exist_ok=True)

    # analyzer.visualize_graph([1,2])
    # plt.savefig(f'{output_dir}/spectral_clustering.png')

    # analyzer.visualize_graph([1,2,3])
    # plt.savefig(f'{output_dir}/spectral_clustering3D.png')


    # import numpy as np
    # import matplotlib.pyplot as plt
    # from sklearn.datasets import make_moons
    # from sklearn.cluster import SpectralClustering

    # # Generate synthetic moon-shaped data
    # X, labels_true = make_moons(n_samples=200, noise=0.1, random_state=0)

    # # Apply spectral clustering
    # sc = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=0)
    # labels_pred = sc.fit_predict(X)

    # # Plot the ground truth and predicted clusters
    # plt.figure(figsize=(8, 4))
    # plt.subplot(121)
    # plt.scatter(X[:, 0], X[:, 1], c=labels_true)
    # plt.title('Ground Truth')
    # plt.subplot(122)
    # plt.scatter(X[:, 0], X[:, 1], c=labels_pred)
    # plt.title('Spectral Clustering')
    # plt.tight_layout()
    # plt.show()


    import networkx as nx
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import SpectralClustering

    # Create a random graph
    # G = nx.erdos_renyi_graph(n=200, p=0.1, seed=0)
    G = nx.from_scipy_sparse_array(adj_target)
    nx.set_node_attributes(G, target_mtype_dict, "mtype")

    # Generate node positions for visualization
    pos = nx.spring_layout(G, seed=0)

    # Convert the graph to an adjacency matrix
    adjacency_matrix = nx.to_numpy_array(G)

    # Perform spectral clustering
    sc = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=0)
    labels = sc.fit_predict(adjacency_matrix)

    # Visualize the graph and the clusters
    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    nx.draw(G, pos, node_color='lightgray', node_size=50)
    plt.title('Random Graph')
    plt.subplot(122)
    nx.draw(G, pos, node_color=labels, cmap='coolwarm', node_size=50)
    plt.title('Spectral Clustering')
    plt.tight_layout()
    plt.show()

    
    output_dir = '../output/spectral_analysis/'
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(f'{output_dir}/spectral_clustering_{target}_spring.png')
