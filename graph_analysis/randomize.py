from scipy import sparse
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import spatial


class RandomModel:
    def __init__(self, adjacency_matrix):
        self.n_nodes = adjacency_matrix.shape[0]
        self.p = np.sum(adjacency_matrix) / (self.n_nodes * (self.n_nodes - 1))
        self.adjacency_matrix = adjacency_matrix

class ErdosRenyiModel(RandomModel):
    def __init__(self, adjacency_matrix):
        super().__init__(adjacency_matrix)

    def generate(self):
        A = sparse.random(self.n_nodes, self.n_nodes, density=self.p, format='csr')
        A = A + A.T
        A = A > 0
        return A.astype(int)

class BarabasiAlbertModel(RandomModel):
    def __init__(self, adjacency_matrix):
        super().__init__(adjacency_matrix)

    def generate(self):
        A = np.zeros((self.n_nodes, self.n_nodes))
        A[:2, :2] = [[0, 1], [1, 0]]
        degrees = np.array([1, 1])
        for i in range(2, self.n_nodes):
            degrees = np.append(degrees, 0)
            for j in range(i):
                degrees[j] = np.sum(A[j, :])
            p = degrees / np.sum(degrees)
            A[i, :] = np.random.choice([0, 1], size=self.n_nodes, p=p)
            A[:, i] = A[i, :]
        return A.astype(int)

class WattsStrogatzModel(RandomModel):
    def __init__(self, adjacency_matrix):
        super().__init__(adjacency_matrix)

    def generate(self):
        A = np.zeros((self.n_nodes, self.n_nodes))
        A[:2, :2] = [[0, 1], [1, 0]]
        for i in range(2, self.n_nodes):
            for j in range(i):
                if np.abs(i - j) <= self.n_nodes // 2:
                    A[i, j] = 1
                    A[j, i] = 1
        return A.astype(int)

class ConfigurationModel(RandomModel):
    def __init__(self, adjacency_matrix):
        super().__init__(adjacency_matrix)

    def generate(self):
        raise NotImplementedError
        # degrees = np.array([np.sum(self.adjacency_matrix[i, :]) for i in range(self.n_nodes)])
        # degrees = np.repeat(np.arange(self.n_nodes), degrees)
        # np.random.shuffle(degrees)
        # A = np.zeros((self.n_nodes, self.n_nodes))
        # for i in range(0, self.n_nodes, 2):
        #     A[degrees[i], degrees[i + 1]] = 1
        #     A[degrees[i + 1], degrees[i]] = 1
        # return A.astype(int)

class DistanceDependentConfigurationalModel(RandomModel):
    '''Randomly connect a number of neurons, keeping their distance dependence intact.
    D: Matrix (AxA) of distances to all other neurons in the circuit
    C: Matrix (AxA) of connections to all other neurons in the circuit (boolean)
    '''

    def __init__(self, adjacency_matrix,positions,nbins=50):
        super().__init__(adjacency_matrix)
        self.positions = positions
        self.nbins = nbins
        self.synaptome = adjacency_matrix
        self.distance_matrix = spatial.distance_matrix(self.positions, self.positions)
        
    def to_connectome(self, synaptome):
        connectome = synaptome.copy()
        connectome[connectome > 0] = 1
        return connectome.astype(bool)

    def generate(self):
        D = self.distance_matrix
        C = self.to_connectome(self.synaptome)

        dbins = np.linspace(0, D.max(),self.nbins + 1) + 0.1
        Di = np.digitize(D, bins=dbins) - 1
        H_connected = np.histogram(Di[C.astype(bool).toarray()], bins=range(self.nbins + 1))[0]
        H_all = np.histogram(Di, bins=range(self.nbins + 1))[0]
        P = H_connected.astype(float) / H_all
        n_eff = np.array(C.sum(axis=1)).transpose()[0]
        indptr = [0]
        indices = []
        for row, n in tqdm(zip(Di, n_eff)):
            p_row = P[row]
            p_row[row == -1] = 0
            p_row = p_row / p_row.sum()
            rnd = np.random.choice(len(row), n, replace=False, p=p_row)
            indices.extend(rnd)
            indptr.append(indptr[-1] + n)
        data = np.ones_like(indices, dtype=bool)
        return sparse.csr_matrix((data, indices, indptr), shape=D.shape)
    
    def plot_distance_dep_connections(self,rnd_connections):
        ax = plt.figure().add_axes([0.1, 0.1, 0.8, 0.8])
        ax.plot(sorted(self.distance_matrix[self.synaptome.astype(bool).toarray()]), label='Data')
        ax.plot(sorted(self.distance_matrix[rnd_connections.astype(bool).toarray()]), label='Control')
        ax.legend(); ax.set_xlabel('Connection #'); ax.set_ylabel('Distance (um)')

class DegreePreservingConnectomeModel(RandomModel):
    '''
    Given a network with weights, shuffle the weights while keeping the degree distribution intact.

    Parameters
    ----------
    adjacency_matrix: scipy.sparse.csr_matrix
        Adjacency matrix of the network to be shuffled
    
    Returns
    -------
    scipy.sparse.csr_matrix
        Shuffled adjacency matrix
    '''
    def __init__(self, adjacency_matrix,keep_total_degrees=True):
        super().__init__(adjacency_matrix)
        self.adjacency_matrix = adjacency_matrix
        self.keep_total_degrees = keep_total_degrees

    def generate(self,seed=42):
        if not self.keep_total_degrees:
            A = self.adjacency_matrix.copy()
            A = A.tolil()
            np.random.seed(seed)
            np.random.shuffle(A.data)
            A.setdiag(0)
            return A.tocsr()
        else:
            M = self.adjacency_matrix.tocoo()
            M.col = np.random.permutation(M.col)
            return M.tocsr()

class WeightPermutedRandomModel(RandomModel):
    def __init__(self, adjacency_matrix):
        super().__init__(adjacency_matrix)
        self.is_weighted = self.check_weighted()
    
    def check_weighted(self):
        return len(self.adjacency_matrix.data) > 2
    
    def generate(self,seed=42):
        if not self.is_weighted:
            raise ValueError('Input adjacency matrix is not weighted')
    
        synaptome_weight_permuted = self.adjacency_matrix.copy().tocoo()
        np.random.seed(seed)
        synaptome_weight_permuted.data = np.random.permutation(synaptome_weight_permuted.data)

        np.testing.assert_equal(synaptome_weight_permuted.count_nonzero(), self.adjacency_matrix.count_nonzero())
        np.testing.assert_equal(synaptome_weight_permuted.data.sum(), self.adjacency_matrix.data.sum())

        return synaptome_weight_permuted.tocsr()

if __name__ == '__main__':
    adj_path = '/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/data/ca1_synaptome.npz'
    synaptome = sparse.load_npz(adj_path)
    connectome = synaptome.copy()
    connectome[connectome > 0] = 1

    target = 'slice10'

    from bluepysnap import Circuit
    CIRCUIT_DIR =  '/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/sonata/circuit_config.json'
    circuit = Circuit(CIRCUIT_DIR)
    nodes = circuit.nodes["hippocampus_neurons"]
    edges = circuit.edges["hippocampus_neurons__hippocampus_neurons__chemical_synapse"]
    target_indices = nodes.ids(target)

    target_synaptome = synaptome[target_indices,:][:,target_indices]
    # mtypes_by_gid = c.nodes['hippocampus_neurons'].get().mtype.values

    xyz = nodes.get(target)[['x','y','z']]

    # model = ErdosRenyiModel(adj)
    # model = BarabasiAlbertModel(adj)
    # model = WattsStrogatzModel(adj)
    # model = ConfigurationModel(adj)
    model = DistanceDependentConfigurationalModel(target_synaptome,xyz)
    dd_adj = model.generate()

    save_dir = '../output/random_models'
    os.makedirs(save_dir,exist_ok=True)

    # compare distribution of distances of connected neurons
    model.plot_distance_dep_connections(dd_adj)
    plt.savefig(f'{save_dir}/connections_dd_configurational.png')
    breakpoint()

    # compare out-degree distribution
    model_outdegree = model.to_connectome(model.synaptome).sum(axis=1).flatten()
    dd_outdegree = dd_adj.sum(axis=1).flatten()
    np.testing.assert_array_equal(model_outdegree,dd_outdegree)

    # save adjacency matrix
    sparse.save_npz(f'{save_dir}/dd_configurational_{target}.npz',dd_adj)