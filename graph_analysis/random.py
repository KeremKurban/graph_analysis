import numpy as np
from scipy import sparse

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
        degrees = np.array([np.sum(self.adjacency_matrix[i, :]) for i in range(self.n_nodes)])
        degrees = np.repeat(np.arange(self.n_nodes), degrees)
        np.random.shuffle(degrees)
        A = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(0, self.n_nodes, 2):
            A[degrees[i], degrees[i + 1]] = 1
            A[degrees[i + 1], degrees[i]] = 1
        return A.astype(int)

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

    def to_connectome(self, synaptome):
        connectome = synaptome.copy()
        connectome[connectome > 0] = 1
        return connectome.astype(bool)

    def distance_matrix(self, xyz): #TODO: Do we need outside target positions to make dd model of inside target?
        from scipy import spatial
        D = spatial.distance_matrix(xyz, xyz)
        return D

    def generate(self):
        D = self.distance_matrix(self.positions)
        C = self.to_connectome(self.synaptome)

        dbins = np.linspace(0, D.max(),self.nbins + 1) + 0.1
        Di = np.digitize(D, bins=dbins) - 1
        H_connected = np.histogram(Di[C.astype(bool).toarray()], bins=range(self.nbins + 1))[0]
        H_all = np.histogram(Di, bins=range(self.nbins + 1))[0]
        P = H_connected.astype(float) / H_all
        n_eff = np.array(C.sum(axis=1)).transpose()[0]
        indptr = [0]
        indices = []
        for row, n in zip(Di, n_eff):
            p_row = P[row]
            p_row[row == -1] = 0
            p_row = p_row / p_row.sum()
            rnd = np.random.choice(len(row), n, replace=False, p=p_row)
            indices.extend(rnd)
            indptr.append(indptr[-1] + n)
        data = np.ones_like(indices, dtype=bool)
        return sparse.csr_matrix((data, indices, indptr), shape=D.shape)
    