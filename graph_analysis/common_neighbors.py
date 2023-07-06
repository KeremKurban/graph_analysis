import numpy as np
from scipy import stats, sparse
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import networkx as nx
# from graph_analysis.degree import GraphAnalysis
import igraph,tqdm
from collections import Counter
import seaborn as sns
import pandas as pd
from scipy.stats import linregress
import logging
from bluepysnap import Circuit
import bluepysnap
import os 
from matplotlib.colors import LogNorm
from scipy import spatial
import json
from master import GraphAnalysis

logging.basicConfig(level=logging.INFO)

class CommonNeighbors(GraphAnalysis):
    def __init__(self, adjacency_matrix,kind):
        super().__init__(adjacency_matrix,kind)

    def _efferent_con_mat(self,conn,pop_gids):
        '''Returns a sparse matrix of the EFFERENT connectivity of neurons in
        the specified population'''
        shape = (len(pop_gids), nodes.size) # The output shape is (number of neurons in population x number of neurons in circuit)
        post = [conn.efferent_nodes(_g) for _g in pop_gids]  # Get identifiers of connected neurons
        
        '''prepare the sparse matrix representation, where the column indices for row i are stored in
        "indices[indptr[i]:indptr[i+1]]" and their corresponding values are stored in "data[indptr[i]:indptr[i+1]]".'''
        indptr = np.hstack((0, np.cumsum(list(map(len, post)))))
        indices = np.hstack(post)
        data = np.ones_like(indices, dtype=bool) # Simple boolean connection matrix. A connection exists or not.
        return sparse.csr_matrix((data, indices, indptr), shape=shape)

    def _get_efferent_from_adjacency(self,pop_gids):
        '''Returns a sparse matrix of the EFFERENT connectivity of neurons in
        the specified population'''
        logging.info('Indexing efferents of selected pop from adjacency matrix')
        return self.adjacency_matrix[pop_gids,:].astype(int)

    def common_neighbor_efferent(self,
                                 nodes,
                                 edges, 
                                 savedir:str, 
                                 analyze_population:str ='Excitatory', 
                                 n_smpl:int=2500,
                                 **kwargs):
        '''
        Calculates common neighbor bias for a given population of neurons.
        type: 'efferent' or 'afferent'
        savedir: directory to save the results
        analyze_population: population target
        n_smpl: number of neurons to analyze

        Returns: 
        plots on the savedir
        '''
        
        # circuit_path = '/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/sonata/circuit_config.json'
        # circuit = Circuit(circuit_path)
        # nodes = circuit.nodes["hippocampus_neurons"]
        # conn = circuit.edges["hippocampus_neurons__hippocampus_neurons__chemical"]

        logging.info('Sampling population...')
        analyze_gids = nodes.ids(analyze_population, sample=n_smpl)  # get the identifiers of target neurons
        # print(len(analyze_gids))
        # connections = self._efferent_con_mat(edges,analyze_gids)
        connections = self._get_efferent_from_adjacency(analyze_gids)
        # Let's look at the result
        vmin = np.min(connections) + 0.01   # +0.01 to avoid log(0) inside the plot
        vmax = np.max(connections)

        # plots adj matrix
        # ax = plt.figure().add_axes([0.1, 0.1, 0.8, 0.8])
        # ax.imshow(connections.toarray()[:1000, 9000:10000], cmap='Reds', norm=LogNorm(vmin=vmin, vmax=vmax))

        logging.info('Calculating common efferent neighbors...')
        def common_efferent_neighbors(M,save_matrix=None):
            saved_matrix = f'{savedir}/common_neighbor_matrix_efferent.npz'
            if save_matrix and os.path.exists(saved_matrix):
                CN = sparse.load_npz(saved_matrix)
            else:
                logging.info('Matrix multiplication in process...')
                CN = M.astype(int) * M.astype(int).transpose() # need to convert to int, so that neighbors are counted
                if save_matrix:
                    sparse.save_npz(f'{savedir}/common_neighbor_matrix_efferent.npz', CN)
            return CN.toarray()

        com_neighs = common_efferent_neighbors(connections,**kwargs)
        vmin = np.min(com_neighs) + 0.01   # +0.01 to avoid log(0) inside the plot
        vmax = np.max(com_neighs)

        ax = plt.figure().add_axes([0.1, 0.1, 0.8, 0.8])
        ax.imshow(com_neighs, cmap='Reds', norm=LogNorm(vmin=vmin, vmax=vmax))
        plt.xlabel('Postsynaptic index')
        plt.ylabel('Presynaptic index')
        plt.title('Common efferent neighbors')	
        plt.savefig(f'{savedir}/common_efferent_neighbors_adj.png')

        logging.info('Calculating the histogram of common neighbors')
        def cn_mat_to_histogram(CN, bins):
            '''limit to upper triangular matrix. This excludes the diagonal entries and redundant entries, because the common
            neighbor matrix is always symmetrical!
            '''
            return np.histogram(np.triu(CN, 1), bins=bins)[0]

        xbins = np.arange(600) # count from 0 to 500 common neighbors #TODO: to be chosen from data
        H = cn_mat_to_histogram(com_neighs, xbins)

        f = plt.figure()
        plt.plot(H)
        plt.yscale('log')
        plt.xlabel('Number of common neighbors')
        plt.ylabel('Number of neuron pairs')
        plt.savefig(f'{savedir}/common_neighbor_occurences.png')

        logging.info('Controlling common neighbor bias against Erdos-Renyi graph')
        def control_erdos_renyi_histogram(CN, N, bins):
            from scipy.stats import hypergeom
            out_degrees = np.diag(CN)
            '''Note: Here, we simply draw a random sample for each pair of neurons.
            Better, but more expensive would be to evaluate the probability mass function
            for all bins and for all pairs.'''
            expected = [hypergeom(N, d_A, out_degrees[(i+1):]).rvs()
                        for i, d_A in enumerate(out_degrees)]
            return np.histogram(np.hstack(expected), bins=bins)[0]

        H_ctrl_er = control_erdos_renyi_histogram(com_neighs, connections.shape[1], xbins)

        ax = plt.figure().add_axes([0.1, 0.1, 0.8, 0.8])
        ax.plot(xbins[:-1], H, color='red', marker='o', label='Experiment')
        ax.plot(xbins[:-1], H_ctrl_er, color='black', marker='o', label='Control (ER)')
        ax.set_yscale('log'); ax.legend(); ax.set_xlabel('Common neighbors'); ax.set_ylabel('Pairs')
        plt.savefig(f'{savedir}/common_neighbor_vs_ER.png')

        logging.info('Controlling common neighbor bias against distance dependent configurational graph')
        nbins = 50
        def connect_keep_dist_dep(D, C, nbins):
            '''Randomly connect a number of neurons, keeping their distance dependence intact.
            D: Matrix (AxN) of distances to all other neurons in the circuit
            C: Matrix (AxN) of connections to all other neurons in the circuit (boolean)'''
            dbins = np.linspace(0, D.max(), nbins + 1) + 0.1
            Di = np.digitize(D, bins=dbins) - 1
            H_connected = np.histogram(Di[C.astype(bool).toarray()], bins=range(nbins + 1))[0]
            H_all = np.histogram(Di, bins=range(nbins + 1))[0]
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

        xyz = nodes.positions()
        # get distance matrix as an input
        D = spatial.distance_matrix(xyz.loc[analyze_gids], xyz)

        # generate random instance
        rnd_connections = connect_keep_dist_dep(D, connections, nbins)

        # compare distribution of distances of connected neurons
        ax = plt.figure().add_axes([0.1, 0.1, 0.8, 0.8])
        ax.plot(sorted(D[connections.astype(bool).toarray()]), label='Data')
        ax.plot(sorted(D[rnd_connections.astype(bool).toarray()]), label='Control')
        ax.legend(); ax.set_xlabel('Connection #'); ax.set_ylabel('Distance (um)')
        plt.savefig(f'{savedir}/connections_dd_configurational.png')

        rnd_com_neighs = common_efferent_neighbors(rnd_connections)
        rnd_H = cn_mat_to_histogram(rnd_com_neighs, xbins)

        ax = plt.figure().add_axes([0.1, 0.1, 0.8, 0.8])
        ax.plot(xbins[:-1], H, color='red', marker='o', label='Experiment')
        ax.plot(xbins[:-1], rnd_H, color='black', marker='o', label='Control (Dist-dep.)')
        ax.set_yscale('log'); ax.legend(); ax.set_xlabel('Common neighbors'); ax.set_ylabel('Pairs');
        plt.savefig(f'{savedir}/common_neighbor_vs_dd_configurational.png')

        def cn_bias_1(H_data, H_ctrl):
            assert len(H_data) == len(H_ctrl)
            log_data = np.log10(H_data[1:]) # exclude the bin at 0
            log_ctrl = np.log10(H_ctrl[1:])
            idx_data = np.nonzero(~np.isinf(log_data))[0] # exclude zero bins
            idx_ctrl = np.nonzero(~np.isinf(log_ctrl))[0]
            slope_data = np.polyfit(idx_data.astype(float), log_data[idx_data], 1)[0]
            slope_ctrl = np.polyfit(idx_ctrl.astype(float), log_ctrl[idx_ctrl], 1)[0]
            return (slope_ctrl - slope_data) / slope_ctrl

        logging.info(cn_bias_1(H, rnd_H))

    def common_neighbor_prob(self,
                                 nodes,
                                 edges, 
                                 savedir:str, 
                                 analyze_population1:str ='Excitatory',
                                 analyze_population2:str ='Inhibitory', 
                                 n_smpl_population:int=2500,
                                 target='cylinder300',
                                 **kwargs):
        
        logging.info('Computing common neighbor probability')

        target_indices = nodes.ids(target)
        target_indices.sort()

        analyze_gids1 = nodes.ids(analyze_population1)
        analyze_gids1 = np.intersect1d(analyze_gids1, target_indices)
        analyze_gids1.sort()

        # analyze_gids1 = nodes.ids(analyze_population1, sample=n_smpl_population)
        # analyze_gids1.sort()

        analyze_gids2 = nodes.ids(analyze_population2)
        analyze_gids2 = np.intersect1d(analyze_gids2, target_indices)


        connections1 =  self._get_efferent_from_adjacency(analyze_gids1)
        connections2 =  self._get_efferent_from_adjacency(analyze_gids2)

        logging.info(f' sliced matrix shapes: {connections1.shape}, {connections2.shape}')

        def common_efferent_neighbors(M, *args):
            if len(args) == 0: # No second matrix provided: Default to the early use case, com. neighs. within the population
                return common_efferent_neighbors(M, M)
            M2 = args[0]
            assert M.shape[1] == M2.shape[1]
            CN = M.astype(int) * M2.astype(int).transpose() # Our new use case: com. neighs. for pairs of the two populations
            return CN.toarray()
        
        com_neighs = common_efferent_neighbors(connections1, connections2)
        logging.info(f'common neighs shape: {com_neighs.shape}')
        # ax = plt.figure().add_axes([0.1, 0.1, 0.8, 0.8])
        # ax.imshow(com_neighs, cmap='Reds')
        # plt.savefig(f'{savedir}/common_neigh_EI.png')

        con_1_to_2 = connections1[:, analyze_gids2.astype(bool)] # minus 1 because neuron gids start counting at 0.
        con_2_to_1 = connections2[:, analyze_gids1.astype(bool)].transpose() # transpose because we need population 1 to be along the first axis.


        def con_prob_for_common_neighbors(cn_mat, connections, min_connections=10):
            cn_x = np.unique(cn_mat)
            smpls = [connections[cn_mat == i] for i in cn_x]
            result = [(x, y.mean()) for x, y in zip(cn_x, smpls)
                    if np.prod(y.shape) >= min_connections]
            return zip(*result)

        x1, y1 = con_prob_for_common_neighbors(com_neighs, con_1_to_2)
        x2, y2 = con_prob_for_common_neighbors(com_neighs, con_2_to_1)
        ax = plt.figure().add_axes([0.1, 0.1, 0.8, 0.8])
        ax.plot(x1, y1, label='%s to %s' % (analyze_population1, analyze_population2))
        ax.plot(x2, y2, label='%s to %s' % (analyze_population2, analyze_population1))
        ax.legend()
        plt.savefig(f'{savedir}/common_neighbor_EI_prob.png')
        logging.info(f'Saving to {savedir}/common_neighbor_EI_prob.png saved: {os.path.exists(f"{savedir}/common_neighbor_EI_prob.png")}')



if __name__ == "__main__":

    adj_path = '/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/data/ca1_synaptome.npz'
    synaptome = sparse.load_npz(adj_path)
    connectome = synaptome.copy()
    connectome[connectome > 0] = 1
    # del synaptome

    analysis = GraphAnalysis(synaptome,'out')

    CIRCUIT_DIR =  '/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/sonata/circuit_config.json'
    circuit = Circuit(CIRCUIT_DIR)
    nodes = circuit.nodes["hippocampus_neurons"]
    edges = circuit.edges["hippocampus_neurons__hippocampus_neurons__chemical_synapse"]
    # save_dir = '/gpfs/bbp.cscs.ch/project/proj112/home/kurban/topology_paper/data/common_neighbor'
    save_dir = '/home/kurban/Documents/graph_analysis/output/common_neighbors/synaptome'
    os.makedirs(save_dir,exist_ok=True)
    # mtypes_by_gid = c.nodes['hippocampus_neurons'].get().mtype.values
    # high,low = analysis.bandwidth_groups(2, mtypes_by_gid)

    logging.info('Testing common neighbor bias')
    analysis.common_neighbor_efferent(nodes,edges,save_dir,'Excitatory',n_smpl=2500,save_matrix=True)
    analysis.common_neighbor_prob(nodes,edges,save_dir,'Excitatory','Inhibitory',n_smpl_population=2500)