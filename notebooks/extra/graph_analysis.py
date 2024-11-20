import numpy as np
from scipy import stats, sparse
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import networkx as nx
import igraph,tqdm
from collections import Counter
import seaborn as sns
import pandas as pd
from scipy.stats import linregress
import logging
from bluepysnap import Circuit
import bluepysnap
import os 

logging.basicConfig(level=logging.INFO)

class GraphAnalysis:
    def __init__(self, adjacency_matrix,kind):
        self.adjacency_matrix = self.assess_input(adjacency_matrix)
        self.is_lognormal_in = None
        self.is_lognormal_out = None
        self.is_lognormal_tot = None
        self.indegrees = self.get_degrees('in')
        self.outdegrees = self.get_degrees('out')
        self.totaldegrees = self.indegrees + self.outdegrees
        self.degrees = self.get_degrees(kind)
        self.kind = kind
    
    def assess_input(self,data):
        '''
        Converts certain known classes to scipy.sparse.cscmatrix
        '''
        import numpy as np
        import igraph
        from scipy import sparse
        
        
        if isinstance(data,nx.Graph):
            return nx.adjacency_matrix(data)
        elif isinstance(data,igraph.Graph):
            return data.get_adjacency_sparse().tocsc()
        elif isinstance(data,np.ndarray):
            return sparse.csc_matrix(data)
        elif isinstance(data,sparse.csc_matrix):
            return data
        elif isinstance(data,sparse.csr_matrix):
            return data.tocsc()
        else:
            raise NotImplementedError
            
    def get_degrees(self, kind):
        """
        Compute and return the desired degrees (in-degrees, out-degrees, or total degrees).
        """
        degrees = None
        # For scipy sparse matrix, convert to csr format for efficient row operations
        if hasattr(self.adjacency_matrix, "tocsr"):
            self.adjacency_matrix = self.adjacency_matrix.tocsr()

        if kind == 'in':
            degrees = np.array(self.adjacency_matrix.sum(axis=0)).flatten()
        elif kind == 'out':
            degrees = np.array(self.adjacency_matrix.sum(axis=1)).flatten()
        elif kind == 'total':
            in_degrees = np.array(self.adjacency_matrix.sum(axis=0)).flatten()
            out_degrees = np.array(self.adjacency_matrix.sum(axis=1)).flatten()
            degrees = in_degrees + out_degrees
        else:
            raise ValueError("Invalid degree type. Choose 'in', 'out', or 'total'.")
        
        degrees = degrees[degrees > 0]  # Remove zero degrees
        return degrees

    def fit_degree_distribution(self, degrees, distributions):
        """
        Fit the degree distribution to multiple statistical distributions and determine the best fit.
        """
        best_fit = None
        best_params = None
        best_sse = np.inf
        
        for distribution in distributions:
            # Fit the distribution to the degrees
            params = distribution.fit(degrees)
            sse = self.calculate_sse(degrees, distribution, params)
            
            # Check if the current fit is better than the previous best
            if sse < best_sse:
                best_fit = distribution
                best_params = params
                best_sse = sse
        
        return best_fit, best_params, best_sse
    
    def calculate_sse(self, degrees, distribution, params):
        """
        Calculate the sum of squared errors (SSE) between the observed degrees and the fitted distribution.
        """
        fitted_degrees = distribution.pdf(degrees, *params)
        sse = np.sum((degrees - fitted_degrees) ** 2)
        return sse

    def check_best_fit(self, distributions):
        """
        Find the best-fitting distribution for the degree distribution.
        """
        degrees = self.get_degrees(self.kind)
        
        best_fit, best_params, best_sse = self.fit_degree_distribution(degrees, distributions)
        
        return best_fit, best_params, best_sse
    
    def check_lognormality(self,alpha=1e-3):
        """
        Check the lognormality of the in and out degree distributions using statistical methods.

        alpha : if p<alpha : reject the null hypothesis that the distribution is normal.
        """
        in_degrees, out_degrees,tot_degrees = self.get_degrees('in'),self.get_degrees('out'),self.get_degrees('total')

        # Compute log-transformed degrees
        log_in_degrees = np.log(in_degrees)
        log_out_degrees = np.log(out_degrees)
        log_tot_degrees = np.log(tot_degrees)

        # Perform statistical tests for lognormality
        in_lognormal, in_pvalue = stats.normaltest(log_in_degrees)
        out_lognormal, out_pvalue = stats.normaltest(log_out_degrees)
        tot_lognormal, tot_pvalue = stats.normaltest(log_tot_degrees)

        if in_pvalue < alpha:  # null hypothesis: x comes from a normal distribution
            print("Indegree: The null hypothesis can be rejected")
            self.is_lognormal_in = False
        else:
            print("Indegree: The null hypothesis cannot be rejected")
            self.is_lognormal_in = True

        if out_pvalue < alpha:  # null hypothesis: x comes from a normal distribution
            print("Outdegree: The null hypothesis can be rejected")
            self.is_lognormal_out = False
        else:
            print("Outdegree: The null hypothesis cannot be rejected")
            self.is_lognormal_out = True

        if tot_pvalue < alpha:  # null hypothesis: x comes from a normal distribution
            print("Total degree: The null hypothesis can be rejected")
            self.is_lognormal_tot = False
        else:
            print("Total degree: The null hypothesis cannot be rejected")
            self.is_lognormal_tot = True

        return self.is_lognormal_in,self.is_lognormal_out,self.is_lognormal_tot

    def plot_degree_distribution(self,title=None):
        degree_counts = dict(Counter(self.degrees))
        degrees = list(degree_counts.keys())
        probabilities = [val / sum(degree_counts.values()) for val in degree_counts.values()]

        log_degrees = np.log10(degrees)
        log_probabilities = np.log10(probabilities)
        
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.subplot(1, 2, 1)
        sns.histplot(degrees, stat='probability', kde=True, bins=30)
        plt.xlabel('Degree (k)')
        plt.ylabel('Probability (P(k))')
        plt.title(f'Degree Distribution: LinLin Plot')
        
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=log_degrees, y=log_probabilities, color='b', alpha=0.5, label='Data')
        slope, intercept = self.calculate_power_law_exponent()
        sns.lineplot(x=log_degrees, y=intercept + slope * log_degrees, color='r', label='Fitted line')
        plt.xlabel('log(Degree)')
        plt.ylabel('log(Probability)')
        plt.title(f'Degree Distribution: LogLog Plot (Linear Binning)')

        
        plt.legend()
        plt.text(0.1, 0.9, f'Slope: {slope:.4f}', transform=plt.gca().transAxes, bbox=dict(facecolor='white', edgecolor='black'))
        
        if title:
            plt.suptitle(title)
        else:
            plt.suptitle(f'{self.kind}-Degree Analysis')
        plt.tight_layout()
        plt.show()

    def calculate_power_law_exponent(self): # deprecated
        degree_counts = dict(Counter(self.degrees))
        degrees = list(degree_counts.keys())
        probabilities = [val / sum(degree_counts.values()) for val in degree_counts.values()]

        log_degrees = np.log10(degrees)
        log_probabilities = np.log10(probabilities)

        slope, intercept, r_value, p_value, std_err = stats.linregress(log_degrees, log_probabilities)
        return slope, intercept
    
    def plot_log_binned_degree_distribution(self, cumulative=False, title=None, fit_min_degree=10,show_plot=True):
        degree_counts = dict(Counter(self.degrees))
        max_degree = np.max(list(degree_counts.keys()))

        # Generate logarithmically spaced bins
        log_bins = np.logspace(0, np.log10(max_degree), num=20)  # Adjust num as required

        # Compute histogram with log bins
        hist, bins = np.histogram(self.degrees, bins=log_bins, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        if cumulative:
            hist = np.cumsum(hist[::-1])[::-1]  # Reverse, compute cumulative sum, then reverse back

        if show_plot:
            # Plot degree distribution in log-log scale
            plt.figure(figsize=(7, 6))
            plt.loglog(bin_centers, hist, 'bo', markersize=5)

        # Calculate linear regression in log-log scale for high degrees only
        fit_indices = bin_centers >= fit_min_degree
        slope, intercept, r_value, p_value, std_err = linregress(np.log10(bin_centers[fit_indices]), np.log10(hist[fit_indices]))
        x_fit = np.linspace(np.log10(fit_min_degree), np.max(np.log10(bin_centers)), num=100)
        y_fit = intercept + slope * x_fit
        if show_plot:
            plt.plot(10 ** x_fit, 10 ** y_fit, 'r-')

            # Add text box with slope and R-squared
            textstr = f'Slope: {slope:.2f}\n$R^2$: {r_value**2:.2f}'
            plt.text(0.8, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.xlabel('Degree (k)')
            if cumulative:
                plt.ylabel('Cumulative Probability (P(K>=k))')
                if title:
                    plt.title(title)
                else:
                    plt.title(f'Log-Binned Cumulative {self.kind}-Degree Distribution')
            else:
                plt.ylabel('Probability (P(k))')
                if title:
                    plt.title(title)
                else:
                    plt.title(f'Log-Binned {self.kind}-Degree Distribution')

            plt.show()
        
        if show_plot:
            return slope
        else:
            return bin_centers, hist, intercept

    def k_hop_distance(self, k):
        """Computes the k-hop distance matrix for the graph."""
        # return self.adjacency_matrix ** k
        """Computes the k-th power of the adjacency matrix using repeated squaring."""
        result = sparse.eye(self.adjacency_matrix.shape[0])
        A = self.adjacency_matrix.copy()
        while k > 0:
            if k % 2 == 1:
                result = result @ A
            A = A @ A
            k //= 2
        return result

    def bandwidth_groups(self, k, groups):
        """Computes the high and low bandwidth groups."""
        distance_matrix = self.k_hop_distance(k)
        logging.info(f'Computed {k}-hop distance matrix')
        group_indices = {group: np.where(np.array(groups) == group)[0] for group in set(groups)}
        group_bandwidth = {group: distance_matrix[indices[:, None], indices].sum() for group, indices in tqdm.tqdm(group_indices.items())}
        median_bandwidth = np.median(list(group_bandwidth.values()))
        high_bandwidth_groups = [group for group, bandwidth in group_bandwidth.items() if bandwidth > median_bandwidth]
        low_bandwidth_groups = [group for group, bandwidth in group_bandwidth.items() if bandwidth <= median_bandwidth]
        return high_bandwidth_groups, low_bandwidth_groups

    def _efferent_con_mat(self,conn,pop_gids):
        '''Returns a sparse matrix of the EFFERENT connectivity of neurons in
        the specified population'''
        shape = (len(pop_gids), nodes.size) # The output shape is (number of neurons in population x number of neurons in circuit)
        post = [conn.efferent_nodes(_g) for _g in pop_gids]  # Get identifiers of connected neurons
        
        '''prepare the sparse matrix representation, where the column indices for row i are stored in
        "indices[indptr[i]:indptr[i+1]]" and their corresponding values are stored in "data[indptr[i]:indptr[i+1]]".'''
        indptr = np.hstack((0, numpy.cumsum(list(map(len, post)))))
        indices = np.hstack(post)
        data = np.ones_like(indices, dtype=bool) # Simple boolean connection matrix. A connection exists or not.
        return sparse.csr_matrix((data, indices, indptr), shape=shape)

    def common_neighbor_efferent(self,
                                 nodes,
                                 edges, 
                                 savedir:str, 
                                 analyze_population:str ='Excitatory', 
                                 n_smpl:int=2500):
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

        # logging.info('Calculating common neighbor bias...')
        analyze_gids = nodes.ids(analyze_population, sample=n_smpl)  # get the identifiers of target neurons
        # print(len(analyze_gids))
        connections = self._efferent_con_mat(edges,analyze_gids)
        # Let's look at the result
        vmin = np.min(connections) + 0.01   # +0.01 to avoid log(0) inside the plot
        vmax = np.max(connections)

        # plots adj matrix
        # ax = plt.figure().add_axes([0.1, 0.1, 0.8, 0.8])
        # ax.imshow(connections.toarray()[:1000, 9000:10000], cmap='Reds', norm=LogNorm(vmin=vmin, vmax=vmax))

        logging.info('Calculating common efferent neighbors...')
        def common_efferent_neighbors(M):
            CN = M.astype(int) * M.astype(int).transpose() # need to convert to int, so that neighbors are counted
            return CN.toarray()

        com_neighs = common_efferent_neighbors(connections)
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
            return numpy.histogram(numpy.triu(CN, 1), bins=bins)[0]

        xbins = numpy.arange(502) # count from 0 to 500 common neighbors #TODO: to be chosen from data
        H = cn_mat_to_histogram(com_neighs, xbins)

        plt.plot(H[:110])
        plt.yscale('log')
        plt.xlabel('Number of common neighbors')
        plt.ylabel('Number of neuron pairs')
        plt.savefig(f'{savedir}/common_neighbor_occurences.png')

        logging.info('Controlling common neighbor bias against Erdos-Renyi graph')
        def control_erdos_renyi_histogram(CN, N, bins):
            from scipy.stats import hypergeom
            out_degrees = numpy.diag(CN)
            '''Note: Here, we simply draw a random sample for each pair of neurons.
            Better, but more expensive would be to evaluate the probability mass function
            for all bins and for all pairs.'''
            expected = [hypergeom(N, d_A, out_degrees[(i+1):]).rvs()
                        for i, d_A in enumerate(out_degrees)]
            return numpy.histogram(numpy.hstack(expected), bins=bins)[0]

        H_ctrl_er = control_erdos_renyi_histogram(com_neighs, connections.shape[1], xbins)

        ax = plt.figure().add_axes([0.1, 0.1, 0.8, 0.8])
        ax.plot(xbins[:-1], H, color='red', marker='o', label='Experiment')
        ax.plot(xbins[:-1], H_ctrl_er, color='black', marker='o', label='Control (ER)')
        ax.set_yscale('log'); ax.legend(); ax.set_xlabel('Common neighbors'); ax.set_ylabel('Pairs')
        plt.savefig(f'{savedir}/common_neighbor_vs_ER.png')

        
if __name__ == "__main__":
    # from sys import argv
    # if argv[1] == 'test_barabasi':
    #     ba = igraph.Graph.Barabasi(10000,30,directed=False)
    #     analysis = GraphAnalysis(ba.to_networkx(),'in')
    #     analysis.plot_degree_distribution()
    #     analysis.plot_log_binned_degree_distribution(cumulative=True, 
    #                                                 fit_min_degree=50)

    #     from scipy.stats import norm, lognorm, expon, gamma
    #     # Define the list of distributions to test
    #     distributions = [norm, lognorm, expon, gamma]

    #     best_fit, best_params, best_sse = analysis.check_best_fit(distributions)

    #     print("Best fit for out-degree distribution:", best_fit.name)
    #     print("Best parameters for out-degree distribution:", best_params)
    #     print("SSE for out-degree distribution:", best_sse)

    # elif argv[1] == 'test_hop':
    # print('Testing Bandwith per group')
    adj_path = '/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/data/ca1_synaptome.npz'
    synaptome = sparse.load_npz(adj_path)

    analysis = GraphAnalysis(synaptome,'out')

    CIRCUIT_DIR =  '/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/sonata/circuit_config.json'
    circuit = Circuit(CIRCUIT_DIR)
    nodes = circuit.nodes["hippocampus_neurons"]
    edges = circuit.edges["hippocampus_neurons__hippocampus_neurons__chemical_synapse"]
    save_dir = '/gpfs/bbp.cscs.ch/project/proj112/home/kurban/topology_paper/data/common_neighbor'
    os.makedirs(save_dir,exist_ok=True)
    # mtypes_by_gid = c.nodes['hippocampus_neurons'].get().mtype.values
    # high,low = analysis.bandwidth_groups(2, mtypes_by_gid)

    logging.info('Testing common neighbor bias')
    analysis.common_neighbor_efferent(nodes,edges,save_dir,'Excitatory',n_smpl=2500)