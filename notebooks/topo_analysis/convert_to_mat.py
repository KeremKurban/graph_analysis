import numpy as np
from scipy.io import savemat
from scipy import sparse

# Load the npz file
data_dir = '/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/data/'
filename = 'ca1_ca3_to_cylinder300_synaptome.npz'


sparse_matrix = sparse.load_npz(f'{data_dir}/{filename}')
sparse_matrix[sparse_matrix>1] = 1
print(np.unique(sparse_matrix))

# Save the data as a MATLAB .mat file

mat_dict = {'sparse_matrix': sparse_matrix}

savemat(f'{data_dir}/ca1_ca3_to_cylinder300_synaptome.mat', mat_dict, do_compression=True)

