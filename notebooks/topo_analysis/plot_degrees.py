import pandas as pd
import matplotlib.pyplot as plt

indegree_ca1 = pd.read_csv('/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/data/indegree_CA1.csv',index_col=0)
outdegree_ca1 = pd.read_csv('/gpfs/bbp.cscs.ch/project/proj112/circuits/CA1/20211110-BioM/data/outdegree_CA1.csv',index_col=0)

plt.figure(figsize=(10,5))
mt_groupby_in= indegree_ca1.groupby('mtype').mean()
mt_groupby_in.plot.bar(yerr=indegree_ca1.groupby('mtype').std())
plt.title('Mean Indegree by m-type')
plt.xlabel('Post m-types')

plt.figure(figsize=(10,5))
mt_groupby_out = outdegree_ca1.groupby('mtype').mean()
mt_groupby_out.plot.bar(yerr=outdegree_ca1.groupby('mtype').std())
plt.title('Mean Outdegree by m-type')
plt.xlabel('Pre m-types')