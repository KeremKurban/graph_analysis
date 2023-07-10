#!/bin/bash
#SBATCH --job-name=rcc
#SBATCH --time=3:00:00
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --constraint=cpu
#SBATCH --partition=prod
#SBATCH --account=proj112

# activate the virtual environment
source /gpfs/bbp.cscs.ch/home/kurban/venv2022/bin/activate

# start the dask worker
python /gpfs/bbp.cscs.ch/project/proj112/home/kurban/topology_paper/toolbox/graph_analysis/examples/norm_rich_club_multi.py $1
