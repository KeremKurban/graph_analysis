#!/bin/bash
#SBATCH --job-name=dask-worker
#SBATCH --time=01:00:00
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --constraint=cpu
#SBATCH --partition=prod
#SBATCH --account=proj112

# activate the virtual environment
source /gpfs/bbp.cscs.ch/home/kurban/venv2022/bin/activate

# start the dask worker
/gpfs/bbp.cscs.ch/home/kurban/venv2022/bin/dask-worker $@
