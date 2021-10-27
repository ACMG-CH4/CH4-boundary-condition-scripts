#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-02:00
#SBATCH -p test
#SBATCH --mem=5000
#SBATCH --mail-type=END

#export OMP_NUM_THREADS=$SLURM_NTASKS
export OMP_NUM_THREADS=1
python run_{run_num}.py
exit 0
#EOC
