#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-10:00
#SBATCH --mem=10000
#SBATCH --mail-type=END

#export OMP_NUM_THREADS=$SLURM_NTASKS
python -u run_{run_num}.py
exit 0
#EOC
