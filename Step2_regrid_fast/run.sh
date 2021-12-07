#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH -t 0-01:50
#SBATCH -p test
#SBATCH --mem=40000
#SBATCH --mail-type=END

#export OMP_NUM_THREADS=$SLURM_NTASKS
python read_daily.py
exit 0
#EOC
