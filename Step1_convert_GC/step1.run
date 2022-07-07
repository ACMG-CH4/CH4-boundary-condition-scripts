#!/bin/bash

#SBATCH -c 8
#SBATCH -N 1
#SBATCH -t 1-20:00
#SBATCH -p huce_intel
#SBATCH --mem=15000
#SBATCH --mail-type=END

###############################################################################
### Sample GEOS-Chem run script for SLURM
### You can increase the number of cores with -c and memory with --mem,
### particularly if you are running at very fine resolution (e.g. nested-grid)
###############################################################################

# Set the proper # of threads for OpenMP
# SLURM_CPUS_PER_TASK ensures this matches the number you set with -c above
export OMP_NUM_THREADS=8

source ~/envs/gcc_cmake.gfortran102_cannon.env
export PYTHONPATH="/n/holyscratch01/jacob_lab/lestrada/IMI/CH4-boundary-condition-scripts"

# Run GEOS_Chem.  The "time" command will return CPU and wall times.
# Stdout and stderr will be directed to the "GC.log" log file
# (you can change the log file name below if you wish)
srun -c $OMP_NUM_THREADS time -p ./template_archive.py >> step1.log

# Exit normally
exit 0
#EOC