#!/bin/bash

#SBATCH -c 8
#SBATCH -N 1
#SBATCH -t 0-24:00
#SBATCH -p huce_intel
#SBATCH --mem=15000
#SBATCH --mail-type=END
#SBATCH -o bc_output.log

source ~/envs/gcc_cmake.gfortran102_cannon.env
export PYTHONPATH="/n/holyscratch01/jacob_lab/lestrada/IMI/CH4-boundary-condition-scripts"

module load gcc/9.3.0-fasrc01 R/4.0.5-fasrc02

cd Step1_convert_GC
./template_archive.py

echo "Done with step 1"

cd ../Step2_regrid_fast
# ./read_daily.py

echo "Done with step 2"

cd ../Step3_correct_background
#Rscript correct_final.R

echo "Done with step 3"
cd ../Step4_write_boundary
# ./write_boundary.py
