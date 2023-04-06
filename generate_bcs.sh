#!/bin/bash

#SBATCH -c 8
#SBATCH -N 1
#SBATCH -t 0-48:00
#SBATCH --partition=seas_compute,huce_cascade,huce_intel
#SBATCH --mem=15000
#SBATCH --mail-type=END
#SBATCH -o bc_output.log

source ~/envs/gcc_cmake.gfortran102_cannon.env
export PYTHONPATH="/n/holylfs05/LABS/jacob_lab/lestrada/IMI/BCs/CH4-boundary-condition-scripts"

set -e

cd Step1_convert_GC
python template_archive.py

echo "Done with step 1"

cd ../Step2_regrid_fast
python read_daily.py

echo "Done with step 2"

cd ../Step3_correct_background
python calculate_bias.py

echo "Done with step 3"
cd ../Step4_write_boundary
python write_boundary.py
