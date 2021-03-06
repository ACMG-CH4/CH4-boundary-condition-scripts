#!/bin/bash
for index in {1..1};do

INP="template_run.sh"
sed -e "s/{run_num}/$index/ig"  $INP > tmp.input
mv tmp.input "optim_"$index".sh"
chmod 755 "optim_"$index".sh"

INP="template_archive.py"
sed -e "s/{run_num}/$index/ig"  $INP > tmp.input
mv tmp.input "run_"$index".py"

# TODO: enable sbatch on aws
# sbatch "optim_"$index".sh"
done
