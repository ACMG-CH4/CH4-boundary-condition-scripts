#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import re
import xarray as xr
import glob
import os
from utilities.utils import mkdir, upload_boundary_conditions, read_config_file

config_filename = os.getcwd().rsplit("/", 1)[0] + "/boundary_condition_config.yml"
config = read_config_file(config_filename)
outputDir = config["paths"]["workdir"] + "smoothed-boundary-conditions"
os.chdir(config["paths"]["GC_datadir"])
file1 = xr.open_dataset(
    config["paths"]["workdir"] + "Bias_4x5_dk_2_updated.nc"
)
upload_to_s3 = True
all_Bias = file1["Bias"].values * 1e-9
strdate = file1["time"].values
file1.close()

all_Bias.mean() * 1e9

files = glob.glob("GEOSChem.BoundaryConditions*.nc4")
files.sort()
mkdir(outputDir)
for ifile in range(len(files)):
    filename = files[ifile]
    print(filename)
    temp = int(re.split("\.|_", filename)[2])
    ind = [index for index, date in enumerate(strdate) if date == temp]
    if len(ind) == 0:
        print("skipping file")
        continue
    ind = ind[0]
    Bias = all_Bias[ind, :, :]

    file2 = xr.open_dataset(filename)
    np.mean(file2["SpeciesBC_CH4"].values)
    orig_data = file2["SpeciesBC_CH4"].values.copy()
    for t in range(orig_data.shape[0]):
        for lev in range(orig_data.shape[1]):
            orig_data[t, lev, :, :] = orig_data[t, lev, :, :] - Bias
    file2["SpeciesBC_CH4"].values = orig_data
    file2.to_netcdf(f"{outputDir}/{filename}")
    file2.close()

# upload files to s3
if upload_to_s3:
    upload_boundary_conditions(outputDir, config["paths"]["outputBucket"])
