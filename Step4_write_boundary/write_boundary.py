#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import re
import xarray as xr
import glob
import os

os.chdir("/n/holyscratch01/jacob_lab/lshen/CH4/GEOS-Chem/Flexgrid/Global_4x5_ND49/OutputDir")
file1=xr.open_dataset("../OutputDir_bias_corrected_dk_2/Bias_4x5_dk_2.nc")
all_Bias = file1['Bias'].values*1e-9
strdate=file1['time'].values
file1.close()

all_Bias.mean()*1e9


files=glob.glob("GEOSChem.BoundaryConditions*.nc4");files.sort()

for ifile in range(len(files)):
    filename=files[ifile]
    print(filename)
    temp=int(re.split("\.|_",filename)[2])
    ind=np.where(strdate==temp)[0]
    if len(ind)==0:
        continue
    ind=ind[0]
    Bias=all_Bias[ind,:,:]
    
    file2=xr.open_dataset(filename)
    np.mean(file2['SpeciesBC_CH4'].values)
    orig_data=file2['SpeciesBC_CH4'].values.copy()
    for t in range(8):
        for lev in range(47):
            orig_data[t,lev,:,:]=orig_data[t,lev,:,:]-Bias
    file2['SpeciesBC_CH4'].values=orig_data
    file2.to_netcdf("../OutputDir_bias_corrected_dk_2/"+filename)
    file2.close()
