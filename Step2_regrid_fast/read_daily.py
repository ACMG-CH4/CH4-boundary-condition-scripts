#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 15:33:46 2017
@author: lulushen
"""
import glob
import numpy as np
import pandas as pd
import os
import pickle 
import re
import xarray as xr
from datetime import timedelta, date
from netCDF4 import Dataset

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name, 'rb') as f:
        return pickle.load(f)

def nearest_loc(loc0,table,tolerance=5):
    temp=np.abs(table-loc0)
    ind=temp.argmin()
    if temp[ind]>=tolerance:
        return np.nan
    else:
        return ind

def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)
        
#==================================================  
workdir="/n/holyscratch01/jacob_lab/lestrada/IMI/"
GC_datadir="/n/holyscratch01/jacob_lab/lshen/CH4/GEOS-Chem/Flexgrid_global/Global_4x5/OutputDir/"
# "/n/holyscratch01/jacob_lab/dvaron/archive/production_output_data/CH4_Jacobian_0000/OutputDir/"
data_converted=workdir+"data_converted_BC1/"
outputdir=workdir+"Step2_regrid_fast/"

GC_data=glob.glob(GC_datadir+'GEOSChem.SpeciesConc*.nc4')[0]
data=xr.open_dataset(GC_data)
LON=data['lon'].values
LAT=data['lat'].values
data.close()

#LON=np.arange(-180,180+0.625,0.625)
#LAT=np.arange(-90,90+0.5,0.5)

start_dt = date(2018, 5, 1)
end_dt = date(2020, 2, 28)
alldates=[]
for dt in daterange(start_dt, end_dt):
    alldates.append(dt.strftime("%Y%m%d"))

daily_OMI=np.zeros((len(LON), len(LAT), len(alldates)))
daily_GC=np.zeros((len(LON), len(LAT), len(alldates)))
daily_count=np.zeros((len(LON), len(LAT), len(alldates)))
daily_altitude=np.zeros((len(LON), len(LAT), len(alldates)))
daily_altitude_std=np.zeros((len(LON), len(LAT), len(alldates)))
daily_albedo1 =np.zeros((len(LON), len(LAT), len(alldates)))
daily_albedo2 =np.zeros((len(LON), len(LAT), len(alldates)))
daily_AOT1 =np.zeros((len(LON), len(LAT), len(alldates)))
daily_AOT2 =np.zeros((len(LON), len(LAT), len(alldates)))

files=glob.glob(data_converted+'*.pkl')


files.sort()
print("Number of files",len(files))

index=12
for index in range(len(files)):
    print(index)
    filename=files[index]    
    temp=re.split('\/', filename)[-1]
    date=re.split('\.|_+|T',temp)[4]
    #date=re.split('\.',temp)[0]
    
    #read converted data
    met=load_obj(filename)
    obs_GC=met['obs_GC']
    NN=obs_GC.shape[0]

    if NN==0:
        continue
    
    time_ind=alldates.index(date)
    for iNN in range(NN):
        c_OMI,c_GC,lon0,lat0=obs_GC[iNN,:4]
        ii=nearest_loc(lon0, LON)
        jj=nearest_loc(lat0, LAT)
        altitude=obs_GC[iNN,6]
        albedo1 =obs_GC[iNN,7]
        albedo2 = obs_GC[iNN,8]
        AOT1 =obs_GC[iNN,10]
        AOT2 = obs_GC[iNN,11]
        altitude_std=obs_GC[iNN,9]
        daily_OMI[ii,jj,time_ind]+=c_OMI
        daily_GC[ii,jj,time_ind]+=c_GC
        daily_count[ii,jj,time_ind]+=1
        daily_altitude[ii,jj,time_ind]+=altitude
        daily_albedo1[ii,jj,time_ind]+=albedo1
        daily_albedo2[ii,jj,time_ind]+=albedo2
        daily_altitude_std[ii,jj,time_ind]+=altitude_std
        daily_AOT1[ii,jj,time_ind]+=AOT1
        daily_AOT2[ii,jj,time_ind]+=AOT2
#---
daily_count[daily_count==0]=np.nan
daily_OMI=daily_OMI/daily_count
daily_GC=daily_GC/daily_count
daily_altitude = daily_altitude/daily_count
daily_altitude_std = daily_altitude_std/daily_count
daily_albedo1 = daily_albedo1/daily_count
daily_albedo2 = daily_albedo2/daily_count
daily_AOT1 = daily_AOT1/daily_count
daily_AOT2 = daily_AOT2/daily_count        
print(np.nanmax(daily_altitude))

diff = daily_OMI - daily_GC
std=np.zeros((len(LON),len(LAT)));std.fill(np.nan)
for i in range(len(LON)):
    for j in range(len(LAT)):
        x=diff[i,j,:]
        if np.sum(~np.isnan(x))>=5:
            std[i,j]=np.nanstd(x,ddof=1)
#std = np.nanstd(diff, axis=2, ddof=1)
m=np.nanmean(std)
std[np.isnan(std)]=m
std[std>=30]=30
std[std<=10]=10

daily_altitude=np.nanmean(daily_altitude,axis=2)
daily_altitude_std=np.nanmean(daily_altitude_std,axis=2)
daily_albedo1=np.nanmean(daily_albedo1,axis=2)
daily_albedo2=np.nanmean(daily_albedo2,axis=2)
daily_AOT1=np.nanmean(daily_AOT1,axis=2)
daily_AOT2=np.nanmean(daily_AOT2,axis=2)

#=======================================
#=======================================
#====save regridded data ===    
regrid_CH4=np.einsum('ijl->lji', daily_OMI)
regrid_GC=np.einsum('ijl->lji', daily_GC)
regrid_error=np.einsum('ij->ji', std)
regrid_altitude=np.einsum('ij->ji', daily_altitude)
regrid_altitude_std=np.einsum('ij->ji', daily_altitude_std)
regrid_albedo1=np.einsum('ij->ji', daily_albedo1)
regrid_albedo2=np.einsum('ij->ji', daily_albedo2)
regrid_AOT1=np.einsum('ij->ji', daily_AOT1)
regrid_AOT2=np.einsum('ij->ji', daily_AOT2)

outputname=outputdir+'Daily_CH4.nc'
if os.path.isfile(outputname):
    os.remove(outputname)

dataset = Dataset(outputname,'w',format='NETCDF4_CLASSIC')

lat = dataset.createDimension('lat', len(LAT))
lon = dataset.createDimension('lon', len(LON))
time = dataset.createDimension('time', len(alldates))

latitudes = dataset.createVariable('lat', 'f8',('lat',))
longitudes = dataset.createVariable('lon', 'f8',('lon',))
dates = dataset.createVariable('date', 'i',('time',))

nc_CH4 = dataset.createVariable('CH4', 'f8',('time','lat','lon'))
nc_GC = dataset.createVariable('GC', 'f8',('time','lat','lon'))
nc_error = dataset.createVariable('error', 'f8',('lat','lon'))
nc_altitude = dataset.createVariable('altitude', 'f8',('lat','lon'))
nc_altitude_std = dataset.createVariable('altitude_std', 'f8',('lat','lon'))
nc_albedo1 = dataset.createVariable('albedo1', 'f8',('lat','lon'))
nc_albedo2 = dataset.createVariable('albedo2', 'f8',('lat','lon'))
nc_AOT1 = dataset.createVariable('AOT1', 'f8',('lat','lon'))
nc_AOT2 = dataset.createVariable('AOT2', 'f8',('lat','lon'))

latitudes[:] = LAT
longitudes[:] = LON
dates[:]=alldates
nc_CH4[:,:,:] = regrid_CH4
nc_GC[:,:,:] = regrid_GC
nc_error[:,:]=regrid_error
nc_altitude[:,:]= regrid_altitude
nc_altitude_std[:,:]= regrid_altitude_std
nc_albedo1[:,:]= regrid_albedo1
nc_albedo2[:,:]= regrid_albedo2
nc_AOT1[:,:]= regrid_AOT1
nc_AOT2[:,:]= regrid_AOT2
dataset.close() 
