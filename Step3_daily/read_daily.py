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
from netCDF4 import Dataset
import xarray as xr
from datetime import date, timedelta
import copy

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name, 'rb') as f:
        return pickle.load(f)

#==================================================    
workdir="/n/holyscratch01/jacob_lab/lshen/CH4/GEOS-Chem/Flexgrid/CPU_global/"
inputdir=workdir+"output/"
outputdir=workdir+"Step3_daily/data/"
sdate = date(2018, 5, 1)   #start date
edate = date(2020, 5, 29)  #end date

diff = edate - sdate
strdates=[]
for i in range(diff.days + 1):
    ap=sdate + timedelta(i)
    strdates.append(int(ap.strftime("%Y%m%d")))

files=glob.glob(inputdir+"*sparse_archive.pkl")
met=load_obj(files[0])
lon_out=met['lon'];lat_out=met['lat']
regrid_CH4=np.zeros((len(lon_out),len(lat_out),len(strdates)));regrid_CH4.fill(np.nan)
regrid_GC=np.zeros((len(lon_out),len(lat_out),len(strdates)));regrid_GC.fill(np.nan)

#====

for idate in range(len(strdates)):
    strdate=strdates[idate]
    print(strdate)
    loc=strdates.index(strdate)
    start=max(0,loc-1);end=min(loc+1,len(strdates))
    select_dates=strdates[(start):(end+1)]
    
    files=[]
    for x in select_dates:
        files.extend(glob.glob(inputdir+"*__"+str(x)+"T*_sparse_archive.pkl"))
    
    #---- assemble all data ----
    daily_result={}
    for ifile in range(len(files)):
        met=load_obj(files[ifile])
        for key, data in met.items():
            if(key in ['lon', 'lat']):
                continue
            if data['YYMMDD']==strdate:
                if key not in daily_result:
                    daily_result[key]={}
                    daily_result[key]=copy.deepcopy(data)
                    daily_result[key]['iterations']=1
                else:
                    item=daily_result[key]
                    item['count']+=data['count']
                    item['CH4']['fenzi']+=data['CH4']['fenzi']
                    item['CH4']['fenmu']+=data['CH4']['fenmu']
                    item['GC']['fenzi']+=data['GC']['fenzi']
                    item['GC']['fenmu']+=data['GC']['fenmu']
                    """
                    item['KK']['fenzi']+=data['KK']['fenzi']
                    item['KK']['fenmu']+=data['KK']['fenmu']
                    """
                    item['iterations']+=1
    
    #---- calculate the average ----                    
    daily={}
    daily['lon']=lon_out
    daily['lat']=lat_out
    for key, data in daily_result.items():
        if data['CH4']['fenmu']==0:
            continue
        TROPOMI=data['CH4']['fenzi']/data['CH4']['fenmu']
        GC=data['GC']['fenzi']/data['CH4']['fenmu']
        #Sensi=data['KK']['fenzi']/data['CH4']['fenmu']
        ij=data['ij']
        #daily[key]={'ij':ij,'TROPOMI':TROPOMI, 'GC':GC, 'Sensi':Sensi}
        daily[key]={'ij':ij,'TROPOMI':TROPOMI, 'GC':GC}
        regrid_CH4[ij[0],ij[1],idate]=TROPOMI
        regrid_GC[ij[0],ij[1],idate]=GC    
    #save_obj(daily,outputdir+str(strdate)+'.pkl')
#====save regridded data ===    
regrid_CH4=np.einsum('ijl->lji', regrid_CH4)
regrid_GC=np.einsum('ijl->lji', regrid_GC)

outputname=outputdir+'Daily_CH4.nc'
if os.path.isfile(outputname):
    os.remove(outputname)
dataset = Dataset(outputname,'w',format='NETCDF4_CLASSIC')

lat = dataset.createDimension('lat', len(lat_out))
lon = dataset.createDimension('lon', len(lon_out))
time = dataset.createDimension('time', len(strdates))

latitudes = dataset.createVariable('lat', 'f8',('lat',))
longitudes = dataset.createVariable('lon', 'f8',('lon',))
dates = dataset.createVariable('date', 'i',('time',))

nc_CH4 = dataset.createVariable('CH4', 'f8',('time','lat','lon'))
nc_GC = dataset.createVariable('GC', 'f8',('time','lat','lon'))

latitudes[:] = lat_out
longitudes[:] = lon_out
dates[:]=strdates
nc_CH4[:,:,:] = regrid_CH4
nc_GC[:,:,:] = regrid_GC
dataset.close()

