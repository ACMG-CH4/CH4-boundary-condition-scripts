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
import datetime

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name, 'rb') as f:
        return pickle.load(f)

def resample(satellite,spdata):
    fenzi=0
    fenmu=0
    for item in satellite:
        name='_'.join([str(item[0]),str(item[1])])
        if name not in lookup:
            print("resample error: not found in lookup")
            continue
        iNN=lookup[name]
        fenzi=fenzi+spdata[iNN]*item[2]
        fenmu=fenmu+item[2]
    met={}
    met['fenzi']=fenzi
    met['fenmu']=fenmu
    return met

def resample_KK(satellite,KK):
    fenzi=0
    fenmu=0
    for item in satellite:
        name='_'.join([str(item[0]),str(item[1])])
        if name not in lookup:
            print("resample error: not found in lookup")
            continue
        iNN=lookup[name]
        fenzi=fenzi+KK[iNN,:]*item[2]
        fenmu=fenmu+item[2]
    met={}
    met['fenzi']=fenzi
    met['fenmu']=fenmu
    return met

#==================================================  
use_Sensi = False    
workdir="/n/holyscratch01/jacob_lab/lshen/CH4/GEOS-Chem/Flexgrid/CPU_global/" 
Sat_datadir=workdir+"data_TROPOMI/"
input_weights=workdir+"weights/"
data_converted=workdir+"data_converted/"
outputdir=workdir+"output/"

GC_startdate=datetime.datetime.strptime("2018-05-01 00:00:00", '%Y-%m-%d %H:%M:%S')
GC_enddate=datetime.datetime.strptime("2019-12-30 23:59:59", '%Y-%m-%d %H:%M:%S')
GC_startdate=datetime.datetime.strptime("2019-12-30 00:00:00", '%Y-%m-%d %H:%M:%S')
GC_enddate=datetime.datetime.strptime("2020-05-29 23:59:59", '%Y-%m-%d %H:%M:%S')
GC_startdate=np.datetime64(GC_startdate)
GC_enddate=np.datetime64(GC_enddate)

allfiles=glob.glob(Sat_datadir+'*.nc')
files=[]
for index in range(len(allfiles)):
    filename=allfiles[index]
    shortname=re.split('\/', filename)[-1]
    shortname=re.split('\.', shortname)[0]
    strdate=re.split('\.|_+|T',shortname)[4]
    strdate2 = datetime.datetime.strptime(strdate, '%Y%m%d')
    if ((strdate2>=GC_startdate) and (strdate2<=GC_enddate)):
        files.append(filename)

files.sort()
print("Number of files",len(files))

index=1
for index in range(({run_num}-1)*600,{run_num}*600):
#for index in range(len(files)):
    print('========================')
    filename=files[index]    
    temp=re.split('\/', filename)[-1]
    date=re.split('\.',temp)[0]

    outputname=outputdir+date+'_sparse_archive.pkl'
    if os.path.isfile(outputname):
        continue
    
    #read converted data
    met=load_obj(data_converted+date+'_GCtoTROPOMI.pkl')
    obs_GC=met['obs_GC']
    #KK=met['KK']    
    NN=obs_GC.shape[0]
            
    #read lon/lat for 4 vertices
    data=xr.open_dataset(files[index],group="PRODUCT/SUPPORT_DATA/GEOLOCATIONS")
    longitude_bounds=data['longitude_bounds'].values[0,:,:,:]
    latitude_bounds=data['latitude_bounds'].values[0,:,:,:]    
    data.close()
    
    #read CH4, time (convert to local time), 
    data=xr.open_dataset(files[index],group="PRODUCT")
    methane=data['methane_mixing_ratio_bias_corrected'].values[0,:,:]
    longitude=data['longitude'].values[0,:,:]
    latitude=data['latitude'].values[0,:,:]
    qa_value=data['qa_value'].values[0,:,:]
    referencetime=data['time'].values
    delta_time=data['delta_time'][0].values
    strdate=[]
    if delta_time.dtype=='<m8[ns]':
        strdate=referencetime+delta_time
    elif delta_time.dtype=='<M8[ns]':
        strdate=delta_time
    else:
        print(delta_time.dtype)
        pass
    timeshift=np.array(longitude/15*60,dtype=int)#convert to minutes
            
    localtimes=np.zeros(shape=timeshift.shape,dtype='datetime64[ns]')
    for kk in range(timeshift.shape[0]):
        item=pd.to_timedelta(timeshift[kk,:], unit='m')
        localtimes[kk,:]=strdate[kk]+item.values
    data.close()
        
    #--- remove the data with low quality ----
    # This is based on the NA domain
    ss2=load_obj(input_weights+date+'.pkl')
    #'lon', 'lat', 'xlim', 'ylim', 'delta_lon', 'delta_lat', 'weights', 'grid_ij'
    grid_ij=ss2['grid_ij']
    weights=ss2['weights']
    delta_lon=ss2['delta_lon'];delta_lat=ss2['delta_lat']
    lon_in=ss2['lon'];lat_in=ss2['lat']

    weights_new={}#It stores the lon_lat and all overlapping satellite pixels
    k=0
    for k in range(len(weights)):
        items=weights[k]
        if (len(items)>0):
            iGC=grid_ij[k][0]
            jGC=grid_ij[k][1]            
            for item in items[:]:
                i=item[0];j=item[1]
                HH=pd.to_datetime(localtimes[i,j]).hour
                #if not ( (qa_value[i,j]>=0.5) and (not np.isnan(methane[i,j])) and (HH>=9) and (HH<=17)):
                if not ( (qa_value[i,j]>=0.5) and (not np.isnan(methane[i,j]))):
                    items.remove(item)
            if len(items)>0:
                species='{:+09.4f}'.format(lon_in[iGC])+'_'+'{:+09.4f}'.format(lat_in[jGC])
                weights_new[species]=items.copy()


    # Now I need to subset the domain
    list1=list(obs_GC[:,4].astype(np.int))
    list2=list(obs_GC[:,5].astype(np.int))
    ii_jj=['_'.join(map(str,i)) for i in zip(list1,list2)]
    lookup={}
    for k in range(len(ii_jj)):
        lookup[ii_jj[k]]=k

    #xlim=[-107.8125,-80.9375];ylim=[10,36]
    #delta_lon=0.3125;delta_lat=0.25
    #lon_out=np.linspace(xlim[0],xlim[1],num=int((xlim[1]-xlim[0])/delta_lon+1))
    #lat_out=np.linspace(ylim[0],ylim[1],num=int((ylim[1]-ylim[0])/delta_lat+1))
    lon_out=lon_in
    lat_out=lat_in
    
    sparse_archive={}
    sparse_archive['lon']=lon_out
    sparse_archive['lat']=lat_out
    for i in range(len(lon_out)):#i=35;j=92
        for j in range(len(lat_out)):
            species='{:+09.4f}'.format(lon_out[i])+'_'+'{:+09.4f}'.format(lat_out[j])
            if species in weights_new:
                satellite=weights_new[species]
                ap=satellite[0]
                ap2=localtimes[ap[0],ap[1]]
                ts = pd.to_datetime(str(ap2))
                name=str(i)+"_"+str(j)
                sparse_archive[name]={}                
                sparse_archive[name]['ij']=[i,j]
                sparse_archive[name]['YYMMDD']=int(ts.strftime('%Y%m%d'))
                sparse_archive[name]['HH']=int(ts.strftime('%H'))
                sparse_archive[name]['count']=len(satellite)
                sparse_archive[name]['CH4']=resample(satellite,obs_GC[:,0])
                sparse_archive[name]['GC']=resample(satellite,obs_GC[:,1])# 0 is for CH4, 1 is for GC
                #sparse_archive[name]['KK']=resample_KK(satellite,KK)
		
                                
    save_obj(sparse_archive,outputname)
