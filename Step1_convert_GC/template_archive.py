#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import numpy as np
import xarray as xr
import re
import pickle
import os
import pandas as pd 
import datetime

#----- define function -------
def save_obj(obj, name ):
    with open(name , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open( name, 'rb') as f:
        return pickle.load(f)
    
def read_tropomi(filename):
    met={}
    data=xr.open_dataset(filename,group="PRODUCT")
    met['methane']  = data['methane_mixing_ratio_bias_corrected'].values[0,:,:]# 3245x215
    met['qa_value'] = data['qa_value'].values[0,:,:]# 3245x215
    met['longitude']= data['longitude'].values[0,:,:]# 3245x215
    met['latitude'] = data['latitude'].values[0,:,:]# 3245x215
    referencetime   = data['time'].values # This is the reference time
    delta_time      = data['delta_time'][0].values # 3245x1
    strdate=[]
    if delta_time.dtype=='<m8[ns]':
        strdate=referencetime+delta_time
    elif delta_time.dtype=='<M8[ns]':
        strdate=delta_time
    else:
        print(delta_time.dtype)
        pass
    timeshift=np.array(met['longitude']/15*60,dtype=int)#convert to minutes
    met['utctime']=strdate
    
    localtimes=np.zeros(shape=timeshift.shape,dtype='datetime64[ns]')
    for kk in range(timeshift.shape[0]):
        #item=pd.to_timedelta(timeshift[kk,:], unit='m')
        #localtimes[kk,:]=strdate[kk]+item.values
        localtimes[kk,:]=strdate[kk]
    data.close()
        
    met['localtime']=localtimes
    
    data=xr.open_dataset(filename,group="PRODUCT/SUPPORT_DATA/DETAILED_RESULTS")
    met['column_AK']=data['column_averaging_kernel'].values[0,:,:,::-1]
    data.close()
    
    data=xr.open_dataset(filename,group="PRODUCT/SUPPORT_DATA/INPUT_DATA")
    met['methane_profile_apriori']=data['methane_profile_apriori'].values[0,:,:,::-1]#mol m-2
    pressure_interval=data['pressure_interval'].values[0,:,:]/100#Pa->hPa
    surface_pressure=data['surface_pressure'].values[0,:,:]/100
    met['dry_air_subcolumns']=data['dry_air_subcolumns'].values[0,:,:,::-1]#Pa -> hPa
    data.close()
    
    N1=met['methane'].shape[0]
    N2=met['methane'].shape[1]

    pressures=np.zeros([N1,N2,13], dtype=np.float)
    pressures.fill(np.nan)
    for i in range(12+1):
        pressures[:,:,i]=surface_pressure-i*pressure_interval
    
    met['pressures']=pressures
    
    return met    


def read_GC(date,use_Sensi=False):
    month=int(date[4:6])
    file_species="GEOSChem.SpeciesConc."+date+"00z.nc4"        
    file_pedge="GEOSChem.LevelEdgeDiags."+date+"00z.nc4"    
    file_troppause="GEOSChem.StateMet."+date+"00z.nc4"    
    
    #-- read CH4 ---
    filename=GC_datadir+'/'+ file_species
    data=xr.open_dataset(filename)
    LON=data['lon'].values
    LAT=data['lat'].values
    CH4 = data['SpeciesConc_CH4'].values[0,:,:,:];
    CH4 =CH4*1e9
    CH4=np.einsum('lij->jil',CH4)
    data.close()

    #-- read PEDGE ---
    filename=GC_datadir+'/'+ file_pedge
    data=xr.open_dataset(filename)
    PEDGE = data['Met_PEDGE'].values[0,:,:,:];
    PEDGE=np.einsum('lij->jil',PEDGE)
    data.close()
    
    #-- read TROPP ---
    filename=GC_datadir+'/'+ file_troppause
    data=xr.open_dataset(filename)
    TROPP = data['Met_TropLev'].values[0,:,:];
    TROPP=np.einsum('ij->ji',TROPP)
    data.close()    

    #--- correct the stratospheric ozone -----
    lat_loc=np.zeros(len(LAT),dtype=int)
    for j in range(len(LAT)):
        lat_loc[j]=np.argmin(abs(lat_mid - LAT[j]))
        
    CH4_adjusted=CH4.copy()
    for i in range(len(LON)):
        for j in range(len(LAT)):
            l=int(TROPP[i,j])
            ind=lat_loc[j]
            CH4_adjusted[i,j,l:]=CH4[i,j,l:]*lat_ratio[ind,month-1]

    met={}
    met['lon']=LON
    met['lat']=LAT
    met['PEDGE']=PEDGE
    met['CH4']=CH4
    met['CH4_adjusted']=CH4_adjusted
    met['TROPP']=TROPP

    #--- read sensitivity ---
    if use_Sensi:
        filename=Sensi_datadir+'/'+date+'.nc'
        data=xr.open_dataset(filename)
        Sensi=data['Sensi'].values
        Sensi=np.einsum('klji->ijlk',Sensi)
        data.close()  
        for i in range(len(LON)):
            for j in range(len(LAT)):
                l=int(TROPP[i,j])
                Sensi[i,j,l:,:]=Sensi[i,j,l:,:]*lat_ratio[j,month-1]
        met['Sensi']=Sensi

    return met

def read_all_GC(all_strdate, use_Sensi =False):
    met={}
    for strdate in all_strdate:
        met[strdate]= read_GC(strdate, use_Sensi) 
    return met

def cal_weights(Sat_p, GC_p):
    """
    Sat_p: pressure edge from TROPOMI (13)
    GC_p: presure edge from  GEOS-Chem (61)
    GC_CH4: methane from GEOS-Chem (60)
    """
    #Step 1: combine Sat_p and GC_p
    Com_p=np.zeros(len(Sat_p)+len(GC_p));Com_p.fill(np.nan)
    data_type=np.zeros(len(Sat_p)+len(GC_p),dtype=int);data_type.fill(-99)
    location=[]
    i=0;j=0;k=0
    while ((i<len(Sat_p)) or (j <len(GC_p))):
        if i==len(Sat_p):
            Com_p[k]=GC_p[j]
            data_type[k]=2
            j=j+1;k=k+1
            continue
        if j==len(GC_p):
            Com_p[k]=Sat_p[i]
            data_type[k]=1  
            location.append(k)        
            i=i+1;k=k+1   
            continue
        if(Sat_p[i]>=GC_p[j]):
            Com_p[k]=Sat_p[i]
            data_type[k]=1        
            location.append(k)        
            i=i+1;k=k+1
        else:
            Com_p[k]=GC_p[j]
            data_type[k]=2
            j=j+1;k=k+1
    
    #Step 2: find the first level of GC
    first_2=-99
    for i in range(len(Sat_p)+len(GC_p)-1):
        if data_type[i]==2:
            first_2=i
            break    
    
    weights={}
    weights['data_type']=data_type
    weights['Com_p']=Com_p
    weights['location']=location
    weights['first_2']=first_2
    
    return weights


def remap(GC_CH4, data_type, Com_p, location, first_2):
    conc=np.zeros(len(Com_p)-1,);conc.fill(np.nan)
    k=0
    for i in range(first_2,len(Com_p)-1):
        conc[i]=GC_CH4[k]
        if data_type[i+1]==2:
            k=k+1
    if first_2>0:
        conc[:first_2]=conc[first_2]
    
    #Step 4: calculate the weighted mean methane for each layer
    delta_p=Com_p[:-1]-Com_p[1:]
    Sat_CH4=np.zeros(12);Sat_CH4.fill(np.nan)
    for i in range(len(location)-1):
        start=location[i]
        end=location[i+1]
        fenzi=sum(conc[start:end]*delta_p[start:end])
        fenmu=sum(delta_p[start:end])
        Sat_CH4[i]=fenzi/fenmu   
    return Sat_CH4


def remap2(Sensi, data_type, Com_p, location, first_2):
    MM=Sensi.shape[1]
    conc=np.zeros((len(Com_p)-1,MM))
    conc.fill(np.nan)
    k=0
    for i in range(first_2,len(Com_p)-1):
        conc[i,:]=Sensi[k,:]
        if data_type[i+1]==2:
            k=k+1
    if first_2>0:
        conc[:first_2,:]=conc[first_2,:]

    Sat_CH4=np.zeros((12,MM));Sat_CH4.fill(np.nan)
    
    delta_p=Com_p[:-1]-Com_p[1:]
    delta_ps=np.transpose(np.tile(delta_p,(MM,1)))
    for i in range(len(location)-1):
        start=location[i]
        end=location[i+1]
        fenzi=np.sum(conc[start:end,:]*delta_ps[start:end,:],0)
        fenmu=np.sum(delta_p[start:end])
        Sat_CH4[i,:]=fenzi/fenmu
    
    return Sat_CH4

def use_AK_to_GC(filename,GC_startdate, GC_enddate, use_Sensi=False):
    TROPOMI=read_tropomi(filename)#read TROPOMI data

    sat_ind=np.where((TROPOMI['qa_value']>=0.5) & (TROPOMI['localtime']>=GC_startdate) & (TROPOMI['localtime']<=GC_enddate))     
    NN=len(sat_ind[0])
    if use_Sensi:
        MM=1009
        temp_KK=np.zeros([NN,MM],dtype=np.float32)#Store the K
    temp_obs_GC=np.zeros([NN,6],dtype=np.float32)#TROPOMI-CH4, GC-CH4, longitude,latitude, II, JJ
    
    #================================
    #--- now compute sensitivity ---
    #================================
    #--- generate all strdate----
    all_strdate=[]
    for iNN in range(NN):
        iSat=sat_ind[0][iNN]
        jSat=sat_ind[1][iNN]
        timeshift=int(TROPOMI['longitude'][iSat,jSat]/15*60)
        timeshift=0#Now I use UTC time instead of local time
        localtime=TROPOMI['utctime'][iSat]+np.timedelta64(timeshift,'m')#local time
        localtime=pd.to_datetime(str(localtime))
        #strdate=localtime.strftime("%Y%m%d_%H")
        strdate=localtime.round('60min').strftime("%Y%m%d_%H")        
        all_strdate.append(strdate)

    all_strdate=list(set(all_strdate))    
    all_date_GC=read_all_GC(all_strdate,use_Sensi)
    
    #temp_obs_GC=np.zeros([NN,4],dtype=np.float32)#
    for iNN in range(NN):
        iSat=sat_ind[0][iNN]
        jSat=sat_ind[1][iNN]
        Sat_p=TROPOMI['pressures'][iSat,jSat,:]
        dry_air_subcolumns=TROPOMI['dry_air_subcolumns'][iSat,jSat,:]#mol m-2
        priori=TROPOMI['methane_profile_apriori'][iSat,jSat,:]
        AK=TROPOMI['column_AK'][iSat,jSat,:]

        timeshift=int(TROPOMI['longitude'][iSat,jSat]/15*60)
        timeshift=0
        localtime=TROPOMI['utctime'][iSat]+np.timedelta64(timeshift,'m')#local time
        localtime=pd.to_datetime(str(localtime))
        #strdate=localtime.strftime("%Y%m%d")        
        strdate=localtime.round('60min').strftime("%Y%m%d_%H")        
        GC=all_date_GC[strdate]
                
        iGC=np.abs(GC['lon']-TROPOMI['longitude'][iSat,jSat]).argmin()
        jGC=np.abs(GC['lat']-TROPOMI['latitude'][iSat,jSat]).argmin()
        GC_p=GC['PEDGE'][iGC,jGC,:]
        
        if(abs(Sat_p[0]-GC_p[0])>=250):
            print('surface pressure large discrepancy',Sat_p[0]-GC_p[0])
        
        GC_CH4=GC['CH4_adjusted'][iGC,jGC,:]
        ww=cal_weights(Sat_p, GC_p)
        Sat_CH4=remap(GC_CH4, ww['data_type'], ww['Com_p'], ww['location'], ww['first_2'])
        Sat_CH4_2=Sat_CH4*1e-9*dry_air_subcolumns ##convert ppb to pressure mol m-2
        GC_base_posteri=sum(priori+AK*(Sat_CH4_2-priori))/sum(dry_air_subcolumns)*1e9

        temp_obs_GC[iNN,0]=TROPOMI['methane'][iSat,jSat]#TROPOMI methane
        temp_obs_GC[iNN,1]=GC_base_posteri # GC methane
        temp_obs_GC[iNN,2]=TROPOMI['longitude'][iSat,jSat] #TROPOMI longitude
        temp_obs_GC[iNN,3]=TROPOMI['latitude'][iSat,jSat]  #TROPOMI latitude 
        temp_obs_GC[iNN,4]=iSat #TROPOMI index of longitude
        temp_obs_GC[iNN,5]=jSat #TROPOMI index of lattitude
        if use_Sensi:
            Sensi=GC['Sensi'][iGC,jGC,:,:]
            Sat_CH4=remap2(Sensi, ww['data_type'], ww['Com_p'], ww['location'], ww['first_2'])
            AKs=np.transpose(np.tile(AK,(MM,1)))
            dry_air_subcolumns_s=np.transpose(np.tile(dry_air_subcolumns,(MM,1)))
            temp_KK[iNN,:]=np.sum(AKs*Sat_CH4*dry_air_subcolumns_s,0)/sum(dry_air_subcolumns)

        
    result={}
    if use_Sensi:
        result['KK']=temp_KK

    result['obs_GC']=temp_obs_GC
        
    return result
    
#==============================================================================
#===========================Define functions ==================================
#==============================================================================
use_Sensi = False    
workdir="/n/holyscratch01/jacob_lab/lshen/CH4/GEOS-Chem/Flexgrid/CPU_global/"
Sat_datadir=workdir+"data_TROPOMI/"
GC_datadir=workdir+"data_GC/"
outputdir=workdir+"data_converted/"
Sensi_datadir=workdir+"Sensi/"

os.chdir(workdir+"Step1_convert_GC")
#==== read lat_ratio ===
df=pd.read_csv("./lat_ratio.csv",index_col=0)
lat_mid=np.array(df.index)
lat_ratio=np.array(df.values)

GC_startdate=datetime.datetime.strptime("2019-12-30 00:00:00", '%Y-%m-%d %H:%M:%S')
GC_enddate=datetime.datetime.strptime("2020-05-29 23:59:59", '%Y-%m-%d %H:%M:%S')
GC_startdate=np.datetime64(GC_startdate)
GC_enddate=np.datetime64(GC_enddate)

#==== read Satellite ===
allfiles=glob.glob(Sat_datadir+'*.nc')
Sat_files=[]
for index in range(len(allfiles)):
    filename=allfiles[index]
    shortname=re.split('\/', filename)[-1]
    shortname=re.split('\.', shortname)[0]
    strdate=re.split('\.|_+|T',shortname)[4]
    strdate2 = datetime.datetime.strptime(strdate, '%Y%m%d')
    if ((strdate2>=GC_startdate) and (strdate2<=GC_enddate)):
        Sat_files.append(filename)

Sat_files.sort()
print("Number of files",len(Sat_files))

index=1
for index in range(({run_num}-1)*200,{run_num}*200):
    print('========================')
    filename=Sat_files[index]    
    temp=re.split('\/', filename)[-1]
    print(temp)
    date=re.split('\.',temp)[0]
    if os.path.isfile(outputdir+date+'_GCtoTROPOMI.pkl'):
        continue
    result=use_AK_to_GC(filename,GC_startdate,GC_enddate)
    save_obj(result,outputdir+date+'_GCtoTROPOMI.pkl')
