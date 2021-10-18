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
from shapely.geometry import Polygon

#----- define function -------
def save_obj(obj, name ):
    with open(name , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open( name, 'rb') as f:
        return pickle.load(f)
    
def read_tropomi(filename):
    met={}
    data=xr.open_dataset(filename)
    data.close()
    met['methane']  = data['xch4_corrected'].values# 51975
    met['qa_value'] = data['qa_value'].values# 51975
    met['longitude']= data['longitude_center'].values# 51975
    met['latitude'] = data['latitude_center'].values# 51975
    met['surface_albedo']=data['surface_albedo'].values
    met['aerosol_optical_thickness']=data['aerosol_optical_thickness'].values        
    met['surface_altitude']=data['surface_altitude'].values
    # met['surface_altitude_stdv']=data['surface_altitude_stdv'].values not needed for IMI workflow
    dates = pd.DataFrame(data['time'].values[:,:-1],
                         columns=['year', 'month', 'day', 'hour', 'minute', 'second'])
    dates = pd.to_datetime(dates)#.dt.strftime('%Y%m%dT%H%M%S')
    met['utctime']=dates
    met['localtime']=dates
    met['column_AK']=data['xch4_column_averaging_kernel'].values[:,::-1]#51975x12    
    met['methane_profile_apriori']=data['ch4_profile_apriori'].values[:,::-1]
    pressure_interval=data['dp'].values #hPa
    surface_pressure=data['surface_pressure'].values #hPa
    met['dry_air_subcolumns']=data['dry_air_subcolumns'].values[:,::-1]
    met['longitude_bounds']=data['longitude_corners'].values
    met['latitude_bounds']=data['latitude_corners'].values    
    N1=met['methane'].shape[0]
    pressures=np.zeros([N1,13], dtype=np.float)
    pressures.fill(np.nan)
    for i in range(12+1):
        pressures[:,i]=surface_pressure-i*pressure_interval
    
    met['pressures']=pressures    
    return met    


def read_GC(date,use_Sensi=False):
    month=int(date[4:6])
    file_species="GEOSChem.SpeciesConc."+date+"00z.nc4"        
    file_pedge="GEOSChem.LevelEdgeDiags."+date+"00z.nc4" 
    
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
    
    # Note: the State Met file is used to adjust methane values in the stratosphere
    # for the IMI workflow we do not adjust for this bias because we are using 
    # nested domains, so we will not use it here either. -lae 10/15/2021   
    # file_troppause="GEOSChem.StateMet."+date+"00z.nc4"   
    #-- read TROPP ---
    # filename=GC_datadir+'/'+ file_troppause
    # data=xr.open_dataset(filename)
    # TROPP = data['Met_TropLev'].values[0,:,:];
    # TROPP=np.einsum('ij->ji',TROPP)
    # data.close()    
        
    #--- read base GC -----
    CH4_adjusted=CH4.copy()
    # for i in range(len(LON)):
    #     for j in range(len(LAT)):
    #         l=int(TROPP[i,j])
    #         ind=np.where(lat_mid == LAT[j])[0][0]### find the location of lat in lat_mid        
    #         CH4_adjusted[i,j,l:]=CH4[i,j,l:]*lat_ratio[ind,month-1]

    met={}
    met['lon']=LON
    met['lat']=LAT
    met['PEDGE']=PEDGE
    met['CH4']=CH4
    met['CH4_adjusted']=CH4_adjusted
    # met['TROPP']=TROPP

    #--- read sensitivity ---
    if use_Sensi:
        filename=Sensi_datadir+'/'+date+'.nc'
        data=xr.open_dataset(filename)
        Sensi=data['Sensi'].values
        Sensi=np.einsum('klji->ijlk',Sensi)
        data.close()
        #--- now adjust the Sensitivity
        Sensi=Sensi*2#Because we perturb the emissions by 50%
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

def use_AK_to_GC(filename,GC_startdate, GC_enddate, use_Sensi,xlim,ylim,):
    TROPOMI=read_tropomi(filename)#read TROPOMI data
    kuadu=np.max(TROPOMI['longitude_bounds'],1) - np.min(TROPOMI['longitude_bounds'],1)
    sat_ind=np.where((TROPOMI['longitude']>xlim[0]) & (TROPOMI['longitude']<xlim[1]) & (TROPOMI['latitude']>ylim[0]) & (TROPOMI['latitude']<ylim[1]) & (TROPOMI['qa_value']>=0.5) & (TROPOMI['utctime']>=GC_startdate) & (TROPOMI['utctime']<=GC_enddate) &  (TROPOMI['methane']<=3000) & (kuadu<=10) )
    NN=len(sat_ind[0])
    print(NN)
    if use_Sensi:
        MM=N_pert
        temp_KK=np.zeros([NN,MM],dtype=np.float32)#Store the K
        temp_KK.fill(np.nan)
        
    temp_obs_GC=np.zeros([NN,6+6],dtype=np.float32)#TROPOMI-CH4, GC-CH4, longitude,latitude, II, JJ
    temp_obs_GC.fill(np.nan)
    
    #================================
    #--- now compute sensitivity ---
    #================================
    #--- generate all strdate----
    all_strdate=[]
    for iNN in range(NN):
        iSat=sat_ind[0][iNN]
        localtime=TROPOMI['utctime'][iSat]
        localtime=pd.to_datetime(str(localtime))
        strdate=localtime.round('60min').strftime("%Y%m%d_%H")
        all_strdate.append(strdate)

    all_strdate=list(set(all_strdate))    
    all_date_GC=read_all_GC(all_strdate,use_Sensi)

    for iNN in range(NN):
        iSat=sat_ind[0][iNN]
        Sat_p=TROPOMI['pressures'][iSat,:]
        dry_air_subcolumns=TROPOMI['dry_air_subcolumns'][iSat,:]#mol m-2
        priori=TROPOMI['methane_profile_apriori'][iSat,:]
        AK=TROPOMI['column_AK'][iSat,:]

        timeshift=0
        localtime=TROPOMI['utctime'][iSat]+np.timedelta64(timeshift,'m')#local time
        localtime=pd.to_datetime(str(localtime))
        strdate=localtime.round('60min').strftime("%Y%m%d_%H")
        GC=all_date_GC[strdate]
                        
        #===========
        longitude_bounds=TROPOMI['longitude_bounds'][iSat,:]
        latitude_bounds=TROPOMI['latitude_bounds'][iSat,:]
        corners_lon=[];corners_lat=[]
        for k in range(4):
            iGC=nearest_loc(longitude_bounds[k], GC['lon'])
            jGC=nearest_loc(latitude_bounds[k], GC['lat'])
            corners_lon.append(iGC)
            corners_lat.append(jGC)

        GC_ij=[(x,y) for x in set(corners_lon) for y in set(corners_lat)]
        # error check for nans to skip creating gc grid if needed
        if np.nan in [item for sublist in GC_ij for item in sublist]:
            continue
        GC_grids=[(GC['lon'][i], GC['lat'][j]) for i,j in GC_ij]
        
        overlap_area=np.zeros(len(GC_grids))
        dlon=GC['lon'][3]-GC['lon'][2]
        dlat=GC['lat'][3]-GC['lat'][2]
        p0=Polygon(np.column_stack((longitude_bounds,latitude_bounds)))
        
        for ipixel in range(len(GC_grids)):
            item=GC_grids[ipixel]
            ap1=[item[0]-dlon/2, item[0]+dlon/2, item[0]+dlon/2, item[0]-dlon/2]
            ap2=[item[1]-dlat/2, item[1]-dlat/2, item[1]+dlat/2, item[1]+dlat/2]        
            p2=Polygon(np.column_stack((ap1,ap2)))
            if p2.intersects(p0):
                  overlap_area[ipixel]=p0.intersection(p2).area

        if sum(overlap_area)==0:
            continue                  

        #===========
        GC_base_posteri=0
        GC_base_sensi=0
        for ipixel in range(len(GC_grids)):
            iGC,jGC=GC_ij[ipixel]
            GC_p=GC['PEDGE'][iGC,jGC,:]                
            GC_CH4=GC['CH4_adjusted'][iGC,jGC,:]
            ww=cal_weights(Sat_p, GC_p)
            Sat_CH4=remap(GC_CH4, ww['data_type'], ww['Com_p'], ww['location'], ww['first_2'])
            Sat_CH4_2=Sat_CH4*1e-9*dry_air_subcolumns ##convert ppb to pressure mol m-2
            GC_base_posteri=GC_base_posteri+overlap_area[ipixel]*sum(priori+AK*(Sat_CH4_2-priori))/sum(dry_air_subcolumns)*1e9
            if use_Sensi:            
                #GC_base_sensi=GC_base_sensi+overlap_area[ipixel]*GC['Sensi'][iGC,jGC,:]
                """
                pedge=GC['PEDGE'][iGC,jGC,:]
                pp=pedge[:47]-pedge[1:]
                pedges=np.transpose(np.tile(pp,(MM,1)))
                ap=GC['Sensi'][iGC,jGC,:,:]*pedges                                
                Sensi=np.sum(ap,0)/(pedge[0]-pedge[-1])
                GC_base_sensi=GC_base_sensi+overlap_area[ipixel]*Sensi
                """
                Sensi=GC['Sensi'][iGC,jGC,:,:]
                Sat_CH4=remap2(Sensi, ww['data_type'], ww['Com_p'], ww['location'], ww['first_2'])                
                AKs=np.transpose(np.tile(AK,(MM,1)))
                dry_air_subcolumns_s=np.transpose(np.tile(dry_air_subcolumns,(MM,1)))
                ap=np.sum(AKs*Sat_CH4*dry_air_subcolumns_s,0)/sum(dry_air_subcolumns)
                GC_base_sensi=GC_base_sensi+overlap_area[ipixel]*ap
                                
        temp_obs_GC[iNN,0]=TROPOMI['methane'][iSat]#TROPOMI methane
        temp_obs_GC[iNN,1]=GC_base_posteri/sum(overlap_area) # GC methane
        temp_obs_GC[iNN,2]=TROPOMI['longitude'][iSat] #TROPOMI longitude
        temp_obs_GC[iNN,3]=TROPOMI['latitude'][iSat]  #TROPOMI latitude 
        temp_obs_GC[iNN,4]=iSat #TROPOMI index of longitude
        temp_obs_GC[iNN,5]=0 #TROPOMI index of lattitude
        temp_obs_GC[iNN,6]=TROPOMI['surface_altitude'][iSat] #TROPOMI index of lattitude        
        temp_obs_GC[iNN,7]=TROPOMI['surface_albedo'][iSat,0] #TROPOMI index of lattitude
        temp_obs_GC[iNN,8]=TROPOMI['surface_albedo'][iSat,1] #TROPOMI index of lattitude
        # temp_obs_GC[iNN,9]=TROPOMI['surface_altitude_stdv'][iSat]
        temp_obs_GC[iNN,9]=TROPOMI['aerosol_optical_thickness'][iSat,0] #AOT for NIR
        temp_obs_GC[iNN,10]=TROPOMI['aerosol_optical_thickness'][iSat,1] #AOT for SWIR
        if use_Sensi:
            temp_KK[iNN,:]=GC_base_sensi/sum(overlap_area)
        
    result={}
    if use_Sensi:
        result['KK']=temp_KK

    result['obs_GC']=temp_obs_GC
        
    return result


def nearest_loc(loc0,table,tolerance=5):
    temp=np.abs(table-loc0)
    ind=temp.argmin()
    if temp[ind]>=tolerance:
        return np.nan
    else:
        return ind

#==============================================================================
#===========================Define functions ==================================
#==============================================================================
use_Sensi = False
N_pert=156
xlim=[-180, 180]
ylim=[-90, 90]

# workdir="/n/holyscratch01/jacob_lab/lshen/CH4/GEOS-Chem/Flexgrid_global/CPU_global_Lorente/"
workdir="/n/holyscratch01/jacob_lab/lestrada/IMI/"
Sat_datadir="/n/seasasfs02/CH4_inversion/InputData/Obs/TROPOMI/"
GC_datadir="/n/holyscratch01/jacob_lab/lshen/CH4/GEOS-Chem/Flexgrid_global/Global_4x5/OutputDir/"
outputdir=workdir+"data_converted_BC/"
Sensi_datadir=workdir+"Sensi/"
scriptdir="/n/home03/lestrada/projects/IMI/CH4-boundary-condition-scripts/" # location of scripts

os.chdir(scriptdir+"Step1_convert_GC")

#==== read GC lon and lat ===
data=xr.open_dataset(glob.glob(GC_datadir+"*.nc4")[0])
GC_lon=data['lon'].values
GC_lat=data['lat'].values
data.close()

#==== read lat_ratio ===
df=pd.read_csv("./lat_ratio.csv",index_col=0)
lat_mid=df.index
lat_ratio=df.values

GC_startdate=datetime.datetime.strptime("2019-07-01 23:59:59", '%Y-%m-%d %H:%M:%S')
GC_enddate=datetime.datetime.strptime("2019-07-02 23:59:59", '%Y-%m-%d %H:%M:%S')
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

for index in list(range(({run_num}-1)*1000,{run_num}*1000)):
    print('========================')
    filename=Sat_files[index]    
    temp=re.split('\/', filename)[-1]
    print(temp)
    date=re.split('\.',temp)[0]
    if os.path.isfile(outputdir+date+'_GCtoTROPOMI.pkl'):
        continue
    result=use_AK_to_GC(filename,GC_startdate,GC_enddate,use_Sensi=use_Sensi,xlim=xlim,ylim=ylim)
    save_obj(result,outputdir+date+'_GCtoTROPOMI.pkl')
