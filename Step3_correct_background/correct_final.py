import numpy as np
import xarray as xr

def smooth_2D(bias_avg_base):
	#Smooth over a 3x3 window over all gridboxes to get rid of extreme values
	bias_avg_base_smoothed = bias_avg_base.copy()
	for i in range(N1):
		for j in range(N2):
			ind1=slice(max(0,i-1), min(N1,i+1))
			ind2=slice(max(0,j-1), min(N2,j+1))
			bias_avg_base_smoothed[i,j]=bias_avg_base[ind1,ind2].mean(skipna=True)
	return(bias_avg_base_smoothed)

def smooth_1D(y):
	y2=y.copy()
	N1=len(y)
	for i in range(N1):
		ind1=slice(max(0,i-1), min(N1,i+1))
		y2[i]=y[ind1].mean(skipna=True)
	return(y2)

def closest_valid_lat(lat, valid_lats):
	closest_lat = min(abs(valid_lats-lat))
	return(closest_lat)



datafile=xr.open_dataset("Daily_CH4.nc")
TROPOMI_lon=datafile["lon"]
TROPOMI_lat=datafile["lat"]
GC_CH4=datafile["GC"]
OMI_CH4=datafile["CH4"]
date=datafile["date"]
datafile.close()
# YYYY=as.numeric(substr(date,1,4))
# MM=as.numeric(substr(date,5,6))
# DD=as.numeric(substr(date,7,8))

#========= correct by latitudes/month ========
N1=len(TROPOMI_lon);N2=len(TROPOMI_lat);N3=len(date)
dk=1
dt=15

# mask=apply(GC_CH4,c(1,2),mean,na.rm=T) #gridcells where there are observations
mask = GC_CH4.mean(dim=["time"], skipna=True)
# GC_bkgd=np.full(GC_CH4.shape, np.nan)
gc_copy = GC_CH4.copy()
GC_bkgd=xr.where(gc_copy != np.nan, np.nan, np.nan)
OMI_bkgd=GC_bkgd.copy()
bias_4x5_new = GC_bkgd.copy()
for i in range(N1):
	print(i)
	ind1=slice(max(0,i-dk), min(N1,i+dk))
	for j in range(N2):
		if np.isnan(mask[i,j]):
			# skip to the next one is nan
			continue
		ind2=slice(max(1,j-dk),min(N2,j+dk))
		for t in range(N3):
			ind3=slice(max(1,t-dt),min(N3,t+dt))
			num1=sum(not np.isnan(OMI_CH4[ind1,ind2,ind3]))
			if(num1>=20):
				GC_bkgd[i,j,t]=np.nanmean(GC_CH4[ind1,ind2,ind3])
				OMI_bkgd[i,j,t]=np.nanmean(OMI_CH4[ind1,ind2,ind3])


bias_4x5 =(GC_bkgd - OMI_bkgd)
# lon.out=TROPOMI.lon;lat.out=TROPOMI.lat

#=================================================
bias_avg_base = bias_4x5.mean(dim=["time"], skipna=True)

# lat_bias_base=apply(bias_4x5,c(2),mean,na.rm=TRUE)
lat_bias_base = bias_4x5.mean(dim=["lon", "time"], skipna=True)
# lat_bias_base[45:46]= lat_bias_base[44]
valid_lats = np.argwhere(np.isnan(lat_bias_base) == False)
invalid_lats = np.argwhere(np.isnan(lat_bias_base))
# replace any NA lat values with closest lat bias
for k in invalid_lats: 
  	lat_bias_base[k] = lat_bias_base[closest_valid_lat(k, valid_lats)]
lat_bias_base = smooth_1D(lat_bias_base)

for k in range(46): #fill ocean gridcells with lat bias
	y = bias_avg_base[:,k]
	y = xr.where(np.isnan(y), lat_bias_base[k], y)
	bias_avg_base[:,k]=y	

#=== now make it smooth =====
# bias_avg_base_smoothed = smooth_2D(bias_avg_base)#smoothed 2D average bias

for index in range(len(date)):
  temp= bias_4x5[:,:,index]
  ind1=((np.isnan(temp)) & (not np.isnan(bias_avg_base.values))) # if it is over the continent, fill with mean bias
  temp = xr.where(ind1, bias_avg_base, temp)
  temp2=smooth_2D(temp)
  bias_4x5_new[:,:,index]=temp2

# bias_4x5_new[bias_4x5_new>=30]=30
# bias_4x5_new[bias_4x5_new<=-30]=-30


print(xr.mean(bias_4x5_new))

#============================================
#============================================
#============================================
# YYYYMMDD= date
# ncfname="Bias_4x5_dk_2_updated.nc"
# mv2 = -1e30
# londim=ncdim_def(name="lon",units="degrees_east",longname="longitude",vals= lon.out)
# latdim=ncdim_def(name="lat",units="degrees_north",longname="latitude", vals= lat.out)
# datedim=ncdim_def(name="time",units="",vals= YYYYMMDD,longname="Time")

# Datadim=ncvar_def(name="Bias",units="ppb",longname="GOSAT methane",dim=list(londim, latdim, datedim),missval=mv2,prec="float")
# ncnew<-nc_create(ncfname,list(Datadim))
# ncvar_put(ncnew, Datadim, bias_4x5_new)
# nc_close(ncnew)
