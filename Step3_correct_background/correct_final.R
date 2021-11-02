rm(list=ls())
library(fields); library(maps); library(ncdf4);library(abind)

smooth_2D=function(bias_avg_base){
	#Smooth over a 3x3 window over all gridboxes to get rid of extreme values
	bias_avg_base_smoothed = bias_avg_base
	for(i in 1:N1){
		for(j in 1:N2){
			ind1=max(1,i-1):min(N1,i+1)
			ind2=max(1,j-1):min(N2,j+1)
			bias_avg_base_smoothed[i,j]=mean(bias_avg_base[ind1,ind2])
		}
	}
	return(bias_avg_base_smoothed)
}

smooth_1D=function(y){
	y2=y
	N1=length(y)
	for(i in 1:length(y)){
		ind1=max(1,i-1):min(N1,i+1)
		y2[i]=mean(y[ind1])
	}
	return(y2)
}

#======================================================
setwd("/home/ubuntu/CH4-boundary-condition-scripts/Step3_correct_background")

datafile=nc_open("Daily_CH4.nc")
TROPOMI.lon=ncvar_get(datafile,varid="lon")
TROPOMI.lat=ncvar_get(datafile,varid="lat")
GC_CH4=ncvar_get(datafile,varid="GC")
OMI_CH4=ncvar_get(datafile,varid="CH4")
date=ncvar_get(datafile,varid="date")
nc_close(datafile)
YYYY=as.numeric(substr(date,1,4))
MM=as.numeric(substr(date,5,6))
DD=as.numeric(substr(date,7,8))

#========= correct by latitudes/month ========
N1=length(TROPOMI.lon);N2=length(TROPOMI.lat);N3=length(YYYY)
dk=1;dt=15

mask=apply(GC_CH4,c(1,2),mean,na.rm=T) #gridcells where there are observations

GC_bkgd=array(NA,dim(GC_CH4))
OMI_bkgd=array(NA,dim(GC_CH4))
for(i in 1:N1){
	print(i)
	ind1=max(1,i-dk):min(N1,i+dk)
	for(j in 1:N2){
		if(is.na(mask[i,j]))next
		ind2=max(1,j-dk):min(N2,j+dk)
		for(t in 1:N3){
			ind3=max(1,t-dt):min(N3,t+dt)
			num1=sum(!is.na(OMI_CH4[ind1,ind2,ind3]))
			if(num1>=20){
     		    GC_bkgd[i,j,t]=mean(GC_CH4[ind1,ind2,ind3],na.rm=T)
	    		OMI_bkgd[i,j,t]=mean(OMI_CH4[ind1,ind2,ind3],na.rm=T)
	    	}			
}
}
}
bias_4x5 =(GC_bkgd - OMI_bkgd)
lon.out=TROPOMI.lon;lat.out=TROPOMI.lat


#=================================================
bias_avg_base = apply(bias_4x5, c(1,2), mean, na.rm=TRUE)

lat_bias_base=apply(bias_4x5,c(2),mean,na.rm=TRUE)
lat_bias_base[45:46]= lat_bias_base[44]
lat_bias_base =smooth_1D(lat_bias_base)

for(k in 1:46){#fill ocean gridcells with lat bias
	y= bias_avg_base[,k]
	y[is.na(y)]= lat_bias_base[k]
	bias_avg_base[,k]=y	
}

#=== now make it smooth =====
bias_avg_base_smoothed = smooth_2D(bias_avg_base)#smoothed 2D average bias

bias_4x5_new=array(NA,dim(bias_4x5))
for(index in 1:dim(bias_4x5)[3]){
  print(index)
  temp= bias_4x5[,,index]
  ind1=(is.na(temp) & !is.na(bias_avg_base)) # if it is over the continent, fill with mean bias
  temp[ind1]= bias_avg_base[ind1]
  temp2=smooth_2D(temp)
  bias_4x5_new[,,index]=temp2
}
bias_4x5_new[bias_4x5_new>=30]=30
bias_4x5_new[bias_4x5_new<=-30]=-30

dev.new(width=5,height=2.8)
avg1=apply(bias_4x5,c(1,2),mean,na.rm=T)
avg2=apply(bias_4x5_new,c(1,2),mean,na.rm=T)
# plot.field(avg1-avg2, lon.out, lat.out,type="sign",zlim=c(-10,10))

print(mean(bias_4x5_new))

#============================================
#============================================
#============================================
YYYYMMDD= date
ncfname="Bias_4x5_dk_2_updated.nc"
mv2 = -1e30
londim=ncdim_def(name="lon",units="degrees_east",longname="longitude",vals= lon.out)
latdim=ncdim_def(name="lat",units="degrees_north",longname="latitude", vals= lat.out)
datedim=ncdim_def(name="time",units="",vals= YYYYMMDD,longname="Time")

Datadim=ncvar_def(name="Bias",units="ppb",longname="GOSAT methane",dim=list(londim, latdim, datedim),missval=mv2,prec="float")
ncnew<-nc_create(ncfname,list(Datadim))
ncvar_put(ncnew, Datadim, bias_4x5_new)
nc_close(ncnew)
