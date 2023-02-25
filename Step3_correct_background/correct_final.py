import numpy as np
import xarray as xr


datafile=xr.open_dataset("Daily_CH4.nc")
TROPOMI_lon=datafile["lon"]
TROPOMI_lat=datafile["lat"]
GC_CH4=datafile["GC"]
OMI_CH4=datafile["CH4"]
date=datafile["date"]
datafile.close()

#========= correct by latitudes/month ========
N1=len(TROPOMI_lon);N2=len(TROPOMI_lat);N3=len(date)
smoothing_lat_window=3
smoothing_lon_window=3
smoothing_time_window=30

# smooth out 3d dataarray over time, lat, lon using desired windows
def smooth_3D_da(da, t_window=smoothing_time_window, lat_window=smoothing_lat_window, lon_window=smoothing_lon_window):
	return da.rolling(time=t_window, lat=lat_window, lon=lon_window, min_periods=1, center=True).mean(skipna=True)

# smooth out background data by 3x3 lat/lon windows and +/-15 day window
GC_bkgd = GC_CH4.rolling(time=smoothing_time_window, lat=smoothing_lat_window, lon=smoothing_lon_window, min_periods=1, center=True).mean(skipna=True)
TROPOMI_bkgd = OMI_CH4.rolling(time=smoothing_time_window, lat=smoothing_lat_window, lon=smoothing_lon_window, min_periods=1, center=True).mean(skipna=True)

# calculate bias between GC background CH4 and and TROPOMI observational background CH4
bias_4x5 =(GC_bkgd - TROPOMI_bkgd)

# smooth out 
bias_avg_base = smooth_3D_da(bias_4x5)
lat_base = bias_avg_base.mean(dim="lon", skipna=True).interpolate_na(dim="lat", method="nearest").bfill(dim="lat").ffill(dim="lat").rolling(time=smoothing_time_window, lat=smoothing_lat_window, center=True, min_periods=1).mean(skipna=True)

# create a dataarray with latitudinal average for each time step 
# expand it into a 3D dataarray
lat_base_full = bias_4x5.copy()
lat_base_full = xr.where(lat_base_full != np.nan, np.nan, np.nan)
for i in range(len(TROPOMI_lon)):
	lat_base_full[:,:,i] = lat_base

# infill nan values in average base with latitudinal averages
# smooth the result
bias_avg_base = smooth_3D_da(bias_avg_base.fillna(lat_base_full))

# infill nan values of raw bias data with the smoothed 
# average background data and smooth the final product
bias_4x5_new = smooth_3D_da(bias_4x5.fillna(bias_avg_base))

print(bias_4x5_new.mean(skipna=True))

# create dataset and export to netcdf file
ds = bias_4x5_new.to_dataset(name="Bias")
ds.to_netcdf("Bias_4x5_dk_2_updated.nc")
