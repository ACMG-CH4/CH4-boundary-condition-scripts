import numpy as np
import xarray as xr
#######################################################
# This script takes the preprocessed daily data 
# from tropomi/GC simulation and calculates the bias 
# between the two using the latitudinal averages and 
# spatial/temporal smoothing. 
# This script is based off of Lu Shen's 
# methods/ R script.
#######################################################

# settings to adjust default smoothing windows
# we use a +/- 15 day time window and 3x3
# smoothing for lat/lon
smoothing_lat_window = 3
smoothing_lon_window = 3
smoothing_time_window = 30

# access the preprocessed CH4 data
filepath = "Daily_CH4.nc"
daily_CH4 = xr.open_dataset(filepath)
TROPOMI_lon = daily_CH4["lon"]
TROPOMI_lat = daily_CH4["lat"]
GC_CH4 = daily_CH4["GC"]
OMI_CH4 = daily_CH4["CH4"]
date = daily_CH4["date"]
daily_CH4.close()


# smooth out 3d dataarray over time, lat, lon using desired windows
def smooth_3D_da(
    da,
    t_window=smoothing_time_window,
    lat_window=smoothing_lat_window,
    lon_window=smoothing_lon_window,
):
    return da.rolling(
        time=t_window,
        lat=lat_window,
        lon=lon_window,
        min_periods=1,
        center=True,
    ).mean(skipna=True)


# dmooth the background GC and TROPOMI data
GC_bkgd = GC_CH4.rolling(
    time=smoothing_time_window,
    lat=smoothing_lat_window,
    lon=smoothing_lon_window,
    min_periods=1,
    center=True,
).mean(skipna=True)

TROPOMI_bkgd = OMI_CH4.rolling(
    time=smoothing_time_window,
    lat=smoothing_lat_window,
    lon=smoothing_lon_window,
    min_periods=1,
    center=True,
).mean(skipna=True)

# calculate bias between GC background CH4 and
# TROPOMI observational background CH4
bias_4x5 = GC_bkgd - TROPOMI_bkgd

# build a smoothed dataset to fill in nan values of raw bias
# start by smoothing the bias
bias_avg_base = smooth_3D_da(bias_4x5)

# create a dataarray with latitudinal average for each time step
# we use nearest neighbor interpolation to fill in data gaps
# for the edges we use backfill and forwardfill which takes 
# the nearest lat and infills all the way to the edge
lat_base = (
    bias_avg_base.mean(dim="lon", skipna=True)
    .interpolate_na(dim="lat", method="nearest")
    .bfill(dim="lat")
    .ffill(dim="lat")
    .rolling(
        time=smoothing_time_window,
        lat=smoothing_lat_window,
        center=True,
        min_periods=1,
    )
    .mean(skipna=True)
)

# expand the latitudinal averages into a 3D dataarray for easier infilling
lat_base_full = bias_4x5.copy()
lat_base_full = xr.where(lat_base_full != np.nan, np.nan, np.nan)
for i in range(len(TROPOMI_lon)):
    lat_base_full[:, :, i] = lat_base

# infill nan values in smoothed bias with latitudinal averages
# for each corresponding day and smooth the result
bias_avg_base = smooth_3D_da(bias_avg_base.fillna(lat_base_full))

# infill nan values of raw bias data with the smoothed
# average background data and smooth the final product
bias_4x5_new = smooth_3D_da(bias_4x5.fillna(bias_avg_base))

print(bias_4x5_new.mean(skipna=True))

# create dataset and export to netcdf file
ds = bias_4x5_new.to_dataset(name="Bias")
ds = ds.assign_coords({"time": date.values})
ds.to_netcdf("Bias_4x5_dk_2_updated.nc")
