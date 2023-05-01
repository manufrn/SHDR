from fitting import fit_time_series
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
from utils import fit_function

data_path = 'data/station7.nc'

with netCDF4.Dataset(data_path, 'r') as f:
    temp = f['temp'][:]
    depth = f['pres'][:]
    time = f['date'][:]
    lat = f['lat'][:]
    lon = f['lon'][:]

a = fit_time_series(time, temp, depth, max_depth=450)
# print(a.loc['D1', 'em'])

# print(fit_function(a.loc[1543396041:1547720560], 10))
print(fit_function(a.iloc[1], np.array([10, 20])))
