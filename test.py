from SHDR_utils import plot_profile_fit, fit_function, compute_stratification, time_series_stratification
from SHDR import fit_time_series, fit_profile
import numpy as np
import pandas as pd
import netCDF4

data_path = 'data/station7.nc'

with netCDF4.Dataset(data_path, 'r') as f:
    temp = f['temp'][:]
    depth = f['pres'][:]
    time = f['date'][:]
    lat = f['lat'][:]
    lon = f['lon'][:]

a = fit_time_series(time, temp, depth, max_depth=450, delta_coding=True, save='results/results_1.csv')

print(time_series_stratification('results_1.csv'))


# p.parentrint(fit_function(10, a.loc[0]))

a = []
for i in range(len(time)):
    a.append(fit_profile(temp[i], depth[i], max_depth=450))

print(compute_stratification(a[0]))
print(fit_function(10, a[0]))




