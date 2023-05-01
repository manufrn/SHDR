import numpy as np
from dataclasses import dataclass
import pandas as pd

@dataclass
class FitOptions:
    only_mld: bool = False
    CR: float = 0.7
    FF: float = 0.6
    num_generations: int = 1200
    num_individuals: int = 60
    max_b2_c2: float = 0.5
    exp_limit: float = 0.5
    min_depth: float = 100
    max_depth: float = 1000
    min_obs: int = 6
    tol: float = 0.00025
    seed: int = None


def process_input_field(arr):
    if isinstance(arr, np.ma.core.MaskedArray):
        processed_array = arr.astype(float).filled(np.nan)

    else:
        processed_array = np.asarray(arr, dtype=np.float64)

    return np.squeeze(processed_array)


def check_input(time, variable, depth, lat, lon):

    # make sure to always work with np.ndarray
    t = process_input_field(time)
    y = process_input_field(variable)
    z = process_input_field(depth)

    # check if latitude and longitude are provided and check their length
    if lat is None or lon is None:

        if lat is not lon:
            raise ValueError('Either neither or both lat and lon must be provided.')

    else:
        lat = process_input_field(lat)
        lon = process_input_field(lon)

        if time.shape != lat.shape or time.shape != lon.shape:
            raise ValueError('lat and lon arrays must have the same length as time')
         
    # length and size checks to ensure input arrays are compatible
    if time.ndim != 1:
        raise ValueError('Time must be 1-D array.')

    if variable.ndim != 2 or depth.ndim != 2:
        raise ValueError('Depth and variable must be 2-D arrays.')

    if time.shape[0] != variable.shape[0] or time.shape[0] != depth.shape[0]:
        raise ValueError('First dimension of variable and depth arrays must have the same length as time')
    
    return time, variable, depth, lat, lon

