'''
This file shouldn't be modified or run for any purpose. Place it in your 
working directory, and import it's core functions to use it. For more
information about using SHDR please refer to the user manual.
'''

import sys
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class _FitOptions:
    '''Class defining options for the fitting algorithm.

    '''

    only_mld: bool = False
    delta_coding: bool = False

    # genetic evolution parameters
    CR: float = 0.7
    FF: float = 0.7
    num_generations: int = 1200
    num_individuals: int = 60
    tol: float = 0.00025

    # fit parameters and safe limits
    max_b2_c2: float = 0.5
    exp_limit: float = 100
    min_depth: float = 50
    max_depth: float = 1000
    min_obs: int = 10
    
    # misc
    seed: int = None
    save: str = None    # only used for time series fit


def _process_input_field(arr):
    '''Basic preprocesing of input arrays.

    '''

    if isinstance(arr, np.ma.core.MaskedArray):
        processed_array = arr.astype(float).filled(np.nan)

    else:
        processed_array = np.asarray(arr, dtype=np.float64)

    return np.squeeze(processed_array)


def _check_time_series_input(time, variable, depth, lat, lon):
    ''' Check dimensional consistency of the input fields 
    for a time series fit and return a processed version.

    '''

    time = _process_input_field(time)
    variable = _process_input_field(variable)
    depth = _process_input_field(depth)

    # check if latitude and longitude are provided and check their length
    if lat is None or lon is None:

        if lat is not lon:
            raise ValueError('Either neither or both lat and lon must be provided.')

    else:
        lat = _process_input_field(lat)
        lon = _process_input_field(lon)

        if time.shape != lat.shape or time.shape != lon.shape:
            raise ValueError('lat and lon arrays must have the same length as time')
         
    # length and size checks to ensure input arrays are compatible
    if time.ndim != 1:
        raise ValueError('Time must be 1-D array.')
    
    if depth.ndim > 2:
        raise ValueError('Depth must be 1-D or 2-D array.')

    # if depth is 1-D, broadcast it to 2-D
    if depth.ndim == 1:
        depth = np.broadcast_to(depth, variable.shape)

    if variable.ndim != 2:
        raise ValueError('Variable must be 2-D array.')

    if time.shape[0] != variable.shape[0] or time.shape[0] != depth.shape[0]:
        raise ValueError('First dimension of variable and depth arrays must have the same length as time')
    
    return time, variable, depth, lat, lon


def _check_save_path(path):
    '''
    Chech if save file for a time series fit already exists, and
    if it does, ask if it should be overwritten. 

    '''
    if not path.endswith('.csv'):
        raise ValueError("Output file must be '.csv'")
    
    if Path(path).exists():
        promt = input('Output file already exists, do you want to overwrite it? [Y/n]: ')
        if promt.lower() == 'n':
            sys.exit() 

    # create parent directory for path if it doesn't exist
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _fit_function(individuals, z, opts: _FitOptions):
    '''Estimate the function a group of individuals at a height z.

    '''
	
    limit = opts.exp_limit
    D1, b2, c2, b3, a2, a1 = np.split(individuals, 6, axis=1)

    pos = np.where(z >= D1, 1.0, 0.0)
    exponent = - (z -D1) * (b2 + (z - D1) * c2)
    
    # chech if exponent is inside limits
    exponent = np.where(exponent > limit, limit, exponent)
    exponent = np.where(exponent < - limit, - limit, exponent)

    return a1 + pos * (b3 * (z - D1) + a2 * (np.exp(exponent) - 1.0))


def _get_fit_limits(y, z, opts: _FitOptions):
    '''Returns the limits for the parametres of the fit function given a certain
       profile with meassures y at depths z.

    '''
       
    z = np.abs(z) # in case depths are defined negative
    
    decay = True if y[0] > y[-1] else False
    min_z, max_z = z.min(), z.max()
    min_y, max_y = y.min(), y.max()
    
    if decay:
        lims = np.array([[1.0, max_z],    # D1
                [0.0, opts.max_b2_c2],    # b2
                [0.0, opts.max_b2_c2],    # c2
                [0.0 if max_z < opts.min_depth else - abs((max_y - min_y) / (max_z - min_z)), 0.0], # b3
                [0.0, max_y - min_y],     # a2
                [min_y, max_y]])          # a1          
    
    else:
        lims = np.array([[1.0, max_z],    # D1
                [0.0, opts.max_b2_c2],    # b2
                [0.0, opts.max_b2_c2],    # c2
                [0.0, 0.0 if max_z < opts.min_depth else abs((max_y - min_y) / (max_z - min_z))], # b3
                [- max_y + min_y, 0.0],     # a2
                [min_y, max_y]])          # a1          

    lims_min = lims[:, 0]
    lims_max = lims[:, 1]
    
    return (lims_min, lims_max)


def _random_init_population(y, z, lims, opts: _FitOptions):
    ''' Returns a random population of solutions with randomly 
    initialized values for the parameters inside the limits for
    a profile with meassures y at depths z.
    
    '''
    
    n = opts.num_individuals 
    lims_min, lims_max = lims
    n_var = np.size(lims_max)
    
    norm = lims_max - lims_min
    individuals = lims_min + norm * np.random.random((n, n_var))

    return individuals


def _population_fitness(individuals, y, z, opts):
    '''Estimate the fitness for a group of individuals for a profile
    with meassures y at depths z via mean squared error.

    '''
    
    fitness = np.sqrt(np.sum((y - _fit_function(individuals, z, opts))**2, axis=1) / len(y))
    return fitness


def _diferential_evolution(individuals, y, z, lims, opts):
    ''' Perform a diferential evolution on a group of individuals 
    for a given profile with meassures y at depths z.

    '''

    n = opts.num_individuals
    lims_min, lims_max = lims
    n_var = np.size(lims_max)
     
    present_fitns = _population_fitness(individuals, y, z, opts)

    best_fit_loc = present_fitns.argmin()
    best_fit = individuals[best_fit_loc]

    for generation in range(opts.num_generations):

        # weight of best indivual is most important in later generations
        best_weight = 0.2 + 0.8 * (generation / opts.num_generations)**2
        
        # generate random permutations 
        perm_1 = np.random.permutation(n)
        perm_2 = np.random.permutation(n)
        new_gen = (1 - best_weight) * individuals + best_weight * best_fit + (opts.FF
                  * (individuals[perm_1] - individuals[perm_2]))
        
        new_gen = np.where(np.random.rand(n, n_var) < opts.CR,
                  new_gen, individuals)
                             

        # seting limits
        new_gen = np.where(new_gen < lims_min.reshape((1,6)), lims_min.reshape((1,6)), new_gen)
        new_gen = np.where(new_gen > lims_max.reshape((1,6)), lims_max.reshape((1,6)), new_gen)

        new_fitns = _population_fitness(new_gen, y, z, opts)

        
        # update individuals to new generation
        individuals = np.where(present_fitns[:, None] < new_fitns[:, None], individuals, new_gen)
        present_fitns = np.where(present_fitns < new_fitns, present_fitns, new_fitns)

        best_fit_loc = present_fitns.argmin()
        best_fit = individuals[best_fit_loc]
        
        if present_fitns.mean() * opts.tol / present_fitns.std() > 1:
            break
     
    return best_fit, present_fitns[best_fit_loc]


def _fit_single_profile(y, z, opts): 
    '''Parse and fit data from a single profile

    '''

    y = _process_input_field(y)
    z = _process_input_field(z)

    if y.ndim > 1 or z.ndim > 1:
        raise ValueError('y and z must be 1-D arrays.')
    
    if y.size != z.size:
        return ValueError('y and z must have the same size')

    
    # remove nans in both arrays
    y = y[np.isfinite(z)]
    z = z[np.isfinite(z)]

    z = z[np.isfinite(y)]
    y = y[np.isfinite(y)]
    
    # only use depths until max_depth
    if (z > opts.max_depth).any():
        max_z_idx = np.argmax(z > opts.max_depth)
        z = z[:max_z_idx]
        y = y[:max_z_idx]
    
    if len(z) < opts.min_obs:
        return np.repeat(np.nan, 8)
    
    lims = _get_fit_limits(y, z, opts)
    
    lims_min, lims_max = lims

    first_gen = _random_init_population(y, z, lims, opts)
    result_1, fitness_1 = _diferential_evolution(first_gen, y, z, lims, opts)  
    
    #### DELTA CODING ####
    # set new limits for fit depending of previous fit result
    # and have them meet the physical limits
    if opts.delta_coding:
        lims_min_d, lims_max_d = 0.85 * result_1, 1.15 * result_1
        for i in np.where(np.sign(result_1) < 0)[0]:
            lims_min_d[i], lims_max_d[i] = lims_max_d[i], lims_min_d[i]

        lims_min_delta = np.where(lims_min_d >= lims_min, lims_min_d,  lims_min)
        lims_max_delta = np.where(lims_max_d <= lims_max, lims_max_d, lims_max)


        lims_delta = (lims_min_delta, lims_max_delta)

        first_gen = _random_init_population(y, z, lims_delta, opts)   # new first generation

        result_delta, fitness_delta = _diferential_evolution(first_gen, y, z, lims_delta, opts)


        if fitness_1 < fitness_delta:
            result = result_1
            fitness = fitness_1 
        else:
            result = result_delta
            fitness = fitness_delta 

    else:
        result = result_1
        fitness = fitness_1

    D1, b2, c2, b3, a2, a1 = result
    em = fitness
    a3 = a1 - a2 

    return np.array([D1, b2, c2, b3, a2, a1, a3, em])


def _format_time_series_result(result, time, lat, lon, opts):
    ''''''
      
    if opts.only_mld == True:
        columns = ['D1']
        result_df = pd.DataFrame([i[0] for i in result], columns=columns)
    
    else:
        columns = ['D1', 'b2', 'c2', 'b3', 'a2', 'a1', 'a3', 'em']
        result_df = pd.DataFrame(result, columns=columns)
    
    result_df.insert(0, 'time', time)

    if lat is not None:
        result_df.insert(1, 'lat', lat)
        result_df.insert(2, 'lon', lon)

    return result_df


def _run_multiprocessing_fit_pool(variable, depth, opts):
    n = variable.shape[0]
    pool_arguments = [[variable[i, :], depth[i, :], opts] for i in range(n)]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results_fit = pool.starmap(_fit_profile, tqdm(pool_arguments,
                                                     total=len(pool_arguments)), chunksize=1)
    return results_fit


def fit_time_series(time, variable, depth, lat=None, lon=None, **opts):
    '''
    Fit a time series record using the SHDR algorithm. 

    Parametres
    ----------
    time : array_like
        Time coordinate. At the moment any format is accepted. Beware of this 
        when using non python time formats (eg. matlab datenum).
    variable : array_like
        2D array containing variable to be fitted (temperature, density or salinity). 
        First dimension is temporal.
    depth : array_like
        2D array defining the vertical coordinate. First dimension is temporal.
    lat : array_like, optional
        Latitude.
    lon : array_like, optional
        Longitude.

    Other parametres
    ----------------
    only_mld : bool default=False
        If True, only the parameter D1 is returned.
    max_depth : float default=1000
        Maximun depth of the profile to consider for fitting.
    min_depth : float default=100
        Minium maximal depth of the profile to perform fitting.
    min_obs : int default=6
        Minimum number of observations in the profile to perform fitting.
    CR : float default=0.7
        Cross probability (diferential evolution algorithm).
    FF : float default=0.6
        Mutation factor (diferential evolution algorithm).
    num_generations : int default=1200
        Number of generations (diferential evolution algorithm).
    num_individuals : int default=60
        Number of individuals (diferential evolution algorithm).
    max_b2_c2 : float default=0.5
        Maximum value for b2 and c2 coefficients.
    exp_limit : float default=0.5
        Maximum decay.
    tol : float default=0.00025
        Tolerance (diferential evolution algorithm).
    seed : int default=None
        Random seed (diferential evolution algorithm).

    Returns
    -------
    pd.DataFrame

    '''
     
    time, variable, depth, lat, lon = _check_time_series_input(time, variable, depth, lat, lon) 
    opts = _FitOptions(**opts) 

    np.random.seed(opts.seed)
    
    results_fit = _run_multiprocessing_fit_pool(variable, depth, opts)
    
    result_df = _format_time_series_result(results_fit, time, lat, lon, opts)
    
    if opts.save is not None:
        _check_save_path(opts.save)
        result_df.to_csv(opts.save, index=False)

    return result_df


def fit_profile(y, z, **opts):
    '''
    Fit a single vertical profile using the SHDR algorithm.

    Parametres
    ---------
    y : array_like
        Variable to be fitted (temperature, density or salinity).
    z : array_like
        Vertical coordinate.

    Other parametres
    ----------------
    only_mld : bool default=False
        If True, only the parameter D1 is returned.
    max_depth : float default=1000
        Maximun depth of the profile to consider for fitting.
    min_depth : float default=100
        Minium maximal depth of the profile to perform fitting.
    min_obs : int default=6
        Minimum number of observations in the profile to perform fitting.
    CR : float default=0.7
        Cross probability (diferential evolution algorithm).
    FF: float default=0.7
        Mutation factor (diferential evolution algorithm).
    num_generations : int default=1200
        Number of generations (diferential evolution algorithm).
    num_individuals : int default=60
        Number of individuals (diferential evolution algorithm).
    max_b2_c2 : float default=0.5
        Maximum value for b2 and c2 coefficients.
    exp_limit : float default=0.5
        Maximum decay.
    tol : float default=0.00025
        Tolerance (diferential evolution algorithm).
    seed : int default=None
        Random seed (diferential evolution algorithm).

    Returns
    -------
    np.ndarray containing fit parametres in the order
    [D1, b2, c2, b3, a2, a1, a3, em]. If only_mld opt is
    True, returns [D1].

    '''

    opts = _FitOptions(**opts)
    np.random.seed(opts.seed)

    result_fit = _fit_single_profile(y, z, opts)

    if opts.only_mld:
        result = np.asarray([result_fit[0]])

    else:
        result  = result_fit

    if opts.save is not None:
        _check_save_path(opts.save)
        result_df.to_csv(opts.save, index=False)

    return result

