import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def fit_function(z, params):
    '''
    Returns the value of an analytical profile with parametres params at
    height z.

    '''

    if len(list(params)) <= 2:
        raise ValueError("Params must be the complete set of parameters. "\
                         "Set only_mld to False when performing the fit "\
                         "to later use this function")

    if isinstance(params, pd.core.series.Series):

        columns = ['D1', 'b2', 'c2', 'b3', 'a2', 'a1']
        D1, b2, c2, b3, a2, a1 = params[columns]

    elif isinstance(params, np.ndarray):
        D1, b2, c2, b3, a2, a1 = params[:6]

    pos = np.where(z >= D1, 1.0, 0.0)
    exponent = - (z - D1) * (b2 + (z - D1) * c2)

    return a1 + pos * (b3 * (z - D1) + a2 * (np.exp(exponent) - 1.0))


def compute_stratification(params, alpha=0.05):
    '''Compute the stratification index G'''

    if len(list(params)) <= 2:
        raise ValueError("Params must be the complete set of parameters. "\
                         "Set only_mld to False when performing the fit "\
                         "to later use this function.")

    if isinstance(params, pd.core.series.Series):

        columns = ['D1', 'b2', 'c2']
        D1, b2, c2 = params[columns]

    elif isinstance(params, np.ndarray):
        D1, b2, c2 = params[:3]
    
    if c2 < 1e-15 and b2 < 1e-15:
        G_alpha = 0
        return G_alpha

    if c2 < 1e-15:
        delta_alpha = -np.log(alpha) / b2
    
    elif b2 == 0:
        delta_alpha = np.sqrt(- np.log(alpha) / c2)
    
    else:
        lambda_ = 2 * c2 / b2**2
        delta_alpha = - b2 / 2 / c2 * (1 - np.sqrt(1 - 2*lambda_*np.log(alpha)))
        
    z_alpha = delta_alpha + D1
    f_z_alpha = fit_function(z_alpha, params)
    G_alpha = (fit_function(D1, params) - f_z_alpha) / delta_alpha
    
    return G_alpha


def time_series_stratification(time_series_fit, alpha=0.05):
    if isinstance(time_series_fit, pd.DataFrame):
        IS_FILE = False

    if isinstance(time_series_fit, str):
        IS_FILE = True
        ts_path = Path(time_series_fit)

        if not ts_path.is_file():
            raise ValueError('Time series fit file not found.')

        elif not time_series_fit.endswith('.csv'):
            raise ValueError('Time series fit file must be a .csv.')

        else:
            time_series_fit = pd.read_csv(ts_path)
    
    stratification = time_series_fit.apply(compute_stratification, alpha=alpha, axis=1)

    if IS_FILE:
        time_series_fit = time_series_fit.assign(G=stratification)
        time_series_fit.to_csv(ts_path, index=False)

    return stratification


def plot_profile_fit(variable, depth, params, figsize=(4, 4.6875)):
    '''Plot measured vertical profile and fit for measure at loc

    '''


    if isinstance(depth, np.ma.core.MaskedArray):
        depth = np.asarray(depth[depth.mask==False])
        variable = np.asarray(variable[variable.mask==False])
    # remove nans in both arrays
    variable = variable[np.isfinite(depth)]
    depth = depth[np.isfinite(depth)]

    depth = depth[np.isfinite(variable)]
    variable = variable[np.isfinite(variable)]

    mld = params.D1
    zz = np.linspace(depth[0], depth[-1] + 5, 300)


    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(variable, depth, marker='o', s=24)
    ax.axhline(mld, c='grey', ls='--') # plot MLD
    ax.set_ylim(depth[-1] + 10, 0)

    ax.plot(fit_function(params, zz), zz, lw=1)
    ax.set_xlabel('Temperature ($^\circ$C)')
    ax.set_ylabel('Depth')
    # ax.set_title(date_i_str)
    fig.tight_layout()
    plt.show()

