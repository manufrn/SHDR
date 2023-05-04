import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def fit_function(params, z):
    '''
    Returns the fit function at a given height for.

    '''
    columns = ['D1', 'b2', 'c2', 'b3', 'a2', 'a1']
    D1, b2, c2, b3, a2, a1 = params[columns]
    pos = np.where(z >= D1, 1.0, 0.0)
    exponent = - (z - D1) * (b2 + (z - D1) * c2)
    print(params)
    return a1 + pos * (b3 * (z - D1) + a2 * (np.exp(exponent) - 1.0))


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

