import numpy as np
import pandas as pd
import matplotlib as plt

def fit_function(params, z):
    '''
    Returns the fit function at a given height for.

    '''
    columns = ['D1', 'b2', 'c2', 'b3', 'a2', 'a1']
    D1, b2, c2, b3, a2, a1 = params[columns]
    pos = np.where(z >= D1, 1.0, 0.0)
    exponent = - (z -D1) * (b2 + (z - D1) * c2)

    return a1 + pos * (b3 * (z - D1) + a2 * (np.exp(exponent) - 1.0))


def plot_profile_fit(variable, depth, params, figsize=):
    '''Plot measured vertical profile and fit for measure at loc
    '''

    zz = np.linspace(1, depth[-1] + 5, 300)

    fig, ax = plt.subplots(figsize=(4, 4.6875))
    ax.scatter(temp_i, pres_i, marker='o', fc='None', ec=colors[1], s=24)
    ax.axhline(mld, c='grey', ls='--') # plot MLD
    ax.set_ylim(pres_i[-1] + 10, 0)

    if xlim is None:
        ax.set_xlim(11, 16)
    
    else:
        ax.set_xlim(*xlim)

    ax.plot(fit_function(zz, df, date_i), zz, lw=1, c=colors[0])
    ax.set_xlabel('Temperature ($^\circ$C)')
    ax.set_ylabel('Depth')
    ax.set_title(date_i_str)
    fig.tight_layout()
    if save:
        fig.savefig(str(save))
    plt.show()

