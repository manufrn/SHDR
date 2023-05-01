import numpy as np
import pandas as pd
import matplotlib as plt

def fit_function(params, z):
    '''
    Returns the fit function at a given height for.

    '''
    if not isinstance()
    
    columns = ['D1', 'b2', 'c2', 'b3', 'a2', 'a1']

    # if isinstance(params, pd.core.series.Series):
    D1, b2, c2, b3, a2, a1 = params[columns]
    print(D1)

    
    # D1, b2, c2, b3, a2, a1 = np.split(params, 6, axis=1)

    pos = np.where(z >= D1, 1.0, 0.0)
    exponent = - (z -D1) * (b2 + (z - D1) * c2)

    return a1 + pos * (b3 * (z - D1) + a2 * (np.exp(exponent) - 1.0))
