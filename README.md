# SHDR_test

Python implementation of the upper ocean structure fitting algorithm described in
[González Pola et al. (2007)](https://www.sciencedirect.com/science/article/abs/pii/S0967063707002026).
The SHDR algorithm (*Sharp Homogenization/Diffusive Retreat*) performs a
differential evolution search to fit an upper ocean profile (temperature,
salinity or density) to an idealized analytical form. This analytical form
defines a region with constant tracer (MLD), followed by a combination of
exponential and gaussian decays (seasonal pycnocline) and a linear decay
(permanent pycnocline). The algorithm can be used as a MLD identifying method,
but it also allows for a physical characterisation of the pycnocline. 

## Installation and basic usage
SHDR is a small package. To allow for simple usage, we decided to keep it small.
To use it, download the file [SHDR.py](SHDR.py) and place it in you working
directory. This files contains all the routines needed to run SHDR, which should
to be imported in another .py file, jupyter notebook or straight from the python console. 
A sample call would be


```python
from SHDR import fit_time_series # SHDR.py file in working directoy

result = fit_time_series(time, density, depth, max_depth=400)
```

Please refer to the user manual and example jupyter notebook.

## Requisites
* ``python``
* ``numpy``, ``pandas``, ``tqdm``, ``matplotlib``


## Citation
If you use SHDR in your project, please cite the
original work [González Pola et al. (2007)](https://www.sciencedirect.com/science/article/abs/pii/S0967063707002026). The code
in this repository can be cited through (TODO: add DOI).




