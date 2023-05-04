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

## Installation
SHDR is a small package. To keep things simple, we decided to source it 
as a standalone module in a single file. To use it, download the file [SHDR.py](SHDR.py) 
and place it in you working directory. This file contains all the routines needed to run 
SHDR, which should be imported in your preferred programming environment. 

A sample call would be

```python
from SHDR import fit_time_series # SHDR.py file in working directoy

result = fit_time_series(time, density, depth, max_depth=400)
```

The user manual contains extensive information on how to use the module. 
For a real use case, please see the [example jupyter notebook](examples.ipynb).

## Requisites
* ``python``
* ``numpy``, ``pandas``, ``tqdm``, ``matplotlib``


## Citation
If you use SHDR in your project, please cite the
original work [González Pola et al. (2007)](https://www.sciencedirect.com/science/article/abs/pii/S0967063707002026). The code
in this repository can be cited through (TODO: add DOI).




