# SHDR_test

Python implementation of the upper ocean structure fitting algorithm described in
[González Pola et al. (2007)](https://www.sciencedirect.com/science/article/abs/pii/S0967063707002026).
The SHDR algorithm (*Sharp Homogenization/Diffusive Retreat*) performs a
differential evolution search to fit an upper ocean profile (temperature,
salinity or density) to an idealized analytical form. It can be used to estimate
the mixed layer depth of a given profile without any further information. It
also allows for a physical characterisation of the seasonal and permanent
pycnoclines. 

$$
f(z) = 
\begin{cases}
    a_1 & \text{if} z < D_1, \\
    a_3 + b_3(z-D_1) + a_2 e^{\left(-b_2(z-D_1)-c_2(z-D_1)^2\right)} & \text{if} z > D_1.
\end{cases}
$$

with the parametres:
* D_1 - Mixed Layer Depth
* a_1 - Mixed Layer Temperature

## Instalation and basic usage
SHDR is a small package. To allow for simple usage, we decided to keep it small.
To use it, download the file [SHDR.py](SHDR.py) and place it in you working
directory. This files contains all the routines needed to run SHDR, which should
to be imported in another .py file, jupyter notebook or straight from the python console. 
A sample call would be


```python
from SHDR import fit_time_series # SHDR.py file in working directoy

result = fit_time_series(time, density, depth, max_depth=400)
```

Please refer to the user manual and example jupyter notebook to understand




