# SHDR_test
Python implementation of the upper ocean fitting algorithm described in
[González Pola et al. (2007)](https://www.sciencedirect.com/science/article/abs/pii/S0967063707002026).
The SHDR algorithm (*Sharp Homogenization/Diffusive Retreat*) performs a
differential evolution search to fit an upper ocean profile (temperature,
salinity or density) to an idealized analytical form

$$
f(z) = 
\begin{cases}
    a_1 & \text{si } z < D_1, \\
    a_3 + b_3(z-D_1) + a_2 e^{\left(-b_2(z-D1)-c_2(z-D)^2\right)} & \text{si } z > D_1.
\end{cases}
$$

The function to, a sum of an exponential and gaussian decay, which define the
pycnocline and a 

A sample call would be

```python
from SHDR import fit_time_series

result = fit_time_series(time, density, depth, max_depth=400)
```




