# SHDR_test
Python implementation of the upper ocean fitting algorithm described in
[González Pola et al (2007)](https://www.sciencedirect.com/science/article/abs/pii/S0967063707002026).

A sample call would be

```python
from SHDR import fit_time_series

result = fit_time_series(time, density, depth, max_depth=400)
```


