# `vdx`
`vdx_py` is the python implementation of the matlab `vdx` package.
It is built on top of the modeling and automatic differentiation tool CasADi.
It is used to keep track of indexed variables e.g. `x_i∈ Rⁿ for i in 1,...N`, as subsets of bigger vectors.
This is particularly useful for writing direct transcription optimal control problems such as those found in `nosnoc_py`.

## Installing
As this software is still in development to install the package, clone this repo and use
```bash
pip install <path to vdx>
```

## Use
The basic interface for `vdx_py` is the `NLP` object which contains 1 scalar objective `f` and 3 vectors `w` (the primal decision variables, `g` the constraints, and `p` problem parameters.

To start do:
```python
import casadi as ca
from vdx_py import NLP, Primal, Parameter, Constraint
nlp = NLP(symbolic_type=ca.SX)

nlp.w.x[range(1,11), range(1,3), range(1,3)] = Primal("x", 4, lb=0.0) # Creates x_i_j_k with i=1,...10, j=1,2, k=1,2
```