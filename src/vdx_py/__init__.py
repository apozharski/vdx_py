from casadi import Function
import casadi
from .variable import IndexResult
from .nlp import NLP
from .vartypes import Primal, Parameter, Constraint, Relaxation, HardConstraint, Ell1Relaxation, Ell2Relaxation, EllInfRelaxation

# CasADi Patches
_original_Function_call = Function.call
def _patched_Function_call(self, args):
    # somewhat slow maybe?
    # TODO(@anton) figure out named arguments
    if isinstance(args, dict):
        new_args = {k: v.getsym() if isinstance(v, IndexResult) else v for (k,v) in args.items()}
    else:
        new_args = tuple([arg.getsym() if isinstance(arg, IndexResult) else arg for arg in args])
    return _original_Function_call(self, new_args)

Function.call = _patched_Function_call

_original_vertcat = casadi.vertcat
def _patched_vertcat(*args):
    new_args = tuple([arg.getsym() if isinstance(arg, IndexResult) else arg for arg in args])
    return _original_vertcat(*new_args)

casadi.vertcat = _patched_vertcat


_original_sum2 = casadi.sum2
def _patched_sum2(*args):
    new_args = tuple([arg.getsym() if isinstance(arg, IndexResult) else arg for arg in args])
    return _original_sum2(*new_args)

casadi.sum2 = _patched_sum2
