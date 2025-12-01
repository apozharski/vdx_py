from casadi import Function
from .variable import IndexResult

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
