import casadi as ca
from .vector import *

class NLP:
    def __init__(self,symbolic_type=ca.SX):
        self.obj = symbolic_type(0)
        self.w = PrimalVector(symbolic_type=symbolic_type)
        self.g = ConstraintVector(symbolic_type=symbolic_type)
        self.p = ParameterVector(symbolic_type=symbolic_type)
