import casadi as ca
from .vector import *

class NLP:
    def __init__(self,symbolic_type=ca.SX):
        self.w = PrimalVector(symbolic_type=symbolic_type)
        self.g = ConstraintVector(symbolic_type=symbolic_type)
        self.p = ConstraintVector(symbolic_type=symbolic_type)
