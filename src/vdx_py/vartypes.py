from dataclasses import dataclass
from typing import Any

import numpy as np

@dataclass
class Primal:
    name : str
    size : int = 1
    lb : Any = -np.inf
    ub : Any = np.inf
    init : Any = 0.0
    init_mult : Any = 0.0

    def __post_init__(self):
        self.lb = self.lb*np.ones(self.size)
        self.ub = self.ub*np.ones(self.size)
        self.init = self.init*np.ones(self.size)
        self.init_mult = self.init_mult*np.ones(self.size)

    def __len__(self):
        return self.size

@dataclass
class Constraint:
    sym : Any
    lb : Any = 0.0
    ub : Any = 0.0
    init_mult : Any = 0.0

    def __post_init__(self):
        size = self.sym.size()[0]
        self.lb = self.lb*np.ones(size)
        self.ub = self.ub*np.ones(size)
        self.init_mult = self.init_mult*np.ones(size)

    def __len__(self):
        return self.sym.size()[0]

@dataclass
class CConstraint(Constraint):
    ub: Any = np.inf

@dataclass
class Parameter:
    name : str
    size : int = 1
    val : Any = 0.0
    def __post_init__(self):
        self.val = self.val*np.ones(self.size)

    def __len__(self):
        return self.size
