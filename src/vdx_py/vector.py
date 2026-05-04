import casadi as ca
import numpy as np
from .variable import Variable
from dataclasses import dataclass
from typing import Any
from termcolor import colored

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
class Parameter:
    name : str
    size : int = 1
    val : Any = 0.0
    def __post_init__(self):
        self.val = self.val*np.ones(self.size)

    def __len__(self):
        return self.size

class Vector:
    def __init__(self, symbolic_type=ca.SX):
        self.symbolic_type = symbolic_type
        self.new_sym = symbolic_type.sym

        # Symbolics
        self.sym = self.symbolic_type([])
        self.nelem = 0
        self.variables = dict()

    def __getattr__(self, name):
        if name not in self.variables:
            var = Variable(self)
            self.variables[name] = var

        return self.variables[name]

class PrimalVector(Vector):
    def __init__(self, symbolic_type=ca.SX):
        super().__init__(symbolic_type=symbolic_type)
        # Initial Values
        self.lb = np.array([])
        self.ub = np.array([])
        self.init = np.array([])
        self.init_mult = np.array([])

        # Results
        self.res = np.array([])
        self.mult = np.array([])

    def add_var(self, value):
        if not isinstance(value, Primal):
            ValueError("You did not pass a Primal object to this initializer")
        # TODO(@anton) implement the caching that vdx does in matlab

        self.sym = ca.vertcat(self.sym, self.new_sym(value.name, value.size))
        self.lb = np.append(self.lb, value.lb)
        self.ub = np.append(self.ub, value.ub)
        self.init = np.append(self.init, value.init)
        self.init_mult = np.append(self.init_mult, value.init_mult)
        self.res = np.append(self.res, value.init)
        self.mult = np.append(self.mult, value.init_mult)
        indices = range(self.nelem, self.nelem + len(value))
        self.nelem += len(value)
        return indices

    def __str__(self):
        lines = []
        max_len = 16 # some minimum length
        for ii in range(self.nelem):
            sym_str = str(self.sym[ii])
            max_len = max(max_len, len(sym_str))
            lines.append((sym_str,
                          self.lb[ii],
                          self.ub[ii],
                          self.init[ii],
                          self.init_mult[ii],
                          self.res[ii],
                          self.mult[ii]))

        ret = f"|{'sym':{max_len}}|{'lb':12}|{'ub':12}|{'init':12}|{'init_mult':12}|{'res':12}|{'mult':12}|\n"
        for sym,lb,ub,init,init_mult,res,mult in lines:
            ret += f"|{sym:{max_len}}|{lb:<12.4g}|{ub:<12.4g}|{init:<12.4g}|{init_mult:<12.4g}|{res:<12.4g}|{mult:<12.4g}|\n"

        return ret


class ConstraintVector(Vector):
    def __init__(self, symbolic_type=ca.SX):
        super().__init__(symbolic_type=symbolic_type)
        self.lb = np.array([])
        self.ub = np.array([])
        self.init_mult = np.array([])

        # Results
        self.val = np.array([])
        self.mult = np.array([])

    def add_var(self, value):
        if not isinstance(value, Constraint):
            ValueError("You did not pass a Constraint object to this initializer")
        # TODO(@anton) implement the caching that vdx does in matlab

        self.sym = ca.vertcat(self.sym, value.sym)
        self.lb = np.append(self.lb, value.lb)
        self.ub = np.append(self.ub, value.ub)
        self.init_mult = np.append(self.init_mult, value.init_mult)
        self.val = np.append(self.mult, 0.0*value.init_mult)
        self.mult = np.append(self.mult, value.init_mult)
        indices = range(self.nelem, self.nelem + len(value))
        self.nelem += len(value)
        return indices

    def __str__(self):
        lines = []
        max_len = 16 # some minimum length
        for ii in range(self.nelem):
            sym_str = str(self.sym[ii])
            max_len = max(max_len, len(sym_str))
            lines.append((sym_str,
                          self.lb[ii],
                          self.ub[ii],
                          self.init_mult[ii],
                          self.val[ii],
                          self.mult[ii]))

        ret = f"|{'sym':{max_len}}|{'lb':12}|{'ub':12}|{'init_mult':12}|{'val':12}|{'mult':12}|\n"
        for sym,lb,ub,init_mult,res,mult in lines:
            ret += f"|{sym:{max_len}}|{lb:<12.4g}|{ub:<12.4g}|{init_mult:<12.4g}|{res:<12.4g}|{mult:<12.4g}|\n"

        return ret

    def print_result(self, tol=1e-6, only_viol=False):
        for ii in range(self.nelem):
            if self.val[ii] < self.lb[ii]-tol or self.val[ii] > self.ub[ii] + tol:
                print(colored(f"{ii}: {self.lb[ii]:.6f}  {self.val[ii]:.6f}  {self.ub[ii]:.6f}", "red"))
            elif not only_viol:
                print(f"{ii}: {self.lb[ii]:.6f}  {self.val[ii]:.6f}  {self.ub[ii]:.6f}")

class ParameterVector(Vector):
    def __init__(self, symbolic_type=ca.SX):
        super().__init__(symbolic_type=symbolic_type)
        self.val = np.array([])

    def add_var(self, value):
        if not isinstance(value, Parameter):
            ValueError("You did not pass a Parameter object to this initializer")
        # TODO(@anton) implement the caching that vdx does in matlab

        self.sym = ca.vertcat(self.sym, self.new_sym(value.name, value.size))
        self.val = np.append(self.val, value.val)
        indices = range(self.nelem, self.nelem + len(value))
        self.nelem += len(value)
        return indices

    def __str__(self):
        lines = []
        max_len = 16 # some minimum length
        for ii in range(self.nelem):
            sym_str = str(self.sym[ii])
            max_len = max(max_len, len(sym_str))
            lines.append((sym_str, self.val[ii]))

        ret = f"|{'sym':{max_len}}|{'val':12}|\n"
        for sym,val in lines:
            ret += f"|{sym:{max_len}}|{val:<12.4g}|\n"

        return ret
