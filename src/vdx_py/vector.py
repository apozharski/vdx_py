from copy import copy, deepcopy
from itertools import chain
from operator import itemgetter

import casadi as ca
import numpy as np

from .vartypes import *
from .variable import Variable
from dataclasses import dataclass
from typing import Any
from termcolor import colored

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
    def __len__(self):
        return self.sym.size(1)
    def __copy__(self):
        """
        Always deepcopy the vector, this is sufficient to get a copy of an NLP
        """
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        new_variables = {name: deepcopy(var) for name, var in result.variables.items()}
        result.__dict__["variables"] = new_variables
        for var in new_variables.values():
            var.vector = result
        return result

    def resort_vector(self):
        idxlst = [((tup,var), varname, ind) for varname,var in self.variables.items() for tup,ind in var.ind_map.items()]
        new_idxlst = sorted(idxlst, key=itemgetter(0))
        reorder = list(chain(ind for _,_,ind in new_idxlst))
        start = 0
        ind_map = [0]*self.nelem
        rev_ind_map = [0]*self.nelem
        for ((tup,var), _,ind) in new_idxlst:
            n = len(ind)
            for ii,jj in zip(var.ind_map[tup],range(start, start+n)):
                ind_map[ii] = jj
                rev_ind_map[jj] = ii
            var.ind_map[tup] = range(start, start+n)
            start += n
        return new_idxlst,reorder,ind_map,rev_ind_map


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
        self.violation = np.array([])

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
        self.violation = np.append(self.violation, np.zeros(len(value)))
        indices = range(self.nelem, self.nelem + len(value))
        self.nelem += len(value)
        return indices

    def __copy__(self):
        result = super().__copy__()
        result.sym = copy(self.sym)
        result.lb = copy(self.lb)
        result.ub = copy(self.ub)
        result.init = copy(self.init)
        result.init_mult = copy(self.init_mult)
        result.res = copy(self.res)
        result.mult = copy(self.mult)
        result.violation = copy(self.violation)
        return result

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
                          self.mult[ii],
                          self.violation[ii],
                          ))

        ret = f"|{'sym':{max_len}}|{'lb':12}|{'ub':12}|{'init':12}|{'init_mult':12}|{'res':12}|{'mult':12}|{'violation':12}|\n"
        for sym,lb,ub,init,init_mult,res,mult,violation in lines:
            ret += f"|{sym:{max_len}}|{lb:<12.4g}|{ub:<12.4g}|{init:<12.4g}|{init_mult:<12.4g}|{res:<12.4g}|{mult:<12.4g}|{violation:<12.4g}|\n"

        return ret

    def resort_vector(self):
        new_idxlst, reorder, ind_map, rev_ind_map = super().resort_vector() # Get new ordering
        self.sym = ca.vertcat(*[self.sym[idx] for idx in reorder])
        self.lb = np.hstack([self.lb[idx] for idx in reorder])
        self.ub = np.hstack([self.ub[idx] for idx in reorder])
        self.init = np.hstack([self.init[idx] for idx in reorder])
        self.init_mult = np.hstack([self.init_mult[idx] for idx in reorder])
        self.res = np.hstack([self.res[idx] for idx in reorder])
        self.mult = np.hstack([self.mult[idx] for idx in reorder])
        self.violation = np.hstack([self.violation[idx] for idx in reorder])
        return ind_map, rev_ind_map

class ConstraintVector(Vector):
    def __init__(self, symbolic_type=ca.SX):
        super().__init__(symbolic_type=symbolic_type)
        self.lb = np.array([])
        self.ub = np.array([])
        self.init_mult = np.array([])

        # Results
        self.val = np.array([])
        self.mult = np.array([])
        self.violation = np.array([])

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
        self.violation = np.append(self.violation, np.zeros(len(value)))
        indices = range(self.nelem, self.nelem + len(value))
        self.nelem += len(value)
        return indices

    def __copy__(self):
        result = super().__copy__()
        result.sym = copy(self.sym)
        result.lb = copy(self.lb)
        result.ub = copy(self.ub)
        result.init_mult = copy(self.init_mult)
        result.val = copy(self.val)
        result.mult = copy(self.mult)
        result.violation = copy(self.violation)
        return result


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
                          self.mult[ii],
                          self.violation[ii],
                          ))

        ret = f"|{'sym':{max_len}}|{'lb':12}|{'ub':12}|{'init_mult':12}|{'val':12}|{'mult':12}|{'violation':12}|\n"
        for sym,lb,ub,init_mult,res,mult,violation in lines:
            ret += f"|{sym:{max_len}}|{lb:<12.4g}|{ub:<12.4g}|{init_mult:<12.4g}|{res:<12.4g}|{mult:<12.4g}|{violation:<12.4g}|\n"

        return ret

    def print_result(self, tol=1e-6, only_viol=False):
        for ii in range(self.nelem):
            if self.val[ii] < self.lb[ii]-tol or self.val[ii] > self.ub[ii] + tol:
                print(colored(f"{ii}: {self.lb[ii]:.6f}  {self.val[ii]:.6f}  {self.ub[ii]:.6f}", "red"))
            elif not only_viol:
                print(f"{ii}: {self.lb[ii]:.6f}  {self.val[ii]:.6f}  {self.ub[ii]:.6f}")

    def resort_vector(self):
        new_idxlst, reorder, ind_map, rev_ind_map = super().resort_vector() # Get new ordering
        self.sym = ca.vertcat(*[self.sym[idx] for idx in reorder])
        self.lb = np.hstack([self.lb[idx] for idx in reorder])
        self.ub = np.hstack([self.ub[idx] for idx in reorder])
        self.init_mult = np.hstack([self.init_mult[idx] for idx in reorder])
        self.val = np.hstack([self.val[idx] for idx in reorder])
        self.mult = np.hstack([self.mult[idx] for idx in reorder])
        self.violation = np.hstack([self.violation[idx] for idx in reorder])
        return ind_map, rev_ind_map

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

    def __copy__(self):
        result = super().__copy__()
        result.sym = copy(self.sym)
        result.val = copy(self.val)
        return result

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

    def resort_vector(self):
        new_idxlst, reorder, ind_map, rev_ind_map = super().resort_vector() # Get new ordering
        self.sym = ca.vertcat(*[self.sym[idx] for idx in reorder])
        self.val = np.hstack([self.val[idx] for idx in reorder])
        return ind_map, rev_ind_map
