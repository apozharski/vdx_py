from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, override

import numpy as np
import casadi as ca

class Relaxation(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def relax(self,idx, constr, name):
        pass

class HardConstraint(Relaxation):
    def __init__(self):
        pass

    @override
    def relax(self, idx, constr, name):
        return constr

class Ell1Relaxation(Relaxation):
    def __init__(self, nlp):
        self.nlp = nlp

    @override
    def relax(self, idx, constr, name):
        if f"rho_{name}" not in self.nlp.p:
            getattr(self.nlp.p, f"rho_{name}")[()] = Parameter(f"rho_{name}", 1, val=1.0)
        rho = getattr(self.nlp.p, f"rho_{name}")[()].sym
        N = len(constr)
        ind_lb = constr.lb != -np.inf
        ind_ub = constr.ub != np.inf
        slack_lb = 0.0
        slack_ub = 0.0
        # Assume no double assignments
        # TODO(@anton) implement checks
        if np.any(ind_lb):
            getattr(self.nlp.w, f"{name}_slack_lb")[idx] = Primal(f"{name}_slack_lb", sum(ind_lb), lb=0.0, ub=np.inf)
            slack_lb = getattr(self.nlp.w, f"{name}_slack_lb")[idx].sym
            self.nlp.f += rho*ca.norm_1(slack_lb)
        if np.any(ind_ub):
            getattr(self.nlp.w, f"{name}_slack_ub")[idx] = Primal(f"{name}_slack_ub", sum(ind_ub), lb=0.0, ub=np.inf)
            slack_ub = getattr(self.nlp.w, f"{name}_slack_ub")[idx].sym
            self.nlp.f += rho*ca.norm_1(slack_ub)

        new_constr = Constraint(
            ca.vertcat(
                constr.sym[ind_lb] + slack_lb,
                constr.sym[ind_ub] - slack_ub,
            ),
            lb=np.hstack([constr.lb[ind_lb], -np.inf*np.ones(sum(ind_ub))]),
            ub=np.hstack([np.inf*np.ones(sum(ind_ub)),constr.ub[ind_ub]]),
        )
        return new_constr

class Ell2Relaxation(Relaxation):
    def __init__(self, nlp):
        self.nlp = nlp

    @override
    def relax(self, idx, constr, name):
        if f"rho_{name}" not in self.nlp.p:
            getattr(self.nlp.p, f"rho_{name}")[()] = Parameter(f"rho_{name}", 1, val=1.0)
        rho = getattr(self.nlp.p, f"rho_{name}")[()].sym
        N = len(constr)
        # Assume no double assignments
        # TODO(@anton) implement checks
        getattr(self.nlp.w, f"{name}_slack")[idx] = Primal(f"{name}_slack_{"_".join([str(i) for i in idx])}", N)
        slack = getattr(self.nlp.w, f"{name}_slack")[idx].sym
        self.nlp.f += 0.5*rho*ca.norm_2(slack)**2
        new_constr = Constraint(
            constr.sym - slack,
            lb=constr.lb,
            ub=constr.ub,
        )
        return new_constr


class EllInfRelaxation(Relaxation):
    def __init__(self, nlp):
        self.nlp = nlp

    @override
    def relax(self, idx, constr, name):
        if f"rho_{name}" not in self.nlp.p:
            getattr(self.nlp.p, f"rho_{name}")[()] = Parameter(f"rho_{name}", 1, val=1.0)
        rho = getattr(self.nlp.p, f"rho_{name}")[()].sym
        N = len(constr)
        ind_lb = constr.lb != -np.inf
        ind_ub = constr.ub != np.inf

        if f"{name}_slack_lb" not in self.nlp.w:
            getattr(self.nlp.w, f"{name}_slack_lb")[()] = Primal(f"{name}_slack_lb", 1, lb=0.0, ub=np.inf)
        slack_lb = getattr(self.nlp.w, f"{name}_slack_lb")[()].sym
        if f"{name}_slack_ub" not in self.nlp.w:
            getattr(self.nlp.w, f"{name}_slack_ub")[()] = Primal(f"{name}_slack_ub", 1, lb=0.0, ub=np.inf)
        slack_ub = getattr(self.nlp.w, f"{name}_slack_ub")[()].sym
        if f"{name}_slack" not in self.nlp.w:
            getattr(self.nlp.w, f"{name}_slack")[()] = Primal(f"{name}_slack", 1, lb=0.0, ub=np.inf)
            slack = getattr(self.nlp.w, f"{name}_slack")[()]
            getattr(self.nlp.g, f"{name}_slack_max")[()] = Constraint(ca.vertcat(slack - slack_lb, slack - slack_ub), lb=0.0, ub=np.inf)
            self.nlp.f += rho*slack
        new_constr = Constraint(
            ca.vertcat(
                constr.sym[ind_lb] + slack_lb,
                constr.sym[ind_ub] - slack_ub,
            ),
            lb=np.hstack([constr.lb[ind_lb], -np.inf*np.ones(sum(ind_ub))]),
            ub=np.hstack([np.inf*np.ones(sum(ind_ub)),constr.ub[ind_ub]]),
        )
        return new_constr


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
    relax : Relaxation = HardConstraint()

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
