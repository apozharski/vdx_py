import casadi as ca
import numpy as np
from casadi import _casadi
from sortedcontainers import SortedSet
from itertools import islice

class IndexResult:
    __initialized = False
    def __init__(self, key, indices, vector):
        self.key = key
        self.indices = indices
        self.vector = vector
        self.__initialized = True

    def getsym(self):
        return ca.horzcat(*[self.vector.sym[idx] for idx in self.indices])

    def __getattr__(self, name):
        if name == "sym":
            return self.getsym()
        else:
            return np.squeeze(np.vstack([self.vector.__getattribute__(name)[idx] for idx in self.indices]))

    def __call__(self, **kwargs):
        if not kwargs:
            return self.vector.sym[self.indices]
        for (name,val) in kwargs.items():
            self.vector.__getattribute__(name)[self.indices] = val
    def __getitem__(self, key): return self.getsym()[key]

    def __matmul__(self, y): return _casadi.mtimes(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.mtimes(self.getsym(), y)
    def __rmatmul__(self, y): return _casadi.mtimes(y.getsym(), self.getsym()) if isinstance(y, IndexResult) else _casadi.mtimes(y, self.getsym())


    def __add__(self, y): return _casadi.plus(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.plus(self.getsym(), y)
    def __radd__(self, y): return _casadi.plus(y.getsym(), self.getsym()) if isinstance(y, IndexResult) else _casadi.plus(y, self.getsym())
    def __sub__(self, y): return _casadi.minus(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.minus(self.getsym(), y)
    def __rsub__(self, y): return _casadi.minus(y.getsym(), self.getsym()) if isinstance(y, IndexResult) else _casadi.minus(y, self.getsym())
    def __mul__(self, y): return _casadi.times(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.times(self.getsym(), y) 
    def __rmul__(self, y): return _casadi.times(y.getsym(), self.getsym()) if isinstance(y, IndexResult) else _casadi.times(y, self.getsym())
    def __div__(self, y): return _casadi.rdivide(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.rdivide(self.getsym(), y) 
    def __rdiv__(self, y): return _casadi.rdivide(y.getsym(), self.getsym()) if isinstance(y, IndexResult) else _casadi.rdivide(y, self.getsym())
    def __truediv__(self, y): return _casadi.rdivide(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.rdivide(self.getsym(), y) 
    def __rtruediv__(self, y): return _casadi.rdivide(y.getsym(), self.getsym()) if isinstance(y, IndexResult) else _casadi.rdivide(y, self.getsym())
    def __lt__(self, y): return _casadi.lt(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.lt(self.getsym(), y) 
    def __rlt__(self, y): return _casadi.lt(y.getsym(), self.getsym()) if isinstance(y, IndexResult) else _casadi.lt(y, self.getsym())
    def __le__(self, y): return _casadi.le(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.le(self.getsym(), y) 
    def __rle__(self, y): return _casadi.le(y.getsym(), self.getsym()) if isinstance(y, IndexResult) else _casadi.le(y, self.getsym())
    def __gt__(self, y): return _casadi.lt(y.getsym(), self.getsym()) if isinstance(y, IndexResult) else _casadi.lt(y, self.getsym())
    def __rgt__(self, y): return _casadi.lt(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.lt(self.getsym(), y) 
    def __ge__(self, y): return _casadi.le(y.getsym(), self.getsym()) if isinstance(y, IndexResult) else _casadi.le(y, self.getsym())
    def __rge__(self, y): return _casadi.le(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.le(self.getsym(), y) 
    def __eq__(self, y): return _casadi.eq(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.eq(self.getsym(), y) 
    def __req__(self, y): return _casadi.eq(y.getsym(), self.getsym()) if isinstance(y, IndexResult) else _casadi.eq(y, self.getsym())
    def __ne__(self, y): return _casadi.ne(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.ne(self.getsym(), y) 
    def __rne__(self, y): return _casadi.ne(y.getsym(), self.getsym()) if isinstance(y, IndexResult) else _casadi.ne(y, self.getsym())
    def __pow__(self, n): return _casadi.power(self.getsym(), n)
    def __rpow__(n, self): return _casadi.power(self.getsym(), n)
    def __arctan2__(self, y): return _casadi.atan2(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.atan2(self.getsym(), y) 
    def __rarctan2__(y, self): return _casadi.atan2(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.atan2(self.getsym(), y) 
    def fmin(self, y): return _casadi.fmin(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.fmin(self.getsym(), y) 
    def fmax(self, y): return _casadi.fmax(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.fmax(self.getsym(), y) 
    def __fmin__(self, y): return _casadi.fmin(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.fmin(self.getsym(), y) 
    def __rfmin__(y, self): return _casadi.fmin(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.fmin(self.getsym(), y) 
    def __fmax__(self, y): return _casadi.fmax(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.fmax(self.getsym(), y) 
    def __rfmax__(y, self): return _casadi.fmax(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.fmax(self.getsym(), y) 
    def logic_and(self, y): return _casadi.logic_and(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.logic_and(self.getsym(), y) 
    def logic_or(self, y): return _casadi.logic_or(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.logic_or(self.getsym(), y) 
    def fabs(self): return _casadi.fabs(self.getsym())
    def sqrt(self): return _casadi.sqrt(self.getsym())
    def sin(self): return _casadi.sin(self.getsym())
    def cos(self): return _casadi.cos(self.getsym())
    def tan(self): return _casadi.tan(self.getsym())
    def arcsin(self): return _casadi.asin(self.getsym())
    def arccos(self): return _casadi.acos(self.getsym())
    def arctan(self): return _casadi.atan(self.getsym())
    def sinh(self): return _casadi.sinh(self.getsym())
    def cosh(self): return _casadi.cosh(self.getsym())
    def tanh(self): return _casadi.tanh(self.getsym())
    def arcsinh(self): return _casadi.asinh(self.getsym())
    def arccosh(self): return _casadi.acosh(self.getsym())
    def arctanh(self): return _casadi.atanh(self.getsym())
    def exp(self): return _casadi.exp(self.getsym())
    def log(x): return _casadi.log(self.getsym())
    def log10(self): return _casadi.log10(self.getsym())
    def log1p(self): return _casadi.log1p(self.getsym())
    def expm1(self): return _casadi.expm1(self.getsym())
    def floor(self): return _casadi.floor(self.getsym())
    def ceil(self): return _casadi.ceil(self.getsym())
    def erf(self): return _casadi.erf(self.getsym())
    def sign(self): return _casadi.sign(self.getsym())
    def fmod(self, y): return _casadi.mod(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.mod(self.getsym(), y) 
    def hypot(self, y): return _casadi.hypot(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.hypot(self.getsym(), y) 
    def remainder(self, y): return _casadi.remainder(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.remainder(self.getsym(), y) 
    def __copysign__(self, y): return _casadi.copysign(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.copysign(self.getsym(), y) 
    def __rcopysign__(y, self): return _casadi.copysign(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.copysign(self.getsym(), y) 
    def copysign(self, y): return _casadi.copysign(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.copysign(self.getsym(), y) 
    def rcopysign(y, self): return _casadi.copysign(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.copysign(self.getsym(), y) 
    def __constpow__(self, y): return _casadi.constpow(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.constpow(self.getsym(), y) 
    def __rconstpow__(y, self): return _casadi.constpow(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.constpow(self.getsym(), y) 
    def constpow(self, y): return _casadi.constpow(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.constpow(self.getsym(), y) 
    def rconstpow(y, self): return _casadi.constpow(self.getsym(), y.getsym()) if isinstance(y, IndexResult) else _casadi.constpow(self.getsym(), y) 


class IndexRange():
    def __init__(self, tup):
        self.ind = tup


        

class Variable:
    def __init__(self,vector):
        self.vector = vector
        self.ind_map = dict() # tuples to lists of indices
        self.ranges = dict()

    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key,slice):
            key = (key,)
        if not isinstance(key, tuple):
            raise KeyError("Argument to variable index must be a tuple or integer")

        inds = self._get_ind_list(key) # TODO(@anton) is this sorted always
        indices = [self.ind_map[ind] for ind in inds]
        return IndexResult(key, indices, self.vector)

    def __setitem__(self, key, value):
        if isinstance(key, int):
            key = (key,)
        if not isinstance(key, tuple):
            raise KeyError("Argument to variable index must be a tuple or integer")

        indices = self.vector.add_var(value)
        self._update_range(key)
        self.ind_map[key] = indices

    def _update_range(self, key):
        if key == ():
            return
        if () in self.ranges:
            self.ranges[()].add(key[0])
        else:
            self.ranges[()] = SortedSet([key[0]])
            
        for i in range(len(key)-1):
            k = key[0:i+1]
            if k in self.ranges:
                self.ranges[k].add(key[i+1])
            else:
                self.ranges[k] = SortedSet([key[i+1]])
            
    def _get_ind_list(self, key):
        inds = []
        if key == ():
            inds = [()]
            return inds
        nkey = len(key)
        # TODO(@anton): there must be a more efficient way
        for ind in self.ind_map.keys():
            still_valid = True
            if not len(ind) == nkey:
                continue
            worklist=[()]
            for (i,(k_key,k_ind)) in enumerate(zip(key,ind)):
                found = False
                new_worklist = []
                for prefix in worklist:
                    if isinstance(k_key,slice):
                        r = self.ranges[prefix]
                        kmin,kmax = r[0],r[-1]
                        s_key = SortedSet(islice(range(kmin,kmax+1),k_key.start,k_key.stop,k_key.step)).intersection(r)
                        if k_ind in s_key:
                            new_worklist.append(prefix+(k_ind,))
                            found = True
                            break
                    else:
                        if k_ind == k_key:
                            new_worklist.append(prefix+(k_ind,))
                            found = True
                            break
                worklist = new_worklist
                if not found:
                    still_valid=False
                    break
                    
                        
            if still_valid:
                inds.append(ind)
        return inds
            
