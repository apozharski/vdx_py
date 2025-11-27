import casadi as ca

class IndexResult:
    __initialized = False
    def __init__(self, key, indices, vector):
        self.key = key
        self.indices = indices
        self.vector = vector
        self.__initialized = True

    def __setattr__(self, name, value):
        if self.__initialized:
            pass# your __setattr__ implementation here
        else:
            object.__setattr__(self, name, value)

    def __call__(self, **kwargs):
        if not kwargs:
            return self.vector.sym[self.indices]
        for (name,val) in kwargs.items():
            self.vector.__getattribute__(name)[self.indices] = val


class Variable:
    def __init__(self,vector):
        self.vector = vector
        self.indices = dict() # tuples to lists of indices

    def __getitem__(self, key):
        if isinstance(key, int):
            key = (key,)
        if not isinstance(key, tuple):
            raise KeyError("Argument to variable index must be a tuple or integer")

        indices = self.indices[key]

        return IndexResult(key, indices, self.vector)

    def __setitem__(self, key, value):
        if isinstance(key, int):
            key = (key,)
        if not isinstance(key, tuple):
            raise KeyError("Argument to variable index must be a tuple or integer")

        indices = self.vector.add_var(value)
        self.indices[key] = indices
