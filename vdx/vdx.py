class Problem():

    def __init__(self):
        self.w =


class Vector():

    def __init__(self, problem):
        self.sym = []
        self.casadi_type = []
        self.problem = problem

        self.variables = {}
        self.pending_assignments = []
        self.length = 0

    def __getattr__(self, item):
        pass

    def __setattr__(self, item, value):
        pass

    def __apply_queued_assignments__(self):
        pass

    def add_variable(self, indices, symbolic, *varargs):
        pass


class PrimalVector(Vector):

    def __init__(self):
        self.lb = []
        self.ub = []
        self.init = []
        self.init_mult = []

        self.res = []
        self.violation = []
        self.mult = []
        
class ParameterVector(Vector):

    def __init__(self):
        self.val = []

        self.mult = []

class ConstraintVector(Vector):

    def __init__(self):
        self.lb = []
        self.ub = []
        self.init_mult = []

        self.eval = []
        self.violation = []
        self.mult = []

class Variable():

    def __init__(self, vector, name):
        self.indicies = []
        self.vector = vector
        self.name = name

    def __getattr__(self, item):
        pass

    def __getitem__(self, key):
        pass

    def __setitem__(self, key):
        pass
