import casadi as ca
from .vector import *
from .nlp import NLP

class MPCC(NLP):
    def __init__(self,symbolic_type=ca.SX, name="nlp"):
        super().__init__(symbolic_type=symbolic_type, name=name)
        self.G = ConstraintVector(symbolic_type=symbolic_type)
        self.H = ConstraintVector(symbolic_type=symbolic_type)

    def to_casadi_dict(self):
        return {'x': self.w.sym,
                'g': self.g.sym,
                'p': self.p.sym,
                'G': self.H.sym,
                'H': self.p.sym,
                'f': self.f}

    def create_solver(self, casadi_opts, plugin="ipopt"):
        # TODO(@anton) implement `mpccsol`
        pass
        #self.solver = ca.nlpsol(self.name, plugin, self.to_casadi_dict(), casadi_opts);

    def solve(self, casadi_opts=dict()):
        if self.solver is None:
            self.create_solver(casadi_opts)

        nlp_results = self.solver(x0=self.w.init,
                                 lbx=self.w.lb,
                                 ubx=self.w.ub,
                                 lbg=self.g.lb,
                                 ubg=self.g.ub,
                                 lam_g0=self.g.init_mult,
                                 lam_x0=self.w.init_mult,
                                 p=self.p.val)
        self.w.res = np.squeeze(nlp_results['x'])
        self.w.mult = np.squeeze(nlp_results['lam_x'])
        self.g.val = np.squeeze(nlp_results['g'])
        self.g.mult = np.squeeze(nlp_results['lam_g'])
        self.p.mult = np.squeeze(nlp_results['lam_p'])
        #
        self.f_result = nlp_results['f']
        # TODO Also solve for #s
