import casadi as ca

class NLP:
    def __init__(symbolic_type=ca.SX):
        self.w = symbolic_type()
