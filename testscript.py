import casadi as ca
from vdx_py.nlp import *
from vdx_py.vector import *
a = ca.SX.sym('a')
f = ca.Function('f',[a], [a+1])
nlp = NLP()
x1 = nlp.w.x[1] = Primal("x",1)
nlp.w.x[2] = Primal("x",1)
print(f(a))
print(f(nlp.w.x[1])+nlp.w.x[1])

print(x1)
