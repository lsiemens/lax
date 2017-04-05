#! /usr/bin/env python3

# the folowing comments are speculative:
#numerical soulution roadmap:
#1.  first given L and A check compatebility
#2.  given U(t) find eigenfunctions and eigenvalues of Lf=\lambda f
#3.  evolve eigenfunctions forward to t+dt using A and if A depends on U(t) compute U(t+dt)
#        by solving Lf=\lambda f when \lambda and f are known.
#4.  repeat step (3) untill t=t2, then compute U(t2) and return.

#note that depending on the structure of L, there may be multiple solutions for U(t).
#   all solutions for U(t) from solving L with know eigenfunctions should also solve the NPDE.   

#note2 it may be posible to add a third equation to replace \partial_t \lambda = 0.

# investigate applications of the LAX equations to multidimesional problems and to linear PDEs

import functions
import operators
import sympy

def simplify(f):
    return sympy.simplify(sympy.expand(f))

def operate(A, psi):
    print(sympy.collect(simplify(A(psi).to_sympy()), psi.to_sympy()))

x = functions.variable("x")
t = functions.variable("t")
c0 = functions.constant("c0")
c1 = functions.constant("c1")
c2 = functions.constant("c2")
c3 = functions.constant("c3")
u = functions.function("u", 2)(x, t)
f = functions.function("f", 2)(x, t)
g = functions.function("g", 2)(x, t)

Dx = operators.partial(x)
Dt = operators.partial(t)
M = operators.multiply

L = operators.add(M(functions.literal("-1"))(Dx(Dx)), M(u))
A = operators.add(M(c0)(Dx(Dx(Dx))), M(c1)(M(u)(Dx)), M(c2)(Dx(M(u))))

L = L(L)
A = A(A)

print("L_t(f)")
print(simplify(operators.commutator(Dt, L)(f).to_sympy()))
print("")

LAX = operators.add(operators.commutator(Dt, L), operators.commutator(L, A))

LAXf = simplify(LAX(f).to_sympy())

PDE = [LAXf.coeff(value.to_sympy()) for key, value in f.derived_functions().items() if "_" not in key]
conditions = [condition for condition in [LAXf.coeff(value.to_sympy()) for key, value in f.derived_functions().items() if "_" in key] if condition != 0]
constants = [const.to_sympy() for const in [c0, c1, c2]]
variables = [variable.to_sympy() for variable in [x, t]]
if len(PDE) > 0:
    PDE = PDE[0]
else:
    PDE = None

print(conditions)
solution = sympy.solve(conditions, constants, exclude=[x.to_sympy(), t.to_sympy()])
print(solution)

if all([simplify(condition.subs(solution)) == 0 for condition in conditions]):
    print("Passed sanity check: LAX(f(x, t))=E*f(x, t) where E is a PDE.")
else:
    raise ValueError("Sympy failed to find a solution.")

PDE = simplify(PDE.subs(solution))
print(PDE)

numerator, denominator = PDE.as_numer_denom()
if len(denominator.free_symbols & set(variables)) != 0:
    # the denominator is a function of x and/or t
    print("Assuming the denominator:" + str(denominator) + " is well behaved.\n")
print(simplify(numerator))
