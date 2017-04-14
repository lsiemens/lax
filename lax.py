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

import sympy
import numpy

import timeout
import functions
import operators

class LaxError(Exception):
    pass

class LaxPair:
    def __init__(self, L, A, constants, x, t):
        self.L = L
        self.A = A
        self.constants = constants
        self.x = x
        self.t = t
                
        self.f = functions.function("f", 2)(self.x, self.t)
        #LAX
#        self.commutator_t = operators.commutator(operators.partial(self.t), self.L)
#        self.commutator_A = operators.commutator(self.L, self.A)
        # partial_t^2 \psi = A\psi
        self.commutator_t = operators.commutator(operators.partial(self.t)(operators.partial(self.t)), self.L)
        self.commutator_A = operators.commutator(self.L, self.A)
        #LAX
#        self.commutator_t = operators.commutator(operators.partial(self.t), self.L)
#        self.commutator_A = operators.commutator(self.L, self.A)
        print("T commutator: " + str(self._simplify(self.commutator_t(self.f).to_sympy())) + "\n")
        print("A commutator: " + str(self._simplify(self.commutator_A(self.f).to_sympy())) + "\n")

        self.LAX = operators.add(self.commutator_t, self.commutator_A)
#       #only comment temporaily ----- READ THIS ------
#        try:
#            self.f = functions.function("f", 2)(self.x, self.t)        
#        except ValueError:
#            raise RuntimeError('The token "f" is already in use and cannot be defined by LaxPair.')
        
        LAXf = self._simplify(self.LAX(self.f).to_sympy())
        PDE = [self._simplify(LAXf.coeff(value.to_sympy())) for key, value in self.f.derived_functions().items() if "_" not in key]
        if len(PDE) == 1:
            PDE = PDE[0]
        elif len(PDE) == 0:
            PDE = 0
        else:
            raise RuntimeError("Failed to seporate the PDE from the compatibility conditions.")
        
        if (PDE == 0):
            raise LaxError("The PDE is trivially zero.")
        
        compatibility = [condition for condition in [self._simplify(LAXf.coeff(value.to_sympy())) for key, value in self.f.derived_functions().items() if "_" in key] if condition != 0]
        print(compatibility)
        constants = [constant.to_sympy() for constant in self.constants]
        variables = [variable.to_sympy() for variable in [self.x, self.t]]
        
        solutions = sympy.solve(compatibility, constants, exclude=variables)
        if isinstance(solutions, dict):
            #ensure that solutions is a list of dicts
            solutions = [solutions]

        print("solutions: " + str(solutions) + "\n")
        
        solutions = [solution for solution in solutions if all([len(value.free_symbols & set(variables)) == 0 for key, value in solution.items()])]

        print("solutions: " + str(solutions) + "\n")
        
        if not (all([self._simplify(condition.subs(solution)) == 0 for condition in compatibility for solution in solutions]) and len(solutions) != 0):
            print(compatibility)
            print([self._simplify(condition.subs(solution)) for condition in compatibility for solution in solutions])
            raise LaxError("One or more compatibility conditions failed.")
        
        PDE = [equation for equation in [self._simplify(PDE.subs(solution)) for solution in solutions] if equation != 0]
        if len(PDE) == 0:
            raise LaxError("All compatible PDEs are trivially zero.")
        
        self.PDE = PDE

        print("PDEs found: [\n" + ",\n\n".join([str(P) for P in self.PDE]) + "\n]")

    def _simplify(self, expression):
        return sympy.simplify(sympy.expand(expression))
### example

constants = []
x = functions.variable("x")
t = functions.variable("t")
u = functions.function("u", 2)(x, t)

Dx = operators.partial(x)
Dt = operators.partial(t)
M = operators.multiply
add = operators.add

def new_constant():
#    const = functions.function("c" + str(len(constants) + 1), 2)(x, t)
    const = functions.constant("c" + str(len(constants) + 1))
    constants.append(const)
    return operators.multiply(const)

#L = add(M(functions.literal("-1"))(Dx(Dx)),M(u))

#A = add(
#        new_constant()(Dx(Dx(Dx))),
#        new_constant()(Dx(M(u))),
#        new_constant()(M(u)(Dx))
#        )

L = add(Dx,operators.commutator(Dx, M(u)))

#A = add(
#        new_constant()(Dx),
#        new_constant()(Dx(Dx)),
#        new_constant()(Dx(M(u))),
#        new_constant()(M(u)(Dx)),
#        new_constant()(Dt),
#        new_constant()(Dt(M(u))),
#        new_constant()(M(u)(Dt)),
#        new_constant()(operators.commutator(Dx, M(u))(Dx)),
#        new_constant()(operators.commutator(Dt, M(u))(Dt))
#        )

A = add(
#####        new_constant()(Dx),
##        new_constant()(Dx(Dx)),
####        new_constant()(Dx(M(u))),
####        new_constant()(M(u)(Dx)),
#        new_constant()(Dt),
###        new_constant()(Dt(M(u))),
###        new_constant()(M(u)(Dt)),
##        new_constant()(operators.commutator(Dx, M(u))(Dx)),
        new_constant()(operators.commutator(Dt, M(u))(Dt))
        )

KdV = LaxPair(L, A, constants, x, t)
