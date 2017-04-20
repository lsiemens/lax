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
    def __init__(self, L, A, constants, x, t, tout=None):
        self.L = L
        self.A = A
        self.constants = constants
        self.x = x
        self.t = t
        
        self.LAX = None
        self.PDE = None
        
        if tout is None:
            tout = 0
                
        #LAX
#        self.commutator_t = operators.commutator(operators.partial(self.t), self.L)
#        self.commutator_A = operators.commutator(self.L, self.A)
        # partial_t^2 \psi = A\psi
        self.commutator_t = operators.commutator(operators.partial(self.t)(operators.partial(self.x)), self.L)
        self.commutator_A = operators.commutator(self.L, self.A)
        #LAX
#        self.commutator_t = operators.commutator(operators.partial(self.t)(operators.partial(self.t)), self.L)
#        self.commutator_A = operators.commutator(self.L, operators.add(operators.commutator(operators.partial(self.t), self.A), self.A(self.A)))

        self.LAX = operators.add(self.commutator_t, self.commutator_A)
        #only comment temporaily ----- READ THIS ------
        try:
            self.f = functions.function("f", 2)(self.x, self.t)        
        except ValueError:
            raise RuntimeError('The token "f" is already in use and cannot be defined by LaxPair.')

        with timeout.timeout(tout):
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
            constants = [constant.to_sympy() for constant in self.constants]
            variables = [variable.to_sympy() for variable in [self.x, self.t]]
            
            solutions = sympy.solve(compatibility, constants, exclude=variables, dict=True)
            if isinstance(solutions, dict):
                #ensure that solutions is a list of dicts
                solutions = [solutions]
    
            solutions = [solution for solution in solutions if all([len(value.free_symbols & set(variables)) == 0 for key, value in solution.items()])]
            
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

class GenerateLax:
    def __init__(self, tout=60):
        self.tout = tout
    
        self.constants = None
        self.x = None
        self.t = None
        self.u = None

        #dict of dicts {"function":function_call, "min_args":0, "max_args":20, "weight":2}
        self.operators = {}
        self.L_operator_distribution = {}
        self.A_operator_distribution = {}
        self.num_args_distribution = numpy.array([1, 1, 1, 1])
        
        self._reset()

    def findLaxPair(self):
        self._reset()

        L = self._generateOperator(self.L_operator_distribution)
        A = self._generateOperator(self.A_operator_distribution)
        
        return LaxPair(L, A, self.constants, self.x, self.t, self.tout)

    def _generateOperator(self, distribution):
        #use sself.operators and distribution
        operator = None
        return operator
    
    def _new_constant(self):
        const = functions.constant("c" + str(len(self.constants) + 1))
        self.constants.append(const)
        return operators.multiply(const)
        
    def _reset(self):
        functions.reset()
        self.constants = []
        self.x = functions.variable("x")
        self.t = functions.variable("t")
        self.u = functions.function("u", 2)(self.x, self.t)

        self.operators = {"add":{"function":operators.add, "min_args":1, "max_args":None, "weight":1},
                          "commutator":{"function":operators.commutator, "min_args":2, "max_args":2, "weight":1},
                          "multiply_u":{"function":operators.multiply(self.u), "min_args":0, "max_args":1, "weight":1},
                          "partial_x":{"function":operators.partal(self.x), "min_args":0, "max_args":1, "weight":1},
                          "partial_t":{"function":operators.partial(self.t), "min_args":0, "max_args":1, "weight":1},
                          "constant":{"function":self._new_constant(), "min_args":1, "max_args":1, "weight":1}}

        self.L_operator_distribution = {"add":1, "commutator":1, "multiply_u":1, "partial_x":1, "partial_t":1, "constant":1}
        self.A_operator_distribution = {"add":1, "commutator":1, "multiply_u":1, "partial_x":1, "partial_t":1, "constant":1}

L = add(Dx(Dx), operators.commutator(Dx, operators.commutator(Dx, M(u))))

A = add(
        new_constant()(operators.commutator(Dx, M(u))(Dt)),
        new_constant()(operators.commutator(Dt, M(u))(Dx)),
        new_constant()(operators.commutator(Dx, M(u))(Dx)),
#        new_constant()(operators.commutator(Dt, M(u))(Dt)),
        new_constant()(operators.commutator(Dx(Dx), M(u))),
        new_constant()(operators.commutator(Dx(Dt), M(u))),
#        new_constant()(operators.commutator(Dt(Dt), M(u))),
#        new_constant()(operators.commutator(Dt, M(u))(Dx))
        )

KdV = LaxPair(L, A, constants, x, t, 120)
