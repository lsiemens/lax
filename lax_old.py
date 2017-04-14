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

import time
import timeout
import functions
import operators

def simplify(f):
    return sympy.simplify(sympy.expand(f))

def operate(A, psi):
    print(sympy.collect(simplify(A(psi).to_sympy()), psi.to_sympy()))

while True:
    functions.reset()
    x = functions.variable("x")
    t = functions.variable("t")
    constants = []
    u = functions.function("u", 2)(x, t)
    f = functions.function("f", 2)(x, t)
    g = functions.function("g", 2)(x, t)

    Dx = operators.partial(x)
    Dt = operators.partial(t)
    M = operators.multiply
    
    def new_constant():
        const = functions.constant("c" + str(len(constants) + 1))
        constants.append(const)
        return operators.multiply(const)
    
    valid_operators = [(operators.add, 1, 3), (operators.partial(x),0, 1), (operators.partial(t), 0, 1),
                 (operators.multiply(u), 0, 1), (operators.multiply(functions.literal("-1")), 0, 1),
                 (new_constant, 0, 1)]
    operator_weights = [30,100,100,60,10,100]
    
    def generate_linear_operator(operators, weights):
        index = numpy.random.choice(len(operators), p=weights/numpy.sum(weights))
        choice = operators[index]
        if choice[0] == new_constant:
            choice = (new_constant(), choice[1], choice[2])
        num_operands = numpy.random.randint(choice[1], choice[2] + 1)
        if num_operands == 0:
            return choice[0]
        elif num_operands == 1:
            argument = generate_linear_operator(operators, weights)
            return choice[0](argument)
        else:
            arguments = [generate_linear_operator(operators, weights) for _ in range(num_operands)]
            return choice[0](*arguments)
    
    try:
        with timeout.timeout(60):
            L = generate_linear_operator(valid_operators, operator_weights)
            A = generate_linear_operator(valid_operators, operator_weights)
            #L = operators.add(M(functions.literal("-1"))(Dx(Dx)), M(u))
            #A = operators.add(new_constant()(Dx(Dx(Dx))), new_constant()(M(u)(Dx)), new_constant()(Dx(M(u))))
            
            print("L(f)")
            print(simplify(L(f).to_sympy()))
            print("\nA(f)")
            print(simplify(A(f).to_sympy()))
            print("\nL_t(f)")
            print(simplify(operators.commutator(Dt, L)(f).to_sympy()))
#            print(simplify(operators.commutator(Dt(Dt), L)(f).to_sympy()))
            print("\n\n")
            
            LAX = operators.add(operators.commutator(Dt, L), operators.commutator(L, A))
#            # second order lax equation
#            LAX = operators.add(operators.commutator(Dt(Dt), L), operators.commutator(L, A))
            
            LAXf = simplify(LAX(f).to_sympy())
            
            print("LAXf")
            print(LAXf)
            
            print(f.derived_functions().keys())
            
            PDE = [simplify(LAXf.coeff(value.to_sympy())) for key, value in f.derived_functions().items() if "_" not in key]
            conditions = [condition for condition in [LAXf.coeff(value.to_sympy()) for key, value in f.derived_functions().items() if "_" in key] if condition != 0]
            constants = [const.to_sympy() for const in constants]
            variables = [variable.to_sympy() for variable in [x, t]]
            if len(PDE) > 0:
                PDE = PDE[0]
            else:
                PDE = 0
            
            if (PDE == 0):
        #        raise ValueError("No f(x) terms in LAXf equation.")
                print("No PDE in LAX")
                continue
                
            print("\nConditions")
            print(conditions)
            solutions = sympy.solve(conditions, constants, exclude=[x.to_sympy(), t.to_sympy()])
            print("sol: " + str(solutions))
            if isinstance(solutions, dict):
                solutions = [solutions]
            print("\nSolutions")
            print(solutions)
            
            if (all([simplify(condition.subs(solution)) == 0 for condition in conditions for solution in solutions])) and (len(solutions) > 0):
                print("Passed sanity check: found " + str(len(solutions)) + " solution(s) to LAX(f(x, t))=E*f(x, t) where E is a PDE.")
                print(PDE)
            else:
        #        raise ValueError("Sympy failed to find any solutions.")
                print("No solution")
                continue
            
            for solution in solutions:
                print("Using solution: " + str(solution))
                PDE = simplify(PDE.subs(solution))
                print(PDE)
                
                numerator, denominator = PDE.as_numer_denom()
                if len(denominator.free_symbols & set(variables)) != 0:
                    # the denominator is a function of x and/or t
                    print("Assuming the denominator:" + str(denominator) + " is well behaved.\n")
                print(simplify(numerator))
                print("")
            break
    except TimeoutError:
        print("NPDE generation timed out.")
        continue            
    except NotImplementedError:
        #I think this should be fixed by substituting variables
        continue
