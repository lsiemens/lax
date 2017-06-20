#! /usr/bin/env python3

import lax.functions

class operator(lax.functions._symbolic_object):
    def __init__(self, token, init_callback, eval_callback, str_callback, copy_callback, *init_arguments):
        self.token = token.upper()
        self._type = "operator"
        self._init_callback = init_callback
        self._eval_callback = eval_callback
        self._str_callback = str_callback
        self._copy_callback = copy_callback
        self._init_arguments = init_arguments

        self.argument = None
        self._init_callback(self, *self._init_arguments)
    
    def __call__(self, argument):
        if argument._type == "operator":
            new_self = self.copy()
            if self.argument is None:
                new_self.argument = argument
                return new_self
            else:
                new_self.argument = self.argument(argument)
                return new_self
        else:
            if self.argument is None:
                return self._eval_callback(self, argument)
            else:
                return self._eval_callback(self, self.argument(argument))
    
    def __str__(self):
        return self.token + "(" + self._str_callback(self) + ")"
    
    def __eq__(self, other):
        return NotImplemented
    
    def copy(self):
        new_copy = operator(self.token, self._init_callback, self._eval_callback, self._str_callback, self._copy_callback, *self._init_arguments)
        return self._copy_callback(self, new_copy)

def add(*vectors):
    def _init(self, *vectors):
        if any([vector._type != "operator" for vector in vectors]):
            raise ValueError("*vectors must all have _type operator.")
        if len(vectors) == 0:
            raise ValueError("*vectors must not be empty list.")
        self._vectors = vectors
    def _eval(self, argument):
        return lax.functions.add(*[vector(argument) for vector in self._vectors])
    def _str(self):
        ###########??????????????
        if self.argument is None:
            return ", ".join([str(vector) for vector in self._vectors])
        else:
            return ", ".join([str(vector(self.argument)) for vector in self._vectors])
    def _copy(self, new_self):
        new_self._vectors = self._vectors
        if self.argument is not None:
            new_self.argument = self.argument.copy()
        return new_self
    return operator("ADD", _init, _eval, _str, _copy, *vectors)

def multiply(scaler):
    def _init(self, scaler):
        if scaler._type == "operator":
            raise ValueError("scaler cannot be an operator.")
        self._scaler = scaler
    def _eval(self, argument):
        return lax.functions.multiply(self._scaler, argument)
    def _str(self):
        if self.argument is None:
            return str(self._scaler) + ", "
        else:
            return str(self._scaler) + ", " + str(self.argument)
    def _copy(self, new_self):
        new_self._scaler = self._scaler
        if self.argument is not None:
            new_self.argument = self.argument.copy()
        return new_self
    return operator("MULTIPLY", _init, _eval, _str, _copy, scaler)

def partial(variable):
    def _init(self, variable):
        if variable._type != "variable":
            raise ValueError("variable must be _type variable.")
        self._variable = variable
    def _eval(self, argument):
        return argument._partial(self._variable)._simplify()
    def _str(self):
        if self.argument is None:
            return str(self._variable) + ", "
        else:
            return str(self._variable) + ", " + str(self.argument)
    def _copy(self, new_self):
        new_self._variable = self._variable
        if self.argument is not None:
            new_self.argument = self.argument.copy()
        return new_self
    return operator("PARTIAL", _init, _eval, _str, _copy, variable)

def commutator(F, G):
    return add(F(G), multiply(lax.functions.literal("-1"))(G(F)))

#x = lax.functions.variable("x")
#t = lax.functions.variable("t")
#u = lax.functions.function("u", 2)(x, t)
#f = lax.functions.function("f", 2)(x, t)

#L = add(multiply(lax.functions.literal("-1"))(partial(x)(partial(x))), multiply(u))
#A = add(multiply(lax.functions.literal("-4"))(partial(x)(partial(x)(partial(x)))),multiply(lax.functions.literal("3"))(add(multiply(u)(partial(x)), partial(x)(multiply(u)))))

#LAX = add(partial(t)(L), commutator(L, A))

#print(LAX(f)._simplify())
