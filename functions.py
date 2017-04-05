#! /usr/bin/env python3

import sympy

class _symbolic_object:
    _tokens = []
    
    def __init__(self, token):
        if len(token) == 0:
            raise ValueError("Empty string is not a valid token.")
    
        if token in self.__class__._tokens:
            raise ValueError("The token \"" + token + "\" is already in use.")

        if not token[0].isalpha():
            raise ValueError("The token \"" + token + "\" must start with a letter.")
            
        if not token.isalnum():
            raise ValueError("The token \"" + token + "\" is not alpha numeric.")

        self.token = token
        self.__class__._tokens.append(self.token)
        self._type = "None"

    def __str__(self):
        return self.token

    def __eq__(self, other):
        if self._type ==  other._type:
            if self.token == other.token:
                return True
        return False    
    
    def _partial(self, variable):
        raise NotImplementedError("Partial derivatives not implemented.")
        
    def _simplify(self):
        return self
        
    def to_sympy(self):
        raise NotImplementedError("Conversion to sympy not supported.")

class literal(_symbolic_object):
    def __init__(self, value):
        self.token = None
        self.value = value
        self._type = "literal"
    
    def __str__(self):
        return self.value
    
    def __eq__(self, other):
        if self._type ==  other._type:
            if self.value == other.value:
                return True
        return False
    
    def _partial(self, variable):
        return literal("0")
    
    def to_sympy(self):
        return sympy.sympify(int(self.value))

class constant(_symbolic_object):
    def __init__(self, token):
        super().__init__(token)
        self._type = "constant"
        self._sympy_symbol = None
    
    def _partial(self, variable):
        return literal("0")

    def to_sympy(self):
        if self._sympy_symbol is None:
            self._sympy_symbol = sympy.Symbol(self.token)
        return self._sympy_symbol

class variable(_symbolic_object):
    def __init__(self, token):
        super().__init__(token)
        self._type = "variable"
        self._sympy_symbol = None

    def _partial(self, variable):
        if variable.token == self.token:
            return literal("1")
        else:
            return literal("0")

    def to_sympy(self):
        if self._sympy_symbol is None:
            self._sympy_symbol = sympy.Symbol(self.token)
        return self._sympy_symbol

class _function(_symbolic_object):
    _functions = {}
    
    def __init__(self, token, arguments, partials=None, partial_callback=None, simplify_callback=None, sympy_callback=None):
        self.token = token.lower()
        self._type = "_function"
        self.arguments = list(arguments)
        self.partials = partials
        self._partial_callback = partial_callback
        self._simplify_callback = simplify_callback
        self._sympy_symbol = None
        self._sympy_callback = sympy_callback

        if self.token not in self.__class__._functions:
            self.__class__._functions[self.token] = self

    def __str__(self):
        return self.token + "(" + ",".join([str(argument) for argument in self.arguments]) + ")"

    def __eq__(self, other):
        return NotImplemented
    
    def _partial(self, variable):
        if self._partial_callback is None:
            if self.partials is None:
                return add(*[multiply(_function(self._partial_token(i), self.arguments), self.arguments[i]._partial(variable)) for i in range(len(self.arguments))])
            else:
                return add(*[multiply(self.partials[i](*self.arguments), self.arguments[i]._partial(variable)) for i in range(len(self.arguments))])
        else:
            return self._partial_callback(self, variable)
    
    def _simplify(self):
        for i in range(len(self.arguments)):
            self.arguments[i] = self.arguments[i]._simplify()

        if self._simplify_callback is None:
            return super()._simplify()
        else:
            return self._simplify_callback(self)

    def to_sympy(self):
        if self._sympy_callback is None:
            if self._sympy_symbol is None:
                self._sympy_symbol = sympy.Function(self.token)
            return self._sympy_symbol(*[argument.to_sympy() for argument in self.arguments])
        else:
            return self._sympy_callback(self)

    def derived_functions(self):
        token = self.token
        if "_" in token:
            token = token.split("_")[0]
        return {key:value for key, value in _function._functions.items() if key.startswith(token)}

    def _partial_token(self, index):
        new_token = None
        if "_" in self.token:
            new_token, partials = self.token.split("_")
            new_token = new_token + "_" + "".join(sorted(partials + str(index)))
        else:
            new_token = self.token + "_" + str(index)
        return new_token

class function(_symbolic_object):
    def __init__(self, token, num_arguments, partials=None):
        self._partials = None
        if num_arguments < 1:
            raise ValueError("The number of arguments must be one or greater")
    
        super().__init__(token.lower())
        self._type = "function"
        self.num_arguments = num_arguments
        self.partials = partials
        
    def __call__(self, *args):
        if len(args) != self.num_arguments:
            raise ValueError("The function \"" + self.token + "\" takes \"" + str(self.num_arguments) + "\" arguments.")
        
        function_instance = _function(self.token, args, self.partials)
        return function_instance
        
    def __str__(self):
        return self.token + "( )"

    def __eq__(self, other):
        return NotImplemented

    @property
    def partials(self):
        return self._partials
        
    @partials.setter
    def partials(self, partials):
        if partials is not None:
            if len(partials) != self.num_arguments:
                raise ValueError("The number of partial derivatives does not match the number of function arguments.")
        self._partials = partials

    def derived_functions(self):
        token = self.token
        if "_" in token:
            token = token.split("_")[0]
        return {key:value for key, value in _function._functions.items() if key.startswith(token)}

    def _simplify(self):
        raise NotImplementedError("Functions without arguments cannot be simplified.")

def add(*args):
    def _simplify(self):
        arguments = []
        for i in range(len(self.arguments)):
            if self.arguments[i] == literal("0"):
                continue
            
            if self.arguments[i]._type == "_function":
                if self.arguments[i].token == "add":
                    arguments = arguments + self.arguments[i].arguments
                    continue
            arguments.append(self.arguments[i])
        self.arguments = arguments
        self.partials = [literal("1") for argument in self.arguments]
        if len(self.arguments) == 0:
            return literal("0")
        elif len(self.arguments) == 1:
            return self.arguments[0]
        return self
        
    partials = [literal("1") for arg in args]
    def _partial(self, variable):
        if self.partials is None:
            raise NotImplementedError("Partial derivatives not implemented")
        else:
            return add(*[multiply(self.partials[i], self.arguments[i]._partial(variable)) for i in range(len(self.arguments))])

    def _sympy(self):
        return sympy.Add(*[argument.to_sympy() for argument in self.arguments])
    
    function_instance = _function("add", args, partials, _partial, _simplify, _sympy)
    return function_instance

def multiply(*args):
    if len(args) == 1:
        return args[0]

    def _simplify(self):
        arguments = []
        for i in range(len(self.arguments)):
            if self.arguments[i] == literal("1"):
                continue
            elif self.arguments[i] == literal("0"):
                return literal("0")

            if self.arguments[i]._type == "_function":
                if self.arguments[i].token == "multiply":
                    arguments = arguments + self.arguments[i].arguments
                    continue
            arguments.append(self.arguments[i])
        self.arguments = arguments
        self.partials = [multiply(*(self.arguments[:i] + self.arguments[i + 1:])) for i in range(len(self.arguments))]    
        if len(self.arguments) == 0:
            return literal("1")
        elif len(self.arguments) == 1:
            return self.arguments[0]
        return self
            
    partials = [multiply(*(args[:i] + args[i + 1:])) for i in range(len(args))]    
    def _partial(self, variable):
        if self.partials is None:
            raise NotImplementedError("Partial derivatives not implemented")
        else:
            return add(*[multiply(self.partials[i], self.arguments[i]._partial(variable)) for i in range(len(self.arguments))])

    def _sympy(self):
        return sympy.Mul(*[argument.to_sympy() for argument in self.arguments])

    function_instance = _function("multiply", args, partials, _partial, _simplify, _sympy)
    return function_instance

#define natural exponentiation
exp = function("exp", 1)
exp.partials = [exp]

#define natural logarithm
ln = function("ln", 1)
ln.partials = [lambda x: power(x, literal("-1"))]

#define exponentiation
def power(x, y):
    return exp(multiply(y, ln(x)))
