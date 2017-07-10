from matplotlib import pyplot
import numpy

class solver:
    def __init__(self, x0, x1, xres, y0, y1, yres):
        self.x0 = x0
        self.x1 = x1
        self.xres = xres
        self.y0 = y0
        self.y1 = y1        
        self.yres = yres
        
        #format self.data[x][y]
        self.x = numpy.linspace(self.x0, self.x1, self.xres, True)
        self.dx = (self.x1 - self.x0)/(self.xres - 1)
        self.y = numpy.linspace(self.y0, self.y1, self.yres, True)
        self.dy = (self.y1 - self.y0)/(self.yres - 1)
        self.data = numpy.zeros((self.xres, self.yres))
        
        #Assuming the folowing: self.U.shape == (n, self.xres) where n >= 1
        self.U = None

    def solve(self, initalization_callback, function_callback):
        initalization_callback(self)
        self.data[:, 0] = self.U[0]
        for i in range(1, self.yres):
            U_t = function_callback(self, self.U)
            self.U = self.integrate_y(self.U, U_t)
            self.data[:, i] = self.U[0]
    
    def derivative(self, profile):
        output = -(numpy.roll(profile, 1) - numpy.roll(profile, -1))/(2*self.dx)
        output = -(-numpy.roll(profile, 2) + 8*numpy.roll(profile, 1) - 8*numpy.roll(profile, -1) + numpy.roll(profile, -2))/(12*self.dx)
        return output

    def integrate_y(self, profile, derivative_y):
        output = profile + derivative_y*self.dy
        return output

    def integrate_x(self, inital_value, U, function_callback, *args):
        output = numpy.zeros((len(inital_value), self.xres))
        output[:, 0] = inital_value
        for i in range(1, self.xres):
            output[:, i] = output[:, i - 1] + function_callback(self, U[:, i - 1], output[:, i - 1], *[arg[:, i - 1] for arg in args])*self.dx
        return output

def init(self):
    a = 2*numpy.pi/(self.x1 - self.x0)
    self.U = numpy.zeros((2, self.xres))
    self.V = numpy.zeros((2, self.yres))
#    self.U = numpy.zeros((2, self.xres))
    self.U[0] = numpy.exp(-self.x**2)
    self.V[0] = numpy.exp(-self.y**2)
    self.U[1] = -2*self.x*numpy.exp(-self.x**2)
    self.V[1] = numpy.exp(-self.y**2)
#    self.U[0] = numpy.cos(4*a*self.x) + numpy.cos(7*a*self.x) + 2
#    self.U[1] = 4*a*numpy.sin(4*a*self.x) + 7*a*numpy.sin(7*a*self.x)
#    self.V[0] = 4*a*numpy.sin(4*a*self.y) + 7*a*numpy.sin(7*a*self.y)
#    self.V[1] = numpy.cos(4*a*self.y) + numpy.cos(7*a*self.y)

def func(self, U):
    output = numpy.zeros(shape=U.shape)
    V = self.V[:, 0]
    self.V = self.V[:, 1:]

    U_x = self.derivative(U)
    U_xx = self.derivative(U_x)
    
    def func2(self, U, V, U_x, U_xx):
        output = numpy.zeros(shape=V.shape)
        output[:-1] = V[1:]
        output[-1] = (V[0] - U_xx[0] - U_x[0]*U[1])/(U[0]**2 + 10)
        return output
    
    u_tm = self.integrate_x(V, U, func2, U_x, U_xx)[0, :]
#    pyplot.plot(self.x, u_tm)
#    pyplot.show()
    output[:-1] = U[1:]
    output[-1] = u_tm
    
#    output[0] = U[0]*self.derivative(U[0])
#    output[0] = self.derivative(U[0])**2 - self.derivative(U[0])
    
#    output[0] = U[1]
#    output[1] = -U[0]*self.derivative(U[1]) + U[1]*self.derivative(U[0])

#    output[0] = U[1]
#    output[1] = self.derivative(self.derivative(U[0]))
    return output

A = solver(-3, 3, 2000, -3, 3, 2000)
A.solve(init, func)

pyplot.plot(A.x, A.data[:, 0])
pyplot.show()
pyplot.imshow(A.data)
pyplot.show()
pyplot.imshow(numpy.log(numpy.abs(A.data)), clim=(0, None))
pyplot.show()
