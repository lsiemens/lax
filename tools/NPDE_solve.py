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
        for i in range(1, self.yres - 1):
            U_t = function_callback(self, self.U)
            self.U = self.integrate_y(self.U, U_t)
            self.data[:, i] = self.U[0]
    
    def derivative(self, profile):
        output = (numpy.roll(profile, 1) - numpy.roll(profile, -1))/(2*self.dx)
#        output = (-numpy.roll(profile, 2) + 8*numpy.roll(profile, 1) - 8*numpy.roll(profile, -1) + numpy.roll(profile, -2))/(12*self.dx)
        return output

    def integrate_y(self, profile, derivative_y):
        output = profile + derivative_y*self.dy
        return output

    def integrate_x(self, inital_value, function_callback):
        output = numpy.zeros(shape=derivative_x.shape)
        output[:, 0] = inital_value
        for i in range(1, self.xres - 1):
            output[:, i] = output[:, i - 1] + function_callback(self.U[:, i - 1], output[:, i - 1])*self.dx
        return output

def init(self):
#    a = 2*numpy.pi/(self.x1 - self.x0)
    self.U = numpy.zeros((1, self.xres))
#    self.U = numpy.zeros((2, self.xres))
    self.U[0] = numpy.exp(-self.x**2)
#    self.U[1] = -2*self.x*numpy.exp(-self.x**2)
#    self.U[0] = numpy.cos(4*a*self.x) + numpy.cos(7*a*self.x)       
#    self.U[1] = 4*a*numpy.sin(4*a*self.x) + 7*a*numpy.sin(7*a*self.x)
    self.guess = numpy.zeros((1, self.xres))
    self.guess[0] = -2*self.x*numpy.exp(-self.x**2)

def func(self, U):
    output = numpy.zeros(shape=U.shape)

    output[0] = U[0]*self.derivative(self.guess[0])
    self.guess[0] = output[0]
#    pyplot.plot(self.x, output[0])
#    pyplot.show()

    output[0] = U[0]*self.derivative(self.guess[0])
    self.guess[0] = output[0]
#    pyplot.plot(self.x, output[0])
#    pyplot.show()

    output[0] = U[0]*self.derivative(self.guess[0])
    self.guess[0] = output[0]
#    pyplot.plot(self.x, output[0])
#    pyplot.show()

#    output[0] = U[0]*self.derivative(U[0])
#    output[0] = self.derivative(U[0])**2 - self.derivative(U[0])
    
#    output[0] = U[1]
#    output[1] = -U[0]*self.derivative(U[1]) + U[1]*self.derivative(U[0])

#    output[0] = U[1]
#    output[1] = self.derivative(self.derivative(U[0]))

    
    return output

A = solver(-3, 3, 200, 0, 2, 200000)
#A.solve(init, func)

#pyplot.plot(A.x, A.data[:, 0])
#pyplot.show()
#pyplot.imshow(A.data[:,::1000], clim=(-0, 1.5))
#pyplot.show()
