import numpy
from matplotlib import pyplot

class solution:
    def __init__(self, xres, dx, yres, dy):
        self.xres = xres
        self.dx = dx
        self.yres = yres
        self.dy = dy
        
        self.x0 = -xres*dx/2.0
        self.x1 = xres*dx/2.0
        self.y0 = -self.yres*self.dy/2.0
        self.y1 = self.yres*self.dy/2.0
        
        #format self.data[x][y]
        self.x = numpy.linspace(self.x0, self.x1, self.xres, False)
        self.y = numpy.linspace(self.y0, self.y1, self.yres, False)
        self.data = numpy.zeros((xres, yres))
        
        self.U = profile(self.xres, self.dx, self.dy)
        self.U_t = profile(self.xres, self.dx, self.dy)
        
    def solve(self, function_callback, initalization_callback):
        initalization_callback(self)
        #assuming self.U has now been initalized add it to self.data
        self.data[:,0] = self.U.data
        for i in range(self.yres - 1):
#            pyplot.plot(self.x, self.U.data)
#            pyplot.show()
            self.U_t = function_callback(self)
            self.U = self.U_t.integrate(self.U)
            self.data[:, i] = self.U.data
        pyplot.plot(self.x, self.data[:, 0])
        pyplot.show()
        pyplot.imshow(self.data, clim=(-2, 2))
        pyplot.show()

class profile:
    def __init__(self, xres, dx, dy):
        self.xres = xres
        self.dx = dx
        self.dy = dy
        
        self.data = numpy.zeros((xres,))

    def derivative(self):
        output = profile(self.xres, self.dx, self.dy)
        output.data[1:-1] = ((self.data[:-2] - self.data[2:])/(2*self.dx))
        output.data[0] = (self.data[0] - self.data[1])/self.dx
        output.data[-1] = (self.data[-2] - self.data[-1])/self.dx
        return output
    
    def integrate(self, inital_profile):
        output = profile(self.xres, self.dx, self.dy)
        output.data = inital_profile.data + self.data*self.dy
        return output

def init(self):
#    self.U.data = numpy.exp(-self.x**2)
#    self.U_t.data = -2*self.x*numpy.exp(-self.x**2)
    self.U.data = numpy.cos(self.x*3) + numpy.cos(self.x*4)
    self.U_t.data = numpy.sin(self.x*3) + numpy.sin(self.x*4)

def sdiv(x, dx=0.01):
    return numpy.tanh(x/dx)/numpy.sqrt(dx**2 + x**2)

def func(self):
    U_t = profile(self.xres, self.dx, self.dy)
#    U_t.data = self.U.data*self.U.derivative().data # u*u_0 + u_1
#    U_t.data = self.U.derivative().data**2 - self.U.derivative().data # u*u_01 + u_0*u_1
#    U_t.data = self.U.data**3*self.U.derivative().data*self.U_t.data + self.U.derivative().data # u*u_01 + u_0*u_1
    U_tt = profile(self.xres, self.dx, self.dy)
    U_tt.data = -self.U.data*self.U_t.derivative().data + self.U_t.data*self.U.derivative().data
    U_t = U_tt.integrate(self.U_t)
    return U_t

#A = solution(2000, 0.005, 2000, 0.001)
a = 2
b = 1
A = solution(2000*a, 0.005/a, 2000*a, 0.001/(a*b))
A.solve(func, init)

