'''
Ideas and strucutre taken from Lab 3 and 
https://fenicsproject.org/pub/tutorial/html/._ftut1010.html#mjx-eqn-3.45
'''


from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

set_log_level(LogLevel.ERROR)

# Define parameters
T = 0.1
dt = 0.0005
tau = Constant(0.0003)
kappa = Constant(1.8)
alpha = Constant(0.9)
Teq = Constant(1.0)
epsilon = constant(0.01)
j = Constant(6.0)
delta = Constant(0.02)
mu = Constant(1.0)
gamma = Constant(10.0)
Pi = Constant(3.1415926)
k = Constant(dt)

# Class representing the intial conditions
class InitialConditions (UserExpression):
    def eval (self, values, x):
        values [0] = 1.* ((x[0]-5) ** 2 +(x[1] - 5) ** 2 - 1)
        values [1] = 0.
    def value_shape (self):
        return (2,)

        
        
# Create mesh and define function space
mesh = RectangleMesh(Point(0,0), Point(10, 10), 50, 50)
P1 = FiniteElement ("Lagrange", mesh.ufl_cell(), 1 )
TH = P1*P1
W = FunctionSpace(mesh, TH) #Space for u,v

# Test functions
v = TestFunction(W)
v_1,v_2 = v[0],v[1]


# Define & set initial condition
indata = InitialConditions()
u_n = TrialFunction(W)
u_n = interpolate(indata, W)


# Define variational problem
u = TrialFunction(W)
u_1, u_2 =u[0], u[1]
u_n1, u_n2 =u_n[0], u_n[1]


F = tau * ((u_1 - u_n1) / k ) * v_1 * dx + ((u_2 - u_n2) / k) * v_2 * dx - kappa*((u_1 - u_n1)/k) * u_2 * dx \
 + epsilon ** 2 * dot(v_1, grad(u_1)) * dx + dot(v_2, grad(u_2)) * dx + u_1*(1-u_1)*(u_1 - 0.5 * alpha/Pi * atan(gamma*(Teq-u_2)) * dx



a,L=lhs(F),rhs(F)
u=Function(W)


#Initial population profiles


plt.clf()
c=plot(u_n1)
plt.colorbar(c)
plt.xlabel("x")
plt.ylabel("y")
plt.title("eta")
plt.savefig("u_0.png")
plt.clf()

c=plot(u_n2)
plt.colorbar(c)
plt.xlabel("x")
plt.ylabel("y")
plt.title("v")
plt.savefig("T_0.png")
plt.clf()

# Time - stepping
tlist=[0]
times=[0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
t=0
while t < T:
    t+=dt
    print("t="+str(t))
    tlist.append(t)
    solve(a==L,u)
    u_n.assign(u)
    
    
    if int(t) in times:
        c=plot(u_n[0])
        plt.colorbar(c)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("eta")
        plt.savefig("eta_"+str(t)+".png")
        plt.clf()

        c=plot(u_n[1])
        plt.colorbar(c)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("T")
        plt.savefig("T_"+str(t)+".png")
        plt.clf()
    
        



c=plot(u_n1)
plt.colorbar(c)
plt.xlabel("x")
plt.ylabel("y")
plt.title("eta")
plt.savefig("eta_t0-_1.png")
plt.clf()

c=plot(u_n2)
plt.colorbar(c)
plt.xlabel("x")
plt.ylabel("y")
plt.title("T")
plt.savefig("T_t0_1.png")
plt.clf()

