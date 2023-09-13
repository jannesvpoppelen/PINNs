"""
Implementation of 2D Allen-Cahn equation in DeepXDE. Uses the tensorflow backend.

d/dt u - k * (d^2/dx^2 + d^2/dy^2) u + 1/eps^2 * u * (u^2 - 1) = 0

Boundary conditions are periodic, and u(x,y,t=0) = sin(4 pi x) * cos(4 pi y)
"""
import deepxde as dde
import numpy as np
from tensorflow import sin, cos

# Global parameters
eps = 1
k = 0.0001  # μm^2/s
iters = 4000  # number of iterations for training

Omega = dde.geometry.geometry_2d.Rectangle((0, 0), (1, 1))  # μm
T = dde.geometry.TimeDomain(0, 1)  # s
geometry = dde.geometry.GeometryXTime(Omega, T)


# Specify PDE and BCs
def pde(x, u):  # x = [x, y, t], u = [u]
    u_t = dde.grad.jacobian(u, x, i=0, j=2)
    u_xx = dde.grad.hessian(u, x, i=0, j=0)
    u_yy = dde.grad.hessian(u, x, i=1, j=1)
    return u_t - k * (u_xx + u_yy) + 1 / (eps ** 2) * u * (u ** 2 * - 1)


# Horizontal boundary, which takes all points on the boundary and everything that is close to x=0 and x=1
def h_boundary(x, on_boundary):
    return on_boundary and (np.isclose(x[0], 0) or np.isclose(x[0], 1))


# Vertical boundary, which takes all points on the boundary and everything that is close to y=0 and y=1
def v_boundary(x, on_boundary):
    return on_boundary and (np.isclose(x[1], 0) or np.isclose(x[1], 1))


# Initial condition
def ic(x):
    return sin(4 * np.pi * x[:, 0:1]) * cos(4 * np.pi * x[:, 1:2])


# Hard implementation of initial condition. Can be applied to the network
def f(x, u):
    xx, yy, t = x[:, 0:1], x[:, 1:2], x[:, 2:]
    return t * u + sin(4 * np.pi * xx) * cos(4 * np.pi * yy)


BC1 = dde.PeriodicBC(geometry, 0, h_boundary)  # Periodic boundary condition on x
BC2 = dde.PeriodicBC(geometry, 1, v_boundary)  # Periodic boundary condition on y
IC = dde.IC(geometry, ic, lambda _, on_initial: on_initial)  # Initial condition

data = dde.data.TimePDE(geometry, pde, [BC1, BC2, IC], num_domain=8000, num_boundary=100)  # Sample data points

# Define network architecture
layers = [2] + [32] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
network = dde.nn.tensorflow.fnn.FNN(layers, activation, initializer)
# network.apply_output_transform(f) # Necessary if using hard implementation of initial condition

# Define model and optimizer
model = dde.Model(data, network)
model.compile("adam", lr=0.001, loss_weights=[10, 1, 1, 1])
model.train(iterations=iters, display_every=100)
model.compile("L-BFGS-B") # Improve loss using L-BFGS-B
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=False, isplot=True)
