from pinnfuncs import *

# hpyer-parameters
t0 = 0.0
tf = 2.0
xmin, xmax = 0., 5.
ymin, ymax = 0., 5.
geometry = [t0, tf, xmin, xmax, ymin, ymax]
n_t = 200
n_x = 128
n_y = 128
n_ic, n_bc = 32, 32
tol = 1.0

xic = jnp.linspace(xmin, xmax, n_ic)
yic = jnp.linspace(ymin, ymax, n_ic)

# Initial condition
eta0 = jnp.array([[eta0(x, y) for x in xic] for y in yic])
mu0 = jnp.array([[mu0(x, y) for x in xic] for y in yic])
phi0 = jnp.array([[phi0(x, y) for x in xic] for y in yic])
u0 = jnp.array([eta0, mu0, phi0])


# Network architecture
d0 = 3#4 * M + 2
layers = [d0, 128, 128, 128, 128, 128, 3]

u = PINN(layers, geometry, u0, n_t, n_x, n_y, n_ic, n_bc, tol)
u.train(1)
#print(u.loss(u.get_params(u.opt_state)))
#print(u.loss_bc(u.get_params(u.opt_state)))
#print(u.loss_ic(u.get_params(u.opt_state)))
#u.train(20000)
#plot_weights(u)
#plot_losses(u)
#plot_dynamics(u, t0, tf, 300)
