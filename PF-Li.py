import equinox as eqx
import jax.random as jr
import numpy as np
import optax
from pdefuncs import *
from pinnfuncs import *

key = jr.PRNGKey(0)
stddev = 2
t_stddev = 1
mapping_size = 10
eps = 1

xmin = 0.  # left boundary in micrometer
xmax = 200.  # right boundary in micrometer
ymin = 0.  # bottom boundary in micrometer
ymax = 100.  # top boundary in micrometer
t0 = 0.  # initial time in s
tf = 100.  # final time in s


xr = 200  # number of spatial collocation points
yr = 100  # number of spatial collocation points
tr = 100  # number of temporal collocation points
Nb = 500  # number of boundary points
Nt = 1000  # number of initial condition points
resWeight = 1
bcWeight = 25  # weight for boundary loss
icWeight = 100  # weight for initial condition loss

xc = jnp.linspace(xmin, xmax, xr)  # collocation points
yc = jnp.linspace(ymin, ymax, yr)  # collocation points
tc = jnp.linspace(t0, tf, tr)  # collocation points
xic = jr.uniform(key, minval=xmin, maxval=xmax, shape=(Nt,))  # initial condition points
yic = jr.uniform(key, minval=ymin, maxval=ymax, shape=(Nt,))  # initial condition points
xbc = jr.uniform(key, minval=xmin, maxval=xmax, shape=(Nb,))  # boundary points
ybc = jr.uniform(key, minval=ymin, maxval=ymax, shape=(Nb,))  # boundary points
tbc = jnp.linspace(t0, tf, Nb)  # boundary points


B = stddev * jr.normal(key, shape=(mapping_size,))  # Fourier features
Btemp = t_stddev * jr.normal(key, shape=(mapping_size,))  # Temporal Fourier features
B = [B, Btemp]
M = jnp.triu(jnp.ones((tr, tr)), k=1).T  # sums the weights of the residual


class PINN(eqx.Module):
    depth: int
    width: int
    layers: list
    bias: jnp.ndarray
    FF: jnp.ndarray

    def __init__(self, key, depth, width, FF):
        # super().__init__()
        key, _ = jr.split(key)
        self.depth = depth
        self.width = width
        self.bias = jr.normal(key, shape=(3,))
        self.FF = FF
        mapsize = 6 * len(self.FF[0].flatten()) + 1
        key, _ = jr.split(key)
        self.layers = [eqx.nn.Linear(mapsize, self.width, key=key)]

        for i in range(self.depth - 1):
            key, _ = jr.split(key)
            self.layers.append(eqx.nn.Linear(self.width, self.width, key=key))

        key, _ = jr.split(key)
        self.layers.append(eqx.nn.Linear(self.width, 3, key=key))

    @eqx.filter_jit
    def __call__(self, x):
        x = input_mapping(x, self.FF[0], self.FF[1])  # map into Fourier features
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        return self.layers[-1](x) + self.bias


u = PINN(key, 3, 128, B)