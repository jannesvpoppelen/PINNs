import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import time

jax.config.update('jax_platform_name', 'cpu')

"""
Solves the Allen-Cahn equation using PINNs

u_t = k * u_xx + 1/sigma^2 * u * (1 - u^2)
u(-1, t) = u(1, t) = -1
u(x, 0) = x^2 * cos(pi * x)


The accuracy of the PINN will be improved using the following:
    - Sequential training
    - Residual based adaptive resampling
    - Gradient enhanced loss function
    - Embedded Fourier Features
"""


# Plotting functions

def plot_1d(net, t_start, t_end, N):
    xarr = jnp.linspace(xmin, xmax, 250)
    for (i, t) in enumerate(jnp.linspace(t_start, t_end, N)):
        u_pred = []
        for x in xarr:
            val = net(jnp.array([x, t]))
            u_pred.append(val)
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title(f"t = {t:.2f}")
        plt.plot(xarr, u_pred)
        plt.savefig(f"u{i}.png")
        plt.clf()


def plot_losses(loss_history, pde_loss, bc_loss, ic_loss):
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss history")
    plt.yscale("log")
    plt.plot(loss_history, label="Total loss")
    plt.plot(pde_loss, label="PDE loss")
    plt.plot(bc_loss, label="BC loss")
    plt.plot(ic_loss, label="IC loss")
    plt.legend()
    plt.savefig("loss.png")


# Define global parameters
lr = 0.0001  # learning rate
k = 0.0001  # diffusion coefficient
sigma = 1 / jnp.sqrt(5)  # epsilon
eps = .5
mapping_size = 10  # amount of fourier features
stddev = 1

xmin = -1.  # left boundary
xmax = 1.  # right boundary
t0 = 0.  # initial time
tf = 1.  # final time

key = jr.PRNGKey(18698)  # Random number generator key
key, _ = jr.split(key)

xr = 100  # number of spatial collocation points
tr = 50  # number of temporal collocation points
Nb = 30  # number of boundary points
Nt = 400  # number of initial condition points
# Nr = 200  # number of residual points
resWeight = 1
bcWeight = 25  # weight for boundary loss
icWeight = 200  # weight for initial condition loss

loss_history = []
pde_history = []
bc_history = []
ic_history = []


x1 = jnp.linspace(xmin, xmax, xr)
t1 = jnp.linspace(t0, tf, tr)

xic = jr.uniform(key, minval=xmin, maxval=xmax, shape=(Nt,))
tic = jnp.zeros(Nt, )
xbc = jnp.ones(Nb, )
tbc = jnp.linspace(t0, tf, Nb)  # jr.uniform(key, minval=t0, maxval=tf, shape=(Nb,))


def input_mapping(arg, B):
    x, t = arg
    x_proj = (2. * jnp.pi * x) * B.T
    return jnp.concatenate([jnp.array([1]), jnp.sin(x_proj), jnp.cos(x_proj), jnp.array([t])], axis=-1)


B = stddev * jr.normal(key, shape=(mapping_size,))
M = jnp.triu(jnp.ones((tr, tr)), k=1).T  # sums the weights of the residual


class PINN(eqx.Module):
    depth: int
    width: int
    layers: list
    bias: jnp.ndarray
    FF: jnp.ndarray

    def __init__(self, key, depth, width, FF):
        super().__init__()
        key, _ = jr.split(key)
        self.depth = depth
        self.width = width
        self.bias = jr.normal(key, shape=(1,))
        self.FF = FF
        mapsize = 2 * len(self.FF.flatten()) + 1 + 1
        key, _ = jr.split(key)
        self.layers = [eqx.nn.Linear(mapsize, self.width, key=key)]

        for i in range(self.depth - 1):
            key, _ = jr.split(key)
            self.layers.append(eqx.nn.Linear(self.width, self.width, key=key))

        key, _ = jr.split(key)
        self.layers.append(eqx.nn.Linear(self.width, 1, key=key))

    @eqx.filter_jit
    def __call__(self, x):
        x = input_mapping(x, self.FF)  # map into Fourier features
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        return (self.layers[-1](x) + self.bias)[0]


@eqx.filter_jit
def residual(net, x, t):
    arg = jnp.array([x, t])
    u_t = jax.jacrev(net)(arg)[1]
    u_xx = jax.jacrev(jax.jacrev(net))(arg)[0][0]
    u = net(arg)
    return u_t - k * u_xx + (1 / sigma ** 2) * (u**3 - u)


@eqx.filter_jit
def res_weights(net, x, t):
    res = jax.vmap(jax.vmap(residual, in_axes=(None, 0, None)), in_axes=(None, None, 0))
    Lt = jnp.mean(jnp.square(res(net, x, t)), axis=1)
    W = jax.lax.stop_gradient(jnp.exp(- eps * (M @ Lt)))
    return Lt, W


@eqx.filter_jit
def ic_res(net, x, t):
    arg = jnp.array([x, t])
    return net(arg) - x ** 2 * jnp.cos(jnp.pi * x)


@eqx.filter_jit
def ic_loss(net, x, t):
    f = jax.vmap(ic_res, in_axes=(None, 0, 0))
    return jnp.mean(jnp.square(f(net, x, t)))


@eqx.filter_jit
def bc_res(net, x, t):
    ul = net(jnp.array([-1. * x, t]))
    ur = net(jnp.array([x, t]))
    ul_x = jax.jacfwd(net)(jnp.array([-1. * x, t]))[0]
    ur_x = jax.jacfwd(net)(jnp.array([x, t]))[0]
    return (ul - ur) ** 2 + (ur_x - ul_x) ** 2


@eqx.filter_jit
def bc_loss(net, x, t):
    f = jax.vmap(bc_res, in_axes=(None, 0, 0))
    return jnp.mean(jnp.square(f(net, x, t)))


@eqx.filter_jit
def loss(net, xcoll, tcoll, xic, tic, xbc, tbc):
    loss_ic = ic_loss(net, xic, tic)
    loss_bc = bc_loss(net, xbc, tbc)
    Lt, W = res_weights(net, xcoll, tcoll)
    loss = jnp.mean(W * Lt) + icWeight * loss_ic + bcWeight * loss_bc
    return loss


@eqx.filter_jit
def step(net, state):
    loss_t, grad = eqx.filter_value_and_grad(loss)(net, x1, t1, xic, tic, xbc, tbc)
    updates, new_state = optimizer.update(grad, state, net)
    new_net = eqx.apply_updates(net, updates)
    return new_net, new_state, loss_t


u = PINN(key, 4, 128, B)
optimizer = optax.adam(lr)
opt_state = optimizer.init(eqx.filter(u, eqx.is_array))

iters = 10000

start = time.time()
for _ in range(iters + 1):
    l, w = res_weights(u, x1, t1)
    pde_history.append(jnp.mean(w * l))
    bc_history.append(bc_loss(u, xbc, tbc))
    ic_history.append(ic_loss(u, xic, tic))
    u, opt_state, netloss = step(u, opt_state)
    loss_history.append(netloss)
    if _ % 100 == 0:
        print(f"Iteration: {_} | Total Weighted Loss: {netloss:.6f} | Residual Loss: {resWeight * pde_history[-1]:.6f} \
| BC Loss: {bcWeight * bc_history[-1]:.6f} | IC Loss: {icWeight * ic_history[-1]:.6f}")

end = time.time()
print(f"Training time: {(end - start):.2f} seconds")

plot_1d(u, t0, tf, 11)
plot_losses(loss_history, pde_history, bc_history, ic_history)
