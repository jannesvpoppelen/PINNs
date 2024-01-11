import jax.numpy as jnp
from jax import grad, vmap, jit
import jax.random as jr
from jax.example_libraries import optimizers
from jax import lax
from jax.flatten_util import ravel_pytree
import itertools
from functools import partial
from tqdm import trange
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rc('font', size=6)  # controls default text sizes
plt.rc('axes', titlesize=6)  # fontsize of the axes title
plt.rc('axes', labelsize=6)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=4)  # fontsize of the tick labels
plt.rc('ytick', labelsize=4)  # fontsize of the tick labels
plt.rc('legend', fontsize=4)  # legend fontsize

# Constants
tau = 0.0003
kappa = 1.8
alpha = 0.9
Teq = 1.
epsilon = 0.01
delta = 0.02
j = 6.
theta0 = 0.2
mu = 1.
gamma = 10.

t0 = 0.0
tf = 0.5
Lx = 10
Ly = 10


def m(T):
    return alpha / jnp.pi * jnp.arctan(gamma * (Teq - T))


def theta(phi_x, phi_y):
    return jnp.arctan(phi_y / (phi_x + 1e-5))


def sigma(theta):
    return 1 + delta * jnp.cos(j * (theta - theta0))


def depsilon(theta):
    return epsilon * (-delta * j * jnp.sin(j * (theta - theta0)))


def phi0(x, y):
    return 1 - ((x - Lx / 2) ** 2 + (y - Ly / 2) ** 2 < 1.)  # 0.5 * (1 - jnp.tanh(2 * (x - 2)))
    # ((x-Lx/2) ** 2 + (y-Ly/2) ** 2 < .5) * 1.


def T0(x, y):
    return 0.  # 1. * (1 - jnp.tanh(2 * (x - 2)))


def plot_losses(net):
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss history")
    plt.yscale("log")
    plt.plot(net.loss_log, label="Total loss")
    plt.plot(net.loss_phi0_log, label="ϕ IC loss")
    plt.plot(net.loss_T0_log, label="T IC loss")
    plt.plot(net.loss_res_phi_log, label="ϕ Residual loss")
    plt.plot(net.loss_res_T_log, label="T Residual loss")
    plt.legend(loc="upper right")
    plt.savefig("loss.png")
    plt.clf()


def plot_vars(net, t):
    # Evaluate the network over the grid
    params = net.get_params(net.opt_state)
    pred = net.vis_net(params, t)
    phi = pred[0]
    T = pred[1]

    # Create a figure with 3 subplots arranged vertically and more vertical spacing
    fig, axs = plt.subplots(2, 1, figsize=(3, 4), sharex=False, gridspec_kw={'hspace': 0.8})
    plt.setp(axs, xticks=[0, Lx], yticks=[0, Ly])

    # Plot eta
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_title(f" ϕ(t = {t:.4f})")
    im = axs[0].imshow(phi, interpolation='nearest', cmap='jet', extent=[0, Lx, 0, Ly],
                       vmin=0, vmax=1)
    cax = fig.add_axes([axs[0].get_position().x1 + 0.025, axs[0].get_position().y0, 0.02, axs[0].get_position().height])
    plt.colorbar(im, cax=cax)

    # Plot mu
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].set_title(f"T(t = {t:.4f})")
    im = axs[1].imshow(T, interpolation='nearest', cmap='jet', extent=[0, Lx, 0, Ly],
                       vmin=0, vmax=2 * Teq)
    cax = fig.add_axes([axs[1].get_position().x1 + 0.025, axs[1].get_position().y0, 0.02, axs[1].get_position().height])
    plt.colorbar(im, cax=cax)

    # Save the figure
    plt.savefig(f"vars_{t}.png")
    plt.clf()
    return None


def plot_phit(net, t):
    params = net.get_params(net.opt_state)
    phi_t = net.phi_t_plot(params, t)
    im = plt.imshow(phi_t, interpolation='nearest', cmap='jet', extent=[0, Lx, 0, Ly])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"ϕ_t(t = {t:.2f})")
    plt.colorbar(im)
    plt.savefig(f"phi_t_{t}.png")
    plt.clf()


def plot_driving(net, t):
    params = net.get_params(net.opt_state)
    T = net.T0_pred_fn(params, t, xic, yic)
    drive = m(T)
    im = plt.imshow(drive, interpolation='nearest', cmap='jet', extent=[0, Lx, 0, Ly])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"m(T) (t = {t:.4f})")
    plt.colorbar(im)
    plt.savefig(f"driving_{t}.png")
    plt.clf()


def plot_residual(net, t):
    params = net.get_params(net.opt_state)
    res_phi, res_T = net.res_plot_fn(params, t, xic, yic)
    # Create a figure with 3 subplots arranged vertically and more vertical spacing
    fig, axs = plt.subplots(2, 1, figsize=(3, 4), sharex=False, gridspec_kw={'hspace': 0.8})
    plt.setp(axs, xticks=[0, Lx], yticks=[0, Ly])

    # Plot eta
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_title(f" ϕ_res(t = {t:.4f})")
    im = axs[0].imshow(res_phi, interpolation='nearest', cmap='jet', extent=[0, Lx, 0, Ly],
                       vmin=0, vmax=1)
    cax = fig.add_axes([axs[0].get_position().x1 + 0.025, axs[0].get_position().y0, 0.02, axs[0].get_position().height])
    plt.colorbar(im, cax=cax)

    # Plot mu
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].set_title(f"T_res(t = {t:.4f})")
    im = axs[1].imshow(res_T, interpolation='nearest', cmap='jet', extent=[0, Lx, 0, Ly],
                       vmin=0, vmax=2 * Teq)
    cax = fig.add_axes([axs[1].get_position().x1 + 0.025, axs[1].get_position().y0, 0.02, axs[1].get_position().height])
    plt.colorbar(im, cax=cax)

    # Save the figure
    plt.savefig(f"res_{t}.png")
    plt.clf()
    return None


# Define the neural net
def init_layer(key, d_in, d_out):
    k1, k2 = jr.split(key)
    glorot_stddev = 1. / jnp.sqrt((d_in + d_out) / 2.)
    W = glorot_stddev * jr.normal(k1, (d_in, d_out))
    b = jnp.zeros(d_out)
    return W, b


M = 128
B = jr.uniform(jr.PRNGKey(1234), shape=(M,), minval=0.0, maxval=10.0)


# Define the neural net
def MLP(layers, L_x=1.0, L_y=1.0, M_t=1, M_x=1, M_y=1, activation=jnp.tanh):
    def xavier_init(key, d_in, d_out):
        glorot_stddev = 1. / jnp.sqrt((d_in + d_out) / 2.)
        W = glorot_stddev * jr.normal(key, (d_in, d_out))
        b = jnp.zeros(d_out)
        return W, b

    w_x = 2.0 * jnp.pi / L_x
    w_y = 2.0 * jnp.pi / L_y
    k_x = jnp.arange(1, M_x + 1)
    k_y = jnp.arange(1, M_y + 1)
    k_xx, k_yy = jnp.meshgrid(k_x, k_y)
    k_xx = k_xx.flatten()
    k_yy = k_yy.flatten()

    # Define input encoding function
    def input_encoding(t, x, y):
        k_t = jnp.power(10.0, jnp.arange(0, M_t + 1))
        out = jnp.hstack([1, k_t * t,
                          jnp.cos(k_x * w_x * x), jnp.cos(k_y * w_y * y),
                          jnp.sin(k_x * w_x * x), jnp.sin(k_y * w_y * y),
                          jnp.cos(k_xx * w_x * x) * jnp.cos(k_yy * w_y * y),
                          jnp.cos(k_xx * w_x * x) * jnp.sin(k_yy * w_y * y),
                          jnp.sin(k_xx * w_x * x) * jnp.cos(k_yy * w_y * y),
                          jnp.sin(k_xx * w_x * x) * jnp.sin(k_yy * w_y * y)])
        return out

    def init(rng_key):
        U1, b1 = xavier_init(jr.PRNGKey(12345), layers[0], layers[1])
        U2, b2 = xavier_init(jr.PRNGKey(54321), layers[0], layers[1])

        def init_layer(key, d_in, d_out):
            k1, k2 = jr.split(key)
            W, b = xavier_init(k1, d_in, d_out)
            return W, b

        key, *keys = jr.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return (params, U1, b1, U2, b2)

    def apply(params, inputs):
        params, U1, b1, U2, b2 = params
        t = inputs[0]
        x = inputs[1]  # /Lx
        y = inputs[2]  # /Ly
        inputs = input_encoding(t, x, y)
        U = activation(jnp.dot(inputs, U1) + b1)
        V = activation(jnp.dot(inputs, U2) + b2)
        for W, b in params[:-1]:
            outputs = activation(jnp.dot(inputs, W) + b)
            inputs = jnp.multiply(outputs, U) + jnp.multiply(1 - outputs, V)
        W, b = params[-1]
        outputs = jnp.dot(inputs, W) + b
        z = jnp.array([outputs[0], outputs[1]])
        return z

    return init, apply


class DataGenerator():
    def __init__(self, t0, t1, n_t=10, n_x=64, rng_key=jr.PRNGKey(1234)):
        self.t0 = t0
        self.t1 = t1 + 0.01 * t1
        self.n_t = n_t
        self.n_x = n_x
        self.key = rng_key

    def __getitem__(self, index):
        self.key, subkey = jr.split(self.key)
        batch = self.__data_generation(subkey)
        return batch

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        subkeys = jr.split(key, 2)
        t_r = jr.uniform(subkeys[0], shape=(self.n_t,), minval=self.t0, maxval=self.t1).sort()
        x_r = jr.uniform(subkeys[1], shape=(self.n_x, 2), minval=0.0, maxval=Lx)
        batch = (t_r, x_r)
        return batch


# Define the model
class PINN:
    def __init__(self, key, layers, M_t, M_x, M_y, state0, t0, t1, n_t, xic, yic, tol):

        self.M_t = M_t
        self.M_x = M_x
        self.M_y = M_y

        # grid
        self.n_t = n_t
        self.t0 = t0
        self.t1 = t1
        eps = 0.01 * t1
        self.t = jnp.linspace(self.t0, self.t1 + eps, n_t)
        self.xic = xic
        self.yic = yic

        # initial state
        self.state0 = state0

        self.tol = tol
        self.M = jnp.triu(jnp.ones((n_t, n_t)), k=1).T

        self.init, self.apply = MLP(layers, L_x=Lx, L_y=Ly, M_t=M_t, M_x=M_x, M_y=M_y,
                                    activation=jnp.tanh)
        params = self.init(rng_key=key)

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(optimizers.exponential_decay(1e-3,
                                                                                                       decay_steps=50000,
                                                                                                       decay_rate=0.9))
        self.opt_state = self.opt_init(params)
        _, self.unravel = ravel_pytree(params)

        self.phi0_pred_fn = vmap(vmap(self.phi_net, (None, None, 0, None)), (None, None, None, 0))
        self.T0_pred_fn = vmap(vmap(self.T_net, (None, None, 0, None)), (None, None, None, 0))
        self.phi_pred_fn = vmap(vmap(vmap(self.phi_net, (None, None, None, 0)), (None, None, 0, None)),
                                (None, 0, None, None))
        self.T_pred_fn = vmap(vmap(vmap(self.T_net, (None, None, None, 0)), (None, None, 0, None)),
                              (None, 0, None, None))
        self.phi_bc_fn = vmap(vmap(self.phi_net, (None, None, 0, 0)), (None, 0, None, None))
        self.T_bc_fn = vmap(vmap(self.T_net, (None, None, 0, 0)), (None, 0, None, None))
        self.r_pred_fn = vmap(vmap(self.residual_net, (None, None, 0, 0)), (None, 0, None, None))
        self.plot_fn = vmap(vmap(self.neural_net, (None, None, 0, None)), (None, None, None, 0))
        self.phi_t_fn = vmap(vmap(self.phi_t_net, (None, None, 0, None)), (None, None, None, 0))
        self.res_plot_fn = vmap(vmap(self.residual_net, (None, None, 0, None)), (None, None, None, 0))

        # Logger
        self.itercount = itertools.count()

        self.loss_log = []
        self.loss_ics_log = []
        self.loss_phi0_log = []
        self.loss_T0_log = []
        self.loss_bcs_log = []
        self.loss_res_phi_log = []
        self.loss_res_T_log = []

    def neural_net(self, params, t, x, y):
        z = jnp.stack([t, x, y])
        outputs = self.apply(params, z)
        phi = outputs[0]
        T = outputs[1]
        return phi, T

    def phi_net(self, params, t, x, y):
        phi, _ = self.neural_net(params, t, x, y)
        return phi

    def T_net(self, params, t, x, y):
        _, T = self.neural_net(params, t, x, y)
        return T

    def term1(self, params, t, x, y):
        phi_x = grad(self.phi_net, argnums=2)(params, t, x, y)
        phi_y = grad(self.phi_net, argnums=3)(params, t, x, y)
        thet = jnp.arctan(phi_y / (phi_x + 1e-5))
        return (epsilon * sigma(thet)) * depsilon(thet) * phi_x

    def term2(self, params, t, x, y):
        phi_x = grad(self.phi_net, argnums=2)(params, t, x, y)
        phi_y = grad(self.phi_net, argnums=3)(params, t, x, y)
        thet = jnp.arctan(phi_y / (phi_x + 1e-5))
        return (epsilon * sigma(thet)) * depsilon(thet) * phi_y

    def residual_net(self, params, t, x, y):

        phi, T = self.neural_net(params, t, x, y)

        # phi_x = grad(self.phi_net, argnums=2)(params, t, x, y)
        # phi_y = grad(self.phi_net, argnums=3)(params, t, x, y)
        phi_t = grad(self.phi_net, argnums=1)(params, t, x, y)
        phi_xx = grad(grad(self.phi_net, argnums=2), argnums=2)(params, t, x, y)
        phi_yy = grad(grad(self.phi_net, argnums=3), argnums=3)(params, t, x, y)
        # thet = theta(phi_x, phi_y)

        T_t = grad(self.T_net, argnums=1)(params, t, x, y)
        T_xx = grad(grad(self.T_net, argnums=2), argnums=2)(params, t, x, y)
        T_yy = grad(grad(self.T_net, argnums=3), argnums=3)(params, t, x, y)

        # aniso = grad(self.term1, argnums=3)(params, t, x, y) + grad(self.term2, argnums=2)(params, t, x, y)

        phi_res = tau * phi_t - (
                phi * (1 - phi) * (phi - 1 / 2 + m(T)) +
                (epsilon) ** 2 * (phi_xx + phi_yy)
        )
        # T
        T_res = T_t - (T_xx + T_yy + kappa * phi_t)

        return phi_res, T_res

    @partial(jit, static_argnums=(0,))
    def vis_net(self, params, t):
        u = self.plot_fn(params, t, self.xic, self.yic)
        return u

    @partial(jit, static_argnums=(0,))
    def phi_t_net(self, params, t, x, y):
        phi_t = grad(self.phi_net, argnums=1)(params, t, x, y)
        return phi_t

    @partial(jit, static_argnums=(0,))
    def phi_t_plot(self, params, t):
        phi_t = self.phi_t_fn(params, t, self.xic, self.yic)
        return phi_t

    @partial(jit, static_argnums=(0,))
    def residuals_and_weights(self, params, tol, batch):
        t_r, x_r = batch
        loss_u0, loss_v0 = self.loss_ics(params)

        L_0 = 1e3 * (loss_u0 + loss_v0)
        res_phi_pred, res_T_pred = self.r_pred_fn(params, t_r, x_r[:, 0], x_r[:, 1])
        L_t = jnp.mean(res_phi_pred ** 2 + res_T_pred ** 2, axis=1)
        W = lax.stop_gradient(jnp.exp(- tol * (self.M @ L_t + L_0)))
        return L_0, L_t, W

    @partial(jit, static_argnums=(0,))
    def loss_ics(self, params):
        # Compute forward pass
        phi0_pred = self.phi0_pred_fn(params, 0.0, self.xic, self.yic)
        T0_pred = self.T0_pred_fn(params, 0.0, self.xic, self.yic)
        # Compute loss
        loss_phi0 = jnp.mean((phi0_pred - self.state0[0]) ** 2)
        loss_T0 = jnp.mean((T0_pred - self.state0[1]) ** 2)
        return loss_phi0, loss_T0

    @partial(jit, static_argnums=(0,))
    def loss_bcs(self, params):
        # Compute forward pass
        phi_left = self.phi_bc_fn(params, self.tbc, jnp.zeros_like(self.ybc), self.ybc)
        phi_right = self.phi_bc_fn(params, self.tbc, Lx * jnp.ones_like(self.ybc), self.ybc)
        phi_bottom = self.phi_bc_fn(params, self.tbc, self.xbc, jnp.zeros_like(self.xbc))
        phi_top = self.phi_bc_fn(params, self.tbc, self.xbc, Ly * jnp.ones_like(self.xbc))

        T_left = self.T_bc_fn(params, self.tbc, jnp.zeros_like(self.ybc), self.ybc)
        T_right = self.T_bc_fn(params, self.tbc, Lx * jnp.ones_like(self.ybc), self.ybc)
        T_bottom = self.T_bc_fn(params, self.tbc, self.xbc, jnp.zeros_like(self.xbc))
        T_top = self.T_bc_fn(params, self.tbc, self.xbc, Ly * jnp.ones_like(self.xbc))

        # Zero Dirichlet BCs
        loss_phi = jnp.mean((phi_left - jnp.ones_like(phi_left)) ** 2 + phi_right ** 2 + phi_bottom ** 2 + phi_top ** 2)
        loss_T = jnp.mean(T_left ** 2 + T_right ** 2 + T_bottom ** 2 + T_top ** 2)

        return loss_phi, loss_T

    @partial(jit, static_argnums=(0,))
    def loss_res(self, params, batch):
        t_r, x_r = batch
        # Compute forward pass
        res_phi_pred, res_T_pred = self.r_pred_fn(params, t_r, x_r[:, 0], x_r[:, 1])
        # Compute loss
        loss_res_phi = jnp.mean(res_phi_pred ** 2)
        loss_res_T = jnp.mean(res_T_pred ** 2)
        return loss_res_phi, loss_res_T

    @partial(jit, static_argnums=(0,))
    def loss(self, params, batch):
        L_0, L_t, W = self.residuals_and_weights(params, self.tol, batch)
        # phibc, Tbc = self.loss_bcs(params)
        Lbc = 0  # 1e1 * (phibc + Tbc)
        # Compute loss
        loss = jnp.mean(W * L_t + L_0 + Lbc)
        return loss

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, batch):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, batch)
        optimizers.clip_grads(g, 1e3)
        return self.opt_update(i, g, opt_state)

    # Optimize parameters in a loop
    def train(self, dataset, nIter=10000):
        res_data = iter(dataset)
        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            batch = next(res_data)
            self.current_count = next(self.itercount)
            self.opt_state = self.step(self.current_count, self.opt_state, batch)

            if it % 500 == 0:
                params = self.get_params(self.opt_state)
                loss_value = self.loss(params, batch)
                loss_phi0_value, loss_T0_value = self.loss_ics(params)
                loss_res_phi_value, loss_res_T_value = self.loss_res(params, batch)
                _, _, W_value = self.residuals_and_weights(params, tol, batch)
                self.loss_log.append(loss_value)
                self.loss_phi0_log.append(loss_phi0_value)
                self.loss_T0_log.append(loss_T0_value)
                self.loss_res_phi_log.append(loss_res_phi_value)
                self.loss_res_T_log.append(loss_res_T_value)

                pbar.set_postfix({
                    'Loss': loss_value,
                    'loss_phi0': loss_phi0_value,
                    'loss_T0': loss_T0_value,
                    'loss_res_T': loss_res_phi_value,
                    'loss_res_phi': loss_res_T_value,
                    'W_min': W_value.min()})

                if W_value.min() > 0.97:
                    break


# Create PINNs model
key = jr.PRNGKey(1234)

M_t = 2
M_x = 5
M_y = 5
d0 = 2 * M_x + 2 * M_y + 4 * M_x * M_y + M_t + 2
# d0 = 3
layers = [d0, 128, 128, 128, 128, 128, 2]

t0 = 0.0
t1 = 0.1
xic = jnp.linspace(0, Lx, 256)
yic = jnp.linspace(0, Ly, 256)
n_t = 200
tol = 1
tol_list = [1e-2, 1e-1]

# Create data set
n_x = 128
dataset = DataGenerator(t0, t1, n_t, n_x)

N = 1 # Number of neural networks over entire temporal domain
params_list = []
losses_list = []

phi0 = jnp.array([[phi0(x, y) for x in xic] for y in yic])
T0 = jnp.array([[T0(x, y) for x in xic] for y in yic])
state0 = jnp.array([phi0, T0])

for k in range(N):
    model = PINN(key, layers, M_t, M_x, M_y, state0, t0, t1, n_t, xic, yic, tol)

    # Train
    for tol in tol_list:
        model.tol = tol
        print('tol:', model.tol)
        # Train
        model.train(dataset, nIter=50000)

    # Store
    params = model.get_params(model.opt_state)
    flat_params, _ = ravel_pytree(params)
    params_list.append(flat_params)
    losses_list.append([model.loss_log,
                        model.loss_phi0_log,
                        model.loss_T0_log,
                        model.loss_res_phi_log,
                        model.loss_res_T_log, ])

    params = model.get_params(model.opt_state)
    phi0_pred = model.phi0_pred_fn(params, t1, xic, yic)
    T0_pred = model.T0_pred_fn(params, t1, xic, yic)
    state0 = jnp.stack([phi0_pred, T0_pred])

plot_vars(model, 0.)
plot_residual(model, 0.)

plot_vars(model, 0.0005)
plot_residual(model, 0.0005)

plot_vars(model, 0.001)
plot_residual(model, 0.001)

plot_vars(model, 0.005)
plot_residual(model, 0.005)

plot_vars(model, 0.01)
plot_residual(model, 0.01)

plot_vars(model, 0.1)
plot_residual(model, 0.1)

plot_losses(model)
