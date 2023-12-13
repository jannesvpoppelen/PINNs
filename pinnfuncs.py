import jax.random as jr
from jax.example_libraries import optimizers
from jax.flatten_util import ravel_pytree
import itertools
from functools import partial
from tqdm import trange
from pdefuncs import *
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
# plt.rcParams["figure.figsize"] = (10, 6)

plt.rc('font', size=6)  # controls default text sizes
plt.rc('axes', titlesize=6)  # fontsize of the axes title
plt.rc('axes', labelsize=6)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=4)  # fontsize of the tick labels
plt.rc('ytick', labelsize=4)  # fontsize of the tick labels
plt.rc('legend', fontsize=4)  # legend fontsize

M = 3
B = jr.normal(jr.PRNGKey(589), (M,))
icWeight = 1000
bcWeight = 50


def plot_vars(net, t, i):
    # Evaluate the network over the grid
    params = net.get_params(net.opt_state)
    upred = net.vis_net(params, t)
    eta = upred[:, :, 0]
    mu = upred[:, :, 1]
    phi = upred[:, :, 2]

    # Create a figure with 3 subplots arranged vertically and more vertical spacing
    fig, axs = plt.subplots(3, 1, figsize=(3, 4), sharex=False, gridspec_kw={'hspace': 0.8})
    plt.setp(axs, xticks=[0, 2.5, 5], yticks=[0, 2.5, 5])

    # Plot eta
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_title(f" η(t = {t:.2f})")
    im = axs[0].imshow(eta, interpolation='nearest', cmap='jet', extent=[net.xmin, net.xmax, net.ymin, net.ymax],
                       vmin=0, vmax=1)
    cax = fig.add_axes([axs[0].get_position().x1 + 0.025, axs[0].get_position().y0, 0.02, axs[0].get_position().height])
    plt.colorbar(im, cax=cax)

    # Plot mu
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].set_title(f"μ(t = {t:.2f})")
    im = axs[1].imshow(mu, interpolation='nearest', cmap='jet', extent=[net.xmin, net.xmax, net.ymin, net.ymax],
                       vmin=-10, vmax=0)
    cax = fig.add_axes([axs[1].get_position().x1 + 0.025, axs[1].get_position().y0, 0.02, axs[1].get_position().height])
    plt.colorbar(im, cax=cax)

    # Plot phi
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")
    axs[2].set_title(f"ϕ(t = {t:.2f})")
    im = axs[2].imshow(phi, interpolation='nearest', cmap='jet', extent=[net.xmin, net.xmax, net.ymin, net.ymax],
                       vmin=phie, vmax=0)
    cax = fig.add_axes([axs[2].get_position().x1 + 0.025, axs[2].get_position().y0, 0.02, axs[2].get_position().height])
    plt.colorbar(im, cax=cax)

    # Save the figure
    plt.savefig(f"vars_{i}.png")
    plt.clf()
    return None


def plot_dynamics(net, t0, tf, frames):
    for (t, i) in enumerate(jnp.linspace(t0, tf, frames)):
        plot_vars(net, i, t)
    return None


def plot_losses(net):
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss history")
    plt.yscale("log")
    plt.plot(net.loss_log, label="Total loss")
    plt.plot(net.loss_res_log, label="PDE loss")
    plt.plot(net.loss_bcs_log, label="BC loss")
    plt.plot(net.loss_ics_log, label="IC loss")
    plt.legend(loc="upper right")
    plt.savefig("loss.png")
    plt.clf()


def plot_weights(net):
    _, W = net.residuals_and_weights(net.get_params(net.opt_state), net.tol)
    nums = jnp.linspace(1, len(W), len(W))
    plt.xlabel("Weight #")
    plt.ylabel("Weight")
    plt.plot(nums, W)
    plt.savefig("weights.png")
    plt.clf()


def MLP(layers):
    def input_mapping(t, x, y):
        out = jnp.hstack([t, 1,
                          jnp.cos(x * B.T), jnp.sin(x * B.T), jnp.cos(y * B.T), jnp.sin(y * B.T)])
        return out

    def init(key):
        def xavier_init(rngkey, d_in, d_out):
            rngkey, _ = jr.split(rngkey)
            glorot_stddev = 1.0 / jnp.sqrt((d_in + d_out) / 2.)
            W = glorot_stddev * jr.normal(k1, (d_in, d_out))
            b = jnp.zeros(d_out)
            return W, b

        k1, k2, key = jr.split(key, 3)
        U1, b1 = xavier_init(k1, layers[0], layers[1])
        U2, b2 = xavier_init(k2, layers[0], layers[1])

        key, *keys = jr.split(key, len(layers))
        params = list(map(xavier_init, keys, layers[:-1], layers[1:]))
        return [params, U1, b1, U2, b2]

    def apply(params, inputs):
        t = inputs[0]
        x = inputs[1]
        y = inputs[2]
        #s = input_mapping(t, x, y)
        s = jnp.array([t, x, y])
        U = jnp.tanh(jnp.dot(s, params[1]) + params[2])
        V = jnp.tanh(jnp.dot(s, params[3]) + params[4])
        for W, b in params[0][:-1]:
            z = jnp.dot(s, W) + b
            s = jnp.multiply(jnp.tanh(z), U) + jnp.multiply(1 - jnp.tanh(z), V)
        W, b = params[0][-1]
        z = jnp.dot(s, W) + b
        #z = jnp.array([jax.nn.sigmoid(z[0]), -10*jax.nn.tanh(z[1]), phie * jax.nn.sigmoid(z[2])])
        z = jnp.array([eta0(x, y) + t * z[0], mu0(x, y) + t * z[1], phi0(x, y) + t * z[2]]) # Hard IC
        z = jnp.array([jax.nn.sigmoid(z[0]), -10*jax.nn.sigmoid(z[1]), phie * jax.nn.sigmoid(z[2])])
        return z

    return init, apply


class PINN:
    def __init__(self, layers, geometry, u0, n_t, n_x, n_y, n_ic, n_bc, tol):
        # collocation
        key = jr.PRNGKey(1234)
        key1 = jr.PRNGKey(5678)
        key2 = jr.PRNGKey(9101112)
        self.current_count = 0
        self.key = key1
        self.key2 = key2
        self.t0 = geometry[0]
        self.tf = geometry[1]
        self.xmin, self.xmax = geometry[2], geometry[3]
        self.ymin, self.ymax = geometry[4], geometry[5]
        self.tr = jnp.linspace(self.t0, self.tf, n_t)
        self.xr = jr.uniform(self.key, minval=self.xmin, maxval=self.xmax, shape=(n_x,))
        self.yr = jr.uniform(self.key2, minval=self.ymin, maxval=self.ymax, shape=(n_y,))
        self.xic = jnp.linspace(self.xmin, self.xmax, n_ic)
        self.yic = jnp.linspace(self.ymin, self.ymax, n_ic)
        self.n_t = n_t
        self.n_x = n_x
        self.n_y = n_y
        self.xc = jnp.linspace(self.xmin, self.xmax, 512)
        self.yc = jnp.linspace(self.ymin, self.ymax, 512)

        # For computing the temporal weights
        self.Mt = jnp.triu(jnp.ones((n_t, n_t)), k=1).T
        self.tol = tol

        # IC
        self.u0 = u0

        # BC
        self.tbc = jnp.linspace(self.t0, self.tf, int(n_bc / 2))
        self.xbc = jnp.linspace(self.xmin, self.xmax, n_bc)
        self.ybc = jnp.linspace(self.ymin, self.ymax, n_bc)

        # Initalize the network
        self.init, self.apply = MLP(layers)
        params = self.init(key=key)
        _, self.unravel = ravel_pytree(params)

        # Use optimizers to set optimizer initialization and update functions
        lr = optimizers.exponential_decay(1e-3, decay_steps=25000, decay_rate=0.9)
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(lr)
        self.opt_state = self.opt_init(params)

        # Evaluate the network and the residual over the grid
        self.u_func = jax.vmap(jax.vmap(self.neural_net, (None, 0, None, None)), (None, None, 0, 0))
        self.uy_func = jax.vmap(jax.vmap(jax.jacfwd(self.neural_net, argnums=3), (None, 0, None, None)),
                                (None, None, 0, 0))
        self.res_func = jax.vmap(jax.vmap(self.residual_net, (None, None, 0, 0)), (None, 0, None, None))
        self.ic_func = jax.vmap(jax.vmap(self.neural_net, (None, None, 0, None)), in_axes=(None, None, None, 0))
        self.plot_func = jax.vmap(jax.vmap(self.neural_net, (None, None, 0, None)), (None, None, None, 0))

        # Logger
        self.itercount = itertools.count()

        self.loss_log = []
        self.loss_ics_log = []
        self.loss_bcs_log = []
        self.loss_res_log = []

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, t, x, y):
        return self.neural_net(self.get_params(self.opt_state), t, x, y)

    @partial(jax.jit, static_argnums=(0,))
    def neural_net(self, params, t, x, y):
        z = jnp.stack([t, x, y])
        outputs = self.apply(params, z)
        return outputs

    @partial(jax.jit, static_argnums=(0,))
    def residual_net(self, params, t, x, y):
        u = self.neural_net(params, t, x, y)
        u_t = jax.jacfwd(self.neural_net, argnums=1)(params, t, x, y)
        u_x = jax.jacfwd(self.neural_net, argnums=2)(params, t, x, y)
        u_y = jax.jacfwd(self.neural_net, argnums=3)(params, t, x, y)
        u_xx = jax.jacfwd(jax.jacfwd(self.neural_net, argnums=2), argnums=2)(params, t, x, y)
        u_yy = jax.jacfwd(jax.jacfwd(self.neural_net, argnums=3), argnums=3)(params, t, x, y)

        eta, mu, phi = u[0], u[1], u[2]
        eta_t, eta_x, eta_y = u_t[0], u_x[0], u_y[0]
        eta_xx, eta_yy = u_xx[0], u_yy[0]

        mu_t, mu_x, mu_y = u_t[1], u_xx[1], u_yy[1]
        mu_xx, mu_yy = u_xx[1], u_yy[1]

        phi_x, phi_y = u_x[2], u_y[2]
        phi_xx, phi_yy = u_xx[2], u_yy[2]

        # ETA
        eta_res = eta_t - (M * (kappa * (eta_xx + eta_yy) - dg(eta))
                           - Mnu * dh(eta) * (
                                   jnp.exp((1 - alpha) * frac * phi) - 1 / c0 * cl(mu) * (1 - h(eta)) * jnp.exp(
                               - alpha * frac * phi)))

        # MU
        gradmuphi = jnp.array([mu_x + frac * phi_x, mu_y + frac * phi_y])
        DD = D0 * dcldmu(mu) * jnp.array([mu_x, mu_y])

        mu_res = chi(eta, mu) * mu_t - (D(eta, mu) * ((mu_xx + mu_yy) + frac * (phi_xx + phi_yy))
                                        + jnp.dot(DD, gradmuphi)
                                        - ft(mu) * dh(eta) * eta_t)

        # PHI
        dsigma = (sigmas - sigmal) * dh(eta) * jnp.array([eta_x, eta_y])
        gradphi = jnp.array([phi_x, phi_y])
        phi_res = (jnp.dot(dsigma, gradphi) + sigma(eta) * (phi_xx + phi_yy)) - fac * eta_t

        return 1*abs(eta_res) + 1*abs(mu_res) + 1*abs(phi_res)

    @partial(jax.jit, static_argnums=(0,))
    def ic_net(self, params, x, y):
        u = self.ic_func(params, 0., x, y)
        return u

    @partial(jax.jit, static_argnums=(0,))
    def vis_net(self, params, t):
        u = self.plot_func(params, t, self.xc, self.yc)
        return u

    @partial(jax.jit, static_argnums=(0,))
    def residuals_and_weights(self, params, tol):
        r_pred = self.res_func(params, self.tr, self.xr, self.yr)
        L_t = jnp.mean(r_pred ** 2, axis=1)
        W = jax.lax.stop_gradient(jnp.exp(- tol * (self.Mt @ L_t)))
        return L_t, W

    @partial(jax.jit, static_argnums=(0,))
    def loss_ic(self, params):
        # Evaluate the network over IC
        u_pred = self.ic_net(params, self.xic, self.yic)
        eta, mu, phi = u_pred[:, :, 0], u_pred[:, :, 1], u_pred[:, :, 2]
        eta_res = jnp.mean((self.u0[0] - eta) ** 2)
        mu_res = jnp.mean((self.u0[1] - mu) ** 2)
        phi_res = jnp.mean((self.u0[2] - phi) ** 2)
        loss_ic = abs(eta_res) + abs(mu_res) + abs(phi_res)
        return loss_ic

    @partial(jax.jit, static_argnums=(0,))
    def loss_res(self, params):
        r_pred = self.res_func(params, self.tr, self.xr, self.yr)
        # Compute loss
        loss_r = jnp.mean(r_pred ** 2)
        return loss_r

    @partial(jax.jit, static_argnums=(0,))
    def loss_bc(self, params):
        u_left = self.u_func(params, self.tbc, jnp.zeros_like(self.ybc), self.ybc)
        u_right = self.u_func(params, self.tbc, self.xmax * jnp.ones_like(self.ybc), self.ybc)
        uy_top = self.uy_func(params, self.tbc, self.xbc, self.ymax * jnp.ones_like(self.xbc))
        uy_bot = self.uy_func(params, self.tbc, self.xbc, jnp.zeros_like(self.xbc))

        eta_l, phi_l = u_left[:, :, 0], u_left[:, :, 2]
        eta_r, mu_r, phi_r = u_right[:, :, 0], u_right[:, :, 1], u_right[:, :, 2]
        eta_top, mu_top, phi_top = uy_top[:, :, 0], uy_top[:, :, 1], uy_top[:, :, 2]
        eta_bot, mu_bot, phi_bot = uy_bot[:, :, 0], uy_bot[:, :, 1], uy_bot[:, :, 2]

        eta_res = jnp.mean(((eta_l - jnp.ones_like(eta_l)) + (eta_r - jnp.zeros_like(eta_r)) +
                            (eta_top - jnp.zeros_like(eta_top)) + (eta_bot - jnp.zeros_like(eta_bot))) ** 2)

        mu_res = jnp.mean(((mu_r - jnp.zeros_like(mu_r)) + (mu_top - jnp.zeros_like(mu_top))
                           + (mu_bot - jnp.zeros_like(mu_bot))) ** 2)
        phi_res = jnp.mean(((phi_l - phie * jnp.ones_like(phi_l)) + (phi_r - jnp.zeros_like(phi_r)) +
                            (phi_top - jnp.zeros_like(phi_top)) + (phi_bot - jnp.zeros_like(phi_bot))) ** 2)

        loss_bc = abs(eta_res) + abs(mu_res) + abs(phi_res)
        return loss_bc

    @partial(jax.jit, static_argnums=(0,))
    def loss(self, params):
        Lic = icWeight * self.loss_ic(params)
        Lbc = bcWeight * self.loss_bc(params)
        L_t, W = self.residuals_and_weights(params, self.tol)
        loss = jnp.mean(W * L_t) + Lic + Lbc
        return loss

    @partial(jax.jit, static_argnums=(0,))
    def step(self, i, opt_state):
        params = self.get_params(opt_state)
        grad = jax.grad(self.loss)(params)
        print(grad)
        return self.opt_update(i, grad, opt_state)

    def train(self, nIter):
        pbar = trange(nIter, miniters=int(nIter / 1000))
        # Main training loop
        for it in pbar:
            self.key, _ = jr.split(self.key)
            self.key2, _ = jr.split(self.key2)
            self.xr = jr.uniform(self.key, minval=0, maxval=1, shape=(self.n_x,))
            self.yr = jr.uniform(self.key2, minval=0, maxval=1, shape=(self.n_y,))
            self.current_count = next(self.itercount)
            self.opt_state = self.step(self.current_count, self.opt_state)
            if it % 1000 == 0:
                params = self.get_params(self.opt_state)
                
                Lt, W = self.residuals_and_weights(params, self.tol)
                loss_value = self.loss(params)
                loss_ics_value = self.loss_ic(params)
                loss_res_value = jnp.mean(Lt*W) #self.loss_res(params)
                loss_bc_value = self.loss_bc(params)

                self.loss_log.append(loss_value)
                self.loss_ics_log.append(loss_ics_value)
                self.loss_res_log.append(loss_res_value)
                self.loss_bcs_log.append(loss_bc_value)

                pbar.set_postfix({'Loss': loss_value, 'loss_res': loss_res_value,
                                  'loss_ics': loss_ics_value, 'loss_bcs': loss_bc_value})
