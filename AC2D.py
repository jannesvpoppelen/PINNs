import jax.nn
import jax.numpy as np
import jax.random as jr
from jax import random, grad, vmap, jit
from jax.example_libraries import optimizers
from jax.flatten_util import ravel_pytree
from jax import lax
from scipy.ndimage import gaussian_filter
import itertools
from functools import partial
from tqdm import trange
import matplotlib.pyplot as plt

M = 1
M_t = 2
M_x = 5
M_y = 5

def MLP(layers, L=1.0, M=1, activation=np.tanh):
    w_x = 2.0 * np.pi
    w_y = 2.0 * np.pi
    k_x = np.arange(1, M_x + 1)
    k_y = np.arange(1, M_y + 1)
    k_xx, k_yy = np.meshgrid(k_x, k_y)
    k_xx = k_xx.flatten()
    k_yy = k_yy.flatten()

    def input_encoding(t, x, y):
        k_t = np.power(10.0, np.arange(0, M_t + 1))
        out = np.hstack([1, k_t * t,
                         np.cos(k_x * w_x * x), np.cos(k_y * w_y * y),
                         np.sin(k_x * w_x * x), np.sin(k_y * w_y * y),
                         np.cos(k_xx * w_x * x) * np.cos(k_yy * w_y * y),
                         np.cos(k_xx * w_x * x) * np.sin(k_yy * w_y * y),
                         np.sin(k_xx * w_x * x) * np.cos(k_yy * w_y * y),
                         np.sin(k_xx * w_x * x) * np.sin(k_yy * w_y * y)])
        return out

    def init(rng_key):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            glorot_stddev = 1.0 / np.sqrt((d_in + d_out) / 2.)
            W = glorot_stddev * random.normal(k1, (d_in, d_out))
            b = np.zeros(d_out)
            return W, b

        k3, k4, k5 = random.split(rng_key, 3)
        U1, b1 = init_layer(k3, layers[0], layers[1])
        U2, b2 = init_layer(k4, layers[0], layers[1])

        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return [params, U1, b1, U2, b2]

    def apply(params, inputs):
        t = inputs[0]
        x = inputs[1]
        y = inputs[2]
        H = input_encoding(t, x, y)
        U = np.tanh(np.dot(H, params[1]) + params[2])
        V = np.tanh(np.dot(H, params[3]) + params[4])
        for W, b in params[0][:-1]:
            outputs = np.dot(H, W) + b
            H = np.multiply(activation(outputs), U) + np.multiply(1-activation(outputs), V)
        W, b = params[0][-1]
        outputs = np.dot(H, W) + b
        return outputs

    return init, apply


class PINN:
    def __init__(self, layers, M, state0, t0, t1, n_t, n_x, n_y, tol):
        icWeight = 100
        # collocation
        key = random.PRNGKey(1234)
        key1 = random.PRNGKey(5678)
        key2 = random.PRNGKey(9101112)
        self.key = key1
        self.key2 = key2
        self.t0 = t0
        self.t1 = t1
        self.t_r = np.linspace(self.t0, self.t1, n_t)
        #self.x_r = np.linspace(0, 1, n_x)
        #self.y_r = np.linspace(0, 1, n_y)
        self.x_r = random.uniform(self.key, minval=0, maxval=1, shape=(n_x,))
        self.y_r = random.uniform(self.key2, minval=0, maxval=1, shape=(n_y,))

        # For computing the temporal weights
        self.Mt = np.triu(np.ones((n_t, n_t)), k=1).T
        self.tol = tol

        # IC
        self.u0 = state0

        # BC
        t_bc = np.linspace(self.t0, self.t1, n_t)

        # Initalize the network
        self.init, self.apply = MLP(layers, L=1.0, M=M, activation=np.tanh)
        params = self.init(rng_key=key)
        _, self.unravel = ravel_pytree(params)

        # Use optimizers to set optimizer initialization and update functions
        lr = optimizers.exponential_decay(1e-3, decay_steps=50000, decay_rate=0.9)
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(lr)
        self.opt_state = self.opt_init(params)

        # Evaluate the network and the residual over the grid
        self.u_pred_fn = vmap(vmap(self.neural_net, (None, 0, None, None)), (None, None, 0, 0))
        self.r_pred_fn = vmap(vmap(self.residual_net, (None, None, 0, 0)), (None, 0, None, None))
        self.vis_fn = vmap(vmap(self.neural_net, (None, None, 0, None)), (None, None, None, 0))

        # Logger
        self.itercount = itertools.count()

        self.loss_log = []
        self.loss_ics_log = []
        self.loss_res_log = []
        self.W_log = []
        self.L_t_log = []
    @partial(jit, static_argnums=(0,))
    def neural_net(self, params, t, x, y):
            z = np.stack([t, x, y])
            outputs = self.apply(params, z)
            return outputs[0]
    @partial(jit, static_argnums=(0,))
    def residual_net(self, params, t, x, y):
            u = self.neural_net(params, t, x, y)
            u_t = grad(self.neural_net, argnums=1)(params, t, x, y)
            #u_x = grad(self.neural_net, argnums=2)(params, t, x, y)
            u_xx = grad(grad(self.neural_net, argnums=2), argnums=2)(params, t, x, y)
            #u_y = grad(self.neural_net, argnums=3)(params, t, x, y)
            u_yy = grad(grad(self.neural_net, argnums=3), argnums=3)(params, t, x, y)
            return u_t + (u ** 3 - u) - nu * (u_xx + u_yy)

    @partial(jit, static_argnums=(0,))
    def vis_net(self, params, t):
        u = self.vis_fn(params, t, xc, yc)
        return u

    @partial(jit, static_argnums=(0,))
    def residuals_and_weights(self, params, tol):
            r_pred = self.r_pred_fn(params, self.t_r, self.x_r, self.y_r)
            L_t = np.mean(r_pred ** 2, axis=1)
            W = lax.stop_gradient(np.exp(- tol * (self.Mt @ L_t)))
            return L_t, W
    @partial(jit, static_argnums=(0,))
    def loss_ics(self, params):
            # Evaluate the network over IC
            u_pred = vmap(vmap(self.neural_net, (None, None, 0, None)),in_axes=(None, None, None, 0))(params, 0., xc, yc)
            # Compute the initial loss
            loss_ics = np.mean((self.u0.flatten() - u_pred.flatten()) ** 2)
            return loss_ics
    @partial(jit, static_argnums=(0,))
    def loss_res(self, params):
            r_pred = self.r_pred_fn(params, self.t_r, self.x_r, self.y_r)
            # Compute loss
            loss_r = np.mean(r_pred ** 2)
            return loss_r
    @partial(jit, static_argnums=(0,))
    def loss(self, params):
            L0 = 1000 * self.loss_ics(params)
            L_t, W = self.residuals_and_weights(params, self.tol)
            # Compute loss
            loss = np.mean(W * L_t) + L0
            return loss
    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params)
        return self.opt_update(i, g, opt_state)
    # Optimize parameters in a loop
    def train(self, nIter):
            pbar = trange(nIter)
            # Main training loop
            for it in pbar:
                self.key, _ = random.split(self.key)
                self.key2, _ = random.split(self.key2)
                self.x_r = random.uniform(self.key, minval=0, maxval=1, shape=(n_x,))
                self.y_r = random.uniform(self.key2, minval=0, maxval=1, shape=(n_y,))
                self.current_count = next(self.itercount)
                self.opt_state = self.step(self.current_count, self.opt_state)
                if it % 1000 == 0:
                    params = self.get_params(self.opt_state)

                    loss_value = self.loss(params)
                    loss_ics_value = self.loss_ics(params)
                    loss_res_value = self.loss_res(params)
                    L_t_value, W_value = self.residuals_and_weights(params, self.tol)

                    self.loss_log.append(loss_value)
                    self.loss_ics_log.append(loss_ics_value)
                    self.loss_res_log.append(loss_res_value)
                    self.W_log.append(W_value)
                    self.L_t_log.append(L_t_value)

                    pbar.set_postfix({'Loss': loss_value,'loss_ics': loss_ics_value, 'loss_res': loss_res_value})



 # Create PINNs model

# Network architecture
d0 = 2 * M_x + 2 * M_y + 4 * M_x * M_y + M_t + 2
layers = [d0, 128, 128, 128, 128, 1]

# hpyer-parameters
nu = 0.0001
t0 = 0.0
t1 = 1.
n_t = 100
n_x = 512
n_y = 512
tol = 1.0


tc = np.linspace(t0, t1, n_t)
xc = np.linspace(0, 1, 256)
yc = np.linspace(0, 1, 256)


# Initial condition
a = jr.uniform(jr.PRNGKey(25), (256, 256), minval=-1, maxval=1) # Random 2D Array
state0 = gaussian_filter(a, sigma=10) # Smear it with a Gaussian filter
state0 = state0 / np.max(state0) * 0.4 # Fix amplitude

u = PINN(layers, M, state0, t0, t1, n_t, n_x, n_y, tol)

u.train(50000)


plt.plot(u.loss_ics_log,label="IC Loss")
plt.plot(u.loss_res_log, label="Residual Loss")
plt.yscale('log')
plt.savefig('loss.png')
plt.clf()

params = u.get_params(u.opt_state)

upred = u.vis_net(params, 0.0)
im = plt.imshow(upred, interpolation='nearest', cmap='seismic', extent=[0, 1, 0, 1],vmin=-1, vmax=1)
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"u(t = {0.0:.2f})")
plt.colorbar(im)
plt.savefig('u_0.png')
plt.clf()


upred = u.vis_net(params, 0.5)
im = plt.imshow(upred, interpolation='nearest', cmap='seismic', extent=[0, 1, 0, 1],vmin=-1, vmax=1)
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"u(t = {0.5:.2f})")
plt.colorbar(im)
plt.savefig('u_05.png')
plt.clf()


upred = u.vis_net(params, 1.0)
im = plt.imshow(upred, interpolation='nearest', cmap='seismic', extent=[0, 1, 0, 1],vmin=-1,vmax=1)
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"u(t = {1.0:.2f})")
plt.colorbar(im)
plt.savefig('u_1.png')
plt.clf()



L_t, W = u.residuals_and_weights(params, u.tol)
fig = plt.figure(figsize=(6, 5))
plt.plot(u.t_r, W)
plt.xlabel('t')
plt.ylabel('w')
plt.savefig('weights.png')
plt.clf()


fig = plt.figure(figsize=(6, 5))
plt.plot(u.t_r, L_t)
plt.xlabel('t')
plt.ylabel('L')
plt.savefig('causal_loss.png')
plt.clf()