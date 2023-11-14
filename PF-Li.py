from pinnfuncs import *
import optax
import time

class PINN(eqx.Module):
    params: list

    def __init__(self, key, layers):
        k1, k2, key = jr.split(key, 3)
        U1, b1 = xavier_init(k1, layers[0], layers[1])
        U2, b2 = xavier_init(k2, layers[0], layers[1])
        key, *keys = jr.split(key, len(layers))
        params = list(map(xavier_init, keys, layers[:-1], layers[1:]))
        self.params = [params, U1, b1, U2, b2]

    @eqx.filter_jit
    def __call__(self, x, y, t):
        # Modified MLP structure
        s = jnp.array([x, y, t])
        # s = input_mapping(x, y, t) # Fourier features
        U = jax.nn.tanh(jnp.dot(s, self.params[1]) + self.params[2])
        V = jax.nn.tanh(jnp.dot(s, self.params[3]) + self.params[4])
        for W, b in self.params[0][:-1]:
            z = jnp.tanh(jnp.dot(s, W) + b)
            s = jnp.multiply(z, U) + jnp.multiply(1 - z, V)
        W, b = self.params[0][-1]
        z = jnp.dot(s, W) + b
        z = jnp.array([jax.nn.sigmoid(z[0]), -10. * jax.nn.sigmoid(z[1]), phie * jax.nn.sigmoid(z[2])])
        # z = output_mapping(x, z)  # Hard force boundary conditions
        # z = ic_mapping(x, y, t, z)  # Hard force initial conditions
        return z


layers = [1 * 3 + 0 * (4 * mapping_size + 2), 128, 128, 128,  128, 3]
u = PINN(key, layers)

lr = optax.exponential_decay(1E-3, 50000, 0.9)
optimizer = optax.adam(lr)
opt_state = optimizer.init(eqx.filter(u, eqx.is_array))

@eqx.filter_jit
def step(net, state):
    loss_t, grad = eqx.filter_value_and_grad(loss)(net, key)
    updates, new_state = optimizer.update(grad, state, net)
    new_net = eqx.apply_updates(net, updates)
    return new_net, new_state, loss_t


iters = 50000
pde_history = []
ic_history = []
bc_history = []
loss_history = []

begin = time.time()
for _ in range(iters + 1):
    key, subkey = jr.split(key, 2)
    u, opt_state, netloss = step(u, opt_state)
    if _ % 500 == 0:
        l, w = res_weights(u, xc, yc, tc)
        pde_history.append(jnp.mean(w * l))
        ic_history.append(ic_loss(u, xic, yic))
        loss_history.append(netloss)
        bc_history.append(bc_loss(u, xbc, ybc, tbc))
        print(f"Iteration: {_} | Total Weighted Loss: {netloss:.6f} | Residual Loss: {pde_history[-1]:.6f} "
              f"| BC Loss : {bc_history[-1]:.6f} | IC Loss: {ic_history[-1]:.6f}")

end = time.time()
print(f"Training time: {end - begin:.2f} seconds")
plot_dynamics(u, t0, tf, 151)
plot_losses(loss_history, pde_history, bc_history, ic_history)
plot_weights(u)
