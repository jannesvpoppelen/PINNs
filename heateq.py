import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import matplotlib.pyplot as plt

"""
PINN for
d/dt u = k * d^2/dx^2 u - e^(-t)(sin(pi * x) - pi^2 * sin(pi * x))
u(-1, t) = u(1, t) = 0
u(x, 0) = sin(pi * x)

Real solution : u(x,t) = e^(-t) * sin(pi * x)
"""

k = 1.0
Nr = 100
Nb = 20
Nt = 20
bcWeight = 10
icWeight = 10
xmin = -1
xmax = 1
t0 = 0
tf = 0.1
lr = 0.0001
iters = 5000

key = jr.PRNGKey(186)
key , subkey = jr.split(key)

u = eqx.nn.MLP(
    in_size=2,
    out_size="scalar",
    width_size=128,
    depth=3,
    activation=jax.nn.tanh,
    key=subkey
)


def pde_res(net, x, t):
    arg = jnp.array([x,t])
    u_t = jax.grad(net)(arg)[1]
    u_xx= jax.jacrev(jax.jacrev(net))(arg)[0][0]
    return u_t -(k * u_xx - jnp.exp(-t) * (jnp.sin(jnp.pi * x) - jnp.pi**2 * jnp.sin(jnp.pi * x)))


def bc_res(net, x, t):
    ul = jnp.array([-x,t])
    ur = jnp.array([x,t])
    gradu = jax.grad(net)
    ux_l = gradu(jnp.array([-x, t]))[0]
    ux_r = gradu(jnp.array([x, t]))[0]

    return jnp.sqrt(abs(net(ul))**2 + abs(net(ur))**2)#jnp.sqrt(abs(net(ul) - net(ur))**2 + abs(ux_l - ux_r)**2)


def ic_res(net, x, t):
    arg = jnp.array([x,t])
    return net(arg) - jnp.sin(jnp.pi * x)


def loss_func(net, x_coll, t_coll, xbc, tbc, xic, tic):
    loss_col = jnp.mean(jnp.square(jax.vmap(pde_res, in_axes=(None, 0, 0))(net, x_coll, t_coll)))
    loss_bc = jnp.mean(jnp.square(jax.vmap(bc_res, in_axes=(None, 0, 0))(net, xbc, tbc)))
    loss_ic = jnp.mean(jnp.square(jax.vmap(ic_res, in_axes=(None, 0, 0))(net, xic, tic)))



    return loss_col + bcWeight * loss_bc + icWeight * loss_ic


@eqx.filter_jit
def step(net, state):
    loss, grad = eqx.filter_value_and_grad(loss_func)(net, x_coll, t_coll, xbc, tbc, xic, tic)
    updates, new_state = optimizer.update(grad, state, net)
    new_net = eqx.apply_updates(net, updates)
    return new_net, new_state, loss

key, samplekey = jr.split(key)

x_coll = jr.uniform(samplekey, minval=xmin, maxval=xmax, shape=(Nr, ))
t_coll = jr.uniform(samplekey, minval=t0, maxval=tf, shape=(Nr, ))
tbc = jr.uniform(samplekey, minval=t0, maxval=tf, shape=(Nb, ))
xbc = xmax * jnp.ones_like(tbc)
xic = jr.uniform(samplekey, minval=xmin, maxval=xmax, shape=(Nt, ))
tic = t0 * jnp.ones_like(xic)

# Training
optimizer = optax.adam(lr)
opt_state = optimizer.init(eqx.filter(u, eqx.is_array))
loss_history = []
for _ in range(iters):
    u, opt_state, loss = step(u, opt_state)
    loss_history.append(loss)
    if _ % 100 == 0:
        print(f"Iteration {_} | Loss {loss}")

# Plot loss history
f = plt.figure()
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.yscale("log")
plt.title("Loss history")
plt.plot(loss_history)
plt.show()

# Plot solution
x = jnp.linspace(xmin, xmax, 100)
t = tf*jnp.ones_like(x)
ustar = []
ureal = jnp.exp(-tf) * jnp.sin(jnp.pi * x)

for i in range(len(x)):
   ustar.append(u(jnp.array([x[i], t[i]])))

ustar = jnp.array(ustar)
g = plt.figure()
plt.xlabel("x")
plt.ylabel("u")
plt.title(f"Solution at t={tf}")
plt.plot(x, ustar, label= 'Predicted solution')
plt.plot(x, ureal, label = 'Real solution', linestyle='--')
plt.legend()
plt.show()

print(f"MSE: {jnp.mean(jnp.square(ustar - ureal))}")
