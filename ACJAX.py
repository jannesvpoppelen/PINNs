import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
import matplotlib.pyplot as plt

"""
Implementation of 1+1D Allen-Cahn equation using JAX and Equinox.

d/dt u - k * (d^2/dx^2 u + 1/eps^2 * u * (u^2 - 1) = 0
k=0.0001, eps=1/sqrt(5)

Boundary conditions are periodic, and u(x,t=0) = x^2 * cos(pi x) (From original Raissi paper)
"""

# Residual from PDE
def pde_residual(net, x, t):
    arg = jnp.array([x, t])
    u = net(arg)
    u_t = jax.grad(net)(arg)[1]
    u_xx = jax.hessian(net)(arg)[0, 0]

    return u_t - k * u_xx + 1 / eps ** 2 * u * (u ** 2 - 1)


# Residual from BC
def bc_residual(net, x, t):
    ur = net(jnp.array([x, t]))
    ul = net(jnp.array([-x, t]))
    ur_x = jax.grad(net)(jnp.array([x, t]))[0]
    ul_x = jax.grad(net)(jnp.array([-x, t]))[0]

    return jnp.sqrt(abs((ur - ul))**2 + abs((ur_x - ul_x))**2)  # Abomination of a loss function


# Residual from IC
def ic_residual(net, x, t):
    u0 = net(jnp.array([x, t]))
    return u0 - x ** 2 * jnp.cos(jnp.pi * x)


# Loss function
def loss_f(net, x_col, t_col, xbc, tbc, xic, tic, w):
    loss_col = jnp.mean(jnp.square(jax.vmap(pde_residual, in_axes=(None, 0, 0))(net, x_col, t_col)))
    loss_bc = jnp.mean(jnp.square(jax.vmap(bc_residual, in_axes=(None, 0, 0))(net, xbc, tbc)))
    loss_ic = jnp.mean(jnp.square(jax.vmap(ic_residual, in_axes=(None, 0, 0))(net, xic, tic)))
    #print(f"{loss_col} | {loss_bc} | {loss_ic}")
    return w[0] * loss_col + w[1] * loss_bc + w[2] * loss_ic


# Step in training
def step(net, state):
    loss, grad = eqx.filter_value_and_grad(loss_f)(net, x_coll, t_coll, xbc, tbc, xic, tic, w)
    updates, new_state = optimizer.update(grad, state, net)
    new_net = eqx.apply_updates(net, updates)
    return new_net, new_state, loss


if __name__ == "__main__":
    '''
    # Global parameters and boundary/initial conditions
    key1 = random.PRNGKey(123456)
    key2 = random.PRNGKey(13456)
    xmin, xmax = -1.0, 1.0
    t0, tf = 0.0, 1.0
    k, eps = 0.0001, 1 / np.sqrt(5)
    Nb = 200  # Number of boundary points
    Nr = 8000  # Number of interior points
    Ni = 400  # Number of initial points

    # Initial condition
    x0 = random.uniform(key=key1, minval=xmin, maxval=xmax, shape=(Ni, 1))
    tin = jnp.zeros_like(x0)

    # Periodic BC
    xbc1 = random.uniform(key=key1, minval=xmin, maxval=xmin, shape=(Nb, 1))
    tbc = random.uniform(key=key2, minval=t0, maxval=tf, shape=(Nb, 1))
    xbc2 = random.uniform(key=key1, minval=xmax, maxval=xmax, shape=(Nb, 1))

    x_col = random.uniform(key=key1, minval=xmin, maxval=xmax, shape=(Nr, 1))
    t_col = random.uniform(key=key2, minval=t0, maxval=tf, shape=(Nr, 1))
    colloc = jnp.concatenate([x_col, t_col], axis=1)

    # Plot collocation points

    f = plt.figure()
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("Position of randomly sampled points")
    plt.scatter(x0, tin, marker='o', color='r', s=5, label='Initial points')
    plt.scatter(xbc1, tbc, marker='x', color='b', s=5, label='Boundary points')
    plt.scatter(xbc2, tbc, marker='x', color='b', s=5, label='Boundary points')
    plt.scatter(x_col, t_col, marker='.', color='k', s=5, label='Collocation points')
    plt.legend(loc='upper left')
    plt.show()
    '''

    k = 0.0001
    eps = 1 / jnp.sqrt(5)
    xmin = -1.0
    xmax = 1.0
    t0 = 0.0
    tf = 0.1
    lr = 1e-3
    w = [1, 100, 100]
    Nr = 5000  # Number of interior points
    Nb = 1000  # Number of boundary points
    Ni = 1000  # Number of initial points
    iters = 3000
    key = jr.PRNGKey(123456)
    key, subkey = jr.split(key)
    pinn = eqx.nn.MLP(
        in_size=2,
        out_size='scalar',
        width_size=128,
        depth=4,
        activation=jax.nn.tanh,
        key=subkey,
    )
    key, samplekey = jr.split(key)

    # Sampled points
    x_coll = jr.uniform(samplekey, minval=xmin, maxval=xmax, shape=(Nr, ))
    t_coll = jr.uniform(samplekey, minval=t0, maxval=tf, shape=(Nr, ))
    tbc = jr.uniform(samplekey, minval=t0, maxval=tf, shape=(Nb, ))
    xbc = xmax * jnp.ones_like(tbc)
    xic = jr.uniform(samplekey, minval=xmin, maxval=xmax, shape=(Ni, ))
    tic = t0 * jnp.ones_like(xic)

    # Training
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(pinn, eqx.is_array))
    loss_history = []
    for _ in range(iters):
        pinn, opt_state, loss = step(pinn, opt_state)
        loss_history.append(loss)
        if _ % 100 == 0:
            print(f"Iteration {_} | Loss {loss}")

    # Plot loss history
    f = plt.figure()
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss history")
    plt.plot(loss_history)
    plt.show()
    plt.clf()

    # Plot solution
    x = jnp.linspace(xmin, xmax, 1000)
    t = tf*jnp.ones_like(x)
    u = []
    for _ in range(len(x)):
        u.append(pinn(jnp.array([x[_], t[_]])))
    g = plt.figure()
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title(f"Solution at t={tf}")
    plt.plot(x, u)
    plt.show()
