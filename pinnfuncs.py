import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_losses(loss_history, pde_loss, bc_loss, ic_loss):
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss history")
    plt.yscale("log")
    plt.plot(loss_history, label="Total loss")
    plt.plot(pde_loss, label="PDE loss")
    plt.plot(bc_loss, label="BC loss")
    plt.plot(ic_loss, label="IC loss")
    plt.legend(loc="upper right")
    plt.savefig("loss.png")
    plt.clf()


def input_mapping(arg, B, Bt):
    x, y, t = arg
    x_proj = (jnp.pi * x) * B.T
    y_proj = (jnp.pi * y) * B.T
    t_proj = (jnp.pi * t) * Bt.T
    return jnp.concatenate(
        [jnp.array([1]), jnp.sin(x_proj), jnp.cos(x_proj), jnp.sin(y_proj), jnp.cos(y_proj), jnp.sin(t_proj),
         jnp.cos(t_proj)], axis=-1)


def residual(net, x, y, t):
    # Need to implement correctly
    arg = jnp.array([x, y, t])
    du = jax.jacfwd(net)
    ddu = jax.hessian(net)
    eta_t = du(arg)[0][2]
    eta_xx = ddu(arg)[0][0][0]
    mu_x, mu_y, mu_t = du(arg)[1][0], du(arg)[1][1], du(arg)[1][2]
    sigma_x, sigma_y = du(arg)[2][0], du(arg)[2][1]

    eta_res = 0
    mu_res = 0
    sigma_res = 0

    return jnp.square(eta_res) + jnp.square(mu_res) + jnp.square(sigma_res)


def res_weights(net, x, y, t):
    f = jax.vmap(jax.vmap(residual, in_axes=(None, 0, 0, None)), in_axes=(None, None, None, 0))
    Lt = jnp.mean(jnp.square(f(net, x, y, t)), axis=1)
    W = jax.lax.stop_gradient(jnp.exp(- eps * (M @ Lt)))
    return Lt, W


def ic_res(net, x, y):
    # IC's according to the paper.
    N = 0.04 * jnp.sin(y)  # Promote faster dendrite formation
    arg = jnp.array([x, y, 0.])
    eta, mu, sigma = net(arg)[0], net(arg)[1], net(arg)[2]

    eta_res = eta - (0.5 * (1 - jnp.tanh(2 * (x - 20 + N))))
    mu_res = mu - (10 * (x < 20))
    sigma_res = sigma - (-0.225 * (1 - jnp.tanh(2 * (x - 20 + N))))

    return jnp.square(eta_res) + jnp.square(mu_res) + jnp.square(sigma_res)


def ic_loss(net, x, y):
    f = jax.vmap(ic_res, in_axes=(None, 0, 0))
    return jnp.mean(jnp.square(f(net, x, y)))


def bc_res(net, x, y, t):
    du = jax.jacfwd(net)
    etal, phil = net(jnp.array([xmin, y, t]))[0], net(jnp.array([xmin, y, t]))[2],
    etar, mur, phir = net(jnp.array([xmax, y, t]))[0], net(jnp.array([xmax, y, t]))[1], net(jnp.array([xmax, y, t]))[2]
    mul_x = du(jnp.array([xmin, y, t]))[1][0]
    etabot_y, etatop_y = du(jnp.array([x, ymin, t]))[0][1], du(jnp.array([x, ymax, t]))[0][1]
    mubot_y, mutop_y = du(jnp.array([x, ymin, t]))[1][1], du(jnp.array([x, ymax, t]))[1][1]
    sigmabot_y, sigmatop_y = du(jnp.array([x, ymin, t]))[2][1], du(jnp.array([x, ymax, t]))[2][1]

    eta_res = (etal - 1) + (etar - 0) + (etabot_y - 0) + (etatop_y - 0)
    mu_res = (mur - 0) + (mul_x - 0) + (mubot_y - 0) + (mutop_y - 0)
    sigma_res = (phil - phie) + (phir - 0) + (sigmabot_y - 0) + (sigmatop_y - 0)

    return jnp.square(eta_res) + jnp.square(mu_res) + jnp.square(sigma_res)


def bc_loss(net, x, y, t):
    f = jax.vmap(jax.vmap(bc_res, in_axes=(None, 0, 0, None)), in_axes=(None, None, None, 0))
    return jnp.mean(jnp.square(f(net, x, y, t)))
