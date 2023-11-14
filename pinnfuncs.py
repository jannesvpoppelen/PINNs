import jax
import jax.random as jr
import matplotlib.pyplot as plt
from pdefuncs import *
import equinox as eqx

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
# plt.rcParams["figure.figsize"] = (10, 6)

plt.rc('font', size=6)  # controls default text sizes
plt.rc('axes', titlesize=6)  # fontsize of the axes title
plt.rc('axes', labelsize=6)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=4)  # fontsize of the tick labels
plt.rc('ytick', labelsize=4)  # fontsize of the tick labels
plt.rc('legend', fontsize=4)  # legend fontsize

xmin = 0.  # left boundary in micrometer
xmax = 200.  # right boundary in micrometer
ymin = 0.  # bottom boundary in micrometer
ymax = 100.  # top boundary in micrometer
t0 = 0.  # initial time in s
tf = 5.  # final time in s
eps = 1.
icWeight = 1000
bcWeight = 100
stddev = 1
t_stddev = 1
mapping_size = 2
key = jr.PRNGKey(17349)
B = stddev * jr.normal(key, shape=(mapping_size,))


def gendata(xr, yr, tr, Nb, Nt, key):
    key1, key2, key3, key4, key5, key6, _ = jr.split(key, 7)
    xc = jr.permutation(key5, jnp.linspace(xmin, xmax, xr))  # collocation points
    yc = jr.permutation(key6, jnp.linspace(ymin, ymax, yr))  # collocation points
    tc = jnp.linspace(t0, tf, tr)  # collocation points
    xic = jr.uniform(key1, minval=xmin, maxval=xmax, shape=(Nt,))  # initial condition points
    yic = jr.uniform(key2, minval=ymin, maxval=ymax, shape=(Nt // 2,))  # initial condition points
    xbc = jr.uniform(key3, minval=xmin, maxval=xmax, shape=(Nb,))  # boundary points
    ybc = jr.uniform(key4, minval=ymin, maxval=ymax, shape=(Nb,))  # boundary points
    tbc = jnp.linspace(t0, tf, 2 * tr)  # boundary points
    return xc, yc, tc, xic, yic, xbc, ybc, tbc


key = jr.PRNGKey(234879)
xr = 1024
yr = 1024
tr = 5
Nb = 128
Nt = 256

Mt = jnp.triu(jnp.ones((tr, tr)), k=1).T  # Sums the weights of the residual
x_, y_, tc, xic, yic, xbc, ybc, tbc = gendata(xr, yr, tr, Nb, Nt, key)

X, Y = jnp.meshgrid(jnp.linspace(xmin, xmax, 50), jnp.linspace(ymin, ymax, 25))  # Coarse mesh to evaluate residual over
xc = X.flatten()
yc = Y.flatten()

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


def plot_vars(net, t, i):
    xarr = jnp.linspace(xmin, xmax, 256)
    yarr = jnp.linspace(ymin, ymax, 256)
    f = jax.jit(lambda net, x, y: net(x, y, t))
    u_pred = jax.vmap(jax.vmap(f, in_axes=(None, 0, None)), in_axes=(None, None, 0))(net, xarr, yarr)
    eta = u_pred[:, :, 0].reshape(xarr.shape[0], yarr.shape[0])
    mu = u_pred[:, :, 1].reshape(xarr.shape[0], yarr.shape[0])
    phi = u_pred[:, :, 2].reshape(xarr.shape[0], yarr.shape[0])

    # Create a figure with 3 subplots arranged vertically and more vertical spacing
    fig, axs = plt.subplots(3, 1, figsize=(3, 4), sharex=False, gridspec_kw={'hspace': 0.8})
    plt.setp(axs, xticks=[0, 50, 100, 150, 200], yticks=[0, 50, 100])

    # Plot eta
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_title(f" η(t = {t:.2f})")
    im = axs[0].imshow(eta, interpolation='nearest', cmap='jet', extent=[xmin, xmax, ymin, ymax])
    cax = fig.add_axes([axs[0].get_position().x1 + 0.025, axs[0].get_position().y0, 0.02, axs[0].get_position().height])
    plt.colorbar(im, cax=cax)

    # Plot mu
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].set_title(f"μ(t = {t:.2f})")
    im = axs[1].imshow(mu, interpolation='nearest', cmap='jet', extent=[xmin, xmax, ymin, ymax])
    cax = fig.add_axes([axs[1].get_position().x1 + 0.025, axs[1].get_position().y0, 0.02, axs[1].get_position().height])
    plt.colorbar(im, cax=cax)

    # Plot phi
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")
    axs[2].set_title(f"ϕ(t = {t:.2f})")
    im = axs[2].imshow(phi, interpolation='nearest', cmap='jet', extent=[xmin, xmax, ymin, ymax])
    cax = fig.add_axes([axs[2].get_position().x1 + 0.025, axs[2].get_position().y0, 0.02, axs[2].get_position().height])
    plt.colorbar(im, cax=cax)

    # Save the figure
    plt.savefig(f"vars_{i}.png")
    plt.clf()
    return None


def plot_dynamics(net, t0, tf, N):
    for (t, i) in enumerate(jnp.linspace(t0, tf, N)):
        plot_vars(net, i, t)
    return None


def plot_weights(net):
    _, W = res_weights(net, xc, yc, tc)
    plt.xlabel("Weight #")
    plt.ylabel("Weight")
    plt.plot(W)
    plt.savefig("weights.png")
    plt.clf()


def input_mapping(x, y, t):
    w = 1 # 2.0 * jnp.pi
    # k = jnp.arange(1, mapping_size + 1)
    # x_proj = w * x * k
    # y_proj = w * y * k
    x_proj = w * x * B.T
    y_proj = w * y * B.T
    out = jnp.hstack([t, 1, jnp.cos(x_proj), jnp.sin(x_proj), jnp.cos(y_proj), jnp.sin(y_proj)])
    return out


def output_mapping(x, z):
    eta, mu, phi = z[0], z[1], z[2]
    return jnp.array(
        [x * (xmax - x) * eta + (1 - x / xmax), (xmax - x) * mu, x * (xmax - x) * phi + phie * (1 - x / xmax)])


def ic_mapping(x, y, t, z):
    N = 0.04 * jnp.sin(y)
    eta, mu, phi = z[0], z[1], z[2]
    eta0 = 0.5 * (1 - jnp.tanh(2 * (x - 20 + N)))
    mu0 = - 10 * (x < 20)
    phi0 = 0.5 * phie * (1 - jnp.tanh(2 * (x - 20 + N)))
    return jnp.array([t * eta + eta0, t * mu + mu0, t * phi + phi0])


def xavier_init(key, d_in, d_out):
    k1, k2 = jr.split(key)
    glorot_stddev = 1.0 / jnp.sqrt((d_in + d_out) / 2.)
    W = glorot_stddev * jr.normal(k1, (d_in, d_out))
    b = jnp.zeros(d_out)
    return W, b


@eqx.filter_jit
def residual(net, x, y, t):
    eta, mu, phi = net(x, y, t)
    u_x, u_y, u_t = jax.jacrev(net, argnums=0), jax.jacrev(net, argnums=1), jax.jacrev(net, argnums=2)
    u_xx, u_yy = jax.jacfwd(u_x, argnums=0), jax.jacfwd(u_y, argnums=1)

    eta_x, eta_y, eta_t = u_x(x, y, t)[0], u_y(x, y, t)[0], u_t(x, y, t)[0]
    mu_x, mu_y, mu_t = u_x(x, y, t)[1], u_y(x, y, t)[1], u_t(x, y, t)[1]
    phi_x, phi_y = u_x(x, y, t)[2], u_y(x, y, t)[2]
    eta_xx, eta_yy = u_xx(x, y, t)[0], u_yy(x, y, t)[0]
    mu_xx, mu_yy = u_xx(x, y, t)[1], u_yy(x, y, t)[1]
    phi_xx, phi_yy = u_xx(x, y, t)[2], u_yy(x, y, t)[2]

    # ETA
    eta_res = eta_t - (M * (kappa * (eta_xx + eta_yy) * 0 - dg(eta))
                       - Mnu * dh(eta) * (jnp.exp(frac * phi * (1 - alpha)) - cl(mu) / c0 * (
                    1 - h(eta) * jnp.exp(-alpha * frac * phi)))
                       )

    # MU
    dD = jax.grad(D, argnums=(0, 1))(eta, mu)
    gradD = jnp.array([dD[0] * eta_x + dD[1] * mu_x, dD[0] * eta_y + dD[1] * mu_y])
    gradmuphi = jnp.array([mu_x + frac * phi_x, mu_y + frac * phi_y])

    mu_res = chi(eta, mu) * mu_t - (
            D(eta, mu) * ((mu_xx + mu_yy) + frac * (phi_xx + phi_yy))
            + jnp.dot(gradD, gradmuphi)
            - ft(mu) * dh(eta) * eta_t
    )

    # PHI
    dsigma = (sigmas - sigmal) * dh(eta) * jnp.array([eta_x, eta_y])
    gradphi = jnp.array([phi_x, phi_y])
    phi_res = (jnp.dot(dsigma, gradphi) + sigma(eta) * (phi_xx + phi_yy)) - fac * eta_t

    return abs(eta_res) + abs(mu_res) + abs(phi_res)


@eqx.filter_jit
def res_weights(net, x, y, t):
    f = jax.vmap(jax.vmap(residual, in_axes=(None, 0, 0, None)), in_axes=(None, None, None, 0))
    Lt = jnp.mean(jnp.square(f(net, x, y, t)), axis=1)
    W = jax.lax.stop_gradient(jnp.exp(- eps * (Mt @ Lt)))
    return Lt, W


@eqx.filter_jit
def ic_res(net, x, y):
    # IC's according to the paper.
    N = 0.04 * jnp.sin(y)  # Promote faster dendrite formation
    eta, mu, phi = net(x, y, 0.)
    eta0 = (0.5 * (1 - jnp.tanh(2 * (x - 20 + N))))
    mu0 = (-10 * (x < 20))
    phi0 = (phie / 2 * (1 - jnp.tanh(2 * (x - 20 + N))))

    eta_res = eta - eta0
    mu_res = mu - mu0
    phi_res = phi - phi0

    return abs(eta_res) + abs(mu_res) + abs(phi_res)


@eqx.filter_jit
def ic_loss(net, x, y):
    f = jax.vmap(jax.vmap(ic_res, in_axes=(None, 0, None)), in_axes=(None, None, 0))
    return jnp.mean(jnp.square(f(net, x, y)))


@eqx.filter_jit
def bc_res(net, x, y, t):
    leftnet = net(xmin, y, t)
    rightnet = net(xmax, y, t)
    # net_x, net_y = jax.jacrev(net, argnums=0), jax.jacrev(net, argnums=1)
    # leftnet_x = net_x(xmin, y, t)
    # botnet_y = net_y(x, ymin, t)
    # topnet_y = net_y(x, ymax, t)
    etal, phil = leftnet[0], leftnet[2]
    etar, mur, phir = rightnet[0], rightnet[1], rightnet[2]
    # mul_x = leftnet_x[1]
    # etabot_y, etatop_y = botnet_y[0], topnet_y[0]
    # mubot_y, mutop_y = botnet_y[1], topnet_y[1]
    # phibot_y, phitop_y = botnet_y[2], topnet_y[2]

    eta_res = (etal - 1) + (etar - 0)  # + (etabot_y - 0) + (etatop_y - 0)
    mu_res = (mur - 0)  # + (mul_x - 0) + (mubot_y - 0) + (mutop_y - 0)
    phi_res = (phil - phie) + (phir - 0)  # + (phibot_y - 0) + (phitop_y - 0)

    return abs(eta_res) + abs(mu_res) + abs(phi_res)


@eqx.filter_jit
def bc_loss(net, x, y, t):
    f = jax.vmap(jax.vmap(bc_res, in_axes=(None, 0, 0, None)), in_axes=(None, None, None, 0))
    return jnp.mean(jnp.square(f(net, x, y, t)))


#@eqx.filter_jit
def loss(net, key):
    key, key1, key2 = jr.split(key, 3)
    # key, key1, key2, key3, key4, key5, key6 = jr.split(key, 7)
    batchsize = 8192
    xc2 = jr.uniform(key1, minval=xmin, maxval=xmax, shape=(batchsize,))
    yc2 = jr.uniform(key2, minval=ymin, maxval=ymax, shape=(batchsize,))
    plt.scatter(xc2, yc2, s=5)
    plt.show()
    loss_ic = ic_loss(net, xic, yic)
    loss_bc = bc_loss(net, xbc, ybc, tbc)
    Lt, W = res_weights(net, xc2, yc2, tc)
    loss = jnp.mean(W * Lt) + icWeight * loss_ic + bcWeight * loss_bc
    return loss