import jax
import jax.numpy as jnp

# Constants
T = 300  # Temperature (in K)
M = 6.25  # Interfacial mobility
kappa = 0.3  # Interface strength
Mnu = 0.001  # Flux mobility
alpha = 0.5  # Transfer coefficient
frac = 38.69  # nF/RT
bar = 2.4  # Barrier height
epss = -13.8  # potential difference solid phase
epsl = 2.631  # potential difference liquid  phase
c0 = 1.0 / 14.89  # Initial lithium molar fraction
dv = 5.5  # Csm/Clm
D0 = 319.7  # Electrolyte diffusion coefficient
sigmas = 1e7  # Solid phase conductivity
sigmal = 1.19  # Liquid phase conductivity
fac = 0.0074  # Factor used in conductivity equation
phie = -0.45  # Initial overpotential
A = 1  # RT/RT
gamma = 0.22  # Surface tension
delta = 1  # Interrace thickness


# Switching function

def h(eta):
    return eta ** 3 * (6 * eta ** 2 - 15 * eta + 10)


def dh(eta):
    return 30 * eta ** 2 * (eta - 1) ** 2


# Barrier function

def g(eta):
    return bar * eta ** 2 * (1 - eta) ** 2


def dg(eta):
    return 2 * bar * eta * (1 - 3 * eta + 2 * (eta * eta))


# Concentrations

def cl(mu):
    return jnp.exp((mu - epsl)) / (1 + jnp.exp((mu - epsl)))


def dcldmu(mu):
    return jnp.exp((mu + epsl)) / ((jnp.exp(mu) + jnp.exp(epsl)) ** 2)


def cs(mu):
    return jnp.exp((mu - epss)) / (1 + jnp.exp((mu - epss)))


def dcsdmu(mu):
    return jnp.exp((mu + epss)) / ((jnp.exp(mu) + jnp.exp(epss)) ** 2)


def ft(mu):
    return cs(mu) * dv - cl(mu)


# Susceptibility

def chi(eta, mu):
    return dcldmu(mu) * (1 - h(eta)) + dcsdmu(mu) * h(eta) * dv


# Diffusion

def D(eta, mu):
    return D0*cl(mu)*(1-h(eta))


# Conductivity
def sigma(eta):
    return sigmas * h(eta) + sigmal * (1 - h(eta))


@jax.jit
def dD(eta, mu):
    return jax.grad(D, argnums=(0, 1))(eta, mu)


# Initial conditions:

def N(y):
    return 0#.5 * jnp.exp(-(y-.5) ** 2/ (2 * .25**2 ))


def eta0(x, y):
    return 0.5 * (1 - jnp.tanh(2 * (x - 1 - N(y))))


def mu0(x, y):
    return - 10 * (x-N(y) < 1)


def phi0(x, y):
    return phie * (1 - jnp.tanh(2 * (x - 1 - N(y)))) #phie * (1-x/50) + (0.5 * phie * (1 - jnp.tanh(2 * (x - 10 - N(y))))) * x/50
