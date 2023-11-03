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
sigmas = 1000  # 1e7  # Solid phase conductivity
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
    return 30 * eta ** 2 * (eta ** 2 - 1) ** 2


# Barrier function

def g(eta):
    return bar * eta ** 2 * (1 - eta) ** 2


def dg(eta):
    return 2 * bar * eta * (1 - 3 * eta + 2 * (eta * eta))


# Concentrations

def cl(mu):
    return jnp.exp((mu - epsl) / A) / (1 + jnp.exp((mu - epsl) / A))


def dcldmu(mu):
    return jnp.exp((mu + epsl) / A) / (A * (jnp.exp(mu / A) + jnp.exp(epsl / A)) ** 2)


def cs(mu):
    return jnp.exp((mu - epss) / A) / (1 + jnp.exp((mu - epss) / A))


def dcsdmu(mu):
    return jnp.exp((mu + epss) / A) / (A * (jnp.exp(mu / A) + jnp.exp(epss / A)) ** 2)


def ft(mu):
    return cs(mu) * dv - cl(mu)


# Susceptibility

def chi(eta, mu):
    return dcldmu(mu) * (1 - h(eta)) + dcsdmu(mu) * h(eta) * dv


# Diffusion

def D(eta, mu):
    return D0 * (1 - h(eta)) * cl(mu) * (1 - h(eta))


# Conductivity
def sigma(eta):
    return sigmas * h(eta) + sigmal * (1 - h(eta))
