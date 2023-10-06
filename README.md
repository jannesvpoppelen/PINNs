# Phase-field modeling of dendrite nucleation on Li-electrodes using PINNs

This repository contains relevant code for my master's thesis done at the Division of Systems and Control at Uppsala University.

As of now (13-09-2023), the repository contains the following files, which are PINN solvers for the Allen-Cahn equation using DeepXDE, or supplementary files:

- `AC_1Dexample.py` - 1D example of the Allen-Cahn equation, as provided by DeepXDE [1].
- `Allen_Cahn.mat` - Reference data for `AC_1Dexample.py`.
- `AC_1D.py` - 1D Allen-Cahn equation using sequential training and Fourier features.
- `AC_2D.py` - 2D Allen-Cahn equation using sequential training and Fourier features.
- `pdefuncs.py` - Functions appearing in the system of PDEs of the phase-field model.
- `pinnfuncs.py` - Residual and loss functions for the PINN. 
- `PF-Li.py` - First draft at PINN solver for phase-field model.

- ~~`AC.py` - My implementation of the 2D Allen-Cahn equation, trying to recreate the results from the bc-PINN paper [2]. ~~ Removed 06-10-2023
- ~~ `ACJAX.py` - Not fully correct implementation of the 2D Allen-Cahn equation using JAX. (OUTDATED) ~~ Removed 06-10-2023
- ~~ `heateq.py` - Working implementation of the heat equation using JAX. ~~ Removed 06-10-2023

## Allen-Cahn equation

On some spatiotemporal domain $\Omega \times [0,T)$, the Allen-Cahn equation is given by

$$ \frac{\partial u}{\partial t} = k\nabla^2 u - \frac{1}{\varepsilon^2} u( u^2-1)$$

For the phase-field model in Li-ion batteries, $u$ would correspond to the phase-field $\xi$, and the Allen-Cahn equation is obtained by minimizing the free energy functional.

## References

[1] L. Lu, X. Meng, Z. Mao, and G. E. Karniadakis, “DeepXDE: A deep learning library for solving differential equations,” SIAM Review, vol. 63, no. 1, pp. 208–228, 2021.

[2] R. Mattey and S. Ghosh, “A novel sequential method to train physics informed neural networks for Allen Cahn and Cahn Hilliard equations,” Computer Methods in Applied Mechanics and Engineering, vol. 390, p. 114474, 2022.
