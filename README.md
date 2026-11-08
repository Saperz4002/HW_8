# HW_8
# 1D FDTD Yee Algorithm

Python implementation of the one-dimensional Yee algorithm, developed for Computational Electrodynamics and Nanophotonics at Rice University.

Four modes are available:

- `raw`: Gaussian pulse in vacuum with reflecting boundaries.
- `abc`: Gaussian pulse in vacuum with approximate absorbing boundaries.
- `dielectric`: Reflection and transmission at a dielectric interface.
- `lossy`: Sinusoidal excitation of a dielectric with finite conductivity.

The script animates the normalized electric field and exports snapshots as PDF and PNG files.

## How to run

Install the dependencies:

```bash
pip install numpy matplotlib pillow
