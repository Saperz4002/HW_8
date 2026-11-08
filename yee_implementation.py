"""Four 1D Yee examples."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# 1. Choose the analysis and grid.
mode = "dielectric"                 # "raw", "abc", "dielectric", "lossy"
assert mode in ("raw", "abc", "dielectric", "lossy")
ke, nsteps, S = 200, 1400, 0.5
kc, source, t0, spread = ke // 2, 25, 40, 10
c0, eps0, dx, frequency = 299792458., 8.8541878128e-12, 0.01, 700e6
dt = S * dx / c0

# 2. Material and normalized fields
eps, sigma = np.ones(ke), np.zeros(ke)
if mode in ("dielectric", "lossy"):
    eps[kc:] = 16.0
if mode == "lossy":
    sigma[kc:] = 0.04
ex, hy = np.zeros(ke), np.zeros(ke)
E, H = [ex.copy()], [hy.copy()]
a = sigma * dt / (2 * eps0 * eps)
ca, cb = (1 - a) / (1 + a), S / (eps * (1 + a))

# Delay ABCs: one-cell travel time; only integer delays supported here.
delays = np.sqrt(eps[[0, -1]]) / S
assert np.allclose(delays, np.rint(delays)), "Use integer ABC delays."
left, right = np.rint(delays).astype(int)

# 3. Update E, apply the source/boundaries, then update H.
for n in range(1, nsteps + 1):
    ex[1:] = ca[1:] * ex[1:] + cb[1:] * (hy[:-1] - hy[1:])
    pulse = (np.sin(2*np.pi*frequency*(n-0.5)*dt) if mode == "lossy"
             else np.exp(-0.5*((n-t0)/spread)**2))
    if mode in ("raw", "abc"):
        ex[source] = pulse          # Hard Gaussian source
    else:
        ex[source] += pulse         # Soft Gaussian/sinusoidal source
    if mode != "raw":               # Approximate ABCs; raw reflects
        if n >= left: ex[0] = E[n-left][1]
        if n >= right: ex[-1] = E[n-right][-2]
    hy[:-1] += S * (ex[:-1] - ex[1:])
    E.append(ex.copy())
    H.append(hy.copy())

# Save the simulation data for future post-processing.
E, H = np.asarray(E), np.asarray(H)
steps = np.arange(nsteps + 1)

folder = Path.cwd() / "outputs"
folder.mkdir(parents=True, exist_ok=True)

output_data = folder / f"yee_{mode}_fields.npz"
np.savez_compressed(
    output_data,
    E_tilde=E,                       # E normalization; shape: (time, position)
    H=H,                             # Magnetic field [A/m]
    z_E=np.arange(ke) * dx,           # Electric-field positions [m]
    z_H=(np.arange(ke) + 0.5) * dx,   # Magnetic-field positions [m]
    t_E=(steps - 0.5) * dt,           # Electric-field times [s]
    t_H=steps * dt,                   # Magnetic-field times [s]
    eps_r=eps,
    sigma=sigma,                     # Conductivity [S/m]
    dx=dx,
    dt=dt,
    S=S,
    c0=c0,
    eps0=eps0,
    mode=mode,
    source_index=source,
    interface_index=kc,
    source_frequency=frequency,
    gaussian_center_step=t0,
    gaussian_spread_steps=spread,
    source_type="hard" if mode in ("raw", "abc") else "soft",
    boundary_type="reflecting" if mode == "raw" else "approximate_delay"
)
print(f"Simulation data saved to: {output_data}")

# 4. Animate only E; skip frames without skipping simulation steps.
fig, ax = plt.subplots(figsize=(8, 4))
line, = ax.plot(np.arange(ke), E[0], label=r'$\widetilde E_x$')
ax.axvline(source, color='gray', ls=':', label='Source')
if mode in ("dielectric", "lossy"):
    ax.axvspan(kc, ke-1, color='teal', alpha=0.12,
               label=f'eps_r=16, sigma={sigma[-1]:g} S/m')
limit = max(1.2, 1.1*np.max(np.abs(E)))
ax.set(xlim=(0, ke-1), ylim=(-limit, limit),
       xlabel=r'Position / $\Delta x$', ylabel=r'$\widetilde E_x$')
ax.legend(loc='upper right', fontsize=8)
ax.grid(alpha=0.2)
title = ax.set_title('')
fig.subplots_adjust(top=0.87, bottom=0.15)
def update(n):
    line.set_ydata(E[n])
    title.set_text(f'{mode} | Step {n}/{nsteps} | E time: {(n-0.5)*dt*1e9:.2f} ns')
    return line, title
frames = list(range(0, nsteps+1, 4))
if frames[-1] != nsteps: frames.append(nsteps)
movie = FuncAnimation(fig, update, frames=frames, interval=40)
output = Path.cwd() / f'yee_{mode}.gif'
movie.save(str(output), writer=PillowWriter(fps=25))
print(f'Saved: {output}')
plt.show()
