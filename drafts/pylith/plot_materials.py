#!/usr/bin/env python3
"""
Material check figures for the SFCVM grids (sfcvm_grid.py): (a) map of rigidity mu = rho vs^2 at 50 m depth, as is
and aligned, with the mesh trace; (b) cross-section of mu through the Tule Pond section at X-X' (lat 37.5612 N),
as is and aligned; (c) the boundary offset b(km) that the alignment removes.
Grid values are interpolated linearly (as PyLith's SimpleGridDB query_type = linear).
Run in opera_disp_feb13:  python drafts/pylith/plot_materials.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 12, "savefig.facecolor": "white", "figure.facecolor": "white"})
lat_xx = 37.5612

grids = {tag: np.load(os.path.join(HERE, "spatialdb", f"sfcvm_v26-0_{tag}_grid.npz")) for tag in ("asis", "aligned")}
g0 = grids["aligned"]
tx, ty, tkm, normal = g0["trace_x"], g0["trace_y"], g0["trace_km"], g0["trace_normal"]
mu = {tag: (g["density"].astype(float) * g["vs"].astype(float) ** 2 / 1e9) for tag, g in grids.items()}
interp = {tag: RegularGridInterpolator((g["x"], g["y"], g["z"]), mu[tag], method="linear") for tag, g in grids.items()}

# X-X': the trace point nearest lat 37.5612 (tmerc y of that latitude on the trace)
import pyproj
to_mesh = pyproj.Transformer.from_crs("EPSG:4326", str(g0["crs"]), always_xy=True)
_, y_xx = to_mesh.transform(-121.97, lat_xx)
i_xx = int(np.argmin(np.abs(ty - y_xx)))
print(f"X-X': trace point km {tkm[i_xx]:.1f} at ({tx[i_xx]:.0f}, {ty[i_xx]:.0f}) m; boundary offset b = "
      f"{g0['boundary_smooth_m'][i_xx]:.0f} m")

fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
norm = LogNorm(0.03, 30)
xs, ys = np.linspace(-6000, 5000, 441), np.linspace(-7000, 7000, 561)
XS, YS = np.meshgrid(xs, ys, indexing="ij")
for ax, tag, letter in zip(axes[0], ("asis", "aligned"), "ab"):
    vals = interp[tag](np.c_[XS.ravel(), YS.ravel(), np.full(XS.size, -50.0)]).reshape(XS.shape)
    pc = ax.pcolormesh(xs / 1e3, ys / 1e3, vals.T, norm=norm, cmap="cividis", shading="auto")
    ax.plot(tx / 1e3, ty / 1e3, color="#D55E00", lw=1.2, label="mesh fault (mapped trace)")
    ax.axhline(ty[i_xx] / 1e3, color="k", ls=":", lw=1)
    ax.set_aspect("equal"); ax.set_xlabel("x east (km, local tmerc)"); ax.set_ylabel("y north (km)")
    ax.set_title(f"({letter}) mu at 50 m depth, SFCVM v26 {'as is' if tag == 'asis' else 'aligned'}", loc="left")
    ax.legend(loc="lower left", fontsize=10)
fig.colorbar(pc, ax=axes[0], label="Shear modulus mu (GPa)", shrink=0.8)
offs, deps = np.linspace(-3000, 3000, 601), np.linspace(0, 1500, 151)
O, D = np.meshgrid(offs, deps, indexing="ij")
for ax, tag, letter in zip(axes[1], ("asis", "aligned"), "cd"):
    px, py = tx[i_xx] + O * normal[i_xx, 0], ty[i_xx] + O * normal[i_xx, 1]
    vals = interp[tag](np.c_[px.ravel(), py.ravel(), -D.ravel()]).reshape(O.shape)
    pc2 = ax.pcolormesh(offs, deps, vals.T, norm=norm, cmap="cividis", shading="auto")
    ax.axvline(0, color="#D55E00", lw=1.5, label="mapped trace")
    ax.set_ylim(1500, 0); ax.set_xlabel("Distance from the trace, east + (m)"); ax.set_ylabel("Depth (m)")
    ax.set_title(f"({letter}) X-X' (km {tkm[i_xx]:.1f}): {'as is' if tag == 'asis' else 'aligned'}", loc="left")
    ax.legend(loc="lower left", fontsize=10)
fig.colorbar(pc2, ax=axes[1], label="Shear modulus mu (GPa)", shrink=0.8)
fig.savefig(os.path.join(FIG, "fig_materials_sfcvm_v26.png"), dpi=300)

fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
ax.plot(tkm, g0["boundary_raw_m"], ".", color="0.5", label="SFCVM block boundary (22|23 vs 7), raw")
ax.plot(tkm, g0["boundary_smooth_m"], "-", color="#0072B2", lw=2, label="5-point running median (shift applied)")
ax.axhline(0, color="#D55E00", lw=1.5, label="mapped trace")
ax.axvspan(56.2, 62.4, color="#E69F00", alpha=0.2, label="Tule Pond SSE section")
ax.set_xlabel("km from Point Pinole"); ax.set_ylabel("Boundary east of trace (m)")
ax.legend(fontsize=10); ax.set_title("SFCVM v26 Hayward block boundary at the surface relative to the mapped trace", loc="left")
fig.savefig(os.path.join(FIG, "fig_materials_boundary_offset.png"), dpi=300)
print("wrote figures/fig_materials_sfcvm_v26.png, figures/fig_materials_boundary_offset.png")
