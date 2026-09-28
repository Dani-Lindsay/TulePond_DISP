#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRAFT (2026-09-28): what the smoothing choice looks like. Reads
sigma_lcurve.csv / sigma_lcurve_slip.npz (drafts/07/sigma_lcurve.py).

A  L-curve: full weighted misfit sqrt(chi2 InSAR + chi2 surface) vs the
   prior-weighted roughness sqrt(m^T C_corr^-1 m) (log-log; the two terms the
   solver trades off),
   corner = maximum curvature (cubic spline in log sigma)
B  curvature vs sigma
C  Mw (30 GPa), peak slip and surface % vs sigma
D-G slip distribution for four sigmas on one colour scale
H  surface-row slip along strike vs the crossing stacks
I  slip vs depth: median over the creep section
Output: Figures_<fig_date>/test_sigma_lcurve.png
Run from the repo root.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from TulePond_filepaths import common_paths
from insar_utils import mpl_paper_style

tests_dir = os.path.join(common_paths["out_dir"], "07_slip_model_tests")
fig_dir = common_paths["fig_dir"]
creep_lo_km, creep_hi_km = 56.3, 62.3
s0_km_pp = 46.4                          # km from Point Pinole at the north end of the mesh

# ------------------------
# Load
# ------------------------
family = pd.read_csv(os.path.join(tests_dir, "sigma_lcurve.csv"))
slip_file = np.load(os.path.join(tests_dir, "sigma_lcurve_slip.npz"))
nodes, faces = slip_file["nodes_sz"], slip_file["faces"]
km_pp, z_top, z_c = slip_file["km_pp"], slip_file["z_top"], slip_file["z_c"]

# ------------------------
# Calculate: the L-curve corner
# ------------------------
log_sigma = np.log10(family.sigma.values)
misfit = np.sqrt(family.chi2_total.values)
roughness = family.prior_norm.values
log_misfit = np.log10(misfit)
log_norm = np.log10(roughness)
spline_misfit = CubicSpline(log_sigma, log_misfit)
spline_norm = CubicSpline(log_sigma, log_norm)
fine_log_sigma = np.linspace(log_sigma[0], log_sigma[-1], 400)
dx, ddx = spline_misfit(fine_log_sigma, 1), spline_misfit(fine_log_sigma, 2)
dy, ddy = spline_norm(fine_log_sigma, 1), spline_norm(fine_log_sigma, 2)
curvature = (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5
inner = (fine_log_sigma > log_sigma[1]) & (fine_log_sigma < log_sigma[-2])   # away from the spline ends
corner_sigma = 10 ** fine_log_sigma[inner][np.argmax(np.abs(curvature[inner]))]
nearest_grid = family.sigma.values[np.argmin(np.abs(log_sigma - np.log10(corner_sigma)))]
print(f"L-curve corner: sigma {corner_sigma:.1f} (nearest grid value {nearest_grid:g})")
print(family.round(3).to_string(index=False))

shown = sorted({8, int(nearest_grid), 48, 128})
shown = [s for s in dict.fromkeys(shown) if s in family.sigma.values]
colours = dict(zip(shown, ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]))

# ------------------------
# Plot
# ------------------------
mpl_paper_style(base=7)
fig = plt.figure(figsize=(7.25, 9.6), constrained_layout=True)
grid = fig.add_gridspec(6, 3, height_ratios=[1.1, 0.75, 0.75, 0.75, 0.75, 1.1])

ax = fig.add_subplot(grid[0, 0])
ax.plot(misfit, roughness, "o-", color="0.4", ms=3, lw=0.8)
for sigma, x_val, y_val in zip(family.sigma, misfit, roughness):
    ax.annotate(f"{sigma:g}", (x_val, y_val), xytext=(3, 2), textcoords="offset points", fontsize=5)
for sigma in shown:
    pick = (family.sigma == sigma).values
    ax.plot(misfit[pick], roughness[pick], "o", color=colours[sigma], ms=5)
ax.plot(10 ** spline_misfit(np.log10(corner_sigma)), 10 ** spline_norm(np.log10(corner_sigma)),
        "k*", ms=8, label=f"corner, sigma {corner_sigma:.0f}")
ax.set_xscale("log"); ax.set_yscale("log")
from matplotlib.ticker import LogLocator, ScalarFormatter, NullFormatter
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 3, 5, 7)))
    axis.set_major_formatter(ScalarFormatter()); axis.set_minor_formatter(NullFormatter())
ax.set_xlabel(r"Weighted misfit $\sqrt{\chi^2}$ (InSAR + surface)"); ax.set_ylabel(r"Roughness $\sqrt{m^T C^{-1} m}$")
ax.legend(fontsize=5, loc="upper right")
ax.set_title(r"$\bf{A}$  L-curve", loc="left", fontsize=7)

ax = fig.add_subplot(grid[0, 1])
ax.plot(10 ** fine_log_sigma, np.abs(curvature), "k-", lw=0.8)
ax.axvline(corner_sigma, color="k", ls=":", lw=0.7)
ax.set_xscale("log"); ax.set_xlabel("sigma (mm)"); ax.set_ylabel("|curvature|")
ax.set_title(r"$\bf{B}$  L-curve curvature", loc="left", fontsize=7)

ax = fig.add_subplot(grid[0, 2])
ax.plot(family.sigma, family.Mw, "ko-", ms=3, lw=0.8)
ax.axvline(corner_sigma, color="k", ls=":", lw=0.7)
ax.set_xscale("log"); ax.set_xlabel("sigma (mm)"); ax.set_ylabel("Mw (30 GPa)")
twin = ax.twinx()
twin.plot(family.sigma, family.peak_slip_mm, "s--", color="#D55E00", ms=2.5, lw=0.8)
twin.set_ylabel("Peak slip (mm)", color="#D55E00")
ax.set_title(r"$\bf{C}$  Mw and peak slip", loc="left", fontsize=7)

vmax = float(np.ceil(np.max(slip_file["slip_mm"][[list(slip_file["sigma"]).index(s) for s in shown]])))
slip_axes = [fig.add_subplot(grid[1 + panel, :]) for panel in range(len(shown))]
letters = "DEFG"
for ax, sigma, letter in zip(slip_axes, shown, letters):
    row = family[family.sigma == sigma].iloc[0]
    slip = slip_file["slip_mm"][list(slip_file["sigma"]).index(sigma)]
    pc = ax.tripcolor(s0_km_pp + nodes[:, 0], nodes[:, 1], faces, facecolors=slip,
                      cmap="viridis", vmin=0, vmax=vmax, edgecolors="none")
    ax.set_xlim(54, 65); ax.set_ylim(2.0, 0)
    ax.set_ylabel("Depth (km)")
    ax.set_title(r"$\bf{" + letter + r"}$  " + f"sigma {sigma:g}: Mw {row.Mw:.2f}, peak {row.peak_slip_mm:.1f} mm, "
                 f"rms {row.rms_mm:.3f} mm, surface {row.surface_pct:.0f}%",
                 loc="left", fontsize=7, color=colours[sigma])
    fig.colorbar(pc, ax=ax, label="Slip (mm)", shrink=0.9, pad=0.01)
slip_axes[-1].set_xlabel("Along strike (km from Point Pinole, arc length)")

ax = fig.add_subplot(grid[5, :2])
ax.errorbar(slip_file["crossing_km"], slip_file["crossing_mm"], yerr=slip_file["crossing_err"],
            fmt="o", color="0.6", ms=2, lw=0.5, label="crossing stacks")
surface_row = z_top <= 1e-6
order = np.argsort(km_pp[surface_row])
for sigma in shown:
    slip = slip_file["slip_mm"][list(slip_file["sigma"]).index(sigma)]
    ax.plot(km_pp[surface_row][order], slip[surface_row][order], "-", color=colours[sigma], lw=0.9,
            label=f"sigma {sigma:g}")
ax.set_xlim(54, 65)
ax.set_xlabel("Distance from Point Pinole (km)"); ax.set_ylabel("Surface slip (mm)")
ax.legend(fontsize=5, ncol=3)
ax.set_title(r"$\bf{H}$  Surface slip", loc="left", fontsize=7)

ax = fig.add_subplot(grid[5, 2])
in_creep = (km_pp >= creep_lo_km) & (km_pp <= creep_hi_km)
depth_bins = np.round(z_c[in_creep], 2)
for sigma in shown:
    slip = slip_file["slip_mm"][list(slip_file["sigma"]).index(sigma)]
    profile = pd.Series(slip[in_creep]).groupby(observed=False, by=pd.cut(z_c[in_creep], np.r_[0, 0.1, 0.2, 0.3, 0.45, 0.65, 0.9, 1.2, 1.6, 2.1, 3.0])).median()
    centres = np.array([interval.mid for interval in profile.index])
    ax.plot(profile.values, centres, "o-", color=colours[sigma], ms=2.5, lw=0.9)
ax.set_ylim(3.0, 0); ax.set_xlabel("Median slip (mm)"); ax.set_ylabel("Depth (km)")
ax.set_title(r"$\bf{I}$  Slip vs depth, km 56.3-62.3", loc="left", fontsize=7)

out_file = os.path.join(fig_dir, "test_sigma_lcurve.png")
fig.savefig(out_file, dpi=250)
print("wrote", out_file)
