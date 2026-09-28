#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRAFT: interim results of the 07 slip-model work, 2026-09-28, from the
products on disk (nothing is re-run). One figure, six panels:
  A  dip sweep on the selected mesh (smooth12, sigma 24): rms by track and Mw
  B  smoothing sigma with the interpolated surface constraint: rms and Mw
  C  fixed-depth scans: current fields (aligned, unaligned, 5 km mesh) and legacy fields
  D  constrained synthetic recovery (batch-2, single-triangle surface link): moment recovered
  E  USGS SFCVM v21.1 rigidity with depth, west and east of the trace
  F  surface slip along strike: crossings vs the line-of-sight and fault-parallel models
Sources: outputs/07_slip_model_tests/*.csv (see interim_logs_2026-09-28.txt).
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo_dir = "/Users/daniellelindsay/Tule_Pond_DISP"
sys.path.insert(0, repo_dir)
from insar_utils import mpl_paper_style

tests_dir = os.path.join(repo_dir, "outputs", "07_slip_model_tests")
fig_file = os.path.join(repo_dir, "Figures_2026-09-28", "interim_07_results.png")
creep_lo_km, creep_hi_km = 56.3, 62.3

# ------------------------
# Load
# ------------------------
dips = pd.read_csv(os.path.join(tests_dir, "dip_on_selected_mesh.csv")).sort_values("dip")
sigmas = pd.read_csv(os.path.join(tests_dir, "sigma_selection.csv"))
sigmas = sigmas[sigmas.family == "sigma_surf"].sort_values("sigma")
scans = {name: pd.read_csv(os.path.join(tests_dir, f"depth_scan{tag}.csv"))
         for name, tag in (("aligned (record)", ""), ("unaligned", "_unaligned"),
                           ("5 km mesh", "_z5"), ("legacy fields", "_legacyfields"))}
recovery = pd.read_csv(os.path.join(tests_dir, "recovery_summary.csv"))
sfcvm = pd.read_csv(os.path.join(tests_dir, "sfcvm_creep_section_profiles.csv"))
sfcvm["side"] = np.where(sfcvm.offset_km < 0, "west", "east")
slip_los = pd.read_csv(os.path.join(tests_dir, "slip_surface_interp_test.csv"))
slip_fp = pd.read_csv(os.path.join(tests_dir, "fp_inversion_surf.csv"))
slip_fp_insar = pd.read_csv(os.path.join(tests_dir, "fp_inversion_insar.csv"))
crossings = pd.read_csv(os.path.join(repo_dir, "outputs", "04_event_detection",
                                     "event_fraction_by_crossing.csv"))
event_cols = ["amp_2017.099", "amp_2019.186", "amp_2021.107", "amp_2023.080", "amp_2024.984"]
crossings["mean_event_mm"] = crossings[event_cols].mean(axis=1)
crossings = crossings[(crossings.km >= 52) & (crossings.km <= 67)]

def surface_median(table, step=0.2, half=0.3):
    surf = table[table.z_top <= 1e-6]
    centres = np.arange(52, 67, step)
    return centres, np.array([np.median(surf.slip_mm[(surf.km_pp >= k - half) & (surf.km_pp <= k + half)])
                              for k in centres])

# ------------------------
# Plot
# ------------------------
mpl_paper_style(base=7)
fig, axes = plt.subplots(2, 3, figsize=(7.25, 5.0), constrained_layout=True)

ax = axes[0, 0]
ax.plot(dips.dip, dips.rms_near_des, "o-", ms=2.5, color="#D55E00", label="desc <200 m")
ax.plot(dips.dip, dips.rms_des, "o--", ms=2.5, color="#D55E00", label="desc full")
ax.plot(dips.dip, dips.rms_near_asc, "o-", ms=2.5, color="#0072B2", label="asc <200 m")
ax.plot(dips.dip, dips.rms_asc, "o--", ms=2.5, color="#0072B2", label="asc full")
ax.axvline(90, color="0.8", lw=0.6)
ax.set_xlabel("dip (deg; >90 = west dip)"); ax.set_ylabel("rms (mm)")
ax.legend(fontsize=5)
twin = ax.twinx(); twin.plot(dips.dip, dips.Mw, "k^-", ms=2.5, lw=0.8); twin.set_ylabel(r"$M_W$ (30 GPa)")
ax.set_title(r"$\bf{A}$  Dip, selected mesh", loc="left", fontsize=7)

ax = axes[0, 1]
ax.plot(sigmas.sigma, sigmas.rms_mm, "o-", ms=2.5, color="#009E73", label="rms")
ax.plot(sigmas.sigma, sigmas.surf_rms_mm, "o-", ms=2.5, color="#E69F00", label="surface-constraint rms")
ax.set_xscale("log"); ax.set_xlabel(r"prior $\sigma$ (mm)"); ax.set_ylabel("rms (mm)")
ax.legend(fontsize=5)
twin = ax.twinx(); twin.plot(sigmas.sigma, sigmas.Mw, "k^-", ms=2.5, lw=0.8); twin.set_ylabel(r"$M_W$ (30 GPa)")
ax.set_title(r"$\bf{B}$  Smoothing, constrained", loc="left", fontsize=7)

ax = axes[0, 2]
colors = {"aligned (record)": "#009E73", "unaligned": "0.6", "5 km mesh": "#8da0cb",
          "legacy fields": "#CC79A7"}
for name, scan in scans.items():
    ax.plot(scan.D_km, scan.dchi2, "o-", ms=2, color=colors[name], label=name)
ax.axhline(4, color="0.5", ls=":", lw=0.7)
ax.set_xscale("log"); ax.set_ylim(-5, 200)
ax.set_xlabel("bottom depth D (km)"); ax.set_ylabel(r"$\Delta\chi^2$")
ax.legend(fontsize=5)
ax.set_title(r"$\bf{C}$  Fixed-depth test", loc="left", fontsize=7)

ax = axes[1, 0]
ax.plot(recovery.D_km, recovery.moment_recovery, "o-", ms=3, color="0.2")
ax.axhline(1, color="0.6", lw=0.6, ls="--")
for d, r in zip(recovery.D_km, recovery.moment_recovery):
    ax.annotate(f"{(2 / 3) * np.log10(r):+.2f}", (d, r), textcoords="offset points",
                xytext=(3, 3), fontsize=5, color="0.35")
ax.set_xlabel("true slip depth D (km)"); ax.set_ylabel("moment recovered / input")
ax.set_title(r"$\bf{D}$  Synthetic recovery ($\Delta M_W$ labelled)", loc="left", fontsize=7)

ax = axes[1, 1]
for side, color in (("west", "#0072B2"), ("east", "#D55E00")):
    med = sfcvm[sfcvm.side == side].groupby("depth_m").mu_gpa.median()
    ax.plot(med.values, med.index / 1000, "o-", ms=2.5, color=color, label=side)
ax.axvline(30, color="0.6", ls="--", lw=0.6)
ax.set_ylim(5, 0); ax.set_xlabel("rigidity (GPa)"); ax.set_ylabel("depth (km)")
ax.legend(fontsize=5)
ax.set_title(r"$\bf{E}$  USGS SFCVM v21.1 rigidity", loc="left", fontsize=7)

ax = axes[1, 2]
ax.axvspan(creep_lo_km, creep_hi_km, color="#0072B2", alpha=0.06, lw=0)
ax.scatter(crossings.km, crossings.mean_event_mm, s=3, c="0.65", linewidths=0,
           label="crossings, mean of 5")
for table, color, label in ((slip_los, "#D55E00", "LOS + constraint"),
                            (slip_fp, "#009E73", "fault-parallel + constraint"),
                            (slip_fp_insar, "0.3", "fault-parallel, InSAR only")):
    km, med = surface_median(table)
    ax.plot(km, med, "-", lw=1.1, color=color, label=label)
ax.set_xlim(52, 67); ax.set_xlabel("km from Point Pinole"); ax.set_ylabel("surface slip (mm)")
ax.legend(fontsize=5)
ax.set_title(r"$\bf{F}$  Surface slip", loc="left", fontsize=7)

fig.savefig(fig_file, dpi=300)
print("wrote", fig_file)
