#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRAFT: which trace does the mesh follow? Map-view comparison of
  1. the raw hand-drawn KML crossing vertices (what 07 loads),
  2. the 20 m CSI discretize2 resampling of them (the mesh top edge),
  3. the 100 m run07 profile points (hayward_fault_profiles.csv),
  4. the 21-pt rolling median of (3) -- the "smoothed" version consumers use.
Full extent plus a zoom on the creep section where the jitter is visible.
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
from TulePond_filepaths import common_paths
from insar_utils import load_trace, mpl_paper_style
from csi import TriangularPatches

# ------------------------
# Constants (mesh window, notebook 07)
# ------------------------
mesh_lat_lo, mesh_lat_hi = 37.47, 37.65
frame_lon0, frame_lat0 = -121.968, 37.557
zoom_lat = (37.525, 37.565)            # creep section, where jitter shows

# ------------------------
# Load: the four trace versions
# ------------------------
hand_drawn = load_trace(common_paths["fault_trace"])
hand_drawn = hand_drawn[(hand_drawn[:, 1] >= mesh_lat_lo) & (hand_drawn[:, 1] <= mesh_lat_hi)]
hand_drawn = hand_drawn[np.argsort(-hand_drawn[:, 1])]

mesh_edge = TriangularPatches("trace check", lon0=frame_lon0, lat0=frame_lat0, verbose=False)
mesh_edge.trace(hand_drawn[:, 0], hand_drawn[:, 1])
mesh_edge.discretize2(every=0.02)                     # the mesh top edge (20 m)

profiles = pd.read_csv(common_paths["fault_profiles"])
profiles = profiles[(profiles.ft_lat >= mesh_lat_lo) & (profiles.ft_lat <= mesh_lat_hi)]
smooth_lon = profiles.ft_lon.rolling(21, center=True, min_periods=1).median()
smooth_lat = profiles.ft_lat.rolling(21, center=True, min_periods=1).median()

# ------------------------
# Plot: full extent + creep-section zoom
# ------------------------
mpl_paper_style(base=7)
fig, axes = plt.subplots(1, 2, figsize=(7.25, 4.4), constrained_layout=True,
                         gridspec_kw=dict(width_ratios=[1, 1.2]))
for ax, (lat_a, lat_b) in zip(axes, [(mesh_lat_lo, mesh_lat_hi), zoom_lat]):
    ax.plot(mesh_edge.loni, mesh_edge.lati, "-", color="#0072B2", lw=1.0,
            label="mesh top edge (20 m discretize2)")
    ax.plot(profiles.ft_lon, profiles.ft_lat, "-", color="#E69F00", lw=0.8,
            label="run07 profile points (100 m)")
    ax.plot(smooth_lon, smooth_lat, "-", color="#009E73", lw=0.8,
            label="21-pt rolling median (consumers)")
    ax.plot(hand_drawn[:, 0], hand_drawn[:, 1], "o", ms=2.5, mfc="none",
            mec="0.2", mew=0.5, label="hand-drawn KML vertices")
    ax.set_ylim(lat_a, lat_b)
    lon_sel = hand_drawn[(hand_drawn[:, 1] >= lat_a) & (hand_drawn[:, 1] <= lat_b), 0]
    pad = 0.004 if ax is axes[1] else 0.01
    ax.set_xlim(lon_sel.min() - pad, lon_sel.max() + pad)
    ax.set_aspect(1 / np.cos(np.deg2rad(frame_lat0)))
    ax.set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
axes[0].set_title(r"$\bf{A}$  Mesh window", loc="left", fontsize=7)
axes[1].set_title(r"$\bf{B}$  Creep section zoom", loc="left", fontsize=7)
axes[1].legend(fontsize=6, loc="upper right")

fig_file = os.path.join(repo_dir, "Figures_2026-09-25", "trace_versions_check.png")
fig.savefig(fig_file, dpi=300)
print("wrote", fig_file)
