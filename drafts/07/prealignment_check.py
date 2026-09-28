#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRAFT: the pre-alignment, shown. Per track: the mean-event field in the
inversion window as loaded (original), the same field after the far-field
plane removal (slip section creep +/- 1 km masked for the fit), and the
along-strike medians before/after -- north and south should sit at zero
after. Mirrors the cells in 07_slip_model exactly.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

repo_dir = "/Users/daniellelindsay/Tule_Pond_DISP"
sys.path.insert(0, repo_dir)
from TulePond_filepaths import common_paths
from insar_utils import load_trace, mpl_paper_style

# ------------------------
# Constants (07_slip_model values)
# ------------------------
lat0, lon0     = 37.557, -121.968
km_per_deg_lat = 110.57
km_per_deg_lon = 111.32 * np.cos(np.deg2rad(lat0))
swath_km       = 4.0
exclude_km     = 0.03
km_window      = (52.0, 67.0)
creep_lo_km, creep_hi_km = 56.3, 62.3
align_mask_km  = 1.0
bin_km         = 0.5                    # along-strike median bins for the profile

# ------------------------
# Load: trace frame and lat -> km map
# ------------------------
trace = load_trace(common_paths["fault_trace"])
trace = trace[(trace[:, 1] >= 37.47) & (trace[:, 1] <= 37.65)]
trace_x = (trace[:, 0] - lon0) * km_per_deg_lon
trace_y = (trace[:, 1] - lat0) * km_per_deg_lat
trace_tree = cKDTree(np.c_[trace_x, trace_y])

profiles = pd.read_csv(common_paths["fault_profiles"])
profile_km = (len(profiles) - 1 - np.arange(len(profiles))) * 0.1
def lat_to_km(lats):
    lats = np.asarray(lats, float)
    out = np.empty(lats.size)
    for lo in range(0, lats.size, 20000):
        chunk = lats[lo:lo + 20000]
        out[lo:lo + 20000] = profile_km[np.abs(profiles.ft_lat.values[None, :]
                                               - chunk[:, None]).argmin(axis=1)]
    return out

# ------------------------
# Calculate: per track, window cut then far-field plane removal
# ------------------------
tracks = {}
for track in ("ascending", "descending"):
    z = np.load(os.path.join(repo_dir, "outputs", "06_event_field_geometry",
                             f"csi_mean_event_{track}.npz"))
    x = (z["lon"] - lon0) * km_per_deg_lon
    y = (z["lat"] - lat0) * km_per_deg_lat
    vel = z["vel_mm"].astype(float)
    dist, _ = trace_tree.query(np.c_[x, y])
    km_pp = lat_to_km(z["lat"])
    keep = (np.isfinite(vel) & (dist <= swath_km) & (dist >= exclude_km)
            & (km_pp >= km_window[0]) & (km_pp <= km_window[1]))
    x, y, vel, km_pp = x[keep], y[keep], vel[keep], km_pp[keep]
    lon_k, lat_k = z["lon"][keep], z["lat"][keep]

    far = (km_pp < creep_lo_km - align_mask_km) | (km_pp > creep_hi_km + align_mask_km)
    plane = np.c_[np.ones(x.size), x, y]
    plane_coef = np.linalg.lstsq(plane[far], vel[far], rcond=None)[0]
    corrected = vel - np.dot(plane, plane_coef)
    tracks[track] = dict(lon=lon_k, lat=lat_k, km=km_pp, vel=vel,
                         corrected=corrected, coef=plane_coef, far=far)
    north = far & (km_pp < creep_lo_km)
    south = far & (km_pp > creep_hi_km)
    print(f"[{track}] plane {plane_coef[0]:+.2f} mm, {plane_coef[1]:+.3f}, "
          f"{plane_coef[2]:+.3f} mm/km | north {np.median(vel[north]):+.2f} -> "
          f"{np.median(corrected[north]):+.2f} mm | south {np.median(vel[south]):+.2f} -> "
          f"{np.median(corrected[south]):+.2f} mm")

# ------------------------
# Plot: rows = tracks; cols = original map, corrected map, medians profile
# ------------------------
mpl_paper_style(base=7)
fig, axes = plt.subplots(2, 3, figsize=(7.25, 5.2), constrained_layout=True,
                         gridspec_kw=dict(width_ratios=[1, 1, 1.35]))
letters = "ABCDEF"
for r, (track, d) in enumerate(tracks.items()):
    for c, (values, title) in enumerate(((d["vel"], "original"),
                                         (d["corrected"], "corrected"))):
        ax = axes[r, c]
        sc = ax.scatter(d["lon"], d["lat"], c=values, s=1.5, cmap="RdBu_r",
                        vmin=-3, vmax=3, linewidths=0)
        ax.plot(trace[:, 0], trace[:, 1], "k-", lw=0.5)
        ax.set_aspect(1 / np.cos(np.deg2rad(lat0)))
        ax.set_xlim(-122.10, -121.86); ax.set_ylim(37.47, 37.65)
        ax.set_title(r"$\bf{" + letters[r * 3 + c] + r"}$  "
                     + f"{track[:4]} {title}", loc="left", fontsize=7)
        if c:
            ax.set_yticklabels([])
        if r == 0:
            ax.set_xticklabels([])
    ax = axes[r, 2]
    bins = np.arange(km_window[0], km_window[1] + bin_km, bin_km)
    centres = 0.5 * (bins[1:] + bins[:-1])
    med_orig = [np.median(d["vel"][(d["km"] >= a) & (d["km"] < b)]) for a, b in zip(bins, bins[1:])]
    med_corr = [np.median(d["corrected"][(d["km"] >= a) & (d["km"] < b)]) for a, b in zip(bins, bins[1:])]
    ax.axvspan(creep_lo_km, creep_hi_km, color="#0072B2", alpha=0.08, lw=0)
    ax.axhline(0, color="0.75", lw=0.6)
    ax.plot(centres, med_orig, "o-", ms=2.5, color="0.5", label="original")
    ax.plot(centres, med_corr, "o-", ms=2.5, color="#D55E00", label="corrected")
    ax.set_xlim(km_window)
    ax.set_ylim(-1.6, 1.6)
    ax.set_ylabel("median LOS (mm)")
    ax.set_title(r"$\bf{" + letters[r * 3 + 2] + r"}$  "
                 + f"{track[:4]} along-strike medians", loc="left", fontsize=7)
    if r == 0:
        ax.legend(fontsize=6)
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Distance from Point Pinole (km)")
axes[1, 0].set_xlabel("Longitude")
axes[0, 0].set_ylabel("Latitude")
axes[1, 0].set_ylabel("Latitude")
fig.colorbar(sc, ax=axes[:, :2], shrink=0.5, label="mean event LOS (mm)",
             orientation="horizontal", pad=0.03)

fig_file = os.path.join(repo_dir, "Figures_2026-09-25", "draft_prealignment_check.png")
fig.savefig(fig_file, dpi=300)
print("wrote", fig_file)
