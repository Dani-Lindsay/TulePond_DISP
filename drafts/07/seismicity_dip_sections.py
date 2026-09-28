#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRAFT (not in the repo yet): is there any evidence in the seismicity for a
dipping fault at Tule Pond?

Science-supplement layout: 2 rows x 4 columns, one row per catalogue
(growclust top, hypoDD bottom), each row = map + sections for boxes 1/2/3.
Box 2 is exactly the creep-event section (registry 04_creep_length_lo_km to
04_creep_length_hi_km, km from Point Pinole); box 1 the section to the
south, box 3 to the north.

Relocated events only: Bo's growclust catalogue with the unrelocated
(nbranch = 1) events dropped, and the Waldhauser & Schaff hypoDD catalogue
NCAeqDD.v202112.1 (relocated throughout).  Signed cross-fault distance
follows csi_io.dist_to_trace (+ east of the trace).  Reference lines at
90/80/70 degree east dip.
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

repo_dir = "/Users/daniellelindsay/Tule_Pond_DISP"
sys.path.insert(0, repo_dir)
from TulePond_filepaths import common_paths, inputs_dir

# ------------------------
# Constants
# ------------------------
lat_lo, lat_hi = 37.47, 37.72         # section band (north edge past the mesh window)
lat0, lon0     = 37.55, -121.968       # local frame centre (csi_io convention)
km_per_deg_lat = 110.57
km_per_deg_lon = 111.32 * np.cos(np.deg2rad(lat0))
max_normal_km  = 4.0                   # cross-fault half-width of each swath box
max_depth_km   = 12.0
mesh_bottom_km = 3.0                   # tri-mesh depth extent in m02
dips_deg       = [90, 80, 70]          # east dips to draw through the trace
trace_step_km  = 0.1                   # spacing of the run07 profile points

fig_dir = os.path.join(repo_dir, "Figures_2026-09-25")
os.makedirs(fig_dir, exist_ok=True)

# ------------------------
# Load: creep-event section limits from the registry (km from Point Pinole)
# ------------------------
registry = pd.read_csv(common_paths["registry_csv"])
creep_lo_km = float(registry[registry.key == "04_creep_length_lo_km"].value.iloc[-1])
creep_hi_km = float(registry[registry.key == "04_creep_length_hi_km"].value.iloc[-1])
print(f"creep-event section: km {creep_lo_km:.1f} to {creep_hi_km:.1f} from Point Pinole")

# ------------------------
# Load: fault trace (smoothed 100 m profile points, run07 product)
# ------------------------
trace = pd.read_csv(common_paths["fault_profiles"])
trace_x = (trace.ft_lon.values - lon0) * km_per_deg_lon
trace_y = (trace.ft_lat.values - lat0) * km_per_deg_lat
trace_tree = cKDTree(np.column_stack([trace_x, trace_y]))
# km from Point Pinole along the trace, run15.lat_to_km_map convention
# (file is ordered south to north; Point Pinole is the north end)
trace_km_pp = (len(trace) - 1 - np.arange(len(trace))) * trace_step_km

# ------------------------
# Load: Bo's growclust catalogue, relocated events only
# ------------------------
bo = pd.read_csv(os.path.join(inputs_dir, "Bo_growclust",
                              "hayward_growclust_catalog_relocated_plus_unrelocated.csv"))
n_all = len(bo)
bo = bo[bo.nbranch > 1].copy()
print(f"growclust: {len(bo)} relocated of {n_all} (nbranch = 1 dropped)")
bo = bo[(bo.latitude >= lat_lo) & (bo.latitude <= lat_hi)]

# ------------------------
# Load: hypoDD catalogue (whitespace columns; awk prefilter keeps the read light)
# ------------------------
dd_file = os.path.join(inputs_dir, "DD_catalog", "NCAeqDD.v202112.1")
awk = ("awk '!/[A-Za-z]/ && NF>=15 && $7>=%f && $7<=%f && $8>=-122.25 && $8<=-121.75'"
       % (lat_lo, lat_hi))
raw = subprocess.run(awk, shell=True, stdin=open(dd_file), capture_output=True, text=True).stdout
dd_cols = ["yr", "mon", "day", "hr", "mn", "sec", "latitude", "longitude", "depth_km",
           "eh1_km", "eh2_km", "az", "ev_km", "magnitude", "evid"]
dd = pd.read_csv(pd.io.common.StringIO(raw), sep=r"\s+", names=dd_cols)

# ------------------------
# Calculate: signed cross-fault distance (+ east) and km from Point Pinole
# ------------------------
for cat in (bo, dd):
    x = (cat.longitude.values - lon0) * km_per_deg_lon
    y = (cat.latitude.values - lat0) * km_per_deg_lat
    dist_km, nearest = trace_tree.query(np.column_stack([x, y]))
    cat["normal_km"] = dist_km * np.sign(x - trace_x[nearest])
    cat["km_pp"] = trace_km_pp[nearest]

# boxes in km from Point Pinole (increases SOUTHWARD): box 3 north of the
# creep section, box 2 the creep section, box 1 south of it
in_band = (trace.ft_lat >= lat_lo) & (trace.ft_lat <= lat_hi)
band_km_lo = trace_km_pp[in_band].min()          # north edge (lat_hi)
band_km_hi = trace_km_pp[in_band].max()          # south edge (lat_lo)
boxes_km_pp = [(creep_hi_km, band_km_hi),        # 1: south
               (creep_lo_km, creep_hi_km),       # 2: creep-event section
               (band_km_lo, creep_lo_km)]        # 3: north
box_names = ["1", "2 (creep section)", "3"]
for name, (km_a, km_b) in zip(box_names, boxes_km_pp):
    print(f"box {name}: km {km_a:.1f} to {km_b:.1f} ({km_b - km_a:.1f} km long)")

# ------------------------
# Plot: 2 rows x 4 columns -- one row per catalogue, each row a map plus
# the box 1/2/3 sections at true scale (Science supplement style)
# ------------------------
from insar_utils import mpl_paper_style
mpl_paper_style(base=7)

catalogues = [("growclust (Bo)", bo, "#0072B2"),
              ("hypoDD (Waldhauser & Schaff)", dd, "#D55E00")]
panel_letters = ["ABCD", "EFGH"]

fig = plt.figure(figsize=(7.25, 5.6), constrained_layout=True)
gs = fig.add_gridspec(2, 4, width_ratios=[1.15, 1, 1, 1])

for row, (cat_label, cat, color) in enumerate(catalogues):
    letters = panel_letters[row]

    # -- map panel
    ax_map = fig.add_subplot(gs[row, 0])
    ax_map.plot(trace.ft_lon, trace.ft_lat, "-", color="0.2", lw=0.8)
    ax_map.scatter(cat.longitude, cat.latitude, s=1.2, c=color, linewidths=0, alpha=0.5)
    for name, (km_a, km_b) in zip(box_names, boxes_km_pp):
        in_box = in_band.values & (trace_km_pp >= km_a) & (trace_km_pp <= km_b)
        seg_x, seg_y = trace_x[in_box], trace_y[in_box]
        strike_vec = np.array([seg_x[-1] - seg_x[0], seg_y[-1] - seg_y[0]])
        strike_vec = strike_vec / np.sqrt(np.dot(strike_vec, strike_vec))
        normal_vec = np.array([-strike_vec[1], strike_vec[0]])
        if normal_vec[0] < 0:
            normal_vec = -normal_vec                   # + east, matching the sections
        corners_xy = np.array([
            [seg_x[0], seg_y[0]] + max_normal_km * normal_vec,
            [seg_x[-1], seg_y[-1]] + max_normal_km * normal_vec,
            [seg_x[-1], seg_y[-1]] - max_normal_km * normal_vec,
            [seg_x[0], seg_y[0]] - max_normal_km * normal_vec,
            [seg_x[0], seg_y[0]] + max_normal_km * normal_vec])
        ax_map.plot(corners_xy[:, 0] / km_per_deg_lon + lon0,
                    corners_xy[:, 1] / km_per_deg_lat + lat0, "-", color="0.35", lw=0.6)
        mid = corners_xy[:4].mean(axis=0) - max_normal_km * 1.45 * normal_vec
        ax_map.text(mid[0] / km_per_deg_lon + lon0, mid[1] / km_per_deg_lat + lat0,
                    name[0], fontsize=7, fontweight="bold", color="0.2",
                    ha="center", va="center")
    ax_map.set_xlim(-122.20, -121.85)
    ax_map.set_ylim(37.44, 37.75)
    ax_map.set_aspect(1 / np.cos(np.deg2rad(lat0)))
    ax_map.set_xticks([-122.2, -122.1, -122.0, -121.9])
    ax_map.set_ylabel("Latitude")
    if row == 1:
        ax_map.set_xlabel("Longitude")
    else:
        ax_map.set_xticklabels([])
    ax_map.set_title(r"$\bf{" + letters[0] + r"}$  " + cat_label, loc="left", fontsize=7)

    # -- one section per box, at true scale
    for col, (name, (km_a, km_b)) in enumerate(zip(box_names, boxes_km_pp), start=1):
        ax = fig.add_subplot(gs[row, col])
        in_box = cat[(cat.km_pp >= km_a) & (cat.km_pp <= km_b)
                     & (np.abs(cat.normal_km) <= max_normal_km)]
        ax.scatter(in_box.normal_km, in_box.depth_km,
                   s=1 + 1.5 * np.clip(in_box.magnitude.values, 0, None),
                   c=color, linewidths=0, alpha=0.55)
        depth_line = np.array([0, max_depth_km])
        for dip in dips_deg:
            east_offset = depth_line / np.tan(np.deg2rad(dip))
            ax.plot(east_offset, depth_line, "-", color="0.45", lw=0.6)
            if row == 0 and col == 3:
                ax.text(min(east_offset[-1], max_normal_km - 0.5), max_depth_km - 0.5,
                        f"{dip}\u00b0", fontsize=6, color="0.3", ha="left")
        ax.axhline(mesh_bottom_km, color="0.6", lw=0.5, ls=":")
        ax.set_xlim(-max_normal_km, max_normal_km)
        ax.set_ylim(max_depth_km, 0)
        ax.set_aspect(1)
        ax.set_xticks([-4, -2, 0, 2, 4])
        ax.set_yticks(np.arange(0, 13, 2))
        title = f"Box {name[0]}" + (" (creep section)" if name.startswith("2") else "")
        ax.set_title(r"$\bf{" + letters[col] + r"}$  " + title + f", n = {len(in_box)}",
                     loc="left", fontsize=7)
        if col == 1:
            ax.set_ylabel("Depth (km)")
        else:
            ax.set_yticklabels([])
        if row == 1 and col == 2:
            ax.set_xlabel("Distance from fault (km, + east)")
        if row == 0:
            ax.set_xticklabels([])

for ext in ("png", "pdf"):
    fig_file = os.path.join(fig_dir, f"seismicity_dip_sections_2x4.{ext}")
    fig.savefig(fig_file, dpi=300)
    print("wrote", fig_file)
