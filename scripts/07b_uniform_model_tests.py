# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
# ---

# %% [markdown]
# # 07 Slip model -- sensitivity tests on the final configuration
#
# ```
# Created on Thu Sep 25 2026; rebuilt on Mon Sep 28 2026 on the final configuration
# @author: daniellelindsay
# Run in opera_disp_feb13. Several hours (about 150 inversions).
# ```
#
# **What this notebook does.** Every test is run on the configuration of
# `07a_uniform_model` (the homogeneous baseline): the fault-parallel mean event
# (06, coherence 0.9), levelled, the `smooth12` vertical mesh, quadtree to one
# 30 m pixel, the crossing surface slip on every surface triangle, slip tied
# no top tie, sigma at 07a_uniform_model's L-curve pick, lambda as 07a_uniform_model. Each
# family changes ONE thing from that baseline. The machinery cells [2] to [6]
# are copied verbatim from `07a_uniform_model`, so the two notebooks run the same
# code; CHECK [21] pins this notebook's baseline to `07a_uniform_model`'s.
#
# 1. **Top tie** at the model sigma: none (baseline), 0.1, 0.2, 0.5 km.
# 2. **Fixed-depth scan** on the fault-parallel field: uniform slip from the
#    surface to D, delta chi2 (no smoothing, no sigma). Variants: the model mesh,
#    a 5 km mesh, the unlevelled field. Then the model difference between the
#    two ends of the bracket against the noise model.
# 3. **Synthetic recovery**: 10 mm to D through this pipeline.
# 4. **Isolation rows**: no surface constraint, unlevelled (with a plane per
#    dataset instead), surface error floors 0.25 and 1.0 mm.
# 5. **Sampling**: quadtree floors 40 and 60 m, and a uniform 100 m grid.
# 6. **Dip**: 60 to 105 degrees east (105 = 75 W).
# 7. **Correlation lengths lambda**: five pairs, each with its own L-curve,
#    compared at their own corners (lambda enters the prior's amplitude as well
#    as its shape, so one sigma does not fit all).
# 8. **Mesh**: plateau 100 m, 50 m, 30 m and a deep plateau, each with its own
#    L-curve.
#
# Replaces the batch-2 LOS suite of 2026-09-28 (now
# `drafts/07/slip_model_tests_LOS_batch2.py`; its numbers are in
# `07_sweep_lessons_2026-09-28.md`). The legacy parity gates of that suite are
# gone: the paper has no legacy chain to check against.
#
# **Inputs**: as `07a_uniform_model` (06 `fp_maps.npz`, 04 crossings, events and
# section lengths, the trace). **Outputs** (`outputs/07b_uniform_model_tests/`):
# `final_suite.csv`, `final_lcurves.csv`, `final_tie_slip_<tie>.csv`,
# `final_depth_scan_<variant>.csv`, `final_depth_vs_noise.csv`,
# `final_recovery.csv`, `final_recovery_profiles.csv`. **Figure**:
# `fig07_sensitivity_final`. Registry keys `07_test_*`.

# %%
## [1]
# ------------------------
# Paths & constants
# ------------------------
import copy
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, cKDTree
from scipy.interpolate import CubicSpline

warnings.simplefilter("ignore")
from TulePond_filepaths import common_paths, repo_dir
import insar_utils as utils
from insar_utils import load_trace, mpl_paper_style

# CSI: Classic Slip Inversion (Jolivet et al.; external/csi at 1277bb4)
from csi import TriangularPatches, insar, surfaceslip
from csi import imagedownsampling as imdown
from csi import transformation as transform
from csi import multisourcesolve as multisrc

notebook = "07b_uniform_model_tests"
out_dir = os.path.join(common_paths["out_dir"], notebook)
baseline_dir = os.path.join(common_paths["out_dir"], "07a_uniform_model")
field_dir = os.path.join(common_paths["out_dir"], "06_event_field_geometry")
events_dir = os.path.join(common_paths["out_dir"], "04_event_detection")
fig_dir = common_paths["fig_dir"]
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# ---- the baseline configuration: identical to 07a_uniform_model
frame_lon0, frame_lat0 = -121.968, 37.557
swath_km       = 4.0
exclude_km     = 0.03
km_window      = (52.0, 67.0)
start_block_km = 2.0
min_block_km   = 0.03
cov_mask_km    = 1.0
cov_distmax_km = 8.0
nug_power      = 0.5
near_field_km  = 0.2
section_lengths = pd.read_csv(os.path.join(events_dir, "section_lengths.csv")).set_index("name")
creep_lo_km = float(section_lengths.loc["tulepond_creep_length", "km_lo"])
creep_hi_km = float(section_lengths.loc["tulepond_creep_length", "km_hi"])
align_mask_km  = 1.0
surf_floor_mm  = 0.5
surf_step_km   = 0.2
surf_win_km    = 0.4
mesh_lat_lo, mesh_lat_hi = 37.47, 37.65
mesh_depth_km  = 3.0
mesh_params    = (0.05, 0.0, 1.2, 1.5)   # smooth12
dip_deg        = 90
fine_km        = (56.0, 63.0)
far_rate       = 0.25
smooth_iters   = 8
lam_xz         = (0.1, 0.05)   # as 07a_uniform_model (along strike 0.1 km, Dani 2026-09-29)
tie_depth_km   = 0.0           # no top tie in the model (Dani 2026-09-29); the tie family still tests 0.1-0.5 km
shear_modulus  = 3.0e10
# sigma: the L-curve pick of 07a_uniform_model on the same data (the rule is fixed, the value is not)
sigma_file     = os.path.join(baseline_dir, "sigma_lcurve.csv")
assert os.path.exists(sigma_file), f"missing {sigma_file}: run 07a_uniform_model first"
sigma_model    = float(pd.read_csv(sigma_file).selected.iloc[0])
print(f"sigma from 07a_uniform_model's L-curve: {sigma_model:g}")

# ---- the families (each changes one thing from the baseline)
tie_depths     = [0.0, 0.1, 0.2, 0.5]    # 0 = no tie
dips_deg       = [60, 70, 75, 80, 85, 90, 95, 100, 105]   # 105 E = 75 W
lam_grid       = [lam_xz] + [pair for pair in [(0.3, 0.05), (0.05, 0.05), (0.3, 0.15), (0.1, 0.15)] if pair != lam_xz]   # the model's pair first
mesh_variants  = {                        # name: (h_top, const_to, growth, h_bot) km
    "smooth12": mesh_params,
    "plateau100": (0.10, 0.5, 1.6, 1.5),
    "plateau100_deep": (0.10, 1.5, 1.6, 1.5),
    "fine50": (0.05, 0.5, 1.6, 1.5),
    "fine30": (0.03, 0.25, 1.6, 1.5),
}
sigma_grid_lc  = [4, 6, 8, 12, 16, 24, 32, 48, 64, 128]   # L-curves of the lambda and mesh families
surf_floors    = [0.25, 1.0]
recovery_depths = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
scan_taper     = 0.5
scan_km        = (creep_lo_km + scan_taper, creep_hi_km - scan_taper)   # uniform slip over the SSE section (04), tapered to zero at its ends
scan_depths    = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65,
                  0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.25, 1.35, 1.5,
                  1.65, 1.85, 2.0, 2.25, 2.5, 3.0, 4.0, 5.0]
noise_bins_km  = [0.03, 0.5, 1.0, 2.0, 3.0, 4.0]   # distance-to-trace bins of the depth-vs-noise table

REG = utils.Registry(notebook, common_paths, repo_dir)
REG.inputs(os.path.join(field_dir, "fp_maps.npz"),
           common_paths["fault_trace"], common_paths["fault_profiles"],
           os.path.join(events_dir, "event_fraction_by_crossing.csv"),
           os.path.join(events_dir, "creep_events.csv"),
           os.path.join(events_dir, "section_lengths.csv"))
print(f"SSE section: km {creep_lo_km:.1f}-{creep_hi_km:.1f} (04 section_lengths.csv)")

# %% [markdown]
# ## [2] The fault mesh
#
# Unrolled (along-strike s, depth z) node rows -> Delaunay -> Laplacian
# smoothing -> wrapped onto the hand-drawn trace as a vertical
# `csi.TriangularPatches` (`tutoFaultMeshing` idiom).

# %%
## [2]
trace_lonlat = load_trace(common_paths["fault_trace"])
trace_lonlat = trace_lonlat[(trace_lonlat[:, 1] >= mesh_lat_lo)
                            & (trace_lonlat[:, 1] <= mesh_lat_hi)]
trace_lonlat = trace_lonlat[np.argsort(-trace_lonlat[:, 1])]     # north -> south

ref_fault = TriangularPatches("trace frame", lon0=frame_lon0, lat0=frame_lat0,
                              verbose=False)
ref_fault.trace(trace_lonlat[:, 0], trace_lonlat[:, 1])
ref_fault.discretize2(every=0.02)                # 20 m sampling along the trace
trace_x, trace_y = ref_fault.xi, ref_fault.yi
arc_s = np.r_[0, np.cumsum(np.hypot(np.diff(trace_x), np.diff(trace_y)))]
arc_total = arc_s[-1]

profile_points = pd.read_csv(common_paths["fault_profiles"])
profile_km_pp = (len(profile_points) - 1 - np.arange(len(profile_points))) * 0.1
def lat_to_km_pp(lats):
    lats = np.asarray(lats, float)
    out = np.empty(lats.size)
    for lo in range(0, lats.size, 20000):
        chunk = lats[lo:lo + 20000]
        out[lo:lo + 20000] = profile_km_pp[np.abs(profile_points.ft_lat.values[None, :]
                                                  - chunk[:, None]).argmin(axis=1)]
    return out
trace_km_pp = lat_to_km_pp(ref_fault.lati)


def build_unrolled(h_top, const_to, growth, h_bot, z_max=mesh_depth_km):
    """Node rows + Delaunay + Laplacian smoothing in the unrolled plane.
    [07b_uniform_model_tests, verbatim]"""
    row_depths = [0.0]
    row_step = h_top
    while row_depths[-1] < z_max:
        row_depths.append(min(row_depths[-1] + row_step, z_max))
        if row_depths[-1] >= const_to - 1e-9:
            row_step = min(row_step * growth, h_bot)
    row_depths = np.array(row_depths)
    row_heights = np.r_[np.diff(row_depths), np.diff(row_depths)[-1]]

    def spacing_at(arc_pos, row_height):
        km_pp = np.interp(arc_pos, arc_s, trace_km_pp)
        outside = np.maximum(0, np.maximum(fine_km[0] - km_pp, km_pp - fine_km[1]))
        return row_height + far_rate * outside

    nodes = []
    for depth, height in zip(row_depths, row_heights):
        row = [0.0]
        while row[-1] < arc_total:
            row.append(row[-1] + spacing_at(row[-1], height))
        row = np.array(row) * arc_total / row[-1]
        nodes += [(pos, depth) for pos in row]
    nodes = np.array(nodes)

    def triangulate(pts):
        faces = Delaunay(pts).simplices
        area2 = np.abs((pts[faces[:, 1], 0] - pts[faces[:, 0], 0])
                       * (pts[faces[:, 2], 1] - pts[faces[:, 0], 1])
                       - (pts[faces[:, 2], 0] - pts[faces[:, 0], 0])
                       * (pts[faces[:, 1], 1] - pts[faces[:, 0], 1]))
        return faces[0.5 * area2 > 1e-6]

    faces = triangulate(nodes)
    fixed = ((nodes[:, 1] < 1e-9) | (nodes[:, 1] > z_max - 1e-9)
             | (nodes[:, 0] < 1e-9) | (nodes[:, 0] > arc_total - 1e-9))
    for _ in range(smooth_iters):
        neighbours = {}
        for tri in faces:
            for node in tri:
                neighbours.setdefault(node, set()).update(tri)
        smoothed = nodes.copy()
        for node in np.nonzero(~fixed)[0]:
            ring = np.array(sorted(neighbours[node] - {node}))
            smoothed[node] = nodes[ring].mean(0)
        nodes = smoothed
        faces = triangulate(nodes)
    return nodes, faces


def wrap_fault(nodes, faces, dip_deg, name):
    """csi.TriangularPatches hinged east about the FIXED surface trace.
    Each fault owns its faces copy (homogeneizeStrike flips winding rows
    in place). [07b_uniform_model_tests, verbatim]"""
    flt = TriangularPatches(name, lon0=frame_lon0, lat0=frame_lat0, verbose=False)
    flt.trace(trace_lonlat[:, 0], trace_lonlat[:, 1])
    flt.discretize2(every=0.02)
    node_x = np.interp(nodes[:, 0], arc_s, trace_x)
    node_y = np.interp(nodes[:, 0], arc_s, trace_y)
    vertices = np.c_[node_x, node_y, nodes[:, 1]]
    if abs(dip_deg - 90.0) > 0.1:
        strike_vec = np.array([trace_x[-1] - trace_x[0], trace_y[-1] - trace_y[0]])
        strike_vec = strike_vec / np.hypot(strike_vec[0], strike_vec[1])
        normal_vec = np.array([strike_vec[1], -strike_vec[0]])
        if normal_vec[0] < 0:
            normal_vec = -normal_vec                   # + east
        east_shift = nodes[:, 1] / np.tan(np.deg2rad(dip_deg))
        vertices[:, 0] += east_shift * normal_vec[0]
        vertices[:, 1] += east_shift * normal_vec[1]
    faces_own = faces.copy()
    flt.Vertices = vertices
    flt.Faces = faces_own
    flt.vertices2ll()
    flt.patch = [np.array([vertices[i] for i in f]) for f in faces_own]
    flt.patch2ll()
    flt.setdepth()
    flt.vertical = abs(dip_deg - 90.0) <= 0.1
    flt.homogeneizeStrike(direction=0, sign=-1.0)
    flt.initializeslip()
    flt.mesh_sz = (nodes.copy(), faces_own, arc_total)
    return flt

mesh_nodes, mesh_faces = build_unrolled(*mesh_params)
fault = wrap_fault(mesh_nodes, mesh_faces, dip_deg, "mean event")
tri_sz = mesh_nodes[mesh_faces]
cell_area_km2 = 0.5 * np.abs((tri_sz[:, 1, 0] - tri_sz[:, 0, 0]) * (tri_sz[:, 2, 1] - tri_sz[:, 0, 1])
                             - (tri_sz[:, 2, 0] - tri_sz[:, 0, 0]) * (tri_sz[:, 1, 1] - tri_sz[:, 0, 1]))
print(f"mesh smooth12: {len(mesh_nodes)} nodes, {len(mesh_faces)} triangles, "
      f"{arc_total:.1f} km of fault, dip {dip_deg}")

# %% [markdown]
# ## [3] Data: the fault-parallel mean event
#
# Notebook 06's decomposition (coherence 0.9, SSE-section strike) as one
# `csi.insar` dataset whose look vector is the horizontal unit vector along
# the strike bearing, so right-lateral motion of the east side is positive.
# Cut to 4 km of the trace and the along-strike window, then the far field
# (slip section +/- 1 km masked) is levelled by a fitted plane.

# %%
## [3]
def dist_to_trace(sar, fault, signed=False):
    """[csi_io, verbatim] + east of the trace when signed."""
    tx, ty = fault.xf, fault.yf
    d, i = cKDTree(np.c_[tx, ty]).query(np.c_[sar.x, sar.y])
    return d * np.sign(sar.x - tx[i]) if signed else d


def align_track(sar):
    """Far-field plane alignment: slip section +/- align_mask_km masked,
    1,x,y fit to the rest, removed from the whole track in place.
    [07b_uniform_model_tests, verbatim]"""
    km_pp = lat_to_km_pp(sar.lat)
    far = (km_pp < creep_lo_km - align_mask_km) | (km_pp > creep_hi_km + align_mask_km)
    plane = np.c_[np.ones(sar.x.size), sar.x, sar.y]
    plane_coef = np.linalg.lstsq(plane[far], sar.vel[far], rcond=None)[0]
    sar.vel[:] = sar.vel - np.dot(plane, plane_coef)
    return plane_coef

fp = np.load(os.path.join(field_dir, "fp_maps.npz"))
strike_bearing = float(fp["strike"])
look = np.array([np.sin(np.deg2rad(strike_bearing)), np.cos(np.deg2rad(strike_bearing)), 0.0])
ok = np.isfinite(fp["fp_event"])
sar = insar("fault-parallel", lon0=frame_lon0, lat0=frame_lat0, verbose=False)
sar.read_from_binary(fp["fp_event"][ok], fp["lon"][ok], fp["lat"][ok], err=np.ones(ok.sum()),
                     los=np.tile(look, (ok.sum(), 1)), remove_nan=True, remove_zeros=False)
trace_dist = dist_to_trace(sar, fault)
sar.vel[(trace_dist > swath_km) | (trace_dist < exclude_km)] = np.nan
km_pp = lat_to_km_pp(sar.lat)
sar.vel[(km_pp < km_window[0]) | (km_pp > km_window[1])] = np.nan
sar.checkNaNs()
plane_coef = align_track(sar)
print(f"fault-parallel field at strike {strike_bearing:.2f} deg: {sar.vel.size} pixels; "
      f"levelling plane {plane_coef[0]:+.2f} mm, {plane_coef[1]:+.3f}, {plane_coef[2]:+.3f} mm/km")

# %% [markdown]
# ## [4] Noise model and quadtree
#
# Far-field semivariogram: nugget + short- and long-range exponentials.
# Distance-based quadtree (`csi.imagedownsampling`) from 2 km blocks to one
# 30 m pixel at the trace; the full data covariance of the samples.

# %%
## [4]
def covariance(sar, fault, npairs=300000, seed=0):
    """[m02, verbatim apart from @ -> np.dot]"""
    from scipy.optimize import curve_fit
    d = dist_to_trace(sar, fault)
    keep = (d > min(cov_mask_km, 0.4 * swath_km)) & np.isfinite(sar.vel)
    x, y, v = sar.x[keep], sar.y[keep], sar.vel[keep]
    design = np.c_[np.ones_like(x), x, y]
    v = v - np.dot(design, np.linalg.lstsq(design, v, rcond=None)[0])
    rng = np.random.default_rng(seed)
    i = rng.integers(0, v.size, npairs); j = rng.integers(0, v.size, npairs)
    h = np.hypot(x[i] - x[j], y[i] - y[j]); dv = v[i] - v[j]
    hmax = min(cov_distmax_km, 4 * swath_km)
    ok = (h > 0) & (h < hmax)
    bins = np.arange(0, hmax + 0.1, 0.1)
    idx = np.digitize(h[ok], bins) - 1
    gam = np.array([0.5 * np.mean(dv[ok][idx == k] ** 2) if (idx == k).sum() > 50 else np.nan
                    for k in range(len(bins) - 1)])
    hb = 0.5 * (bins[1:] + bins[:-1])
    fin = np.isfinite(gam)
    model = lambda hh, n2, s1, l1, s2, l2: n2 + s1 * (1 - np.exp(-hh / l1)) + s2 * (1 - np.exp(-hh / l2))
    sill = np.nanmean(gam[fin][-5:])
    p, _ = curve_fit(model, hb[fin], gam[fin], p0=(0.4 * sill, 0.3 * sill, 0.7, 0.5 * sill, 5.0),
                     bounds=([0, 0, 0.1, 0, 2.0], [2 * sill, 2 * sill, 2.0, 3 * sill, 50]), maxfev=20000)
    return p


def downsample(sar, fault, start_km, min_km):
    """Distance-based quadtree (csi.imagedownsampling). [m02, verbatim]"""
    ds = imdown(sar.name, sar, fault, verbose=False)
    ds.initialstate(start_km, min_km, tolerance=0.1, plot=False)
    if start_km > min_km + 1e-9:
        ds.distanceBased(chardist=0.0, expodist=1, plot=False)
    return ds.newimage


def build_cd(ds, cov_p, asc_scale=1.0):
    """Two exponentials + nugget/blocksize. [m02; 07b_uniform_model_tests, verbatim]"""
    nug2, s1, l1, s2, l2 = cov_p
    dx = ds.x[:, None] - ds.x[None, :]; dy = ds.y[:, None] - ds.y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    cd = s1 * np.exp(-dist / l1) + s2 * np.exp(-dist / l2)
    n_px = np.asarray(getattr(ds, "wgt", np.ones(ds.vel.size)), dtype=float)
    n_px = np.where(np.isfinite(n_px) & (n_px > 0), n_px, 1.0)
    cd[np.diag_indices_from(cd)] += nug2 / n_px ** nug_power
    ds.Cd = cd * asc_scale ** 2
    return ds

cov_p = covariance(sar, fault)
samples = downsample(sar, fault, start_block_km, min_block_km)
build_cd(samples, cov_p)
noise_model = pd.Series(dict(nugget_mm=np.sqrt(cov_p[0]), short_mm=np.sqrt(cov_p[1]),
                             short_km=cov_p[2], long_mm=np.sqrt(cov_p[3]), long_km=cov_p[4]))
print(noise_model.round(3).to_string())
print(f"quadtree: {samples.vel.size} samples")

# %% [markdown]
# ## [5] The surface constraint
#
# The five-event mean surface slip at every fault crossing (04), as a
# running median every 0.2 km (+/- 0.2 km), error 1.4826 MAD / sqrt(n)
# floored at 0.5 mm. CSI's default link ties each surface point to its
# single nearest triangle (`Fault.surfaceGFs`), which reaches ~7% of the
# creep-section surface row; here the profile is interpolated to the centre
# of every surface triangle, errors scaled by sqrt(triangles per 0.2 km) so
# the total weight is unchanged.

# %%
## [5]
crossings_all = pd.read_csv(os.path.join(events_dir, "event_fraction_by_crossing.csv"))
creep_events = pd.read_csv(os.path.join(events_dir, "creep_events.csv"))
tulepond_onsets = creep_events[creep_events.segment == "TULEPOND"].onset_yr.values
event_cols = [f"amp_{onset:.3f}" for onset in tulepond_onsets]    # 04's column naming
print(f"mean event of {len(event_cols)} Tule Pond events: {', '.join(event_cols)}")
crossings_all["mean_event_mm"] = crossings_all[event_cols].mean(axis=1)
trace_south_north = trace_lonlat[np.argsort(trace_lonlat[:, 1])]

def make_surface(floor_mm):
    """csi.surfaceslip: running median of the 5-event crossing means,
    error = 1.4826 MAD / sqrt(n) floored at floor_mm. [07b_uniform_model_tests, verbatim]"""
    rows = []
    for k_centre in np.arange(np.ceil(crossings_all.km.min()),
                              crossings_all.km.max(), surf_step_km):
        in_win = np.abs(crossings_all.km - k_centre) <= surf_win_km / 2
        amps = crossings_all.mean_event_mm[in_win].dropna()
        if len(amps) < 4:
            continue
        med = float(np.median(amps))
        err = max(1.4826 * float(np.median(np.abs(amps - med))) / np.sqrt(len(amps)),
                  floor_mm)
        lat_c = float(np.median(crossings_all.lat[in_win]))
        lon_c = float(np.interp(lat_c, trace_south_north[:, 1], trace_south_north[:, 0]))
        rows.append((lon_c, lat_c, k_centre, med, err))
    rows = np.array(rows)
    rows = rows[(rows[:, 2] >= km_window[0]) & (rows[:, 2] <= km_window[1])]
    ss = surfaceslip(f"crossing stacks f{floor_mm:g}", lon0=frame_lon0,
                     lat0=frame_lat0, verbose=False)
    ss.read_from_binary(rows[:, 3], rows[:, 0], rows[:, 1], err=rows[:, 4])
    ss.buildCd()
    return ss

def make_surface_interp(fault, floor_mm):
    """The running-median profile interpolated to the centre of EVERY
    surface triangle of this fault. [07b_uniform_model_tests, verbatim]"""
    base = make_surface(floor_mm)
    base_km = lat_to_km_pp(base.lat)
    order = np.argsort(base_km)
    surf = np.where([0 in p[:, 2] for p in fault.patch])[0]
    ll = np.array([fault.patchll[i].mean(0) for i in surf])
    tri_km = lat_to_km_pp(ll[:, 1])
    keep = (tri_km >= base_km.min()) & (tri_km <= base_km.max())
    ll, tri_km = ll[keep], tri_km[keep]
    vals = np.interp(tri_km, base_km[order], base.vel[order])
    errs = np.interp(tri_km, base_km[order], base.err[order])
    per_step = np.array([np.sum(np.abs(tri_km - k) <= surf_step_km / 2) for k in tri_km])
    errs = errs * np.sqrt(np.maximum(per_step, 1))
    ss = surfaceslip(f"crossing interp f{floor_mm:g}", lon0=frame_lon0,
                     lat0=frame_lat0, verbose=False)
    ss.read_from_binary(vals, ll[:, 0], ll[:, 1], err=errs)
    ss.buildCd()
    return ss

crossing_profile = make_surface(surf_floor_mm)
crossing_km = lat_to_km_pp(crossing_profile.lat)
crossing_median = float(np.median(crossing_profile.vel[(crossing_km >= creep_lo_km)
                                                       & (crossing_km <= creep_hi_km)]))
print(f"surface constraint: {crossing_profile.vel.size} profile points; creep-section "
      f"median {crossing_median:.2f} mm")

# %% [markdown]
# ## [6] The inversion
#
# Meade triangular dislocations (right-lateral positive) for the samples;
# for the surface data the Green's function is 1 on the triangle's own
# slip. Per-dataset constant (`csi.transformation`), exponential prior
# (`buildCmXY`), the top tie (every cell shallower than 0.2 km equal to the
# shallowest cell of its 0.1 km along-strike column; after Jolivet et al.
# 2023, who tie 0-1 km), NNLS (`multisourcesolve`).
# [07b_uniform_model_tests, verbatim]

# %%
## [6]
def tie_rows(fault, n_params, tie_depth_km, tie_bin_km=0.1, tie_err_mm=0.01):
    """Rows of [cell - reference] = 0 with a tiny error (near-hard constraint)."""
    nodes, faces, _ = fault.mesh_sz
    centroid = nodes[faces].mean(1)
    shallow = np.nonzero(centroid[:, 1] < tie_depth_km)[0]
    bins = np.floor(centroid[shallow, 0] / tie_bin_km).astype(int)
    rows = []
    for b in np.unique(bins):
        members = shallow[bins == b]
        if members.size < 2:
            continue
        ref = members[np.argmin(centroid[members, 1])]
        for m in members[members != ref]:
            r = np.zeros(n_params); r[m] = 1.0; r[ref] = -1.0
            rows.append(r)
    E = np.array(rows) if rows else np.zeros((0, n_params))
    return E, np.full(E.shape[0], tie_err_mm ** 2)


def invert(fault, insar_datas, sigma, lam=lam_xz, prior="xy", slipdir="s",
           ramp_spec="1", surface=None, tie_depth_km=None):
    solve_datas = list(insar_datas) + ([surface] if surface is not None else [])
    for ds in insar_datas:
        fault.buildGFs(ds, vertical=True, slipdir=slipdir, verbose=False)
        fault.G[ds.name]["strikeslip"] *= -1.0        # right-lateral positive
    if surface is not None:
        fault.buildGFs(surface, slipdir=slipdir, verbose=False)   # NO flip
    trans = transform("ramps", lon0=frame_lon0, lat0=frame_lat0, verbose=False)
    specs = [ramp_spec] * len(insar_datas) + (["1"] if surface is not None else [])
    trans.buildGFs(solve_datas, specs, verbose=False)
    fault.assembleGFs(solve_datas, slipdir=slipdir); fault.assembled(solve_datas)
    fault.assembleCd(solve_datas)
    trans.assembleGFs(solve_datas); trans.assembled(solve_datas)
    trans.assembleCd(solve_datas)
    if prior == "diag":
        fault.buildCmGaussian([sigma ** 2] * (1 if slipdir == "s" else 2), verbose=False)
    else:
        fault.buildCmXY(sigma, lam, verbose=False)
    trans.buildCm(1e4, verbose=False)
    solver = multisrc("mean event", [fault, trans], verbose=False)
    solver.assembleGFs(); solver.assembleCm()
    if tie_depth_km:                                  # uniform slip over the top tie_depth_km
        E, e_var = tie_rows(fault, solver.G.shape[1], tie_depth_km)
        solver.G = np.vstack([solver.G, E])
        solver.d = np.concatenate([solver.d, np.zeros(E.shape[0])])
        n_d = solver.Cd.shape[0]
        Cd_new = np.zeros((n_d + E.shape[0], n_d + E.shape[0]))
        Cd_new[:n_d, :n_d] = solver.Cd
        Cd_new[n_d:, n_d:] = np.diag(e_var)
        solver.Cd = Cd_new
    if slipdir == "s":
        mprior = np.zeros(solver.Np)
        ramp_idx = [i for i, t in enumerate(solver.paramTypes) if t[0] == trans.name]
        mprior[ramp_idx] = -1e3                       # NNLS: ramps may be negative
        solver.ConstrainedLeastSquareSoln(method="nnls", mprior=mprior)
    else:
        solver.GeneralizedLeastSquareSoln()           # rake free: sign unconstrained
    solver.distributem()
    for ds in solve_datas:
        ds.buildsynth([fault], direction=slipdir)
        ds.removeTransformation(trans)                # residuals are ramp-removed
    return solver


# %% [markdown]
# ## [7] The suite machinery: samples per geometry, one case, one L-curve
#
# `get_samples` returns a fresh copy of the quadtree samples for a mesh, dip,
# quadtree floor and field version (levelled or not), with its covariance
# rebuilt; the quadtree is computed once per geometry. `run_case` builds a
# fresh fault and inverts once; `model_numbers` gives the numbers every row
# carries. `lcurve` runs one case per sigma and finds the corner exactly as
# `07a_uniform_model` [8] does.

# %%
## [7]
def load_fp_field(levelled=True):
    """Cell [3]'s fault-parallel field, levelled (the baseline) or not."""
    field = insar("fault-parallel", lon0=frame_lon0, lat0=frame_lat0, verbose=False)
    field.read_from_binary(fp["fp_event"][ok], fp["lon"][ok], fp["lat"][ok], err=np.ones(ok.sum()),
                           los=np.tile(look, (ok.sum(), 1)), remove_nan=True, remove_zeros=False)
    dist = dist_to_trace(field, fault)
    field.vel[(dist > swath_km) | (dist < exclude_km)] = np.nan
    field_km = lat_to_km_pp(field.lat)
    field.vel[(field_km < km_window[0]) | (field_km > km_window[1])] = np.nan
    field.checkNaNs()
    if levelled:
        align_track(field)
    return field

fields = {True: load_fp_field(levelled=True), False: load_fp_field(levelled=False)}
assert np.array_equal(fields[True].vel, sar.vel), "load_fp_field does not reproduce cell [3]"
cov_models = {True: cov_p, False: covariance(fields[False], fault)}
unrolled = {name: build_unrolled(*params) for name, params in mesh_variants.items()}
for name, (nodes_, faces_) in unrolled.items():
    print(f"mesh {name}: {len(faces_)} triangles")

sample_cache = {}
def get_samples(mesh_name="smooth12", dip=dip_deg, min_km=min_block_km,
                start_km=start_block_km, levelled=True):
    key = (mesh_name, dip, min_km, start_km, levelled)
    if key not in sample_cache:
        geometry = wrap_fault(*unrolled[mesh_name], dip, "samples")
        sample_cache[key] = downsample(fields[levelled], geometry, start_km, min_km)
    ds = copy.deepcopy(sample_cache[key])
    build_cd(ds, cov_models[levelled])
    return ds

assert np.array_equal(get_samples().vel, samples.vel), "get_samples does not reproduce cell [4]"


def model_numbers(flt, ds, surface):
    """Misfit, surface fit, peak, potency and Mw of one solved fault."""
    nodes_, faces_, _ = flt.mesh_sz
    tri = nodes_[faces_]
    area_km2 = np.array([0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])) for p in flt.patch])
    slip = flt.slip[:, 0]
    z_top = tri[:, :, 1].min(1)
    z_c = tri[:, :, 1].mean(1)
    cell_km = lat_to_km_pp(np.array([pll.mean(0) for pll in flt.patchll])[:, 1])
    surface_cells = (z_top <= 1e-6) & (cell_km >= creep_lo_km) & (cell_km <= creep_hi_km)
    resid = ds.vel - ds.synth
    near = dist_to_trace(ds, flt) <= near_field_km
    potency = float(np.sum(area_km2 * 1e6 * slip * 1e-3))
    row = dict(chi2=float(np.dot(resid, np.linalg.solve(ds.Cd, resid))),
               rms_mm=float(np.sqrt(np.mean(resid ** 2))),
               rms_near_mm=float(np.sqrt(np.mean(resid[near] ** 2))),
               chi2_surf=np.nan, surf_rms_mm=np.nan,
               surface_pct=float(100 * np.median(slip[surface_cells]) / crossing_median),
               peak_slip_mm=float(slip.max()), peak_depth_m=float(1000 * z_c[np.argmax(slip)]),
               potency_m3=potency, Mw=(2.0 / 3.0) * (np.log10(shear_modulus * max(potency, 1e-9)) - 9.1),
               n_samples=int(ds.vel.size), n_cells=len(flt.patch))
    if surface is not None:
        surf_resid = surface.vel - surface.synth
        row["chi2_surf"] = float(np.dot(surf_resid, np.linalg.solve(surface.Cd, surf_resid)))
        row["surf_rms_mm"] = float(np.sqrt(np.mean(surf_resid ** 2)))
    return row


def slip_table(flt):
    nodes_, faces_, _ = flt.mesh_sz
    tri = nodes_[faces_]
    return pd.DataFrame(dict(s_km=tri[:, :, 0].mean(1), z_c=tri[:, :, 1].mean(1),
                             z_top=tri[:, :, 1].min(1), z_bot=tri[:, :, 1].max(1),
                             km_pp=lat_to_km_pp(np.array([pll.mean(0) for pll in flt.patchll])[:, 1]),
                             slip_mm=flt.slip[:, 0]))


suite_rows = []
def run_case(family, label, mesh_name="smooth12", dip=dip_deg, sigma=sigma_model, lam=lam_xz,
             tie=tie_depth_km, use_surface=True, floor=surf_floor_mm, min_km=min_block_km,
             start_km=start_block_km, levelled=True, ramp_spec="1", record=True):
    flt = wrap_fault(*unrolled[mesh_name], dip, f"{family} {label}")
    ds = get_samples(mesh_name, dip, min_km, start_km, levelled)
    surface = make_surface_interp(flt, floor) if use_surface else None
    invert(flt, [ds], sigma, lam=lam, ramp_spec=ramp_spec, surface=surface,
           tie_depth_km=tie if tie > 0 else None)
    row = model_numbers(flt, ds, surface)
    row.update(family=family, label=label, mesh=mesh_name, dip=dip, sigma=sigma, lam_s=lam[0],
               lam_z=lam[1], tie_km=tie, use_surface=use_surface, floor=floor if use_surface else np.nan,
               min_block_km=min_km, start_block_km=start_km, levelled=levelled, ramp=ramp_spec)
    if record:
        suite_rows.append(row)
    print(f"[{family}] {label}: rms {row['rms_mm']:.3f} mm, surface {row['surface_pct']:.0f}%, "
          f"peak {row['peak_slip_mm']:.1f} mm at {row['peak_depth_m']:.0f} m, Mw {row['Mw']:.3f}")
    return row, flt, ds, surface


def save_suite():
    """Written after every family, so a partial run keeps what it did."""
    pd.DataFrame(suite_rows).to_csv(os.path.join(out_dir, "final_suite.csv"), index=False)


lcurve_rows = []
def lcurve(label, mesh_name="smooth12", lam=lam_xz, sigmas=sigma_grid_lc):
    """One case per sigma; the corner and the grid sigma the rule picks
    (07a_uniform_model [8], verbatim). Returns the L-curve table, the corner,
    the pick, and the row at the pick."""
    rows = []
    for sigma in sigmas:
        row, flt, ds, surface = run_case("lcurve", f"{label} sigma{sigma:g}", mesh_name=mesh_name,
                                         sigma=sigma, lam=lam, record=False)
        slip_vec = flt.slip[:, 0]
        row["prior_norm"] = float(np.sqrt(np.dot(slip_vec, np.linalg.solve(flt.Cm / sigma ** 2, slip_vec))))
        row["chi2_total"] = row["chi2"] + row["chi2_surf"]
        row["curve"] = label
        rows.append(row)
    table = pd.DataFrame(rows)
    log_sigma = np.log10(table.sigma.values)
    spline_misfit = CubicSpline(log_sigma, np.log10(np.sqrt(table.chi2_total.values)))
    spline_rough = CubicSpline(log_sigma, np.log10(table.prior_norm.values))
    fine_log_sigma = np.linspace(log_sigma[1], log_sigma[-2], 400)
    d_misfit, dd_misfit = spline_misfit(fine_log_sigma, 1), spline_misfit(fine_log_sigma, 2)
    d_rough, dd_rough = spline_rough(fine_log_sigma, 1), spline_rough(fine_log_sigma, 2)
    curvature = np.abs(d_misfit * dd_rough - d_rough * dd_misfit) / (d_misfit ** 2 + d_rough ** 2) ** 1.5
    corner = float(10 ** fine_log_sigma[np.argmax(curvature)])
    pick = float(table.sigma[table.sigma >= corner].min())
    table["corner"], table["pick"] = corner, pick
    lcurve_rows.extend(table.to_dict("records"))
    pd.DataFrame(lcurve_rows).to_csv(os.path.join(out_dir, "final_lcurves.csv"), index=False)
    print(f"L-curve {label}: corner sigma {corner:.1f}, rule picks {pick:g}")
    return table, corner, pick, table[table.sigma == pick].iloc[0].to_dict()

# %% [markdown]
# ## [8] Family 1: the top tie at the model sigma
#
# Slip uniform from the surface to the tie depth in every 0.1 km column (the
# baseline is 0.2 km, chosen on 2026-09-28 at sigma 24 on the 0.7 field). The
# slip table of each is kept for the depth profiles.

# %%
## [8]
tie_tables = {}
for tie in tie_depths:
    label = f"tie{tie * 1000:.0f}m"
    row, flt, _, _ = run_case("tie", label, tie=tie)
    tie_tables[tie] = slip_table(flt)
    tie_tables[tie].to_csv(os.path.join(out_dir, f"final_tie_slip_{label}.csv"), index=False)
save_suite()
baseline_row = [r for r in suite_rows if r["family"] == "tie" and r["tie_km"] == tie_depth_km][0]

# %% [markdown]
# ## [9] Family 2: the fixed-depth scan on the fault-parallel field
#
# Uniform slip from the surface to D over the scan window (tapered over
# 0.5 km at each end), its amplitude and a constant solved by whitened least
# squares with the full data covariance: chi2(D), no prior, no sigma. The
# bracket is delta chi2 <= 4. Mw per D at 30 GPa from the fitted amplitude.
# Run on the model mesh (`smooth12`), the earlier scan mesh (`plateau100`),
# a 5 km deep version of it (so the mesh bottom does not cap the bracket) and
# the unlevelled field.

# %%
## [9]
def depth_scan(mesh_name, tag, levelled=True, nodes_faces=None):
    nodes_, faces_ = nodes_faces if nodes_faces is not None else unrolled[mesh_name]
    flt = wrap_fault(nodes_, faces_, dip_deg, f"scan {tag}")
    geometry_key = mesh_name if nodes_faces is None else tag
    if nodes_faces is not None:
        unrolled[tag] = nodes_faces
    ds = get_samples(geometry_key, dip_deg, levelled=levelled)
    flt.buildGFs(ds, vertical=True, slipdir="s", verbose=False)
    gmat = -flt.G[ds.name]["strikeslip"]                       # right-lateral positive
    table = slip_table(flt)
    area_km2 = np.array([0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])) for p in flt.patch])
    in_window = np.clip(np.minimum(table.km_pp - scan_km[0], scan_km[1] - table.km_pp) / scan_taper + 1, 0, 1).values
    chol = np.linalg.cholesky(ds.Cd + 1e-9 * np.eye(ds.Cd.shape[0]))
    data_w = np.linalg.solve(chol, ds.vel)
    rows, predictions = [], {}
    for depth_bottom in [d for d in scan_depths if d <= table.z_bot.max() + 1e-6]:
        shape = in_window * (table.z_c.values < depth_bottom)
        design = np.c_[np.dot(gmat, shape), np.ones(ds.vel.size)]
        coef, *_ = np.linalg.lstsq(np.linalg.solve(chol, design), data_w, rcond=None)
        resid_w = data_w - np.dot(np.linalg.solve(chol, design), coef)
        potency = float(np.sum(area_km2 * 1e6 * shape * coef[0] * 1e-3))
        rows.append(dict(D_km=depth_bottom, slip_mm=coef[0], constant_mm=coef[1],
                         chi2=float(np.dot(resid_w, resid_w)), potency_m3=potency,
                         Mw=(2.0 / 3.0) * (np.log10(shear_modulus * max(potency, 1e-9)) - 9.1)))
        predictions[depth_bottom] = np.dot(design, coef)
    scan = pd.DataFrame(rows)
    scan["dchi2"] = scan.chi2 - scan.chi2.min()
    scan.to_csv(os.path.join(out_dir, f"final_depth_scan_{tag}.csv"), index=False)
    best, bracket = scan.loc[scan.chi2.idxmin()], scan[scan.dchi2 <= 4]
    print(f"depth scan {tag}: best D {best.D_km:g} km ({best.slip_mm:.1f} mm, Mw {best.Mw:.2f}); "
          f"delta chi2 <= 4: {bracket.D_km.min():g}-{bracket.D_km.max():g} km, "
          f"Mw {bracket.Mw.min():.2f}-{bracket.Mw.max():.2f}")
    return scan, predictions, ds, flt

scans = {}
scans["smooth12"], predictions, scan_samples, scan_fault = depth_scan("smooth12", "smooth12")
scans["plateau100"], _, _, _ = depth_scan("plateau100", "plateau100")
scans["plateau100_z5"], _, _, _ = depth_scan("plateau100", "plateau100_z5",
                                              nodes_faces=build_unrolled(0.10, 0.5, 1.6, 1.5, z_max=5.0))
scans["unlevelled"], _, _, _ = depth_scan("smooth12", "unlevelled", levelled=False)

# %% [markdown]
# ### The two ends of the bracket against the noise
#
# The fault-parallel displacement the shallowest and the deepest accepted
# models predict at the samples (model mesh), their difference binned by
# distance from the trace, against the noise model: the standard deviation
# of one sample (nugget + both exponentials) and the rms difference between
# two samples that far apart, sqrt(2 gamma(h)). Numbers only; delta chi2
# above is the full-covariance version of the same question.

# %%
## [10]
record_scan = scans["smooth12"]
bracket = record_scan[record_scan.dchi2 <= 4]
d_shallow, d_deep = float(bracket.D_km.min()), float(bracket.D_km.max())
model_diff = predictions[d_deep] - predictions[d_shallow]
sample_dist = dist_to_trace(scan_samples, scan_fault)
nug2, s1, l1, s2, l2 = cov_p
gamma = lambda h: nug2 + s1 * (1 - np.exp(-h / l1)) + s2 * (1 - np.exp(-h / l2))
noise_rows = []
for lo, hi in zip(noise_bins_km[:-1], noise_bins_km[1:]):
    in_bin = (sample_dist >= lo) & (sample_dist < hi)
    h_mid = 0.5 * (lo + hi)
    noise_rows.append(dict(dist_lo_km=lo, dist_hi_km=hi, n_samples=int(in_bin.sum()),
                           median_abs_diff_mm=float(np.median(np.abs(model_diff[in_bin]))) if in_bin.any() else np.nan,
                           noise_sd_mm=float(np.sqrt(nug2 + s1 + s2)),
                           noise_pair_rms_mm=float(np.sqrt(2 * gamma(h_mid)))))
depth_vs_noise = pd.DataFrame(noise_rows)
depth_vs_noise.to_csv(os.path.join(out_dir, "final_depth_vs_noise.csv"), index=False)
print(f"model difference, D {d_deep:g} km minus D {d_shallow:g} km (fault-parallel):")
print(depth_vs_noise.round(3).to_string(index=False))

# %% [markdown]
# ## [11] Family 3: synthetic recovery through this pipeline
#
# 10 mm of uniform slip from the surface to D over the scan window, forward
# through the model mesh to the samples (noise drawn from the full data
# covariance) and to the surface constraint (noise from its errors), then
# inverted exactly as the baseline (model sigma, tie 0.2 km, surface constraint).

# %%
## [11]
recovery_rng = np.random.default_rng(7)
recovery_rows, recovery_profiles = [], []
for depth_bottom in recovery_depths:
    rec_fault = wrap_fault(*unrolled["smooth12"], dip_deg, f"recovery {depth_bottom:g}")
    rec_table = slip_table(rec_fault)
    in_window = np.clip(np.minimum(rec_table.km_pp - scan_km[0], scan_km[1] - rec_table.km_pp)
                        / scan_taper + 1, 0, 1).values
    slip_input = 10.0 * in_window * (rec_table.z_c.values < depth_bottom)
    rec_ds = get_samples()
    rec_fault.buildGFs(rec_ds, vertical=True, slipdir="s", verbose=False)
    chol = np.linalg.cholesky(rec_ds.Cd + 1e-9 * np.eye(rec_ds.Cd.shape[0]))
    rec_ds.vel[:] = (np.dot(-rec_fault.G[rec_ds.name]["strikeslip"], slip_input)
                     + np.dot(chol, recovery_rng.standard_normal(rec_ds.vel.size)))
    rec_surface = make_surface_interp(rec_fault, surf_floor_mm)
    rec_fault.buildGFs(rec_surface, slipdir="s", verbose=False)
    rec_surface.vel[:] = (np.dot(rec_fault.G[rec_surface.name]["strikeslip"], slip_input)
                          + np.sqrt(np.diag(rec_surface.Cd)) * recovery_rng.standard_normal(rec_surface.vel.size))
    invert(rec_fault, [rec_ds], sigma_model, surface=rec_surface, tie_depth_km=tie_depth_km)
    area_km2 = np.array([0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])) for p in rec_fault.patch])
    input_potency = float(np.sum(area_km2 * 1e6 * slip_input * 1e-3))
    row = model_numbers(rec_fault, rec_ds, rec_surface)
    recovery_rows.append(dict(D_km=depth_bottom, moment_recovery=row["potency_m3"] / input_potency,
                              mw_input=(2.0 / 3.0) * (np.log10(shear_modulus * input_potency) - 9.1),
                              mw_recovered=row["Mw"], peak_recovered_mm=row["peak_slip_mm"],
                              peak_depth_m=row["peak_depth_m"], surface_pct=row["surface_pct"]))
    out_table = slip_table(rec_fault)
    out_table["input_mm"], out_table["D_km"] = slip_input, depth_bottom
    recovery_profiles.append(out_table)
    print(f"input 10 mm to {depth_bottom:g} km: peak {row['peak_slip_mm']:.1f} mm at {row['peak_depth_m']:.0f} m, "
          f"moment recovered {row['potency_m3'] / input_potency:.2f}")
recovery = pd.DataFrame(recovery_rows)
recovery.to_csv(os.path.join(out_dir, "final_recovery.csv"), index=False)
pd.concat(recovery_profiles).to_csv(os.path.join(out_dir, "final_recovery_profiles.csv"), index=False)

# %% [markdown]
# ## [12] Families 4 and 5: isolation rows and sampling
#
# Each switches one baseline choice: no surface constraint; the unlevelled
# field with a plane (1, x, y) solved per dataset in its place; surface error
# floors 0.25 and 1.0 mm; quadtree floors 40 and 60 m; a uniform 100 m grid.

# %%
## [12]
run_case("isolation", "insar_only", use_surface=False)
run_case("isolation", "unlevelled_plane", levelled=False, ramp_spec="1,x,y")
for floor in surf_floors:
    run_case("isolation", f"floor{floor:g}", floor=floor)
run_case("sampling", "minblock40", min_km=0.04)
run_case("sampling", "minblock60", min_km=0.06)
run_case("sampling", "uniform100", min_km=0.1, start_km=0.1)
save_suite()

# %% [markdown]
# ## [13] Family 6: dip
#
# The mesh hinged east about the fixed surface trace, 60 to 105 degrees
# (105 = 75 W), at the model sigma. The quadtree follows each geometry.

# %%
## [13]
for dip in dips_deg:
    run_case("dip", f"dip{dip}", dip=dip)
save_suite()

# %% [markdown]
# ## [14] Family 7: correlation lengths, each at its own L-curve corner
#
# In CSI (`Fault.buildCmXY`) Cm = sigma^2 (lam0^2 / (lamx lamz))
# exp(-0.5 dx/lamx) exp(-0.5 dz/lamz): lambda sets the shape AND enters the
# amplitude, so the model sigma is the corner only for the model's pair. Each pair
# gets its own L-curve; the suite row is the model at its own pick (and the
# baseline pair's curve, on a coarser grid, is a check on 07a_uniform_model's corner).

# %%
## [14]
lambda_corners = []
for lam in lam_grid:
    label = f"lam{lam[0]:g}x{lam[1]:g}"
    _, corner, pick, pick_row = lcurve(label, lam=lam)
    pick_row.update(family="lam", label=label)
    suite_rows.append(pick_row)
    lambda_corners.append(dict(lam_s=lam[0], lam_z=lam[1], corner=corner, pick=pick, Mw=pick_row["Mw"]))
save_suite()
lambda_corners = pd.DataFrame(lambda_corners)
print(lambda_corners.round(3).to_string(index=False))

# %% [markdown]
# ## [15] Family 8: mesh, each at its own L-curve corner
#
# Surface cells 100, 50 and 30 m with a plateau, the smooth x1.2 growth of
# the baseline (its curve is in [14]), and a plateau of 100 m cells to 1.5 km.
# The prior is per cell, so each mesh gets its own L-curve.

# %%
## [15]
mesh_corners = []
for mesh_name in mesh_variants:
    if mesh_name == "smooth12":
        continue
    _, corner, pick, pick_row = lcurve(mesh_name, mesh_name=mesh_name)
    pick_row.update(family="mesh", label=mesh_name)
    suite_rows.append(pick_row)
    mesh_corners.append(dict(mesh=mesh_name, corner=corner, pick=pick, Mw=pick_row["Mw"]))
save_suite()
mesh_corners = pd.DataFrame(mesh_corners)
print(mesh_corners.round(3).to_string(index=False))

# %% [markdown]
# ### CHECK -- repeatable, and the same model as 07a_uniform_model
#
# The baseline case run again must reproduce the tie family's 0.2 km row
# exactly; and where `07a_uniform_model` has run on the same inputs, its slip
# table must equal this notebook's baseline to round-off (same machinery,
# same data, same sigma).

# %%
## [16]
repeat_row, repeat_fault, _, _ = run_case("gate", "baseline_repeat")
assert abs(repeat_row["chi2"] - baseline_row["chi2"]) < 1e-6 and abs(repeat_row["Mw"] - baseline_row["Mw"]) < 1e-9, \
    "the baseline configuration did not reproduce exactly"
print("PASS: the baseline configuration reproduces exactly")
baseline_file = os.path.join(baseline_dir, "slip_homogeneous_baseline.csv")
if os.path.exists(baseline_file):
    baseline_slip = pd.read_csv(baseline_file).slip_mm.values
    slip_diff = float(np.max(np.abs(baseline_slip - repeat_fault.slip[:, 0])))
    print(f"max |slip - 07a_uniform_model baseline| = {slip_diff:.2e} mm")
    assert slip_diff < 1e-6, "this notebook's baseline differs from 07a_uniform_model's"
    print("PASS: same model as 07a_uniform_model")
else:
    print("SKIPPED: 07a_uniform_model has not written slip_homogeneous_baseline.csv yet")
save_suite()

# %% [markdown]
# ## [17] The collated figure
#
# A Mw across the families (baseline dashed). B slip with depth at the peak
# column for each tie depth. C the fixed-depth scans. D the synthetic
# recovery. E the L-curves of the lambda pairs, corners marked. F dip: rms
# and Mw.

# %%
## [17]
mpl_paper_style(base=7)
suite = pd.read_csv(os.path.join(out_dir, "final_suite.csv"))
suite = suite[suite.family != "gate"]
fig = plt.figure(figsize=(7.25, 6.6), constrained_layout=True)
grid = fig.add_gridspec(3, 3)
family_colors = {"tie": "#000000", "isolation": "#D55E00", "sampling": "#56B4E9", "dip": "#0072B2",
                 "lam": "#CC79A7", "mesh": "#E69F00"}

ax = fig.add_subplot(grid[0, :])
x_pos, ticks, tick_labels = 0, [], []
for family, color in family_colors.items():
    group = suite[suite.family == family]
    xs = np.arange(x_pos, x_pos + len(group))
    ax.plot(xs, group.Mw, "o", ms=4, color=color)
    ticks += list(xs); tick_labels += list(group.label)
    if len(group):
        ax.text(xs.mean(), 1.02, family, transform=ax.get_xaxis_transform(), ha="center", fontsize=6)
        ax.axvline(x_pos + len(group) - 0.5, color="0.85", lw=0.6)
    x_pos += len(group)
ax.axhline(baseline_row["Mw"], color="0.4", ls="--", lw=0.7)
ax.set_xticks(ticks); ax.set_xticklabels(tick_labels, rotation=60, ha="right", fontsize=5)
ax.set_ylabel(r"$M_W$ (30 GPa)")
ax.set_title(r"$\bf{A}$  Moment across the tests", loc="left", fontsize=7, pad=12)

ax = fig.add_subplot(grid[1, 0])
for tie, table in tie_tables.items():
    peak_s = table.loc[table.slip_mm.idxmax(), "s_km"]
    column = table[(table.s_km - peak_s).abs() < 0.3]
    by_depth = column.groupby(np.round(column.z_c, 3)).slip_mm.max()
    ax.plot(by_depth.values, by_depth.index, "o-", ms=2, label=f"tie {tie * 1000:.0f} m")
ax.set_ylim(1.5, 0); ax.set_xlabel("Slip (mm)"); ax.set_ylabel("Depth (km)")
ax.legend(fontsize=5)
ax.set_title(r"$\bf{B}$  Top tie, sigma " + f"{sigma_model:g}", loc="left", fontsize=7)

ax = fig.add_subplot(grid[1, 1])
for tag, color in (("smooth12", "#000000"), ("plateau100", "#0072B2"), ("plateau100_z5", "#56B4E9"),
                   ("unlevelled", "0.6")):
    ax.plot(scans[tag].D_km, scans[tag].dchi2, "o-", ms=2, color=color, label=tag)
ax.axhline(4, color="0.5", ls=":", lw=0.7)
ax.set_xscale("log"); ax.set_ylim(-2, min(100, scans["smooth12"].dchi2.max() * 1.05))
ax.set_xlabel("Bottom depth D (km)"); ax.set_ylabel(r"$\Delta\chi^2$")
ax.legend(fontsize=5)
ax.set_title(r"$\bf{C}$  Fixed-depth scan", loc="left", fontsize=7)

ax = fig.add_subplot(grid[1, 2])
profiles_all = pd.concat(recovery_profiles)
for depth_bottom, group in profiles_all.groupby("D_km"):
    peak_s = group.loc[group.slip_mm.idxmax(), "s_km"]
    column = group[(group.s_km - peak_s).abs() < 0.3]
    by_depth = column.groupby(np.round(column.z_c, 3)).slip_mm.max()
    ax.plot(by_depth.values, by_depth.index, "o-", ms=2, label=f"to {depth_bottom:g} km")
ax.axvline(10, color="0.6", ls="--", lw=0.6)
ax.set_ylim(3, 0); ax.set_xlabel("Recovered slip (mm)"); ax.set_ylabel("Depth (km)")
ax.legend(fontsize=4.5, ncols=2)
ax.set_title(r"$\bf{D}$  Recovery of 10 mm", loc="left", fontsize=7)

ax = fig.add_subplot(grid[2, 0:2])
curves = pd.DataFrame(lcurve_rows)
for label, group in curves[curves.curve.str.startswith("lam")].groupby("curve", sort=False):
    ax.plot(np.sqrt(group.chi2_total), group.prior_norm, "o-", ms=2, lw=0.8, label=label)
    pick = group[group.sigma == group.pick.iloc[0]]
    ax.plot(np.sqrt(pick.chi2_total), pick.prior_norm, "k*", ms=6)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"Weighted misfit $\sqrt{\chi^2}$"); ax.set_ylabel(r"Roughness $\sqrt{m^T C^{-1} m}$")
ax.legend(fontsize=5)
ax.set_title(r"$\bf{E}$  L-curves of the correlation lengths (star: pick)", loc="left", fontsize=7)

ax = fig.add_subplot(grid[2, 2])
dips = suite[suite.family == "dip"].sort_values("dip")
ax.plot(dips.dip, dips.rms_mm, "o-", ms=3, color="#0072B2", label="rms")
ax.plot(dips.dip, dips.rms_near_mm, "o--", ms=3, color="#0072B2", label="rms < 200 m")
ax.set_xlabel("Dip (deg E)"); ax.set_ylabel("rms (mm)")
ax2 = ax.twinx(); ax2.plot(dips.dip, dips.Mw, "s-", ms=3, color="#D55E00"); ax2.set_ylabel(r"$M_W$", color="#D55E00")
ax.legend(fontsize=5)
ax.set_title(r"$\bf{F}$  Dip", loc="left", fontsize=7)

fig_file = os.path.join(fig_dir, "fig07_sensitivity_final.png")
fig.savefig(fig_file, dpi=300); fig.savefig(fig_file.replace(".png", ".pdf"))
print("wrote", fig_file)

# %% [markdown]
# ## [17b] Inputs for a stress-drop estimate (saved, not reported)
#
# Depth is not constrained, so no stress drop is reported (Dani 2026-09-29);
# a ballpark is quoted as a range. Following Materna et al. (2024, Text S3),
# stress drop = mu d / x with d the slip and x half the depth extent. For the
# distributed model (x = half the depth holding 95% of its potency) and for
# every uniform-slip depth D (x = D / 2), mu is the SFCVM v26.0 rigidity
# weighted by that model's own potency (as 07a_uniform_model), and d is the mean
# band-stack event slip of notebook 04. Written to
# `stress_drop_inputs.csv`; no registry rows.

# %%
## [17b]
sfcvm = pd.read_csv(os.path.join(out_dir, "sfcvm_creep_section_profiles_v26-0.csv"))
mu_by_depth = sfcvm.groupby("depth_m").mu_gpa.median()
registry = pd.read_csv(common_paths["registry_csv"], dtype=str).groupby("key").last()
d_band_mm = float(registry.loc["04_band90_steps_mean_mm", "value"])
def mu_weighted(table, slip_mm):
    """SFCVM rigidity weighted by potency (GPa), floored at 0.5 GPa as 07a_uniform_model."""
    mu_cell = np.maximum(np.interp(table.z_c.values * 1000, mu_by_depth.index.values, mu_by_depth.values), 0.5)
    return float(np.sum(mu_cell * slip_mm) / np.sum(slip_mm))
stress_rows = []
base_table = tie_tables[tie_depth_km]
order = np.argsort(base_table.z_c.values)
cum = np.cumsum(base_table.slip_mm.values[order]) / base_table.slip_mm.sum()   # slip-weighted (cell areas are similar at the top)
depth95_km = float(base_table.z_c.values[order][np.searchsorted(cum, 0.95)])
mu_dist = mu_weighted(base_table, base_table.slip_mm.values)
stress_rows.append(dict(model="distributed (the model)", depth_extent_km=depth95_km, mu_gpa=mu_dist, d_mm=d_band_mm,
                        stress_drop_kpa=mu_dist * 1e9 * d_band_mm * 1e-3 / (depth95_km * 500) / 1e3, dchi2=np.nan))
scan_table = slip_table(scan_fault)
in_window_scan = np.clip(np.minimum(scan_table.km_pp - scan_km[0], scan_km[1] - scan_table.km_pp) / scan_taper + 1, 0, 1).values
for r in record_scan.itertuples():
    shape = in_window_scan * (scan_table.z_c.values < r.D_km)
    if shape.sum() == 0:
        continue
    mu_d = mu_weighted(scan_table, shape)
    stress_rows.append(dict(model=f"uniform slip to {r.D_km:g} km", depth_extent_km=r.D_km, mu_gpa=mu_d, d_mm=d_band_mm,
                            stress_drop_kpa=mu_d * 1e9 * d_band_mm * 1e-3 / (r.D_km * 500) / 1e3, dchi2=r.dchi2))
stress_inputs = pd.DataFrame(stress_rows)
stress_inputs.to_csv(os.path.join(out_dir, "stress_drop_inputs.csv"), index=False)
accepted = stress_inputs[(stress_inputs.dchi2 <= 4) | stress_inputs.dchi2.isna()]
print(accepted.round(2).to_string(index=False))
print(f"ballpark (distributed model and the accepted uniform-slip depths): "
      f"{accepted.stress_drop_kpa.min():.0f}-{accepted.stress_drop_kpa.max():.0f} kPa")

# %% [markdown]
# ## [18] Registry -- the sensitivity numbers

# %%
## [18]
by_family = {family: group for family, group in suite.groupby("family")}
tie_rows = by_family["tie"].set_index("tie_km")
sidecar = REG.sidecar("07_tests_final", [
    f"baseline: fault-parallel, smooth12, dip {dip_deg}, sigma {sigma_model}, lambda {lam_xz}, tie {tie_depth_km} km: "
    f"Mw {baseline_row['Mw']:.3f}, rms {baseline_row['rms_mm']:.3f} mm, surface {baseline_row['surface_pct']:.0f}%",
    "tie depth: " + "; ".join(f"{t * 1000:.0f} m Mw {r.Mw:.3f} peak {r.peak_slip_mm:.1f} mm at {r.peak_depth_m:.0f} m "
                              f"surface {r.surface_pct:.0f}%" for t, r in tie_rows.iterrows()),
    f"depth scan (smooth12): best {record_scan.loc[record_scan.chi2.idxmin(), 'D_km']:g} km, "
    f"bracket {d_shallow:g}-{d_deep:g} km, Mw {bracket.Mw.min():.2f}-{bracket.Mw.max():.2f}",
    "depth scan variants: " + "; ".join(
        f"{tag} {scan[scan.dchi2 <= 4].D_km.min():g}-{scan[scan.dchi2 <= 4].D_km.max():g} km" for tag, scan in scans.items()),
    "recovery (moment fraction): " + "; ".join(f"{r.D_km:g} km {r.moment_recovery:.2f}" for r in recovery.itertuples()),
    f"dip Mw {by_family['dip'].Mw.min():.3f}-{by_family['dip'].Mw.max():.3f}",
    "lambda at own corners: " + "; ".join(f"({r.lam_s:g}, {r.lam_z:g}) corner {r.corner:.1f} pick {r.pick:g} Mw {r.Mw:.3f}"
                                          for r in lambda_corners.itertuples()),
    "mesh at own corners: " + "; ".join(f"{r.mesh} corner {r.corner:.1f} pick {r.pick:g} Mw {r.Mw:.3f}"
                                        for r in mesh_corners.itertuples()),
])
REG.register("07b_n_samples", baseline_row["n_samples"], "count", 0, where="M7", cell="baseline", sidecar=sidecar)
for tie, r in tie_rows.iterrows():
    REG.register(f"07b_tie{tie * 1000:.0f}m_mw", r.Mw, "Mw", 2, where="M7", cell="tie", sidecar=sidecar)
    REG.register(f"07b_tie{tie * 1000:.0f}m_surface_pct", r.surface_pct, "%", 0, where="M7", cell="tie", sidecar=sidecar)
REG.register("07b_depth_best_km", record_scan.loc[record_scan.chi2.idxmin(), "D_km"], "km", 2, where="M7", cell="depth-scan", sidecar=sidecar)
REG.register("07b_depth_lo_km", d_shallow, "km", 2, where="M7", cell="depth-scan", sidecar=sidecar)
REG.register("07b_depth_hi_km", d_deep, "km", 2, where="M7", cell="depth-scan", sidecar=sidecar)
REG.register("07b_depth_mw_lo", bracket.Mw.min(), "Mw", 2, where="M7", cell="depth-scan", sidecar=sidecar)
REG.register("07b_depth_mw_hi", bracket.Mw.max(), "Mw", 2, where="M7", cell="depth-scan", sidecar=sidecar)
REG.register("07b_dip_mw_min", by_family["dip"].Mw.min(), "Mw", 2, where="M7", cell="dip", sidecar=sidecar)
REG.register("07b_dip_mw_max", by_family["dip"].Mw.max(), "Mw", 2, where="M7", cell="dip", sidecar=sidecar)
REG.register("07b_lam_mw_min", lambda_corners.Mw.min(), "Mw", 2, where="M7", cell="lambda", sidecar=sidecar)
REG.register("07b_lam_mw_max", lambda_corners.Mw.max(), "Mw", 2, where="M7", cell="lambda", sidecar=sidecar)
REG.register("07b_mesh_mw_min", mesh_corners.Mw.min(), "Mw", 2, where="M7", cell="mesh", sidecar=sidecar)
REG.register("07b_mesh_mw_max", mesh_corners.Mw.max(), "Mw", 2, where="M7", cell="mesh", sidecar=sidecar)
for r in recovery.itertuples():
    REG.register(f"07b_recovery_{r.D_km * 1000:.0f}m", r.moment_recovery, "fraction", 2, where="M7", cell="recovery", sidecar=sidecar)
REG.render()
print("07b_uniform_model_tests (final configuration) complete")
