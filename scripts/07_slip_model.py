# %% [markdown]
# # 07 Slip model of the mean event -- the homogeneous baseline
#
# ```
# Created on Thu Sep 25 2026; rewritten on Mon Sep 28 2026 as the clean baseline
# @author: daniellelindsay
# ```
#
# The best slip model with **uniform-rock (homogeneous half-space) Green's
# functions**, with every lesson of the 2026-09-25 to 28 tests applied
# (`07_sweep_lessons_2026-09-28.md`, `07_slip_model_tests.ipynb`,
# `drafts/07/`). It is the baseline that a 1D layered model (PyLith Green's
# functions from the USGS SFCVM profile) and later a 3D model have to beat,
# measured the same way. The rock enters the INVERSION as uniform; the USGS
# rigidity only rescales the moment afterwards (Mw with SFCVM mu(z)).
# Registry keys are prefixed `07_hom_`.
#
# | Choice | Value | Evidence |
# |---|---|---|
# | Data | fault-parallel mean event (06 `fp_maps.npz`, five events) | residual maps: removes the ascending east-side patch |
# | Decomposition strike | mean local strike of the SSE section, 146.44° | 06; Dani 2026-09-28 |
# | Coherence | 0.9 for the field | 06 `coh_maps`; Dani 2026-09-28 |
# | Levelling | plane fitted to the far field (section ± 1 km masked) | ascending tilt ~2.5 mm |
# | Fault | vertical, on the hand-drawn trace | dip unresolved 60°E-75°W, Mw 3.82-3.89 |
# | Mesh | `smooth12`: 50 m cells at the surface, x1.2 per row, 3 km | no resolution plateau |
# | Sampling | distance-based quadtree, 2 km to one 30 m pixel | |
# | Prior | `buildCmXY`, lambda (0.3, 0.05) km, sigma at the L-curve corner, leaning under-smoothed (12) | `drafts/07/sigma_lcurve.py` |
# | Surface | crossing-stack slip interpolated to every surface triangle | 40% -> 86% of the crossing slip |
# | Top tie | uniform slip 0-0.2 km per 0.1 km column | removes the sub-surface overshoot |
# | Solver | NNLS (right-lateral >= 0), per-dataset constant | |
#
# The machinery (mesh, levelling, noise model, quadtree, surface
# constraint, tie, inversion) is ported verbatim from
# `07_slip_model_tests.py`, where it is pinned by the zero-difference mesh
# gate and the m02 parity gate; CHECK [7] below pins this notebook to the
# draft the tests were run on. CSI calls follow the tutorials
# (`external/csi/notebooks/`).

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

warnings.simplefilter("ignore")
from TulePond_filepaths import common_paths, repo_dir
import insar_utils as utils
from insar_utils import load_trace, mpl_paper_style

# CSI: Classic Slip Inversion (Jolivet et al.; external/csi at 1277bb4)
from csi import TriangularPatches, insar, surfaceslip
from csi import imagedownsampling as imdown
from csi import transformation as transform
from csi import multisourcesolve as multisrc

notebook = "07_slip_model"
out_dir = os.path.join(common_paths["out_dir"], notebook)
tests_dir = os.path.join(common_paths["out_dir"], "07_slip_model_tests")
field_dir = os.path.join(common_paths["out_dir"], "06_event_field_geometry")
events_dir = os.path.join(common_paths["out_dir"], "04_event_detection")
fig_dir = common_paths["fig_dir"]
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# local frame centre (csi_io convention: the creep-section centre)
frame_lon0, frame_lat0 = -121.968, 37.557

# data selection
swath_km       = 4.0      # keep pixels within this distance of the trace
exclude_km     = 0.03     # drop the pixel straddling the trace
km_window      = (52.0, 67.0)   # along-strike data window (km from Point Pinole)
start_block_km = 2.0      # quadtree starting block
min_block_km   = 0.03     # quadtree minimum block = one 30 m pixel
cov_mask_km    = 1.0      # covariance from pixels beyond this distance
cov_distmax_km = 8.0
nug_power      = 0.5      # nugget variance / n_pixels**nug_power per block
near_field_km  = 0.2      # "near field" for the fit metrics

# levelling and surface constraint
section_lengths = pd.read_csv(os.path.join(events_dir, "section_lengths.csv")).set_index("name")
creep_lo_km = float(section_lengths.loc["tulepond_creep_length", "km_lo"])   # the SSE section (04)
creep_hi_km = float(section_lengths.loc["tulepond_creep_length", "km_hi"])
align_mask_km  = 1.0      # slip section +/- this masked for the far-field plane
surf_floor_mm  = 0.5      # crossing-stack error floor
surf_step_km   = 0.2
surf_win_km    = 0.4

# mesh, prior, tie
mesh_lat_lo, mesh_lat_hi = 37.47, 37.65
mesh_depth_km  = 3.0
mesh_params    = (0.05, 0.0, 1.2, 1.5)   # smooth12: h_top, const_to, growth, h_bot (km)
dip_deg        = 90
fine_km        = (56.0, 63.0)
far_rate       = 0.25
smooth_iters   = 8
lam_xz         = (0.3, 0.05)   # prior correlation lengths along strike, down dip (km)
tie_depth_km   = 0.2           # slip uniform from the surface to this depth
shear_modulus  = 3.0e10        # uniform rigidity of the reference Mw
mu_floor_gpa   = 0.5           # SFCVM rigidity floor (the model is ~0 at the surface)
sigma_grid     = [4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128]   # L-curve (larger = LESS smoothing)
sigma_check    = 24            # sigma of the draft the CHECK compares against
sigma_fixed    = 12            # Dani 2026-09-28: sigma fixed at 12 (the L-curve pick on the data
                               # through 2025, corner 10.1); the L-curve is still run and its pick reported
sfcvm_file     = os.path.join(tests_dir, "sfcvm_creep_section_profiles.csv")   # drafts/07/sfcvm_extract.py
assert os.path.exists(sfcvm_file), (f"missing {sfcvm_file}: download USGS_SFCVM_v21-1_detailed.h5 "
                                    "(doi:10.5066/P9TRDCHE) and run drafts/07/sfcvm_extract.py first")
n_boot         = 24

REG = utils.Registry(notebook, common_paths, repo_dir)
REG.inputs(os.path.join(field_dir, "fp_maps.npz"),
           common_paths["fault_trace"], common_paths["fault_profiles"],
           os.path.join(events_dir, "event_fraction_by_crossing.csv"),
           os.path.join(events_dir, "creep_events.csv"),
           os.path.join(events_dir, "section_lengths.csv"),
           sfcvm_file)
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
    [07_slip_model_tests, verbatim]"""
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
    in place). [07_slip_model_tests, verbatim]"""
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
    [07_slip_model_tests, verbatim]"""
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
    """Two exponentials + nugget/blocksize. [m02; 07_slip_model_tests, verbatim]"""
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
    error = 1.4826 MAD / sqrt(n) floored at floor_mm. [07_slip_model_tests, verbatim]"""
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
    surface triangle of this fault. [07_slip_model_tests, verbatim]"""
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
# [07_slip_model_tests, verbatim]

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
# ### CHECK -- this notebook reproduces the draft the tests were run on
#
# At sigma = 24 the slip must equal `drafts/07/fig2_model_row.py`'s
# product (pinned as `fig2_model_row_draft_sigma24_reference.npz`, made on
# the data through 2025) to round-off. The gate applies only when the
# quadtree samples are the draft's (same positions, data equal up to the
# fitted constant); on new data it is skipped and says so. Every solve
# starts from a fresh fault and a copy of the samples:
# `removeTransformation` edits `vel` in place.

# %%
## [7]
draft_file = os.path.join(tests_dir, "fig2_model_row_draft_sigma24_reference.npz")
same_data = False
if os.path.exists(draft_file):
    draft = np.load(draft_file)
    same_data = (draft["sample_lon"].size == samples.lon.size
                 and np.array_equal(draft["sample_lon"], samples.lon)
                 and np.array_equal(draft["sample_lat"], samples.lat)
                 and np.ptp(samples.vel - draft["data_mm"]) < 1e-9)
if same_data:
    fault = wrap_fault(mesh_nodes, mesh_faces, dip_deg, "mean event")
    ds = copy.deepcopy(samples)
    surface = make_surface_interp(fault, surf_floor_mm)
    invert(fault, [ds], sigma_check, surface=surface, tie_depth_km=tie_depth_km)
    slip_diff = float(np.max(np.abs(fault.slip[:, 0] - draft["slip_mm"])))
    print(f"sigma {sigma_check}: max |slip - draft| = {slip_diff:.2e} mm over {len(mesh_faces)} cells")
    assert slip_diff < 1e-6, "the clean notebook does not reproduce the tested draft"
    print("PASS: the clean notebook reproduces the draft model")
else:
    slip_diff = np.nan
    print("SKIPPED: the input data differ from the draft's (new data); the gate does not apply")

# %% [markdown]
# ## [8] Sigma: the L-curve corner (Hansen 1992)
#
# One inversion per sigma on the grid (4-128; ported from
# `drafts/07/sigma_lcurve.py`, and equal to it on the data through 2025). In
# CSI sigma is the prior standard deviation of slip (Cm = sigma^2
# exp(-d/lambda)): LARGER sigma = WEAKER smoothing. The L-curve plots the
# two terms the solver trades off: the full weighted misfit
# sqrt(chi2 InSAR + chi2 surface) against the prior-weighted roughness
# sqrt(m^T C_corr^-1 m), both log. Corner = maximum curvature (cubic spline
# in log sigma, spline end intervals excluded). Selected: the smallest grid
# sigma at or above the corner, i.e. the corner leaning to under-smoothing
# (Dani 2026-09-28). Larger sigma piles slip into a band just below the
# 0.2 km tie (sigma >= 48; `test_sigma_lcurve.png`).

# %%
## [8]
from scipy.interpolate import CubicSpline

lcurve_rows = []
for sigma in sigma_grid:
    fault = wrap_fault(mesh_nodes, mesh_faces, dip_deg, f"lcurve {sigma}")
    ds = copy.deepcopy(samples)
    surface = make_surface_interp(fault, surf_floor_mm)
    invert(fault, [ds], sigma, surface=surface, tie_depth_km=tie_depth_km)
    insar_resid, surf_resid = ds.vel - ds.synth, surface.vel - surface.synth
    chi2_insar = float(np.dot(insar_resid, np.linalg.solve(ds.Cd, insar_resid)))
    chi2_surf = float(np.dot(surf_resid, np.linalg.solve(surface.Cd, surf_resid)))
    slip_vec = fault.slip[:, 0]
    prior_norm = float(np.sqrt(np.dot(slip_vec, np.linalg.solve(fault.Cm / sigma ** 2, slip_vec))))
    potency = float(np.sum(cell_area_km2 * 1e6 * slip_vec * 1e-3))
    surface_cells = np.array([0 in p[:, 2] for p in fault.patch])
    cell_km = lat_to_km_pp(np.array([pll.mean(0) for pll in fault.patchll])[:, 1])
    in_creep = surface_cells & (cell_km >= creep_lo_km) & (cell_km <= creep_hi_km)
    lcurve_rows.append(dict(sigma=sigma, chi2=chi2_insar, chi2_surf=chi2_surf,
                            chi2_total=chi2_insar + chi2_surf, prior_norm=prior_norm,
                            rms_mm=float(np.sqrt(np.mean(insar_resid ** 2))),
                            surface_pct=float(100 * np.median(slip_vec[in_creep]) / crossing_median),
                            peak_slip_mm=float(slip_vec.max()),
                            Mw=(2.0 / 3.0) * (np.log10(shear_modulus * potency) - 9.1)))
    print(f"  sigma {sigma:>4g}: chi2 {chi2_insar:.0f} + {chi2_surf:.0f}, roughness {prior_norm:.1f}, "
          f"Mw {lcurve_rows[-1]['Mw']:.2f}")
sigma_family = pd.DataFrame(lcurve_rows)
sigma_family.to_csv(os.path.join(out_dir, "sigma_lcurve.csv"), index=False)
if same_data:                                          # CHECK: equal to the draft L-curve
    draft_lcurve = pd.read_csv(os.path.join(tests_dir, "sigma_lcurve.csv"))
    lcurve_diff = float(np.max(np.abs(sigma_family.chi2_total.values - draft_lcurve.chi2_total.values)))
    assert lcurve_diff < 1e-6, f"L-curve differs from the draft by {lcurve_diff:.2e}"
    print(f"PASS: L-curve equals drafts/07/sigma_lcurve.py (max |chi2 diff| {lcurve_diff:.1e})")
print(sigma_family[["sigma", "chi2_total", "prior_norm", "rms_mm", "surface_pct",
                    "peak_slip_mm", "Mw"]].round(3).to_string(index=False))
log_sigma = np.log10(sigma_family.sigma.values)
spline_misfit = CubicSpline(log_sigma, np.log10(np.sqrt(sigma_family.chi2_total.values)))
spline_rough = CubicSpline(log_sigma, np.log10(sigma_family.prior_norm.values))
fine_log_sigma = np.linspace(log_sigma[1], log_sigma[-2], 400)
d_misfit, dd_misfit = spline_misfit(fine_log_sigma, 1), spline_misfit(fine_log_sigma, 2)
d_rough, dd_rough = spline_rough(fine_log_sigma, 1), spline_rough(fine_log_sigma, 2)
curvature = np.abs(d_misfit * dd_rough - d_rough * dd_misfit) / (d_misfit ** 2 + d_rough ** 2) ** 1.5
sigma_corner = float(10 ** fine_log_sigma[np.argmax(curvature)])
sigma_rule = float(sigma_family.sigma[sigma_family.sigma >= sigma_corner].min())
sigma_sel = float(sigma_fixed)
print(f"L-curve corner: sigma {sigma_corner:.1f}; the rule picks grid sigma {sigma_rule:g}; "
      f"used: sigma {sigma_sel:g} (fixed)")
if sigma_rule != sigma_sel:
    print(f"NOTE: on this data the L-curve rule picks {sigma_rule:g}, not the fixed {sigma_sel:g}")

mpl_paper_style(base=7)
fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.3), constrained_layout=True)
ax = axes[0]
ax.plot(np.sqrt(sigma_family.chi2_total), sigma_family.prior_norm, "o-", color="0.4", ms=3, lw=0.8)
for sigma, x_val, y_val in zip(sigma_family.sigma, np.sqrt(sigma_family.chi2_total), sigma_family.prior_norm):
    ax.annotate(f"{sigma:g}", (x_val, y_val), xytext=(3, 2), textcoords="offset points", fontsize=5)
pick = (sigma_family.sigma == sigma_sel).values
ax.plot(np.sqrt(sigma_family.chi2_total[pick]), sigma_family.prior_norm[pick], "o", color="#D55E00", ms=5,
        label=f"selected, sigma {sigma_sel:g}")
ax.set_xscale("log"); ax.set_yscale("log")
from matplotlib.ticker import LogLocator, ScalarFormatter, NullFormatter
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 3, 5, 7)))
    axis.set_major_formatter(ScalarFormatter()); axis.set_minor_formatter(NullFormatter())
ax.set_xlabel(r"Weighted misfit $\sqrt{\chi^2}$"); ax.set_ylabel(r"Roughness $\sqrt{m^T C^{-1} m}$")
ax.legend(fontsize=5)
ax.set_title(r"$\bf{A}$  L-curve", loc="left", fontsize=7)
ax = axes[1]
ax.plot(10 ** fine_log_sigma, curvature, "k-", lw=0.8)
ax.axvline(sigma_corner, color="k", ls=":", lw=0.7)
ax.axvline(sigma_sel, color="#D55E00", lw=0.8)
ax.set_xscale("log"); ax.set_xlabel("sigma (mm)"); ax.set_ylabel("|curvature|")
ax.set_title(r"$\bf{B}$  Curvature", loc="left", fontsize=7)
fig.savefig(os.path.join(fig_dir, "fig07_hom_lcurve.png"), dpi=300)
fig.savefig(os.path.join(fig_dir, "fig07_hom_lcurve.pdf"))
print("wrote fig07_hom_lcurve.png/.pdf")

# %% [markdown]
# ## [9] The baseline model and its numbers
#
# Mw at 30 GPa is the reference. Mw with the USGS SFCVM rigidity: each
# cell's moment recomputed with the median mu(z) of the creep-section
# profiles (`drafts/07/sfcvm_extract.py`, floored at 0.5 GPa); the slip is
# the uniform-rock slip, so this is a rescale, not a layered inversion.

# %%
## [9]
fault = wrap_fault(mesh_nodes, mesh_faces, dip_deg, "mean event")
ds = copy.deepcopy(samples)
surface = make_surface_interp(fault, surf_floor_mm)
solver = invert(fault, [ds], sigma_sel, surface=surface, tie_depth_km=tie_depth_km)
nodes, faces, _ = fault.mesh_sz
table = pd.DataFrame(dict(
    lon=np.array([pll.mean(0) for pll in fault.patchll])[:, 0],
    lat=np.array([pll.mean(0) for pll in fault.patchll])[:, 1],
    s_km=nodes[faces].mean(1)[:, 0], z_c=nodes[faces].mean(1)[:, 1],
    z_top=nodes[faces][:, :, 1].min(1), z_bot=nodes[faces][:, :, 1].max(1),
    area_km2=cell_area_km2, slip_mm=fault.slip[:, 0]))
table["km_pp"] = lat_to_km_pp(table.lat.values)

profiles_sfcvm = pd.read_csv(sfcvm_file)
mu_by_depth = profiles_sfcvm.groupby("depth_m").mu_gpa.median()
table["mu_sfcvm_gpa"] = np.maximum(np.interp(table.z_c.values * 1000, mu_by_depth.index.values,
                                             mu_by_depth.values), mu_floor_gpa)
potency_m3 = float(np.sum(table.area_km2 * 1e6 * table.slip_mm * 1e-3))
moment_30 = shear_modulus * potency_m3
moment_sfcvm = float(np.sum(table.mu_sfcvm_gpa * 1e9 * table.area_km2 * 1e6 * table.slip_mm * 1e-3))

residual = ds.vel - ds.synth
near = dist_to_trace(ds, fault) <= near_field_km
in_creep_surface = (table.z_top <= 1e-6) & (table.km_pp >= creep_lo_km) & (table.km_pp <= creep_hi_km)
results = pd.Series(dict(
    sigma=sigma_sel, n_samples=ds.vel.size,
    chi2=float(np.dot(residual, np.linalg.solve(ds.Cd, residual))),
    rms_mm=float(np.sqrt(np.mean(residual ** 2))),
    rms_near_mm=float(np.sqrt(np.mean(residual[near] ** 2))),
    surf_rms_mm=float(np.sqrt(np.mean((surface.vel - surface.synth) ** 2))),
    surface_pct=float(100 * table.slip_mm[in_creep_surface].median() / crossing_median),
    peak_slip_mm=float(table.slip_mm.max()),
    peak_depth_m=float(1000 * table.z_c[table.slip_mm.idxmax()]),
    potency_m3=potency_m3,
    mw_30gpa=(2.0 / 3.0) * (np.log10(moment_30) - 9.1),
    mw_sfcvm=(2.0 / 3.0) * (np.log10(moment_sfcvm) - 9.1),
    mu_effective_gpa=moment_sfcvm / potency_m3 / 1e9))
print(results.to_string(float_format=lambda v: f"{v:.4g}"))

# %% [markdown]
# ### CHECK -- the model's fault-normal motion is small
#
# The fault-parallel decomposition assumes no horizontal motion
# perpendicular to the fault. The model's own fault-normal displacement at
# the samples (look vector rotated to the fault-normal bearing) must be
# small against its fault-parallel signal.

# %%
## [10]
look_normal = np.array([np.sin(np.deg2rad(strike_bearing + 90)),
                        np.cos(np.deg2rad(strike_bearing + 90)), 0.0])
normal = insar("fault-normal", lon0=frame_lon0, lat0=frame_lat0, verbose=False)
normal.read_from_binary(np.zeros(ds.vel.size), ds.lon, ds.lat, err=np.ones(ds.vel.size),
                        los=np.tile(look_normal, (ds.vel.size, 1)),
                        remove_nan=False, remove_zeros=False)
fault.buildGFs(normal, vertical=True, slipdir="s", verbose=False)
fault.G[normal.name]["strikeslip"] *= -1.0
normal.buildsynth([fault], direction="s")
normal_max_mm = float(np.max(np.abs(normal.synth)))
parallel_max_mm = float(np.max(np.abs(ds.synth)))
print(f"model fault-normal motion: max {normal_max_mm:.2f} mm "
      f"({100 * normal_max_mm / parallel_max_mm:.0f}% of the fault-parallel max "
      f"{parallel_max_mm:.2f} mm)")

# %% [markdown]
# ## [11] Bootstrap -- the formal error on Mw
#
# n_boot noise realisations from the full data covariances (Cholesky)
# added to the model's own synthetics, re-inverted at the selected sigma,
# each on a fresh fault and fresh copies of the data. The spread is the
# formal error of THIS configuration; the structural range across
# configurations is in `07_sweep_lessons_2026-09-28.md`.

# %%
## [11]
rng = np.random.default_rng(11)
chol_insar = np.linalg.cholesky(ds.Cd + 1e-9 * np.eye(ds.Cd.shape[0]))
chol_surface = np.linalg.cholesky(surface.Cd + 1e-9 * np.eye(surface.Cd.shape[0]))
synth_insar, synth_surface = ds.synth.copy(), surface.synth.copy()
boot_mw = []
for k in range(n_boot):
    boot_fault = wrap_fault(mesh_nodes, mesh_faces, dip_deg, f"boot {k}")
    boot_ds = copy.deepcopy(samples)
    boot_surface = make_surface_interp(boot_fault, surf_floor_mm)
    boot_ds.vel[:] = synth_insar + np.dot(chol_insar, rng.standard_normal(boot_ds.vel.size))
    boot_surface.vel[:] = synth_surface + np.dot(chol_surface, rng.standard_normal(boot_surface.vel.size))
    invert(boot_fault, [boot_ds], sigma_sel, surface=boot_surface, tie_depth_km=tie_depth_km)
    boot_potency = float(np.sum(cell_area_km2 * 1e6 * boot_fault.slip[:, 0] * 1e-3))
    boot_mw.append((2.0 / 3.0) * (np.log10(shear_modulus * boot_potency) - 9.1))
    print(f"  boot {k + 1}/{n_boot}: Mw {boot_mw[-1]:.3f}")
boot_mw = np.array(boot_mw)
mw_boot_sd = float(boot_mw.std(ddof=1))
print(f"bootstrap: Mw {boot_mw.mean():.3f} +/- {mw_boot_sd:.3f} (30 GPa, n={n_boot})")

# %% [markdown]
# ## [12] Figures
#
# A-C: fault-parallel data, model and residual at the quadtree samples.
# Then the slip distribution (single panel).

# %%
## [12]
mpl_paper_style(base=7)
trace_plot = np.c_[fault.lon, fault.lat]
fig, axes = plt.subplots(1, 3, figsize=(7.25, 3.0), constrained_layout=True)
panels = ((ds.vel, "Data (fault-parallel)"), (ds.synth, "Model"), (residual, "Residual"))
for ax, letter, (values, title) in zip(axes, "ABC", panels):
    sc = ax.scatter(ds.lon, ds.lat, c=values, s=4, cmap="RdBu_r", vmin=-3, vmax=3, linewidths=0)
    ax.plot(trace_plot[:, 0], trace_plot[:, 1], "k-", lw=0.5)
    ax.set_aspect(1 / np.cos(np.deg2rad(frame_lat0)))
    ax.set_xlim(-122.10, -121.84); ax.set_ylim(37.46, 37.65)
    ax.set_title(r"$\bf{" + letter + r"}$  " + title, loc="left", fontsize=7)
    if letter != "A":
        ax.set_yticklabels([])
fig.colorbar(sc, ax=axes, shrink=0.6, label="Fault-parallel displacement (mm)",
             orientation="horizontal", pad=0.04)
fig.savefig(os.path.join(fig_dir, "fig07_hom_fit_maps.png"), dpi=300)
print("wrote fig07_hom_fit_maps.png")

fig, ax = plt.subplots(figsize=(7.25, 2.4), constrained_layout=True)
pc = ax.tripcolor(nodes[:, 0], nodes[:, 1], faces, facecolors=table.slip_mm.values,
                  cmap="viridis", vmin=0, vmax=max(table.slip_mm.max(), 1), edgecolors="none")
ax.set_xlim(0, arc_total); ax.set_ylim(mesh_depth_km, 0)
ax.set_xlabel("Along-strike distance from north end (km)")
ax.set_ylabel("Depth (km)")
fig.colorbar(pc, ax=ax, label="Right-lateral slip (mm)", shrink=0.8)
ax.set_title(f"Homogeneous baseline, sigma {sigma_sel:g}: Mw {results.mw_30gpa:.2f} (30 GPa), "
             f"{results.mw_sfcvm:.2f} (SFCVM rigidity)", loc="left", fontsize=7)
fig.savefig(os.path.join(fig_dir, "fig07_hom_slip.png"), dpi=300)
fig.savefig(os.path.join(fig_dir, "fig07_hom_slip.pdf"))
print("wrote fig07_hom_slip.png/.pdf")

# %% [markdown]
# ## [13] Products
#
# The slip table, and Fig 2's model row (`fig2_model_row.npz`, read by
# notebook 06), in the format of the draft it replaces.

# %%
## [13]
table.to_csv(os.path.join(out_dir, "slip_homogeneous_baseline.csv"), index=False)
np.savez(os.path.join(tests_dir, "fig2_model_row.npz"),
         sample_lon=ds.lon, sample_lat=ds.lat, data_mm=ds.vel, model_mm=ds.synth,
         residual_mm=residual, nodes_sz=nodes, faces=faces, slip_mm=table.slip_mm.values,
         s0_km_pp=float(lat_to_km_pp([ref_fault.lati[0]])[0]),
         mw_30gpa=results.mw_30gpa, mw_sfcvm=results.mw_sfcvm, rms_mm=results.rms_mm,
         surf_ratio_pct=results.surface_pct, sigma=sigma_sel, strike_deg=strike_bearing,
         tie_depth_km=tie_depth_km)
print("wrote slip_homogeneous_baseline.csv and fig2_model_row.npz")

# %% [markdown]
# ## [14] Registry -- the homogeneous baseline (not the final model)

# %%
## [14]
sidecar = REG.sidecar("07_hom_baseline", [
    f"configuration: fault-parallel (strike {strike_bearing:.2f}), coherence 0.9, levelled, "
    f"smooth12, dip {dip_deg}, quadtree {min_block_km * 1000:.0f} m, surface interp, "
    f"tie {tie_depth_km * 1000:.0f} m, sigma {sigma_sel:g}, homogeneous GFs",
    f"L-curve corner sigma {sigma_corner:.1f} -> rule picks grid sigma {sigma_rule:g}; used sigma {sigma_sel:g} (fixed)",
    f"draft CHECK at sigma {sigma_check}: max |slip diff| {slip_diff:.2e} mm",
    f"noise model: {noise_model.round(3).to_dict()}",
    f"bootstrap Mw: {boot_mw.mean():.3f} +/- {mw_boot_sd:.3f} (n={n_boot})",
    f"fault-normal max {normal_max_mm:.2f} mm vs fault-parallel max {parallel_max_mm:.2f} mm",
] + [f"{key}: {value:.4g}" for key, value in results.items()])
REG.register("07_hom_mw_30gpa", results.mw_30gpa, "Mw", 2, where="M7", cell="baseline", sidecar=sidecar)
REG.register("07_hom_mw_sfcvm", results.mw_sfcvm, "Mw", 2, where="M7", cell="baseline", sidecar=sidecar)
REG.register("07_hom_mw_boot_sd", mw_boot_sd, "Mw", 3, where="M7", cell="bootstrap", sidecar=sidecar)
REG.register("07_hom_potency_m3", results.potency_m3, "m3", 0, where="M7", cell="baseline", sidecar=sidecar)
REG.register("07_hom_mu_effective_gpa", results.mu_effective_gpa, "GPa", 1, where="M7", cell="baseline", sidecar=sidecar)
REG.register("07_hom_surface_pct", results.surface_pct, "%", 0, where="M7", cell="baseline", sidecar=sidecar)
REG.register("07_hom_peak_slip_mm", results.peak_slip_mm, "mm", 1, where="M7", cell="baseline", sidecar=sidecar)
REG.register("07_hom_rms_mm", results.rms_mm, "mm", 2, where="M7", cell="baseline", sidecar=sidecar)
REG.register("07_hom_rms_near_mm", results.rms_near_mm, "mm", 2, where="M7", cell="baseline", sidecar=sidecar)
REG.register("07_hom_sigma", sigma_sel, "mm", 0, where="M7", cell="sigma", sidecar=sidecar)
REG.register("07_hom_sigma_rule", sigma_rule, "mm", 0, where="M7", cell="sigma", sidecar=sidecar)
REG.register("07_hom_sigma_corner", sigma_corner, "mm", 1, where="M7", cell="sigma", sidecar=sidecar)
REG.register("07_hom_tie_depth_m", tie_depth_km * 1000, "m", 0, where="M7", cell="constants", sidecar=sidecar)
REG.register("07_hom_n_samples", int(results.n_samples), "count", 0, where="M7", cell="baseline", sidecar=sidecar)
REG.register("07_hom_fault_normal_max_mm", normal_max_mm, "mm", 2, where="M7", cell="fault-normal", sidecar=sidecar)
REG.render()
print("07_slip_model (homogeneous baseline) complete")
