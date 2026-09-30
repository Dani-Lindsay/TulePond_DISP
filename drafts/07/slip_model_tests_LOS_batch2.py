# %% [markdown]
# # 07 Slip model -- sensitivity tests (batch 2)
#
# ```
# Created on Thu Sep 25 2026, batch 2 on Sun Sep 28 2026
# @author: daniellelindsay
# ```
#
# Every modelling choice for the mean-event slip inversion, tested. The
# BASELINE configuration of this batch matches the selected model
# (2026-09-27 decisions): far-field plane alignment per track, per-dataset
# CONSTANT ramp, the crossing-stack surface constraint IN the inversion,
# equal track weights, quadtree to one 30 m pixel. Dedicated rows isolate
# each of those choices (`insar_only`, `unaligned`, `asc_x2`, floors).
#
# Families: smoothing sigma with and without the surface constraint, dip
# 60-95 (95 = 85 W, closing the boundary), mesh (surface cells 100/50/30 m,
# plateau vs smooth growth, deep plateau), prior correlation lengths
# (never swept before), sampling, prior form, free rake (slip table saved).
# Then the parametric depth scan (aligned fields, densified grid, 5 km
# mesh and legacy-field variants) and the synthetic recovery WITH the
# constraint. Ported from `Hayward_DISP-S1/modelling/` (m02, m05, m15),
# CSI-tutorial style; CHECK cells pin the port against the legacy chain.

# %%
## [1]
# ------------------------
# Paths & constants
# ------------------------
import copy
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, cKDTree

warnings.simplefilter("ignore")
from TulePond_filepaths import common_paths, repo_dir
from insar_utils import load_trace, mpl_paper_style

# CSI: Classic Slip Inversion (Jolivet et al.; external/csi at 1277bb4)
from csi import TriangularPatches, insar, surfaceslip
from csi import imagedownsampling as imdown
from csi import transformation as transform
from csi import multisourcesolve as multisrc

notebook = "07_slip_model_tests"
out_dir = os.path.join(common_paths["out_dir"], notebook)
fig_dir = common_paths["fig_dir"]
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# local frame centre (csi_io convention: the creep-section centre)
frame_lon0, frame_lat0 = -121.968, 37.557

# data selection
field_dir      = os.path.join(common_paths["out_dir"], "06_event_field_geometry")
tracks         = ("ascending", "descending")
swath_km       = 4.0      # keep pixels within this distance of the trace
exclude_km     = 0.03     # drop the pixel straddling the trace
km_window      = (52.0, 67.0)   # along-strike data window (km from Point Pinole)
start_block_km = 2.0      # quadtree starting block
min_block_km   = 0.03     # quadtree minimum block = one 30 m pixel
cov_mask_km    = 1.0      # covariance from pixels beyond this distance
cov_distmax_km = 8.0
nug_power      = 0.5      # nugget variance / n_pixels**nug_power per block
near_field_km  = 0.2      # "near field" for the fit metrics

# alignment and surface constraint (selected-model conventions)
creep_lo_km, creep_hi_km = 56.3, 62.3   # registry 04_creep_length_lo/hi
align_mask_km  = 1.0      # slip section +/- this masked for the far-field plane
surf_floor_mm  = 0.5      # crossing-stack error floor (baseline)
surf_step_km   = 0.2
surf_win_km    = 0.4

# mesh and prior
mesh_lat_lo, mesh_lat_hi = 37.47, 37.65
mesh_depth_km  = 3.0
fine_km        = (56.0, 63.0)
far_rate       = 0.25
smooth_iters   = 8
lam_xz         = (0.3, 0.05)   # prior correlation lengths along strike, down dip (km)
shear_modulus  = 3.0e10

# the suite
dips_deg       = [60, 70, 75, 80, 85, 90, 95]   # 95 deg E = 85 deg W
sigmas_scan    = [8, 16, 24, 32, 48, 64]
lam_grid       = [(0.15, 0.05), (0.6, 0.05), (0.3, 0.15), (0.6, 0.15)]
mesh_variants  = {
    # name: (h_top, const_to, growth, h_bot)
    "plateau100": (0.10, 0.5, 1.6, 1.5),
    "plateau100_deep": (0.10, 1.5, 1.6, 1.5),
    "fine50": (0.05, 0.5, 1.6, 1.5),
    "fine30": (0.03, 0.25, 1.6, 1.5),      # one 30 m pixel per cell at the surface
    "smooth12": (0.05, 0.0, 1.2, 1.5),     # no plateau (selected)
}
mesh_sigma     = {"plateau100": [32], "plateau100_deep": [64], "fine50": [48],
                  "fine30": [64, 96], "smooth12": [32, 64]}
surf_floors    = [0.25, 1.0]              # baseline 0.5 has its own family row
recovery_depths = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
scan_km        = (57.5, 61.5)
scan_taper     = 0.5
scan_depths    = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65,
                  0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.25, 1.35, 1.5,
                  1.65, 1.85, 2.0, 2.25, 2.5, 3.0, 4.0, 5.0]

# %% [markdown]
# ## Section 1 -- the fault mesh
#
# Unrolled (s, z) node rows -> Delaunay -> Laplacian smoothing -> wrapped
# onto the hand-drawn trace, hinged east about the FIXED surface trace
# (`tutoFaultMeshing` idiom; ported from csi_io.load_fault_tri, proven
# zero-difference below).

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
print(f"trace: {len(trace_lonlat)} vertices, {arc_total:.1f} km of fault")

# %%
## [3]
def build_unrolled(h_top, const_to, growth, h_bot, z_max=mesh_depth_km):
    """Node rows + Delaunay + Laplacian smoothing in the unrolled plane."""
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
    """csi.TriangularPatches hinged east about the FIXED surface trace
    (dip > 90 hinges west). Each fault owns its faces copy
    (homogeneizeStrike flips winding rows in place)."""
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

unrolled = {name: build_unrolled(*params) for name, params in mesh_variants.items()}
for name, (nodes, faces) in unrolled.items():
    print(f"mesh {name}: {len(nodes)} nodes, {len(faces)} triangles")

# %% [markdown]
# ### CHECK -- zero-difference mesh port gate

# %%
## [4]
legacy_dir = os.path.join(repo_dir, "Hayward_DISP-S1")
for extra in (legacy_dir, os.path.join(legacy_dir, "modelling"),
              os.path.join(legacy_dir, "analysis")):
    sys.path.insert(0, extra)
import csi_io

for dip_check in (90.0, 70.0):
    legacy_fault = csi_io.load_fault_tri(dip=dip_check)
    ported_fault = wrap_fault(*unrolled["plateau100"], dip_check, "check")
    dv = np.max(np.abs(legacy_fault.Vertices - ported_fault.Vertices))
    same_faces = np.array_equal(legacy_fault.Faces, ported_fault.Faces)
    print(f"dip {dip_check:g}: max |vertex difference| {dv:.3e} km, "
          f"faces identical: {same_faces}")
    assert dv == 0.0 and same_faces, "mesh port differs from csi_io.load_fault_tri"
print("PASS: mesh port is zero-difference against the original")

# %% [markdown]
# ## Section 2 -- data, alignment, covariance, downsampling, the constraint
#
# Per track: notebook 06's five-event mean, cut to the window; the far
# field (slip section +/- 1 km masked) levelled by a fitted plane BEFORE
# anything else (the ascending mean event tilts ~2.5 mm end to end); the
# far-field semivariogram noise model; the distance-based quadtree to one
# 30 m pixel. The crossing-stack surface constraint is built per error
# floor.

# %%
## [5]
def dist_to_trace(sar, fault, signed=False):
    """[csi_io, verbatim] + east of the trace when signed."""
    tx, ty = fault.xf, fault.yf
    d, i = cKDTree(np.c_[tx, ty]).query(np.c_[sar.x, sar.y])
    return d * np.sign(sar.x - tx[i]) if signed else d


def load_track(track, field_file, fault, keep_side=None):
    """csi.insar cut to the swath and along-strike window. [csi_io port]"""
    z = np.load(field_file)
    sar = insar(track, lon0=frame_lon0, lat0=frame_lat0, verbose=False)
    sar.read_from_binary(z["vel_mm"], z["lon"], z["lat"], err=z["err_mm"],
                         incidence=z["incidence"], azimuth=z["azimuth"],
                         remove_nan=True, remove_zeros=False)
    trace_dist = dist_to_trace(sar, fault)
    sar.vel[trace_dist > swath_km] = np.nan
    sar.vel[trace_dist < exclude_km] = np.nan
    if keep_side is not None:
        signed = dist_to_trace(sar, fault, signed=True)
        sar.vel[signed > 0 if keep_side == "west" else signed < 0] = np.nan
    km_pp = lat_to_km_pp(sar.lat)
    sar.vel[(km_pp < km_window[0]) | (km_pp > km_window[1])] = np.nan
    sar.checkNaNs()
    return sar


def align_track(sar):
    """Far-field plane alignment: slip section +/- align_mask_km masked,
    1,x,y fit to the rest, removed from the whole track in place."""
    km_pp = lat_to_km_pp(sar.lat)
    far = (km_pp < creep_lo_km - align_mask_km) | (km_pp > creep_hi_km + align_mask_km)
    plane = np.c_[np.ones(sar.x.size), sar.x, sar.y]
    plane_coef = np.linalg.lstsq(plane[far], sar.vel[far], rcond=None)[0]
    sar.vel[:] = sar.vel - np.dot(plane, plane_coef)
    return plane_coef


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
    """Two exponentials + nugget/blocksize; asc_scale**2 on the whole
    matrix (1 = the equal-weights baseline). [m02 + optional inflation]"""
    nug2, s1, l1, s2, l2 = cov_p
    dx = ds.x[:, None] - ds.x[None, :]; dy = ds.y[:, None] - ds.y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    cd = s1 * np.exp(-dist / l1) + s2 * np.exp(-dist / l2)
    n_px = np.asarray(getattr(ds, "wgt", np.ones(ds.vel.size)), dtype=float)
    n_px = np.where(np.isfinite(n_px) & (n_px > 0), n_px, 1.0)
    cd[np.diag_indices_from(cd)] += nug2 / n_px ** nug_power
    ds.Cd = cd * asc_scale ** 2
    return ds

# %%
## [6]
geometry_fault = wrap_fault(*unrolled["plateau100"], 90, "geometry")
full_tracks, full_tracks_raw, cov_models, cov_models_raw, align_coefs = {}, {}, {}, {}, {}
for track in tracks:
    field_file = os.path.join(field_dir, f"csi_mean_event_{track}.npz")
    raw = load_track(track, field_file, geometry_fault)
    aligned = copy.deepcopy(raw)
    align_coefs[track] = align_track(aligned)
    full_tracks_raw[track] = raw
    full_tracks[track] = aligned
    cov_models_raw[track] = covariance(raw, geometry_fault)
    cov_models[track] = covariance(aligned, geometry_fault)
    c = align_coefs[track]
    print(f"[{track}] {aligned.vel.size} px; alignment plane {c[0]:+.2f} mm, "
          f"{c[1]:+.3f}, {c[2]:+.3f} mm/km")

# %%
## [6b]
# the surface constraint (port of csi_io.load_surface_slip on 04's product)
crossings_all = pd.read_csv(os.path.join(common_paths["out_dir"],
                                         "04_event_detection",
                                         "event_fraction_by_crossing.csv"))
event_cols = ["amp_2017.099", "amp_2019.186", "amp_2021.107",
              "amp_2023.080", "amp_2024.984"]
crossings_all["mean_event_mm"] = crossings_all[event_cols].mean(axis=1)
trace_south_north = trace_lonlat[np.argsort(trace_lonlat[:, 1])]

def make_surface(floor_mm):
    """csi.surfaceslip: running median of the 5-event crossing means,
    error = 1.4826 MAD / sqrt(n) floored at floor_mm."""
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
    surface triangle of this fault; errors x sqrt(triangles per step) so
    the total weight equals the original points (departure from CSI's
    single-nearest-triangle link, Fault.surfaceGFs)."""
    base = get_surface(floor_mm)
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

surface_cache = {}
def get_surface(floor_mm):
    if floor_mm not in surface_cache:
        surface_cache[floor_mm] = make_surface(floor_mm)
    return surface_cache[floor_mm]
print(f"surface constraint: {get_surface(surf_floor_mm).vel.size} points, "
      f"floor {surf_floor_mm:g} mm")

# %%
## [7]
downsampled = {}
def get_downsampled(track, mesh_name, dip, min_km=min_block_km, aligned=True,
                    start_km=start_block_km, asc_scale=1.0):
    """Quadtree samples for one geometry/data version, cached in-session;
    Cd rebuilt on every call so options never stick to the cache."""
    key = (track, mesh_name, dip, min_km, aligned, start_km)
    if key not in downsampled:
        fault = wrap_fault(*unrolled[mesh_name], dip, f"{mesh_name}_d{dip}")
        source = full_tracks[track] if aligned else full_tracks_raw[track]
        downsampled[key] = downsample(source, fault, start_km, min_km)
    ds = downsampled[key]
    cov_p = cov_models[track] if aligned else cov_models_raw[track]
    build_cd(ds, cov_p, asc_scale if track == "ascending" else 1.0)
    return ds

# %% [markdown]
# ## Section 3 -- the inversion machinery
#
# Meade GFs (right-lateral positive), per-dataset ramp, buildCmXY prior,
# NNLS (or GLS when the rake is freed). [m02, verbatim apart from
# @ -> np.dot and the surface-constraint dataset]

# %%
## [8]
def tie_rows(fault, n_params, tie_depth_km, tie_bin_km=0.1, tie_err_mm=0.01):
    """Equality constraints tying slip uniform from the surface to
    tie_depth_km (after Jolivet et al. 2023, who tie 0-1 km): cells whose
    centroid is shallower than tie_depth_km, grouped in along-strike bins of
    tie_bin_km, each equal to its bin's shallowest cell. Rows of
    [cell - reference] = 0 with a tiny error (near-hard constraint)."""
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


def patch_table(fault):
    nodes, faces, _ = fault.mesh_sz
    centres_sz = nodes[faces].mean(1)
    centres_ll = np.array([pll.mean(0) for pll in fault.patchll])
    return pd.DataFrame(dict(
        lon=centres_ll[:, 0], lat=centres_ll[:, 1],
        s_km=centres_sz[:, 0], z_c=centres_sz[:, 1],
        z_top=nodes[faces][:, :, 1].min(1), z_bot=nodes[faces][:, :, 1].max(1),
        km_pp=lat_to_km_pp(centres_ll[:, 1]),
        slip_mm=fault.slip[:, 0]))


def metrics(fault, insar_datas, surface=None):
    """Per-track misfit, near-field rms, surface-constraint fit, peak,
    moment, Mw, depth profile (InSAR datasets only for the track stats)."""
    row = {}
    chi2_total = 0.0; rss = 0.0; n_total = 0
    for ds in insar_datas:
        track_key = ds.name.split()[-1][:3]
        resid = ds.vel - ds.synth
        chi2 = float(np.dot(resid, np.linalg.solve(ds.Cd, resid)))
        row[f"chi2_{track_key}"] = chi2
        row[f"rms_{track_key}"] = float(np.sqrt(np.mean(resid ** 2)))
        near = dist_to_trace(ds, fault) <= near_field_km
        row[f"rms_near_{track_key}"] = (float(np.sqrt(np.mean(resid[near] ** 2)))
                                        if near.any() else np.nan)
        row[f"n_{track_key}"] = int(resid.size)
        chi2_total += chi2; rss += float(np.sum(resid ** 2)); n_total += resid.size
    row["chi2"] = chi2_total
    row["rms_mm"] = float(np.sqrt(rss / n_total))
    row["n_samples"] = n_total
    row["surf_rms_mm"] = (float(np.sqrt(np.mean((surface.vel - surface.synth) ** 2)))
                          if surface is not None else np.nan)
    table = patch_table(fault)
    moment = 0.0
    for patch, slip in zip(fault.patch, table.slip_mm.values):
        area_km2 = 0.5 * np.linalg.norm(np.cross(patch[1] - patch[0], patch[2] - patch[0]))
        moment += shear_modulus * area_km2 * 1e6 * slip * 1e-3
    row["M0"] = moment
    row["Mw"] = (2.0 / 3.0) * (np.log10(max(moment, 1.0)) - 9.1)
    row["peak_slip_mm"] = float(table.slip_mm.max())
    row["dipslip_rms_mm"] = (float(np.sqrt(np.mean(fault.slip[:, 1] ** 2)))
                             if fault.slip.shape[1] > 1 and np.any(fault.slip[:, 1]) else 0.0)
    peak_s = table.loc[table.slip_mm.idxmax(), "s_km"]
    near_peak = table[(table.s_km - peak_s).abs() < 0.3]
    depth_profile = near_peak.groupby(np.round(0.5 * (near_peak.z_top + near_peak.z_bot), 2)).slip_mm.max()
    row["depth_profile"] = depth_profile.to_dict()
    return row, table

# %% [markdown]
# ## Section 4 -- the suite
#
# Baseline = the selected-model configuration: aligned, constant ramp,
# surface constraint (floor 0.5 mm), equal weights, 30 m quadtree floor.
# Every case builds its fault and GFs FRESH (nothing persists between
# cases except the quadtree samples, which are re-Cd'd and deep-copied).

# %%
## [9]
suite_rows = []
def run_case(family, label, mesh_name, dip, sigma, lam=lam_xz,
             min_km=min_block_km, start_km=start_block_km, aligned=True,
             use_surface=True, surf_floor=surf_floor_mm, asc_scale=1.0,
             prior="xy", ramp_spec="1", slipdir="s", keep_side=None):
    fault = wrap_fault(*unrolled[mesh_name], dip, f"{mesh_name}_d{dip}_{label}")
    insar_datas = []
    for track in tracks:
        if keep_side is not None and track == "ascending":
            field_file = os.path.join(field_dir, f"csi_mean_event_{track}.npz")
            full = load_track(track, field_file, fault, keep_side=keep_side)
            if aligned:
                align_track(full)
            ds = downsample(full, fault, start_km, min_km)
            build_cd(ds, cov_models[track] if aligned else cov_models_raw[track],
                     asc_scale)
        else:
            ds = get_downsampled(track, mesh_name, dip, min_km, aligned,
                                 start_km, asc_scale)
        insar_datas.append(copy.deepcopy(ds))         # invert mutates state
    surface = make_surface_interp(fault, surf_floor) if use_surface else None
    invert(fault, insar_datas, sigma, lam, prior, slipdir, ramp_spec, surface)
    row, table = metrics(fault, insar_datas, surface)
    row.update(family=family, label=label, mesh=mesh_name, dip=dip, sigma=sigma,
               lam_s=lam[0], lam_z=lam[1], min_block_km=min_km,
               start_block_km=start_km, aligned=aligned, use_surface=use_surface,
               surf_floor=surf_floor if use_surface else np.nan,
               asc_scale=asc_scale, prior=prior, slipdir=slipdir,
               n_patches=len(fault.patch))
    suite_rows.append(row)
    print(f"[{family}] {label}: chi2 {row['chi2']:.0f}, rms {row['rms_mm']:.2f} mm, "
          f"near des {row['rms_near_des']:.2f} mm, surf {row['surf_rms_mm']:.2f} mm, "
          f"peak {row['peak_slip_mm']:.1f} mm, Mw {row['Mw']:.2f}")
    return row, table, fault, insar_datas

# %%
## [10]
# family 1: smoothing sigma WITH the surface constraint (the record axis);
# the plateau rule then selects sigma for everything downstream
for sigma in sigmas_scan:
    run_case("sigma_surf", f"sigmaS{sigma:g}", "smooth12", 90, sigma)
surf_family = pd.DataFrame([r for r in suite_rows if r["family"] == "sigma_surf"])
plateau = surf_family[surf_family.rms_mm <= surf_family.rms_mm.min() + 0.005]
sigma_sel = float(plateau.sigma.min())
print(f"selected sigma (constrained plateau onset): {sigma_sel:g}")

# %%
## [11]
# family 2: sigma WITHOUT the constraint -- the Mw-ramp demonstration
for sigma in (8, 32, 64):
    run_case("sigma", f"sigma{sigma:g}", "smooth12", 90, sigma, use_surface=False)

# %%
## [12]
# family 3: dip, 60 to 95 east (95 = 85 west: does the optimum close?)
for dip in dips_deg:
    run_case("dip", f"dip{dip}", "plateau100", dip, sigma_sel)

# %%
## [13]
# family 4: mesh -- surface cells 100/50/30 m, plateau vs smooth growth,
# each at (approximately) matched-misfit sigma; smooth12 covered by family 1
for mesh_name, sig_list in mesh_sigma.items():
    if mesh_name == "smooth12":
        continue
    for sigma in sig_list:
        run_case("mesh", f"{mesh_name}_s{sigma:g}", mesh_name, 90, sigma)

# %%
## [14]
# family 5: prior correlation lengths (never swept before)
for lam_s, lam_z in lam_grid:
    run_case("lam", f"lam{lam_s:g}x{lam_z:g}", "smooth12", 90, sigma_sel,
             lam=(lam_s, lam_z))

# %%
## [15]
# family 6: sampling -- 40 and 60 m floors, and a uniform 100 m grid
run_case("sampling", "minblock40", "plateau100", 90, sigma_sel, min_km=0.04)
run_case("sampling", "minblock60", "plateau100", 90, sigma_sel, min_km=0.06)
run_case("sampling", "uniform100", "plateau100", 90, sigma_sel,
         min_km=0.1, start_km=0.1)

# %%
## [16]
# family 7: isolation rows -- each selected-model choice switched off alone
run_case("isolation", "insar_only", "smooth12", 90, sigma_sel, use_surface=False)
run_case("isolation", "unaligned", "smooth12", 90, sigma_sel, aligned=False,
         ramp_spec="1,x,y")
run_case("isolation", "asc_x2", "smooth12", 90, sigma_sel, asc_scale=2.0)
for floor in surf_floors:
    run_case("isolation", f"floor{floor:g}", "smooth12", 90, sigma_sel,
             surf_floor=floor)

# %%
## [17]
# family 8: prior form brackets and the free rake (slip table saved)
run_case("prior", "diag", "plateau100", 90, sigma_sel, prior="diag",
         use_surface=False)
run_case("prior", "unregularized", "plateau100", 90, 1000, prior="diag",
         use_surface=False)
rake_row, rake_table, rake_fault, _ = run_case(
    "rake", "rake_free", "plateau100", 90, sigma_sel, use_surface=False,
    slipdir="sd")
rake_table["dipslip_mm"] = rake_fault.slip[:, 1]
rake_table.to_csv(os.path.join(out_dir, "rake_free_slip.csv"), index=False)

# %%
## [18]
# determinism gate: an exact repeat of one configuration, then the csv
gate_row, _, _, _ = run_case("gate", "dip90_repeat", "plateau100", 90, sigma_sel)
suite = pd.DataFrame(suite_rows)
first = suite[(suite.family == "dip") & (suite.label == "dip90")].iloc[0]
assert abs(first.chi2 - gate_row["chi2"]) < 1e-6 and abs(first.Mw - gate_row["Mw"]) < 1e-9, \
    "identical configuration did not reproduce exactly"
print("PASS: repeated configuration reproduces exactly")
suite_csv = os.path.join(out_dir, "suite_results.csv")
suite.drop(columns=["depth_profile"]).to_csv(suite_csv, index=False)
np.save(os.path.join(out_dir, "suite_depth_profiles.npy"),
        np.array([r["depth_profile"] for r in suite_rows], dtype=object),
        allow_pickle=True)
print(f"wrote {suite_csv} ({len(suite)} configurations)")

# %% [markdown]
# ### CHECK -- machinery parity against the legacy m02 chain
#
# Legacy fields, unaligned, 1,x,y ramps, no surface data, 40 m floor,
# sigma 16, lam (0.3, 0.05): must reproduce the m02 CLI numbers.

# %%
## [19]
legacy_lcurve = pd.read_csv(os.path.join(common_paths["out_dir"], "modelling",
                                         "inversion",
                                         "lcurve_tri100m-500m_sw4_km52-67_lam0.3x0.05.csv"))
legacy_row = legacy_lcurve[legacy_lcurve.sigma == 16].iloc[0]
check_fault = wrap_fault(*unrolled["plateau100"], 90, "parity")
check_datas = []
for track in tracks:
    legacy_field = os.path.join(common_paths["out_dir"], "modelling", track,
                                "csi_mean_event.npz")
    full = load_track(track, legacy_field, check_fault)
    cov_p = covariance(full, check_fault)
    ds = downsample(full, check_fault, start_block_km, 0.04)
    build_cd(ds, cov_p)
    check_datas.append(ds)
invert(check_fault, check_datas, 16, lam=(0.3, 0.05), ramp_spec="1,x,y")
check_row, _ = metrics(check_fault, check_datas)
print(f"ported: chi2 {check_row['chi2']:.1f}, Mw {check_row['Mw']:.3f} | "
      f"legacy: chi2 {legacy_row.chi2:.1f}, Mw {legacy_row.Mw:.3f}")
assert abs(check_row["chi2"] - legacy_row.chi2) < 1.0
assert abs(check_row["Mw"] - legacy_row.Mw) < 0.005
print("PASS: inversion machinery reproduces the legacy m02 numbers")

# %% [markdown]
# ## Section 5 -- depth: the scans and the constrained recovery
#
# chi2(D) for uniform slip 0-D (whitened LSQ with per-track constants) on
# the ALIGNED fields; variants: unaligned (was the deep answer an
# alignment artifact?), legacy fields (field-version systematic), and a
# 5 km mesh (so the upper bracket is not imposed by the mesh bottom).
# Then the synthetic recovery WITH the surface constraint.

# %%
## [20]
def run_depth_scan(fault, scan_datas, tag):
    scan_G = []
    for ds in scan_datas:
        fault.buildGFs(ds, vertical=True, slipdir="s", verbose=False)
        fault.G[ds.name]["strikeslip"] *= -1.0
        scan_G.append(fault.G[ds.name]["strikeslip"])
    table = patch_table(fault)
    in_window = np.clip(np.minimum(table.km_pp - scan_km[0],
                                   scan_km[1] - table.km_pp) / scan_taper + 1, 0, 1)
    chols, data_w = [], []
    for ds in scan_datas:
        chol = np.linalg.cholesky(ds.Cd + 1e-9 * np.eye(ds.Cd.shape[0]))
        chols.append(chol)
        data_w.append(np.linalg.solve(chol, ds.vel))
    depth_rows = []
    grid = [d for d in scan_depths if d <= table.z_bot.max() + 1e-6]
    for depth_bottom in grid:
        slip_shape = in_window.values * (table.z_c.values < depth_bottom)
        blocks_A, blocks_b = [], []
        for k, (gmat, chol) in enumerate(zip(scan_G, chols)):
            col = np.dot(gmat, slip_shape)
            design = np.zeros((col.size, 1 + len(scan_G)))
            design[:, 0] = col
            design[:, 1 + k] = 1.0                    # per-track constant
            blocks_A.append(np.linalg.solve(chol, design))
            blocks_b.append(data_w[k])
        design = np.vstack(blocks_A); b = np.concatenate(blocks_b)
        coef, *_ = np.linalg.lstsq(design, b, rcond=None)
        resid = b - np.dot(design, coef)
        depth_rows.append(dict(D_km=depth_bottom, slip_mm=coef[0],
                               chi2=float(np.dot(resid, resid))))
    scan = pd.DataFrame(depth_rows)
    scan["dchi2"] = scan.chi2 - scan.chi2.min()
    scan.to_csv(os.path.join(out_dir, f"depth_scan{tag}.csv"), index=False)
    best = scan.loc[scan.chi2.idxmin()]
    ok = scan[scan.dchi2 <= 4]
    print(f"depth scan {tag or '(record)'}: best D {best.D_km:g} km, "
          f"slip {best.slip_mm:.1f} mm; bracket {ok.D_km.min():g}-{ok.D_km.max():g} km")
    return scan

scan_fault = wrap_fault(*unrolled["plateau100"], 90, "depth_scan")
scan_datas = [copy.deepcopy(get_downsampled(track, "plateau100", 90))
              for track in tracks]
depth_scan = run_depth_scan(scan_fault, scan_datas, "")

scan_raw = [copy.deepcopy(get_downsampled(track, "plateau100", 90, aligned=False))
            for track in tracks]
run_depth_scan(wrap_fault(*unrolled["plateau100"], 90, "scan_raw"),
               scan_raw, "_unaligned")

scan_legacy_fault = wrap_fault(*unrolled["plateau100"], 90, "scan_legacy")
legacy_scan = []
for track in tracks:
    legacy_field = os.path.join(common_paths["out_dir"], "modelling", track,
                                "csi_mean_event.npz")
    full = load_track(track, legacy_field, scan_legacy_fault)
    align_track(full)
    cov_p = covariance(full, scan_legacy_fault)
    ds = downsample(full, scan_legacy_fault, start_block_km, min_block_km)
    build_cd(ds, cov_p)
    legacy_scan.append(ds)
run_depth_scan(scan_legacy_fault, legacy_scan, "_legacyfields")

deep_nodes, deep_faces = build_unrolled(0.10, 0.5, 1.6, 1.5, z_max=5.0)
unrolled["plateau100_z5"] = (deep_nodes, deep_faces)
scan5_fault = wrap_fault(deep_nodes, deep_faces, 90, "scan_z5")
scan5_datas = [copy.deepcopy(get_downsampled(track, "plateau100_z5", 90))
               for track in tracks]
run_depth_scan(scan5_fault, scan5_datas, "_z5")

# %%
## [21]
# synthetic recovery WITH the surface constraint: 10 mm to D, forward
# through the full pipeline (covariance-drawn noise on the InSAR and the
# constraint), inverted at the selected configuration
recovery_rng = np.random.default_rng(7)
recovery_profiles = {}
recovery_rows = []
base_surface = get_surface(surf_floor_mm)
for depth_bottom in recovery_depths:
    rec_fault = wrap_fault(*unrolled["smooth12"], 90, f"recovery_{depth_bottom:g}")
    rec_table = patch_table(rec_fault)
    in_window = np.clip(np.minimum(rec_table.km_pp - scan_km[0],
                                   scan_km[1] - rec_table.km_pp) / scan_taper + 1, 0, 1)
    slip_input = in_window.values * (rec_table.z_c.values < depth_bottom) * 10.0
    rec_datas = []
    for track in tracks:
        ds = copy.deepcopy(get_downsampled(track, "smooth12", 90))
        rec_fault.buildGFs(ds, vertical=True, slipdir="s", verbose=False)
        rec_fault.G[ds.name]["strikeslip"] *= -1.0
        chol = np.linalg.cholesky(ds.Cd + 1e-9 * np.eye(ds.Cd.shape[0]))
        ds.vel[:] = (np.dot(rec_fault.G[ds.name]["strikeslip"], slip_input)
                     + np.dot(chol, recovery_rng.standard_normal(ds.vel.size)))
        rec_datas.append(ds)
    rec_surface = make_surface_interp(rec_fault, surf_floor_mm)
    rec_fault.buildGFs(rec_surface, slipdir="s", verbose=False)
    ss_noise = np.sqrt(np.diag(rec_surface.Cd)) * recovery_rng.standard_normal(
        rec_surface.vel.size)
    rec_surface.vel[:] = (np.dot(rec_fault.G[rec_surface.name]["strikeslip"],
                                 slip_input) + ss_noise)
    invert(rec_fault, rec_datas, sigma_sel, surface=rec_surface)
    rec_row, rec_out = metrics(rec_fault, rec_datas, rec_surface)
    recovery_profiles[depth_bottom] = rec_row["depth_profile"]
    input_m0 = np.sum([shear_modulus * 0.5 * np.linalg.norm(
        np.cross(p[1] - p[0], p[2] - p[0])) * 1e6 * s * 1e-3
        for p, s in zip(rec_fault.patch, slip_input)])
    recovery_rows.append(dict(D_km=depth_bottom,
                              moment_recovery=rec_row["M0"] / input_m0,
                              mw_input=(2.0 / 3.0) * (np.log10(input_m0) - 9.1),
                              mw_recovered=rec_row["Mw"],
                              peak_recovered=rec_row["peak_slip_mm"]))
    print(f"input 10 mm to {depth_bottom:g} km (constrained): peak "
          f"{rec_row['peak_slip_mm']:.1f} mm, moment recovery "
          f"{rec_row['M0'] / input_m0:.2f}")
np.save(os.path.join(out_dir, "recovery_profiles.npy"),
        np.array([recovery_profiles], dtype=object), allow_pickle=True)
pd.DataFrame(recovery_rows).to_csv(os.path.join(out_dir, "recovery_summary.csv"),
                                   index=False)

# %% [markdown]
# ## Section 6 -- the collated sensitivity figure
#
# Assembled from the SAVED products, so this cell redraws without
# re-running the suite.

# %%
## [22]
mpl_paper_style(base=7)
suite = pd.read_csv(os.path.join(out_dir, "suite_results.csv"))
suite = suite[suite.family != "gate"]
depth_scan = pd.read_csv(os.path.join(out_dir, "depth_scan.csv"))
depth_scan_un = pd.read_csv(os.path.join(out_dir, "depth_scan_unaligned.csv"))
depth_scan_z5 = pd.read_csv(os.path.join(out_dir, "depth_scan_z5.csv"))
recovery_profiles = np.load(os.path.join(out_dir, "recovery_profiles.npy"),
                            allow_pickle=True)[0]

fig = plt.figure(figsize=(7.25, 6.6), constrained_layout=True)
gs = fig.add_gridspec(3, 3)

family_names = {"sigma_surf": "smoothing $\\sigma$ (constr.)",
                "sigma": "$\\sigma$ (InSAR only)", "dip": "dip angle",
                "mesh": "mesh", "lam": "correlation $\\lambda$",
                "sampling": "sampling", "isolation": "isolation",
                "prior": "prior form", "rake": "rake"}
family_colors = {"sigma_surf": "#009E73", "sigma": "#66c2a5", "dip": "#0072B2",
                 "mesh": "#E69F00", "lam": "#8da0cb", "sampling": "#56B4E9",
                 "isolation": "#D55E00", "prior": "#CC79A7", "rake": "#999999"}
family_order = list(family_names)

# A: Mw across the suite, families bracketed and named
ax = fig.add_subplot(gs[0, :])
x_pos = 0; ticks = []; tick_labels = []; brackets = []
for family in family_order:
    grp = suite[suite.family == family]
    if grp.empty:
        continue
    xs = np.arange(x_pos, x_pos + len(grp))
    ax.plot(xs, grp.Mw, "o", ms=4, color=family_colors[family])
    ticks += list(xs); tick_labels += list(grp.label)
    brackets.append((xs[0], xs[-1], family_names[family]))
    ax.axvline(x_pos + len(grp) - 0.5, color="0.85", lw=0.6)
    x_pos += len(grp)
mw_lo, mw_hi = suite.Mw.min(), suite.Mw.max()
ax.set_ylim(mw_lo - 0.05, mw_hi + 0.28)
for k, (x_start, x_end, name) in enumerate(brackets):
    row_up = (x_end - x_start) < 4 and k % 2
    bracket_y = mw_hi + (0.17 if row_up else 0.07)
    ax.plot([x_start - 0.3, x_end + 0.3], [bracket_y, bracket_y], "-",
            color="0.25", lw=0.8)
    ax.text(0.5 * (x_start + x_end), bracket_y + 0.015, name, ha="center",
            va="bottom", fontsize=6, color="0.15")
ax.set_xticks(ticks)
ax.set_xticklabels(tick_labels, rotation=60, ha="right", fontsize=5)
ax.set_ylabel(r"$M_W$")
ax.set_title(r"$\bf{A}$  Geodetic moment across the suite", loc="left", fontsize=7)

# B: misfit vs moment, family colours, outliers clipped
ax = fig.add_subplot(gs[1, 0])
for family in family_order:
    grp = suite[suite.family == family]
    if grp.empty:
        continue
    ax.scatter(grp.rms_mm, grp.Mw, s=10, c=family_colors[family], linewidths=0,
               label=family_names[family])
ax.legend(fontsize=4, loc="lower right", handletextpad=0.1)
core = suite[suite.rms_mm < suite.rms_mm.quantile(0.95)]
ax.set_xlim(core.rms_mm.min() - 0.02, core.rms_mm.max() + 0.04)
ax.set_xlabel("rms (mm)"); ax.set_ylabel(r"$M_W$")
ax.set_title(r"$\bf{B}$  Misfit vs moment", loc="left", fontsize=7)

# C: depth scans -- record, unaligned, 5 km mesh
ax = fig.add_subplot(gs[1, 1])
ax.plot(depth_scan_un.D_km, depth_scan_un.dchi2, "o-", ms=2, color="0.6",
        label="unaligned")
ax.plot(depth_scan_z5.D_km, depth_scan_z5.dchi2, "o-", ms=2, color="#8da0cb",
        label="5 km mesh")
ax.plot(depth_scan.D_km, depth_scan.dchi2, "o-", ms=2.5, color="#009E73",
        label="aligned (record)")
ax.legend(fontsize=5)
ax.axhline(1, color="0.5", ls="--", lw=0.7); ax.axhline(4, color="0.5", ls=":", lw=0.7)
ax.set_xscale("log")
ax.set_ylim(-5, min(depth_scan.dchi2.max() * 1.05, 200))
ax.set_xlabel("bottom depth D (km)"); ax.set_ylabel(r"$\Delta\chi^2$")
ax.set_title(r"$\bf{C}$  Parametric depth scan", loc="left", fontsize=7)

# D: constrained synthetic recovery
ax = fig.add_subplot(gs[1, 2])
rec_colors = {0.3: "#0072B2", 0.5: "#009E73", 0.8: "#E69F00", 1.0: "#56B4E9",
              1.5: "#D55E00", 2.0: "#CC79A7"}
for depth_bottom, profile in recovery_profiles.items():
    depths = np.array(sorted(profile))
    ax.plot([profile[z] for z in depths], depths, "o-", ms=2,
            color=rec_colors[float(depth_bottom)], label=f"to {depth_bottom:g} km")
ax.axvline(10, color="0.6", lw=0.6, ls="--")
ax.set_ylim(3, 0)
ax.set_xlabel("recovered slip (mm)"); ax.set_ylabel("depth (km)")
ax.legend(fontsize=4.5, ncols=2)
ax.set_title(r"$\bf{D}$  Constrained recovery (10 mm input)", loc="left", fontsize=7)

# E: descending near-field misfit per configuration (panel A order/colours)
ax = fig.add_subplot(gs[2, 0])
x_pos = 0
for family in family_order:
    grp = suite[suite.family == family]
    if grp.empty:
        continue
    xs = np.arange(x_pos, x_pos + len(grp))
    ax.plot(xs, grp.rms_near_des, "o", ms=3, color=family_colors[family])
    ax.axvline(x_pos + len(grp) - 0.5, color="0.9", lw=0.5)
    x_pos += len(grp)
ax.set_xlabel("configuration (panel A order)")
ax.set_ylabel("rms (mm)")
ax.set_title(r"$\bf{E}$  Descending near-field (<200 m) rms", loc="left", fontsize=7)

# F: dip -- every rms subset (the trade-off panel)
ax = fig.add_subplot(gs[2, 1])
dip_grp = suite[suite.family == "dip"].sort_values("dip")
ax.plot(dip_grp.dip, dip_grp.rms_near_asc, "o--", ms=3, color="#0072B2",
        label="asc near")
ax.plot(dip_grp.dip, dip_grp.rms_asc, "o-", ms=3, color="#0072B2", label="asc full")
ax.plot(dip_grp.dip, dip_grp.rms_near_des, "o--", ms=3, color="#D55E00",
        label="desc near")
ax.plot(dip_grp.dip, dip_grp.rms_des, "o-", ms=3, color="#D55E00", label="desc full")
ax.axvline(90, color="0.8", lw=0.6)
ax.set_xlabel("dip (deg E)"); ax.set_ylabel("rms (mm)")
ax.legend(fontsize=4.5)
ax.set_title(r"$\bf{F}$  Dip: every rms subset", loc="left", fontsize=7)

# G: sigma with and without the constraint -- does the constraint pin Mw?
ax = fig.add_subplot(gs[2, 2])
with_c = suite[suite.family == "sigma_surf"].sort_values("sigma")
without = suite[suite.family == "sigma"].sort_values("sigma")
ax.plot(with_c.sigma, with_c.Mw, "o-", ms=3, color="#009E73", label="constrained")
ax.plot(without.sigma, without.Mw, "o-", ms=3, color="0.5", label="InSAR only")
ax.set_xscale("log")
ax.set_xlabel(r"prior $\sigma$ (mm)"); ax.set_ylabel(r"$M_W$")
ax.legend(fontsize=5)
ax.set_title(r"$\bf{G}$  Smoothing: does the constraint pin $M_W$?",
             loc="left", fontsize=7)

fig_file = os.path.join(fig_dir, "fig07_sensitivity_suite.png")
fig.savefig(fig_file, dpi=300)
fig.savefig(fig_file.replace(".png", ".pdf"))
print("wrote", fig_file)

# %% [markdown]
# The selected configuration and its registry numbers live in
# `07_slip_model.ipynb`, which reads `suite_results.csv` for the ranges.
