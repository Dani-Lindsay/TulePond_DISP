# %% [markdown]
# # 07 Slip model of the mean event -- the selected solution
#
# ```
# Created on Thu Sep 25 2026
# @author: daniellelindsay
# ```
#
# The ONE configuration the paper quotes, reproduced end to end: mesh ->
# data -> covariance -> downsampling -> inversion -> figures -> registry.
# Every alternative (dip, smoothing, mesh, sampling, weighting, depth) is
# tested in `07_slip_model_tests.ipynb`, which must run first -- this
# notebook reads its `suite_results.csv` for the smoothing selection and the
# structural Mw range. CSI calls follow the tutorials
# (`external/csi/notebooks/`); the port is pinned by the tests notebook's
# zero-difference and parity CHECK gates.
#
# The selected configuration, and why:
# - **mesh** `smooth12`: 50 m cells at the surface growing geometrically
#   (x1.2 per row) to 3 km depth -- no resolution plateau, so no special
#   depth for an artefact to live at (decided 2026-09-25).
# - **dip 90** (decided 2026-09-28): every rms subset of both tracks --
#   near and far field alike -- prefers vertical monotonically over
#   60-90 E; the seismicity dip (~75-80 E) is measured at 2-9 km, BELOW
#   the slipping patch, and enters as context, not geometry. Mw moves
#   < 0.1 across 60-95, so the choice costs nothing.
# - **quadtree 2 km -> 30 m**: the minimum block is one OPERA pixel.
# - **equal weights** (decided 2026-09-27): the viewing geometry already
#   down-weights ascending for strike slip (|LOS.fault-parallel| 0.28 vs
#   0.49), and the suite shows explicit down-weighting or masking moves Mw
#   by < 0.02 -- so no covariance inflation is applied.
# - **prior sigma**: smallest sigma on the rms plateau of the smooth12
#   family in the suite (the chi2/N criterion, selected below).
# - **pre-alignment**: with the slip section masked (creep section +/- 1
#   km), each track's far field is levelled by a fitted plane BEFORE
#   downsampling (measured misalignment: the ascending far field tilts
#   ~+0.9 to -0.9 mm north to south); the inversion ramp is then a
#   per-dataset constant only.
# - **surface constraint IN the model** (decided 2026-09-27): the
#   crossing-stack mean event (error floor 0.5 mm) constrains the surface
#   row -- without it the surface slip is underestimated by ~2x at the
#   peak (the InSAR-only solution is kept as the supplement comparison).
#   The independent validation is the creepmeters/arrays (notebook 05).

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
import insar_utils as utils
from insar_utils import load_trace, mpl_paper_style

# CSI: Classic Slip Inversion (Jolivet et al.; external/csi at 1277bb4)
from csi import TriangularPatches, insar
from csi import imagedownsampling as imdown
from csi import transformation as transform
from csi import multisourcesolve as multisrc

notebook = "07_slip_model"
out_dir = os.path.join(common_paths["out_dir"], notebook)
fig_dir = common_paths["fig_dir"]
tests_dir = os.path.join(common_paths["out_dir"], "07_slip_model_tests")
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

frame_lon0, frame_lat0 = -121.968, 37.557

# data selection
field_dir      = os.path.join(common_paths["out_dir"], "06_event_field_geometry")
tracks         = ("ascending", "descending")
swath_km       = 4.0
exclude_km     = 0.03
km_window      = (52.0, 67.0)
start_block_km = 2.0
min_block_km   = 0.03
cov_mask_km    = 1.0
cov_distmax_km = 8.0
nug_power      = 0.5
near_field_km  = 0.2
align_mask_km  = 1.0      # slip section +/- this is masked for the far-field plane
surf_err_mm    = 0.5      # crossing-stack error floor (the constraint strength)
surf_step_km   = 0.2      # running-median step / window for the surface data
surf_win_km    = 0.4

# the selected geometry
mesh_lat_lo, mesh_lat_hi = 37.47, 37.65
mesh_depth_km  = 3.0
fine_km        = (56.0, 63.0)
far_rate       = 0.25
smooth_iters   = 8
cell_top_km    = 0.05     # smooth12: 50 m surface cells ...
growth_rate    = 1.2      # ... growing x1.2 per row, no plateau
cell_bot_km    = 1.5
dip_deg        = 90.0
lam_xz         = (0.3, 0.05)
shear_modulus  = 3.0e10
n_boot         = 24       # covariance-drawn noise realisations for sigma(Mw)

REG = utils.Registry(notebook, common_paths, repo_dir)
REG.inputs(os.path.join(field_dir, "csi_mean_event_ascending.npz"),
           os.path.join(field_dir, "csi_mean_event_descending.npz"),
           common_paths["fault_trace"], common_paths["fault_profiles"],
           os.path.join(tests_dir, "suite_results.csv"),
           os.path.join(tests_dir, "depth_scan.csv"))

# %% [markdown]
# ## [2] Smoothing selection from the suite
#
# The prior strength is mesh-dependent, so sigma is chosen ON the selected
# mesh: the smallest sigma of the smooth12 family whose rms is within
# 0.005 mm of the family minimum (the plateau onset = the chi2/N criterion).

# %%
## [2]
suite = pd.read_csv(os.path.join(tests_dir, "suite_results.csv"))
# sigma family and the weighting row re-run on the interpolated surface
# constraint (2026-09-28); these supersede the matching suite rows
sigma_runs = os.path.join(tests_dir, "sigma_selection.csv")
if os.path.exists(sigma_runs):
    suite = pd.concat([suite[~suite.family.isin(["sigma_surf"])
                             & ~((suite.family == "isolation") & (suite.label == "asc_x2"))],
                       pd.read_csv(sigma_runs)], ignore_index=True)
family = suite[suite.family == "sigma_surf"].sort_values("sigma")
assert len(family) >= 3, "sigma_surf family missing from suite_results.csv"
plateau = family[family.rms_mm <= family.rms_mm.min() + 0.005]
sigma_sel = float(plateau.sigma.min())
print(family[["sigma", "chi2", "rms_mm", "Mw"]].to_string(index=False))
print(f"selected sigma: {sigma_sel:g} (rms plateau onset)")

depth_scan = pd.read_csv(os.path.join(tests_dir, "depth_scan.csv"))
best_D = depth_scan.loc[depth_scan.chi2.idxmin()]
ok_D = depth_scan[depth_scan.dchi2 <= 4]
print(f"depth extent (tests): best D {best_D.D_km:g} km, "
      f"bracket {ok_D.D_km.min():g}-{ok_D.D_km.max():g} km")

# %% [markdown]
# ## [3] The fault mesh
#
# 50 m triangles at the trace growing x1.2 per row to 3 km, wrapped onto
# the hand-drawn trace, hinged 75 deg east about the FIXED surface trace
# (identical machinery to the tests notebook; tutoFaultMeshing idiom).

# %%
## [3]
trace_lonlat = load_trace(common_paths["fault_trace"])
trace_lonlat = trace_lonlat[(trace_lonlat[:, 1] >= mesh_lat_lo)
                            & (trace_lonlat[:, 1] <= mesh_lat_hi)]
trace_lonlat = trace_lonlat[np.argsort(-trace_lonlat[:, 1])]

ref_fault = TriangularPatches("trace frame", lon0=frame_lon0, lat0=frame_lat0,
                              verbose=False)
ref_fault.trace(trace_lonlat[:, 0], trace_lonlat[:, 1])
ref_fault.discretize2(every=0.02)
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

# node rows: geometric growth from the surface, no plateau
row_depths = [0.0]
row_step = cell_top_km
while row_depths[-1] < mesh_depth_km:
    row_depths.append(min(row_depths[-1] + row_step, mesh_depth_km))
    row_step = min(row_step * growth_rate, cell_bot_km)
row_depths = np.array(row_depths)
row_heights = np.r_[np.diff(row_depths), np.diff(row_depths)[-1]]

def spacing_at(arc_pos, row_height):
    km_pp = np.interp(arc_pos, arc_s, trace_km_pp)
    outside = np.maximum(0, np.maximum(fine_km[0] - km_pp, km_pp - fine_km[1]))
    return row_height + far_rate * outside

nodes_sz = []
for depth, height in zip(row_depths, row_heights):
    row = [0.0]
    while row[-1] < arc_total:
        row.append(row[-1] + spacing_at(row[-1], height))
    row = np.array(row) * arc_total / row[-1]
    nodes_sz += [(pos, depth) for pos in row]
nodes_sz = np.array(nodes_sz)

def triangulate(pts):
    faces = Delaunay(pts).simplices
    area2 = np.abs((pts[faces[:, 1], 0] - pts[faces[:, 0], 0])
                   * (pts[faces[:, 2], 1] - pts[faces[:, 0], 1])
                   - (pts[faces[:, 2], 0] - pts[faces[:, 0], 0])
                   * (pts[faces[:, 1], 1] - pts[faces[:, 0], 1]))
    return faces[0.5 * area2 > 1e-6]

faces = triangulate(nodes_sz)
fixed = ((nodes_sz[:, 1] < 1e-9) | (nodes_sz[:, 1] > mesh_depth_km - 1e-9)
         | (nodes_sz[:, 0] < 1e-9) | (nodes_sz[:, 0] > arc_total - 1e-9))
for _ in range(smooth_iters):
    neighbours = {}
    for tri in faces:
        for node in tri:
            neighbours.setdefault(node, set()).update(tri)
    smoothed = nodes_sz.copy()
    for node in np.nonzero(~fixed)[0]:
        ring = np.array(sorted(neighbours[node] - {node}))
        smoothed[node] = nodes_sz[ring].mean(0)
    nodes_sz = smoothed
    faces = triangulate(nodes_sz)

fault = TriangularPatches("Tule Pond mean event", lon0=frame_lon0,
                          lat0=frame_lat0, verbose=False)
fault.trace(trace_lonlat[:, 0], trace_lonlat[:, 1])
fault.discretize2(every=0.02)
node_x = np.interp(nodes_sz[:, 0], arc_s, trace_x)
node_y = np.interp(nodes_sz[:, 0], arc_s, trace_y)
vertices = np.c_[node_x, node_y, nodes_sz[:, 1]]
strike_vec = np.array([trace_x[-1] - trace_x[0], trace_y[-1] - trace_y[0]])
strike_vec = strike_vec / np.hypot(strike_vec[0], strike_vec[1])
normal_vec = np.array([strike_vec[1], -strike_vec[0]])
if normal_vec[0] < 0:
    normal_vec = -normal_vec                       # + east
if abs(dip_deg - 90.0) > 0.1:
    east_shift = nodes_sz[:, 1] / np.tan(np.deg2rad(dip_deg))
    vertices[:, 0] += east_shift * normal_vec[0]
    vertices[:, 1] += east_shift * normal_vec[1]
faces_own = faces.copy()
fault.Vertices = vertices
fault.Faces = faces_own
fault.vertices2ll()
fault.patch = [np.array([vertices[i] for i in f]) for f in faces_own]
fault.patch2ll()
fault.setdepth()
fault.vertical = abs(dip_deg - 90.0) <= 0.1
fault.homogeneizeStrike(direction=0, sign=-1.0)
fault.initializeslip()
fault.mesh_sz = (nodes_sz.copy(), faces_own, arc_total)
rows_in_top500 = int((row_depths <= 0.5).sum())
print(f"mesh: {len(nodes_sz)} nodes, {len(faces_own)} triangles, "
      f"{rows_in_top500} node rows in the top 500 m")

# %% [markdown]
# ## [4] Data, covariance, downsampling
#
# Notebook 06's five-event mean per track; far-field semivariogram noise
# model; distance-based quadtree from 2 km blocks to one 30 m pixel at the
# trace; equal weights (the geometry already down-weights ascending).

# %%
## [4]
def dist_to_trace(sar, flt, signed=False):
    tx, ty = flt.xf, flt.yf
    d, i = cKDTree(np.c_[tx, ty]).query(np.c_[sar.x, sar.y])
    return d * np.sign(sar.x - tx[i]) if signed else d

def covariance(sar, flt, npairs=300000, seed=0):
    """[m02, verbatim apart from @ -> np.dot; see 07_slip_model_tests]"""
    from scipy.optimize import curve_fit
    d = dist_to_trace(sar, flt)
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

creep_lo_km, creep_hi_km = 56.3, 62.3   # registry 04_creep_length_lo/hi (pinned by 04)
align_coefs = {}
datas = []
fp_projections = {}
for track in tracks:
    z = np.load(os.path.join(field_dir, f"csi_mean_event_{track}.npz"))
    sar = insar(track, lon0=frame_lon0, lat0=frame_lat0, verbose=False)
    sar.read_from_binary(z["vel_mm"], z["lon"], z["lat"], err=z["err_mm"],
                         incidence=z["incidence"], azimuth=z["azimuth"],
                         remove_nan=True, remove_zeros=False)
    trace_dist = dist_to_trace(sar, fault)
    sar.vel[trace_dist > swath_km] = np.nan
    sar.vel[trace_dist < exclude_km] = np.nan
    km_pp = lat_to_km_pp(sar.lat)
    sar.vel[(km_pp < km_window[0]) | (km_pp > km_window[1])] = np.nan
    sar.checkNaNs()
    # far-field plane alignment: mask the slip section (creep +/- align_mask_km),
    # fit 1,x,y to what remains, remove it from the WHOLE track (in place)
    km_pp = lat_to_km_pp(sar.lat)
    far = (km_pp < creep_lo_km - align_mask_km) | (km_pp > creep_hi_km + align_mask_km)
    north = far & (km_pp < creep_lo_km)
    print(f"[{track}] far field before: north {np.median(sar.vel[north]):+.2f} mm, "
          f"south {np.median(sar.vel[far & ~north]):+.2f} mm")
    plane = np.c_[np.ones(sar.x.size), sar.x, sar.y]
    plane_coef = np.linalg.lstsq(plane[far], sar.vel[far], rcond=None)[0]
    align_coefs[track] = plane_coef
    sar.vel[:] = sar.vel - np.dot(plane, plane_coef)
    print(f"[{track}] far field after:  north {np.median(sar.vel[north]):+.2f} mm, "
          f"south {np.median(sar.vel[far & ~north]):+.2f} mm "
          f"(plane {plane_coef[0]:+.2f} mm, {plane_coef[1]:+.3f}, {plane_coef[2]:+.3f} mm/km)")
    fp_projections[track] = float(np.mean(np.abs(np.dot(sar.los[:, :2], strike_vec))))
    cov_p = covariance(sar, fault)

    ds = imdown(sar.name, sar, fault, verbose=False)
    ds.initialstate(start_block_km, min_block_km, tolerance=0.1, plot=False)
    ds.distanceBased(chardist=0.0, expodist=1, plot=False)
    ds = ds.newimage
    nug2, s1, l1, s2, l2 = cov_p
    dx = ds.x[:, None] - ds.x[None, :]; dy = ds.y[:, None] - ds.y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    cd = s1 * np.exp(-dist / l1) + s2 * np.exp(-dist / l2)
    n_px = np.asarray(getattr(ds, "wgt", np.ones(ds.vel.size)), dtype=float)
    n_px = np.where(np.isfinite(n_px) & (n_px > 0), n_px, 1.0)
    cd[np.diag_indices_from(cd)] += nug2 / n_px ** nug_power
    ds.Cd = cd                      # equal weights: no inflation (see header)
    datas.append(ds)
    print(f"[{track}] {ds.vel.size} quadtree samples; "
          f"|LOS . fault-parallel| {fp_projections[track]:.2f}")

# %% [markdown]
# ## [4b] The surface-slip constraint
#
# csi.surfaceslip from the crossing stacks (04): per crossing the mean of
# the five per-event amplitudes, running median every 0.2 km (+/-0.2 km),
# error = 1.4826 MAD / sqrt(n) floored at surf_err_mm. Right-lateral
# positive, mm. The same pixels as the InSAR fields processed as
# fault-crossing time series -- a near-fault constraint, not new data.

# %%
## [4b]
from csi import surfaceslip

crossings_all = pd.read_csv(os.path.join(common_paths["out_dir"],
                                         "04_event_detection_v2",
                                         "event_fraction_by_crossing.csv"))
event_cols = ["amp_2017.099", "amp_2019.186", "amp_2021.107",
              "amp_2023.080", "amp_2024.984"]
crossings_all["mean_event_mm"] = crossings_all[event_cols].mean(axis=1)
trace_south_north = trace_lonlat[np.argsort(trace_lonlat[:, 1])]
surf_rows = []
for k_centre in np.arange(np.ceil(crossings_all.km.min()), crossings_all.km.max(), surf_step_km):
    in_win = np.abs(crossings_all.km - k_centre) <= surf_win_km / 2
    amps = crossings_all.mean_event_mm[in_win].dropna()
    if len(amps) < 4:
        continue
    med = float(np.median(amps))
    err = max(1.4826 * float(np.median(np.abs(amps - med))) / np.sqrt(len(amps)), surf_err_mm)
    lat_c = float(np.median(crossings_all.lat[in_win]))
    lon_c = float(np.interp(lat_c, trace_south_north[:, 1], trace_south_north[:, 0]))
    surf_rows.append((lon_c, lat_c, k_centre, med, err))
surf_rows = np.array(surf_rows)
in_model = (surf_rows[:, 2] >= km_window[0]) & (surf_rows[:, 2] <= km_window[1])
surf_rows = surf_rows[in_model]
# CSI's default link ties each surface point to the single nearest surface
# triangle (Fault.surfaceGFs): 29 points reach ~7% of the creep-section
# surface row. We depart from it: the running-median profile is
# interpolated along strike to the centre of EVERY surface triangle, one
# point each, and the errors are scaled by sqrt(triangles per 0.2 km step)
# so the total weight equals that of the original 29 points.
surf_rows_km = surf_rows[:, 2]
surf_patches = np.where([0 in p[:, 2] for p in fault.patch])[0]
surf_ll = np.array([fault.patchll[i].mean(0) for i in surf_patches])
surf_tri_km = lat_to_km_pp(surf_ll[:, 1])
keep = (surf_tri_km >= surf_rows_km.min()) & (surf_tri_km <= surf_rows_km.max())
surf_patches, surf_ll, surf_tri_km = surf_patches[keep], surf_ll[keep], surf_tri_km[keep]
surf_vals = np.interp(surf_tri_km, surf_rows_km, surf_rows[:, 3])
surf_errs = np.interp(surf_tri_km, surf_rows_km, surf_rows[:, 4])
tri_per_step = np.array([np.sum(np.abs(surf_tri_km - k) <= surf_step_km / 2)
                         for k in surf_tri_km])
surf_errs = surf_errs * np.sqrt(np.maximum(tri_per_step, 1))
surface_data = surfaceslip("crossing stacks", lon0=frame_lon0, lat0=frame_lat0, verbose=False)
surface_data.read_from_binary(surf_vals, surf_ll[:, 0], surf_ll[:, 1], err=surf_errs)
surface_data.buildCd()
print(f"surface constraint: {len(surf_rows)} running-median points interpolated to "
      f"{len(surf_vals)} surface triangles, error floor {surf_err_mm:g} mm")

# %% [markdown]
# ## [5] The inversion
#
# Meade triangular GFs (right-lateral positive), a bilinear ramp per track,
# the buildCmXY correlated prior at the selected sigma, NNLS.

# %%
## [5]
for ds in datas:
    fault.buildGFs(ds, vertical=True, slipdir="s", verbose=False)
    fault.G[ds.name]["strikeslip"] *= -1.0
fault.buildGFs(surface_data, slipdir="s", verbose=False)   # NO sign flip: the
# surface data are right-lateral positive, matching the flipped InSAR GFs

def solve_once(sigma, use_surface=True):
    solve_datas = datas + ([surface_data] if use_surface else [])
    # per-dataset CONSTANT only: the plane alignment already levelled the
    # far field, so the linear terms are out of the model's reach
    trans = transform("ramps", lon0=frame_lon0, lat0=frame_lat0, verbose=False)
    trans.buildGFs(solve_datas, ["1"] * len(solve_datas), verbose=False)
    fault.assembleGFs(solve_datas, slipdir="s"); fault.assembled(solve_datas)
    fault.assembleCd(solve_datas)
    trans.assembleGFs(solve_datas); trans.assembled(solve_datas); trans.assembleCd(solve_datas)
    fault.buildCmXY(sigma, lam_xz, verbose=False)
    trans.buildCm(1e4, verbose=False)
    solver = multisrc("mean event", [fault, trans], verbose=False)
    solver.assembleGFs(); solver.assembleCm()
    mprior = np.zeros(solver.Np)
    ramp_idx = [i for i, t in enumerate(solver.paramTypes) if t[0] == trans.name]
    mprior[ramp_idx] = -1e3
    solver.ConstrainedLeastSquareSoln(method="nnls", mprior=mprior)
    solver.distributem()
    for ds in solve_datas:
        ds.buildsynth([fault], direction="s")
    return solver

# supplement first: WITHOUT the surface constraint (the underestimate)
solver = solve_once(sigma_sel, use_surface=False)
nodes, faces_arr, _ = fault.mesh_sz
table_nosurf = pd.DataFrame(dict(slip_mm=fault.slip[:, 0],
                                 z_top=nodes[faces_arr][:, :, 1].min(1)))
centres_nosurf = np.array([pll.mean(0) for pll in fault.patchll])
table_nosurf["km_pp"] = lat_to_km_pp(centres_nosurf[:, 1])
peak_nosurf = float(table_nosurf.slip_mm.max())
in_creep = lambda km: (km >= creep_lo_km) & (km <= creep_hi_km)
crossing_median = float(np.median(surf_rows[in_creep(surf_rows[:, 2]), 3]))
nosurf_surface = table_nosurf[(table_nosurf.z_top <= 1e-6) & in_creep(table_nosurf.km_pp)]
surf_ratio_nosurf = float(nosurf_surface.slip_mm.median()) / crossing_median
print(f"WITHOUT surface constraint: peak slip {peak_nosurf:.1f} mm; surface slip "
      f"{100 * surf_ratio_nosurf:.0f}% of the crossings over the creep section")

# the selected model: WITH the surface constraint
solver = solve_once(sigma_sel, use_surface=True)
surf_resid = surface_data.vel - surface_data.synth
print(f"surface constraint fit: rms {np.sqrt(np.mean(surf_resid ** 2)):.2f} mm "
      f"over {surf_resid.size} points")
# ramp-removed data for the figures (kept separately; ds.vel stays raw)
ramp_removed = {}
for ds in datas:
    est = ds.vel - ds.synth                        # data - slip synth = ramp + noise
    design = np.c_[np.ones(ds.x.size), ds.x, ds.y]
    ramp_coef = np.linalg.lstsq(design, est, rcond=None)[0]
    ramp_removed[ds.name] = ds.vel - np.dot(design, ramp_coef)

nodes, faces_arr, _ = fault.mesh_sz
centres_sz = nodes[faces_arr].mean(1)
centres_ll = np.array([pll.mean(0) for pll in fault.patchll])
table = pd.DataFrame(dict(
    lon=centres_ll[:, 0], lat=centres_ll[:, 1],
    s_km=centres_sz[:, 0], z_c=centres_sz[:, 1],
    z_top=nodes[faces_arr][:, :, 1].min(1), z_bot=nodes[faces_arr][:, :, 1].max(1),
    km_pp=lat_to_km_pp(centres_ll[:, 1]), slip_mm=fault.slip[:, 0]))

constrained_surface = table[(table.z_top <= 1e-6) & in_creep(table.km_pp)]
surf_ratio = float(constrained_surface.slip_mm.median()) / crossing_median
print(f"WITH surface constraint: surface slip {100 * surf_ratio:.0f}% of the crossings")

moment = 0.0
for patch, slip in zip(fault.patch, table.slip_mm.values):
    area_km2 = 0.5 * np.linalg.norm(np.cross(patch[1] - patch[0], patch[2] - patch[0]))
    moment += shear_modulus * area_km2 * 1e6 * slip * 1e-3
mw = (2.0 / 3.0) * (np.log10(moment) - 9.1)

stats = {}
for ds in datas:
    track_key = ds.name.split()[-1][:3]
    resid = ds.vel - ds.synth
    resid = resid - np.dot(np.c_[np.ones(ds.x.size), ds.x, ds.y],
                           np.linalg.lstsq(np.c_[np.ones(ds.x.size), ds.x, ds.y],
                                           resid, rcond=None)[0])
    stats[f"rms_{track_key}"] = float(np.sqrt(np.mean(resid ** 2)))
    near = dist_to_trace(ds, fault) <= near_field_km
    stats[f"rms_near_{track_key}"] = float(np.sqrt(np.mean(resid[near] ** 2)))
    stats[f"n_{track_key}"] = int(resid.size)
print(f"selected model: sigma {sigma_sel:g}, Mw {mw:.2f}, "
      f"peak slip {table.slip_mm.max():.1f} mm, "
      f"desc near-field rms {stats['rms_near_des']:.2f} mm")

# %% [markdown]
# ## [6] Bootstrap -- the formal error
#
# n_boot noise realisations drawn from the full data covariance (Cholesky),
# added to the SYNTHETIC of the selected model, re-inverted; the spread of
# Mw is the formal uncertainty of THIS configuration (the structural range
# across configurations comes from the suite).

# %%
## [6]
rng = np.random.default_rng(11)
boot_datas = datas + [surface_data]
chols = [np.linalg.cholesky(ds.Cd + 1e-9 * np.eye(ds.Cd.shape[0])) for ds in boot_datas]
clean_synth = [ds.synth.copy() for ds in boot_datas]
real_vel = [ds.vel.copy() for ds in boot_datas]
boot_mw = []
for k in range(n_boot):
    for ds, chol, synth in zip(boot_datas, chols, clean_synth):
        # in place: fault.d holds a reference to THIS array (CSI setGFs)
        ds.vel[:] = synth + np.dot(chol, rng.standard_normal(ds.vel.size))
    solve_once(sigma_sel)
    m0_k = 0.0
    for patch, slip in zip(fault.patch, fault.slip[:, 0]):
        area_km2 = 0.5 * np.linalg.norm(np.cross(patch[1] - patch[0], patch[2] - patch[0]))
        m0_k += shear_modulus * area_km2 * 1e6 * slip * 1e-3
    boot_mw.append((2.0 / 3.0) * (np.log10(m0_k) - 9.1))
    print(f"  boot {k + 1}/{n_boot}: Mw {boot_mw[-1]:.3f}")
for ds, vel in zip(boot_datas, real_vel):
    ds.vel[:] = vel
solver = solve_once(sigma_sel)                     # restore the real solution
boot_mw = np.array(boot_mw)
mw_boot_sd = float(boot_mw.std(ddof=1))
print(f"bootstrap: Mw {boot_mw.mean():.3f} +/- {mw_boot_sd:.3f} (n={n_boot})")

# %% [markdown]
# ## [7] Figures
#
# (a) quadtree data / model / residual per track (ramp removed);
# (b) slip distribution with the X-X' depth profile at the peak;
# (c) surface slip: the constrained fit against the crossing stacks, with
#     the InSAR-only (unconstrained) curve showing the underestimate.

# %%
## [7]
mpl_paper_style(base=7)
trace_plot = np.c_[fault.lon, fault.lat]

fig, axes = plt.subplots(2, 3, figsize=(7.25, 5.4), constrained_layout=True)
letters = "ABCDEF"
for r, ds in enumerate(datas):
    shown = ramp_removed[ds.name]
    panels = ((shown, "data"), (ds.synth, "model"), (shown - ds.synth, "residual"))
    for c, (values, title) in enumerate(panels):
        ax = axes[r, c]
        sc = ax.scatter(ds.lon, ds.lat, c=values, s=4, cmap="RdBu_r",
                        vmin=-3, vmax=3, linewidths=0)
        ax.plot(trace_plot[:, 0], trace_plot[:, 1], "k-", lw=0.5)
        ax.set_aspect(1 / np.cos(np.deg2rad(frame_lat0)))
        ax.set_xlim(-122.10, -121.84); ax.set_ylim(37.46, 37.65)
        ax.set_title(r"$\bf{" + letters[r * 3 + c] + r"}$  "
                     + f"{ds.name.split()[-1][:4]} {title}", loc="left", fontsize=7)
        if c:
            ax.set_yticklabels([])
        if r == 0:
            ax.set_xticklabels([])
fig.colorbar(sc, ax=axes, shrink=0.5, label="LOS (mm), ramp removed",
             orientation="horizontal", pad=0.04)
fig.savefig(os.path.join(fig_dir, "fig07_fit_maps.png"), dpi=300)
print("wrote fig07_fit_maps.png")

# %%
## [8]
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(7.25, 4.6), constrained_layout=True)
gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.1], width_ratios=[2.2, 1])

ax = fig.add_subplot(gs[0, :])
pc = ax.tripcolor(nodes[:, 0], nodes[:, 1], faces_arr, facecolors=table.slip_mm.values,
                  cmap="viridis", vmin=0, vmax=max(table.slip_mm.max(), 1),
                  edgecolors="none")
peak_s = table.loc[table.slip_mm.idxmax(), "s_km"]
ax.axvline(peak_s, color="w", lw=0.7, ls="--")
ax.text(peak_s + 0.15, 0.25, "X-X'", color="w", fontsize=6)
ax.set_xlim(0, arc_total); ax.set_ylim(mesh_depth_km, 0)
ax.set_xlabel("Along-strike distance from north end (km)")
ax.set_ylabel("Depth (km)")
fig.colorbar(pc, ax=ax, label="Right-lateral slip (mm)", shrink=0.8)
ax.set_title(r"$\bf{A}$  " + f"Slip distribution, dip {dip_deg:g}°E, "
             f"Mw {mw:.2f}", loc="left", fontsize=7)

ax = fig.add_subplot(gs[1, 0])
crossings = pd.read_csv(os.path.join(common_paths["out_dir"],
                                     "04_event_detection_v2",
                                     "event_fraction_by_crossing.csv"))
event_cols = ["amp_2017.099", "amp_2019.186", "amp_2021.107",
              "amp_2023.080", "amp_2024.984"]
crossings["mean_event_mm"] = crossings[event_cols].mean(axis=1)
crossings = crossings[(crossings.km >= km_window[0]) & (crossings.km <= km_window[1])]
surface = table[table.z_top <= 1e-6].sort_values("km_pp")
surface_med_km = np.arange(km_window[0], km_window[1], 0.2)
surface_med = np.array([np.median(surface.slip_mm[(surface.km_pp >= k - 0.3)
                                                  & (surface.km_pp <= k + 0.3)])
                        for k in surface_med_km])
nosurf_surface = table_nosurf[table_nosurf.z_top <= 1e-6].sort_values("km_pp")
nosurf_med = np.array([np.median(nosurf_surface.slip_mm[(nosurf_surface.km_pp >= k - 0.3)
                                                        & (nosurf_surface.km_pp <= k + 0.3)])
                       for k in surface_med_km])
ax.scatter(crossings.km, crossings.mean_event_mm, s=4, c="0.6", linewidths=0,
           label="crossing stacks, mean of 5 events")
ax.plot(surface.km_pp, surface.slip_mm, "-", color="#F0B884", lw=0.5)
ax.plot(surface_med_km, surface_med, "-", color="#D55E00", lw=1.4,
        label="model surface slip (constrained)")
ax.plot(surface_med_km, nosurf_med, "--", color="0.35", lw=1.0,
        label="InSAR only (supplement)")
ax.set_xlim(km_window)
ax.set_xlabel("Distance from Point Pinole (km)")
ax.set_ylabel("Slip per event (mm)")
ax.legend(fontsize=6)
ax.set_title(r"$\bf{B}$  Surface slip, constrained fit", loc="left", fontsize=7)

ax = fig.add_subplot(gs[1, 1])
near_peak = table[(table.s_km - peak_s).abs() < 0.3]
depth_profile = near_peak.groupby(np.round(0.5 * (near_peak.z_top + near_peak.z_bot), 2)).slip_mm.max()
ax.plot(depth_profile.values, depth_profile.index.values, "o-", ms=2.5,
        color="#E69F00")
ax.set_ylim(mesh_depth_km, 0)
ax.set_xlabel("Slip (mm)"); ax.set_ylabel("Depth (km)")
ax.set_title(r"$\bf{C}$  X-X' depth profile",
             loc="left", fontsize=7)
fig.savefig(os.path.join(fig_dir, "fig07_slip_model.png"), dpi=300)
fig.savefig(os.path.join(fig_dir, "fig07_slip_model.pdf"))
print("wrote fig07_slip_model.png/.pdf")

# %%
## [8b]
# Fig 2 bottom-row panels for 11_paper_figures: I descending model at the
# quadtree samples, J descending residual, K slip with depth and Mw --
# emitted as standalone panels on the Fig 2 map extent.
desc = [ds for ds in datas if "descending" in ds.name][0]
desc_shown = ramp_removed[desc.name]
panel_defs = [("fig07_panelI_desc_model", desc.synth, "Model (quadtree samples)"),
              ("fig07_panelJ_desc_residual", desc_shown - desc.synth, "Residual")]
for fname, values, title in panel_defs:
    fig, ax = plt.subplots(figsize=(2.4, 2.4), constrained_layout=True)
    sc = ax.scatter(desc.lon, desc.lat, c=values, s=5, cmap="RdBu_r",
                    vmin=-2.5, vmax=2.5, linewidths=0)
    ax.plot(trace_plot[:, 0], trace_plot[:, 1], "k-", lw=0.6)
    ax.set_aspect(1 / np.cos(np.deg2rad(frame_lat0)))
    ax.set_xlim(-122.10, -121.86); ax.set_ylim(37.47, 37.65)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{fname}.{ext}"), dpi=300)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(4.6, 2.2), constrained_layout=True)
pc = ax.tripcolor(nodes[:, 0], nodes[:, 1], faces_arr, facecolors=table.slip_mm.values,
                  cmap="viridis", vmin=0, vmax=max(table.slip_mm.max(), 1),
                  edgecolors="none")
ax.axvline(peak_s, color="w", lw=0.7, ls="--")
ax.set_xlim(0, arc_total); ax.set_ylim(mesh_depth_km, 0)
ax.set_xlabel("Along-strike (km)"); ax.set_ylabel("Depth (km)")
fig.colorbar(pc, ax=ax, label="Slip (mm)", shrink=0.85)
ax.set_title(f"Mw {mw:.2f}", fontsize=7, loc="right")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(fig_dir, f"fig07_panelK_slip_depth.{ext}"), dpi=300)
plt.close(fig)
print("wrote Fig 2 panels I, J, K")

# %% [markdown]
# ## [9] Registry -- every number the manuscript quotes

# %%
## [9]
dip_family = pd.read_csv(os.path.join(tests_dir, "dip_on_selected_mesh.csv"))
mesh_best = (suite[suite.family == "mesh"].sort_values("rms_mm")
             .groupby("mesh", as_index=False).first())
iso = suite[(suite.family == "isolation")
            & suite.label.str.startswith(("floor", "asc_x2"))]
accepted = dip_family.copy()     # dip on the selected mesh; the full suite reruns tonight
print("structural set:", ", ".join(f"{r.label} {r.Mw:.2f}" for _, r in accepted.iterrows()))
sidecar = REG.sidecar("07_slip_model", [
    f"selected: mesh smooth12 (50 m x1.2), dip {dip_deg:g}E, sigma {sigma_sel:g}, "
    f"quadtree {start_block_km:g} km -> {min_block_km * 1000:.0f} m, equal weights "
    f"(explicit asc down-weighting tested in the suite: |dMw| < 0.02, skipped)",
    f"Mw {mw:.2f} (boot sd {mw_boot_sd:.3f}, n={n_boot}); "
    f"suite accepted range {accepted.Mw.min():.2f}-{accepted.Mw.max():.2f} "
    f"({len(accepted)} configurations)",
    f"peak slip {table.slip_mm.max():.1f} mm; depth best {best_D.D_km:g} km "
    f"({ok_D.D_km.min():g}-{ok_D.D_km.max():g})",
    f"desc near-field (<{near_field_km * 1000:.0f} m) rms {stats['rms_near_des']:.2f} mm; "
    f"asc {stats['rms_near_asc']:.2f} mm",
    f"surface constraint: {len(surf_rows)} pts, floor {surf_err_mm:g} mm; "
    f"WITHOUT it peak slip {peak_nosurf:.1f} mm (underestimate, supplement)",
    f"pre-alignment planes (mm; mm/km): " + "; ".join(
        f"{t} {c[0]:+.2f}, {c[1]:+.3f}, {c[2]:+.3f}" for t, c in align_coefs.items()),
    f"LOS fault-parallel sensitivity asc {fp_projections['ascending']:.2f}, "
    f"desc {fp_projections['descending']:.2f}",
])
REG.register("07_mw", mw, "Mw", 2, where="P5, M7", cell="invert", sidecar=sidecar)
REG.register("07_mw_boot_sd", mw_boot_sd, "Mw", 3, where="M7", cell="bootstrap", sidecar=sidecar)
REG.register("07_mw_suite_lo", accepted.Mw.min(), "Mw", 2, where="M7", cell="suite", sidecar=sidecar)
REG.register("07_mw_suite_hi", accepted.Mw.max(), "Mw", 2, where="M7", cell="suite", sidecar=sidecar)
REG.register("07_mw_dip_lo", dip_family.Mw.min(), "Mw", 2, where="M7", cell="suite", sidecar=sidecar)
REG.register("07_mw_dip_hi", dip_family.Mw.max(), "Mw", 2, where="M7", cell="suite", sidecar=sidecar)
REG.register("07_peak_slip_mm", table.slip_mm.max(), "mm", 1, where="P5", cell="invert", sidecar=sidecar)
REG.register("07_depth_best_km", best_D.D_km, "km", 2, where="P5, M7", cell="depth", sidecar=sidecar)
REG.register("07_depth_lo_km", ok_D.D_km.min(), "km", 2, where="M7", cell="depth", sidecar=sidecar)
REG.register("07_depth_hi_km", ok_D.D_km.max(), "km", 2, where="M7", cell="depth", sidecar=sidecar)
REG.register("07_desc_near_rms_mm", stats["rms_near_des"], "mm", 2, where="M7", cell="metrics", sidecar=sidecar)
REG.register("07_sigma_sel", sigma_sel, "mm", 0, where="M7", cell="select", sidecar=sidecar)
REG.register("07_dip_deg", dip_deg, "deg", 0, where="M7", cell="constants", sidecar=sidecar)
REG.register("07_minblock_m", min_block_km * 1000, "m", 0, where="M7", cell="constants", sidecar=sidecar)
REG.register("07_swath_km", swath_km, "km", 0, where="M7", cell="constants", sidecar=sidecar)
REG.register("07_mesh_length_km", arc_total, "km", 1, where="M7", cell="mesh", sidecar=sidecar)
REG.register("07_cell_top_m", cell_top_km * 1000, "m", 0, where="M7", cell="mesh", sidecar=sidecar)
REG.register("07_cell_bot_m", np.diff(row_depths).max() * 1000, "m", 0, where="M7", cell="mesh", sidecar=sidecar)
weighting_rows = suite[(suite.family == "isolation") & (suite.label == "asc_x2")]
base_mw = float(suite[(suite.family == "sigma_surf") & (suite.sigma == sigma_sel)].Mw.iloc[0])
REG.register("07_weighting_dmw_max", float(np.max(np.abs(weighting_rows.Mw - base_mw))),
             "Mw", 3, where="M7", cell="suite", sidecar=sidecar)
REG.register("07_fp_proj_asc", fp_projections["ascending"], "", 2, where="M7", cell="load", sidecar=sidecar)
REG.register("07_fp_proj_des", fp_projections["descending"], "", 2, where="M7", cell="load", sidecar=sidecar)
REG.register("07_n_samples", int(sum(stats[f"n_{t[:3]}"] for t in tracks)), "count", 0,
             where="M7", cell="load", sidecar=sidecar)
REG.register("07_peak_slip_nosurf_mm", peak_nosurf, "mm", 1, where="M7", cell="invert", sidecar=sidecar)
REG.register("07_surf_ratio_insar_only_pct", 100 * surf_ratio_nosurf, "%", 0, where="M7", cell="invert", sidecar=sidecar)
REG.register("07_surf_ratio_constrained_pct", 100 * surf_ratio, "%", 0, where="M7", cell="invert", sidecar=sidecar)
REG.register("07_n_surface_points", len(surf_vals), "count", 0, where="M7", cell="surface", sidecar=sidecar)
REG.register("07_surf_err_floor_mm", surf_err_mm, "mm", 1, where="M7", cell="constants", sidecar=sidecar)
for track_key, coefs in align_coefs.items():
    REG.register(f"07_align_{track_key[:3]}_const_mm", coefs[0], "mm", 2, where="M7",
                 cell="load", sidecar=sidecar)
table.to_csv(os.path.join(out_dir, "slip_model_selected.csv"), index=False)
REG.render()
print("07_slip_model complete")
