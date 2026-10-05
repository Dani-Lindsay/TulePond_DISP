# %% [markdown]
# # 07a Uniform model -- the homogeneous half-space baseline (CSI)
#
# ```
# Created on Thu Sep 25 2026; rewritten on Mon Sep 28 2026 as the clean baseline
# @author: daniellelindsay
# Run in opera_disp_feb13, after 04 and 06. 38 inversions (13 for the L-curve, the model, 24 bootstrap).
# ```
#
# **What this notebook does.** The slip model of the five-event mean fault-parallel field with
# **uniform-rock (homogeneous half-space) Green's functions** (CSI, Meade triangular
# dislocations). It is the baseline against which the paper's selected model, the 3D PyLith
# model in the USGS SFCVM (`08a_3d_model`; labelled SFSVM in the figures), is measured. The rock
# enters the INVERSION as uniform; the USGS rigidity only rescales the moment afterwards (Mw with
# SFCVM mu(z)).
# PREPUB: Every lesson of the 2026-09-25 to 28 tests is applied (`07_sweep_lessons_2026-09-28.md`, `drafts/07/`).
#
# | Choice | Value | Evidence |
# |---|---|---|
# | Data | fault-parallel mean event (06 `fp_maps.npz`, five events) | residual maps: removes the ascending east-side patch |
# | Decomposition strike | mean local strike of the SSE section (06 `strike`, registry `06_strike_local_deg`) | 06; Dani 2026-09-28 |
# | Coherence | 0.9 for the field | 06 `coh_maps`; Dani 2026-09-28 |
# | Levelling | plane fitted to the far field (section ± 1 km masked) | removes a far-field tilt; the plane is printed in [3] |
# | Fault | vertical, on the hand-drawn trace | dip unresolved 60°E-75°W; Mw across it in 07b (registry `07b_dip_mw_min`, `07b_dip_mw_max`) |
# | Mesh | `smooth12`: 50 m cells at the surface, x1.2 per row, 3 km | no resolution plateau (07b mesh family) |
# | Sampling | distance-based quadtree, 2 km to one 30 m pixel | |
# | Prior | `buildCmXY`, lambda (0.1, 0.05) km, sigma at the L-curve corner (grid value at or above it) | L-curve in [8]; lambda family in 07b |
# | Surface | crossing-stack running median interpolated to every surface triangle | 07b isolation row `insar_only` (no surface constraint) against the baseline (`07a_hom_surface_pct`) |
# | Top tie | none (dropped 2026-09-29) | at the L-curve smoothing the untied model peaks at the surface (07b tie family) |
# | Solver | NNLS (right-lateral >= 0), per-dataset constant | |
#
# PREPUB: Evidence scripts: lambda `drafts/07/along_strike_smoothing.py`; the top tie `drafts/07/tie_decision.py` (the tie was needed only at sigma >= 24).
#
# Cells [2]-[6] are the master copy of the machinery (mesh, levelling, noise model, quadtree,
# surface constraint, tie, inversion). `07b_uniform_model_tests` carries a verbatim copy (its
# CHECK [16] asserts the same slip to 1e-6 mm) and `08a_3d_model` execs cells [1]-[6] unchanged,
# splitting this file on the first `### CHECK` heading (cell [7]). Keep that heading, the single
# registry line and `REG.inputs` call and the names 08a takes from [1]-[6], and make
# every edit of [2]-[6] in 07b's copy too. CSI calls follow the tutorials (`external/csi/notebooks/`).
#
# **Inputs**: 06 `fp_maps.npz`, `fig2_inputs.npz` (the profile lines, [13]) and
# `csi_mean_event_{ascending,descending}.npz` (the LOS geometry, [10]); 04
# `event_fraction_by_crossing.csv`, `creep_events.csv`, `section_lengths.csv`; the trace and
# `hayward_fault_profiles.csv`; the USGS SFCVM v26.0 creep-section profiles
# `07b_uniform_model_tests/sfcvm_creep_section_profiles_v26-0.csv`.
#
# **Outputs** (`outputs/07a_uniform_model/`): `sigma_lcurve.csv` (the sigma pick, read by 07b),
# `slip_homogeneous_baseline.csv` (read by 07b's CHECK [16]), the stats sidecar
# `07a_hom_baseline_stats.txt`; and `fig2_model_row.npz`, written to
# `outputs/07b_uniform_model_tests/`, where 08a [7] (the uniform model on the same samples) and
# 07c (the sample check) read it. **Figures** (QC, not paper figures): `fig07_hom_lcurve`,
# `fig07_hom_fit_maps`, `fig07_hom_slip`. **Registry keys** `07a_hom_*`.
#
# **What to check**
# - [3] the levelling plane is small (tenths of a mm, mm/km).
# - [4] the noise fit: a long-range sill of 0 means the noise model is a nugget plus one exponential.
# - [8] the L-curve corner lies inside the searched range (between the second and second-last grid
#   sigma) and the pick is the next grid sigma at or above it.
# - [9] Mw at 30 GPa and with the SFCVM rigidity; the surface % of the crossing slip.
# - [10] PASS: the model's own fault-normal motion leaks into the fault-parallel field below the noise.
# - [11] the bootstrap mean against the model Mw (printed): re-inverted noisy synthetics can come
#   back low, so the bootstrap sd is the scatter of this configuration, not an error bar centred on
#   the model.

# %% [markdown]
# ## [1] Paths and constants
#
# Every number the model depends on, the folders, and the registry with its inputs. Cells [1]-[6]
# are exec'd by `08a_3d_model` (see the header).

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
import pyproj
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter, NullFormatter
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
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

notebook = "07a_uniform_model"
out_dir = os.path.join(common_paths["out_dir"], notebook)
tests_dir = os.path.join(common_paths["out_dir"], "07b_uniform_model_tests")
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
lam_xz         = (0.1, 0.05)   # prior correlation lengths along strike, down dip (km). Along strike 0.1 km (Dani 2026-09-29):
                               # the shortest that keeps the sigma pick stable, sharper section edges, Mw unchanged;
                               # down dip 0.05 km so the prior does not push slip deep
                               # (drafts/07/along_strike_smoothing.py)  # PREPUB
tie_depth_km   = 0.0           # no top tie (Dani 2026-09-29): at the L-curve smoothing the untied model peaks at the surface;
                               # the tie code in [6] stays so [2]-[6] equal 07b's copy, whose tie family uses it
                               # (drafts/07/tie_decision.py)  # PREPUB
shear_modulus  = 3.0e10        # uniform rigidity of the reference Mw
mu_floor_gpa   = 0.5           # SFCVM rigidity floor (the model is ~0 at the surface)
sigma_grid     = [4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128]   # L-curve (larger = LESS smoothing)
sigma_check    = 24            # sigma of the draft the CHECK compares against  # PREPUB
sfcvm_version  = "v26-0"       # USGS SFCVM v26.0 (2026-08-14; new rules west of the Hayward fault), Dani 2026-09-29
# the creep-section profiles are made by drafts/07/sfcvm_extract.py v26-0  # PREPUB: the extract script lives in drafts/
sfcvm_file     = os.path.join(tests_dir, f"sfcvm_creep_section_profiles_{sfcvm_version}.csv")
assert os.path.exists(sfcvm_file), (f"missing {sfcvm_file}: download USGS_SFCVM_{sfcvm_version}_detailed.h5 "
                                    "(doi:10.5066/P13AEJFI) and run the SFCVM extract script (comment above) first")
n_boot         = 24

REG = utils.Registry(notebook, common_paths, repo_dir)
REG.inputs(os.path.join(field_dir, "fp_maps.npz"),
           common_paths["fault_trace"], common_paths["fault_profiles"],
           os.path.join(events_dir, "event_fraction_by_crossing.csv"),
           os.path.join(events_dir, "creep_events.csv"),
           os.path.join(events_dir, "section_lengths.csv"),
           os.path.join(field_dir, "fig2_inputs.npz"),
           os.path.join(field_dir, "csi_mean_event_ascending.npz"),
           os.path.join(field_dir, "csi_mean_event_descending.npz"),
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
    """km from Point Pinole of each latitude: the km of the nearest vertex of the fault-profile
    trace (vertices 0.1 km apart, km = (n - 1 - index) x 0.1). Worked in chunks of 20000
    latitudes so the latitude-difference matrix stays small."""
    lats = np.asarray(lats, float)
    out = np.empty(lats.size)
    for lo in range(0, lats.size, 20000):
        chunk = lats[lo:lo + 20000]
        lat_diff = np.abs(profile_points.ft_lat.values[None, :] - chunk[:, None])
        out[lo:lo + 20000] = profile_km_pp[lat_diff.argmin(axis=1)]
    return out


trace_km_pp = lat_to_km_pp(ref_fault.lati)


def build_unrolled(h_top, const_to, growth, h_bot, z_max=mesh_depth_km):
    """The mesh in the unrolled (s, z) plane, in km. Node rows from the surface to z_max: the
    first row h_top deep, the row height constant down to const_to, then multiplied by
    `growth` per row up to h_bot. Along strike the node spacing equals the row height inside
    fine_km and widens by far_rate per km outside it. Delaunay triangles (slivers dropped),
    then smooth_iters rounds of Laplacian smoothing of the interior nodes.
    Returns the nodes (s, z) and the faces."""
    # [07b_uniform_model_tests, verbatim]  # PREPUB
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
        for pos in row:
            nodes.append((pos, depth))
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
    in place)."""
    # [07b_uniform_model_tests, verbatim]  # PREPUB
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
    flt.patch = []
    for face in faces_own:
        flt.patch.append(vertices[face])               # the three corners of the triangle (x, y, z)
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
def dist_to_trace(sar, fault):
    """Horizontal distance (km) from each sample of `sar` to the nearest point of the
    fault's discretised trace (fault.xf, fault.yf)."""
    # [csi_io, verbatim]  # PREPUB
    trace_tree = cKDTree(np.c_[fault.xf, fault.yf])
    dist, _ = trace_tree.query(np.c_[sar.x, sar.y])
    return dist


def align_track(sar):
    """Far-field plane alignment: slip section +/- align_mask_km masked,
    1,x,y fit to the rest, removed from the whole track in place."""
    # [07b_uniform_model_tests, verbatim]  # PREPUB
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
# Far-field semivariogram: nugget + short- and long-range exponentials
# (`semivariogram_model`), fitted to pixel pairs beyond 1 km of the trace with a
# plane removed. Check the printed fit: when the long-range sill comes out 0 (its
# range then sits at a bound and means nothing) the noise model is a nugget plus
# one exponential. Distance-based quadtree (`csi.imagedownsampling`) from 2 km
# blocks to one 30 m pixel at the trace; the full data covariance of the samples.

# %%
## [4]
def semivariogram_model(distance_km, nugget2, sill_short, range_short_km, sill_long, range_long_km):
    """Semivariance (mm^2) at a distance h (km): a nugget plus a short- and a long-range
    exponential, gamma(h) = n2 + s1 (1 - exp(-h / l1)) + s2 (1 - exp(-h / l2))."""
    short_term = sill_short * (1 - np.exp(-distance_km / range_short_km))
    long_term = sill_long * (1 - np.exp(-distance_km / range_long_km))
    return nugget2 + short_term + long_term


def covariance(sar, fault, npairs=300000, seed=0):
    """Fit semivariogram_model to the far field: the pixels beyond min(cov_mask_km, 0.4 swath_km)
    of the trace, a plane (1, x, y) removed; npairs random pixel pairs closer than
    min(cov_distmax_km, 4 swath_km), binned every 0.1 km (bins with more than 50 pairs).
    Returns (nugget2, sill_short, range_short_km, sill_long, range_long_km), in mm^2 and km."""
    # [m02, verbatim apart from @ -> np.dot]  # PREPUB
    dist = dist_to_trace(sar, fault)
    keep = (dist > min(cov_mask_km, 0.4 * swath_km)) & np.isfinite(sar.vel)
    x, y, values = sar.x[keep], sar.y[keep], sar.vel[keep]
    design = np.c_[np.ones_like(x), x, y]
    values = values - np.dot(design, np.linalg.lstsq(design, values, rcond=None)[0])
    rng = np.random.default_rng(seed)
    first = rng.integers(0, values.size, npairs)
    second = rng.integers(0, values.size, npairs)
    pair_dist = np.hypot(x[first] - x[second], y[first] - y[second])
    pair_diff = values[first] - values[second]
    max_dist_km = min(cov_distmax_km, 4 * swath_km)
    in_range = (pair_dist > 0) & (pair_dist < max_dist_km)
    bins = np.arange(0, max_dist_km + 0.1, 0.1)
    bin_index = np.digitize(pair_dist[in_range], bins) - 1
    diff_in_range = pair_diff[in_range]
    gamma = np.full(len(bins) - 1, np.nan)
    for k in range(len(bins) - 1):
        in_bin = bin_index == k
        if in_bin.sum() > 50:
            gamma[k] = 0.5 * np.mean(diff_in_range[in_bin] ** 2)
    bin_centres = 0.5 * (bins[1:] + bins[:-1])
    fitted = np.isfinite(gamma)
    sill = np.nanmean(gamma[fitted][-5:])
    params, _ = curve_fit(semivariogram_model, bin_centres[fitted], gamma[fitted],
                          p0=(0.4 * sill, 0.3 * sill, 0.7, 0.5 * sill, 5.0),
                          bounds=([0, 0, 0.1, 0, 2.0], [2 * sill, 2 * sill, 2.0, 3 * sill, 50]), maxfev=20000)
    return params


def downsample(sar, fault, start_km, min_km):
    """Distance-based quadtree (csi.imagedownsampling): start_km blocks cut down to min_km
    toward the trace; a regular grid when start_km == min_km."""
    # [m02, verbatim]  # PREPUB
    sampler = imdown(sar.name, sar, fault, verbose=False)
    sampler.initialstate(start_km, min_km, tolerance=0.1, plot=False)
    if start_km > min_km + 1e-9:
        sampler.distanceBased(chardist=0.0, expodist=1, plot=False)
    return sampler.newimage


def build_cd(ds, cov_p):
    """The data covariance of the quadtree samples (mm^2), set on ds.Cd:
    Cd_ij = s1 exp(-r_ij / l1) + s2 exp(-r_ij / l2), plus nugget2 / n_pixels**nug_power on the
    diagonal (n_pixels = the pixels each block averages, CSI's `wgt`)."""
    # [m02; 07b_uniform_model_tests, verbatim]  # PREPUB
    nugget2, sill_short, range_short_km, sill_long, range_long_km = cov_p
    dx = ds.x[:, None] - ds.x[None, :]
    dy = ds.y[:, None] - ds.y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    cd = sill_short * np.exp(-dist / range_short_km) + sill_long * np.exp(-dist / range_long_km)
    n_px = np.asarray(ds.wgt, dtype=float)
    assert n_px.shape == (ds.vel.size,) and np.all(np.isfinite(n_px) & (n_px > 0)), \
        "the quadtree block pixel counts (wgt) are missing or not positive"
    cd[np.diag_indices_from(cd)] += nugget2 / n_px ** nug_power
    ds.Cd = cd
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
# the total weight is unchanged. This running median is the uniform model's
# surface constraint only: the 3D model (08a) ties each measured crossing
# directly (`pylith_helpers/surface_crossings.py`), and uses `crossing_median`
# (the creep-section median of this profile) as the denominator of its surface %.

# %%
## [5]
crossings_all = pd.read_csv(os.path.join(events_dir, "event_fraction_by_crossing.csv"))
creep_events = pd.read_csv(os.path.join(events_dir, "creep_events.csv"))
tulepond_onsets = creep_events[creep_events.segment == "TULEPOND"].onset_lo.values    # the step epoch (last pre-event acquisition)
event_cols = [f"amp_{onset:.3f}" for onset in tulepond_onsets]    # 04's column naming
print(f"mean event of {len(event_cols)} Tule Pond events: {', '.join(event_cols)}")
crossings_all["mean_event_mm"] = crossings_all[event_cols].mean(axis=1)
trace_south_north = trace_lonlat[np.argsort(trace_lonlat[:, 1])]

def make_surface(floor_mm):
    """csi.surfaceslip: running median of the 5-event crossing means,
    error = 1.4826 MAD / sqrt(n) floored at floor_mm."""
    # [07b_uniform_model_tests, verbatim]  # PREPUB
    rows = []
    for k_centre in np.arange(np.ceil(crossings_all.km.min()),
                              crossings_all.km.max(), surf_step_km):
        in_win = np.abs(crossings_all.km - k_centre) <= surf_win_km / 2 + 1e-6   # 1e-6: keep the edge crossings (float roundoff dropped them, fixed 2026-10-02)
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
    surface_data = surfaceslip(f"crossing stacks f{floor_mm:g}", lon0=frame_lon0,
                               lat0=frame_lat0, verbose=False)
    surface_data.read_from_binary(rows[:, 3], rows[:, 0], rows[:, 1], err=rows[:, 4])
    surface_data.buildCd()
    return surface_data

def make_surface_interp(fault, floor_mm):
    """The running-median profile interpolated to the centre of EVERY
    surface triangle of this fault, each error scaled by sqrt(triangles within
    +/- surf_step_km / 2) so the total weight per profile step is unchanged."""
    # [07b_uniform_model_tests, verbatim]  # PREPUB
    base = make_surface(floor_mm)
    base_km = lat_to_km_pp(base.lat)
    order = np.argsort(base_km)
    surface_tri = []
    for index, patch in enumerate(fault.patch):
        if 0 in patch[:, 2]:
            surface_tri.append(index)
    centre_ll = np.array([fault.patchll[index].mean(0) for index in surface_tri])
    tri_km = lat_to_km_pp(centre_ll[:, 1])
    keep = (tri_km >= base_km.min()) & (tri_km <= base_km.max())
    centre_ll, tri_km = centre_ll[keep], tri_km[keep]
    vals = np.interp(tri_km, base_km[order], base.vel[order])
    errs = np.interp(tri_km, base_km[order], base.err[order])
    per_step = np.zeros(tri_km.size)
    for k, km in enumerate(tri_km):
        per_step[k] = np.sum(np.abs(tri_km - km) <= surf_step_km / 2 + 1e-6)
    errs = errs * np.sqrt(np.maximum(per_step, 1))
    surface_data = surfaceslip(f"crossing interp f{floor_mm:g}", lon0=frame_lon0,
                               lat0=frame_lat0, verbose=False)
    surface_data.read_from_binary(vals, centre_ll[:, 0], centre_ll[:, 1], err=errs)
    surface_data.buildCd()
    return surface_data

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
# (`buildCmXY`), NNLS (`multisourcesolve`). The optional top tie (every cell
# shallower than tie_depth_km equal to the shallowest cell of its 0.1 km
# along-strike column; after Jolivet et al. 2023, who tie 0-1 km) is off in
# the uniform model (tie_depth_km = 0) and used only by 07b's tie family.
# PREPUB: [07b_uniform_model_tests, verbatim]

# %%
## [6]
def tie_rows(fault, n_params, tie_depth_km, tie_bin_km=0.1, tie_err_mm=0.01):
    """The top tie as extra data rows: in every tie_bin_km along-strike column, each cell
    shallower than tie_depth_km minus the column's shallowest cell = 0, with a tiny error
    (a near-hard constraint). Returns the rows (n_ties x n_params) and their variances."""
    nodes, faces, _ = fault.mesh_sz
    centroid = nodes[faces].mean(1)
    shallow = np.nonzero(centroid[:, 1] < tie_depth_km)[0]
    column_of = np.floor(centroid[shallow, 0] / tie_bin_km).astype(int)
    rows = []
    for column in np.unique(column_of):
        members = shallow[column_of == column]
        if members.size < 2:
            continue
        reference = members[np.argmin(centroid[members, 1])]
        for member in members[members != reference]:
            row = np.zeros(n_params)
            row[member] = 1.0
            row[reference] = -1.0
            rows.append(row)
    if rows:
        tie_matrix = np.array(rows)
    else:
        tie_matrix = np.zeros((0, n_params))
    return tie_matrix, np.full(tie_matrix.shape[0], tie_err_mm ** 2)


def invert(fault, insar_datas, sigma, lam=lam_xz, ramp_spec="1", surface=None, tie_depth_km=None):
    """Right-lateral slip on `fault` from the InSAR samples (and the surface constraint when
    given): Meade triangular dislocations, one transformation per dataset (ramp_spec for the
    InSAR, a constant for the surface data), the exponential prior buildCmXY(sigma, lam), the
    optional top tie, NNLS (slip >= 0, the transformation free). Fills fault.slip and each
    dataset's synth, and removes the fitted transformation from each dataset's vel IN PLACE
    (pass copies). Returns the CSI solver."""
    solve_datas = list(insar_datas)
    if surface is not None:
        solve_datas.append(surface)
    for ds in insar_datas:
        fault.buildGFs(ds, vertical=True, slipdir="s", verbose=False)
        fault.G[ds.name]["strikeslip"] *= -1.0        # right-lateral positive
    if surface is not None:
        fault.buildGFs(surface, slipdir="s", verbose=False)   # NO flip
    trans = transform("ramps", lon0=frame_lon0, lat0=frame_lat0, verbose=False)
    specs = [ramp_spec] * len(insar_datas)
    if surface is not None:
        specs.append("1")
    trans.buildGFs(solve_datas, specs, verbose=False)
    fault.assembleGFs(solve_datas, slipdir="s")
    fault.assembled(solve_datas)
    fault.assembleCd(solve_datas)
    trans.assembleGFs(solve_datas)
    trans.assembled(solve_datas)
    trans.assembleCd(solve_datas)
    fault.buildCmXY(sigma, lam, verbose=False)
    trans.buildCm(1e4, verbose=False)
    solver = multisrc("mean event", [fault, trans], verbose=False)
    solver.assembleGFs()
    solver.assembleCm()
    if tie_depth_km:                                  # uniform slip over the top tie_depth_km
        tie_matrix, tie_var = tie_rows(fault, solver.G.shape[1], tie_depth_km)
        solver.G = np.vstack([solver.G, tie_matrix])
        solver.d = np.concatenate([solver.d, np.zeros(tie_matrix.shape[0])])
        n_data = solver.Cd.shape[0]
        cd_tied = np.zeros((n_data + tie_matrix.shape[0], n_data + tie_matrix.shape[0]))
        cd_tied[:n_data, :n_data] = solver.Cd
        cd_tied[n_data:, n_data:] = np.diag(tie_var)
        solver.Cd = cd_tied
    mprior = np.zeros(solver.Np)
    for index, param_type in enumerate(solver.paramTypes):
        if param_type[0] == trans.name:
            mprior[index] = -1e3                      # NNLS: the transformation may be negative
    solver.ConstrainedLeastSquareSoln(method="nnls", mprior=mprior)
    solver.distributem()
    for ds in solve_datas:
        ds.buildsynth([fault], direction="s")
        ds.removeTransformation(trans)                # residuals are ramp-removed
    return solver

# %% [markdown]
# ### CHECK -- this notebook reproduces the draft the tests were run on
#
# PREPUB: This whole cell is a comparison with an earlier draft run; remove it before publication.
# PREPUB: 08a splits this file on the heading above, so move 08a's split to the `## [8]` heading when removing it.
# PREPUB: At sigma = 24 the slip must equal `drafts/07/fig2_model_row.py`'s
# PREPUB: product (pinned as `fig2_model_row_draft_sigma24_reference.npz`, made on
# PREPUB: the data through 2025) to round-off. The gate applies only when the
# PREPUB: quadtree samples are the draft's (same positions, data equal up to the
# PREPUB: fitted constant); on new data it is skipped and says so (and since the
# PREPUB: 2026-10-02 surface-window fix it could not pass even on the draft's data).
#
# Every solve starts from a fresh fault and a copy of the samples:
# `removeTransformation` edits `vel` in place.

# %%
## [7]
# PREPUB-START: comparison with the 2025 draft run (skipped on the paper data)
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
# PREPUB-END

# %% [markdown]
# ## [8] Sigma: the L-curve corner (Hansen 1992)
#
# One inversion per sigma on the grid (4-128). In CSI (`Fault.buildCmXY`)
# Cm = sigma^2 (lambda0^2 / (lambda_s lambda_z)) exp(-0.5 d_s / lambda_s)
# exp(-0.5 d_z / lambda_z), with lambda0 CSI's mean cell spacing: LARGER sigma
# = WEAKER smoothing (sigma scales the prior standard deviation of slip but is
# not equal to it). The L-curve plots the two terms the solver trades off: the
# full weighted misfit sqrt(chi2 InSAR + chi2 surface) against the
# prior-weighted roughness sqrt(m^T C_corr^-1 m), C_corr = Cm / sigma^2, both
# log. Corner = maximum curvature (`lcurve_corner`: cubic splines in log sigma,
# spline end intervals excluded). Selected: the smallest grid sigma at or above
# the corner, i.e. the corner leaning to under-smoothing (Dani 2026-09-28).
# Larger sigma raises the peak slip and lowers the surface fraction (the table
# printed below).
# PREPUB: The sigma loop was ported from `drafts/07/sigma_lcurve.py` and equals it on the data through 2025.

# %%
## [8]
def moment_magnitude(moment_nm):
    """Moment magnitude from the seismic moment M0 in N m: Mw = 2/3 (log10 M0 - 9.1)."""
    return (2.0 / 3.0) * (np.log10(moment_nm) - 9.1)


def potency_from_slip(area_km2, slip_mm):
    """Potency (m^3) = the sum over cells of area x slip (km^2 -> m^2, mm -> m)."""
    return float(np.sum(area_km2 * 1e6 * slip_mm * 1e-3))


def lcurve_corner(sigmas, chi2_total, roughness):
    """The L-curve corner (Hansen 1992). x = log10 sqrt(chi2_total) and y = log10 roughness as
    cubic splines of log10 sigma; curvature = |x' y'' - y' x''| / (x'^2 + y'^2)^1.5 on 400
    points between the second and the second-last grid sigma (the spline end intervals are
    excluded). Returns the sigma of maximum curvature, the 400 log10 sigmas and the curvature."""
    log_sigma = np.log10(sigmas)
    spline_misfit = CubicSpline(log_sigma, np.log10(np.sqrt(chi2_total)))
    spline_rough = CubicSpline(log_sigma, np.log10(roughness))
    fine_log_sigma = np.linspace(log_sigma[1], log_sigma[-2], 400)
    d_misfit = spline_misfit(fine_log_sigma, 1)
    dd_misfit = spline_misfit(fine_log_sigma, 2)
    d_rough = spline_rough(fine_log_sigma, 1)
    dd_rough = spline_rough(fine_log_sigma, 2)
    curvature = np.abs(d_misfit * dd_rough - d_rough * dd_misfit) / (d_misfit ** 2 + d_rough ** 2) ** 1.5
    corner = float(10 ** fine_log_sigma[np.argmax(curvature)])
    return corner, fine_log_sigma, curvature


lcurve_rows = []
for sigma in sigma_grid:
    fault = wrap_fault(mesh_nodes, mesh_faces, dip_deg, f"lcurve {sigma}")
    ds = copy.deepcopy(samples)
    surface = make_surface_interp(fault, surf_floor_mm)
    invert(fault, [ds], sigma, surface=surface, tie_depth_km=tie_depth_km)
    insar_resid = ds.vel - ds.synth
    surf_resid = surface.vel - surface.synth
    chi2_insar = float(np.dot(insar_resid, np.linalg.solve(ds.Cd, insar_resid)))
    chi2_surf = float(np.dot(surf_resid, np.linalg.solve(surface.Cd, surf_resid)))
    slip_vec = fault.slip[:, 0]
    prior_norm = float(np.sqrt(np.dot(slip_vec, np.linalg.solve(fault.Cm / sigma ** 2, slip_vec))))
    potency = potency_from_slip(cell_area_km2, slip_vec)
    surface_cells = np.array([0 in patch[:, 2] for patch in fault.patch])
    cell_km = lat_to_km_pp(np.array([pll.mean(0) for pll in fault.patchll])[:, 1])
    in_creep = surface_cells & (cell_km >= creep_lo_km) & (cell_km <= creep_hi_km)
    lcurve_rows.append(dict(sigma=sigma, chi2=chi2_insar, chi2_surf=chi2_surf,
                            chi2_total=chi2_insar + chi2_surf, prior_norm=prior_norm,
                            rms_mm=float(np.sqrt(np.mean(insar_resid ** 2))),
                            surface_pct=float(100 * np.median(slip_vec[in_creep]) / crossing_median),
                            peak_slip_mm=float(slip_vec.max()),
                            Mw=moment_magnitude(shear_modulus * potency)))
    print(f"  sigma {sigma:>4g}: chi2 {chi2_insar:.0f} + {chi2_surf:.0f}, roughness {prior_norm:.1f}, "
          f"Mw {lcurve_rows[-1]['Mw']:.2f}")
sigma_family = pd.DataFrame(lcurve_rows)
# PREPUB-START: comparison with the draft L-curve (runs only on the draft's data)
if same_data:                                          # CHECK: equal to the draft L-curve
    draft_lcurve = pd.read_csv(os.path.join(tests_dir, "sigma_lcurve.csv"))
    lcurve_diff = float(np.max(np.abs(sigma_family.chi2_total.values - draft_lcurve.chi2_total.values)))
    assert lcurve_diff < 1e-6, f"L-curve differs from the draft by {lcurve_diff:.2e}"
    print(f"PASS: L-curve equals drafts/07/sigma_lcurve.py (max |chi2 diff| {lcurve_diff:.1e})")
# PREPUB-END
print(sigma_family[["sigma", "chi2_total", "prior_norm", "rms_mm", "surface_pct",
                    "peak_slip_mm", "Mw"]].round(3).to_string(index=False))
sigma_corner, fine_log_sigma, curvature = lcurve_corner(sigma_family.sigma.values, sigma_family.chi2_total.values,
                                                        sigma_family.prior_norm.values)
sigma_sel = float(sigma_family.sigma[sigma_family.sigma >= sigma_corner].min())
print(f"L-curve corner: sigma {sigma_corner:.1f}; selected grid sigma {sigma_sel:g}")
print("(12 on the data through 2025, corner 10.1)")  # PREPUB
sigma_family["corner"] = sigma_corner                  # read by 07b_uniform_model_tests
sigma_family["selected"] = sigma_sel
sigma_family.to_csv(os.path.join(out_dir, "sigma_lcurve.csv"), index=False)

mpl_paper_style(base=7)
fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.3), constrained_layout=True)
ax = axes[0]
ax.plot(np.sqrt(sigma_family.chi2_total), sigma_family.prior_norm, "o-", color="0.4", ms=3, lw=0.8)
for sigma, x_val, y_val in zip(sigma_family.sigma, np.sqrt(sigma_family.chi2_total), sigma_family.prior_norm):
    ax.annotate(f"{sigma:g}", (x_val, y_val), xytext=(3, 2), textcoords="offset points", fontsize=6)
pick = (sigma_family.sigma == sigma_sel).values
ax.plot(np.sqrt(sigma_family.chi2_total[pick]), sigma_family.prior_norm[pick], "o", color="#D55E00", ms=5,
        label=f"selected, sigma {sigma_sel:g}")
ax.set_xscale("log")
ax.set_yscale("log")
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 3, 5, 7)))
    axis.set_major_formatter(ScalarFormatter())
    axis.set_minor_formatter(NullFormatter())
ax.set_xlabel(r"Weighted misfit $\sqrt{\chi^2}$")
ax.set_ylabel(r"Roughness $\sqrt{m^T C^{-1} m}$")
ax.legend(fontsize=6)
ax.set_title(r"$\bf{A}$  L-curve", loc="left", fontsize=7)
ax = axes[1]
ax.plot(10 ** fine_log_sigma, curvature, "k-", lw=0.8)
ax.axvline(sigma_corner, color="k", ls=":", lw=0.7)
ax.axvline(sigma_sel, color="#D55E00", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel("sigma (mm)")
ax.set_ylabel("|curvature|")
ax.set_title(r"$\bf{B}$  Curvature", loc="left", fontsize=7)
fig.savefig(os.path.join(fig_dir, "fig07_hom_lcurve.png"), dpi=300)
fig.savefig(os.path.join(fig_dir, "fig07_hom_lcurve.pdf"))
print("wrote fig07_hom_lcurve.png/.pdf")

# %% [markdown]
# ## [9] The baseline model and its numbers
#
# Mw at 30 GPa is the reference. Mw with the USGS SFCVM v26.0 rigidity,
# following Materna et al. (2024): one shear modulus for the depth that
# slips, taken from the regional velocity model. Here it is the SFCVM
# rigidity weighted by the model's own potency (each cell's median mu(z) of
# the creep-section profiles, floored at 0.5 GPa), reported as
# `mu_effective_gpa`; the slip is the uniform-rock slip, so this is a
# rescale, not a layered inversion.
# PREPUB: The creep-section profiles are made by `drafts/07/sfcvm_extract.py v26-0`.

# %%
## [9]
fault = wrap_fault(mesh_nodes, mesh_faces, dip_deg, "mean event")
ds = copy.deepcopy(samples)
surface = make_surface_interp(fault, surf_floor_mm)
solver = invert(fault, [ds], sigma_sel, surface=surface, tie_depth_km=tie_depth_km)
nodes, faces, _ = fault.mesh_sz
cell_centre_ll = np.array([pll.mean(0) for pll in fault.patchll])
cell_centre_sz = nodes[faces].mean(1)
table = pd.DataFrame(dict(
    lon=cell_centre_ll[:, 0],
    lat=cell_centre_ll[:, 1],
    s_km=cell_centre_sz[:, 0],
    z_c=cell_centre_sz[:, 1],
    z_top=nodes[faces][:, :, 1].min(1),
    z_bot=nodes[faces][:, :, 1].max(1),
    area_km2=cell_area_km2,
    slip_mm=fault.slip[:, 0]))
table["km_pp"] = lat_to_km_pp(table.lat.values)

profiles_sfcvm = pd.read_csv(sfcvm_file)
mu_by_depth = profiles_sfcvm.groupby("depth_m").mu_gpa.median()
table["mu_sfcvm_gpa"] = np.maximum(np.interp(table.z_c.values * 1000, mu_by_depth.index.values,
                                             mu_by_depth.values), mu_floor_gpa)
potency_m3 = potency_from_slip(table.area_km2, table.slip_mm)
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
    mw_30gpa=moment_magnitude(moment_30),
    mw_sfcvm=moment_magnitude(moment_sfcvm),
    mu_effective_gpa=moment_sfcvm / potency_m3 / 1e9))
print(results.to_string(float_format="{:.4g}".format))

# %% [markdown]
# ### CHECK -- the model's fault-normal motion is small
#
# The fault-parallel decomposition (06) assumes no horizontal motion
# perpendicular to the fault. Strike-slip on a curved trace, and near the
# ends of the slipping patch, produces some. The model's own fault-normal
# displacement at the samples (look vector rotated to the fault-normal
# bearing) is pushed through the same two-track solve as 06 (median LOS
# geometry of the section); the error it would put into the fault-parallel
# field must be below the data noise.

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

# the two-track solve of 06 for (fault-parallel, up), one row per track; `look` is the
# fault-parallel unit vector of [3]
leak_window_km = 1.0                                   # the leakage rms is taken within this distance of the trace
up_hat = np.array([0.0, 0.0, 1.0])
design = np.zeros((2, 2))
normal_in_los = np.zeros(2)
for row, track in enumerate(("ascending", "descending")):
    geom = np.load(os.path.join(field_dir, f"csi_mean_event_{track}.npz"))
    los_vec = utils.los_unit_vector(np.median(geom["incidence"]), np.median(geom["azimuth"]))
    design[row] = [np.dot(los_vec, look), np.dot(los_vec, up_hat)]
    normal_in_los[row] = np.dot(los_vec, look_normal)
leak_fp = np.linalg.solve(design, normal_in_los)[0]   # fault-parallel error per mm of fault-normal motion
leak_field = leak_fp * normal.synth
near_trace = dist_to_trace(ds, fault) <= leak_window_km
leak_rms_mm = float(np.sqrt(np.mean(leak_field[near_trace] ** 2)))
leak_max_mm = float(np.max(np.abs(leak_field)))
noise_sd_mm = float(np.sqrt(cov_p[0] + cov_p[1] + cov_p[3]))
print(f"leakage into the fault-parallel field: {leak_fp:+.2f} mm per mm of fault-normal motion; "
      f"max {leak_max_mm:.2f} mm, rms within 1 km {leak_rms_mm:.2f} mm (data noise sd {noise_sd_mm:.2f} mm)")
assert leak_rms_mm < noise_sd_mm, "the model's fault-normal motion biases the fault-parallel field above the noise"
print("PASS: the model's own fault-normal motion leaks into the fault-parallel field below the noise")

# %% [markdown]
# ## [11] Bootstrap -- the formal error on Mw
#
# n_boot noise realisations from the full data covariances (Cholesky)
# added to the model's own synthetics, re-inverted at the selected sigma,
# each on a fresh fault and fresh copies of the data. The spread is the
# formal error of THIS configuration; the structural range across
# configurations is in `07b_uniform_model_tests` (registry `07b_*`). The
# printed offset of the bootstrap mean from the model Mw shows whether the
# re-inverted synthetics come back biased: if they do, the sd is the scatter of
# this configuration, not an error bar centred on the model.
# PREPUB: The earlier structural range is in `07_sweep_lessons_2026-09-28.md`.

# %%
## [11]
rng = np.random.default_rng(11)
chol_insar = np.linalg.cholesky(ds.Cd + 1e-9 * np.eye(ds.Cd.shape[0]))
chol_surface = np.linalg.cholesky(surface.Cd + 1e-9 * np.eye(surface.Cd.shape[0]))
synth_insar = ds.synth.copy()
synth_surface = surface.synth.copy()
boot_mw = []
for k in range(n_boot):
    boot_fault = wrap_fault(mesh_nodes, mesh_faces, dip_deg, f"boot {k}")
    boot_ds = copy.deepcopy(samples)
    boot_surface = make_surface_interp(boot_fault, surf_floor_mm)
    boot_ds.vel[:] = synth_insar + np.dot(chol_insar, rng.standard_normal(boot_ds.vel.size))
    boot_surface.vel[:] = synth_surface + np.dot(chol_surface, rng.standard_normal(boot_surface.vel.size))
    invert(boot_fault, [boot_ds], sigma_sel, surface=boot_surface, tie_depth_km=tie_depth_km)
    boot_potency = potency_from_slip(cell_area_km2, boot_fault.slip[:, 0])
    boot_mw.append(moment_magnitude(shear_modulus * boot_potency))
    print(f"  boot {k + 1}/{n_boot}: Mw {boot_mw[-1]:.3f}")
boot_mw = np.array(boot_mw)
mw_boot_sd = float(boot_mw.std(ddof=1))
boot_offset = float(boot_mw.mean() - results.mw_30gpa)
print(f"bootstrap: Mw {boot_mw.mean():.3f} +/- {mw_boot_sd:.3f} (30 GPa, n={n_boot}); "
      f"mean minus the model's Mw {results.mw_30gpa:.3f}: {boot_offset:+.3f}")

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
    ax.set_xlim(-122.10, -121.84)
    ax.set_ylim(37.46, 37.65)
    ax.set_title(r"$\bf{" + letter + r"}$  " + title, loc="left", fontsize=7)
    if letter != "A":
        ax.set_yticklabels([])
fig.colorbar(sc, ax=axes, shrink=0.6, label="Fault-parallel displacement (mm)",
             orientation="horizontal", pad=0.04)
fig.savefig(os.path.join(fig_dir, "fig07_hom_fit_maps.png"), dpi=300)
fig.savefig(os.path.join(fig_dir, "fig07_hom_fit_maps.pdf"))
print("wrote fig07_hom_fit_maps.png/.pdf")

fig, ax = plt.subplots(figsize=(7.25, 2.4), constrained_layout=True)
pc = ax.tripcolor(nodes[:, 0], nodes[:, 1], faces, facecolors=table.slip_mm.values,
                  cmap="magma", vmin=0, vmax=max(table.slip_mm.max(), 1), edgecolors="none")
ax.set_xlim(0, arc_total)
ax.set_ylim(mesh_depth_km, 0)
ax.set_xlabel("Along-strike distance from north end (km)")
ax.set_ylabel("Depth (km)")
fig.colorbar(pc, ax=ax, label="Right-lateral slip (mm)", shrink=0.8)
ax.set_title(f"Homogeneous baseline, sigma {sigma_sel:g}: Mw {results.mw_30gpa:.2f} (30 GPa), "
             f"{results.mw_sfcvm:.2f} ({results.mu_effective_gpa:.1f} GPa, SFSVM {sfcvm_version})", loc="left", fontsize=7)
fig.savefig(os.path.join(fig_dir, "fig07_hom_slip.png"), dpi=300)
fig.savefig(os.path.join(fig_dir, "fig07_hom_slip.pdf"))
print("wrote fig07_hom_slip.png/.pdf")

# %% [markdown]
# ## [13] Products
#
# The slip table, and the model row `fig2_model_row.npz` (written to
# `outputs/07b_uniform_model_tests/`): read by `08a_3d_model` [7] (the uniform
# model on the same quadtree samples, and the crossing columns its own Fig 2
# products copy) and by `07c_fig2` (the check that 08a's products are on these
# samples). The model is also evaluated along the profile lines X-X' and Y-Y'
# of notebook 06 (fault-parallel and vertical, every 10 m, zeroed on the far
# field exactly as 06 zeroes the observed event profiles), and along strike at
# the surface next to the crossing slip.

# %%
## [13]
table.to_csv(os.path.join(out_dir, "slip_homogeneous_baseline.csv"), index=False)

# the model along 06's profile lines: the same ends, the same far-field zeroing
fig2_in = np.load(os.path.join(field_dir, "fig2_inputs.npz"))
to_ll_field = pyproj.Transformer.from_crs(int(fig2_in["epsg"]), 4326, always_xy=True)
profile_cross_m = float(fig2_in["cross_m"])
event_far_m = 1500.0                                   # 06: the event profile is zeroed on |x| >= this (06's event_far_m)
look_up = np.array([0.0, 0.0, 1.0])
model_profiles = {}
for name in [str(n) for n in fig2_in["names"]]:
    x_west, y_west, x_east, y_east = fig2_in[f"prof_fp_{name}_ends"]
    d_line = np.arange(-profile_cross_m, profile_cross_m + 1.0, 10.0)
    d_line = d_line[np.abs(d_line) > 5.0]              # the trace itself is a discontinuity
    frac = (d_line + profile_cross_m) / (2 * profile_cross_m)
    lon_line, lat_line = to_ll_field.transform(x_west + frac * (x_east - x_west), y_west + frac * (y_east - y_west))
    for component, look_vec in (("fp", look), ("up", look_up)):
        line = insar(f"profile {name} {component}", lon0=frame_lon0, lat0=frame_lat0, verbose=False)
        line.read_from_binary(np.zeros(d_line.size), np.asarray(lon_line), np.asarray(lat_line), err=np.ones(d_line.size),
                              los=np.tile(look_vec, (d_line.size, 1)), remove_nan=False, remove_zeros=False)
        fault.buildGFs(line, vertical=True, slipdir="s", verbose=False)
        fault.G[line.name]["strikeslip"] *= -1.0
        line.buildsynth([fault], direction="s")
        values = line.synth - np.median(line.synth[np.abs(d_line) >= event_far_m])
        model_profiles[f"prof_{name}_{component}_d"] = d_line
        model_profiles[f"prof_{name}_{component}_mm"] = values
    step_fp = model_profiles[f"prof_{name}_fp_mm"]
    print(f"{name}-{name}' model fault-parallel: west {step_fp[d_line < 0].min():+.2f} mm, east {step_fp[d_line > 0].max():+.2f} mm")

# the quadtree blocks as polygons, so Fig 2 can fill each block at its true size
# (equal-size dots over-weight the many small blocks at the trace). CSI keeps two
# opposite corners of every block in its local km frame (`xycorner`, one row per
# sample, in sample order); the four corners go back to lon/lat with the image's xy2ll.
block_corners_xy = np.asarray(ds.xycorner, dtype=float)
assert block_corners_xy.shape == (ds.vel.size, 4), f"xycorner {block_corners_xy.shape} does not match {ds.vel.size} samples"
corner_x = np.c_[block_corners_xy[:, 0], block_corners_xy[:, 2], block_corners_xy[:, 2], block_corners_xy[:, 0]]
corner_y = np.c_[block_corners_xy[:, 1], block_corners_xy[:, 1], block_corners_xy[:, 3], block_corners_xy[:, 3]]
block_lon, block_lat = ds.xy2ll(corner_x.ravel(), corner_y.ravel())
block_lon = np.reshape(block_lon, corner_x.shape)
block_lat = np.reshape(block_lat, corner_x.shape)
inside = ((block_lon.min(1) <= ds.lon) & (ds.lon <= block_lon.max(1)) & (block_lat.min(1) <= ds.lat) & (ds.lat <= block_lat.max(1)))
assert inside.mean() > 0.99, f"only {inside.mean():.1%} of samples fall inside their own block: corner order is off"

# along strike: the model's surface cells next to the crossing slip it was fitted to
surface_row = table[table.z_top <= 1e-6].sort_values("km_pp")
np.savez(os.path.join(tests_dir, "fig2_model_row.npz"),
         sample_lon=ds.lon, sample_lat=ds.lat, block_lon=block_lon, block_lat=block_lat, data_mm=ds.vel, model_mm=ds.synth,
         residual_mm=residual, nodes_sz=nodes, faces=faces, slip_mm=table.slip_mm.values,
         s0_km_pp=float(lat_to_km_pp([ref_fault.lati[0]])[0]),
         mw_30gpa=results.mw_30gpa, mw_sfcvm=results.mw_sfcvm, mu_effective_gpa=results.mu_effective_gpa,
         sfcvm_version=sfcvm_version, rms_mm=results.rms_mm,
         surf_ratio_pct=results.surface_pct, sigma=sigma_sel, strike_deg=strike_bearing,
         tie_depth_km=tie_depth_km,
         surface_km=surface_row.km_pp.values, surface_model_mm=surface_row.slip_mm.values,
         crossing_km=crossings_all.km.values, crossing_mm=crossings_all.mean_event_mm.values,
         crossing_profile_km=crossing_km, crossing_profile_mm=crossing_profile.vel,
         crossing_profile_err_mm=crossing_profile.err, **model_profiles)
print("wrote slip_homogeneous_baseline.csv and fig2_model_row.npz")

# %% [markdown]
# ## [14] Registry -- the homogeneous baseline (not the final model)
#
# The stats sidecar `07a_hom_baseline_stats.txt` and the `07a_hom_*` registry
# keys. The paper's selected model is the 3D model of `08a_3d_model`.

# %%
## [14]
sidecar = REG.sidecar("07a_hom_baseline", [
    f"configuration: fault-parallel (strike {strike_bearing:.2f}), coherence 0.9, levelled, "
    f"smooth12, dip {dip_deg}, quadtree {min_block_km * 1000:.0f} m, surface interp, "
    f"tie {tie_depth_km * 1000:.0f} m, sigma {sigma_sel:g}, homogeneous GFs",
    f"L-curve corner sigma {sigma_corner:.1f} -> selected grid sigma {sigma_sel:g}",
    f"draft CHECK at sigma {sigma_check}: max |slip diff| {slip_diff:.2e} mm",  # PREPUB
    f"noise model: {noise_model.round(3).to_dict()}",
    f"bootstrap Mw: {boot_mw.mean():.3f} +/- {mw_boot_sd:.3f} (n={n_boot}); mean minus the model Mw {boot_offset:+.3f}",
    f"fault-normal max {normal_max_mm:.2f} mm vs fault-parallel max {parallel_max_mm:.2f} mm; "
    f"leakage into the fault-parallel field {leak_fp:+.2f} mm/mm, max {leak_max_mm:.2f} mm, rms within 1 km {leak_rms_mm:.2f} mm "
    f"(noise sd {noise_sd_mm:.2f} mm)",
] + [f"{key}: {value:.4g}" for key, value in results.items()])
REG.register("07a_hom_mw_30gpa", results.mw_30gpa, "Mw", 2, where="M7", cell="baseline", sidecar=sidecar)
REG.register("07a_hom_mw_sfcvm", results.mw_sfcvm, "Mw", 2, where="M7", cell="baseline", sidecar=sidecar)
REG.register("07a_hom_mw_boot_sd", mw_boot_sd, "Mw", 3, where="M7", cell="bootstrap", sidecar=sidecar)
REG.register("07a_hom_potency_m3", results.potency_m3, "m3", 0, where="M7", cell="baseline", sidecar=sidecar)
REG.register("07a_hom_mu_effective_gpa", results.mu_effective_gpa, "GPa", 1, where="M7", cell="baseline", sidecar=sidecar)
REG.register("07a_hom_surface_pct", results.surface_pct, "%", 0, where="M7", cell="baseline", sidecar=sidecar)
REG.register("07a_hom_peak_slip_mm", results.peak_slip_mm, "mm", 1, where="M7", cell="baseline", sidecar=sidecar)
REG.register("07a_hom_rms_mm", results.rms_mm, "mm", 2, where="M7", cell="baseline", sidecar=sidecar)
REG.register("07a_hom_rms_near_mm", results.rms_near_mm, "mm", 2, where="M7", cell="baseline", sidecar=sidecar)
REG.register("07a_hom_sigma", sigma_sel, "mm", 0, where="M7", cell="sigma", sidecar=sidecar)
REG.register("07a_hom_sigma_corner", sigma_corner, "mm", 1, where="M7", cell="sigma", sidecar=sidecar)
REG.register("07a_hom_tie_depth_m", tie_depth_km * 1000, "m", 0, where="M7", cell="constants", sidecar=sidecar)
REG.register("07a_hom_n_samples", int(results.n_samples), "count", 0, where="M7", cell="baseline", sidecar=sidecar)
REG.register("07a_hom_fault_normal_max_mm", normal_max_mm, "mm", 2, where="M7", cell="fault-normal", sidecar=sidecar)
REG.register("07a_hom_fault_normal_leak_rms_mm", leak_rms_mm, "mm", 2, where="M7", cell="fault-normal", sidecar=sidecar)
REG.render()
print("07a_uniform_model (homogeneous baseline) complete")
