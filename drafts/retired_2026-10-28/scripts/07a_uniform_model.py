# %% [markdown]
# # 07a Uniform model -- the model data and the homogeneous half-space baseline (CSI)
#
# ```
# Created on Thu Sep 25 2026; rewritten on Mon Sep 28 2026 as the clean baseline;
# 2026-10-06: the modelling functions moved to model_utils.py, the model data product and the sample sets added
# @author: daniellelindsay
# Run in opera_disp_feb13, after 04 and 06. About 40 min: 38 inversions (13 for the L-curve, the model, 24 bootstrap)
# and three quadtree sample sets.
# ```
#
# **What this notebook does.** Two things, in this order:
#
# 1. **The data side of every slip model** (cells [1]-[6]): the five-event mean fault-parallel field of 06 (no levelling),
#    its far-field noise model, the quadtree samples and their covariance, the surface constraint, and the two
#    sensitivity sample sets (swath 7 km; the field at map coherence 0.7). Written ONCE to
#    `outputs/07a_uniform_model/model_data_07a.npz` (cell [6b]) and to the PyLith point lists in
#    `modeling_inputs/points/`; the 3D-model notebooks `08c_3d_model_sensitivity`, `08a_3d_model` and
#    `08b_3d_model_tests` read those files (no notebook runs another notebook's code).
# 2. **The uniform-rock (homogeneous half-space) model** (cells [7]-[15]): CSI with Meade triangular dislocations.
#    It is the baseline against which the paper's selected model, the 3D PyLith model in the USGS SFCVM
#    (`08a_3d_model`; labelled SFSVM in the figures), is measured. The rock is uniform throughout: Mw is quoted at
#    30 GPa only. The SFSVM rigidity enters the paper through the 3D model (08a: its per-triangle rigidity table).
# PREPUB: Until 2026-10-05 this notebook also rescaled its moment by a median SFCVM mu(z) profile
# PREPUB: (`07a_hom_mw_sfcvm`, `07a_hom_mu_effective_gpa`), read from an orphan csv made by
# PREPUB: drafts/07/sfcvm_extract.py with a superseded section (km 56.5-62.3) and strike (143.05);
# PREPUB: retired as superseded by 08a's rigidity table (Dani's modelling paragraph: 30 GPa + harmonic SFSVM).
# PREPUB: Every lesson of the 2026-09-25 to 28 tests is applied (`07_sweep_lessons_2026-09-28.md`, `drafts/07/`).
#
# | Choice | Value | Evidence |
# |---|---|---|
# | Data | fault-parallel mean event (06 `fp_maps.npz`, five events) | residual maps: removes the ascending east-side patch |
# | Decomposition strike | mean local strike of the SSE section (06 `strike`, registry `06_strike_local_deg`) | 06; Dani 2026-09-28 |
# | Coherence | 0.9 for the field | 06 `coh_maps`; Dani 2026-09-28 |
# | Levelling | none: the inversion's constant is the only datum | Dani 2026-10-08; a far-field plane (section ± 1 km masked) was removed until then, changed Mw by < 0.01, and is gone from the code |
# | Fault | vertical, on the hand-drawn trace | dip unresolved 60°E-75°W; the PyLith dip runs in 08c |
# | Mesh | `smooth12`: 50 m cells at the surface, x1.2 per row, 3 km | no resolution plateau (07b mesh family) |
# | Sampling | distance-based quadtree, 2 km to one 30 m pixel | |
# | Prior | `buildCmXY`, lambda (0.1, 0.05) km, sigma at the L-curve corner (grid value at or above it) | L-curve in [9]; lambda family in 07b |
# | Surface | crossing-stack running median interpolated to every surface triangle | 07b isolation row `insar_only` (no surface constraint) against the baseline (`07a_hom_surface_pct`) |
# | Top tie | none (dropped 2026-09-29) | at the L-curve smoothing the untied model peaks at the surface (07b tie family) |
# | Solver | NNLS (right-lateral >= 0), per-dataset constant | |
#
# PREPUB: Evidence scripts: lambda `drafts/07/along_strike_smoothing.py`; the top tie `drafts/07/tie_decision.py` (the tie was needed only at sigma >= 24).
#
# Every function used here is in `model_utils.py` (one definition each, with its source): the mesh (`unrolled_mesh`,
# `wrap_fault`), the data (`build_sample_set`), the surface constraint (`crossing_profile_on_triangles`), the
# inversion (`invert_triangles`), the L-curve (`lcurve_corner`). `07b_uniform_model_tests` (retired from the paper,
# not re-run) still carries its own older copy of this machinery. CSI calls follow the tutorials
# (`external/csi/notebooks/`).
#
# **Inputs** (made by): 06 `fp_maps.npz` (the field), `tulepond_steps_{ascending,descending}.h5` (the coherence-0.7
# field, [6]), `fig2_inputs.npz` (the profile lines, [14]) and `csi_mean_event_{ascending,descending}.npz` (the LOS
# geometry, [11]); 04 `event_fraction_by_crossing.csv`, `creep_events.csv`, `section_lengths.csv`; the trace KML,
# `hayward_fault_profiles.csv` and the temporal-coherence 0.9 masks (data_prep).
#
# **Outputs** (`outputs/07a_uniform_model/`): `model_data_07a.npz` (the data side of every model, [6b]; read by
# 08a, 08b, 08c), `sigma_lcurve.csv` (the sigma pick), `slip_homogeneous_baseline.csv`, `fig2_model_row.npz` (the
# baseline on the quadtree samples; read by 08a and 07c), the stats sidecar `07a_hom_baseline_stats.txt`; and the
# three sample sets `modeling_inputs/points/insar_points{,_swath7,_coh07}.{txt,npz}` (PyLith output points + data,
# [6]) with the coherence-0.7 field `fp_event_coh07.npz`. **Figures** (QC, not paper figures): `fig07_hom_lcurve`,
# `fig07_hom_fit_maps`, `fig07_hom_slip`. **Registry keys** `07a_hom_*`.
#
# **What to check**
# - [3] the pixel count; nothing is removed from the field, so a far-field tilt is left to the residual.
# - [4] the noise fit: a long-range sill of 0 means the noise model is a nugget plus one exponential.
# - [6] PASS: the coherence-0.9 field rebuilt from 06's saved steps equals 06's `fp_maps.npz` exactly, and the
#   three sample sets keep the positions the PyLith Green's functions were computed at.
# - [9] the L-curve corner lies inside the searched range (between the second and second-last grid
#   sigma) and the pick is the next grid sigma at or above it.
# - [10] Mw at 30 GPa; the surface % of the measured crossing slip.
# - [11] PASS: the model's own fault-normal motion leaks into the fault-parallel field below the noise.
# - [12] the bootstrap mean against the model Mw (printed): re-inverted noisy synthetics can come
#   back low, so the bootstrap sd is the scatter of this configuration, not an error bar centred on
#   the model.

# %% [markdown]
# ## [1] Paths and constants
#
# Every number the model depends on, the folders, and the registry with its inputs.

# %%
## [1]
# ------------------------
# Paths & constants
# ------------------------
import copy
import os
import time
import warnings

import numpy as np
import pandas as pd
import pyproj
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter, NullFormatter

warnings.simplefilter("ignore")
from TulePond_filepaths import (common_paths, repo_dir, paths_out, paths_inputs, insar_dic_list, model_crs,
                                model_frame_lonlat, coh_timeseries)
import insar_utils as utils
from insar_utils import load_trace, mpl_paper_style
import model_utils                                  # every modelling function, once (with its source)

# CSI: Classic Slip Inversion (Jolivet et al.; external/csi at 1277bb4)
from csi import insar

notebook = "07a_uniform_model"
out_dir = paths_out[notebook]
field_dir = paths_out["06_event_field_geometry"]
events_dir = paths_out["04_event_detection"]
draft_dir = paths_out["07b_uniform_model_tests"]     # PREPUB: the frozen draft reference of the CHECK in [8] lives there
fig_dir = common_paths["fig_dir"]
points_dir = paths_inputs["points"]
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# local frame centre (csi_io convention: the creep-section centre = the PyLith mesh CRS origin)
frame_lon0, frame_lat0 = model_frame_lonlat

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

# the two sensitivity sample sets of the 3D model (08c): a wider swath, and the field at map coherence 0.7
swath_wide_km  = 7.0      # the swath-7 set (otherwise 07a's quadtree)
coh_low        = coh_timeseries   # the coherence-0.7 set: 06's field at the fit mask (0.7) instead of coh_maps 0.9
decomposition_window_px = 20      # MintPy's asc_desc2horz_vert window: 06's geom_step (the coherence-0.9 gate in [6] proves it)

# surface constraint
section_lengths = pd.read_csv(os.path.join(events_dir, "section_lengths.csv")).set_index("name")
creep_lo_km = float(section_lengths.loc["tulepond_creep_length", "km_lo"])   # the SSE section (04)
creep_hi_km = float(section_lengths.loc["tulepond_creep_length", "km_hi"])
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
                               # model_utils.invert_triangles keeps the tie for 07b's tie family
                               # (drafts/07/tie_decision.py)  # PREPUB
shear_modulus  = 3.0e10        # uniform rigidity of the reference Mw (the SFSVM rigidity of the 3D model is in 08a)
sigma_grid     = [4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128]   # L-curve (larger = LESS smoothing)
sigma_check    = 24            # sigma of the draft the CHECK compares against  # PREPUB
n_boot         = 24

fp_file = os.path.join(field_dir, "fp_maps.npz")
steps_files = {}
for dic in insar_dic_list:
    steps_files[dic["ORBIT_DIRECTION"]] = os.path.join(field_dir, f"tulepond_steps_{dic['ORBIT_DIRECTION']}.h5")
mask09_files = {}
for dic in insar_dic_list:
    mask09_files[dic["ORBIT_DIRECTION"]] = dic["mask_file"][0.9]
REG = utils.Registry(notebook, common_paths, repo_dir)
REG.inputs(fp_file, *steps_files.values(), *mask09_files.values(),
           common_paths["fault_trace"], common_paths["fault_profiles"],
           os.path.join(events_dir, "event_fraction_by_crossing.csv"),
           os.path.join(events_dir, "creep_events.csv"),
           os.path.join(events_dir, "section_lengths.csv"),
           os.path.join(field_dir, "fig2_inputs.npz"),
           os.path.join(field_dir, "csi_mean_event_ascending.npz"),
           os.path.join(field_dir, "csi_mean_event_descending.npz"))
print(f"SSE section: km {creep_lo_km:.1f}-{creep_hi_km:.1f} (04 section_lengths.csv)")

# %% [markdown]
# ## [2] The fault mesh
#
# Unrolled (along-strike s, depth z) node rows -> Delaunay -> Laplacian
# smoothing -> wrapped onto the hand-drawn trace as a vertical
# `csi.TriangularPatches` (`tutoFaultMeshing` idiom): `model_utils.model_trace`, `unrolled_mesh`, `wrap_fault`.

# %%
## [2]
profile_points = pd.read_csv(common_paths["fault_profiles"])
profile_lats = profile_points.ft_lat.values               # km from Point Pinole = the 100 m index rule on these


def lat_to_km_pp(lats):
    """km from Point Pinole of each latitude (model_utils.km_from_pinole on the fault-profile trace)."""
    return model_utils.km_from_pinole(lats, profile_lats)


trace = model_utils.model_trace(load_trace(common_paths["fault_trace"]), mesh_lat_lo, mesh_lat_hi,
                                model_frame_lonlat, profile_lats)
trace_lonlat = trace["lonlat"]                            # north -> south
ref_fault = trace["ref_fault"]
arc_total = trace["arc_total"]
mesh_nodes, mesh_faces = model_utils.unrolled_mesh(*mesh_params, mesh_depth_km, trace, fine_km, far_rate, smooth_iters)
fault = model_utils.wrap_fault(mesh_nodes, mesh_faces, dip_deg, "mean event", trace, model_frame_lonlat)
cell_area_km2 = model_utils.triangle_areas_km2(mesh_nodes, mesh_faces)
print(f"mesh smooth12: {len(mesh_nodes)} nodes, {len(mesh_faces)} triangles, "
      f"{arc_total:.1f} km of fault, dip {dip_deg}")

# %% [markdown]
# ## [3] Data: the fault-parallel mean event
#
# Notebook 06's decomposition (coherence 0.9, SSE-section strike) as one
# `csi.insar` dataset whose look vector is the horizontal unit vector along
# the strike bearing, so right-lateral motion of the east side is positive.
# Cut to 4 km of the trace and the along-strike window (`model_utils.fault_parallel_dataset`,
# inside `build_sample_set`). Nothing is removed from the field: the inversion's per-dataset
# constant is its only datum and a far-field tilt is left in the residual (Dani 2026-10-08).
#
# ## [4] Noise model and quadtree
#
# Far-field semivariogram: nugget + short- and long-range exponentials
# (`model_utils.semivariogram_model`), fitted to pixel pairs beyond 1 km of the trace with a
# plane removed. Check the printed fit: when the long-range sill comes out 0 (its
# range then sits at a bound and means nothing) the noise model is a nugget plus
# one exponential. Distance-based quadtree (`csi.imagedownsampling`) from 2 km
# blocks to one 30 m pixel at the trace; the full data covariance of the samples
# (`model_utils.build_cd`). One call does [3] and [4]: `model_utils.build_sample_set`.

# %%
## [3]-[4]
fp = np.load(fp_file)
strike_bearing = float(fp["strike"])
sample_settings = dict(fault=fault, frame_lonlat=model_frame_lonlat, profile_lats=profile_lats,
                       exclude_km=exclude_km, km_window=km_window,
                       cov_mask_km=cov_mask_km, cov_distmax_km=cov_distmax_km, start_block_km=start_block_km,
                       min_block_km=min_block_km, nug_power=nug_power)
sample_set = model_utils.build_sample_set(fp["fp_event"], fp["lon"], fp["lat"], strike_bearing, swath_km=swath_km,
                                          **sample_settings)
sar = sample_set["sar"]
look = sample_set["look"]
cov_p = sample_set["cov_p"]
samples = sample_set["samples"]
print(f"fault-parallel field at strike {strike_bearing:.2f} deg: {sar.vel.size} pixels (nothing removed from the field)")
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
# the total weight is unchanged (`model_utils.crossing_profile_on_triangles`). This running median is the uniform
# model's surface constraint only: the 3D model (08a) ties each measured crossing
# directly (`model_utils.crossing_constraint`). The denominator of every
# surface % (here, 08a, 08b, 08c) is `crossing_median`:
# the median of the MEASURED crossings in the SSE section, from the one shared
# function `model_utils.surface_pct_denominator_mm` (Dani 2026-10-05).

# %%
## [5]
crossings_all = pd.read_csv(os.path.join(events_dir, "event_fraction_by_crossing.csv"))
creep_events = pd.read_csv(os.path.join(events_dir, "creep_events.csv"))
tulepond_onsets = creep_events[creep_events.segment == "TULEPOND"].onset_lo.values    # the step epoch (last pre-event acquisition)
event_cols = [f"amp_{onset:.3f}" for onset in tulepond_onsets]    # 04's column naming
print(f"mean event of {len(event_cols)} Tule Pond events: {', '.join(event_cols)}")
crossings_all["mean_event_mm"] = crossings_all[event_cols].mean(axis=1)
trace_south_north = trace_lonlat[np.argsort(trace_lonlat[:, 1])]
surface_settings = dict(crossings=crossings_all, floor_mm=surf_floor_mm, step_km=surf_step_km, win_km=surf_win_km,
                        km_window=km_window, trace_south_north=trace_south_north, frame_lonlat=model_frame_lonlat)


def make_surface_interp(fault_model):
    """The uniform model's surface constraint on this fault (model_utils.crossing_profile_on_triangles)."""
    return model_utils.crossing_profile_on_triangles(fault_model, profile_lats=profile_lats, **surface_settings)


crossing_profile = model_utils.crossing_profile(**surface_settings)
crossing_km = lat_to_km_pp(crossing_profile.lat)
crossing_median = model_utils.surface_pct_denominator_mm(crossings_all.km.values, crossings_all.mean_event_mm.values,
                                                         creep_lo_km, creep_hi_km)   # median of the measured crossings in the SSE section
print(f"surface constraint: {crossing_profile.vel.size} profile points; surface % denominator: median of the "
      f"measured crossings in the SSE section {crossing_median:.2f} mm")

# %% [markdown]
# ## [6] The sample sets of the 3D models (PyLith output points)
#
# The 3D models invert the same data on PyLith Green's functions computed AT the quadtree samples, so every sample
# set is written as a PyLith point list (`modeling_inputs/points/insar_points<tag>.txt`, "name x y z" in the mesh
# CRS) and an npz carrying the data, the block pixel counts, the block corners and the noise model
# (`model_utils.write_points_list`). Three sets: the paper's (above), and the two sensitivity sets of 08c, each built
# by the same `model_utils.build_sample_set`:
# - **swath 7 km**: the same field, pixels within 7 km of the trace (its own variogram);
# - **coherence 0.7**: 06's field at map coherence 0.7 (the time-series fit mask) instead of 0.9, swath 4 km. 06
#   saves the per-track mean event at the fit mask with its geometry (`tulepond_steps_<track>.h5`), so the field is
#   re-decomposed here (`model_utils.fault_parallel_from_steps`, MintPy's `asc_desc2horz_vert` as in 06). GATE: the
#   same call with 06's coherence-0.9 masks reproduces 06's `fp_maps.npz` field exactly (NaN pattern and values).
#
# CHECK: if a set exists already, its sample positions must be unchanged (the cached Green's functions were computed
# at them; asserted by `write_points_list`). A set with new positions needs new PyLith runs (pylith/README.md).

# %%
## [6]
t0 = time.time()
fp_event_coh09 = model_utils.fault_parallel_from_steps(steps_files, strike_bearing, decomposition_window_px, mask09_files)
assert np.array_equal(np.isfinite(fp_event_coh09), np.isfinite(fp["fp_event"])), "the rebuilt 0.9 field has another NaN pattern than 06's"
gate_diff = float(np.nanmax(np.abs(fp_event_coh09 - fp["fp_event"])))
assert gate_diff == 0.0, f"the coherence-0.9 field rebuilt from 06's saved steps differs from 06's fp_maps.npz by {gate_diff} mm"
print(f"PASS: 06's fault-parallel field rebuilt from its saved steps at coherence 0.9 equals fp_maps.npz exactly "
      f"({np.isfinite(fp_event_coh09).sum()} pixels)")
fp_event_coh07 = model_utils.fault_parallel_from_steps(steps_files, strike_bearing, decomposition_window_px)
np.savez(os.path.join(points_dir, "fp_event_coh07.npz"), fp_event=fp_event_coh07, strike=strike_bearing, coherence=coh_low,
         provenance=f"07a_uniform_model {time.strftime('%Y-%m-%d %H:%M')}: 06's mean event at the fit mask (coherence {coh_low}), "
                    f"decomposed with asc_desc2horz_vert (window {decomposition_window_px} px); grid = 06 fp_maps.npz lon/lat")
print(f"coherence-{coh_low} field: {np.isfinite(fp_event_coh07).sum()} pixels (0.9: {np.isfinite(fp['fp_event']).sum()}); "
      f"{time.time() - t0:.0f} s")

field_fingerprint = model_utils.field_fingerprint(fp_file)        # the 06 field every model fits
sample_sets = {"default": dict(set=sample_set, swath_km=swath_km, field="06 fp_maps.npz (coherence 0.9)")}
sample_sets["swath7"] = dict(set=model_utils.build_sample_set(fp["fp_event"], fp["lon"], fp["lat"], strike_bearing,
                                                              swath_km=swath_wide_km, **sample_settings),
                             swath_km=swath_wide_km, field="06 fp_maps.npz (coherence 0.9)")
sample_sets["coh07"] = dict(set=model_utils.build_sample_set(fp_event_coh07, fp["lon"], fp["lat"], strike_bearing,
                                                             swath_km=swath_km, **sample_settings),
                            swath_km=swath_km, field=f"06 mean event at coherence {coh_low} (fp_event_coh07.npz)")
points_files = {}
for tag, entry in sample_sets.items():
    one_set = entry["set"]
    ds = one_set["samples"]
    stem = os.path.join(points_dir, "insar_points" + ("" if tag == "default" else f"_{tag}"))
    model_utils.write_points_list(
        stem, ds.lon, ds.lat, model_crs,
        header=f"quadtree samples of 07a_uniform_model, set {tag} ({ds.vel.size} points of {one_set['n_pixels']} pixels, "
               f"swath {entry['swath_km']:g} km, {entry['field']}), written {time.strftime('%Y-%m-%d %H:%M')}",
        extra=dict(x_km=np.asarray(ds.x, float), y_km=np.asarray(ds.y, float), vel_mm=np.asarray(ds.vel, float),
                   wgt=np.asarray(ds.wgt, float), block_lon=one_set["block_lon"], block_lat=one_set["block_lat"],
                   cov_p=np.asarray(one_set["cov_p"], float), nug_power=nug_power, swath_km=entry["swath_km"],
                   n_pixels=one_set["n_pixels"], strike_deg=strike_bearing, field=entry["field"],
                   field_fingerprint=field_fingerprint,
                   provenance=f"07a_uniform_model {time.strftime('%Y-%m-%d %H:%M')}: model_utils.build_sample_set, set {tag}"))
    points_files[tag] = stem
    print(f"set {tag:7s}: {one_set['n_pixels']} pixels -> {ds.vel.size} samples; noise "
          f"{np.round(np.sqrt(np.asarray(one_set['cov_p'])[[0, 1, 3]]), 3)} mm, ranges {np.round(np.asarray(one_set['cov_p'])[[2, 4]], 2)} km")

# %% [markdown]
# ## [6b] The model data product
#
# Everything the 3D models take from this notebook, in ONE file (`model_data_07a.npz`): the samples (positions in
# lon/lat and in CSI's local km frame, data, block pixel counts, block corners), the noise model and the covariance
# exponent (the full Cd is rebuilt from them by `model_utils.build_cd`), the strike and look vector, the measured
# crossings of the surface constraint and the surface % denominator, the SSE section, the prior and sigma grid, the
# uniform model's mesh and trace (for the CSI validation of the PyLith Green's functions in 08a), the bootstrap size,
# and the fingerprint of the 06 field.

# %%
## [6b]
model_data_file = os.path.join(out_dir, "model_data_07a.npz")
np.savez(model_data_file,
         sample_lon=samples.lon, sample_lat=samples.lat, sample_x_km=samples.x, sample_y_km=samples.y,
         data_mm=samples.vel, wgt=samples.wgt, block_lon=sample_set["block_lon"], block_lat=sample_set["block_lat"],
         cov_p=cov_p, nug_power=nug_power, strike_deg=strike_bearing, look=look, frame_lonlat=np.array(model_frame_lonlat),
         crs=model_crs, crossing_km=crossings_all.km.values, crossing_lat=crossings_all.lat.values,
         crossing_mean_mm=crossings_all.mean_event_mm.values, crossing_median_mm=crossing_median,
         creep_lo_km=creep_lo_km, creep_hi_km=creep_hi_km, km_window=np.array(km_window), surf_floor_mm=surf_floor_mm,
         lam_xz=np.array(lam_xz), sigma_grid=np.array(sigma_grid), n_boot=n_boot,
         trace_lonlat=trace_lonlat, trace_every_km=trace["every_km"], mesh_nodes_sz=mesh_nodes, mesh_faces=mesh_faces,
         mesh_dip_deg=dip_deg, mesh_lat_range=np.array([mesh_lat_lo, mesh_lat_hi]),
         points_default=points_files["default"], points_swath7=points_files["swath7"], points_coh07=points_files["coh07"],
         field_fingerprint=field_fingerprint,
         provenance=f"07a_uniform_model {time.strftime('%Y-%m-%d %H:%M')}; {utils.provenance_stamp(notebook, repo_dir)}; "
                    f"06 field {field_fingerprint}")
print(f"wrote {model_data_file}: {samples.vel.size} samples, strike {strike_bearing:.4f}, field {field_fingerprint}")

# %% [markdown]
# ## [7] The inversion
#
# `model_utils.invert_triangles`: Meade triangular dislocations (right-lateral positive) for the samples;
# for the surface data the Green's function is 1 on the triangle's own
# slip. Per-dataset constant (`csi.transformation`), exponential prior
# (`buildCmXY`), NNLS (`multisourcesolve`). The optional top tie (every cell
# shallower than tie_depth_km equal to the shallowest cell of its 0.1 km
# along-strike column; after Jolivet et al. 2023, who tie 0-1 km) is off in
# the uniform model (tie_depth_km = 0) and used only by 07b's tie family.
# Every solve starts from a fresh fault and a copy of the samples:
# `removeTransformation` edits `vel` in place.

# %%
## [7]
def solve_uniform(name, sigma, insar_data=None, surface=None):
    """One uniform-model solve on a fresh fault: returns (fault, InSAR data, surface data, solver)."""
    fault_model = model_utils.wrap_fault(mesh_nodes, mesh_faces, dip_deg, name, trace, model_frame_lonlat)
    if insar_data is None:
        insar_data = copy.deepcopy(samples)
    if surface is None:
        surface = make_surface_interp(fault_model)
    solver = model_utils.invert_triangles(fault_model, [insar_data], sigma, lam_xz, model_frame_lonlat,
                                          surface=surface, tie_depth_km=tie_depth_km)
    return fault_model, insar_data, surface, solver

# %% [markdown]
# ### CHECK -- this notebook reproduces the draft the tests were run on
#
# PREPUB: This whole cell is a comparison with an earlier draft run; remove it before publication.
# PREPUB: At sigma = 24 the slip must equal `drafts/07/fig2_model_row.py`'s
# PREPUB: product (pinned as `fig2_model_row_draft_sigma24_reference.npz`, made on
# PREPUB: the data through 2025) to round-off. The gate applies only when the
# PREPUB: quadtree samples are the draft's (same positions, data equal up to the
# PREPUB: fitted constant); on new data it is skipped and says so (and since the
# PREPUB: 2026-10-02 surface-window fix it could not pass even on the draft's data).

# %%
## [8]
# PREPUB-START: comparison with the 2025 draft run (skipped on the paper data)
draft_file = os.path.join(draft_dir, "fig2_model_row_draft_sigma24_reference.npz")
same_data = False
if os.path.exists(draft_file):
    draft = np.load(draft_file)
    same_data = (draft["sample_lon"].size == samples.lon.size
                 and np.array_equal(draft["sample_lon"], samples.lon)
                 and np.array_equal(draft["sample_lat"], samples.lat)
                 and np.ptp(samples.vel - draft["data_mm"]) < 1e-9)
if same_data:
    fault, ds, surface, _ = solve_uniform("mean event", sigma_check)
    slip_diff = float(np.max(np.abs(fault.slip[:, 0] - draft["slip_mm"])))
    print(f"sigma {sigma_check}: max |slip - draft| = {slip_diff:.2e} mm over {len(mesh_faces)} cells")
    assert slip_diff < 1e-6, "the clean notebook does not reproduce the tested draft"
    print("PASS: the clean notebook reproduces the draft model")
else:
    slip_diff = np.nan
    print("SKIPPED: the input data differ from the draft's (new data); the gate does not apply")
# PREPUB-END

# %% [markdown]
# ## [9] Sigma: the L-curve corner (Hansen 1992)
#
# One inversion per sigma on the grid (4-128). In CSI (`Fault.buildCmXY`)
# Cm = sigma^2 (lambda0^2 / (lambda_s lambda_z)) exp(-0.5 d_s / lambda_s)
# exp(-0.5 d_z / lambda_z), with lambda0 CSI's mean cell spacing: LARGER sigma
# = WEAKER smoothing (sigma scales the prior standard deviation of slip but is
# not equal to it). The L-curve plots the two terms the solver trades off: the
# full weighted misfit sqrt(chi2 InSAR + chi2 surface) against the
# prior-weighted roughness sqrt(m^T C_corr^-1 m), C_corr = Cm / sigma^2, both
# log. Corner = maximum curvature (`model_utils.lcurve_corner`: cubic splines in log sigma,
# spline end intervals excluded). Selected: the smallest grid sigma at or above
# the corner, i.e. the corner leaning to under-smoothing (Dani 2026-09-28; `model_utils.lcurve_pick`).
# Larger sigma raises the peak slip and lowers the surface fraction (the table
# printed below).
# PREPUB: The sigma loop was ported from `drafts/07/sigma_lcurve.py` and equals it on the data through 2025.

# %%
## [9]
lcurve_rows = []
for sigma in sigma_grid:
    fault, ds, surface, _ = solve_uniform(f"lcurve {sigma}", sigma)
    insar_resid = ds.vel - ds.synth
    surf_resid = surface.vel - surface.synth
    chi2_insar = float(np.dot(insar_resid, np.linalg.solve(ds.Cd, insar_resid)))
    chi2_surf = float(np.dot(surf_resid, np.linalg.solve(surface.Cd, surf_resid)))
    slip_vec = fault.slip[:, 0]
    prior_norm = float(np.sqrt(np.dot(slip_vec, np.linalg.solve(fault.Cm / sigma ** 2, slip_vec))))
    potency = model_utils.potency_from_slip(cell_area_km2, slip_vec)
    surface_cells = np.array([0 in patch[:, 2] for patch in fault.patch])
    cell_km = lat_to_km_pp(np.array([pll.mean(0) for pll in fault.patchll])[:, 1])
    in_creep = surface_cells & (cell_km >= creep_lo_km) & (cell_km <= creep_hi_km)
    lcurve_rows.append(dict(sigma=sigma, chi2=chi2_insar, chi2_surf=chi2_surf,
                            chi2_total=chi2_insar + chi2_surf, prior_norm=prior_norm,
                            rms_mm=float(np.sqrt(np.mean(insar_resid ** 2))),
                            surface_pct=float(100 * np.median(slip_vec[in_creep]) / crossing_median),
                            peak_slip_mm=float(slip_vec.max()),
                            Mw=model_utils.moment_magnitude(shear_modulus * potency)))
    print(f"  sigma {sigma:>4g}: chi2 {chi2_insar:.0f} + {chi2_surf:.0f}, roughness {prior_norm:.1f}, "
          f"Mw {lcurve_rows[-1]['Mw']:.2f}")
sigma_family = pd.DataFrame(lcurve_rows)
# PREPUB-START: comparison with the draft L-curve (runs only on the draft's data)
if same_data:                                          # CHECK: equal to the draft L-curve
    draft_lcurve = pd.read_csv(os.path.join(draft_dir, "sigma_lcurve.csv"))
    lcurve_diff = float(np.max(np.abs(sigma_family.chi2_total.values - draft_lcurve.chi2_total.values)))
    assert lcurve_diff < 1e-6, f"L-curve differs from the draft by {lcurve_diff:.2e}"
    print(f"PASS: L-curve equals drafts/07/sigma_lcurve.py (max |chi2 diff| {lcurve_diff:.1e})")
# PREPUB-END
print(sigma_family[["sigma", "chi2_total", "prior_norm", "rms_mm", "surface_pct",
                    "peak_slip_mm", "Mw"]].round(3).to_string(index=False))
sigma_corner, fine_log_sigma, curvature = model_utils.lcurve_corner(sigma_family.sigma.values,
                                                                    np.sqrt(sigma_family.chi2_total.values),
                                                                    sigma_family.prior_norm.values)
sigma_sel = model_utils.lcurve_pick(sigma_family.sigma.values, sigma_corner)
print(f"L-curve corner: sigma {sigma_corner:.1f}; selected grid sigma {sigma_sel:g}")
print("(12 on the data through 2025, corner 10.1)")  # PREPUB
sigma_family["corner"] = sigma_corner
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
# ## [10] The baseline model and its numbers
#
# Mw at 30 GPa (the uniform rock of these Green's functions); surface % =
# the median slip of the surface cells in the SSE section over the median of
# the measured crossings there ([5]). The SFSVM rigidity is applied to the 3D
# model only (08a [15c]).

# %%
## [10]
fault, ds, surface, solver = solve_uniform("mean event", sigma_sel)
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

potency_m3 = model_utils.potency_from_slip(table.area_km2, table.slip_mm)
moment_30 = shear_modulus * potency_m3

residual = ds.vel - ds.synth
near = model_utils.dist_to_trace(ds, fault) <= near_field_km
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
    mw_30gpa=model_utils.moment_magnitude(moment_30)))
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
## [11]
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
near_trace = model_utils.dist_to_trace(ds, fault) <= leak_window_km
leak_rms_mm = float(np.sqrt(np.mean(leak_field[near_trace] ** 2)))
leak_max_mm = float(np.max(np.abs(leak_field)))
noise_sd_mm = float(np.sqrt(cov_p[0] + cov_p[1] + cov_p[3]))
print(f"leakage into the fault-parallel field: {leak_fp:+.2f} mm per mm of fault-normal motion; "
      f"max {leak_max_mm:.2f} mm, rms within 1 km {leak_rms_mm:.2f} mm (data noise sd {noise_sd_mm:.2f} mm)")
assert leak_rms_mm < noise_sd_mm, "the model's fault-normal motion biases the fault-parallel field above the noise"
print("PASS: the model's own fault-normal motion leaks into the fault-parallel field below the noise")

# %% [markdown]
# ## [12] Bootstrap -- the formal error on Mw
#
# n_boot noise realisations from the full data covariances (Cholesky)
# added to the model's own synthetics, re-inverted at the selected sigma,
# each on a fresh fault and fresh copies of the data. The spread is the
# formal error of THIS configuration. The
# printed offset of the bootstrap mean from the model Mw shows whether the
# re-inverted synthetics come back biased: if they do, the sd is the scatter of
# this configuration, not an error bar centred on the model.
# PREPUB: The earlier structural range is in `07_sweep_lessons_2026-09-28.md`.

# %%
## [12]
rng = np.random.default_rng(11)
chol_insar = np.linalg.cholesky(ds.Cd + 1e-9 * np.eye(ds.Cd.shape[0]))
chol_surface = np.linalg.cholesky(surface.Cd + 1e-9 * np.eye(surface.Cd.shape[0]))
synth_insar = ds.synth.copy()
synth_surface = surface.synth.copy()
boot_mw = []
for k in range(n_boot):
    boot_fault = model_utils.wrap_fault(mesh_nodes, mesh_faces, dip_deg, f"boot {k}", trace, model_frame_lonlat)
    boot_ds = copy.deepcopy(samples)
    boot_surface = make_surface_interp(boot_fault)
    boot_ds.vel[:] = synth_insar + np.dot(chol_insar, rng.standard_normal(boot_ds.vel.size))
    boot_surface.vel[:] = synth_surface + np.dot(chol_surface, rng.standard_normal(boot_surface.vel.size))
    model_utils.invert_triangles(boot_fault, [boot_ds], sigma_sel, lam_xz, model_frame_lonlat, surface=boot_surface,
                                 tie_depth_km=tie_depth_km)
    boot_potency = model_utils.potency_from_slip(cell_area_km2, boot_fault.slip[:, 0])
    boot_mw.append(model_utils.moment_magnitude(shear_modulus * boot_potency))
    print(f"  boot {k + 1}/{n_boot}: Mw {boot_mw[-1]:.3f}")
boot_mw = np.array(boot_mw)
mw_boot_sd = float(boot_mw.std(ddof=1))
boot_offset = float(boot_mw.mean() - results.mw_30gpa)
print(f"bootstrap: Mw {boot_mw.mean():.3f} +/- {mw_boot_sd:.3f} (30 GPa, n={n_boot}); "
      f"mean minus the model's Mw {results.mw_30gpa:.3f}: {boot_offset:+.3f}")

# %% [markdown]
# ## [13] Figures
#
# A-C: fault-parallel data, model and residual at the quadtree samples.
# Then the slip distribution (single panel).

# %%
## [13]
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
ax.set_title(f"Homogeneous baseline, sigma {sigma_sel:g}: Mw {results.mw_30gpa:.2f} (30 GPa)", loc="left", fontsize=7)
fig.savefig(os.path.join(fig_dir, "fig07_hom_slip.png"), dpi=300)
fig.savefig(os.path.join(fig_dir, "fig07_hom_slip.pdf"))
print("wrote fig07_hom_slip.png/.pdf")

# %% [markdown]
# ## [14] Products
#
# The slip table, and the model row `fig2_model_row.npz` (in this notebook's output folder): read by
# `08a_3d_model` (the uniform model on the same quadtree samples, and the crossing columns its own Fig 2
# products copy) and by `07c_fig2` (the check that 08a's products are on these
# samples). The model is also evaluated along the profile lines X-X' and Y-Y'
# of notebook 06 (fault-parallel and vertical, every 10 m, zeroed on the far
# field exactly as 06 zeroes the observed event profiles), and along strike at
# the surface next to the crossing slip.
# PREPUB: Until 2026-10-06 fig2_model_row.npz was written to outputs/07b_uniform_model_tests/.

# %%
## [14]
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

# the quadtree blocks as polygons, so Fig 2 can fill each block at its true size (model_utils.quadtree_block_corners)
block_lon, block_lat = model_utils.quadtree_block_corners(ds)

# along strike: the model's surface cells next to the crossing slip it was fitted to
surface_row = table[table.z_top <= 1e-6].sort_values("km_pp")
np.savez(os.path.join(out_dir, "fig2_model_row.npz"),
         sample_lon=ds.lon, sample_lat=ds.lat, block_lon=block_lon, block_lat=block_lat, data_mm=ds.vel, model_mm=ds.synth,
         residual_mm=residual, nodes_sz=nodes, faces=faces, slip_mm=table.slip_mm.values,
         s0_km_pp=float(lat_to_km_pp([ref_fault.lati[0]])[0]),
         mw_30gpa=results.mw_30gpa, rms_mm=results.rms_mm,
         surf_ratio_pct=results.surface_pct, sigma=sigma_sel, strike_deg=strike_bearing,
         tie_depth_km=tie_depth_km,
         surface_km=surface_row.km_pp.values, surface_model_mm=surface_row.slip_mm.values,
         crossing_km=crossings_all.km.values, crossing_mm=crossings_all.mean_event_mm.values,
         crossing_profile_km=crossing_km, crossing_profile_mm=crossing_profile.vel,
         crossing_profile_err_mm=crossing_profile.err, **model_profiles)
print("wrote slip_homogeneous_baseline.csv and fig2_model_row.npz")

# %% [markdown]
# ## [15] Registry -- the homogeneous baseline (not the final model)
#
# The stats sidecar `07a_hom_baseline_stats.txt` and the `07a_hom_*` registry
# keys. The paper's selected model is the 3D model of `08a_3d_model`.

# %%
## [15]
sidecar = REG.sidecar("07a_hom_baseline", [
    f"configuration: fault-parallel (strike {strike_bearing:.2f}), coherence 0.9, no levelling, "
    f"smooth12, dip {dip_deg}, quadtree {min_block_km * 1000:.0f} m, surface interp, "
    f"tie {tie_depth_km * 1000:.0f} m, sigma {sigma_sel:g}, homogeneous GFs",
    f"L-curve corner sigma {sigma_corner:.1f} -> selected grid sigma {sigma_sel:g}",
    f"draft CHECK at sigma {sigma_check}: max |slip diff| {slip_diff:.2e} mm",  # PREPUB
    f"noise model: {noise_model.round(3).to_dict()}",
    f"bootstrap Mw: {boot_mw.mean():.3f} +/- {mw_boot_sd:.3f} (n={n_boot}); mean minus the model Mw {boot_offset:+.3f}",
    f"fault-normal max {normal_max_mm:.2f} mm vs fault-parallel max {parallel_max_mm:.2f} mm; "
    f"leakage into the fault-parallel field {leak_fp:+.2f} mm/mm, max {leak_max_mm:.2f} mm, rms within 1 km {leak_rms_mm:.2f} mm "
    f"(noise sd {noise_sd_mm:.2f} mm)",
    f"model data product: {model_data_file} (06 field {field_fingerprint}); sample sets "
    + ", ".join(f"{tag} {entry['set']['samples'].vel.size}" for tag, entry in sample_sets.items()),
] + [f"{key}: {value:.4g}" for key, value in results.items()])
REG.register("07a_hom_mw_30gpa", results.mw_30gpa, "Mw", 2, where="M7", cell="baseline", sidecar=sidecar)
REG.register("07a_hom_mw_boot_sd", mw_boot_sd, "Mw", 3, where="M7", cell="bootstrap", sidecar=sidecar)
REG.register("07a_hom_potency_m3", results.potency_m3, "m3", 0, where="M7", cell="baseline", sidecar=sidecar)
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
