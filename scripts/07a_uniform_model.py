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
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 07a The data of the slip models, and the best homogeneous model
#
# ```
# Rewritten from scratch on Fri Oct 30 2026 (the rewire, code_audit_2026-10-28/REWIRE_BRIEF.md); the earlier 07a is in
# drafts/retired_2026-10-28/.
# @author: daniellelindsay
# Run in opera_disp_feb13, after 04 and 06. About 15 min with the analytic Green's functions cached (about 2 h without).
# ```
#
# **What this notebook does.** Two things, in this order.
#
# 1. **The data side of every slip model** (cells [2]-[6]): notebook 06's five-event mean fault-parallel field, its
#    quadtree samples (the PyLith output points: their positions must not change), the far-field noise model
#    (semivariogram beyond 5 km of the trace), and the coherence-0.7 sample set of the sensitivity row. Written once as
#    products that 08a, 08b and `scripts/sensitivity/` read.
# 2. **The best homogeneous model** (cells [7]-[11]): the same inversion as the selected 3D model of 08a, on the same
#    fault vertices, with Green's functions of a homogeneous elastic half-space (CSI triangular dislocations, Meade 2007,
#    register #13). It is the comparison line of Fig 2 and of 08a.
#
# Every setting comes from `model_config.py` (the one parameter block; register rows in brackets). Every modelling
# function is in `model_utils.py`.
#
# **Inputs**: 06 `fp_maps.npz` (the field), `tulepond_steps_{ascending,descending}.h5` (the coherence-0.7 field);
# 04 `section_lengths.csv`; the trace KML and `hayward_fault_profiles.csv`; the temporal-coherence 0.9 masks; the
# archived G of the selected PyLith run (its fault vertices).
#
# **Outputs** (`outputs/07a_uniform_model/`): `model_data.npz` (the samples, the noise model, the trace),
# `noise_model_5km.csv` and `semivariogram_bins_5km.csv` (the noise model as a product), `homogeneous_lcurve.csv`,
# `homogeneous_model.npz` (slip, prediction, plane at the samples), `07a_homogeneous_stats.txt`; the point lists
# `modeling_inputs/points/insar_points{,_coh07}.{txt,npz}` (positions asserted unchanged); the analytic Green's
# functions `modeling_uniform/gf/G_analytic_<run>_nsub<n>.npz` (cache). **Registry keys** `07a_*`.
#
# **What to check**
# - [3] PASS: the quadtree samples are the PyLith output points (same positions, same data).
# - [4] the noise fit: nugget ~0.90 mm, one exponential ~0.89 mm / 1.7 km, the second term at 0 (its range on a bound).
# - [7] the n_sub check: the largest relative difference of the 20 test tents decides n_sub.
# - [9] the corner lies inside the grid; the refined corner is near the base corner.
# - [10] the RMS of the fitted samples, the near-field RMS west / east, Mw (30 GPa), depth50 / depth90.

# %% [markdown]
# ## [1] Imports, the locked settings, the registry
#
# The settings are printed from `model_config.summary()`, so the run record shows what was used.

# %%
## [1]
import copy
import os
import time
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyproj

warnings.filterwarnings("ignore", category=DeprecationWarning)   # library noise only; numerical warnings stay visible
warnings.filterwarnings("ignore", category=FutureWarning)
from TulePond_filepaths import (common_paths, repo_dir, paths_out, paths_inputs, insar_dic_list, model_crs,
                                model_frame_lonlat, coh_timeseries)
import insar_utils as utils
from insar_utils import load_trace
import model_config as cfg                          # THE parameter block of the slip models
import model_utils                                  # every modelling function, once (with its source)

notebook = "07a_uniform_model"
out_dir = paths_out[notebook]
field_dir = paths_out["06_event_field_geometry"]
points_dir = paths_inputs["points"]
os.makedirs(out_dir, exist_ok=True)
made = time.strftime("%Y-%m-%d %H:%M")

fp_file = os.path.join(field_dir, "fp_maps.npz")
steps_files = {}
mask09_files = {}
for dic in insar_dic_list:
    steps_files[dic["ORBIT_DIRECTION"]] = os.path.join(field_dir, f"tulepond_steps_{dic['ORBIT_DIRECTION']}.h5")
    mask09_files[dic["ORBIT_DIRECTION"]] = dic["mask_file"][0.9]
creep_lo_km, creep_hi_km = cfg.sse_section_km()
REG = utils.Registry(notebook, common_paths, repo_dir)
REG.inputs(fp_file, *steps_files.values(), *mask09_files.values(), common_paths["fault_trace"], common_paths["fault_profiles"],
           os.path.join(paths_out["04_event_detection"], "section_lengths.csv"), cfg.gf_files["G"])
print("\n".join(cfg.summary()))

# %% [markdown]
# ## [2] The mapped trace
#
# The hand-drawn trace (KML) between the latitudes of the model region, as a CSI trace discretised every 20 m
# (`model_utils.model_trace`). Its points give every pixel's distance to the fault (the quadtree refines toward it and
# the noise model masks around it) and every sample's signed distance and along-strike km.

# %%
## [2]
profile_lats = pd.read_csv(common_paths["fault_profiles"]).ft_lat.values      # km from Point Pinole = the 100 m index rule on these
trace = model_utils.model_trace(load_trace(common_paths["fault_trace"]), *cfg.fault_trace_lat_range, model_frame_lonlat, profile_lats)
trace_fault = trace["ref_fault"]                                              # CSI object: xf, yf = the trace points
trace_xy_m = (trace["x_km"] * 1e3, trace["y_km"] * 1e3)                       # mesh CRS = CSI frame (same origin), m
print(f"trace: {trace['lonlat'].shape[0]} KML vertices, {trace['arc_total']:.1f} km, discretised every "
      f"{trace['every_km'] * 1e3:.0f} m; SSE section km {creep_lo_km:.2f}-{creep_hi_km:.2f}")

# %% [markdown]
# ## [3] The data: the fault-parallel mean event and its quadtree samples
#
# Notebook 06's decomposition (coherence 0.9, the SSE-section strike) as one `csi.insar` data set whose look vector is
# the horizontal unit vector along strike, so right-lateral motion of the east side is positive
# (`model_utils.fault_parallel_dataset`): pixels within 4 km of the trace (#3), the pixel straddling the trace dropped,
# km 52-67 along strike. Distance-based quadtree from 2 km blocks to one 30 m pixel at the trace (#4,
# `model_utils.downsample`, CSI `imagedownsampling`).
#
# CHECK: the samples are the PyLith output points. Every Green's function (PyLith and analytic) was computed at these
# positions, so they must be the positions (and data) of `insar_points.npz` exactly.

# %%
## [3]
fp = np.load(fp_file)
strike_deg = float(fp["strike"])
field_fingerprint = model_utils.field_fingerprint(fp_file)
sar, look = model_utils.fault_parallel_dataset(fp["fp_event"], fp["lon"], fp["lat"], strike_deg, trace_fault, model_frame_lonlat,
                                               cfg.swath_km, cfg.exclude_km, cfg.km_window, profile_lats)
quadtree = model_utils.downsample(sar, trace_fault, cfg.start_block_km, cfg.min_block_km)
block_lon, block_lat = model_utils.quadtree_block_corners(quadtree)
n_pixels = int(np.isfinite(sar.vel).sum())
points_old = np.load(os.path.join(points_dir, "insar_points.npz"))
assert quadtree.vel.size == points_old["lon"].size, f"{quadtree.vel.size} samples, the PyLith points hold {points_old['lon'].size}"
for name, new, old in (("lon", quadtree.lon, points_old["lon"]), ("lat", quadtree.lat, points_old["lat"]),
                       ("x_km", quadtree.x, points_old["x_km"]), ("y_km", quadtree.y, points_old["y_km"]),
                       ("data", quadtree.vel, points_old["vel_mm"]), ("block pixels", quadtree.wgt, points_old["wgt"])):
    assert np.array_equal(np.asarray(new, float), np.asarray(old, float)), f"the quadtree {name} differ from insar_points.npz"
positions_key = model_utils.positions_fingerprint(quadtree.lon, quadtree.lat)
print(f"fault-parallel field at strike {strike_deg:.2f} deg: {n_pixels:,} pixels -> {quadtree.vel.size} samples")
print(f"PASS: the samples are the PyLith output points (positions, data and block sizes identical; key {positions_key})")

# %% [markdown]
# ## [4] The noise model: semivariogram of the far field (register #5)
#
# Exponential + nugget semivariogram (`model_utils.semivariogram_model`: nugget + two exponentials) fitted to the pixels
# more than 5 km from the trace, a plane removed (`model_utils.covariance`; Sudhaus & Jonsson 2009, Vavra & Fialko
# 2026). The model swath (4 km) holds no such pixels, so the field is rebuilt on a 15 km swath for this fit only
# (`model_config.noise_swath_km`). Saved as products: the fitted parameters and the binned semivariogram (08b draws it).
# Under the locked weighting (#6) only the nugget enters the inversion (every sample has variance nugget^2); the full
# covariance is used for the Monte Carlo noise of 08a and for the full-Cd sensitivity row.

# %%
## [4]
sar_wide, _ = model_utils.fault_parallel_dataset(fp["fp_event"], fp["lon"], fp["lat"], strike_deg, trace_fault, model_frame_lonlat,
                                                 cfg.noise_swath_km, cfg.exclude_km, cfg.km_window, profile_lats)
cov_p, semivariogram_bins, n_noise_pixels = model_utils.covariance(sar_wide, trace_fault, cfg.noise_swath_km, cfg.noise_mask_km,
                                                                   cfg.noise_distmax_km, cfg.noise_npairs, cfg.noise_seed,
                                                                   return_bins=True)
cov_p = np.asarray(cov_p, float)
noise_model = pd.DataFrame([dict(mask_km=cfg.noise_mask_km, swath_km=cfg.noise_swath_km, distmax_km=cfg.noise_distmax_km,
                                 n_pixels=n_noise_pixels, npairs=cfg.noise_npairs, seed=cfg.noise_seed,
                                 nugget2_mm2=cov_p[0], sill_short_mm2=cov_p[1], range_short_km=cov_p[2],
                                 sill_long_mm2=cov_p[3], range_long_km=cov_p[4], nugget_mm=np.sqrt(cov_p[0]),
                                 short_sd_mm=np.sqrt(cov_p[1]), long_sd_mm=np.sqrt(cov_p[3]),
                                 total_sd_mm=np.sqrt(cov_p[0] + cov_p[1] + cov_p[3]),
                                 provenance=f"{notebook} {made}: model_utils.covariance, 06 field {field_fingerprint}")])
noise_file = os.path.join(out_dir, "noise_model_5km.csv")
bins_file = os.path.join(out_dir, "semivariogram_bins_5km.csv")
noise_model.to_csv(noise_file, index=False)
semivariogram_bins.to_csv(bins_file, index=False)
assert 0.5 < np.sqrt(cov_p[0]) < 1.5, f"nugget {np.sqrt(cov_p[0]):.3f} mm outside the plausible range"
print(f"{n_noise_pixels:,} pixels beyond {cfg.noise_mask_km:g} km: nugget {np.sqrt(cov_p[0]):.3f} mm; short "
      f"{np.sqrt(cov_p[1]):.3f} mm / {cov_p[2]:.2f} km; long {np.sqrt(cov_p[3]):.2e} mm / {cov_p[4]:.2f} km "
      f"(a long sill of ~0 with its range on the 2 km bound = a nugget plus ONE exponential)")
print(f"wrote {noise_file}, {bins_file}")

# %% [markdown]
# ## [5] The coherence-0.7 sample set (the sensitivity row "Data coherence 0.7")
#
# 06's mean event at the time-series fit mask (coherence 0.7) instead of 0.9, re-decomposed here from 06's saved
# per-track steps (`model_utils.fault_parallel_from_steps`, MintPy `asc_desc2horz_vert`). GATE: the same call with the
# coherence-0.9 masks reproduces 06's `fp_maps.npz` exactly. Its quadtree uses the same rule; its PyLith run
# (`gf_default_sfcvm_coh07_order2`) was computed at its positions, asserted unchanged when the list is rewritten in [6].

# %%
## [5]
t0 = time.time()
fp_event_coh09 = model_utils.fault_parallel_from_steps(steps_files, strike_deg, cfg.decomposition_window_px, mask09_files)
assert np.array_equal(np.isfinite(fp_event_coh09), np.isfinite(fp["fp_event"])), "the rebuilt 0.9 field has another NaN pattern than 06's"
gate_diff = float(np.nanmax(np.abs(fp_event_coh09 - fp["fp_event"])))
assert gate_diff == 0.0, f"the coherence-0.9 field rebuilt from 06's saved steps differs from fp_maps.npz by {gate_diff} mm"
print("PASS: 06's field rebuilt from its saved steps at coherence 0.9 equals fp_maps.npz exactly")
fp_event_coh07 = model_utils.fault_parallel_from_steps(steps_files, strike_deg, cfg.decomposition_window_px)
sar_coh07, _ = model_utils.fault_parallel_dataset(fp_event_coh07, fp["lon"], fp["lat"], strike_deg, trace_fault, model_frame_lonlat,
                                                  cfg.swath_km, cfg.exclude_km, cfg.km_window, profile_lats)
quadtree_coh07 = model_utils.downsample(sar_coh07, trace_fault, cfg.start_block_km, cfg.min_block_km)
block_lon_coh07, block_lat_coh07 = model_utils.quadtree_block_corners(quadtree_coh07)
print(f"coherence-{coh_timeseries} field: {int(np.isfinite(sar_coh07.vel).sum()):,} pixels -> {quadtree_coh07.vel.size} samples "
      f"({time.time() - t0:.0f} s)")

# %% [markdown]
# ## [6] The products of the data side
#
# - The PyLith point lists (`model_utils.write_points_list`, "name x y z" in the mesh CRS + an npz with the data, block
#   pixel counts, block corners and the noise model): the selected set and the coherence-0.7 set. The positions are
#   ASSERTED unchanged (the cached Green's functions are computed at them). Both carry the 5 km noise model and the
#   locked exponent (until 2026-10-30 they carried 07a's 1 km fit and N^0.5).
# - `model_data.npz`: everything the slip models take from here, in one file.

# %%
## [6]
sample_sets = {"": (quadtree, block_lon, block_lat, "06 fp_maps.npz (coherence 0.9)"),
               "_coh07": (quadtree_coh07, block_lon_coh07, block_lat_coh07, f"06 mean event at coherence {coh_timeseries}")}
for tag, (ds, b_lon, b_lat, field) in sample_sets.items():
    model_utils.write_points_list(
        os.path.join(points_dir, "insar_points" + tag), ds.lon, ds.lat, model_crs,
        header=f"quadtree samples of 07a_uniform_model{tag} ({ds.vel.size} points, swath {cfg.swath_km:g} km, {field}), written {made}",
        extra=dict(x_km=np.asarray(ds.x, float), y_km=np.asarray(ds.y, float), vel_mm=np.asarray(ds.vel, float),
                   wgt=np.asarray(ds.wgt, float), block_lon=b_lon, block_lat=b_lat, cov_p=cov_p, nug_power=cfg.nugget_power,
                   swath_km=cfg.swath_km, strike_deg=strike_deg, field=field, field_fingerprint=field_fingerprint,
                   positions_fingerprint=model_utils.positions_fingerprint(ds.lon, ds.lat),
                   provenance=f"{notebook} {made}: quadtree of model_utils.downsample; noise model = the 5 km fit of [4] (both sets)"))
model_data_file = os.path.join(out_dir, "model_data.npz")
np.savez(model_data_file,
         sample_lon=np.asarray(quadtree.lon, float), sample_lat=np.asarray(quadtree.lat, float),
         sample_x_km=np.asarray(quadtree.x, float), sample_y_km=np.asarray(quadtree.y, float),
         data_mm=np.asarray(quadtree.vel, float), wgt=np.asarray(quadtree.wgt, float), block_lon=block_lon, block_lat=block_lat,
         cov_p=cov_p, nugget_power=cfg.nugget_power, strike_deg=strike_deg, look=look, frame_lonlat=np.array(model_frame_lonlat),
         crs=model_crs, trace_lonlat=trace["lonlat"], trace_every_km=trace["every_km"],
         creep_lo_km=creep_lo_km, creep_hi_km=creep_hi_km, n_pixels=n_pixels,
         points_default=os.path.join(points_dir, "insar_points.npz"), points_coh07=os.path.join(points_dir, "insar_points_coh07.npz"),
         positions_fingerprint=positions_key, field_fingerprint=field_fingerprint,
         provenance=f"{notebook} {made}; {utils.provenance_stamp(notebook, repo_dir)}; 06 field {field_fingerprint}")
print(f"wrote {model_data_file}: {quadtree.vel.size} samples, strike {strike_deg:.4f}")

# %% [markdown]
# ## [7] The homogeneous Green's functions on the PyLith tents (register #13)
#
# The selected PyLith run (`model_config.gf_run`) fixes the fault: its vertices, triangles and impulse vertices (the
# "tents": unit slip at one vertex, linear to zero at its neighbours). `model_utils.analytic_tent_G` gives the same tents
# in a homogeneous half-space (Poisson solid) with CSI's triangular dislocations (Meade 2007): every triangle is split
# into n_sub^2 constant-slip sub-triangles.
#
# The n_sub check (rule of the brief): 20 test tents in the SSE section (5 at the surface, 15 spread over depth by rank),
# with n_sub 4 and 8; if any column differs by more than 1 % (relative L2 norm over the samples), n_sub 8 is used.
# The full matrix is cached, keyed by run, n_sub, sample positions and impulse vertices (`analytic_tent_G_cached`).
# CHECK: uniform slip on every tent moves the east side along +look (right-lateral), as the PyLith G does.

# %%
## [7]
to_mesh = pyproj.Transformer.from_crs("EPSG:4326", model_crs, always_xy=True)
gf_npz = model_utils.load_G(cfg.gf_run, os.path.join(points_dir, "insar_points.npz"), strike_deg, cfg.gf_pylith_output_dir, cfg.gf_gf_dir)
tents = model_utils.TentFault(gf_npz, to_mesh, profile_lats)
in_section_tents = np.nonzero((tents.imp_km > creep_lo_km + 0.5) & (tents.imp_km < creep_hi_km - 0.5))[0]
surface_tents = in_section_tents[tents.imp_depth_km[in_section_tents] < 1e-6]
buried_tents = in_section_tents[tents.imp_depth_km[in_section_tents] > 1e-6]
buried_by_depth = buried_tents[np.argsort(tents.imp_depth_km[buried_tents])]
n_check = cfg.analytic_nsub_check_tents
test_tents = np.r_[surface_tents[np.linspace(0, surface_tents.size - 1, 5).astype(int)],
                   buried_by_depth[np.linspace(0, buried_by_depth.size - 1, n_check - 5).astype(int)]]
test_fault = SimpleNamespace(fault_xyz=tents.fault_xyz, fault_cells=tents.fault_cells, vertex_index=tents.vertex_index[test_tents],
                             n_imp=test_tents.size)
G_test = {}
for n_sub in cfg.analytic_nsub_options:
    G_test[n_sub] = model_utils.analytic_tent_G(test_fault, quadtree.lon, quadtree.lat, look, model_frame_lonlat, to_mesh, n_sub=n_sub)
n_coarse, n_fine = cfg.analytic_nsub_options
nsub_rel_diff = np.linalg.norm(G_test[n_fine] - G_test[n_coarse], axis=0) / np.linalg.norm(G_test[n_fine], axis=0)
nsub_table = pd.DataFrame(dict(tent_depth_m=np.round(tents.imp_depth_km[test_tents] * 1e3), tent_km=np.round(tents.imp_km[test_tents], 2),
                               rel_diff=nsub_rel_diff))
print(nsub_table.to_string(index=False))
n_sub = n_fine if nsub_rel_diff.max() > cfg.analytic_nsub_tolerance else n_coarse
print(f"largest n_sub {n_coarse} vs {n_fine} difference {nsub_rel_diff.max():.3f} "
      f"({'>' if n_sub == n_fine else '<='} {cfg.analytic_nsub_tolerance:g}): n_sub = {n_sub}")
analytic_cache = os.path.join(cfg.analytic_G_dir, f"G_analytic_{cfg.gf_run}_nsub{n_sub}.npz")
G_homogeneous = model_utils.analytic_tent_G_cached(analytic_cache, tents, quadtree.lon, quadtree.lat, look, model_frame_lonlat, to_mesh,
                                                   n_sub, cfg.gf_run, positions_key)
assert G_homogeneous.shape == gf_npz["G_rl_m_per_m"].shape, "analytic and PyLith G differ in shape"
signed_d_km = model_utils.signed_distance_m(trace_xy_m[0], trace_xy_m[1], points_old["x"], points_old["y"]) / 1e3
near_band = np.abs(signed_d_km) < 0.3
for label, G_check in (("analytic", G_homogeneous), ("PyLith 3D", gf_npz["G_rl_m_per_m"])):
    uniform = G_check.sum(axis=1)
    step = uniform[near_band & (signed_d_km > 0)].mean() - uniform[near_band & (signed_d_km < 0)].mean()
    print(f"{label}: uniform unit slip on every tent, east-minus-west within 300 m = {step:+.3f} m/m")
    assert step > 0, f"{label}: right-lateral slip does not move the east side along +look"
print(f"G homogeneous {G_homogeneous.shape} from {analytic_cache}")

# %% [markdown]
# ## [8] The inversion set-up (the locked method, the same as 08a)
#
# `model_utils.tent_model_setup` on the selected run gives the tent fault, the CSI template (`TriangularTents` on the
# impulse vertices), every sample's signed distance and along-strike km, `keep` (the fitted samples: 60 m or more from
# the trace, #2) and the near-field masks (fitted samples within 500 m west / east, in the SSE section). The samples
# are read back from `model_data.npz` (`model_utils.samples_from_arrays`), exactly as 08a reads them. Data covariance:
# diagonal, nugget^2 for every sample ("the data were equally weighted", #6; `model_utils.data_cd`). The plane (1, x, y)
# is solved jointly (#7), slip >= 0 (#14), CSI exponential prior with lambda (0.1, 0.05) km (#15), no surface
# constraint (#17): all inside `model_utils.invert_tents`.

# %%
## [8]
model_data = np.load(model_data_file)
samples = model_utils.samples_from_arrays("fault-parallel samples", model_data["sample_lon"], model_data["sample_lat"],
                                          model_data["sample_x_km"], model_data["sample_y_km"], model_data["data_mm"],
                                          model_data["wgt"], look, model_frame_lonlat, cov_p, cfg.nugget_power)
setup = model_utils.tent_model_setup({"sfcvm": cfg.gf_run}, str(model_data["points_default"]), strike_deg,
                                     {"sfcvm": (cfg.gf_pylith_output_dir, cfg.gf_gf_dir)}, cfg.km_window, creep_lo_km, creep_hi_km,
                                     trace_xy_m, model_frame_lonlat, model_crs, profile_lats, cfg.near_trace_cut_km,
                                     cfg.near_field_km, trace["km_pp"])
assert np.array_equal(setup["tents"].imp_xyz, tents.imp_xyz), "the set-up's tents are not the tents of the analytic G"
keep = setup["keep"]
Cd = model_utils.data_cd(cfg.weighting, samples, cov_p, cfg.nugget_power, keep=keep)
print(f"{keep.sum()} fitted samples of {keep.size} (cut {cfg.near_trace_cut_km * 1e3:.0f} m); near field: "
      f"{setup['west_near'].sum()} west, {setup['east_near'].sum()} east; every fitted sample sd {np.sqrt(Cd[keep, keep][0]):.3f} mm")


def solve_homogeneous(sigma):
    """The homogeneous model at one sigma (the locked inversion, model_utils.invert_tents)."""
    return model_utils.invert_tents(setup["template"], samples, G_homogeneous, Cd, None, None, sigma, cfg.lam_xz_km,
                                    model_frame_lonlat, use_surface=cfg.use_surface, name=f"homogeneous sigma {sigma:.3f}",
                                    ramp_spec=cfg.ramp_spec)

# %% [markdown]
# ## [9] The L-curve and the model's own corner (register #16)
#
# `model_utils.own_corner`: the L-curve (weighted misfit of the fitted samples vs prior-weighted roughness) on the base
# grid, its corner (maximum curvature, Hansen 1992), then nine more sigmas around that corner and the corner of the
# union. The model is solved at that corner (not snapped to a grid value).

# %%
## [9]
t0 = time.time()
sigma_hom, sigma_hom_base, lcurve_hom = model_utils.own_corner(solve_homogeneous, cfg.sigma_grid_base, cfg.sigma_refine_factors, tents,
                                                               cfg.mu_ref_pa, "homogeneous", keep)
lcurve_file = os.path.join(out_dir, "homogeneous_lcurve.csv")
lcurve_hom.assign(corner=sigma_hom, base_corner=sigma_hom_base).to_csv(lcurve_file, index=False)
print(f"homogeneous corner {sigma_hom:.2f} (base grid {sigma_hom_base:.2f}); {len(lcurve_hom)} solves in {(time.time() - t0) / 60:.1f} min; "
      f"wrote {lcurve_file}")

# %% [markdown]
# ## [10] The best homogeneous model and its numbers
#
# Solved at its corner. The numbers (`model_utils.tent_numbers`, the metric policy of 2026-10-28): RMS of the
# plane-removed data minus the model over the fitted samples and within 500 m west / east in the SSE section; potency;
# Mw at 30 GPa (the medium is uniform: no other rigidity); peak slip and its depth; the depths above which 50 % and
# 90 % of the potency lie.

# %%
## [10]
fault_hom, data_hom, _ = solve_homogeneous(sigma_hom)
slip_hom = fault_hom.slip[:, 0].copy()
numbers_hom = model_utils.tent_numbers(slip_hom, data_hom, tents, keep, setup["west_near"], setup["east_near"], cfg.mu_ref_pa)
plane_hom = samples.vel - data_hom.vel                                   # the fitted plane at every sample (CSI removed it from vel)
design = np.c_[np.ones(samples.vel.size), samples.x, samples.y]
plane_coef = np.linalg.lstsq(design, plane_hom, rcond=None)[0]
assert np.allclose(design @ plane_coef, plane_hom, atol=1e-6), "the removed reference frame is not a plane in (x, y)"
assert np.allclose(G_homogeneous @ slip_hom, data_hom.synth, atol=1e-6), "CSI's prediction is not G x slip"
for key, value in numbers_hom.items():
    print(f"  {key:18s} {value:.4g}")
print(f"  fitted plane: {plane_coef[0]:+.3f} mm {plane_coef[1]:+.3f} mm/km (x) {plane_coef[2]:+.3f} mm/km (y)")

# %% [markdown]
# ## [11] Products and registry
#
# `homogeneous_model.npz`: the slip on the tents, the model (G x slip) at every sample, the fitted plane, the
# plane-removed data, `keep` and the sigma (08a compares it with the 3D model; 07c draws its line in Fig 2).

# %%
## [11]
model_file = os.path.join(out_dir, "homogeneous_model.npz")
np.savez(model_file, slip_mm=slip_hom, imp_xyz=tents.imp_xyz, model_mm=data_hom.synth.copy(), plane_mm=plane_hom,
         data_plane_removed_mm=data_hom.vel.copy(), keep=keep, sigma=sigma_hom, sigma_base_corner=sigma_hom_base, n_sub=n_sub,
         sample_lon=model_data["sample_lon"], sample_lat=model_data["sample_lat"], positions_fingerprint=positions_key,
         analytic_G=analytic_cache, **{k: v for k, v in numbers_hom.items()},
         provenance=f"{notebook} {made}: analytic (CSI Meade, n_sub {n_sub}) tents of {cfg.gf_run}; locked method (model_config)")
stats = REG.sidecar("07a_homogeneous", cfg.summary() + [f"noise model: {noise_model.iloc[0].drop('provenance').to_dict()}",
                                                         f"n_sub check: {nsub_table.to_dict('list')}",
                                                         f"sigma corner {sigma_hom:.3f} (base grid {sigma_hom_base:.3f})"]
                    + [f"{k} = {v}" for k, v in numbers_hom.items()])
where = "M8"
REG.register("07a_n_samples", int(samples.vel.size), "count", 0, where, "[3]", stats)
REG.register("07a_n_samples_fitted", int(keep.sum()), "count", 0, where, "[8]", stats)
REG.register("07a_noise_nugget_mm", float(np.sqrt(cov_p[0])), "mm", 2, where, "[4]", stats)
REG.register("07a_noise_sill_mm", float(np.sqrt(cov_p[1])), "mm", 2, where, "[4]", stats)
REG.register("07a_noise_range_km", float(cov_p[2]), "km", 1, where, "[4]", stats)
REG.register("07a_analytic_nsub", n_sub, "count", 0, where, "[7]", stats)
REG.register("07a_hom_sigma", sigma_hom, "mm", 1, where, "[9]", stats)
REG.register("07a_hom_mw_30gpa", numbers_hom["Mw_30gpa"], "Mw", 2, where, "[10]", stats)
REG.register("07a_hom_potency_m3", numbers_hom["potency_m3"], "m3", 0, where, "[10]", stats)
REG.register("07a_hom_peak_slip_mm", numbers_hom["peak_slip_mm"], "mm", 1, where, "[10]", stats)
REG.register("07a_hom_peak_depth_m", numbers_hom["peak_depth_m"], "m", 0, where, "[10]", stats)
REG.register("07a_hom_depth50_m", numbers_hom["depth50_m"], "m", 0, where, "[10]", stats)
REG.register("07a_hom_depth90_m", numbers_hom["depth90_m"], "m", 0, where, "[10]", stats)
REG.register("07a_hom_rms_mm", numbers_hom["rms_mm"], "mm", 2, where, "[10]", stats)
REG.register("07a_hom_rms_west500_mm", numbers_hom["rms_west_mm"], "mm", 2, where, "[10]", stats)
REG.register("07a_hom_rms_east500_mm", numbers_hom["rms_east_mm"], "mm", 2, where, "[10]", stats)
REG.render()
print(f"wrote {model_file}")
