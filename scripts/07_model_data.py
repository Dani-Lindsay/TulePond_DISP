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
# # 07 Model data and the homogeneous slip model
#
# ```
# Author: Danielle Lindsay
# Environment: environment.yml. Runs after 04 and 06. About 15 min with the analytic Green's functions cached, about 2 h without.
# ```
#
# **What this notebook does**
#
# 1. **Data of the slip models** (sections 2 to 6). The five-event mean fault-parallel field of notebook 06 is sampled
#    with a quadtree; the samples are also the output points of the PyLith runs. The noise model is a semivariogram of
#    the pixels more than 5 km from the trace. A second sample set, from the field at temporal coherence 0.7, serves
#    the coherence sensitivity test. Notebooks 08 and 10 and the scripts in `scripts/sensitivity/` read these products.
# 2. **Homogeneous slip model** (sections 7 to 11). The inversion of notebook 08 is repeated on the same fault (the fault
#    of the PyLith run `model_config.gf_run`, dipping 70° to the west) with the Green's functions of a homogeneous
#    elastic half-space, computed with CSI triangular dislocations (Meade 2007). It is the homogeneous comparison model
#    of notebook 08 and Fig 2.
#
# The settings are in `model_config.py` and the modelling functions in `model_utils.py`.
#
# **Meshes and PyLith Green's functions.** No notebook builds a mesh or runs PyLith. The meshes are built by
# `pylith/build/generate_gmsh_vertical.py` and `pylith/build/generate_gmsh_dip.py`, and the Green's functions are
# computed by `pylith/run_gf.sh`. Both are run by hand, as documented in `pylith/README.md`. This notebook writes the
# point lists that the PyLith runs use as output points (section 6) and reads the stored products of the run
# `model_config.gf_run`: its fault vertices and its G matrix, assembled from the PyLith output by
# `model_utils.build_G` (section 7).
#
# **Inputs**: notebook 06 `fp_maps.npz` (the field) and `tulepond_steps_{ascending,descending}.h5` (for the
# coherence-0.7 field); notebook 04 `section_lengths.csv`; the trace KML and `hayward_fault_profiles.csv`; the
# temporal-coherence 0.9 masks; the stored G matrix of the PyLith run `model_config.gf_run` (its fault vertices).
#
# **Outputs** (`outputs/07_model_data/`): `model_data.npz` (the samples, the noise model, the trace),
# `noise_model_5km.csv` and `semivariogram_bins_5km.csv` (the noise model), `homogeneous_lcurve.csv`,
# `homogeneous_model.npz` (slip, prediction and plane at the samples), `07_homogeneous_stats.txt`; the point lists
# `modeling_inputs/points/insar_points{,_coh07}.{txt,npz}` (positions asserted unchanged); the analytic Green's
# functions `modeling_uniform/gf/G_analytic_<run>_nsub<n>.npz` (cache). **Registry keys** `07_*`.
#
# **What to check**
# - Section 3: the check that the quadtree samples are the PyLith output points (same positions, same data) passes.
# - Section 4: the printed noise fit. The sill of the second exponential term should be near zero.
# - Section 7: the n_sub table. The largest relative difference of the 20 test tents sets n_sub.
# - Section 9: the L-curve corner lies inside the sigma grid (asserted).
# - Section 10: the printed RMS of the fitted samples, the near-field RMS west and east, Mw at 30 GPa, and depth50 and
#   depth90.

# %% [markdown]
# ## 1. Imports, settings and the number registry
#
# The settings are printed from `model_config.summary()` so that the executed notebook records them.

# %%
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
import model_config as cfg                          # the settings of the slip models
import model_utils                                  # the modelling functions

notebook = "07_model_data"
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
# ## 2. The mapped trace
#
# The mapped trace (KML) between the latitudes of the model region is discretised every 20 m as a CSI trace
# (`model_utils.model_trace`; CSI, Jolivet 2024). It gives each pixel's distance from the fault (used by the quadtree
# and the noise mask) and each sample's signed distance from the trace and position along strike. Positions along
# strike are in km from Point Pinole, taken from the nearest vertex of `hayward_fault_profiles.csv` (vertices 100 m
# apart).

# %%
profile_lats = pd.read_csv(common_paths["fault_profiles"]).ft_lat.values      # latitudes of the 100 m fault-profile vertices
trace = model_utils.model_trace(load_trace(common_paths["fault_trace"]), *cfg.fault_trace_lat_range, model_frame_lonlat, profile_lats)
trace_fault = trace["ref_fault"]                                              # CSI object; xf, yf are the trace points
trace_xy_m = (trace["x_km"] * 1e3, trace["y_km"] * 1e3)                       # m; the mesh CRS and the CSI frame share the origin
print(f"trace: {trace['lonlat'].shape[0]} KML vertices, {trace['arc_total']:.1f} km, discretised every "
      f"{trace['every_km'] * 1e3:.0f} m; SSE section km {creep_lo_km:.2f}-{creep_hi_km:.2f}")

# %% [markdown]
# ## 3. The data: the fault-parallel mean event and its quadtree samples
#
# The fault-parallel field of notebook 06 (temporal coherence 0.9, projected on the strike of the SSE section) is read as
# one `csi.insar` data set (`model_utils.fault_parallel_dataset`). Its look vector is the horizontal unit vector along
# strike,
#
# $$\hat{\mathbf{e}} = (\sin\theta,\ \cos\theta,\ 0),$$
#
# where $\hat{\mathbf{e}}$ has east, north and up components and $\theta$ is the strike (degrees clockwise from north).
# With this vector, right-lateral motion of the east side is positive. The data are the pixels within 4 km of the
# trace, without the pixel on the trace, between km 52 and 67 along strike. A distance-based quadtree reduces them to
# samples, with blocks from 2 km far from the fault down to single 30 m pixels at the trace (`model_utils.downsample`,
# CSI `imagedownsampling`; Lohman & Simons 2005).
#
# Check: the PyLith and analytic Green's functions are computed at the sample positions in `insar_points.npz`, so the
# samples must match that file in position, data and block size.

# %%
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
print(f"Check passed: the samples are the PyLith output points (positions, data and block sizes identical; key {positions_key})")

# %% [markdown]
# ## 4. The noise model: semivariogram of the far field
#
# The noise model is fitted to the pixels more than 5 km from the trace, after a best-fitting plane is removed from them
# (`model_utils.covariance`; Lohman & Simons 2005; Sudhaus & Jónsson 2009; Vavra & Fialko 2026). The 4 km model swath
# holds no such pixels, so the field is rebuilt on a 15 km swath for this fit (`model_config.noise_swath_km`). From
# 300,000 random pixel pairs closer than 8 km, the empirical semivariogram is computed in 0.1 km distance bins (bins
# with more than 50 pairs),
#
# $$\hat\gamma(h_k) = \frac{1}{2N_k} \sum_{(i,j) \in k} (u_i - u_j)^2 ,$$
#
# and the model (`model_utils.semivariogram_model`)
#
# $$\gamma(h) = n^2 + s_1\left(1 - e^{-h/l_1}\right) + s_2\left(1 - e^{-h/l_2}\right)$$
#
# is fitted to it by bounded least squares (`scipy.optimize.curve_fit`), where $h$ is the distance between two pixels
# (km), $h_k$ the centre of bin $k$, $N_k$ the number of pairs in bin $k$, $u_i$ the plane-removed fault-parallel
# displacement of pixel $i$ (mm), $\gamma$ the semivariance (mm$^2$), $n^2$ the nugget (mm$^2$), $s_1$ and $s_2$ the
# sills of a short- and a long-range exponential term (mm$^2$), and $l_1$ and $l_2$ their ranges (km). The fitted
# parameters and the binned semivariogram are saved; notebook 10 (section 7) plots them.
#
# The covariance of two quadtree samples $i$ and $j$ that follows from this model (`model_utils.covariance_matrix`) is
#
# $$C_{ij} = s_1 e^{-r_{ij}/l_1} + s_2 e^{-r_{ij}/l_2} + \delta_{ij}\,\frac{n^2}{N_i^{\,p}} ,$$
#
# where $r_{ij}$ is the distance between the samples (km), $\delta_{ij}$ the Kronecker delta, $N_i$ the number of pixels
# that sample $i$ averages, and $p$ the nugget exponent. The inversion uses only the diagonal with $p = 0$, so every
# sample has the variance $n^2$ (equal weights; section 8). The full covariance with $p = 0.5$ is used for the
# Monte Carlo noise realisations of notebook 08 and for the full-covariance sensitivity test.

# %%
sar_wide, _= model_utils.fault_parallel_dataset(fp["fp_event"], fp["lon"], fp["lat"], strike_deg, trace_fault, model_frame_lonlat,
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
      f"(a second sill of ~0 at its range bound: a nugget plus one exponential term)")
print(f"wrote {noise_file}, {bins_file}")

# %% [markdown]
# ## 5. The coherence-0.7 sample set (sensitivity test)
#
# The slip models fit the mean event at temporal coherence 0.9. The "Data coherence 0.7" row of the sensitivity table
# (notebook 10, section 3) tests whether the model depends on this mask, and this section makes its data. The mean event
# of notebook 06 is masked at temporal coherence 0.7, the mask of the time-series fit, instead of 0.9. The field is
# rebuilt from the per-track steps that notebook 06 saved and decomposed into fault-parallel and vertical motion with
# MintPy `asc_desc2horz_vert` (Yunjun et al. 2019), with the same window as in 06
# (`model_utils.fault_parallel_from_steps`). It is sampled with the same quadtree rule as the main set (section 3).
#
# Check: the same call with the coherence-0.9 masks must reproduce 06's `fp_maps.npz`. The PyLith run
# `gf_default_sfcvm_coh07_order2` (vertical fault, default mesh) was computed at the positions of this sample set, so
# section 6 asserts that they are unchanged.

# %%
t0 = time.time()
fp_event_coh09 = model_utils.fault_parallel_from_steps(steps_files, strike_deg, cfg.decomposition_window_px, mask09_files)
assert np.array_equal(np.isfinite(fp_event_coh09), np.isfinite(fp["fp_event"])), "the rebuilt 0.9 field has another NaN pattern than 06's"
gate_diff = float(np.nanmax(np.abs(fp_event_coh09 - fp["fp_event"])))
assert gate_diff == 0.0, f"the coherence-0.9 field rebuilt from 06's saved steps differs from fp_maps.npz by {gate_diff} mm"
print("Check passed: 06's field rebuilt from its saved steps at coherence 0.9 equals fp_maps.npz")
fp_event_coh07 = model_utils.fault_parallel_from_steps(steps_files, strike_deg, cfg.decomposition_window_px)
sar_coh07, _ = model_utils.fault_parallel_dataset(fp_event_coh07, fp["lon"], fp["lat"], strike_deg, trace_fault, model_frame_lonlat,
                                                  cfg.swath_km, cfg.exclude_km, cfg.km_window, profile_lats)
quadtree_coh07 = model_utils.downsample(sar_coh07, trace_fault, cfg.start_block_km, cfg.min_block_km)
block_lon_coh07, block_lat_coh07 = model_utils.quadtree_block_corners(quadtree_coh07)
print(f"coherence-{coh_timeseries} field: {int(np.isfinite(sar_coh07.vel).sum()):,} pixels -> {quadtree_coh07.vel.size} samples "
      f"({time.time() - t0:.0f} s)")

# %% [markdown]
# ## 6. Data products
#
# - The PyLith point lists of the main and the coherence-0.7 sample sets (`model_utils.write_points_list`): a text file
#   of "name x y z" in the mesh CRS, and an npz with the data, the block pixel counts, the block corners and the noise
#   model. The stored Green's functions are computed at these positions, so the positions are asserted unchanged.
# - `model_data.npz`: the data inputs of the slip models in one file.

# %%
sample_sets = {"": (quadtree, block_lon, block_lat, "06 fp_maps.npz (coherence 0.9)"),
               "_coh07": (quadtree_coh07, block_lon_coh07, block_lat_coh07, f"06 mean event at coherence {coh_timeseries}")}
for tag, (ds, b_lon, b_lat, field) in sample_sets.items():
    model_utils.write_points_list(
        os.path.join(points_dir, "insar_points" + tag), ds.lon, ds.lat, model_crs,
        header=f"quadtree samples of 07_model_data{tag} ({ds.vel.size} points, swath {cfg.swath_km:g} km, {field}), written {made}",
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
# ## 7. Homogeneous Green's functions on the PyLith fault
#
# The PyLith run `model_config.gf_run` defines the fault: its vertices, its triangles and its impulse vertices. Each
# impulse is a "tent", unit slip at one vertex that decreases linearly to zero at the neighbouring vertices.
# `model_utils.analytic_tent_G` computes the same tents in a homogeneous elastic half-space (Poisson solid) with CSI
# triangular dislocations (Meade 2007; Jolivet 2024). Each fault triangle is split into $n_{sub}^2$ sub-triangles of
# constant slip, and the Green's function of tent $j$ is
#
# $$\mathbf{g}_j = \sum_{t \ni j} \sum_{q=1}^{n_{sub}^2} \phi_j(\mathbf{c}_{tq})\, \mathbf{g}_{tq} ,$$
#
# where $t$ runs over the fault triangles that touch vertex $j$, $q$ over their sub-triangles, $\mathbf{c}_{tq}$ is the
# centroid of sub-triangle $q$ of triangle $t$, $\phi_j$ the tent function of vertex $j$ (1 at the vertex, 0 at the
# other two corners, linear in between), and $\mathbf{g}_{tq}$ the fault-parallel displacement at the samples (m) for
# 1 m of right-lateral slip on the sub-triangle.
#
# Choice of $n_{sub}$: 20 test tents in the SSE section (5 at the surface, 15 spread over depth) are computed with
# $n_{sub} = 4$ and $8$, and their relative difference is
#
# $$\varepsilon_j = \frac{\lVert \mathbf{g}_j^{(8)} - \mathbf{g}_j^{(4)} \rVert_2}{\lVert \mathbf{g}_j^{(8)} \rVert_2} ,$$
#
# where the norms run over the samples. If any $\varepsilon_j$ exceeds 1 %, $n_{sub} = 8$ is used, otherwise 4. The
# full matrix is cached and keyed by run, $n_{sub}$, sample positions and impulse vertices (`analytic_tent_G_cached`).
#
# Check: uniform right-lateral slip on every tent moves the east side along $+\hat{\mathbf{e}}$, as in the PyLith
# Green's functions.

# %%
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
                                                   n_sub, cfg.gf_run, positions_key, n_workers=cfg.analytic_n_workers)
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
# ## 8. Inversion set-up (as in notebook 08)
#
# `model_utils.tent_model_setup` gives the tent fault, its CSI template (`TriangularTents` on the impulse vertices), each
# sample's signed distance from the trace and position along strike, `keep` (the fitted samples, 60 m or more from the
# trace) and the near-field masks (fitted samples within 500 m west or east of the trace in the SSE section). The samples
# are read from `model_data.npz` (`model_utils.samples_from_arrays`), as in notebook 08.
#
# The inversion (`model_utils.invert_tents`, CSI; Jolivet 2024) solves for the slip and a plane together:
#
# $$\min_{\mathbf{s} \ge 0,\ \mathbf{a}} \;
#   (\mathbf{d} - G\mathbf{s} - B\mathbf{a})^{T} C_d^{-1} (\mathbf{d} - G\mathbf{s} - B\mathbf{a})
#   + \mathbf{s}^{T} C_m^{-1} \mathbf{s}
#   + (\mathbf{a} - \mathbf{a}_0)^{T} C_a^{-1} (\mathbf{a} - \mathbf{a}_0) ,$$
#
# where $\mathbf{d}$ is the vector of fault-parallel samples (mm), $G$ the Green's functions (fault-parallel
# displacement per unit right-lateral slip, m/m), $\mathbf{s}$ the right-lateral slip at the impulse vertices (mm),
# $B$ the plane design matrix with rows $(1, x_i, y_i)$ ($x_i$, $y_i$ the sample position in km in the CSI frame),
# and $\mathbf{a}$ the plane coefficients (mm, mm/km, mm/km).
#
# - $C_d$ is the data covariance (`model_utils.data_cd`, mode "nugget" with exponent 0): diagonal, with the nugget
#   variance $n^2$ of section 4 for every fitted sample (equal weights). Samples closer than 60 m to the trace are given
#   the variance $10^8 n^2$, so they stay in the products but do not influence the fit.
# - $C_m$ is the CSI exponential prior covariance of the slip (`Fault.buildCmXY`; Radiguet et al. 2011),
#
#   $$C_{m,jk} = \sigma^2 \,\frac{\lambda_0^2}{\lambda_h \lambda_v}\,
#     \exp\!\left(-\frac{h_{jk}}{2\lambda_h}\right) \exp\!\left(-\frac{v_{jk}}{2\lambda_v}\right) ,$$
#
#   where $\sigma$ is the regularisation strength (mm), $h_{jk}$ and $v_{jk}$ the horizontal and vertical distances
#   between impulse vertices $j$ and $k$ (km), $\lambda_h = 0.1$ km and $\lambda_v = 0.05$ km the correlation lengths
#   along strike and down dip (`model_config.lam_xz_km`), and $\lambda_0$ a normalising length that CSI computes from
#   the vertex spacing (km). CSI divides the distances by $2\lambda$, so the e-folding lengths are 0.2 km and 0.1 km.
# - $C_a = 10^4 I$ and $\mathbf{a}_0 = -10^3$ form a weak prior on the plane. CSI solves the system with
#   non-negative least squares on $\mathbf{m} - \mathbf{m}_0$, with $\mathbf{m} = (\mathbf{s}, \mathbf{a})$ and
#   $\mathbf{m}_0 = (\mathbf{0}, \mathbf{a}_0)$, so the slip is non-negative and the plane coefficients can take either
#   sign.
#
# No surface constraint is used (`model_config.use_surface`).

# %%
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
    """Solve the homogeneous model at one regularisation strength.

    Parameters
    ----------
    sigma : float
        Regularisation strength of the CSI exponential prior (mm).

    Returns
    -------
    tuple
        (fault, InSAR data set, None) from `model_utils.invert_tents`.
    """
    return model_utils.invert_tents(setup["template"], samples, G_homogeneous, Cd, None, None, sigma, cfg.lam_xz_km,
                                    model_frame_lonlat, use_surface=cfg.use_surface, name=f"homogeneous sigma {sigma:.3f}",
                                    ramp_spec=cfg.ramp_spec)

# %% [markdown]
# ## 9. L-curve and corner
#
# The regularisation strength is the corner of the model's own L-curve (`model_utils.own_corner`; Hansen 1992). For
# each $\sigma$ the L-curve takes the weighted misfit of the fitted samples and the prior-weighted roughness of the slip
# (`model_utils.tent_lcurve`),
#
# $$\chi^2 = \mathbf{r}_K^{T} C_{d,K}^{-1} \mathbf{r}_K , \qquad
#   R = \sqrt{\mathbf{s}^{T} \left(C_m / \sigma^2\right)^{-1} \mathbf{s}} ,$$
#
# where $\mathbf{r}_K = \mathbf{d} - G\mathbf{s} - B\mathbf{a}$ is the residual over the fitted samples $K$ (mm),
# $C_{d,K}$ the data covariance of those samples, and $C_m/\sigma^2$ the prior covariance of section 8 without
# $\sigma^2$. With $x = \log_{10}\sqrt{\chi^2}$ and $y = \log_{10} R$ as cubic splines of $\log_{10}\sigma$
# (`model_utils.lcurve_corner`), the curvature is
#
# $$\kappa = \frac{\lvert x' y'' - y' x'' \rvert}{\left(x'^2 + y'^2\right)^{3/2}} ,$$
#
# where the primes are derivatives with respect to $\log_{10}\sigma$. $\kappa$ is evaluated at 400 points between the
# second and the second-last $\sigma$ of the grid, and the corner is the $\sigma$ of maximum curvature. The corner is
# found first on the base grid (`model_config.sigma_grid_base`), then on the base grid plus nine values
# $\sigma_c \, 2^{k/16}$ ($k = -4, \dots, 4$) around the first corner $\sigma_c$. The model is solved at the second
# corner.

# %%
t0 = time.time()
sigma_hom, sigma_hom_base, lcurve_hom = model_utils.own_corner(solve_homogeneous, cfg.sigma_grid_base, cfg.sigma_refine_factors, tents,
                                                               cfg.mu_ref_pa, "homogeneous", keep)
lcurve_file = os.path.join(out_dir, "homogeneous_lcurve.csv")
lcurve_hom.assign(corner=sigma_hom, base_corner=sigma_hom_base).to_csv(lcurve_file, index=False)
print(f"homogeneous corner {sigma_hom:.2f} (base grid {sigma_hom_base:.2f}); {len(lcurve_hom)} solves in {(time.time() - t0) / 60:.1f} min; "
      f"wrote {lcurve_file}")

# %% [markdown]
# ## 10. The homogeneous model and its numbers
#
# The model is solved at the corner $\sigma$, and `model_utils.tent_numbers` gives its numbers.
#
# - RMS of the residual $r_i = d_i - (G\mathbf{s})_i - (B\mathbf{a})_i$ over a set of samples $S$,
#   $\mathrm{RMS}_S = \sqrt{\frac{1}{|S|}\sum_{i \in S} r_i^2}$ (mm), where $S$ is the fitted samples, or the fitted
#   samples within 500 m west or east of the trace in the SSE section.
# - Potency, $P = \sum_t A_t \bar{s}_t$ (m$^3$), where $A_t$ is the area of fault triangle $t$ (m$^2$) and $\bar{s}_t$
#   the mean of the slip at its three vertices (m). Because the slip is linear within each triangle, this is the
#   integral of the slip over the fault.
# - Seismic moment and moment magnitude (Hanks & Kanamori 1979), $M_0 = \mu P$ and
#   $M_w = \frac{2}{3}\left(\log_{10} M_0 - 9.1\right)$, where $\mu = 30$ GPa is the reference rigidity
#   (`model_config.mu_ref_pa`) and $M_0$ is in N m.
# - Peak slip, $\max_j s_j$ (mm), and the depth of its vertex (m).
# - depth50 and depth90, the depths above which 50 % and 90 % of the potency lie (`model_utils.potency_extent`):
#   $z_q = \min\{\, z_t : \sum_{t' :\, z_{t'} \le z_t} A_{t'} \bar{s}_{t'} \ge q P \,\}$, where $z_t$ is the depth of
#   the centroid of triangle $t$ (m) and $q$ is 0.5 or 0.9.
#
# CSI removes the fitted plane from the data, so the plane at each sample is the data minus the plane-removed data. It
# is checked to be a plane in $(x, y)$, and the CSI prediction is checked to equal $G\mathbf{s}$.

# %%
fault_hom, data_hom, _ = solve_homogeneous(sigma_hom)
slip_hom = fault_hom.slip[:, 0].copy()
numbers_hom = model_utils.tent_numbers(slip_hom, data_hom, tents, keep, setup["west_near"], setup["east_near"], cfg.mu_ref_pa)
plane_hom = samples.vel - data_hom.vel                                   # the fitted plane at every sample (CSI removes it from vel)
design = np.c_[np.ones(samples.vel.size), samples.x, samples.y]
plane_coef = np.linalg.lstsq(design, plane_hom, rcond=None)[0]
assert np.allclose(design @ plane_coef, plane_hom, atol=1e-6), "the removed reference frame is not a plane in (x, y)"
assert np.allclose(G_homogeneous @ slip_hom, data_hom.synth, atol=1e-6), "CSI's prediction is not G x slip"
for key, value in numbers_hom.items():
    print(f"  {key:18s} {value:.4g}")
print(f"  fitted plane: {plane_coef[0]:+.3f} mm {plane_coef[1]:+.3f} mm/km (x) {plane_coef[2]:+.3f} mm/km (y)")

# %% [markdown]
# ## 11. Products and registry
#
# `homogeneous_model.npz` holds the slip on the tents, the model prediction $G\mathbf{s}$ at every sample, the fitted
# plane, the plane-removed data, `keep` and $\sigma$. Notebook 08 reads it, and notebook 09 draws it in Fig 2. The
# numbers of the paper are registered with the stats file `07_homogeneous_stats.txt`.

# %%
model_file = os.path.join(out_dir, "homogeneous_model.npz")
np.savez(model_file, slip_mm=slip_hom, imp_xyz=tents.imp_xyz, model_mm=data_hom.synth.copy(), plane_mm=plane_hom,
         data_plane_removed_mm=data_hom.vel.copy(), keep=keep, sigma=sigma_hom, sigma_base_corner=sigma_hom_base, n_sub=n_sub,
         sample_lon=model_data["sample_lon"], sample_lat=model_data["sample_lat"], positions_fingerprint=positions_key,
         analytic_G=analytic_cache, **{k: v for k, v in numbers_hom.items()},
         provenance=f"{notebook} {made}: analytic (CSI Meade, n_sub {n_sub}) tents of {cfg.gf_run}; settings of model_config")
stats = REG.sidecar("07_homogeneous", cfg.summary() + [f"noise model: {noise_model.iloc[0].drop('provenance').to_dict()}",
                                                         f"n_sub check: {nsub_table.to_dict('list')}",
                                                         f"sigma corner {sigma_hom:.3f} (base grid {sigma_hom_base:.3f})"]
                    + [f"{k} = {v}" for k, v in numbers_hom.items()])
where = "M8"
REG.register("07_n_samples", int(samples.vel.size), "count", 0, where, "[3]", stats)
REG.register("07_n_samples_fitted", int(keep.sum()), "count", 0, where, "[8]", stats)
REG.register("07_noise_nugget_mm", float(np.sqrt(cov_p[0])), "mm", 2, where, "[4]", stats)
REG.register("07_noise_sill_mm", float(np.sqrt(cov_p[1])), "mm", 2, where, "[4]", stats)
REG.register("07_noise_range_km", float(cov_p[2]), "km", 1, where, "[4]", stats)
REG.register("07_analytic_nsub", n_sub, "count", 0, where, "[7]", stats)
REG.register("07_hom_sigma", sigma_hom, "mm", 1, where, "[9]", stats)
REG.register("07_hom_mw_30gpa", numbers_hom["Mw_30gpa"], "Mw", 2, where, "[10]", stats)
REG.register("07_hom_potency_m3", numbers_hom["potency_m3"], "m3", 0, where, "[10]", stats)
REG.register("07_hom_peak_slip_mm", numbers_hom["peak_slip_mm"], "mm", 1, where, "[10]", stats)
REG.register("07_hom_peak_depth_m", numbers_hom["peak_depth_m"], "m", 0, where, "[10]", stats)
REG.register("07_hom_depth50_m", numbers_hom["depth50_m"], "m", 0, where, "[10]", stats)
REG.register("07_hom_depth90_m", numbers_hom["depth90_m"], "m", 0, where, "[10]", stats)
REG.register("07_hom_rms_mm", numbers_hom["rms_mm"], "mm", 2, where, "[10]", stats)
REG.register("07_hom_rms_west500_mm", numbers_hom["rms_west_mm"], "mm", 2, where, "[10]", stats)
REG.register("07_hom_rms_east500_mm", numbers_hom["rms_east_mm"], "mm", 2, where, "[10]", stats)
REG.render()
print(f"wrote {model_file}")
