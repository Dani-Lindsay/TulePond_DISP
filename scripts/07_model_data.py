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
# # 07 Model data
#
# ```
# Author: Danielle Lindsay
# Environment: environment.yml. Runs after 04 and 06. About 5 min.
# ```
#
# **What this notebook does**
#
# 1. **Samples** (sections 2 and 3). The five-event mean fault-parallel field of notebook 06 is sampled with a quadtree.
#    The samples are also the output points of the PyLith runs.
# 2. **Noise model** (section 4). A semivariogram of the pixels more than 5 km from the trace.
# 3. **Coherence-0.7 sample set** (section 5). The same field at temporal coherence 0.7, for the coherence sensitivity
#    test.
# 4. **Products** (section 6). The PyLith point lists and `model_data.npz`, read by notebooks 08, 09 and 10 and by the
#    scripts in `scripts/sensitivity/`.
#
# The settings are in `model_config.py` and the modelling functions in `model_utils.py`.
#
# **Meshes and PyLith Green's functions.** No notebook builds a mesh or runs PyLith. The meshes are built by
# `pylith/build/generate_gmsh_vertical.py` and `pylith/build/generate_gmsh_dip.py`, and the Green's functions are
# computed by `pylith/run_gf.sh`. Both are run by hand, as documented in `pylith/README.md`. This notebook writes the
# point lists that the PyLith runs use as output points (section 6).
#
# **Inputs**: notebook 06 `fp_maps.npz` (the field) and `tulepond_steps_{ascending,descending}.h5` (for the
# coherence-0.7 field); notebook 04 `section_lengths.csv`; the trace KML and `hayward_fault_profiles.csv`; the
# temporal-coherence 0.9 masks.
#
# **Outputs** (`outputs/07_model_data/`): `model_data.npz` (the samples, the noise model, the trace),
# `noise_model_5km.csv` and `semivariogram_bins_5km.csv` (the noise model), `07_model_data_stats.txt`; the point lists
# `modeling_inputs/points/insar_points{,_coh07}.{txt,npz}` (positions asserted unchanged). **Registry keys** `07_*`.
#
# **What to check**
# - Section 3: the check that the quadtree samples are the PyLith output points (same positions, same data) passes.
# - Section 4: the printed noise fit. The sill of the second exponential term should be near zero.
# - Section 5: the check that the coherence-0.9 field rebuilt from 06's saved steps equals 06's field passes.

# %% [markdown]
# ## 1. Imports, settings and the number registry
#
# The settings are printed from `model_config.summary()` so that the executed notebook records them.

# %%
import os
import time
import warnings

import numpy as np
import pandas as pd

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
           os.path.join(paths_out["04_event_detection"], "section_lengths.csv"))
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
# sample has the variance $n^2$ (equal weights; notebook 08, section 4). The full covariance with $p = 0.5$ is used for
# the Monte Carlo noise realisations of notebook 09 and for the full-covariance sensitivity test.

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
                   provenance=f"{notebook} {made}: quadtree of model_utils.downsample; noise model = the 5 km fit of section 4 (both sets)"))
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
# ## 7. Registry
#
# The numbers of the paper from this notebook, with the stats file `07_model_data_stats.txt`.

# %%
stats = REG.sidecar("07_model_data", cfg.summary() + [f"noise model: {noise_model.iloc[0].drop('provenance').to_dict()}",
                                                     f"samples: {quadtree.vel.size} ({n_pixels} pixels); coherence-0.7 set: "
                                                     f"{quadtree_coh07.vel.size}"])
where = "M8"
REG.register("07_n_samples", int(quadtree.vel.size), "count", 0, where, "section 3", stats)
REG.register("07_noise_nugget_mm", float(np.sqrt(cov_p[0])), "mm", 2, where, "section 4", stats)
REG.register("07_noise_sill_mm", float(np.sqrt(cov_p[1])), "mm", 2, where, "section 4", stats)
REG.register("07_noise_range_km", float(cov_p[2]), "km", 1, where, "section 4", stats)
REG.render()
