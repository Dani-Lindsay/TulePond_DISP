# %% [markdown]
# # 01 Prepared inputs
#
# ```
# Author: Danielle Lindsay
# Environment: opera_disp_feb13. Run time: seconds.
# ```
#
# **What this notebook does.**
#
# 1. **Fault crossings.** Resamples the hand-mapped Hayward Fault trace at equal
#    100 m spacing along the fault to define the crossings used by all later
#    notebooks, and checks that the result equals the archived copy that those
#    notebooks read (section 2).
# 2. **Fingerprints.** Records the shape, size and a fingerprint of every
#    prepared InSAR product, mask and mosaic, and fingerprints of the trace and
#    the catalogues, so that every registered number can be traced to the input
#    files that produced it (section 3).
#
# The preparation chain is in `data_prep/run00` to `run05` and is described in
# Methods M1. It downloads and subsets the OPERA DISP-S1 products (NASA/JPL/OPERA
# 2025; Staniewicz et al. 2026), rebases the mini-stacks to one reference date
# with `opera-utils` [citation needed], applies the solid-earth-tide and
# ionosphere layers and the OPERA TROPO correction, converts the stacks to
# MintPy time series (Yunjun et al. 2019), masks them, fits the velocities and
# ties them to GNSS. Its products are archived on Zenodo, so the analysis can
# start at notebook 02 without re-running the chain.
#
# **Inputs.** Per frame (09157 ascending track 35, 11115 descending track 42),
# in `<frame>/mintpy/`:
# - `timeseries_tropo.h5`: the displacement cube (m), troposphere-corrected,
#   unmasked, referenced at the pixel co-located with GNSS station ZOA1
# - `maskTempCoh_07.h5`, `maskTempCoh_09.h5`: temporal-coherence masks at 0.7
#   and 0.9
# - `velocity_masked.h5`: LOS velocity, masked at temporal coherence 0.7, zero
#   at ZOA1
# - `velocity_NA.h5`: the same after the GNSS tie (a quadratic ramp per track
#   and one constant per frame), in the North America reference frame, zero at
#   ZOA1
# - `geometryGeo.h5`: incidence and azimuth angles of every pixel
#
# Mosaics (`mosaic_velfield/`): the two ascending frames merged, and the
# descending frame, as `<orbit>_velocity_NA.h5` and `<orbit>_geometry.h5` on one
# grid.
#
# External inputs (`external_inputs/`): the hand-mapped fault trace and its
# archived 100 m crossings, the USGS creepmeter records (Langbein et al. 2024),
# the USGS alignment-array archive v3.0 (McFarland et al. 2017), the Johnson et
# al. (2022) creep-rate compilation, the GHCN-Daily rain gauges (Menne et al.
# 2012), the repeating-earthquake catalogues of T. Taira (BSL, GEN3) and of
# Waldhauser and Schaff, the GrowClust relocated catalogue of B. Xie, and the
# archived aperture sweep.
#
# **Outputs** (`outputs/01_prep_inputs/`): `hayward_fault_profiles.csv` (must
# equal the archived copy) and `01_inputs_stats.txt` (the fingerprints). This
# notebook registers no numbers. The acquisition counts are registered per
# frame by 01_prep_figures and per track by 03_creep_timeseries.
#
# **Figures.** None. Figs. S1 and S2 are drawn by 01_prep_figures.
#
# **What to check.** The line printed in section 2 gives the number of trace
# points and crossings, the trace length and the ranges of latitude and
# cross-fault azimuth. Section 2 asserts that the regenerated crossings equal
# the archived copy. Section 3 prints the fingerprint table.

# %% [markdown]
# ## 1. Imports and settings

# %%
import os

import insar_utils as utils
from TulePond_filepaths import common_paths, insar_dic_list, paths_mosaic, paths_out, repo_dir

import numpy as np                 # Matrix calculations
import pandas as pd                # Tables
import h5py                        # MintPy HDF5 products
from pyproj import Transformer     # lon/lat <-> UTM

notebook = "01_prep_inputs"
os.makedirs(paths_out[notebook], exist_ok=True)
REG = utils.Registry(notebook, common_paths, repo_dir)
REG.inputs(common_paths["fault_trace"], common_paths["fault_profiles"])

# %% [markdown]
# ## 2. Fault trace to 100 m crossings
#
# The Hayward Fault trace was re-mapped by hand on the InSAR fault-parallel
# velocity field, following the USGS Quaternary fault traces (U.S. Geological
# Survey and California Geological Survey 2019) and the steps visible in the
# velocities. The hand-drawn line is stored as points in
# `hayward_fault_trace_2026-07-24.kml` (`common_paths["fault_trace"]`). A
# placemark whose name is not a number is an annotation; the shared reader drops
# it and prints its name.
#
# This section resamples the hand-drawn line at equal 100 m spacing along the
# fault. Each resampled point is one crossing. The steps are:
#
# 1. Read the point coordinates.
# 2. Order the points from north to south and project them to UTM zone 10N
#    (EPSG 32610), the grid of the frames.
# 3. Resample the line every $\Delta s$ = 100 m of arc length.
# 4. Compute the cross-fault azimuth of each crossing from the local tangent.
# 5. Sort the crossings from south to north and write `ft_lon, ft_lat,
#    profile_az`.
#
# With $(E_j, N_j)$ the UTM easting and northing (m) of point $j$, ordered from
# north to south, the arc length is
#
# $$ s_0 = 0, \qquad s_j = \sum_{i=1}^{j} \sqrt{(E_i - E_{i-1})^2 + (N_i - N_{i-1})^2}, $$
#
# and the crossings lie at $s = k\,\Delta s$ ($k = 0, 1, 2, \dots$, up to the
# end of the line), with $E$ and $N$ linearly interpolated in $s$. Ordering by
# latitude gives a continuous line only if the trace does not double back, so
# the path length must be less than 1.05 times the straight-line distance
# between the two end points (asserted).
#
# The cross-fault azimuth of crossing $k$ is
#
# $$ \phi_k = \left[\operatorname{atan2}(\delta E_k, \delta N_k) - 90^\circ\right] \bmod 360^\circ, $$
#
# where $\delta E_k$ and $\delta N_k$ are the central differences of the
# resampled easting and northing (one-sided at the two ends; `np.gradient`).
# The atan2 term is the compass bearing of the local tangent (clockwise from
# north). Because the points run from north to south, the tangent points along
# strike to the south-east, and subtracting 90° turns it to the north-east,
# toward the east side of the fault. $\phi_k$ is stored as `profile_az`, rounded
# to 0.1°. It keeps the jitter of hand digitising at 100 m spacing; notebooks
# 02, 03 and 06 smooth it with a 21-crossing (2.1 km) rolling median.
#
# **Check.** The regenerated crossings are compared with the archived copy
# `external_inputs/hayward_fault_profiles.csv` (`common_paths["fault_profiles"]`),
# which is the file that every other notebook reads. Both come from the same
# hand-drawn trace; neither is the USGS trace. The two tables must have the same
# shape and agree to 1e-9 in every column (asserted). To re-map the trace, edit
# the KML, run this section and copy the new csv over the archived copy.

# %%
spacing_km = 0.1        # along-fault sample spacing (km)
utm_epsg = 32610        # UTM zone 10N, the grid of the frames

# 1) the point coordinates (the shared reader drops and prints the annotation placemarks)
trace = utils.load_trace(common_paths["fault_trace"])

# 2) north to south, one continuous line, in metres
trace = trace[np.argsort(-trace[:, 1])]
to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
to_ll = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
easting, northing = to_utm.transform(trace[:, 0], trace[:, 1])

# 3) resample every spacing_km along the arc length
step_m = np.hypot(np.diff(easting), np.diff(northing))
# Check: the latitude ordering is valid only if the trace does not double back,
# in which case the path length is close to the end-to-end length
path_ratio = step_m.sum() / np.hypot(easting[-1] - easting[0], northing[-1] - northing[0])
assert path_ratio < 1.05, f"trace path is {path_ratio:.3f} x its end-to-end length: the latitude ordering does not give a continuous line"
arc_m = np.r_[0.0, np.cumsum(step_m)]
arc_new_m = np.arange(0.0, arc_m[-1], spacing_km * 1000.0)
easting_new = np.interp(arc_new_m, arc_m, easting)
northing_new = np.interp(arc_new_m, arc_m, northing)

# 4) cross-fault azimuth: bearing of the local tangent (clockwise from north) minus 90, toward the east side
d_easting, d_northing = np.gradient(easting_new), np.gradient(northing_new)
cross_az = (np.degrees(np.arctan2(d_easting, d_northing)) - 90.0) % 360.0
lon_new, lat_new = to_ll.transform(easting_new, northing_new)
profiles = pd.DataFrame({"ft_lon": lon_new, "ft_lat": lat_new, "profile_az": np.round(cross_az, 1)})

# 5) south to north, and save
profiles = profiles.sort_values("ft_lat").reset_index(drop=True)
profiles_file = os.path.join(paths_out[notebook], "hayward_fault_profiles.csv")
profiles.to_csv(profiles_file, index=False)
print(f"{len(trace)} trace points -> {len(profiles)} crossings every {spacing_km} km, "
      f"{arc_m[-1] / 1000:.1f} km, lat {profiles.ft_lat.min():.4f} to {profiles.ft_lat.max():.4f}, "
      f"profile_az {profiles.profile_az.min():.1f} to {profiles.profile_az.max():.1f} deg")
print("Saved crossings to:", profiles_file)

# Check: the result must equal the archived copy under external_inputs/ that the other notebooks read
inputs_copy = pd.read_csv(common_paths["fault_profiles"])
same = profiles.shape == inputs_copy.shape and np.allclose(profiles.values, inputs_copy.values, atol=1e-9)
print(f"identical to {common_paths['fault_profiles']}: {same}")
assert same, "the regenerated crossings differ from the archived external_inputs/hayward_fault_profiles.csv: the trace or the code has changed"
trace_note = f"   fault trace: {len(trace)} KML points -> {len(profiles)} crossings at {spacing_km} km; regenerated copy identical"

# %% [markdown]
# ## 3. Fingerprints of the prepared inputs
#
# For every frame product, mask and mosaic, and for the trace, the crossings
# and the catalogues, the main dataset shape, the file size and a fingerprint
# are written to `01_inputs_stats.txt`, so that each registry row can be traced
# to its input files. The fingerprint (`utils.file_fingerprint`) is the first
# 12 hexadecimal characters of the SHA-256 hash of the file. Files larger than
# 256 MiB are hashed on their size and their first and last 4 MiB only, and the
# fingerprint is then prefixed with `~`.

# %%
dataset_of = {"ts_file": "timeseries", "vel_file": "velocity", "vel_NA_file": "velocity", "geo_file": "height"}   # the main dataset of each product
lines = ["01 PREPARED INPUTS (fingerprints)", trace_note]
for dic in insar_dic_list:
    lines.append(f"{dic['ORBIT_DIRECTION']} frame {dic['Frame']} (track {dic['PATH']})")
    for key in ("ts_file", "vel_file", "vel_NA_file", "geo_file"):
        with h5py.File(dic[key], "r") as h:
            shape = h[dataset_of[key]].shape
        size_gb = os.path.getsize(dic[key]) / 1e9
        lines.append(f"   {os.path.basename(dic[key]):22s} {str(shape):22s} {size_gb:5.2f} GB  sha {utils.file_fingerprint(dic[key])}")
    for coherence, mask_path in dic["mask_file"].items():
        lines.append(f"   {os.path.basename(mask_path):22s} coherence {coherence}                  sha {utils.file_fingerprint(mask_path)}")
for orbit, mosaic_paths in paths_mosaic.items():
    for key, path in mosaic_paths.items():
        lines.append(f"   mosaic {orbit} {key:12s} sha {utils.file_fingerprint(path)}")
for key in ("fault_trace", "fault_profiles", "creepmeter_locations", "aa_summary", "creep_rates_Johnson22", "CRE_Taira", "CRE_Wald", "growclust_cat"):
    lines.append(f"   {key:22s} sha {utils.file_fingerprint(common_paths[key])}")
print("\n".join(lines))
REG.sidecar("01_inputs", lines)
REG.render()
