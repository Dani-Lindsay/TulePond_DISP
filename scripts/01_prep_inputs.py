# %% [markdown]
# # [1] 01 Prepared inputs
#
# ```
# Author: Danielle Lindsay
# Environment: opera_disp_feb13. Run time: seconds.
# ```
#
# **Purpose.** This notebook regenerates the 100 m fault crossings from the
# mapped trace and checks them against the `external_inputs/` copy read by all
# later notebooks. It then records the inputs of the analysis and computes
# fingerprints of the prepared products, the fault trace and the catalogues, so
# that every registered number can be traced to the inputs that produced it.
# The preparation chain (download and subsetting of the OPERA DISP-S1 products,
# rebasing of the mini-stacks to one reference date with `opera-utils`, the
# solid-earth-tide and ionosphere layers, the OPERA TROPO correction, MintPy
# conversion, masking, the velocity fit and the GNSS tie) is in `data_prep/run00`
# to `run05` and is described in Methods M1. Its products are archived on
# Zenodo, so the analysis can be started at notebook 02 without re-running the
# preparation chain.
#
# **Per frame** (09157 ascending track 35, 11115 descending track 42), in
# `<frame>/mintpy/`:
# - `timeseries_tropo.h5`: the displacement cube, troposphere-corrected,
#   unmasked, referenced at the pixel co-located with GNSS station ZOA1 (m)
# - `maskTempCoh_07.h5`, `_09.h5`: temporal-coherence masks at 0.7 and 0.9
# - `velocity_masked.h5`: LOS velocity, masked at temporal coherence 0.7, ZOA1 datum
# - `velocity_NA.h5`: the same after the GNSS tie (quadratic ramp per track, one
#   constant per frame), North America reference frame, zero at ZOA1
# - `geometryGeo.h5`: incidence and azimuth angles per pixel
#
# **Mosaics** (`mosaic_velfield/`): the two ascending frames merged, and the
# descending frame, as `<orbit>_velocity_NA.h5` and `<orbit>_geometry.h5` on one grid.
#
# **Inputs** (`external_inputs/`): the mapped fault trace and its 100 m
# crossings, the USGS creepmeter records, the USGS alignment-array archive v3.0,
# the Johnson et al. (2022) creep-rate compilation, the GHCN rain gauges, the
# repeating-earthquake catalogues of T. Taira (BSL, GEN3) and of Waldhauser and
# Schaff, the GrowClust relocated catalogue of B. Xie, and the archived
# aperture sweep.
#
# **Outputs** (`outputs/01_prep_inputs/`): `hayward_fault_profiles.csv` (must
# equal the `external_inputs/` copy) and `01_inputs_stats.txt` (the
# fingerprints). This notebook registers no numbers (the acquisition counts are
# registered per frame by 01_prep_figures and per track by 03_creep_timeseries)
# and draws no figures (Figs. S1 and S2 are drawn by 01_prep_figures).
#
# **Check.** The crossings cell asserts that the regenerated crossings are
# identical to the `external_inputs/` copy and prints the number of crossings
# and the trace length.

# %%
# [2]
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
# ## [3] Fault trace to 100 m crossings
#
# The trace is `hayward_fault_trace_2026-07-24.kml`: hand-placed points along
# one continuous strand, mapped in Google Earth from the USGS Quaternary fault
# traces and the steps visible in the InSAR velocity fields. The placemark names
# are not used; a placemark whose name is not a number is an annotation and is
# dropped.
#
# This cell converts the points into `hayward_fault_profiles.csv`, the file read
# by all later notebooks (`ft_lon, ft_lat, profile_az`, south to north):
#
# 1. read the placemark coordinates
# 2. order them north to south as one polyline, in UTM zone 10N metres
# 3. resample every 100 m along the arc length
# 4. cross-fault azimuth = bearing of the local tangent - 90 deg, pointing NE
#    to the east side of the fault, rounded to 0.1 deg
# 5. sort south to north
#
# The result is written to the output directory and must equal the copy under
# `external_inputs/`. To re-map the trace, edit the KML, run this cell and copy
# the new csv over the `external_inputs/` copy. The digitising jitter at 100 m
# spacing is retained in `profile_az`; notebooks 02 and 03 smooth it with a
# 2.1 km rolling median.

# %%
# [4]
spacing_km = 0.1        # along-fault sample spacing
utm_epsg = 32610        # UTM zone 10N, the grid of the frames

# 1) the placemark coordinates (the shared reader drops and prints the annotation placemarks)
trace = utils.load_trace(common_paths["fault_trace"])

# 2) north -> south, one continuous polyline, in metres
trace = trace[np.argsort(-trace[:, 1])]
to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
to_ll = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
easting, northing = to_utm.transform(trace[:, 0], trace[:, 1])

# 3) resample every spacing_km along the arc length
step_m = np.hypot(np.diff(easting), np.diff(northing))
# Check: ordering by latitude assumes the trace does not double back, in which case the path length
# is close to the end-to-end length
path_ratio = step_m.sum() / np.hypot(easting[-1] - easting[0], northing[-1] - northing[0])
assert path_ratio < 1.05, f"trace path is {path_ratio:.3f} x its end-to-end length: the latitude ordering does not give a continuous polyline"
arc_m = np.r_[0.0, np.cumsum(step_m)]
arc_new_m = np.arange(0.0, arc_m[-1], spacing_km * 1000.0)
easting_new = np.interp(arc_new_m, arc_m, easting)
northing_new = np.interp(arc_new_m, arc_m, northing)

# 4) cross-fault azimuth: bearing of the local tangent (clockwise from north) - 90, pointing to the east side
d_easting, d_northing = np.gradient(easting_new), np.gradient(northing_new)
cross_az = (np.degrees(np.arctan2(d_easting, d_northing)) - 90.0) % 360.0
lon_new, lat_new = to_ll.transform(easting_new, northing_new)
profiles = pd.DataFrame({"ft_lon": lon_new, "ft_lat": lat_new, "profile_az": np.round(cross_az, 1)})

# 5) south -> north, and save
profiles = profiles.sort_values("ft_lat").reset_index(drop=True)
profiles_file = os.path.join(paths_out[notebook], "hayward_fault_profiles.csv")
profiles.to_csv(profiles_file, index=False)
print(f"{len(trace)} trace points -> {len(profiles)} crossings every {spacing_km} km, "
      f"{arc_m[-1] / 1000:.1f} km, lat {profiles.ft_lat.min():.4f} to {profiles.ft_lat.max():.4f}, "
      f"profile_az {profiles.profile_az.min():.1f} to {profiles.profile_az.max():.1f} deg")
print("Saved crossings to:", profiles_file)

# Check: the result must equal the copy under external_inputs/ read by the other notebooks
inputs_copy = pd.read_csv(common_paths["fault_profiles"])
same = profiles.shape == inputs_copy.shape and np.allclose(profiles.values, inputs_copy.values, atol=1e-9)
print(f"identical to {common_paths['fault_profiles']}: {same}")
assert same, "the regenerated crossings differ from external_inputs/hayward_fault_profiles.csv: the trace or the code has changed"
trace_note = f"   fault trace: {len(trace)} KML points -> {len(profiles)} crossings at {spacing_km} km; regenerated copy identical"

# %% [markdown]
# ## [5] Fingerprints of the prepared inputs
#
# The shape, size and fingerprint of every frame product, mask and mosaic, and
# the fingerprints of the trace and the catalogues, are written to
# `01_inputs_stats.txt`, so that each registry row can be traced to its input
# files.

# %%
# [6]
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
