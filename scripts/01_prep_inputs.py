# %% [markdown]
# # [1] 01 The prepared inputs: what this repo starts from
#
# ```
# Created on Mon Sep 21 2026
# @author: daniellelindsay
# Run in opera_disp_feb13. Seconds.
# ```
#
# **What this notebook does.** It rebuilds the 100 m fault crossings from the
# mapped trace and checks them against the `external_inputs/` copy every later
# notebook reads, then records what the analysis starts from and fingerprints
# the prepared products, the trace and the catalogues, so the registry can show
# which inputs produced every number. The preparation chain itself (download
# and subsetting of the OPERA DISP-S1 products, rebasing the mini-stacks to one
# reference date with `opera-utils`, the solid-earth-tide and ionosphere
# layers, the OPERA TROPO correction, MintPy conversion, masking, the velocity
# fit and the GNSS tie) lives in `data_prep/run00` to `run05` (ported from
# `Hayward_DISP-S1/analysis/`) and is described in Methods M1. Its products are what Zenodo will hold, so
# a reader can start at notebook 02 without re-running any of it.
#
# **Per frame** (09157 ascending track 35, 11115 descending track 42), in
# `<frame>/mintpy/`:
# - `timeseries_tropo.h5`: the displacement cube, tropo-corrected, unmasked,
#   referenced at the pixel co-located with GNSS station ZOA1 (m)
# - `maskTempCoh_07.h5`, `_09.h5`: temporal-coherence masks at 0.7 and 0.9
# - `velocity_masked.h5`: LOS velocity, coherence-0.7 masked, ZOA1 datum
# - `velocity_NA.h5`: the same after the GNSS tie (quadratic ramp per track, one
#   constant per frame), North America frame, zero at ZOA1
# - `geometryGeo.h5`: incidence and azimuth angles per pixel
#
# **Mosaics** (`mosaic_velfield/`): the two ascending frames merged, and the descending
# frame, as `<orbit>_velocity_NA.h5` and `<orbit>_geometry.h5` on one grid.
#
# **Inputs** (`external_inputs/`): the mapped fault trace and its 100 m crossings, the
# USGS creepmeter records, the USGS alignment-array archive v3.0, the Johnson
# et al. (2022) creep-rate compilation, the GHCN rain gauges, T. Taira's and
# Waldhauser and Schaff's repeating-earthquake catalogues, Bo Xie's GrowClust
# catalogue, and the frozen aperture sweep.
#
# **Outputs** (`outputs/01_prep_inputs/`): `hayward_fault_profiles.csv` (must equal
# the `external_inputs/` copy) and `01_inputs_stats.txt` (the fingerprints). No
# registry keys (the acquisition counts are registered by 01_prep_figures, per
# frame, and 03_creep_timeseries, per track). No figures (Figs S1 and S2 are
# 01_prep_figures).
#
# **What to check.** The crossings cell prints "identical to ...: True" (it
# asserts it); 791 crossings over 79.0 km.

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
# ## [3] Fault trace -> 100 m crossings
#
# The trace is `hayward_fault_trace_2026-07-24.kml`: 698 hand-placed crossing
# points along one continuous strand (plus one annotation placemark, dropped), re-mapped in Google Earth from the USGS
# Quaternary fault traces and the steps visible in the InSAR velocity fields
# (Dani, July 2026). The placemark names are the azimuths of an earlier
# version and are ignored; a placemark whose name is not a number is an
# annotation and is dropped.
#
# This cell turns the points into `hayward_fault_profiles.csv`, the file every
# later notebook reads (`ft_lon, ft_lat, profile_az`, south to north):
#
# 1. read the placemark coordinates
# 2. order them north to south as one polyline, in UTM 10N metres
# 3. resample every 100 m along the arc length
# 4. cross-fault azimuth = bearing of the local tangent - 90, i.e. pointing
#    NE to the EAST side (~54 deg), rounded to 0.1 deg
# 5. sort south to north
#
# Ported from `Hayward_DISP-S1/analysis/run07_fault_profiles.py`. The result
# is written next to the outputs and must equal the copy under `external_inputs/`;
# to re-map the trace, edit the KML, run this cell and copy the new csv over the
# `external_inputs/` one. The 100 m jitter of hand digitising is left in `profile_az`
# here; notebooks 02 and 03 smooth it with a 2.1 km rolling median.

# %%
# [4]
spacing_km = 0.1        # along-fault sample spacing
utm_epsg = 32610        # UTM 10N, the grid the frames live on

# 1) the placemark coordinates (the shared reader drops the annotation placemarks and prints them)
trace = utils.load_trace(common_paths["fault_trace"])

# 2) north -> south, one continuous polyline, in metres
trace = trace[np.argsort(-trace[:, 1])]
to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
to_ll = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
easting, northing = to_utm.transform(trace[:, 0], trace[:, 1])

# 3) resample every spacing_km along the arc length
step_m = np.hypot(np.diff(easting), np.diff(northing))
# ordering by latitude assumes the trace never doubles back: then the path is barely longer than end to end
# (1.006 for this trace; the few long steps, up to 1.6 km, are the sparsely digitised far south, south of 37.47 N)
path_ratio = step_m.sum() / np.hypot(easting[-1] - easting[0], northing[-1] - northing[0])
assert path_ratio < 1.05, f"trace path {path_ratio:.3f} x its end-to-end length: the latitude ordering broke the polyline"
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

# must equal the copy under external_inputs/ that the other notebooks read
inputs_copy = pd.read_csv(common_paths["fault_profiles"])
same = profiles.shape == inputs_copy.shape and np.allclose(profiles.values, inputs_copy.values, atol=1e-9)
print(f"identical to {common_paths['fault_profiles']}: {same}")
assert same, "the regenerated crossings differ from external_inputs/hayward_fault_profiles.csv: the trace or the code changed"
trace_note = f"   fault trace: {len(trace)} KML points -> {len(profiles)} crossings at {spacing_km} km; regenerated copy identical"

# %% [markdown]
# ## [5] Fingerprints of the prepared inputs
#
# The shape, size and fingerprint of every frame product, mask and mosaic, and
# the fingerprint of the trace and the catalogues, written to
# `01_inputs_stats.txt`, so each registry row can be traced to the files that
# made it.

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
