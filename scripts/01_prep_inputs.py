# %% [markdown]
# # [1] 01 The prepared inputs: what this repo starts from
#
# ```
# Created on Mon Sep 21 2026
# @author: daniellelindsay
# Run in opera_disp_feb13. Seconds.
# ```
#
# **What this notebook does.** Nothing is computed here. It records what the
# analysis starts from and fingerprints every file so the registry can show
# which inputs produced every number. The preparation chain itself (download
# and subsetting of the OPERA DISP-S1 products, rebasing the mini-stacks to one
# reference date with `opera-utils`, the solid-earth-tide and ionosphere
# layers, the OPERA TROPO correction, MintPy conversion, masking, the velocity
# fit and the GNSS tie) lives in `Hayward_DISP-S1/analysis/run00` to `run05`
# and is described in Methods M1. Its products are what Zenodo will hold, so
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
# **Mosaics** (`mosaic/`): the two ascending frames merged, and the descending
# frame, as `<orbit>_velocity_NA.h5` and `<orbit>_geometry.h5` on one grid.
#
# **Inputs** (`Inputs/`): the mapped fault trace and its 100 m crossings, the
# USGS creepmeter records, the USGS alignment-array archive v3.0, the Johnson
# et al. (2022) creep-rate compilation, the GHCN rain gauges, T. Taira's and
# Waldhauser and Schaff's repeating-earthquake catalogues, Bo Xie's GrowClust
# catalogue, and the frozen aperture sweep.

# %%
# [2]
import os
import re
import xml.etree.ElementTree as ET

import insar_utils as utils
from TulePond_filepaths import common_paths, insar_dic_list, paths_mosaic, paths_out, repo_dir

import numpy as np                 # Matrix calculations
import pandas as pd                # Tables
import h5py                        # MintPy HDF5 products
from pyproj import Transformer     # lon/lat <-> UTM

notebook = "01_prep_inputs"
os.makedirs(paths_out[notebook], exist_ok=True)
REG = utils.Registry(notebook, common_paths, repo_dir)

# %% [markdown]
# ## [3] Fault trace -> 100 m crossings
#
# The trace is `hayward_fault_trace_2026-07-24.kml`: 699 hand-placed crossing
# points along one continuous strand, re-mapped in Google Earth from the USGS
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
# is written next to the outputs and must equal the copy under `Inputs/`; to
# re-map the trace, edit the KML, run this cell and copy the new csv over the
# `Inputs/` one. The 100 m jitter of hand digitising is left in `profile_az`
# here; notebooks 02 and 03 smooth it with a 2.1 km rolling median.

# %%
# [4]
spacing_km = 0.1        # along-fault sample spacing
utm_epsg = 32610        # UTM 10N, the grid the frames live on

# 1) the placemark coordinates
kml = re.sub(r'xmlns="[^"]+"', "", open(common_paths["fault_trace"]).read())
trace = []
for placemark in ET.fromstring(kml).findall(".//Placemark"):
    if placemark.find(".//Point") is None:
        continue
    name = placemark.findtext("name")
    try:
        float(name)
    except (TypeError, ValueError):
        print(f"dropped annotation placemark '{name}'")
        continue
    lon, lat = map(float, placemark.find(".//coordinates").text.strip().split(",")[:2])
    trace.append((lon, lat))
trace = np.array(trace)

# 2) north -> south, one continuous polyline, in metres
trace = trace[np.argsort(-trace[:, 1])]
to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
to_ll = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
X, Y = to_utm.transform(trace[:, 0], trace[:, 1])

# 3) resample every spacing_km along the arc length
s = np.r_[0.0, np.cumsum(np.hypot(np.diff(X), np.diff(Y)))]
s_new = np.arange(0.0, s[-1], spacing_km * 1000.0)
X_new, Y_new = np.interp(s_new, s, X), np.interp(s_new, s, Y)

# 4) cross-fault azimuth: bearing of the local tangent (clockwise from north) - 90, pointing to the east side
dX, dY = np.gradient(X_new), np.gradient(Y_new)
cross_az = (np.degrees(np.arctan2(dX, dY)) - 90.0) % 360.0
lon_new, lat_new = to_ll.transform(X_new, Y_new)
profiles = pd.DataFrame({"ft_lon": lon_new, "ft_lat": lat_new, "profile_az": np.round(cross_az, 1)})

# 5) south -> north, and save
profiles = profiles.sort_values("ft_lat").reset_index(drop=True)
profiles_file = os.path.join(paths_out[notebook], "hayward_fault_profiles.csv")
profiles.to_csv(profiles_file, index=False)
print(f"{len(trace)} trace points -> {len(profiles)} crossings every {spacing_km} km, "
      f"{s[-1] / 1000:.1f} km, lat {profiles.ft_lat.min():.4f} to {profiles.ft_lat.max():.4f}, "
      f"profile_az {profiles.profile_az.min():.1f} to {profiles.profile_az.max():.1f} deg")
print("Saved crossings to:", profiles_file)

# must equal the copy under Inputs/ that the other notebooks read
inputs_copy = pd.read_csv(common_paths["fault_profiles"])
same = profiles.shape == inputs_copy.shape and np.allclose(profiles.values, inputs_copy.values, atol=1e-9)
print(f"identical to {common_paths['fault_profiles']}: {same}")
assert same, "the regenerated crossings differ from Inputs/hayward_fault_profiles.csv: the trace or the code changed"
trace_note = f"   fault trace: {len(trace)} KML points -> {len(profiles)} crossings at {spacing_km} km; regenerated copy identical"

# %%
# [5]
L = ["01 PREPARED INPUTS (fingerprints)", trace_note]
for dic in insar_dic_list:
    L.append(f"{dic['ORBIT_DIRECTION']} frame {dic['Frame']} (track {dic['PATH']})")
    for key in ("ts_file", "vel_file", "vel_NA_file", "geo_file"):
        with h5py.File(dic[key], "r") as h:
            a = h.attrs
            shape = h["timeseries"].shape if "timeseries" in h else h[list(h.keys())[0]].shape
            L.append(f"   {os.path.basename(dic[key]):22s} {str(shape):22s} {os.path.getsize(dic[key]) / 1e9:5.2f} GB  sha {utils.file_fingerprint(dic[key])}")
    for c, m in dic["mask_file"].items():
        L.append(f"   {os.path.basename(m):22s} coherence {c}                  sha {utils.file_fingerprint(m)}")
    with h5py.File(dic["ts_file"], "r") as h:
        dates = [d.decode() for d in h["date"][:]]
        REG.register(f"01_n_acquisitions_{dic['ORBIT_DIRECTION'][:3]}", len(dates), "count", 0, where="M1", cell="fingerprints")
        REG.register(f"01_first_date_{dic['ORBIT_DIRECTION'][:3]}", dates[0], "date", 0, where="M1", cell="fingerprints")
        REG.register(f"01_last_date_{dic['ORBIT_DIRECTION'][:3]}", dates[-1], "date", 0, where="M1", cell="fingerprints")
for orbit, d in paths_mosaic.items():
    for key, path in d.items():
        L.append(f"   mosaic {orbit} {key:12s} sha {utils.file_fingerprint(path)}")
for key in ("fault_profiles", "creepmeter_locations", "aa_summary", "creep_rates_Johnson22", "CRE_Taira", "CRE_Wald", "growclust_cat"):
    L.append(f"   {key:22s} sha {utils.file_fingerprint(common_paths[key])}")
print("\n".join(L))
REG.sidecar("01_inputs", L)
REG.render()
