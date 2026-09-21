# %% [markdown]
# # 01 The prepared inputs: what this repo starts from
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
import os

import insar_utils as utils
from TulePond_filepaths import common_paths, insar_dic_list, paths_mosaic, paths_out, repo_dir

import h5py

notebook = "01_prep_inputs"
os.makedirs(paths_out[notebook], exist_ok=True)
REG = utils.Registry(notebook, common_paths, repo_dir)

# %%
L = ["01 PREPARED INPUTS (fingerprints)"]
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
