#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DRAFT (2026-09-28): elastic properties across the Tule Pond creep section
from the USGS San Francisco Bay region 3D seismic velocity model v21.1
(Aagaard & Hirakawa 2021, doi:10.5066/P9TRDCHE), read straight from the
GeoModelGrids HDF5 with h5py (the GeoModelGrids query tool is not on
conda-forge).

Coordinate handling, verified: the four documented WGS84 bounding-box
corners map to (0,0), (135,0), (135,285), (0,285) km in model x/y.
  - project lon/lat into the model CRS (attrs["crs"], a NAD83 transverse
    Mercator), subtract (origin_x, origin_y), rotate by y_azimuth
  - depth below the ground surface d -> logical z = -d (squashing of the
    top surface is negligible here: tens of m of topography over 45 km)
  - blocks: vres25m (0 to -500 m), vres50m (-500 to -3500), vres125m
    (-3500 to -10000), vres250m (-10000 to -45000); nearest grid point

Output: outputs/07_slip_model_tests/sfcvm_creep_section_profiles.csv
(km, offset_km, depth_m, rho, vp, vs, mu_gpa = rho vs^2 / 1e9, zone).
Run from the repo root in opera_disp_feb13 (PROJ_LIB pointed at the env).
"""
import json
import os

import h5py
import numpy as np
import pandas as pd
import pyproj

# ------------------------
# Paths & constants
# ------------------------
from TulePond_filepaths import common_paths, inputs_dir      # was hard-coded to the modelling laptop
import sys
version = sys.argv[1] if len(sys.argv) > 1 else "v21-1"     # "v21-1" (used so far) or "v26-0" (2026-08-14)
model_file = os.path.join(inputs_dir, f"USGS_SFCVM_{version}", f"USGS_SFCVM_{version}_detailed.h5")
assert os.path.exists(model_file), f"missing {model_file}"
profiles_file = common_paths["fault_profiles"]
out_name = "sfcvm_creep_section_profiles.csv" if version == "v21-1" else f"sfcvm_creep_section_profiles_{version}.csv"
out_file = os.path.join(common_paths["out_dir"], "07_slip_model_tests", out_name)
os.makedirs(os.path.dirname(out_file), exist_ok=True)

strike_deg = 143.05                         # local strike of the creep section (notebook 06)
km_along = np.arange(56.5, 62.31, 0.5)      # creep section, km from Point Pinole
offsets_km = (-3.0, -1.5, -0.5, -0.2, 0.2, 0.5, 1.5, 3.0)   # + = east (NE) of the trace
depths_m = (0, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000)

# ------------------------
# Load: model and its coordinate system
# ------------------------
model = h5py.File(model_file, "r")
attrs = model.attrs
crs_wkt = attrs["crs"] if isinstance(attrs["crs"], str) else attrs["crs"].decode()
to_model_crs = pyproj.Transformer.from_crs("EPSG:4326", pyproj.CRS.from_wkt(crs_wkt), always_xy=True)
origin_x, origin_y = float(attrs["origin_x"]), float(attrs["origin_y"])
y_azimuth = float(attrs["y_azimuth"])
zone_names = {int(v): k for k, v in json.loads(attrs["auxiliary"])["zone_ids"].items() if v > -1e10}
blocks = [("vres25m", 100, 25, 0.0), ("vres50m", 200, 50, -500.0),
          ("vres125m", 500, 125, -3500.0), ("vres250m", 1000, 250, -10000.0)]


def get_model_values(lon, lat, depth_m):
    """density, Vp, Vs, Qp, Qs, fault_block_id, zone_id at the nearest
    grid point (None outside the blocks)."""
    easting, northing = to_model_crs.transform(lon, lat)
    d_east, d_north = easting - origin_x, northing - origin_y
    y = d_east * np.sin(np.deg2rad(y_azimuth)) + d_north * np.cos(np.deg2rad(y_azimuth))
    x = d_east * np.sin(np.deg2rad(y_azimuth + 90)) + d_north * np.cos(np.deg2rad(y_azimuth + 90))
    z = -depth_m
    for name, h_res, dz, z_top in blocks:
        nz = model["blocks/" + name].shape[2]
        if z_top - (nz - 1) * dz - 1e-6 <= z <= z_top + 1e-6:
            ix, iy, iz = int(round(x / h_res)), int(round(y / h_res)), int(round((z_top - z) / dz))
            return model["blocks/" + name][ix, iy, iz, :]
    return None

# ------------------------
# Calculate: columns across the creep section
# ------------------------
trace = pd.read_csv(profiles_file)
trace_km = (len(trace) - 1 - np.arange(len(trace))) * 0.1
normal = np.deg2rad(strike_deg - 90)          # NE, the east side
rows = []
for km in km_along:
    i = int(np.argmin(np.abs(trace_km - km)))
    lat0, lon0 = trace.ft_lat[i], trace.ft_lon[i]
    for off in offsets_km:
        lat = lat0 + off * np.cos(normal) / 110.57
        lon = lon0 + off * np.sin(normal) / (111.32 * np.cos(np.deg2rad(lat0)))
        for depth in depths_m:
            values = get_model_values(lon, lat, depth)
            if values is None or values[2] <= 0:
                continue
            rows.append(dict(km=km, offset_km=off, depth_m=depth, rho=values[0],
                             vp=values[1], vs=values[2],
                             mu_gpa=values[0] * values[2] ** 2 / 1e9,
                             zone=zone_names.get(int(round(values[6])), "?")))
profiles = pd.DataFrame(rows)
profiles.to_csv(out_file, index=False)
print(f"wrote {out_file} ({len(profiles)} samples)")
