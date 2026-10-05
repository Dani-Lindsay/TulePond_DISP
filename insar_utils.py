#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 2026

@author: daniellelindsay

Shared helper functions for the TulePond_DISP publication repository.

This file owns HOW the shared maths is done (NC_ALOS-2 layout): reading and
writing MintPy HDF5 products, dates, the LOS projection conventions, the
along-fault distance axis, robust statistics, creepmeter readers, and the
number registry that every notebook appends its manuscript numbers to.

Only functions that MORE THAN ONE notebook uses live here. Anything that one
notebook does on its own is written out in that notebook so it can be read
top to bottom. Every function here is short and has a test in tests/.

Sections
    run_command          run a MintPy / GMT command-line tool and print it
    Date-time            decimal years
    Read-Write           h5py readers and writers for MintPy files
    Projections          LOS unit vector, fault-parallel design matrix, decomposition
    Along-fault axis     km from Point Pinole along the mapped trace; the map strike frame (u, v)
    Statistics           robust sigma, detrend, linear rate with std_err
    Creepmeters          USGS .10min reader; the daily glitch-filtered record
    Provenance/Registry  git hash, file fingerprints, the registry class
"""
import csv
import hashlib
import os
import re
import sys
import xml.etree.ElementTree as ET
import shutil
import subprocess
import uuid
from datetime import datetime

import numpy as np                 # Matrix calculations
import pandas as pd                # Tables
import h5py                        # MintPy HDF5 products
from scipy import stats            # linregress
from scipy.ndimage import gaussian_filter1d      # gaussian_smooth_km
import geopy.distance              # geodesic distance along the fault trace
import pyproj                      # lon/lat <-> UTM for the strike frame (strike_frame)


def run_command(cmd):
    """Run a MintPy or GMT command-line tool from THIS python environment and
    print the command first, so the notebook output records exactly what ran.

    The tool is looked up next to the running python (the env's bin/), which
    avoids the wrong-environment problem we hit with `conda activate` in
    non-interactive shells.
    """
    import sys
    env_bin = os.path.dirname(sys.executable)
    exe = os.path.join(env_bin, cmd[0])
    if os.path.exists(exe):
        cmd = [exe] + list(cmd[1:])
    print("Running:", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


# ------------------------------
# Date-time
# ------------------------------

RESERVE_GB = 50.0      # never plan to eat into the last this-many GB of disk


def require_disk_space(path, need_gb, what=""):
    """Abort unless `path` has room for `need_gb` plus a safety reserve.

    A download that fills the disk does not just fail, it takes the machine
    down with it, so the big-download scripts in `data_prep/` check space up
    front and refuse to start rather than dying halfway through. `need_gb` is a
    rough estimate; the check catches "this was never going to fit", not the
    last megabyte.
    """
    total, used, free = (x / 1e9 for x in shutil.disk_usage(path))
    label = f" for {what}" if what else ""
    print(f"  disk: {free:.0f} GB free of {total:.0f} GB; "
          f"need ~{need_gb:.1f} GB{label} (+{RESERVE_GB:.0f} GB reserve)")
    if free - need_gb >= RESERVE_GB:
        return True
    raise SystemExit(f"ABORT: insufficient disk space{label}: {free:.0f} GB free, "
                     f"need ~{need_gb:.1f} GB plus a {RESERVE_GB:.0f} GB reserve.")


def date_to_decimal_year(date_input, format_str=None):
    """Convert a date (str, bytes, or datetime) to a decimal year.

    Uses the exact length of the year in seconds (leap years handled), so a
    date and its decimal year round-trip to the second. MintPy stores dates as
    bytes b'20160717'; those are accepted directly.
    """
    if isinstance(date_input, bytes):
        date_input = date_input.decode("utf-8")

    if isinstance(date_input, datetime):
        date = date_input
    else:
        if format_str is None:
            if "-" in date_input:
                format_str = "%Y-%m-%d"
            elif "/" in date_input:
                format_str = "%m/%d/%Y"
            elif len(date_input) == 8:
                format_str = "%Y%m%d"
            else:
                raise ValueError("Unknown date format. Please provide a format string.")
        date = datetime.strptime(date_input, format_str)

    start_of_year = datetime(year=date.year, month=1, day=1)
    start_of_next_year = datetime(year=date.year + 1, month=1, day=1)
    year_length = (start_of_next_year - start_of_year).total_seconds()
    year_progress = (date - start_of_year).total_seconds()
    return date.year + year_progress / year_length


def decimal_year_to_date(dec_year):
    """Inverse of date_to_decimal_year, to the nearest second."""
    year = int(np.floor(dec_year))
    start_of_year = datetime(year=year, month=1, day=1)
    start_of_next_year = datetime(year=year + 1, month=1, day=1)
    seconds = (dec_year - year) * (start_of_next_year - start_of_year).total_seconds()
    return start_of_year + pd.Timedelta(seconds=round(seconds))


def decimal_year_to_day(dec_year):
    """The calendar day nearest to a decimal year, as a Timestamp at midnight.

    Acquisition dates are whole days but their decimal years are stored rounded (2017.0986 is
    6 Feb 2017 less 16 minutes), so cutting the time off would give the day before; rounding gives
    the acquisition day."""
    return pd.Timestamp(decimal_year_to_date(dec_year)).round("D")


# ------------------------------
# Read-Write (MintPy HDF5)
# ------------------------------

def load_h5_attrs(h5_file):
    """Return the MintPy attributes of a file as a plain dict of strings."""
    with h5py.File(h5_file, "r") as hf:
        return {k: (v.decode() if isinstance(v, bytes) else str(v)) for k, v in hf.attrs.items()}


def load_h5_data(h5_file, dataset, to_mm=False):
    """Read ONE 2-D dataset from a MintPy file as float64, with zeros set to
    NaN (MintPy writes 0 for no-data). `to_mm=True` multiplies by 1000.

    Returns (array, attrs).
    """
    with h5py.File(h5_file, "r") as hf:
        arr = np.array(hf[dataset][:]).astype(np.float64)
        attrs = {k: (v.decode() if isinstance(v, bytes) else str(v)) for k, v in hf.attrs.items()}
    arr[arr == 0] = np.nan
    if to_mm:
        arr = arr * 1000.0
    return arr, attrs


def load_geometry(geo_file):
    """Read incidenceAngle and azimuthAngle (degrees) from geometryGeo.h5 as
    float64. Zeros (no-data) become NaN."""
    inc, _ = load_h5_data(geo_file, "incidenceAngle")
    azi, _ = load_h5_data(geo_file, "azimuthAngle")
    return inc, azi


def grid_coords(attrs):
    """Pixel-centre X and Y coordinate arrays (metres, EPSG in attrs) from the
    MintPy attributes X_FIRST/Y_FIRST/X_STEP/Y_STEP/WIDTH/LENGTH."""
    x0, y0 = float(attrs["X_FIRST"]), float(attrs["Y_FIRST"])
    dx, dy = float(attrs["X_STEP"]), float(attrs["Y_STEP"])
    nx, ny = int(attrs["WIDTH"]), int(attrs["LENGTH"])
    x = x0 + dx * np.arange(nx)
    y = y0 + dy * np.arange(ny)
    return x, y


def lonlat_grid(attrs):
    """2-D longitude and latitude arrays for a UTM-gridded MintPy product."""
    import pyproj
    epsg = int(attrs.get("EPSG", 32610))
    x, y = grid_coords(attrs)
    X, Y = np.meshgrid(x, y)
    to_ll = pyproj.Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon, lat = to_ll.transform(X, Y)
    return lon, lat


def write_new_h5(data, out_file, src_file, dataset="velocity", extra_attrs=None):
    """Write a 2-D array to a NEW MintPy-style HDF5 file, grafting every
    attribute from `src_file` so MintPy tools (save_gmt.py, view.py) still
    work on the product. `extra_attrs` are added on top (provenance).

    NaN is written as 0 to match MintPy's no-data convention.
    """
    arr = np.array(data, dtype=np.float32)
    arr[~np.isfinite(arr)] = 0.0
    with h5py.File(src_file, "r") as src, h5py.File(out_file, "w") as dst:
        dst.create_dataset(dataset, data=arr)
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        dst.attrs["FILE_TYPE"] = dataset
        dst.attrs["DATA_TYPE"] = "float32"
        for key, value in (extra_attrs or {}).items():
            dst.attrs[key] = str(value)
    print(f"Saved {dataset} to: {out_file}")
    return out_file


# ------------------------------
# Projections
#
# Conventions (MintPy / OPERA), verified against OPERA's own los_east and
# los_north rasters in tests/test_insar_utils.py:
#   inc_deg  incidence angle of the LOS vector, measured from vertical
#   azi_deg  azimuth of the GROUND -> SATELLITE LOS vector, from north,
#            ANTI-CLOCKWISE positive. Ours: ascending ~100 (satellite to the
#            west), descending ~260 (satellite to the east).
#   strike   compass bearing, from north, CLOCKWISE positive. Hayward ~144.
# Mixing the anti-clockwise azimuth with the clockwise bearing is the
# 90-degree bug that once made ascending creep 2x too small; keep them apart.
# ------------------------------

def los_unit_vector(inc_deg, azi_deg):
    """Unit vector (E, N, U) of the LOS direction from the GROUND to the SATELLITE."""
    inc, azi = np.deg2rad(inc_deg), np.deg2rad(azi_deg)
    return np.array([
        -np.sin(inc) * np.sin(azi),   # East
        np.sin(inc) * np.cos(azi),    # North
        np.cos(inc),                  # Up
    ])


def horizontal_unit_vector(bearing_deg):
    """Unit vector (E, N, 0) of a horizontal direction given as a COMPASS
    BEARING clockwise from north (0 = N, 90 = E, 180 = S, 270 = W)."""
    b = np.deg2rad(bearing_deg)
    return np.array([np.sin(b), np.cos(b), 0.0])


def design_matrix_up_fp(inc_deg, azi_deg, strike_deg):
    """Design matrix G mapping [U, FP] -> LOS, one row per track:

        d_los_i = U * (L_i . up) + FP * (L_i . strike_hat)

    Assumes all horizontal motion is fault-parallel (the two-look assumption).
    Returns G with shape (n_tracks, 2), columns [U, FP].
    """
    inc_deg = np.atleast_1d(inc_deg)
    azi_deg = np.atleast_1d(azi_deg)
    up = np.array([0.0, 0.0, 1.0])
    strike_hat = horizontal_unit_vector(strike_deg)
    G = np.zeros((len(inc_deg), 2), dtype=float)
    for i in range(len(inc_deg)):
        L = los_unit_vector(inc_deg[i], azi_deg[i])
        G[i, 0] = L @ up
        G[i, 1] = L @ strike_hat
    return G


def decompose_asc_desc(d_asc, d_desc, inc_asc_deg, azi_asc_deg,
                       inc_desc_deg, azi_desc_deg, strike_deg):
    """Decompose ascending + descending LOS series into vertical (U) and
    fault-parallel (FP), solving the 2x2 system per epoch by least squares.

    d_asc / d_desc must be LOS values on a COMMON set of epochs. FP is positive
    along `strike_deg`: with strike ~144 (S36E), the east block moving SE
    relative to the west block (right-lateral Hayward creep) gives POSITIVE FP
    when the series is (east side - west side).

    Returns (U, FP), each the length of the input series.
    """
    G = design_matrix_up_fp([inc_asc_deg, inc_desc_deg],
                            [azi_asc_deg, azi_desc_deg], strike_deg)
    D = np.vstack([np.asarray(d_asc, dtype=float),
                   np.asarray(d_desc, dtype=float)])   # (2, n_epochs)
    UV_FP = np.linalg.lstsq(G, D, rcond=None)[0]       # (2, n_epochs)
    return UV_FP[0, :], UV_FP[1, :]


# ------------------------------
# pygmt helpers
# ------------------------------

def utm_to_lonlat_grid(arr, x0, y0, dx, dy, epsg, region, dlon, dlat):
    """Nearest-pixel resampling of a north-up UTM raster (`x0`, `y0` the corner
    of the first pixel, `dy` negative) onto a regular lon/lat grid covering
    `region` = [W, E, S, N], for pygmt.grdimage. Nothing is interpolated: each
    lon/lat node takes the value of the UTM pixel it falls in (NaN outside).
    Returns an xarray.DataArray (lat, lon)."""
    import xarray as xr
    from pyproj import Transformer
    lon = np.arange(region[0], region[1] + dlon / 2, dlon)
    lat = np.arange(region[2], region[3] + dlat / 2, dlat)
    LON, LAT = np.meshgrid(lon, lat)
    X, Y = Transformer.from_crs(4326, epsg, always_xy=True).transform(LON, LAT)
    col = np.floor((X - x0) / dx).astype(int)
    row = np.floor((Y - y0) / dy).astype(int)
    ok = (row >= 0) & (row < arr.shape[0]) & (col >= 0) & (col < arr.shape[1])
    out = np.full(LON.shape, np.nan, dtype="float32")
    out[ok] = arr[row[ok], col[ok]]
    return xr.DataArray(out, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))


def utm_grid_region(x0, y0, dx, dy, width, length, epsg):
    """[W, E, S, N] in lon/lat enclosing a north-up UTM raster."""
    from pyproj import Transformer
    xs = [x0, x0 + width * dx]; ys = [y0, y0 + length * dy]
    lon, lat = Transformer.from_crs(epsg, 4326, always_xy=True).transform(
        [xs[0], xs[0], xs[1], xs[1]], [ys[0], ys[1], ys[0], ys[1]])
    return [min(lon), max(lon), min(lat), max(lat)]


def gmt_map_size(region, projection):
    """Width and height (cm) GMT gives a region/projection pair (mapproject -W)."""
    import pygmt, tempfile
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
        f_ = tf.name
    with pygmt.clib.Session() as lib:
        lib.call_module("mapproject", f"-R{'/'.join(str(r) for r in region)} -J{projection} -W ->{f_}")
    w, h = (float(x) for x in open(f_).read().split())
    os.remove(f_)
    return w, h


def write_cpt(path, cmap_name, vmin, vmax, n=64):
    """Write a GMT colour table sampled from a matplotlib colormap, so the
    pygmt figures use the same colours as the matplotlib QC plots. Returns path."""
    import matplotlib.pyplot as plt
    cm_ = plt.get_cmap(cmap_name)
    edges = np.linspace(vmin, vmax, n + 1)
    with open(path, "w") as fh:
        for i in range(n):
            r1, g1, b1 = (int(255 * v) for v in cm_(i / n)[:3])
            r2, g2, b2 = (int(255 * v) for v in cm_((i + 1) / n)[:3])
            fh.write(f"{edges[i]:.6g} {r1}/{g1}/{b1} {edges[i + 1]:.6g} {r2}/{g2}/{b2}\n")
        fh.write(f"B {'/'.join(str(int(255 * v)) for v in cm_(0.0)[:3])}\nF {'/'.join(str(int(255 * v)) for v in cm_(1.0)[:3])}\nN 255/255/255\n")
    return path


# ------------------------------
# Along-fault axis
# ------------------------------

def along_fault_km(ft_lon, ft_lat):
    """Along-fault distance of every trace vertex, in km, with ZERO AT POINT
    PINOLE (the northern end = the LAST row of hayward_fault_profiles.csv) and
    increasing SOUTHWARD, the published along-fault convention.

    Summed segment by segment along the curved trace with a geodesic, not
    projected onto a straight line (the strand bends up to 1.1 km off any
    straight line, which would compress the southern half by ~0.4 km).
    """
    ft_lon = np.asarray(ft_lon, dtype=float)
    ft_lat = np.asarray(ft_lat, dtype=float)
    if ft_lat[-1] < ft_lat[0]:
        raise ValueError("trace must run south -> north (Point Pinole, the zero, last)")
    seg_km = [geopy.distance.geodesic((ft_lat[i], ft_lon[i]),
                                      (ft_lat[i + 1], ft_lon[i + 1])).km
              for i in range(len(ft_lat) - 1)]
    from_south = np.concatenate([[0.0], np.cumsum(seg_km)])
    return from_south[-1] - from_south       # zero at the north end


def lat_to_km_map(profiles_csv):
    """Return a function lat -> km from Point Pinole, by nearest trace vertex.
    The trace is sampled at 100 m so km = (n-1 - index) * 0.1 to within 10 m
    (pinned by tests/test_filepaths.py)."""
    prof = pd.read_csv(profiles_csv)
    plat = prof["ft_lat"].to_numpy(float)
    km = (len(plat) - 1 - np.arange(len(plat))) * 0.1
    return lambda lats: km[np.abs(plat[None, :] - np.atleast_1d(np.asarray(lats, float))[:, None]).argmin(axis=1)]


def strike_frame(epsg, strike_deg, trace_lon, trace_lat):
    """The straight strike frame the map panels of notebooks 04 and 05 are drawn
    in: u along the strike from Point Pinole (km, + toward the strike bearing,
    i.e. south-east), v across the fault (km, + to the north-east = the east
    side), both in the UTM zone `epsg`.

    Unit vectors in (east, north): along = (sin s, cos s), across = (-cos s, sin s)
    for the compass strike s. The origin of u is the northern end of the trace
    (Point Pinole); the origin of v is the median across-fault position of the
    trace. Returns a dict: the two transformers (to_utm, from_utm), the unit
    vectors (along_e, along_n, cross_e, cross_n) and the origins in metres
    (u_origin, v_origin)."""
    to_utm = pyproj.Transformer.from_crs(4326, int(epsg), always_xy=True)
    from_utm = pyproj.Transformer.from_crs(int(epsg), 4326, always_xy=True)
    strike_rad = np.deg2rad(strike_deg)
    along_e, along_n = np.sin(strike_rad), np.cos(strike_rad)
    cross_e, cross_n = -np.cos(strike_rad), np.sin(strike_rad)
    trace_e, trace_n = to_utm.transform(np.asarray(trace_lon, float), np.asarray(trace_lat, float))
    trace_u = trace_e * along_e + trace_n * along_n
    trace_v = trace_e * cross_e + trace_n * cross_n
    u_origin = float(trace_u[int(np.argmax(trace_lat))])       # Point Pinole, the northern end
    v_origin = float(np.median(trace_v))
    return dict(to_utm=to_utm, from_utm=from_utm, along_e=along_e, along_n=along_n,
                cross_e=cross_e, cross_n=cross_n, u_origin=u_origin, v_origin=v_origin, strike_deg=float(strike_deg))


def to_uv(frame, lon, lat):
    """lon/lat -> strike-frame km (see strike_frame): returns (u along strike
    from Point Pinole, v across the fault)."""
    easting, northing = frame["to_utm"].transform(np.asarray(lon, float), np.asarray(lat, float))
    u_km = ((easting * frame["along_e"] + northing * frame["along_n"]) - frame["u_origin"]) / 1e3
    v_km = ((easting * frame["cross_e"] + northing * frame["cross_n"]) - frame["v_origin"]) / 1e3
    return u_km, v_km


def uv_to_lonlat(frame, u_km, v_km):
    """Strike-frame km (u along, v across; see strike_frame) -> (lon, lat), for
    the few things defined in the frame (map corners, ticks, brackets)."""
    u_m = np.asarray(u_km, float) * 1e3 + frame["u_origin"]
    v_m = np.asarray(v_km, float) * 1e3 + frame["v_origin"]
    easting = u_m * frame["along_e"] + v_m * frame["cross_e"]
    northing = u_m * frame["along_n"] + v_m * frame["cross_n"]
    return frame["from_utm"].transform(easting, northing)


# The rotated fault maps (04 Fig 1A, the 05 and 09 supplements) all show the same rectangle of the strike frame
FAULT_MAP_U_KM = (-3.0, 70.0)     # along strike: from 3 km north of Point Pinole to just south of CFW1
FAULT_MAP_HALF_KM = 10.9          # across the fault, each side (3.4 cm wide for 73 km tall at true scale)


def fault_map_corners(frame):
    """The rectangle of the rotated fault maps in the strike frame (see strike_frame): u from
    FAULT_MAP_U_KM[0] to FAULT_MAP_U_KM[1], v within +/- FAULT_MAP_HALF_KM of the trace.
    Returns (the GMT region in its lower-left/upper-right '+r' form, (lon, lat) of the centre,
    the four corners as a (4, 2) array of lon, lat). A grid drawn on the map must cover the corners."""
    u_north, u_south = FAULT_MAP_U_KM
    half_km = FAULT_MAP_HALF_KM
    lon_ll, lat_ll = uv_to_lonlat(frame, u_south, -half_km)          # lower-left = south end, west side
    lon_ur, lat_ur = uv_to_lonlat(frame, u_north, half_km)           # upper-right = north end, east side
    lon_c, lat_c = uv_to_lonlat(frame, 0.5 * (u_north + u_south), 0.0)
    corners = []
    for corner_u, corner_v in ((u_south, -half_km), (u_south, half_km), (u_north, half_km), (u_north, -half_km)):
        corners.append(uv_to_lonlat(frame, corner_u, corner_v))
    region = f"{float(lon_ll)}/{float(lat_ll)}/{float(lon_ur)}/{float(lat_ur)}+r"
    return region, (float(lon_c), float(lat_c)), np.array(corners, float)


def creepmeter_u_km(frame, trace_lon, trace_lat, creepmeters):
    """Where each creepmeter sits along the rotated fault maps: u (km) of the trace vertex nearest it.
    The instruments sit on the trace to within a few tens of metres. creepmeters is the
    TulePond_filepaths list of (code, lat, lon, name). Returns ({code: u km}, {code: distance to that vertex, m})."""
    trace_u_km, trace_v_km = to_uv(frame, trace_lon, trace_lat)
    u_by_code = {}
    distance_by_code = {}
    for code, lat, lon, name in creepmeters:
        u_cm, v_cm = to_uv(frame, lon, lat)
        nearest_vertex = int(np.argmin(np.hypot(trace_u_km - u_cm, trace_v_km - v_cm)))
        u_by_code[code] = float(trace_u_km[nearest_vertex])
        distance_by_code[code] = float(1e3 * np.hypot(trace_u_km[nearest_vertex] - u_cm, trace_v_km[nearest_vertex] - v_cm))
    return u_by_code, distance_by_code


def utm_grid_to_lonlat(values, x_first, x_step, y_first, y_step, to_utm, corners, lon_step=0.00035, lat_step=0.00028):
    """A UTM-gridded field looked up on a regular lon/lat grid that covers the map corners (plus 0.01 deg),
    nearest UTM pixel, NaN off the UTM grid (GMT draws regular lon/lat grids). x_first/y_first are the
    pixel-centre coordinates of the first column/row, x_step/y_step the pixel size (y_step < 0 for north-up).
    Returns (lon_axis, lat_axis, values on the lon/lat grid, lat x lon)."""
    lat_axis = np.arange(corners[:, 1].min() - 0.01, corners[:, 1].max() + 0.01, lat_step)
    lon_axis = np.arange(corners[:, 0].min() - 0.01, corners[:, 0].max() + 0.01, lon_step)
    lon_grid, lat_grid = np.meshgrid(lon_axis, lat_axis)
    easting, northing = to_utm.transform(lon_grid, lat_grid)
    col_index = np.round((easting - x_first) / x_step).astype(int)
    row_index = np.round((northing - y_first) / y_step).astype(int)
    on_grid = (col_index >= 0) & (col_index < values.shape[1]) & (row_index >= 0) & (row_index < values.shape[0])
    values_lonlat = np.full(lon_grid.shape, np.nan)
    values_lonlat[on_grid] = values[row_index[on_grid], col_index[on_grid]]
    return lon_axis, lat_axis, values_lonlat


def fault_map_panel(fig, frame, grid, cpt, map_h_cm, trace_lon, trace_lat, creepmeters, sse_u_km,
                    letter="A", title="Hayward Fault", colorbar_frame=None, extra_symbols=(), extra_labels=()):
    """Draw the rotated fault map of Fig 1A at the current origin of a pygmt figure. The same panel is used
    by 04 (Fig 1A), 05 (the validation supplement) and 09 (the alignment-array supplement), so all three
    look alike; only the field, its colours, the extra symbols and the panel letter differ.

    What it draws, in this order: the field `grid` (an xarray lon/lat grid covering fault_map_corners) in
    the colour table `cpt`; the mapped trace; the extra symbols and their labels (lists of keyword dicts
    for fig.plot and text_backed); the creepmeter codes on the west side; the SSE extent bracket on the
    east side, from sse_u_km[0] to sse_u_km[1] (km along the strike frame) with its label; the city
    names on the east side; the creepmeters as yellow triangles on top; North and South; the panel letter
    and title above the top-left corner; the 5 km scale bar; the distance from Point Pinole down the
    left edge; and, if colorbar_frame is given (e.g. "xa2f1+lFault-parallel velocity (mm/yr)"), a 2.5 cm
    colour bar centred under the map.

    The map is GMT's oblique Mercator with its x axis along strike - 90 deg, so the fault runs up the
    page with north at the top, at true scale: the height is map_h_cm and the width follows from it.
    Fonts and frame pens are the caller's pygmt.config. Returns the map width in cm."""
    import pygmt
    region, (lon_c, lat_c), corners = fault_map_corners(frame)
    u_north, u_south = FAULT_MAP_U_KM
    strike_deg = frame["strike_deg"]
    trace_u_km, trace_v_km = to_uv(frame, trace_lon, trace_lat)
    u_creepmeter, _ = creepmeter_u_km(frame, trace_lon, trace_lat, creepmeters)
    sse_lo_km, sse_hi_km = sse_u_km
    gold = "#F0E442"                                       # Okabe-Ito yellow: the creepmeters, as Fig 1D
    colorbar_w_cm = 2.5                                    # every colour bar of Fig 1 the same size

    # the width that makes the map exactly map_h_cm tall at true scale (GMT's own size for a trial width)
    trial_w_cm = 3.4
    _, trial_h_cm = gmt_map_size([region], f"Oa{lon_c}/{lat_c}/{strike_deg - 90}/{trial_w_cm}c")   # region passed whole (+r form)
    map_w_cm = trial_w_cm * map_h_cm / trial_h_cm
    projection = f"Oa{lon_c}/{lat_c}/{strike_deg - 90}/{map_w_cm:.3f}c"
    print(f"map panel {map_w_cm:.2f} x {map_h_cm:.2f} cm at true scale")

    fig.basemap(region=region, projection=projection, frame=["lrtb"])
    fig.grdimage(grid=grid, cmap=cpt, nan_transparent=True, interpolation="c")   # -nc: bicubic when GMT resamples the grid
    #                                                                              to the page (display only, the grid is the data)
    fig.plot(x=trace_lon, y=trace_lat, pen="0.6p,black")
    for symbol in extra_symbols:                           # e.g. 09's alignment arrays, under the creepmeter labels
        fig.plot(**symbol)
    for label in extra_labels:
        text_backed(fig, **label)
    for code, lat, lon, name in creepmeters:               # labels on the west side; the triangles go on top, below
        text_backed(fig, x=lon, y=lat, text=code, justify="RM", offset="-0.15c/0c", font="7p,Helvetica,black")
    # the SSE extent bracket, east of the fault so it stays clear of the creepmeter labels, ticks reaching
    # almost to the trace; labelled in two rows (GMT has no newline outside paragraph mode), one size up
    bracket_lon, bracket_lat = uv_to_lonlat(frame, [sse_lo_km, sse_hi_km], [2.0, 2.0])
    fig.plot(x=bracket_lon, y=bracket_lat, pen="1.1p,black")
    for u_tick in (sse_lo_km, sse_hi_km):
        tick_lon, tick_lat = uv_to_lonlat(frame, [u_tick, u_tick], [0.4, 2.0])
        fig.plot(x=tick_lon, y=tick_lat, pen="1.1p,black")
    u_mid = 0.5 * (sse_lo_km + sse_hi_km)
    label_lon, label_lat = uv_to_lonlat(frame, [u_mid - 1.0, u_mid + 1.0], [2.9, 2.9])   # left edge 0.9 km east of the bracket
    text_backed(fig, x=label_lon, y=label_lat, text=["SSE", "extent"], justify="LM", font="8p,Helvetica,black")
    # the cities along the fault, east side opposite the creepmeter labels, each within the city's stretch of the
    # fault (labels only, nothing is measured from them):
    #   Berkeley, Union City: the city centre's along-strike position
    #   Oakland: midway between CTM1 (Temescal) and COZ1 (Oakland Zoo), both in Oakland
    #   Hayward: at CHP1 (Palisade Street, Hayward)
    #   Fremont: its centre (Central Park) sits beside the SSE bracket, so just south of the bracket, still in the city
    city_u = {"Berkeley": float(to_uv(frame, [-122.2727], [37.8716])[0][0]),
              "Oakland": 0.5 * (u_creepmeter["CTM1"] + u_creepmeter["COZ1"]),
              "Hayward": u_creepmeter["CHP1"],
              "Union City": float(to_uv(frame, [-122.0439], [37.5934])[0][0]),
              "Fremont": max(float(to_uv(frame, [-121.9886], [37.5485])[0][0]), sse_hi_km + 2.0)}
    for city, u_city in city_u.items():
        assert u_north < u_city < u_south, f"{city} (u {u_city:.1f} km) falls off the map"
        nearest_vertex = int(np.argmin(np.abs(trace_u_km - u_city)))
        city_lon, city_lat = uv_to_lonlat(frame, [u_city], [trace_v_km[nearest_vertex] + 0.6])
        text_backed(fig, x=city_lon[0], y=city_lat[0], text=city, justify="LM", font="7p,Helvetica-Oblique,black")
    for code, lat, lon, name in creepmeters:               # the creepmeter symbol of Fig 1D, on top of every label
        fig.plot(x=[lon], y=[lat], style="i0.22c", fill=gold, pen="0.5p,black")
    fig.text(position="TR", text="North", offset="-0.1c/-0.1c", font="7p,Helvetica,gray20", fill="white", clearance="0.05c")
    fig.text(position="BR", text="South", offset="-0.1c/0.1c", font="7p,Helvetica,gray20", fill="white", clearance="0.05c")
    if letter is not None:
        fig.text(position="TL", text=f"@%Helvetica-Bold%{letter}@%% @:7p:{title}@::", justify="BL", offset="0c/0.12c",
                 font="9p,Helvetica", no_clip=True)
    fig.basemap(region=region, projection=projection, frame=["lrtb"], map_scale="jBL+w5k+o0.3c/0.4c")   # frame on top, scale bar

    # distance from Point Pinole down the left edge: the map's vertical IS the strike frame u (true scale), so a
    # linear axis over the same box reads u directly. Checked: GMT's own page y of the creepmeters must match the
    # linear axis to 0.02 cm.
    creepmeter_lonlat = []
    creepmeter_axis_y = []                                 # where the linear axis puts each creepmeter (cm from the map bottom)
    for code, lat, lon, name in creepmeters:
        creepmeter_lonlat.append([lon, lat])
        creepmeter_axis_y.append(map_h_cm * (u_south - u_creepmeter[code]) / (u_south - u_north))
    with pygmt.clib.Session() as lib, pygmt.helpers.GMTTempFile() as page_xy:
        with pygmt.helpers.GMTTempFile() as lonlat_file:
            np.savetxt(lonlat_file.name, creepmeter_lonlat)
            lib.call_module("mapproject", f"{lonlat_file.name} -R{region} -J{projection} ->{page_xy.name}")
        creepmeter_page_y = np.loadtxt(page_xy.name, ndmin=2)[:, 1]
    axis_error_cm = float(np.max(np.abs(creepmeter_page_y - np.array(creepmeter_axis_y))))
    assert axis_error_cm < 0.02, f"distance axis off the map by {axis_error_cm:.3f} cm"
    fig.basemap(region=[0, 1, u_north, u_south], projection=f"X{map_w_cm:.3f}c/-{map_h_cm}c", frame=["W", "ya10f5+lDistance from Point Pinole (km)"])
    if colorbar_frame is not None:
        with pygmt.config(FONT_ANNOT_PRIMARY="13p", FONT_LABEL="13p"):     # GMT scales colour-bar text down with the bar thickness (-> ~6.5p)
            fig.colorbar(cmap=cpt, position=f"x{0.5 * (map_w_cm - colorbar_w_cm)}c/-0.4c+w{colorbar_w_cm}c/0.3c+h+jTL", frame=colorbar_frame)
    return map_w_cm


# ------------------------------
# Statistics
# ------------------------------

def register_helvetica_faces():
    """Make Helvetica's bold and oblique faces available to matplotlib.

    matplotlib registers only the FIRST face of a .ttc collection, so macOS's
    Helvetica.ttc gives it Helvetica Regular alone and a bold panel letter
    would silently render regular. The other faces are extracted once (with
    fontTools, which matplotlib ships) into the matplotlib config directory
    and registered. Returns True when Helvetica Bold is available."""
    import matplotlib
    from matplotlib import font_manager as fm
    ttc = "/System/Library/Fonts/Helvetica.ttc"
    cache = os.path.join(matplotlib.get_configdir(), "tulepond_fonts")
    try:
        if not os.path.exists(ttc):
            return False
        os.makedirs(cache, exist_ok=True)
        if not any(name.endswith(".ttf") for name in os.listdir(cache)):
            from fontTools.ttLib import TTCollection
            for face in TTCollection(ttc).fonts:
                face.save(os.path.join(cache, face["name"].getDebugName(6) + ".ttf"))     # PostScript name, e.g. Helvetica-Bold
        for name in os.listdir(cache):
            if name.endswith(".ttf"):
                fm.fontManager.addfont(os.path.join(cache, name))
        return "Helvetica-Bold" in fm.findfont(fm.FontProperties(family="Helvetica", weight="bold"), fallback_to_default=False)
    except Exception as err:
        print("Helvetica faces not registered, falling back to Arial:", err)
        return False


def mpl_paper_style(base=8):
    """Matplotlib settings for the paper figures: Helvetica (the font of the
    pygmt figures; Arial if its bold face cannot be registered), `base` pt
    text, and TrueType (Type 42) embedding so PDF text stays editable."""
    import matplotlib
    family = "Helvetica" if register_helvetica_faces() else "Arial"
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [family, "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": base,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        # mathtext ($\\sigma$, $\\pm$, the bold panel letters) in the same face, not DejaVu
        "mathtext.fontset": "custom", "mathtext.rm": family, "mathtext.it": f"{family}:italic",
        "mathtext.bf": f"{family}:bold", "mathtext.default": "regular",
        # legends sit on a white box so no line runs through their text (Science figure check, 2026-09-29)
        "legend.frameon": True, "legend.facecolor": "white", "legend.edgecolor": "none", "legend.framealpha": 0.85,
        "legend.fontsize": max(base - 2, 6),
    })
    return dict(matplotlib.rcParams)


# Okabe-Ito colour-blind-safe categorical colours (no blue/green pair: Dani cannot tell them apart)
okabe_ito = ["#0072B2", "#E69F00", "#D55E00", "#CC79A7", "#000000", "#56B4E9", "#F0E442"]


def panel_title(ax, letter, text="", size=7):
    """Science panel label on a matplotlib axis: a bold capital letter, then a
    plain `size`-pt description, left-aligned above the axis."""
    label = rf"$\bf{{{letter}}}$" + (f"  {text}" if text else "")
    ax.set_title(label, loc="left", fontsize=size)


def text_backed(fig, transparency=30, **kw):
    """A pygmt map label on a translucent white box with CRISP text: GMT applies the
    transparency to the glyphs as well as the box, so the label is drawn twice,
    first with the box and the transparency, then again with neither."""
    fig.text(fill="white", transparency=transparency, **kw)
    fig.text(**kw)


def robust_sigma(x):
    """Robust standard deviation via the median absolute deviation
    (1.4826 * MAD). Insensitive to the outlier tail a plain std is inflated by."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def detrend(t, y):
    """Residuals of a linear fit y(t), ignoring NaN; NaN preserved."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    out = np.full_like(y, np.nan, dtype=float)
    if m.sum() > 1:
        b = np.polyfit(t[m], y[m], 1)
        out[m] = y[m] - np.polyval(b, t[m])
    return out


def linear_rate(t, y):
    """Linear rate by scipy.stats.linregress, ignoring NaN.
    Returns (slope, intercept, std_err, n) -- the rate and its standard error."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan, np.nan, int(m.sum())
    slope, intercept, r_value, p_value, std_err = stats.linregress(t[m], y[m])
    return slope, intercept, std_err, int(m.sum())


def gaussian_smooth_km(x, lats_of_x, sigma_km):
    """NaN-aware, gap-preserving Gaussian smoothing along the fault (crossings
    in fault order; spacing from the latitudes, 1 deg = 111.19 km). Weights are
    renormalised over the finite neighbours; NaN where fewer than 15 % of the
    kernel weight is data, and NaN wherever the input is NaN. Used by 03, 04
    (Fig 1C, sigma 0.15 km) and 07c (Fig 2K, the same 0.15 km)."""
    spacing_km = float(np.median(np.abs(np.diff(lats_of_x)))) * 111.19
    sig = max(sigma_km / spacing_km, 0.5)
    finite = np.isfinite(x).astype(float)
    filled = np.where(finite > 0, x, 0.0)
    den = gaussian_filter1d(finite, sig, mode="nearest")
    out = np.divide(gaussian_filter1d(filled, sig, mode="nearest"), den, out=np.full_like(filled, np.nan), where=den > 0.15)
    out[~np.isfinite(x)] = np.nan
    return out


def annual_increment_rate(t, y):
    """Rate and 1-sigma of a continuous record (creepmeter) from its slip gained
    in each whole year from the first sample: rate = mean(dy), sigma =
    std(dy, ddof=1) / sqrt(n). Daily samples are not independent (episodic
    creep), so linregress's std_err (~0.01 mm/yr) understates the uncertainty;
    one-year blocks are treated as the independent samples instead.
    Returns (rate, sigma, n_years)."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    edges = t[0] + np.arange(0, np.floor(t[-1] - t[0]) + 1)
    dy = np.diff(np.interp(edges, t, y))
    assert dy.size >= 3, f"annual_increment_rate needs >= 3 whole years, got {dy.size}"
    return float(dy.mean()), float(dy.std(ddof=1) / np.sqrt(dy.size)), int(dy.size)


# ------------------------------
# Creepmeters (USGS, Langbein et al. 2024)
# ------------------------------

def creepmeter_file(creepmeter_dir, code):
    """Path to a creepmeter record, preferring the refreshed `.10min.gz`
    (read transparently by pandas) over a stale uncompressed `.10min`."""
    gz = os.path.join(creepmeter_dir, code.lower() + ".10min.gz")
    return gz if os.path.exists(gz) else os.path.join(creepmeter_dir, code.lower() + ".10min")


def read_creepmeter_10min(filepath, date_start=None, date_end=None):
    """Read a USGS creepmeter `.10min`/`.10min.gz` file (whitespace columns:
    year, fractional day-of-year, slip_mm; no header) into DAILY means with
    columns ['decimal_year', 'slip_mm'], zeroed to the first day.

    The published values are already divided by cos(30 deg) at source, i.e.
    they are right-lateral fault-parallel slip; no obliquity correction here.
    """
    df = pd.read_csv(filepath, sep=r"\s+", usecols=[0, 1, 2], header=None,
                     names=["year", "doy", "slip_mm"])
    df["decimal_year"] = df["year"] + (df["doy"] / df["year"].apply(
        lambda y: 366 if (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0) else 365))
    if date_start is not None:
        df = df[df["decimal_year"] >= date_start]
    if date_end is not None:
        df = df[df["decimal_year"] <= date_end]
    df = df.reset_index(drop=True)
    # Group on the INTEGER day: the doy column is fractional (one value per
    # 10-minute sample), so grouping on it directly would average nothing.
    df["doy_int"] = df["doy"].astype(int)
    df_daily = df.groupby(["year", "doy_int"], as_index=False).agg(
        {"decimal_year": "first", "slip_mm": "mean"})
    df_daily["slip_mm"] = df_daily["slip_mm"] - df_daily["slip_mm"].iloc[0]
    return df_daily[["decimal_year", "slip_mm"]]


def daily_creepmeter(creepmeter_dir, code, t_start, t_end, glitches):
    """Daily creepmeter record of one instrument between t_start and t_end
    (decimal years), the listed glitch days removed, zeroed at the first day
    kept. Used by notebooks 04 and 05.

    `glitches` is the explicit list in TulePond_filepaths (creepmeter_glitches):
    (code, first day, last day, reason) rows, days as 'YYYY-MM-DD' in this
    reader's day convention (the day of a daily mean is the floor of its
    decimal year). No automatic filter: only those days are dropped, and each
    listed day inside the window must be in the record. Prints the days dropped.
    Returns (decimal_year, slip_mm) as two arrays."""
    cm = read_creepmeter_10min(creepmeter_file(creepmeter_dir, code), date_start=t_start, date_end=t_end)
    dates = cm["decimal_year"].values
    slip = cm["slip_mm"].values
    days = pd.DatetimeIndex([pd.Timestamp(decimal_year_to_date(date)).floor("D") for date in dates])
    first_day = pd.Timestamp(decimal_year_to_date(t_start)).floor("D")
    last_day = pd.Timestamp(decimal_year_to_date(t_end)).floor("D")
    dropped = np.zeros(len(dates), bool)
    dropped_text = []
    for glitch_code, glitch_first, glitch_last, reason in glitches:
        if glitch_code != code:
            continue
        glitch_first = pd.Timestamp(glitch_first)
        glitch_last = pd.Timestamp(glitch_last)
        assert glitch_last >= glitch_first, f"{code}: glitch {glitch_first.date()} ends before it starts"
        if glitch_last < first_day or glitch_first > last_day:
            continue                                            # outside this window
        in_glitch = (days >= glitch_first) & (days <= glitch_last)
        assert in_glitch.any(), f"{code}: listed glitch {glitch_first.date()} to {glitch_last.date()} is not in the record"
        dropped |= in_glitch
        dropped_text.append(f"{glitch_first.date()} to {glitch_last.date()} ({in_glitch.sum()} d: {reason})")
    print(f"  {code}: {dropped.sum()} listed glitch days dropped of {len(slip)}"
          + (": " + "; ".join(dropped_text) if dropped_text else ""))
    dates = dates[~dropped]
    slip = slip[~dropped]
    return dates, slip - slip[0]


# ------------------------------
# Catalogues (seismicity and repeating earthquakes), used by 05 and 10
# ------------------------------

def read_creepmeter_10min_raw(filepath, date_start=None, date_end=None):
    """The same USGS `.10min` file at its native 10-minute sampling (no daily
    averaging): columns ['decimal_year', 'slip_mm'], as published. Used where a
    creep-event STEP has to be measured (steps take hours; daily means smear
    them across two days)."""
    df = pd.read_csv(filepath, sep=r"\s+", usecols=[0, 1, 2], header=None,
                     names=["year", "doy", "slip_mm"])
    df["decimal_year"] = df["year"] + (df["doy"] / df["year"].apply(
        lambda y: 366 if (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0) else 365))
    if date_start is not None:
        df = df[df["decimal_year"] >= date_start]
    if date_end is not None:
        df = df[df["decimal_year"] <= date_end]
    return df[["decimal_year", "slip_mm"]].reset_index(drop=True)


def _decimal_year_from_ymdhms(yr, mo, dy, hr, mn, sc):
    """Vectorised decimal year from year/month/day/hour/minute/second columns."""
    t = pd.to_datetime(dict(year=yr, month=mo, day=dy, hour=hr, minute=mn), errors="coerce") + pd.to_timedelta(sc, unit="s")
    start = pd.to_datetime(dict(year=yr, month=1, day=1))
    end = pd.to_datetime(dict(year=yr + 1, month=1, day=1))
    return yr + (t - start).dt.total_seconds() / (end - start).dt.total_seconds()


def load_growclust_catalog(path, bbox=None):
    """Bo Xie's GrowClust catalogue (2000-2025), relocated PLUS unrelocated:
    latitude/longitude/depth_km hold the relocation where the event clustered
    (nbranch > 1) and the NCSS location otherwise. Returns the common schema
    YYYY, Lon, Lat, Depth, Mag, evID, source; clipped to bbox (W, S, E, N)."""
    df = pd.read_csv(path)
    need = ["yr", "mon", "day", "hr", "min", "sec", "latitude", "longitude", "depth_km", "magnitude", "evid", "nbranch"]
    missing = [c for c in need if c not in df.columns]
    assert not missing, f"growclust catalogue missing columns: {missing}"
    out = df[["latitude", "longitude", "depth_km", "magnitude", "evid", "nbranch"]].rename(
        columns={"latitude": "Lat", "longitude": "Lon", "depth_km": "Depth", "magnitude": "Mag", "evid": "evID"})
    out["YYYY"] = _decimal_year_from_ymdhms(df.yr.astype(int), df.mon.astype(int), df.day.astype(int),
                                            df.hr.astype(int), df["min"].astype(int), df.sec.astype(float))
    out["source"] = np.where(out.nbranch > 1, "growclust", "ncss_unrelocated")
    out = out.dropna(subset=["YYYY", "Lat", "Lon", "Depth", "Mag"])
    assert not out.evID.duplicated().any(), "duplicate evid in the growclust file"
    if bbox is not None:
        w, s, e, n = bbox
        out = out[(out.Lon >= w) & (out.Lon <= e) & (out.Lat >= s) & (out.Lat <= n)]
    return out.reset_index(drop=True)


def load_aa(aa_dir, code):
    """One USGS Bay Area alignment array (v3.0 archive): year, cumulative dextral
    movement (mm), its std (mm), sorted by year. Two file layouts exist."""
    d = pd.read_csv(os.path.join(aa_dir, f"{code}.csv"), encoding="latin-1")
    d = d.rename(columns={c: c.replace("\xa0", " ").strip() for c in d.columns})
    if "Measurement_yr" in d.columns:
        d = d[["Measurement_yr", "Cumulative_dextral_movement_mm", "Dextral_movement_st_dev_mm"]]
    elif "Year" in d.columns:
        d = d[["Year", "Cumulative movement (mm)", "Std. dev."]]
    else:
        raise ValueError(f"{code}: unrecognised alignment-array file layout ({list(d.columns)[:4]}...)")
    d.columns = ["year", "cum_mm", "std_mm"]
    d = d.apply(pd.to_numeric, errors="coerce").dropna(subset=["year", "cum_mm"])
    d["std_mm"] = d["std_mm"].fillna(0.0).clip(lower=0.0)
    return d.sort_values("year").reset_index(drop=True)


def load_aa_summary(aa_summary, fault="Hayward"):
    """The archive's summary table (site, lon, lat, long-term rates, record span)."""
    s = pd.read_csv(aa_summary, encoding="latin-1")
    s.columns = [("site_code" if c.endswith("site_code") else c.replace("\xa0", " ").strip()) for c in s.columns]
    return s[s["fault"] == fault].reset_index(drop=True)


def load_cre_taira(path):
    """T. Taira's HFRC linkage-cluster repeating-earthquake catalogue (GEN3).
    Whitespace (or comma) separated: YYYY Lat Lon Depth cumD(cm) EVID CSID Mag D(cm)."""
    names = ['YYYY', 'Lat', 'Lon', 'Depth', 'cumD(cm)', 'EVID', 'CSID', 'Mag', 'D(cm)']
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in (line.split(",") if "," in line else line.split()) if p.strip() != ""]
            if len(parts) != len(names):
                continue
            try:
                float(parts[0])
            except ValueError:
                continue
            rows.append(parts)
    df = pd.DataFrame(rows, columns=names)
    for col in ('YYYY', 'Lat', 'Lon', 'Depth', 'Mag'):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    out = df[["YYYY", "Lon", "Lat", "Depth", "Mag", "CSID"]].rename(columns={"CSID": "seqID"})
    out["cat"] = "Taira26"
    return out


def load_cre_waldhauser(path):
    """Waldhauser & Schaff (2021) repeaters: '#' sequence headers (last field =
    sequence id) followed by one line per event; 39-line file header."""
    columns = ['YR', 'MO', 'DY', 'HR', 'MN', 'SC', 'DAYS', 'Lat', 'Lon', 'Depth', 'EX', 'EY', 'EZ',
               'MAG', 'DMAG', 'DMAGE', 'CCm', 'evID', 'seqID']
    rows, seq = [], None
    with open(path) as fh:
        for _ in range(39):
            next(fh)
        for line in fh:
            line = line.strip()
            if line.startswith('#'):
                seq = line[1:].split()[-1]
            elif seq and line:
                rows.append(line.split() + [seq])
    df = pd.DataFrame(rows, columns=columns)
    for col in columns[:-2]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["YYYY"] = _decimal_year_from_ymdhms(df.YR.astype(int), df.MO.astype(int), df.DY.astype(int),
                                           df.HR.astype(int), df.MN.astype(int), df.SC)
    out = df[["YYYY", "Lon", "Lat", "Depth", "MAG", "seqID"]].rename(columns={"MAG": "Mag"})
    out["cat"] = "Waldhauser21"
    return out


def project_to_fault(df, trace, lon_col="Lon", lat_col="Lat"):
    """Add `dist_fault_km` (signed, + east) and `ft_lat` (latitude of the
    nearest trace vertex) to a catalogue, on a local tangent plane."""
    lat0 = float(trace.ft_lat.mean())
    kx, ky = 111.195 * np.cos(np.radians(lat0)), 111.195
    tx, ty = trace.ft_lon.to_numpy() * kx, trace.ft_lat.to_numpy() * ky
    ex, ey = df[lon_col].to_numpy() * kx, df[lat_col].to_numpy() * ky
    d2 = (ex[:, None] - tx[None, :]) ** 2 + (ey[:, None] - ty[None, :]) ** 2
    idx = np.argmin(d2, axis=1)
    dist = np.sqrt(d2[np.arange(len(ex)), idx])
    j = np.clip(idx, 1, len(tx) - 1)
    cross = (tx[j] - tx[j - 1]) * (ey - ty[idx]) - (ty[j] - ty[j - 1]) * (ex - tx[idx])
    out = df.copy()
    out["dist_fault_km"] = dist * np.where(cross < 0, 1.0, -1.0)     # trace runs S->N, so -cross is east
    out["ft_lat"] = trace.ft_lat.to_numpy()[idx]
    return out


# ------------------------------
# Provenance and the number registry
#
# Every number that ends up in the manuscript is registered ONCE, from the
# notebook that computes it, into outputs/registry.csv (append-only log) and
# rendered to outputs/registry.txt (latest value per key, grouped by notebook
# and by where it goes in the paper). Dani copies from registry.txt into
# Overleaf, so the row says exactly which notebook and cell produced it.
#
# Re-running a notebook that produces the SAME numbers adds identical rows
# and is the reproducibility proof. A run that CHANGES a registered value
# raises RegistryConflict unless the environment variable TULEPOND_SUPERSEDE=1
# is set, so a number cannot drift without someone deciding it should.
# ------------------------------

REGISTRY_COLUMNS = ["key", "value", "units", "precision", "rendered", "where",
                    "notebook", "cell", "sidecar", "inputs", "git_hash", "git_dirty",
                    "timestamp", "run_id"]


class RegistryConflict(Exception):
    """A registered number changed at its stated precision."""


def git_hash(repo_dir):
    """Short git hash of the repo and whether the tree has uncommitted changes."""
    try:
        h = subprocess.run(["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", repo_dir, "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip() != ""
        return h, dirty
    except Exception:
        return "nogit", True


def file_fingerprint(path, full_limit_mb=256):
    """sha256 of a file, shortened to 12 hex characters. Files larger than
    `full_limit_mb` are fingerprinted on (size, first 4 MiB, last 4 MiB) and
    the result is prefixed with '~' so the registry shows it was partial."""
    size = os.path.getsize(path)
    h = hashlib.sha256()
    chunk = 4 * 1024 * 1024
    with open(path, "rb") as f:
        if size <= full_limit_mb * 1024 * 1024:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
            return h.hexdigest()[:12]
        h.update(str(size).encode())
        h.update(f.read(chunk))
        f.seek(max(size - chunk, 0))
        h.update(f.read(chunk))
        return "~" + h.hexdigest()[:12]


def provenance_stamp(notebook, repo_dir):
    """One-line provenance string for figure metadata and sidecar headers."""
    h, dirty = git_hash(repo_dir)
    return f"{notebook} | git {h}{'+' if dirty else ''} | {datetime.now():%Y-%m-%d %H:%M}"


def render_value(value, precision):
    """The string to paste into the manuscript: rounded to `precision`
    decimals, integers without a decimal point, dates left as they are."""
    if isinstance(value, str):
        return value
    if precision == 0:
        return f"{int(round(float(value)))}"
    return f"{float(value):.{precision}f}"


class Registry:
    """Append-only registry of manuscript numbers for ONE notebook run.

        REG = utils.Registry("05_event_detection", common_paths, repo_dir)
        REG.inputs(path1, path2)                       # files this notebook read
        REG.register("05_n_events", 5, "count", 0, where="P3", cell="detector")
        REG.sidecar("events_stats", ["n events: 5", ...])
        REG.render()                                   # rewrite registry.txt
    """

    def __init__(self, notebook, common_paths, repo_dir):
        self.notebook = notebook
        self.csv_path = common_paths["registry_csv"]
        self.txt_path = common_paths["registry_txt"]
        self.out_dir = os.path.join(common_paths["out_dir"], notebook)
        os.makedirs(self.out_dir, exist_ok=True)
        self.repo_dir = repo_dir
        self.git, self.dirty = git_hash(repo_dir)
        self.run_id = uuid.uuid4().hex[:8]
        self._inputs = []
        self.supersede = os.environ.get("TULEPOND_SUPERSEDE", "0") == "1"
        if not os.path.exists(self.csv_path):
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(REGISTRY_COLUMNS)
        print(f"Registry for {notebook}: run {self.run_id}, git {self.git}{'+' if self.dirty else ''}")

    def inputs(self, *paths):
        """Declare the input files; each is fingerprinted once."""
        for p in paths:
            assert os.path.exists(p), f"input missing: {p}"
            self._inputs.append(f"{os.path.basename(p)}:{file_fingerprint(p)}")
        return self

    def latest(self):
        """Latest row per key over the whole registry (all notebooks)."""
        df = pd.read_csv(self.csv_path, dtype=str)
        if df.empty:
            return df
        return df.groupby("key", as_index=False).last()

    def register(self, key, value, units, precision, where, cell, sidecar=None):
        """Register one manuscript number. Raises RegistryConflict if `key`
        already exists with a different rendered value and TULEPOND_SUPERSEDE
        is not set."""
        rendered = render_value(value, precision)
        latest = self.latest()
        if not latest.empty and key in set(latest["key"]):
            old = latest.loc[latest["key"] == key].iloc[0]
            if str(old["rendered"]) != rendered and not self.supersede:
                raise RegistryConflict(
                    f"{key}: was {old['rendered']} (git {old['git_hash']}, {old['timestamp']}), "
                    f"now {rendered}. Set TULEPOND_SUPERSEDE=1 to accept the change.")
        row = [key, value, units, precision, rendered, where, self.notebook, cell,
               sidecar or "", ";".join(self._inputs), self.git, self.dirty,
               datetime.now().isoformat(timespec="seconds"), self.run_id]
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"  registered {key} = {rendered} {units}  -> {where}")
        return value

    def register_many(self, prefix, values, units, precision, where, cell,
                      labels=None, sidecar=None):
        """Register a list of values as prefix_<label>."""
        labels = labels or [str(i) for i in range(len(values))]
        for lab, val in zip(labels, values):
            self.register(f"{prefix}_{lab}", val, units, precision, where, cell, sidecar)

    def sidecar(self, name, lines):
        """Write a human-readable stats file next to the products, with a
        provenance header. Returns the path (relative to outputs/)."""
        path = os.path.join(self.out_dir, f"{name}_stats.txt")
        with open(path, "w") as f:
            f.write(f"# {provenance_stamp(self.notebook, self.repo_dir)}\n")
            f.write(f"# inputs: {'; '.join(self._inputs)}\n")
            for line in lines:
                f.write(str(line).rstrip("\n") + "\n")
        print(f"Saved stats to: {path}")
        return os.path.relpath(path, os.path.dirname(self.csv_path))

    def render(self):
        """Rewrite registry.txt: latest value per key, grouped by notebook,
        then an index grouped by manuscript location."""
        latest = self.latest()
        # keys whose notebook is no longer in the repo (renamed or retired) are listed apart, at the end, and are
        # left out of the by-location index, so a stale value is never copied by mistake (Dani 2026-10-02:
        # 10_seasonality_forecast_v2 rows sat beside 11_forecast's). registry.csv itself stays append-only.
        current = {os.path.splitext(n)[0] for n in os.listdir(self.repo_dir) if n.endswith(".ipynb")}
        # a key is live only if its notebook still exists AND the notebook's latest run registered it (a key the
        # notebook no longer produces, e.g. 09's interval keys moved to 11 on 2026-10-02, is retired with it)
        rows = pd.read_csv(self.csv_path, dtype=str)
        last_run = rows.sort_values("timestamp").groupby("notebook")["run_id"].last()
        in_last_run = latest.apply(lambda r: r["run_id"] == last_run.get(r["notebook"]), axis=1).astype(bool)
        is_live = latest["notebook"].isin(current) & in_last_run
        live, retired = latest[is_live], latest[~is_live]

        def lines(f, frame):
            for nb, grp in frame.groupby("notebook"):
                f.write(f"## {nb}\n")
                for _, r in grp.sort_values("key").iterrows():
                    f.write(f"{r['key']} = {r['rendered']} {r['units']} ({r['precision']} dp) -> {r['where']}"
                            f"  [{r['cell']}; git {r['git_hash']}; {str(r['timestamp'])[:10]}]\n")
                f.write("\n")
        with open(self.txt_path, "w") as f:
            f.write(f"# TulePond_DISP number registry, rendered {datetime.now():%Y-%m-%d %H:%M}\n")
            f.write("# One line per key: key = rendered units (precision) -> where [notebook/cell; git; date]\n\n")
            lines(f, live)
            f.write("## By manuscript location\n")
            for where, grp in live.groupby("where"):
                f.write(f"{where}: " + ", ".join(f"{r['key']}={r['rendered']}" for _, r in grp.iterrows()) + "\n")
            if len(retired):
                f.write(f"\n# ===== RETIRED: {len(retired)} keys from notebooks no longer in the repo or not produced by "
                        f"their notebook's latest run ({', '.join(sorted(retired['notebook'].unique()))}). DO NOT QUOTE. =====\n\n")
                lines(f, retired)
        print(f"Rendered registry to: {self.txt_path}")
        return self.txt_path


# ---------------------------------------------------------------------------
# Fault trace file reader -- ported VERBATIM from
# Hayward_DISP-S1/analysis/run07_fault_profiles.py (notebook 07 port round,
# 2026-09-25); the hand-drawn trace KML is read the same way everywhere.
# ---------------------------------------------------------------------------
def _is_number(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def load_trace(path):
    """Vertices of the trace as one (N, 2) lon/lat array.

    KML of Points: hand-edited crossing placemarks; placemarks whose name is
    not a number (annotations like "multi-splay zone") are dropped.
    KML LineString / GMT / CSV: the line vertices ('#'/'>' headers ignored;
    lat,lon column order auto-detected).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".kml":
        txt = re.sub(r'xmlns="[^"]+"', "", open(path).read())
        root = ET.fromstring(txt)
        pt_pms = [pm for pm in root.findall(".//Placemark")
                  if pm.find(".//Point") is not None]
        if pt_pms:                                   # KML of crossing points
            pts, dropped = [], []
            for pm in pt_pms:
                name = pm.findtext("name")
                c = pm.find(".//coordinates")
                lo, la = map(float, c.text.strip().split(",")[:2])
                if _is_number(name):
                    pts.append((lo, la))
                else:
                    dropped.append(name)
            if dropped:
                print(f"dropped {len(dropped)} non-numeric placemark(s): {dropped}")
            return np.array(pts)
        m = re.search(r"<coordinates>(.*?)</coordinates>", txt, re.S)
        if not m:
            sys.exit(f"no Points or <coordinates> in {path}")
        return np.array([tuple(map(float, tok.split(",")[:2]))
                         for tok in m.group(1).split() if tok.strip()])
    # GMT / CSV / txt: two numeric columns per line
    pts = []
    for ln in open(path):
        if ln.startswith(("#", ">", "@")):
            continue
        a = ln.replace(",", " ").split()
        if len(a) >= 2:
            try:
                pts.append((float(a[0]), float(a[1])))
            except ValueError:
                continue
    p = np.array(pts)
    if np.nanmedian(p[:, 0]) > 0 and np.nanmedian(p[:, 1]) < 0:   # cols are lat,lon
        p = p[:, ::-1]
    return p
