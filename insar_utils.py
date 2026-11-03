#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Danielle Lindsay

insar_utils.py: shared helper functions of the TulePond_DISP repository: reading MintPy HDF5 attributes, dates,
the LOS projection conventions, the along-fault distance axis and the rotated fault maps, statistics, creepmeter
and catalogue readers, and the provenance and number-registry helpers.

Most functions here are shared by several notebooks or scripts; the few used by one notebook belong with a shared
convention (the time and LOS conventions, the readers, the rotated fault maps). Functions specific to one step of
the analysis are written in its notebook. Most have a test in tests/.

Sections
    Disk space, dates    disk-space check before downloads; decimal years
    Read                 MintPy attributes (h5py)
    Projections          LOS unit vector, fault-parallel design matrix, decomposition of series and of maps
    pygmt helpers        resampling to lon/lat grids, map sizes, colour tables
    Along-fault axis     km from Point Pinole along the mapped trace; the map strike frame (u, v); the fault map panel
    Figure style         matplotlib paper style, panel titles, pygmt labels
    Statistics           robust sigma, detrend, linear rate, the joint-model design matrix, sigma from a 95 % interval,
                         along-trace smoothing, annual-increment rate
    Creepmeters          USGS .10min reader; the daily record with listed glitch days removed
    Catalogues           seismicity, repeating-earthquake and alignment-array readers
    Provenance/Registry  git hash, file fingerprints, the registry class
    Fault trace          KML / text trace reader
"""
import csv
import hashlib
import json
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
import pyproj                      # lon/lat <-> UTM for the strike frame (strike_frame)


# ------------------------------
# Disk space and date-time
# ------------------------------

RESERVE_GB = 50.0      # GB of free disk space that a download must leave untouched


def require_disk_space(path, need_gb, what=""):
    """Exit (SystemExit) unless the disk holding `path` has room for `need_gb`
    (GB, an estimate of the job size) plus RESERVE_GB; `what` names the job in
    the message. Returns True otherwise. The download scripts in `data_prep/`
    call it before starting. Source: written for this project.
    """
    total, used, free = (x / 1e9 for x in shutil.disk_usage(path))
    label = f" for {what}" if what else ""
    print(f"  disk: {free:.0f} GB free of {total:.0f} GB; "
          f"need ~{need_gb:.1f} GB{label} (+{RESERVE_GB:.0f} GB reserve)")
    if free - need_gb >= RESERVE_GB:
        return True
    raise SystemExit(f"ABORT: insufficient disk space{label}: {free:.0f} GB free, "
                     f"need ~{need_gb:.1f} GB plus a {RESERVE_GB:.0f} GB reserve.")


# The one time convention of the project: a time t (UTC) is the decimal year
#     y = 2000 + (t - 2000-01-01 00:00 UTC) / 365.25 d,
# a uniform year of 365.25 days (the convention of the UNR GNSS series). A difference of two decimal years times
# 365.25 is then the exact number of days between them. Every notebook converts through the functions below.
EPOCH_2000 = pd.Timestamp("2000-01-01 00:00:00")
DAYS_PER_YEAR = 365.25


def decimal_years(times):
    """Decimal years of UTC times (array-like of datetimes, Timestamps or date strings), as a numpy array.

    Times without a time zone are taken as UTC; times with one are converted to UTC.
    """
    times = pd.DatetimeIndex(pd.to_datetime(times))
    if times.tz is not None:
        times = times.tz_convert("UTC").tz_localize(None)
    return 2000.0 + np.asarray((times - EPOCH_2000) / pd.Timedelta(days=1), float) / DAYS_PER_YEAR


def from_calendar_decimal_years(values):
    """Calendar decimal years (year + elapsed fraction of that year's length, 1 January 00:00 = year + 0), the
    convention of external files, in the project convention of `decimal_years`.

    Parameters
    ----------
    values : array-like of float

    Returns
    -------
    numpy.ndarray
    """
    values = np.asarray(values, float)
    whole_year = np.floor(values).astype(int)
    is_leap = ((whole_year % 4 == 0) & (whole_year % 100 != 0)) | (whole_year % 400 == 0)
    first_of_year = pd.DatetimeIndex(pd.to_datetime(pd.DataFrame(dict(year=whole_year, month=1, day=1))))
    times = first_of_year + pd.to_timedelta((values - whole_year) * np.where(is_leap, 366.0, 365.0), unit="D")
    return decimal_years(times)


def date_to_decimal_year(date_input, format_str=None):
    """Decimal year of one date or time (str, bytes or datetime), UTC.

    MintPy stores dates as bytes b'20160717'; those are accepted directly. Strings are parsed with
    `format_str`, or as YYYY-MM-DD, MM/DD/YYYY or YYYYMMDD when it is None. A date without a time is
    00:00 UTC of that day.
    """
    if isinstance(date_input, bytes):
        date_input = date_input.decode("utf-8")
    if isinstance(date_input, (datetime, pd.Timestamp)):
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
    return float(decimal_years([date])[0])


def decimal_year_to_date(dec_year):
    """Inverse of date_to_decimal_year, to the nearest second. Returns a datetime (UTC, no time zone)."""
    seconds = (dec_year - 2000.0) * DAYS_PER_YEAR * 86400.0
    return (EPOCH_2000 + pd.Timedelta(seconds=round(seconds))).to_pydatetime()


def decimal_year_to_day(dec_year):
    """The UTC calendar day of an acquisition epoch stored at midnight, as a Timestamp at midnight.

    Rounds to the nearest midnight, so that an epoch stored to a few decimal places (notebook 03's columns, to 4
    places) gives its own day. For a time of day (an exact acquisition time, a forecast date) use
    `decimal_year_to_utc_date`."""
    return pd.Timestamp(decimal_year_to_date(dec_year)).round("D")


def decimal_year_to_utc_date(dec_year):
    """The UTC calendar day that contains a time given as a decimal year, as a Timestamp at midnight."""
    return pd.Timestamp(decimal_year_to_date(dec_year)).floor("D")


def acquisition_times(stack_file):
    """UTC time of every acquisition in a rebased OPERA DISP-S1 stack (`<frame>_stack.nc`, data_prep run01).

    The stack's `time` coordinate holds the secondary acquisition time of each layer, the zero-Doppler start time
    of the OPERA product (`identification/secondary_datetime` of each granule).

    Parameters
    ----------
    stack_file : str
        The stack (netCDF).

    Returns
    -------
    pandas.Series
        UTC time of each acquisition, indexed by its UTC day.
    """
    import xarray as xr
    with xr.open_dataset(stack_file) as stack:
        times = pd.DatetimeIndex(stack["time"].values)
    assert times.is_monotonic_increasing and times.normalize().is_unique, f"{stack_file}: times not one per day, in order"
    return pd.Series(times, index=times.normalize())


# ------------------------------
# Read (MintPy HDF5)
# ------------------------------

def load_h5_attrs(h5_file):
    """Return the MintPy attributes of a file as a plain dict of strings."""
    with h5py.File(h5_file, "r") as hf:
        return {k: (v.decode() if isinstance(v, bytes) else str(v)) for k, v in hf.attrs.items()}

# ------------------------------
# Projections
#
# Conventions (MintPy / OPERA), tested against OPERA's own los_east and
# los_north rasters in tests/test_insar_utils.py:
#   inc_deg  incidence angle of the LOS vector, measured from vertical
#   azi_deg  azimuth of the ground -> satellite LOS vector, from north,
#            anticlockwise positive. Tracks used here: ascending ~100
#            (satellite to the west), descending ~260 (satellite to the east).
#   strike   compass bearing, from north, clockwise positive. Hayward ~144.
# The LOS azimuth (anticlockwise) and the strike bearing (clockwise) use
# opposite conventions and must not be mixed.
# ------------------------------

def los_unit_vector(inc_deg, azi_deg):
    """Unit vector (E, N, U) of the LOS direction from the ground to the satellite,
    from the incidence and azimuth angles (deg, MintPy convention above)."""
    inc, azi = np.deg2rad(inc_deg), np.deg2rad(azi_deg)
    return np.array([
        -np.sin(inc) * np.sin(azi),   # East
        np.sin(inc) * np.cos(azi),    # North
        np.cos(inc),                  # Up
    ])


def horizontal_unit_vector(bearing_deg):
    """Unit vector (E, N, 0) of a horizontal direction given as a compass
    bearing in deg clockwise from north (0 = N, 90 = E, 180 = S, 270 = W)."""
    b = np.deg2rad(bearing_deg)
    return np.array([np.sin(b), np.cos(b), 0.0])


def design_matrix_up_fp(inc_deg, azi_deg, strike_deg):
    """Design matrix G mapping [U, FP] -> LOS, one row per track. The same
    matrix as MintPy's get_design_matrix4horz_vert (Yunjun et al. 2019) with
    the horizontal direction set to the fault strike:

        d_los_i = U * cos(inc_i) + FP * sin(inc_i) * cos(azi_i - horz_az),    horz_az = -strike

    inc is the incidence angle from vertical, azi the azimuth of the ground ->
    satellite LOS vector (anticlockwise from north, MintPy convention) and
    strike a compass bearing (clockwise from north), so the strike enters
    MintPy's anticlockwise horizontal azimuth as -strike. The two columns are
    the dot products of the LOS unit vector with up and with the strike unit
    vector (los_unit_vector, horizontal_unit_vector); the two forms are
    asserted equal in tests/test_insar_utils.py.

    Assumes all horizontal motion is fault-parallel (the two-look assumption).
    Returns G with shape (n_tracks, 2), columns [U, FP].
    Source: written for this project, after MintPy.
    """
    inc = np.deg2rad(np.atleast_1d(inc_deg).astype(float))
    azi = np.deg2rad(np.atleast_1d(azi_deg).astype(float))
    horz_az = np.deg2rad(-float(strike_deg))
    G = np.column_stack([np.cos(inc), np.sin(inc) * np.cos(azi - horz_az)])
    return G


def decompose_asc_desc(d_asc, d_desc, inc_asc_deg, azi_asc_deg,
                       inc_desc_deg, azi_desc_deg, strike_deg):
    """Decompose ascending + descending LOS series into vertical (U) and
    fault-parallel (FP), solving the 2x2 system per epoch by least squares.

    d_asc / d_desc must be LOS values on a common set of epochs. FP is positive
    along `strike_deg`: with strike ~144 (S36E), the east block moving SE
    relative to the west block (right-lateral Hayward creep) gives positive FP
    when the series is (east side - west side). Angles in deg.

    Returns (U, FP), each the length of the input series, in the units of the input.
    """
    G = design_matrix_up_fp([inc_asc_deg, inc_desc_deg],
                            [azi_asc_deg, azi_desc_deg], strike_deg)
    D = np.vstack([np.asarray(d_asc, dtype=float),
                   np.asarray(d_desc, dtype=float)])   # (2, n_epochs)
    UV_FP = np.linalg.lstsq(G, D, rcond=None)[0]       # (2, n_epochs)
    return UV_FP[0, :], UV_FP[1, :]


def decompose_asc_desc_maps(los_asc, los_desc, inc_both, azi_both, strike_deg, window_px):
    """Fault-parallel and vertical maps from the ascending and descending LOS maps, with MintPy's
    `asc_desc2horz_vert` (Yunjun et al. 2019). Used by notebook 06 and model_utils.fault_parallel_from_steps (07).

    Parameters
    ----------
    los_asc, los_desc : numpy.ndarray
        (n_rows, n_cols) LOS maps (mm or mm/yr, positive toward the satellite); only pixels valid on both are used.
    inc_both, azi_both : numpy.ndarray
        (2, n_rows, n_cols) incidence and azimuth angles (deg, MintPy convention above), ascending first.
    strike_deg : float
        Strike (deg, compass bearing); MintPy's horizontal direction is horz_az_angle = -strike_deg, as MintPy counts
        anticlockwise.
    window_px : int
        MintPy's decomposition window (`step`, pixels).

    Returns
    -------
    fault_parallel, up : numpy.ndarray
        (n_rows, n_cols) components along the strike and vertical, in the units of the input (float64).
    """
    from mintpy.asc_desc2horz_vert import asc_desc2horz_vert
    valid_both = np.isfinite(los_asc) & np.isfinite(los_desc)
    los_stack = np.stack([los_asc, los_desc]).astype(np.float32)
    los_stack[:, ~valid_both] = np.nan
    fault_parallel, up = asc_desc2horz_vert(los_stack, inc_both, azi_both, horz_az_angle=-strike_deg, step=window_px)
    return fault_parallel.astype(float), up.astype(float)


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
    """[W, E, S, N] in lon/lat (deg) enclosing a north-up UTM raster (x0, y0 the
    corner of the first pixel, dx, dy the pixel size in m, width x length pixels)."""
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
    """Write a GMT colour table with n slices between vmin and vmax sampled from
    a matplotlib colormap, so that the pygmt figures use the same colours as
    the matplotlib figures. Returns path."""
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

def lat_to_km_map(profiles_csv):
    """Return a function lat (deg) -> km from Point Pinole, by nearest vertex of
    the trace in profiles_csv (column ft_lat). The trace is sampled every 100 m,
    so km = (n-1 - index) * 0.1 to within 10 m (tested in tests/test_filepaths.py)."""
    prof = pd.read_csv(profiles_csv)
    plat = prof["ft_lat"].to_numpy(float)
    km = (len(plat) - 1 - np.arange(len(plat))) * 0.1
    return lambda lats: km[np.abs(plat[None, :] - np.atleast_1d(np.asarray(lats, float))[:, None]).argmin(axis=1)]


def strike_frame(epsg, strike_deg, trace_lon, trace_lat):
    """The straight strike frame of the rotated map panels (notebooks 04, 05,
    11 and 14): u along the strike from Point Pinole (km, positive toward the
    strike bearing, i.e. south-east), v across the fault (km, positive to the
    north-east = the east side), both in the UTM zone `epsg`; strike_deg is a
    compass bearing (deg), trace_lon / trace_lat the trace vertices (deg).

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
    """Strike-frame km (u along, v across; see strike_frame) -> (lon, lat) in deg,
    for the map elements defined in the frame (map corners, ticks, brackets)."""
    u_m = np.asarray(u_km, float) * 1e3 + frame["u_origin"]
    v_m = np.asarray(v_km, float) * 1e3 + frame["v_origin"]
    easting = u_m * frame["along_e"] + v_m * frame["cross_e"]
    northing = u_m * frame["along_n"] + v_m * frame["cross_n"]
    return frame["from_utm"].transform(easting, northing)


# The rotated fault maps (Fig 1A, the 05 and 11 supplementary figures) all show the same rectangle of the strike frame
FAULT_MAP_U_KM = (-3.0, 70.0)     # along strike: from 3 km north of Point Pinole to just south of CFW1
FAULT_MAP_HALF_KM = 10.9          # across the fault, each side (3.4 cm wide for 73 km tall at true scale)


def fault_map_corners(frame, u_range_km=FAULT_MAP_U_KM, half_km=FAULT_MAP_HALF_KM):
    """The rectangle of the rotated fault maps in the strike frame (see strike_frame): u from
    u_range_km[0] to u_range_km[1] (default the whole fault, FAULT_MAP_U_KM), v within +/- half_km of the trace.
    Returns (the GMT region in its lower-left/upper-right '+r' form, (lon, lat) of the centre,
    the four corners as a (4, 2) array of lon, lat). A grid drawn on the map must cover the corners."""
    u_north, u_south = u_range_km
    lon_ll, lat_ll = uv_to_lonlat(frame, u_south, -half_km)          # lower-left = south end, west side
    lon_ur, lat_ur = uv_to_lonlat(frame, u_north, half_km)           # upper-right = north end, east side
    lon_c, lat_c = uv_to_lonlat(frame, 0.5 * (u_north + u_south), 0.0)
    corners = []
    for corner_u, corner_v in ((u_south, -half_km), (u_south, half_km), (u_north, half_km), (u_north, -half_km)):
        corners.append(uv_to_lonlat(frame, corner_u, corner_v))
    region = f"{float(lon_ll)}/{float(lat_ll)}/{float(lon_ur)}/{float(lat_ur)}+r"
    return region, (float(lon_c), float(lat_c)), np.array(corners, float)


def creepmeter_u_km(frame, trace_lon, trace_lat, creepmeters):
    """Position of each creepmeter along the rotated fault maps: u (km) of the trace vertex nearest to it.
    The instruments lie on the trace to within a few tens of metres. creepmeters is the
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
    """A UTM-gridded field sampled on a regular lon/lat grid (lon_step, lat_step in deg) that covers the map corners
    (plus 0.01 deg), by nearest UTM pixel, NaN off the UTM grid (GMT draws regular lon/lat grids). x_first/y_first
    are the pixel-centre coordinates of the first column/row (m), x_step/y_step the pixel size (m; y_step < 0 for
    north-up); to_utm is a pyproj transformer lon/lat -> UTM. Returns (lon_axis, lat_axis, values on the lon/lat grid,
    lat x lon)."""
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
                    letter="A", title="Hayward Fault", colorbar_frame=None, extra_symbols=(), extra_labels=(),
                    u_range_km=FAULT_MAP_U_KM, half_km=FAULT_MAP_HALF_KM, scale_km=5):
    """Draw the rotated fault map of Fig 1A at the current origin of a pygmt figure. The same panel is used
    by 14 (Fig 1A), 05 (the validation supplementary figure) and 11 (the alignment-array supplementary
    figure); only the field, its colours, the extra symbols and the panel letter differ.

    It draws, in this order: the field `grid` (an xarray lon/lat grid covering fault_map_corners) in
    the colour table `cpt`; the mapped trace; the extra symbols and their labels (lists of keyword dicts
    for fig.plot and text_backed); the creepmeter codes on the west side; the SSE extent bracket on the
    east side, from sse_u_km[0] to sse_u_km[1] (km along the strike frame) with its label; the city
    names on the east side; the creepmeters as yellow triangles on top; North and South; the panel letter
    and title above the top-left corner; the 5 km scale bar; the distance from Point Pinole down the
    left edge; and, if colorbar_frame is given (e.g. "xa2f1+lFault-parallel velocity (mm/yr)"), a 2.5 cm
    colour bar centred under the map.

    The map is GMT's oblique Mercator with its x axis along strike - 90 deg, so the fault runs up the
    page with north at the top, at true scale: the height is map_h_cm and the width follows from it.
    Fonts and frame pens are the caller's pygmt.config. u_range_km and half_km crop the map to part of the
    fault (11 draws km 42-70 only); creepmeters and city names outside the crop are left out. scale_km is the
    length of the scale bar (km). Returns the map
    width in cm.
    Source: written for this project (pygmt)."""
    import pygmt
    region, (lon_c, lat_c), corners = fault_map_corners(frame, u_range_km, half_km)
    u_north, u_south = u_range_km
    strike_deg = frame["strike_deg"]
    trace_u_km, trace_v_km = to_uv(frame, trace_lon, trace_lat)
    u_creepmeter, _ = creepmeter_u_km(frame, trace_lon, trace_lat, creepmeters)
    city_u_all = {"Berkeley": float(to_uv(frame, [-122.2727], [37.8716])[0][0]),
                  "Oakland": 0.5 * (u_creepmeter["CTM1"] + u_creepmeter["COZ1"]),
                  "Hayward": u_creepmeter["CHP1"],
                  "Union City": float(to_uv(frame, [-122.0439], [37.5934])[0][0]),
                  "Fremont": max(float(to_uv(frame, [-121.9886], [37.5485])[0][0]), sse_u_km[1] + 2.0)}
    whole_fault = tuple(u_range_km) == tuple(FAULT_MAP_U_KM)
    creepmeters = [c for c in creepmeters if u_north < u_creepmeter[c[0]] < u_south]     # those on this map
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
    fig.grdimage(grid=grid, cmap=cpt, nan_transparent=True, interpolation="c")   # -nc: bicubic resampling to the page (display only)
    fig.plot(x=trace_lon, y=trace_lat, pen="0.6p,black")
    for symbol in extra_symbols:                           # e.g. 11's alignment arrays, under the creepmeter labels
        fig.plot(**symbol)
    for label in extra_labels:
        text_backed(fig, **label)
    for code, lat, lon, name in creepmeters:               # labels on the west side; the triangles go on top, below
        text_backed(fig, x=lon, y=lat, text=code, justify="RM", offset="-0.15c/0c", font="7p,Helvetica,black")
    # the SSE extent bracket, east of the fault so it stays clear of the creepmeter labels, ticks reaching
    # almost to the trace; labelled in two rows (GMT has no newline outside paragraph mode), one size up
    bracket_v_km = 2.0 if whole_fault else 0.9          # a cropped (narrower) map keeps the label inside the frame
    bracket_lon, bracket_lat = uv_to_lonlat(frame, [sse_lo_km, sse_hi_km], [bracket_v_km, bracket_v_km])
    fig.plot(x=bracket_lon, y=bracket_lat, pen="1.1p,black")
    for u_tick in (sse_lo_km, sse_hi_km):
        tick_lon, tick_lat = uv_to_lonlat(frame, [u_tick, u_tick], [0.4, bracket_v_km])
        fig.plot(x=tick_lon, y=tick_lat, pen="1.1p,black")
    u_mid = 0.5 * (sse_lo_km + sse_hi_km)
    row_km = 1.0 * (u_south - u_north) / (FAULT_MAP_U_KM[1] - FAULT_MAP_U_KM[0])     # the rows 1 km apart on the whole-fault map, the same page gap on a crop
    label_v_km = bracket_v_km + (0.9 if whole_fault else 0.7)                        # left edge just east of the bracket
    label_lon, label_lat = uv_to_lonlat(frame, [u_mid - row_km, u_mid + row_km], [label_v_km, label_v_km])
    text_backed(fig, x=label_lon, y=label_lat, text=["SSE", "extent"], justify="LM", font="8p,Helvetica,black")
    # the cities along the fault, east side opposite the creepmeter labels, each within the city's stretch of the
    # fault (labels only, nothing is measured from them):
    #   Berkeley, Union City: the city centre's along-strike position
    #   Oakland: midway between CTM1 (Temescal) and COZ1 (Oakland Zoo), both in Oakland
    #   Hayward: at CHP1 (Palisade Street, Hayward)
    #   Fremont: its centre (Central Park) sits beside the SSE bracket, so just south of the bracket, still in the city
    # (city_u_all, above; on a cropped map only the cities inside the crop are named)
    city_u = {city: u_city for city, u_city in city_u_all.items() if whole_fault or u_north + 1.0 < u_city < u_south - 1.0}
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
    fig.basemap(region=region, projection=projection, frame=["lrtb"], map_scale=f"jBL+w{scale_km:g}k+o0.3c/0.4c")   # frame on top, scale bar

    # distance from Point Pinole down the left edge: the map's vertical axis is the strike frame u (true scale), so a
    # linear axis over the same box reads u directly. Asserted: GMT's own page y of the creepmeters matches the
    # linear axis to 0.16 km on the ground (0.02 cm on the whole-fault map; the creepmeters sit up to ~0.1 km from a vertex).
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
    axis_error_km = axis_error_cm * (u_south - u_north) / map_h_cm          # on the ground: the check holds for any crop
    assert axis_error_km < 0.16, f"distance axis off the map by {axis_error_km:.3f} km ({axis_error_cm:.3f} cm)"
    fig.basemap(region=[0, 1, u_north, u_south], projection=f"X{map_w_cm:.3f}c/-{map_h_cm}c", frame=["W", "ya10f5+lDistance from Point Pinole (km)"])
    if colorbar_frame is not None:
        with pygmt.config(FONT_ANNOT_PRIMARY="13p", FONT_LABEL="13p"):     # GMT scales colour-bar text down with the bar thickness (-> ~6.5p)
            fig.colorbar(cmap=cpt, position=f"x{0.5 * (map_w_cm - colorbar_w_cm)}c/-0.4c+w{colorbar_w_cm}c/0.3c+h+jTL", frame=colorbar_frame)
    return map_w_cm


# ------------------------------
# Figure style
# ------------------------------

def register_helvetica_faces():
    """Make Helvetica's bold and oblique faces available to matplotlib.

    matplotlib registers only the first face of a .ttc collection, so macOS's
    Helvetica.ttc provides Helvetica Regular alone and a bold panel letter
    would render in the regular face. The other faces are extracted once (with
    fontTools, a matplotlib dependency) into the matplotlib config directory
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
    text, and TrueType (Type 42) embedding so PDF text stays editable.
    Returns the resulting rcParams as a dict."""
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
        # legends sit on a white box so no line runs through their text
        "legend.frameon": True, "legend.facecolor": "white", "legend.edgecolor": "none", "legend.framealpha": 0.85,
        "legend.fontsize": max(base - 2, 6),
    })
    return dict(matplotlib.rcParams)


# Okabe-Ito colour-blind-safe categorical colours (Okabe & Ito 2008); no blue/green pairs
okabe_ito = ["#0072B2", "#E69F00", "#D55E00", "#CC79A7", "#000000", "#56B4E9", "#F0E442"]


def panel_title(ax, letter, text="", size=7):
    """Science panel label on a matplotlib axis: a bold capital letter, then a
    plain `size`-pt description, left-aligned above the axis."""
    label = rf"$\bf{{{letter}}}$" + (f"  {text}" if text else "")
    ax.set_title(label, loc="left", fontsize=size)


def text_backed(fig, transparency=30, **kw):
    """A pygmt map label on a translucent white box with opaque text: GMT applies the
    transparency (percent) to the glyphs as well as the box, so the label is drawn
    twice, first with the box and the transparency, then again with neither.
    kw: the arguments of pygmt Figure.text."""
    fig.text(fill="white", transparency=transparency, **kw)
    fig.text(**kw)


# ------------------------------
# Statistics
# ------------------------------

def robust_sigma(x):
    """Robust standard deviation via the median absolute deviation
    (1.4826 * MAD) of the finite values; NaN if there are none."""
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
    Returns (slope, intercept, std_err, n): the rate, intercept, the standard error
    of the rate, and the number of finite samples (NaN values with fewer than 3)."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan, np.nan, int(m.sum())
    slope, intercept, r_value, p_value, std_err = stats.linregress(t[m], y[m])
    return slope, intercept, std_err, int(m.sum())


def joint_design(dates, step_epochs):
    """Design matrix of the joint model of a creep series (notebook 04, section 3; also used by 05 and 11).

    Parameters
    ----------
    dates : numpy.ndarray
        Epochs (decimal years).
    step_epochs : sequence of float
        Step epoch of each event (decimal years); an empty sequence gives the model without steps.

    Returns
    -------
    numpy.ndarray
        (n_dates, 4 + n_steps). Columns: constant, epoch minus its mean (the rate), sin(2 pi t), cos(2 pi t) (the
        annual cycle), then one Heaviside step H(t > step epoch) per event.
    """
    columns = [np.ones_like(dates), dates - dates.mean(), np.sin(2 * np.pi * dates), np.cos(2 * np.pi * dates)]
    for step_epoch in step_epochs:
        columns.append((dates > step_epoch).astype(float))
    return np.column_stack(columns)


def sigma_from_ci95(ci_lo, ci_hi):
    """1-sigma from a 95 % interval, taken as Gaussian (the mean +/- 1.96 sigma): (hi - lo) / 3.92.

    Parameters
    ----------
    ci_lo, ci_hi : float or numpy.ndarray
        Bounds of the 95 % interval.

    Returns
    -------
    float or numpy.ndarray
        The 1-sigma, in the units of the bounds.
    """
    return (ci_hi - ci_lo) / 3.92


def gaussian_smooth_km(x, km_of_x, sigma_km):
    """NaN-aware, gap-preserving Gaussian smoothing along the trace. `km_of_x`
    is the along-trace km of each sample (km from Point Pinole, e.g. from
    `lat_to_km_map`), in fault order (strictly monotonic). The samples are
    placed on a regular grid of their median spacing (the 100 m crossings), so
    a missing crossing is a gap in the grid, not a closer neighbour, and
    sigma_km (km) is a true along-trace width. Weights are renormalised over
    the finite neighbours; NaN where less than 15 % of the kernel weight is
    data, and NaN wherever the input is NaN. Used by notebooks 03 and 04
    (Fig 1C). Source: written for this project (scipy gaussian_filter1d)."""
    x = np.asarray(x, float)
    km_of_x = np.asarray(km_of_x, float)
    assert km_of_x.shape == x.shape, f"km_of_x {km_of_x.shape} does not match x {x.shape}"
    assert np.isfinite(km_of_x).all(), "km_of_x has NaN: every sample needs an along-trace position"
    step_km = np.diff(km_of_x)
    assert (step_km > 0).all() or (step_km < 0).all(), "km_of_x is not strictly monotonic: samples must be in fault order"
    spacing_km = float(np.median(np.abs(step_km)))
    # the crossings are 0.1 km apart; a spacing of ~0.001 means latitudes were passed instead of km
    assert 0.02 <= spacing_km <= 2.0, (f"median spacing {spacing_km:.4g} km is not an along-trace spacing: pass km "
                                       f"(utils.lat_to_km_map), not latitude")
    node = np.round((km_of_x - km_of_x.min()) / spacing_km).astype(int)
    assert np.allclose(node * spacing_km + km_of_x.min(), km_of_x, rtol=0, atol=0.25 * spacing_km), \
        "the samples are not on a regular along-trace grid of their median spacing"
    grid = np.full(node.max() + 1, np.nan)                     # the regular along-trace grid, holes NaN
    grid[node] = x
    sig = max(sigma_km / spacing_km, 0.5)
    finite = np.isfinite(grid).astype(float)
    filled = np.where(finite > 0, grid, 0.0)
    den = gaussian_filter1d(finite, sig, mode="nearest")
    smooth = np.divide(gaussian_filter1d(filled, sig, mode="nearest"), den, out=np.full_like(filled, np.nan), where=den > 0.15)
    out = smooth[node]
    out[~np.isfinite(x)] = np.nan
    return out


def annual_increment_rate(t, y):
    """Rate and 1-sigma of a continuous record (creepmeter) from its slip gained
    in each whole year from the first sample: rate = mean(dy), sigma =
    std(dy, ddof=1) / sqrt(n). Daily samples are not independent (episodic
    creep), so the standard error of a linear regression understates the
    uncertainty; one-year increments are treated as the independent samples.
    t in decimal years. Returns (rate, sigma, n_years), in units of y per year."""
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
    """Path to the record of creepmeter `code`, preferring the `.10min.gz` file
    (read directly by pandas) over an uncompressed `.10min` file."""
    gz = os.path.join(creepmeter_dir, code.lower() + ".10min.gz")
    return gz if os.path.exists(gz) else os.path.join(creepmeter_dir, code.lower() + ".10min")


def creepmeter_decimal_year(year, doy):
    """Decimal year of USGS creepmeter samples. The fractional day-of-year is
    1-based: 1 January 00:00 is doy 1.000 (`cfw1.10min.gz`: the first sample of
    2020 is `2020 1.000000000`, the last of 2019 `2019 365.993055556`). The times
    are UTC (the site visits in the station headers fall at 16-23 h, working hours
    in California). Returns decimal years in the convention of `decimal_years`."""
    year = np.asarray(year).astype(int)
    doy = np.asarray(doy, float)
    is_leap = ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)
    days_in_year = np.where(is_leap, 366.0, 365.0)
    assert np.all((doy >= 1.0) & (doy < days_in_year + 1.0)), "creepmeter day-of-year outside 1 to days-in-year + 1"
    first_of_year = pd.to_datetime(pd.DataFrame(dict(year=year, month=1, day=1)))
    days_since_2000 = np.asarray((first_of_year - EPOCH_2000) / pd.Timedelta(days=1), float) + (doy - 1.0)
    return 2000.0 + days_since_2000 / DAYS_PER_YEAR


def read_creepmeter_10min(filepath, date_start=None, date_end=None):
    """Read a USGS creepmeter `.10min`/`.10min.gz` file (whitespace columns:
    year, fractional 1-based day-of-year, slip_mm; no header) into daily means
    with columns ['decimal_year', 'slip_mm'], zeroed to the first day, between
    date_start and date_end (decimal years, optional). The decimal year of a
    day is that of its first sample (`creepmeter_decimal_year`).

    The published values are already divided by cos(30 deg) at source, i.e.
    they are right-lateral fault-parallel slip; no obliquity correction is
    applied here.
    """
    df = pd.read_csv(filepath, sep=r"\s+", usecols=[0, 1, 2], header=None,
                     names=["year", "doy", "slip_mm"])
    df["decimal_year"] = creepmeter_decimal_year(df["year"].to_numpy(), df["doy"].to_numpy())
    if date_start is not None:
        df = df[df["decimal_year"] >= date_start]
    if date_end is not None:
        df = df[df["decimal_year"] <= date_end]
    df = df.reset_index(drop=True)
    # Group on the integer day: the doy column is fractional (one value per
    # 10-minute sample).
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
    (code, first day, last day, reason) rows, days as 'YYYY-MM-DD' calendar
    days (the day of a daily mean is the floor of its decimal year, which with
    the 1-based day-of-year conversion is the calendar day of its samples).
    Only the listed days are dropped (no automatic filter), and each listed
    day inside the window must be in the record. Prints the days dropped.
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


def read_creepmeter_10min_raw(filepath, date_start=None, date_end=None):
    """The same USGS `.10min` file at its native 10-minute sampling (no daily
    averaging): columns ['decimal_year', 'slip_mm'], as published, between
    date_start and date_end (decimal years, optional). Used where a creep-event
    step is measured (steps take hours; daily means spread them across two days)."""
    df = pd.read_csv(filepath, sep=r"\s+", usecols=[0, 1, 2], header=None,
                     names=["year", "doy", "slip_mm"])
    df["decimal_year"] = creepmeter_decimal_year(df["year"].to_numpy(), df["doy"].to_numpy())   # 1-based doy
    if date_start is not None:
        df = df[df["decimal_year"] >= date_start]
    if date_end is not None:
        df = df[df["decimal_year"] <= date_end]
    return df[["decimal_year", "slip_mm"]].reset_index(drop=True)


# ------------------------------
# Catalogues (seismicity, repeating earthquakes, alignment arrays)
# ------------------------------

def _decimal_year_from_ymdhms(yr, mo, dy, hr, mn, sc):
    """Vectorised decimal year (`decimal_years` convention) from UTC year/month/day/hour/minute/second columns."""
    t = pd.to_datetime(dict(year=yr, month=mo, day=dy, hour=hr, minute=mn), errors="coerce") + pd.to_timedelta(sc, unit="s")
    return 2000.0 + (t - EPOCH_2000).dt.total_seconds() / (DAYS_PER_YEAR * 86400.0)


def load_growclust_catalog(path, bbox=None):
    """GrowClust relocated catalogue of the Hayward area (2000-2025, B. Xie),
    relocated and unrelocated events: latitude/longitude/depth_km hold the
    relocation where the event clustered (nbranch > 1) and the NCSS location
    otherwise. Returns the common schema YYYY (decimal year), Lon, Lat, Depth
    (km), Mag, evID, source; clipped to bbox (W, S, E, N, deg) if given."""
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
    """One USGS Bay Area alignment array `code` (v3.0 archive in aa_dir): decimal
    year, cumulative dextral movement (mm), its standard deviation (mm), sorted
    by year. Two file layouts exist.

    The archive's decimal year (`Measurement_yr`, or `Year` in the older layout)
    is year + doy / days with a 1-based day of year (HRKT `1979.726, 9/22/1979`:
    265/365), so it falls one day after the survey day. Each archive year is
    turned into its calendar time, one day is subtracted, and the result is
    returned in the convention of `decimal_years` (a survey is placed at 00:00 of
    its day; the time of the survey is not recorded). The survey-date column
    (`Measurement_date`, or the third column of the older layout) is used as a
    check: over each file the median of (archive time - date) must lie within
    0.5 to 1.5 d (asserted, so an archive issued in the other convention fails).
    Rows where the corrected time and the date still disagree by more than 1.5 d
    are inconsistencies within the archive (e.g. HAPP 2024.68 against 10/5/2024);
    they keep the archive's time and are printed."""
    d = pd.read_csv(os.path.join(aa_dir, f"{code}.csv"), encoding="latin-1")
    d = d.rename(columns={c: c.replace("\xa0", " ").strip() for c in d.columns})
    if "Measurement_yr" in d.columns:
        date_column = "Measurement_date"
        d = d[["Measurement_yr", "Cumulative_dextral_movement_mm", "Dextral_movement_st_dev_mm", date_column]]
    elif "Year" in d.columns:
        date_column = d.columns[2]                                  # the unnamed survey-date column after Year
        d = d[["Year", "Cumulative movement (mm)", "Std. dev.", date_column]]
    else:
        raise ValueError(f"{code}: unrecognised alignment-array file layout ({list(d.columns)[:4]}...)")
    d.columns = ["year", "cum_mm", "std_mm", "date"]
    for column in ("year", "cum_mm", "std_mm"):
        d[column] = pd.to_numeric(d[column], errors="coerce")
    d = d.dropna(subset=["year", "cum_mm"]).copy()
    d["std_mm"] = d["std_mm"].fillna(0.0).clip(lower=0.0)
    # the archive's calendar decimal year (1-based day of year) -> its calendar time, one day earlier
    archive_year = d["year"].to_numpy()
    whole_year = np.floor(archive_year).astype(int)
    is_leap = ((whole_year % 4 == 0) & (whole_year % 100 != 0)) | (whole_year % 400 == 0)
    first_of_year = pd.DatetimeIndex(pd.to_datetime(pd.DataFrame(dict(year=whole_year, month=1, day=1))))
    archive_time = first_of_year + pd.to_timedelta((archive_year - whole_year) * np.where(is_leap, 366.0, 365.0), unit="D")
    d["year"] = decimal_years(archive_time - pd.Timedelta(days=1))
    # the check against the survey dates
    survey_day = pd.DatetimeIndex(pd.to_datetime(d["date"].astype(str).str.strip(), format="%m/%d/%Y", errors="coerce"))
    has_date = survey_day.notna()
    assert has_date.sum() >= 0.9 * len(d), f"{code}: only {has_date.sum()} of {len(d)} surveys have a readable date"
    archive_minus_date_d = np.asarray((archive_time[has_date] - survey_day[has_date]) / pd.Timedelta(days=1), float)
    median_offset_d = float(np.median(archive_minus_date_d))
    assert 0.5 <= median_offset_d <= 1.5, (f"{code}: the archive year runs {median_offset_d:+.2f} d from the survey dates; "
                                           f"the 1-based day-of-year correction assumes ~ +1 d")
    disagree = np.abs(archive_minus_date_d - 1.0) > 1.5
    if disagree.any() or (~has_date).any():
        listed = ", ".join(f"{year:.3f} vs {day}" for year, day in zip(archive_year[has_date][disagree], d["date"].to_numpy()[has_date][disagree]))
        print(f"  AA {code}: {disagree.sum()} surveys whose archive year and date disagree by > 1.5 d after the 1-day "
              f"correction (kept at the archive year){': ' + listed if listed else ''}; {(~has_date).sum()} without a readable date")
    return d[["year", "cum_mm", "std_mm"]].sort_values("year").reset_index(drop=True)


def load_aa_summary(aa_summary, fault="Hayward"):
    """The alignment-array archive's summary table (site, lon, lat, long-term
    rates, record span), rows of `fault` only."""
    s = pd.read_csv(aa_summary, encoding="latin-1")
    s.columns = [("site_code" if c.endswith("site_code") else c.replace("\xa0", " ").strip()) for c in s.columns]
    return s[s["fault"] == fault].reset_index(drop=True)


def load_cre_taira(path):
    """Repeating-earthquake catalogue of T. Taira (BSL), HFRC linkage clusters,
    GEN3. Whitespace (or comma) separated: YYYY Lat Lon Depth cumD(cm) EVID CSID
    Mag D(cm). YYYY is a calendar decimal year; it is returned in the convention
    of `decimal_years`. Returns YYYY, Lon, Lat, Depth, Mag, seqID, cat."""
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
    out = out.dropna(subset=["YYYY"]).reset_index(drop=True)
    out["YYYY"] = from_calendar_decimal_years(out["YYYY"])      # the catalogue's calendar decimal years -> decimal_years
    out["cat"] = "Taira26"
    return out


def load_cre_waldhauser(path):
    """Waldhauser & Schaff (2021) repeaters: '#' sequence headers (last field =
    sequence id) followed by one line per event; 39-line file header. Returns
    YYYY, Lon, Lat, Depth, Mag, seqID, cat."""
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
    """Add `dist_fault_km` (signed, east positive) and `ft_lat` (latitude of the
    nearest trace vertex) to a catalogue, on a local tangent plane. trace: a
    DataFrame with ft_lon, ft_lat running south to north. Returns a copy."""
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
# Every number quoted in the manuscript is registered once, by the notebook
# that computes it, into outputs/registry.csv (an append-only log with the
# notebook, cell, input fingerprints and git hash of each value) and written
# to outputs/paper_numbers.md: the current value of every number, one table
# per notebook.
#
# Re-running a notebook that produces the same numbers adds identical rows.
# A run that changes a registered value raises RegistryConflict unless the
# environment variable TULEPOND_SUPERSEDE=1 is set.
# ------------------------------

REGISTRY_COLUMNS = ["key", "value", "units", "precision", "rendered", "where",
                    "notebook", "cell", "sidecar", "inputs", "git_hash", "git_dirty",
                    "timestamp", "run_id"]


class RegistryConflict(Exception):
    """A registered number changed at its stated precision."""


def git_hash(repo_dir):
    """Short git hash of the repo and whether the tree has uncommitted changes.

    Under run_all.sh the hash and the tree state are those recorded before the first notebook ran
    (TULEPOND_RUN_COMMIT, TULEPOND_RUN_CLEAN): the notebooks are executed in place, so the tree is dirty after the first
    one whatever the state of the code."""
    if os.environ.get("TULEPOND_RUN_COMMIT"):
        return os.environ["TULEPOND_RUN_COMMIT"], os.environ.get("TULEPOND_RUN_CLEAN") != "1"
    try:
        h = subprocess.run(["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", repo_dir, "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip() != ""
        return h, dirty
    except Exception:
        return "nogit", True


def notebook_source(path):
    """The code of a notebook: its code cells joined in order, as one Python source string.

    The notebooks are the only copy of the analysis code (no paired .py scripts since 2 Nov 2026), so the tests, the
    deposit packer and the extension check read the code from here.

    Parameters
    ----------
    path : str
        Path to an .ipynb file.

    Returns
    -------
    str
    """
    with open(path) as fh:
        cells = json.load(fh)["cells"]
    code = []
    for cell in cells:
        if cell["cell_type"] == "code":
            source = cell["source"]
            code.append(source if isinstance(source, str) else "".join(source))
    assert code, f"{path} has no code cells"
    return "\n\n".join(code)


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


_UNIT_LABELS = {"1": "", "count": "", "date": "", "fraction": "", "Mw": "", "pct": "%", "decyear": "decimal year",
                "d": "days", "m3": "m^3", "sigma": "sigma"}
_UNIT_SUFFIXES = ("_mm_yr", "_mm", "_km", "_m3", "_m", "_pct", "_mpa", "_kpa", "_gpa", "_deg", "_yr", "_days", "_d")


def _notebook_title(notebook):
    """'04_event_detection' -> '04 Event detection'."""
    number, _, name = notebook.partition("_")
    return f"{number} {name.replace('_', ' ').capitalize().replace('3d', '3D')}"


def _quantity_label(key, units):
    """A readable name for a registry key: notebook prefix and unit suffix dropped, underscores as spaces."""
    name = key.partition("_")[2] or key
    for suffix in _UNIT_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            name = name[: -len(suffix)]
            break
    name = name.replace("_", " ")
    return name[:1].upper() + name[1:]


def _unit_label(units):
    """The unit column of paper_numbers.md."""
    return _UNIT_LABELS.get(str(units), str(units))


class Registry:
    """Registry of the manuscript numbers of one notebook run.

        REG = utils.Registry("05_event_detection", common_paths, repo_dir)
        REG.inputs(path1, path2)                       # files this notebook read
        REG.register("05_n_events", 5, "count", 0, where="P3", cell="detector")
        REG.sidecar("events_stats", ["n events: 5", ...])
        REG.render()                                   # rewrite paper_numbers.md
    """

    def __init__(self, notebook, common_paths, repo_dir):
        self.notebook = notebook
        self.common_paths = common_paths
        self.csv_path = common_paths["registry_csv"]
        self.numbers_path = common_paths["paper_numbers"]
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
        provenance header (notebook, git, time; then this run's id; then the
        input fingerprints). The header is also the record that the run
        happened: render() uses it to retire the keys of a notebook whose
        latest run registered nothing. Returns the path (relative to outputs/)."""
        path = os.path.join(self.out_dir, f"{name}_stats.txt")
        with open(path, "w") as f:
            f.write(f"# {provenance_stamp(self.notebook, self.repo_dir)}\n")
            f.write(f"# run {self.run_id}\n")                  # render() reads it: a run that wrote this and registered nothing
            f.write(f"# inputs: {'; '.join(self._inputs)}\n")
            for line in lines:
                f.write(str(line).rstrip("\n") + "\n")
        print(f"Saved stats to: {path}")
        return os.path.relpath(path, os.path.dirname(self.csv_path))

    def render(self):
        """Rewrite paper_numbers.md: the current value of every registered number, one table per notebook, in the
        order the notebook registers them. A number is current if its notebook is in the repo, is not retired from
        the paper (common_paths["retired_notebooks"]) and its latest run registered it; other rows of registry.csv
        are left out of the file. Returns the path."""
        rows = pd.read_csv(self.csv_path, dtype=str)
        rows["row"] = np.arange(len(rows))
        latest = rows.groupby("key", as_index=False).last()
        current = {os.path.splitext(n)[0] for n in os.listdir(self.repo_dir) if n.endswith(".ipynb")}
        last_run = rows.sort_values("timestamp").groupby("notebook")["run_id"].last()
        in_last_run = latest.apply(lambda r: r["run_id"] == last_run.get(r["notebook"]), axis=1).astype(bool)
        # a run that wrote its sidecar but registered nothing retires all of its notebook's numbers
        silent = self._notebooks_whose_latest_run_registered_nothing(rows)
        retired_by_decision = set(self.common_paths.get("retired_notebooks", ()))
        is_live = (latest["notebook"].isin(current) & in_last_run & ~latest["notebook"].isin(silent)
                   & ~latest["notebook"].isin(retired_by_decision))
        live = latest[is_live].sort_values("row")
        for nb in sorted(silent):
            print(f"  {nb}: its latest run registered no numbers; its numbers are left out")

        with open(self.numbers_path, "w") as f:
            f.write("# Numbers of the paper, from the notebooks\n\n")
            f.write(f"Written {datetime.now():%Y-%m-%d %H:%M} from the latest run of each notebook "
                    f"({len(live)} numbers). Values are rounded to the precision each number was registered with; "
                    "the key finds the number in its notebook and in `registry.csv` (full value, inputs, git hash).\n\n")
            f.write("| Notebook | Numbers | Run | Git |\n|---|---:|---|---|\n")
            for nb, grp in live.groupby("notebook"):
                f.write(f"| [{_notebook_title(nb)}](#{nb.replace('_', '-')}) | {len(grp)} | "
                        f"{str(grp['timestamp'].max())[:16].replace('T', ' ')} | {grp['git_hash'].iloc[-1]} |\n")
            for nb, grp in live.groupby("notebook"):
                f.write(f"\n<a id=\"{nb.replace('_', '-')}\"></a>\n## {_notebook_title(nb)}\n\n")
                f.write("| Quantity | Value | Unit | Cell | Key |\n|---|---:|---|---|---|\n")
                for _, r in grp.iterrows():
                    f.write(f"| {_quantity_label(r['key'], r['units'])} | {r['rendered']} | {_unit_label(r['units'])} | "
                            f"{r['cell']} | `{r['key']}` |\n")
        print(f"Wrote the numbers to: {self.numbers_path}")
        return self.numbers_path

    def _notebooks_whose_latest_run_registered_nothing(self, rows):
        """Notebooks whose newest sidecar (outputs/<notebook>/*_stats.txt with that notebook's provenance header)
        comes from a run that registered no row. A sidecar from 2026-10-05 on carries its run id ('# run <id>',
        second line): that run registered nothing if the id is not in registry.csv and the sidecar is not older
        than the notebook's last registry row (the header time is to the minute). An older sidecar has no run id:
        it counts only if its minute is strictly after the minute of the last registry row (a run writes its
        sidecar before it registers, so the sidecar of the run that made the last rows is never later)."""
        outputs_root = os.path.dirname(self.out_dir)
        silent = set()
        for nb, nb_rows in rows.groupby("notebook"):
            last_row_minute = pd.Timestamp(nb_rows["timestamp"].max()).floor("min")
            registered_runs = set(nb_rows["run_id"])
            folder = os.path.join(outputs_root, nb)
            if not os.path.isdir(folder):
                continue
            for name in os.listdir(folder):
                if not name.endswith("_stats.txt"):
                    continue
                with open(os.path.join(folder, name)) as f:
                    first_line = f.readline()
                    second_line = f.readline()
                if not first_line.startswith(f"# {nb} | git "):
                    continue                                        # another notebook's (or an old name's) sidecar
                written = pd.Timestamp(first_line.rsplit("|", 1)[1].strip())
                if second_line.startswith("# run "):
                    run_id = second_line[len("# run "):].strip()
                    if run_id not in registered_runs and written >= last_row_minute:
                        silent.add(nb)
                elif written > last_row_minute:
                    silent.add(nb)
        return silent


# ---------------------------------------------------------------------------
# Fault trace file reader (written for this project); the hand-drawn trace
# KML is read with this function everywhere.
# ---------------------------------------------------------------------------
def _is_number(s):
    """True if `s` converts to a float."""
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def load_trace(path):
    """Vertices of the trace as one (N, 2) lon/lat array (deg).

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
