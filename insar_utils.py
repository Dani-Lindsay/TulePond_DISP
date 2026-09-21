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
    Along-fault axis     km from Point Pinole along the mapped trace
    Statistics           robust sigma, detrend, linear rate with std_err
    Creepmeters          USGS .10min reader
    Provenance/Registry  git hash, file fingerprints, the registry class
"""
import csv
import hashlib
import os
import subprocess
import uuid
from datetime import datetime

import numpy as np                 # Matrix calculations
import pandas as pd                # Tables
import h5py                        # MintPy HDF5 products
from scipy import stats            # linregress
import geopy.distance              # geodesic distance along the fault trace


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


# ------------------------------
# Statistics
# ------------------------------

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


# ------------------------------
# Catalogues (seismicity and repeating earthquakes), used by 05 and 10
# ------------------------------

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
        with open(self.txt_path, "w") as f:
            f.write(f"# TulePond_DISP number registry, rendered {datetime.now():%Y-%m-%d %H:%M}\n")
            f.write("# One line per key: key = rendered units (precision) -> where [notebook/cell; git; date]\n\n")
            for nb, grp in latest.groupby("notebook"):
                f.write(f"## {nb}\n")
                for _, r in grp.sort_values("key").iterrows():
                    f.write(f"{r['key']} = {r['rendered']} {r['units']} ({r['precision']} dp) -> {r['where']}"
                            f"  [{r['cell']}; git {r['git_hash']}; {str(r['timestamp'])[:10]}]\n")
                f.write("\n")
            f.write("## By manuscript location\n")
            for where, grp in latest.groupby("where"):
                f.write(f"{where}: " + ", ".join(f"{r['key']}={r['rendered']}" for _, r in grp.iterrows()) + "\n")
        print(f"Rendered registry to: {self.txt_path}")
        return self.txt_path
