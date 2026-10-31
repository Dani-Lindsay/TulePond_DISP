#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Danielle Lindsay

creep_profiles.py: the cross-fault creep time series of notebook 03 (Methods M3), as functions shared by notebook 03
and the aperture sweep (data_prep/run07_aperture_sweep.py), so that both run one code. Notebook 03 calls them with
the parameters of TulePond_filepaths.py; run07 calls them with every aperture bin and coherence that notebook 05
scores. The notebook explains the method and the equations; each function here states what it does and its inputs
and outputs with units.

Sections
    Fault crossings      the 100 m crossings with the smoothed cross-fault azimuth, km and UTM position
    Extraction           per track: the MintPy products, the pixels of one crossing, the west/east split, the rate
                         mask and the east - west difference of the side medians (Li et al. 2023)
    Common date grid     gap-limited interpolation of both tracks onto the union of their dates
    Decomposition        the per-crossing geometry and strike, fault-parallel (FP) series by least squares
    Rates and flags      linear FP rate per crossing and the crossing-level landslide flag

Source: written for this project, following the cross-fault differencing of Li et al. (2023); the decomposition uses
insar_utils.design_matrix_up_fp and decompose_asc_desc.
"""
import json

import numpy as np                               # Matrix calculations
import pandas as pd                              # Tables
import h5py                                      # MintPy HDF5 products
from pyproj import Transformer                   # lon/lat -> UTM

import insar_utils as utils

UTM_EPSG = 32610         # UTM zone 10N, the grid of the frames
TREND_WINDOW = 21        # crossings (~2 km) in the running median used to flag landslide crossings
TREND_NSIGMA = 3.0       # a crossing whose FP rate departs from that trend by more than this (robust sigma) is dropped


# ------------------------------
# Fault crossings
# ------------------------------

def load_crossings(fault_profiles, az_smooth_n, utm_epsg=UTM_EPSG):
    """The 100 m crossings of the mapped trace, with the smoothed azimuth, along-trace km and UTM position.

    Parameters
    ----------
    fault_profiles : str
        `hayward_fault_profiles.csv` (notebook 01): `ft_lon, ft_lat, profile_az`, south to north, where
        `profile_az` is the cross-fault azimuth (deg, compass bearing toward the east side).
    az_smooth_n : int
        Crossings in the centred rolling median of the azimuth (at least 3 values).
    utm_epsg : int, optional
        EPSG code of the UTM grid of the frames.

    Returns
    -------
    x_fault : pandas.DataFrame
        The crossings, with the added columns `az_smooth` (deg), `km` (km from Point Pinole, from
        `utils.lat_to_km_map`), `easting` and `northing` (m).
    km_of : callable
        latitude -> km from Point Pinole (`utils.lat_to_km_map`).
    """
    x_fault = pd.read_csv(fault_profiles)
    x_fault["az_smooth"] = x_fault["profile_az"].rolling(az_smooth_n, center=True, min_periods=3).median()
    km_of = utils.lat_to_km_map(fault_profiles)      # lat -> km from Point Pinole
    x_fault["km"] = km_of(x_fault["ft_lat"].values)

    # UTM position of every crossing, for the pixel windows
    ll2utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    x_fault["easting"], x_fault["northing"] = ll2utm.transform(x_fault["ft_lon"].values, x_fault["ft_lat"].values)
    return x_fault, km_of


# ------------------------------
# Extraction, one track
# ------------------------------

def read_insar_data(dic, coh, dist_max, x_width, unit):
    """Open the MintPy products of one track.

    The displacement cube is not loaded: `ts` is the open h5py dataset and is
    read per crossing window. Close `ts_h5` when done.

    Parameters
    ----------
    dic : dict
        Track entry of `insar_dic_list` (file paths, orbit direction, frame).
    coh : float
        Temporal-coherence threshold; the mask read is `dic["mask_file"][coh]`.
    dist_max : float
        Outer distance from the trace (km); sets the window read around each crossing.
    x_width : float
        Half width along strike of each crossing window (km).
    unit : float
        Factor from the units of the products (m) to those of the analysis (mm).

    Returns
    -------
    dict
        Incidence and azimuth (deg), coherence mask, LOS velocity (mm/yr),
        the open time-series file and dataset, the dates (decimal years), the
        grid (`x_first`, `y_first`, `x_step`, `y_step` in m; `length`,
        `width` in pixels) and `win`, the half-width (pixels) of the window
        read around each crossing.
    """
    print(f"Loading {dic['ORBIT_DIRECTION']} {dic['Frame']}")
    with h5py.File(dic["geo_file"], "r") as hf:
        inc = hf["incidenceAngle"][:].astype(np.float64)     # stored float16; cast before any arithmetic
        azi = hf["azimuthAngle"][:].astype(np.float64)
    with h5py.File(dic["mask_file"][coh], "r") as hf:
        mask = hf["mask"][:].astype(bool)
    with h5py.File(dic["vel_file"], "r") as hf:
        vel = hf["velocity"][:].astype(np.float64) * unit     # mm/yr, the rate-mask discriminator
    ts_h5 = h5py.File(dic["ts_file"], "r")
    attrs = ts_h5.attrs
    ts_dates = np.array([utils.date_to_decimal_year(d.decode()) for d in ts_h5["date"][:]])
    x_step = float(attrs["X_STEP"])
    y_step = float(attrs["Y_STEP"])
    # half-width of the pixel window read around each crossing: the dist_max x x_width box
    # corners are hypot(dist_max, x_width) from the crossing, plus one pixel
    win = int(np.ceil(np.hypot(dist_max, x_width) * 1000.0 / abs(x_step))) + 1
    print(f"   cube {ts_h5['timeseries'].shape} (dates x rows x cols), {x_step:.0f} m grid, "
          f"coherence-{coh} mask keeps {100 * mask.mean():.1f}% of pixels")
    return {"inc": inc, "azi": azi, "mask": mask, "vel": vel,
            "ts_h5": ts_h5, "ts": ts_h5["timeseries"], "ts_dates": ts_dates,
            "x_first": float(attrs["X_FIRST"]), "y_first": float(attrs["Y_FIRST"]),
            "x_step": x_step, "y_step": y_step, "length": inc.shape[0], "width": inc.shape[1],
            "win": win}


def get_window_points(insar, easting_0, northing_0, az, x_width):
    """Coherent pixels in a square window around one crossing.

    Parameters
    ----------
    insar : dict
        Track products from `read_insar_data`.
    easting_0, northing_0 : float
        UTM position of the crossing (m).
    az : float
        Cross-fault azimuth of the crossing (deg, compass bearing toward the
        east side).
    x_width : float
        Half width along strike of the crossing window (km).

    Returns
    -------
    pandas.DataFrame or None
        One row per pixel: `row`, `col` (pixel), `p` (distance across the
        fault, km, positive toward `az`), `q` (distance along the fault, km),
        `vel` (mm/yr), `inc`, `azi` (deg) and `coherent`. Only coherent pixels
        with a finite velocity and |q| <= x_width are returned. None if the
        crossing is off the frame.
    """
    # The frame is a regular UTM grid: pixel (row, col) sits at
    #     easting  = x_first + col * x_step        (x_step = +30 m, col 0 is the west edge)
    #     northing = y_first + row * y_step        (y_step = -30 m, row 0 is the north edge)
    # so the pixel under the crossing is found by inverting that.
    col_centre = int(round((easting_0 - insar["x_first"]) / insar["x_step"]))
    row_centre = int(round((northing_0 - insar["y_first"]) / insar["y_step"]))
    # The square of +/- win pixels around it (large enough to contain the whole dist_max x x_width
    # box at any orientation), clipped at the frame edge. No pixels are selected here; the
    # selection is the p, q filter below and get_fault_pairs. Fewer than 3 pixels on a side means
    # the crossing is off the frame.
    row_min = max(0, row_centre - insar["win"])
    row_max = min(insar["length"], row_centre + insar["win"] + 1)
    col_min = max(0, col_centre - insar["win"])
    col_max = min(insar["width"], col_centre + insar["win"] + 1)
    if row_max - row_min < 3 or col_max - col_min < 3:
        return None
    # easting and northing of every pixel in the window, then the distance of each pixel from the
    # crossing along the cross-fault azimuth (p) and along the fault (q)
    cols, rows = np.meshgrid(np.arange(col_min, col_max), np.arange(row_min, row_max))
    easting = insar["x_first"] + cols * insar["x_step"]
    northing = insar["y_first"] + rows * insar["y_step"]
    az = np.deg2rad(az)
    p = ((easting - easting_0) * np.sin(az) + (northing - northing_0) * np.cos(az)) / 1000.0
    q = ((easting - easting_0) * np.sin(az + np.pi / 2) + (northing - northing_0) * np.cos(az + np.pi / 2)) / 1000.0
    points = pd.DataFrame({"row": rows.ravel(), "col": cols.ravel(), "p": p.ravel(), "q": q.ravel(),
                           "vel": insar["vel"][row_min:row_max, col_min:col_max].ravel(),
                           "inc": insar["inc"][row_min:row_max, col_min:col_max].ravel(),
                           "azi": insar["azi"][row_min:row_max, col_min:col_max].ravel(),
                           "coherent": insar["mask"][row_min:row_max, col_min:col_max].ravel()})
    points = points[points.coherent & np.isfinite(points.vel) & (np.abs(points.q) <= x_width)]
    return points


def get_fault_pairs(points, dist_min, dist_max):
    """Split the pixels of one crossing into the west and east sides.

    Parameters
    ----------
    points : pandas.DataFrame
        Pixels from `get_window_points`.
    dist_min, dist_max : float
        Inner and outer distance from the trace (km).

    Returns
    -------
    w_subset, e_subset : pandas.DataFrame
        Pixels with -dist_max < p < -dist_min (west) and
        dist_min < p < dist_max (east).
    """
    w_subset = points[(points.p > -dist_max) & (points.p < -dist_min)]
    e_subset = points[(points.p > dist_min) & (points.p < dist_max)]
    return w_subset, e_subset


def read_creep_ts(path):
    """Read a latitude x decimal-year table written by `write_track` (or notebook 03).

    Parameters
    ----------
    path : str
        csv file with latitudes as the index and decimal years as columns.

    Returns
    -------
    pandas.DataFrame
        The table, with float index and float columns.
    """
    df = pd.read_csv(path, index_col=0)
    df.columns = df.columns.astype(float)
    df.index = df.index.astype(float)
    return df


def ts_data_key(dic, coh, fault_profiles):
    """Identity of the inputs of one track, for the reuse test of notebook 03.

    Parameters
    ----------
    dic : dict
        Track entry of `insar_dic_list`.
    coh : float
        Temporal-coherence threshold (the mask is `dic["mask_file"][coh]`).
    fault_profiles : str
        The fault crossings file.

    Returns
    -------
    dict
        Number, first and last of the time-series dates, the decimal year of the first (so that a change of the time
        convention is seen), and the
        fingerprints of every file the extraction reads (time series,
        velocity, geometry, coherence mask, fault crossings).
    """
    with h5py.File(dic["ts_file"], "r") as h:
        dates = h["date"][:]
    key = dict(n_dates=len(dates), first_date=dates[0].decode(), last_date=dates[-1].decode())
    key["first_decimal_year"] = round(utils.date_to_decimal_year(dates[0].decode()), 8)     # the time convention of the columns
    key["sha_ts_file"] = utils.file_fingerprint(dic["ts_file"])
    key["sha_vel_file"] = utils.file_fingerprint(dic["vel_file"])
    key["sha_geo_file"] = utils.file_fingerprint(dic["geo_file"])
    key["sha_mask_file"] = utils.file_fingerprint(dic["mask_file"][coh])
    key["sha_fault_profiles"] = utils.file_fingerprint(fault_profiles)
    return key


def extract_track(insar, x_fault, dist_min, dist_max, x_width, min_points, rate_nsigma, unit, track):
    """Cross-fault LOS series of one track at every crossing, with the rate mask (the two loops of notebook 03).

    Loop 1 splits each crossing into its west and east pixels and pools the deviation of every pixel's LOS rate from
    the median rate of its own side; the robust sigma of that pool sets the cut. Loop 2 removes the pixels beyond the
    cut, reads the displacement cube in the window of the crossing and forms the east median minus the west median
    at every date. A crossing with fewer than `min_points` pixels on either side, before or after the cut, is dropped.

    Parameters
    ----------
    insar : dict
        Track products from `read_insar_data`; `insar["ts"]` is read as `ts[:, rows, cols]`.
    x_fault : pandas.DataFrame
        The crossings from `load_crossings`.
    dist_min, dist_max : float
        Inner and outer distance from the trace (km).
    x_width : float
        Half width along strike of each crossing window (km).
    min_points : int
        Minimum number of pixels on each side.
    rate_nsigma : float
        Rate-mask cut in robust standard deviations.
    unit : float
        Factor from m (products) to mm.
    track : str
        Track name, for the printed progress.

    Returns
    -------
    dict
        `creep_ts` (DataFrame: latitude x decimal year rounded to 4 places, mm LOS, east - west, in crossing order),
        `geometry` (DataFrame: latitude x [inc_deg, azi_deg], the mean over the pixels used), `rate_dev` (the pooled
        rate deviations, mm/yr), `sigma` and `rate_cut` (mm/yr), `n_rejected` and `n_total` (pixels) and `n_dropped`
        (crossings emptied by the cut).
    """
    # ---------- Loop 1: pixel pairs at every crossing, rate deviation of every pixel from its own side
    print(f"[{track}] loop 1: pixel pairs and rate deviations ...")
    pairs = {}                  # index in x_fault -> (w_subset, e_subset)
    rate_dev = []
    for index, row in x_fault.iterrows():
        points = get_window_points(insar, row["easting"], row["northing"], row["az_smooth"], x_width)
        if points is None:
            continue
        w_subset, e_subset = get_fault_pairs(points, dist_min, dist_max)
        if len(w_subset) < min_points or len(e_subset) < min_points:
            continue
        pairs[index] = (w_subset, e_subset)
        rate_dev.append(w_subset["vel"] - w_subset["vel"].median())
        rate_dev.append(e_subset["vel"] - e_subset["vel"].median())
    rate_dev = pd.concat(rate_dev).to_numpy()
    sigma = utils.robust_sigma(rate_dev)
    rate_cut = rate_nsigma * sigma

    # ---------- Loop 2: drop the pixels beyond the cut, read the cube, median of each side, east - west
    print(f"[{track}] loop 2: reading the cube at {len(pairs)} crossings ...")
    creep_ts = {}               # ft_lat -> east - west LOS (mm) at every date
    geometry = {}               # ft_lat -> (mean inc, mean azi) of the pixels used
    n_rejected = 0
    n_total = 0
    n_dropped = 0
    for index, (w_subset, e_subset) in pairs.items():
        w_keep = w_subset[np.abs(w_subset["vel"] - w_subset["vel"].median()) <= rate_cut]
        e_keep = e_subset[np.abs(e_subset["vel"] - e_subset["vel"].median()) <= rate_cut]
        n_total += len(w_subset) + len(e_subset)
        n_rejected += (len(w_subset) - len(w_keep)) + (len(e_subset) - len(e_keep))
        if len(w_keep) < min_points or len(e_keep) < min_points:
            n_dropped += 1
            continue
        # the smallest box holding both sides, read from the cube in one go (m -> mm)
        both = pd.concat([w_keep, e_keep]).sort_values(["row", "col"])
        box_row_min = both["row"].min()
        box_row_max = both["row"].max() + 1
        box_col_min = both["col"].min()
        box_col_max = both["col"].max() + 1
        ts_win = insar["ts"][:, box_row_min:box_row_max, box_col_min:box_col_max] * unit
        w_rows = (w_keep["row"] - box_row_min).to_numpy()
        w_cols = (w_keep["col"] - box_col_min).to_numpy()
        e_rows = (e_keep["row"] - box_row_min).to_numpy()
        e_cols = (e_keep["col"] - box_col_min).to_numpy()
        w_avg = np.nanmedian(ts_win[:, w_rows, w_cols], axis=1)
        e_avg = np.nanmedian(ts_win[:, e_rows, e_cols], axis=1)
        west_east_ts = e_avg - w_avg
        ft_lat = float(x_fault.loc[index, "ft_lat"])
        creep_ts[ft_lat] = west_east_ts
        geometry[ft_lat] = (float(np.nanmean(both["inc"])), float(np.nanmean(both["azi"])))

    creep_ts = pd.DataFrame.from_dict(creep_ts, orient="index", columns=np.round(insar["ts_dates"], 4))
    creep_ts.index.name = "lat"
    geometry = pd.DataFrame.from_dict(geometry, orient="index", columns=["inc_deg", "azi_deg"])
    geometry.index.name = "lat"
    return dict(creep_ts=creep_ts, geometry=geometry, rate_dev=rate_dev, sigma=sigma, rate_cut=rate_cut,
                n_rejected=n_rejected, n_total=n_total, n_dropped=n_dropped)


def write_track(extracted, ts_file, geom_file, params_file, dev_file, params_key):
    """Save one track's extraction: the series and the geometry (rows south -> north), the parameters, the deviations.

    Parameters
    ----------
    extracted : dict
        Output of `extract_track`.
    ts_file, geom_file, params_file : str
        Paths of `creep_ts_LOS_<track>.csv`, `creep_geom_<track>.csv` and `creep_params_<track>.json`.
    dev_file : str or None
        Path of `rate_deviation_<track>.csv`; not written if None.
    params_key : dict
        Written to `params_file` (in notebook 03, the reuse key).
    """
    extracted["creep_ts"].sort_index().to_csv(ts_file)
    extracted["geometry"].sort_index().to_csv(geom_file)
    with open(params_file, "w") as fh:
        json.dump(params_key, fh, indent=1)
    if dev_file is not None:
        pd.DataFrame({"rate_deviation_mm_yr": extracted["rate_dev"]}).to_csv(dev_file, index=False)


# ------------------------------
# Common date grid
# ------------------------------

def interp_to_dates(target_dates, dates, values, max_interp_gap_days):
    """Linearly interpolate one crossing's series onto new dates.

    Parameters
    ----------
    target_dates : numpy.ndarray
        Dates of the common grid (decimal years).
    dates : numpy.ndarray
        Acquisition dates of the track (decimal years).
    values : numpy.ndarray
        Series of the crossing at `dates` (mm).
    max_interp_gap_days : float
        Longest interval between acquisitions (days) across which a value is interpolated.

    Returns
    -------
    numpy.ndarray
        Series at `target_dates`; NaN inside gaps longer than
        max_interp_gap_days, outside the observed range, and everywhere if
        fewer than two values are finite.
    """
    out = np.full(len(target_dates), np.nan)
    ok = np.isfinite(values)
    if ok.sum() < 2:
        return out
    dates, values = dates[ok], values[ok]
    interp = np.interp(target_dates, dates, values)
    left = np.clip(np.searchsorted(dates, target_dates, side="right") - 1, 0, len(dates) - 1)   # acquisition date at or before
    right = np.clip(left + 1, 0, len(dates) - 1)                                                # acquisition date after
    is_real = np.isclose(target_dates, dates[left], rtol=0, atol=0.5 / 365.25)                  # an acquisition date
    #                                       (rtol=0: on decimal years the default rtol=1e-5 alone is ~7 days)
    gap_days = (dates[right] - dates[left]) * 365.25
    inside = (target_dates >= dates[0] - 0.5 / 365.25) & (target_dates <= dates[-1] + 0.5 / 365.25)
    keep = inside & (is_real | (gap_days <= max_interp_gap_days))
    out[keep] = interp[keep]
    return out


def common_grid(asc_los, desc_los, max_interp_gap_days):
    """Both tracks on the union of their dates over their overlap, at the crossings present on both.

    Parameters
    ----------
    asc_los, desc_los : pandas.DataFrame
        Per-track series from `read_creep_ts` (latitude x decimal year, mm LOS).
    max_interp_gap_days : float
        Longest interval (days) bridged by `interp_to_dates`.

    Returns
    -------
    dict
        `los_interp` ({"asc", "desc"} -> DataFrame, latitude x common date, mm), `lats` (the crossings on both
        tracks, sorted), `common_dates` (decimal years), `grid_start` and `grid_end` (the overlap of the two tracks).
    """
    # the common grid runs over the overlap of the two tracks
    grid_start = max(asc_los.columns.min(), desc_los.columns.min())
    grid_end = min(asc_los.columns.max(), desc_los.columns.max())
    common_dates = np.array(sorted(set(asc_los.columns) | set(desc_los.columns)))
    common_dates = common_dates[(common_dates >= grid_start) & (common_dates <= grid_end)]
    lats = np.array(sorted(set(asc_los.index) & set(desc_los.index)))

    los_interp = {}
    for name, los in (("asc", asc_los), ("desc", desc_los)):
        dates = los.columns.values.astype(float)
        matrix = np.full((len(lats), len(common_dates)), np.nan)
        for i, lat in enumerate(lats):
            matrix[i] = interp_to_dates(common_dates, dates, los.loc[lat].values, max_interp_gap_days)
        los_interp[name] = pd.DataFrame(matrix, index=lats, columns=common_dates)
        los_interp[name].index.name = "lat"
    return dict(los_interp=los_interp, lats=lats, common_dates=common_dates, grid_start=grid_start, grid_end=grid_end)


# ------------------------------
# Decomposition
# ------------------------------

def crossing_geometry(geom_file, lats, key):
    """Incidence and azimuth of every kept crossing on one track.

    Parameters
    ----------
    geom_file : str
        `creep_geom_<track>.csv` (latitude x [inc_deg, azi_deg]).
    lats : numpy.ndarray
        Latitudes of the crossings on the common grid; each must have a row.
    key : str
        Track label for the assertion messages ("asc" or "desc").

    Returns
    -------
    inc, azi : numpy.ndarray
        Mean incidence and azimuth (deg) of the pixels used at each crossing.
    """
    geometry = pd.read_csv(geom_file, index_col=0)
    inc = np.full(len(lats), np.nan)
    azi = np.full(len(lats), np.nan)
    for i, lat in enumerate(lats):
        nearest = int(np.argmin(np.abs(geometry.index.values - lat)))
        assert np.isclose(geometry.index.values[nearest], lat, rtol=0, atol=1e-9), \
            f"{key}: crossing {lat} has no geometry row"
        inc[i] = geometry["inc_deg"].values[nearest]
        azi[i] = geometry["azi_deg"].values[nearest]
    assert np.isfinite(inc).all() and np.isfinite(azi).all(), f"{key}: missing geometry"
    return inc, azi


def crossing_strike(x_fault, lats):
    """Local strike of every kept crossing: the smoothed cross-fault azimuth of its trace point + 90.

    Parameters
    ----------
    x_fault : pandas.DataFrame
        The crossings from `load_crossings`.
    lats : numpy.ndarray
        Latitudes of the crossings; each must be a trace point.

    Returns
    -------
    numpy.ndarray
        Strike (deg, compass bearing, pointing south-east along the fault).
    """
    trace_lats = x_fault["ft_lat"].values
    strike = np.full(len(lats), np.nan)
    for i, lat in enumerate(lats):
        nearest = int(np.argmin(np.abs(trace_lats - lat)))
        assert np.isclose(trace_lats[nearest], lat, rtol=0, atol=1e-9), f"crossing {lat} is not a trace point"
        strike[i] = x_fault["az_smooth"].values[nearest] + 90.0
    return strike


def decompose_fp(los_interp, crossing_inc, crossing_azi, strike, lats, common_dates):
    """Fault-parallel series of every crossing from the two tracks, with its own geometry and strike.

    Parameters
    ----------
    los_interp : dict
        {"asc", "desc"} -> DataFrame on the common grid (from `common_grid`).
    crossing_inc, crossing_azi : dict
        {"asc", "desc"} -> incidence and azimuth per crossing (deg; from `crossing_geometry`).
    strike : numpy.ndarray
        Local strike per crossing (deg; from `crossing_strike`).
    lats, common_dates : numpy.ndarray
        Crossings and dates of the common grid.

    Returns
    -------
    pandas.DataFrame
        FP (mm, positive right-lateral), latitude x decimal year; NaN where either track is NaN.
    """
    fp_matrix = np.full((len(lats), len(common_dates)), np.nan)
    for i, lat in enumerate(lats):
        d_asc = los_interp["asc"].values[i]
        d_desc = los_interp["desc"].values[i]
        ok = np.isfinite(d_asc) & np.isfinite(d_desc)
        if ok.sum() == 0:
            continue
        _, fp_matrix[i, ok] = utils.decompose_asc_desc(
            d_asc[ok], d_desc[ok], crossing_inc["asc"][i], crossing_azi["asc"][i],
            crossing_inc["desc"][i], crossing_azi["desc"][i], strike[i])
    fp_ts = pd.DataFrame(fp_matrix, index=lats, columns=common_dates)
    fp_ts.index.name = "lat"
    return fp_ts


# ------------------------------
# Rates and the landslide flag
# ------------------------------

def rate_per_crossing(ts):
    """Least-squares linear rate of every crossing.

    Parameters
    ----------
    ts : pandas.DataFrame
        Latitude x decimal-year table (mm).

    Returns
    -------
    numpy.ndarray
        Rate of each row (mm/yr); NaN if fewer than 6 dates are finite.
    """
    dates = ts.columns.values.astype(float)
    rate = np.full(len(ts), np.nan)
    for i, values in enumerate(ts.values):
        ok = np.isfinite(values)
        if ok.sum() > 5:
            rate[i] = np.polyfit(dates[ok], values[ok], 1)[0]
    return rate


def flag_landslide_crossings(fp_rate_raw, trend_window=TREND_WINDOW, trend_nsigma=TREND_NSIGMA):
    """Flag the crossings whose FP rate departs from the running median of the rate along the fault.

    Parameters
    ----------
    fp_rate_raw : numpy.ndarray
        FP rate per crossing (mm/yr), in fault order.
    trend_window : int, optional
        Crossings in the centred running median (at least 5 finite values).
    trend_nsigma : float, optional
        Cut in robust standard deviations of the residuals from the running median.

    Returns
    -------
    landslide_flag : numpy.ndarray of bool
        True for a flagged crossing (finite rate beyond the cut).
    trend_sigma : float
        Robust sigma of the residuals (mm/yr).
    """
    trend = pd.Series(fp_rate_raw).rolling(trend_window, center=True, min_periods=5).median().to_numpy()
    resid = fp_rate_raw - trend
    trend_sigma = utils.robust_sigma(resid)
    landslide_flag = (np.abs(resid) > trend_nsigma * trend_sigma) & np.isfinite(fp_rate_raw)
    return landslide_flag, trend_sigma
