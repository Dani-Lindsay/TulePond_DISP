# %% [markdown]
# # 03 Cross-fault creep time series
#
# ```
# Author: Danielle Lindsay
# Environment: opera_disp_feb13. Run time: about 15 min per track for the time-series read; the other sections take seconds.
# ```
#
# **What this notebook does.** Following Li et al. (2023), it measures creep as
# the difference between the InSAR displacement time series on the two sides of
# the fault, at every 100 m crossing of the hand-mapped Hayward Fault trace
# (Methods M3). The difference removes the reference datum and the per-date
# noise common to both sides.
#
# 1. **Fault trace**: the cross-fault azimuth of every crossing is smoothed
#    along the fault (section 3).
# 2. **Cross-fault series, per track**: at each crossing the pixels in a box
#    oriented by the local strike are split into a west and an east side; pixels
#    whose LOS rate departs from the median rate of their side by more than 3
#    robust standard deviations are rejected (landslides, quarries, edges); and
#    the median of the east side minus the median of the west side is formed at
#    every date (sections 4 and 5). This is done separately for the ascending
#    (09157) and descending (11115) frames. Section 6 checks one crossing step
#    by step, and Fig. S3 shows the rate mask (section 7).
# 3. **Common date grid**: each track is interpolated onto the union of the
#    dates of both tracks, only across gaps shorter than 45 days (section 8).
# 4. **Decomposition**: the ascending and descending series of each crossing are
#    decomposed into fault-parallel (FP) and vertical (Up) components with the
#    local strike and the local incidence and azimuth (section 9).
# 5. **Creep rate and landslide flag**: a linear FP rate is fitted at every
#    crossing; a crossing whose rate departs from a 21-crossing running median by
#    more than 3 robust standard deviations is removed from every product; and
#    the rate profile is smoothed along the fault (sections 10 to 12).
# 6. **Sampling**: Fig. S4 shows which epochs of each track are acquired and
#    which are interpolated (section 13). The numbers are registered in
#    section 14.
#
# Only crossings present on both tracks are kept on the common grid. The
# northern limit is Point Pinole; in the south the common crossings end where
# either track has too few valid pixels (the latitude range is printed in
# section 8).
#
# **Inputs**: per track, the MintPy products of `data_prep/run04` (Yunjun et al.
# 2019): `timeseries_tropo.h5`, `velocity_masked.h5`, `geometryGeo.h5` and
# `maskTempCoh_07.h5`; the fault crossings `hayward_fault_profiles.csv`
# (notebook 01); the creepmeter list in `TulePond_filepaths.py`.
#
# **Outputs** (`outputs/03_creep_timeseries/`):
# - per track: `creep_ts_LOS_<track>.csv` (latitude x decimal-year matrix, mm
#   LOS; read by notebooks 04, 05 and 12), `creep_geom_<track>.csv` (mean
#   incidence and azimuth of the pixels used at each crossing),
#   `creep_params_<track>.json` (the reuse key of section 5),
#   `rate_deviation_<track>.csv` (the rate deviations of section 5)
# - common grid: `fp_ts.csv` (read by notebooks 04, 05, 11 and 13)
# - along-fault rates: `creep_rate_fp.csv` (read by notebook 04)
# - `03_creep_stats.txt` and the registry rows `03_*`
#
# **Figures**: Fig. S3 `figS_ratemask_<params>_nsigma3` (the rate mask) and
# Fig. S4 `figS_acquisitions_<params>_gap45d` (acquired and interpolated
# epochs).
#
# **What to check.** Section 6 asserts that the crossing at CFW1, recomputed
# step by step, equals the extracted series, and section 12 asserts that its FP
# rate is finite. The printed output and the sidecar give the rate-mask
# standard deviation and the rejected percentage per track (section 5), the
# number of common dates and of dates acquired on both tracks (section 8), the
# number of landslide-flagged crossings and the median FP creep rate
# (section 10), and the sampling statistics (section 13).

# %% [markdown]
# ## 1. Imports

# %%
import os
import json

import insar_utils as utils
from TulePond_filepaths import (common_paths, insar_dic_list, creepmeters, unit, params,
                                dist_min, dist_max, x_width, min_points, coh_timeseries,
                                rate_nsigma, az_smooth_n, max_interp_gap_days, rate_smooth_km,
                                repo_dir, frames)

import numpy as np                               # Matrix calculations
import pandas as pd                              # Tables
import h5py                                      # MintPy HDF5 products
from pyproj import Transformer                   # lon/lat -> UTM
import matplotlib.pyplot as plt                  # Figures

# %% [markdown]
# ## 2. Paths and parameters
#
# The aperture, mask and smoothing settings are read from
# `TulePond_filepaths.py`, so that every notebook uses the same values. The
# parameters set below are used only in this notebook.

# %%
notebook = "03_creep_timeseries"
out_dir = os.path.join(common_paths["out_dir"], notebook)
fig_dir = common_paths["fig_dir"]
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

utm_epsg = 32610         # UTM zone 10N, the grid of the frames
trend_window = 21        # crossings (~2 km) in the running median used to flag landslide crossings
trend_nsigma = 3.0       # a crossing whose FP rate departs from that trend by more than this (robust sigma) is dropped
s1b_death = utils.date_to_decimal_year("2021-12-23")     # Sentinel-1B failure: the end of 6-day sampling (Fig S4)

print(f"aperture {dist_min * 1000:.0f}-{dist_max * 1000:.0f} m, along-strike +/-{x_width * 1000:.0f} m, "
      f"min {min_points} px per side, coherence {coh_timeseries}, rate mask {rate_nsigma:g} sigma")
print("tracks: " + ", ".join(f"{d['ORBIT_DIRECTION']} {d['Frame']}" for d in insar_dic_list))

REG = utils.Registry(notebook, common_paths, repo_dir)
input_files = []
for dic in insar_dic_list:
    input_files.append(dic["ts_file"])
    input_files.append(dic["vel_file"])
    input_files.append(dic["geo_file"])
    input_files.append(dic["mask_file"][coh_timeseries])
input_files.append(common_paths["fault_profiles"])
REG.inputs(*input_files)

# %% [markdown]
# ## 3. Fault trace
#
# `hayward_fault_profiles.csv` is the hand-mapped trace resampled every 100 m
# (notebook 01): `ft_lon, ft_lat, profile_az`, south to north, where
# `profile_az` is the cross-fault azimuth $\phi_k$ of crossing $k$ (compass
# bearing, toward the east side).
#
# The raw azimuth includes the jitter of hand digitising at 100 m spacing. A
# centred rolling median over 21 crossings (2.1 km; `az_smooth_n`, at least 3
# values),
#
# $$ \tilde\phi_k = \operatorname{median}\left(\phi_{k-10}, \dots, \phi_{k+10}\right), $$
#
# removes the digitising kinks and keeps the bends of the trace; notebook 02
# uses the same smoothing. The local strike is $\Psi_k = \tilde\phi_k + 90^\circ$
# (compass bearing, pointing south-east along the fault). The along-trace
# distance of each crossing, in km from Point Pinole, is taken from
# `utils.lat_to_km_map` (nearest crossing by latitude, 0.1 km per crossing).

# %%
x_fault = pd.read_csv(common_paths["fault_profiles"])
x_fault["az_smooth"] = x_fault["profile_az"].rolling(az_smooth_n, center=True, min_periods=3).median()
km_of = utils.lat_to_km_map(common_paths["fault_profiles"])      # lat -> km from Point Pinole
x_fault["km"] = km_of(x_fault["ft_lat"].values)

# UTM position of every crossing, for the pixel windows
ll2utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
x_fault["easting"], x_fault["northing"] = ll2utm.transform(x_fault["ft_lon"].values, x_fault["ft_lat"].values)

print(f"{len(x_fault)} crossings from {x_fault.ft_lat.min():.4f} N (south, {x_fault.km.max():.1f} km) "
      f"to {x_fault.ft_lat.max():.4f} N (Point Pinole, 0 km)")
print(f"cross-fault azimuth: raw {x_fault.profile_az.min():.1f} to {x_fault.profile_az.max():.1f}, "
      f"smoothed {x_fault.az_smooth.min():.1f} to {x_fault.az_smooth.max():.1f} deg; "
      f"|raw - smoothed| > 10 deg at {(np.abs(x_fault.profile_az - x_fault.az_smooth) > 10).sum()} crossings")

# %% [markdown]
# ## 4. Functions
#
# `get_window_points` returns the coherent pixels around one crossing with
# their position relative to the crossing, computed in UTM:
#
# $$ p = \frac{(E - E_0)\sin\tilde\phi + (N - N_0)\cos\tilde\phi}{1000}, \qquad
#    q = \frac{(E - E_0)\sin(\tilde\phi + 90^\circ) + (N - N_0)\cos(\tilde\phi + 90^\circ)}{1000}, $$
#
# where $(E, N)$ are the easting and northing of the pixel (m), $(E_0, N_0)$
# those of the crossing, $\tilde\phi$ the smoothed cross-fault azimuth of the
# crossing, $p$ the distance across the fault (km, positive toward the east
# side) and $q$ the distance along the fault (km). A pixel is kept if it is in
# the coherence mask, has a finite LOS velocity and has $|q| \le x_w$
# (`x_width`). `get_fault_pairs` then splits the pixels into the west side,
# $-d_{max} < p < -d_{min}$, and the east side, $d_{min} < p < d_{max}$
# (`dist_min`, `dist_max`).
#
# The coherence mask `maskTempCoh_07.h5` keeps pixels with temporal coherence of
# at least 0.7 and a valid value at every epoch (set in `data_prep/run04`).
#
# The displacement cube (13 GB per track) is read one window at a time. The
# window is a square of $\pm w$ pixels around the crossing, with
# $w = \lceil \sqrt{d_{max}^2 + x_w^2} / \Delta x \rceil + 1$ and $\Delta x$ the
# pixel size (30 m): the corners of the $d_{max} \times x_w$ box are
# $\sqrt{d_{max}^2 + x_w^2}$ from the crossing, so the window contains the whole
# box at any orientation. The window does not select pixels; any window that
# contains the box gives the same result.

# %%
def read_insar_data(dic):
    """Open the MintPy products of one track.

    The displacement cube is not loaded: `ts` is the open h5py dataset and is
    read per crossing window. Close `ts_h5` when done.

    Parameters
    ----------
    dic : dict
        Track entry of `insar_dic_list` (file paths, orbit direction, frame).

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
    with h5py.File(dic["mask_file"][coh_timeseries], "r") as hf:
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
          f"coherence-{coh_timeseries} mask keeps {100 * mask.mean():.1f}% of pixels")
    return {"inc": inc, "azi": azi, "mask": mask, "vel": vel,
            "ts_h5": ts_h5, "ts": ts_h5["timeseries"], "ts_dates": ts_dates,
            "x_first": float(attrs["X_FIRST"]), "y_first": float(attrs["Y_FIRST"]),
            "x_step": x_step, "y_step": y_step, "length": inc.shape[0], "width": inc.shape[1],
            "win": win}


def get_window_points(insar, easting_0, northing_0, az):
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
    """Read a latitude x decimal-year table written by this notebook.

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

# %% [markdown]
# ## 5. Extract the cross-fault time series, one track at a time
#
# The method follows Li et al. (2023). There are two loops over the crossings
# of each track.
#
# **Loop 1: pixel pairs and the rate mask.** At every crossing $k$ the window is
# read and split into the west side $W_k$ and the east side $E_k$ (at least
# `min_points` pixels each). The LOS rate $v_i$ of every pixel (mm/yr, from
# `velocity_masked.h5`) is expressed as its deviation from the median rate of
# its own side,
#
# $$ \delta_i = v_i - \operatorname{median}_{j \in S_i} v_j , $$
#
# where $S_i$ is the side ($W_k$ or $E_k$) of pixel $i$. Fault creep is a
# difference between the two sides and therefore does not enter these
# deviations, and the deviations are centred on zero along the whole fault. They
# are pooled over all crossings of the track (a pixel enters once for each
# crossing whose box contains it), and one robust standard deviation per track
# [citation needed] sets the cut:
#
# $$ \sigma_r = 1.4826\,\operatorname{median}_i \left| \delta_i - \operatorname{median}_j \delta_j \right| , $$
#
# where 1.4826 scales the median absolute deviation to the standard deviation of
# a normal distribution (`utils.robust_sigma`).
#
# **Loop 2: the difference.** Pixels with $|\delta_i| > n_\sigma\,\sigma_r$
# ($n_\sigma$ = `rate_nsigma` = 3) are removed, and a crossing with fewer than
# `min_points` pixels left on either side is dropped. The displacement cube is
# read in the window of the crossing, and the cross-fault series is
#
# $$ \Delta d_k(t) = \operatorname{median}_{i \in E_k} d_i(t) - \operatorname{median}_{i \in W_k} d_i(t), $$
#
# where $d_i(t)$ is the LOS displacement of pixel $i$ at date $t$ (mm, positive
# toward the satellite) and the medians ignore NaN. The mean incidence and
# azimuth of the pixels used are saved for the decomposition (section 9).
#
# The cube read takes about 15 min per track. If the per-track files are
# already on disk with the same reuse key, the extraction of that track is
# skipped and the mask statistics are recomputed from the saved deviations. The
# reuse key consists of the parameters and the identity of the inputs: the dates
# of the time series and the fingerprints of the time series, velocity,
# geometry, coherence mask and fault crossings. A change in the data, the mask
# or the trace therefore forces a re-run; deleting the files also forces a
# re-run.

# %%
params_now = dict(coh=coh_timeseries, dist_min=dist_min, dist_max=dist_max, x_width=x_width,
                  min_points=min_points, rate_nsigma=rate_nsigma)


def ts_data_key(dic):
    """Identity of the inputs of one track, for the reuse test.

    Parameters
    ----------
    dic : dict
        Track entry of `insar_dic_list`.

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
    key["sha_mask_file"] = utils.file_fingerprint(dic["mask_file"][coh_timeseries])
    key["sha_fault_profiles"] = utils.file_fingerprint(common_paths["fault_profiles"])
    return key


mask_stats = {}

for dic in insar_dic_list:
    track = dic["ORBIT_DIRECTION"]
    ts_file = os.path.join(out_dir, f"creep_ts_LOS_{track}.csv")
    geom_file = os.path.join(out_dir, f"creep_geom_{track}.csv")
    params_file = os.path.join(out_dir, f"creep_params_{track}.json")
    dev_file = os.path.join(out_dir, f"rate_deviation_{track}.csv")

    # ---------- already on disk with this reuse key: recompute only the mask statistics
    cache_key = dict(params_now)
    cache_key.update(ts_data_key(dic))
    saved_key = None
    if os.path.exists(params_file) and os.path.exists(ts_file):
        with open(params_file) as fh:
            saved_key = json.load(fh)
    if saved_key == cache_key:
        print(f"[{track}] extraction already on disk with this reuse key; reusing")
        rate_dev = pd.read_csv(dev_file)["rate_deviation_mm_yr"].to_numpy()
        creep_ts = read_creep_ts(ts_file)
        sigma = utils.robust_sigma(rate_dev)
        rate_cut = rate_nsigma * sigma
        mask_stats[track] = dict(track=track, n_crossings=len(creep_ts), n_dates=creep_ts.shape[1],
                                 ts_dates=creep_ts.columns.to_numpy(), sigma=sigma, rate_cut=rate_cut,
                                 rej_pct=100.0 * np.mean(np.abs(rate_dev) > rate_cut), n_px=rate_dev.size,
                                 max_dev=float(np.abs(rate_dev).max()))
        continue

    insar = read_insar_data(dic)

    # ---------- Loop 1: pixel pairs at every crossing, rate deviation of every pixel from its own side
    print(f"[{track}] loop 1: pixel pairs and rate deviations ...")
    pairs = {}                  # index in x_fault -> (w_subset, e_subset)
    rate_dev = []
    for index, row in x_fault.iterrows():
        points = get_window_points(insar, row["easting"], row["northing"], row["az_smooth"])
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
    insar["ts_h5"].close()

    # ---------- save: lat x decimal-year matrix (rows south -> north), geometry, parameters, deviations
    creep_ts = pd.DataFrame.from_dict(creep_ts, orient="index", columns=np.round(insar["ts_dates"], 4))
    creep_ts.index.name = "lat"
    creep_ts.sort_index().to_csv(ts_file)
    geometry = pd.DataFrame.from_dict(geometry, orient="index", columns=["inc_deg", "azi_deg"])
    geometry.index.name = "lat"
    geometry.sort_index().to_csv(geom_file)
    with open(params_file, "w") as fh:
        json.dump(cache_key, fh, indent=1)
    pd.DataFrame({"rate_deviation_mm_yr": rate_dev}).to_csv(dev_file, index=False)

    rej_pct = 100.0 * n_rejected / n_total
    print(f"[{track}] {len(creep_ts)}/{len(x_fault)} crossings, {len(insar['ts_dates'])} dates")
    print(f"          rate mask: sigma {sigma:.3f} mm/yr, cut +/-{rate_cut:.3f} ({rate_nsigma:g} sigma) -> "
          f"{rej_pct:.1f}% of pixels rejected, {n_dropped} crossings dropped, max |dev| {np.abs(rate_dev).max():.1f}")
    print("")
    mask_stats[track] = dict(track=track, n_crossings=len(creep_ts), n_dates=len(insar["ts_dates"]),
                             ts_dates=insar["ts_dates"], sigma=sigma, rate_cut=rate_cut, rej_pct=rej_pct,
                             n_px=rate_dev.size, max_dev=float(np.abs(rate_dev).max()))

# %% [markdown]
# ## 6. Check: one crossing recomputed step by step
#
# The crossing at the latitude of CFW1 is recomputed on the ascending track at
# the last epoch, step by step outside the loop: the window is read and split
# into west and east, the pixels beyond the rate cut of the track are removed,
# and the side medians are differenced. Check: the result equals the value in
# `creep_ts_LOS_ascending.csv` to 1e-3 mm.

# %%
for dic in insar_dic_list:
    if dic["ORBIT_DIRECTION"] == "ascending":
        asc_dic = dic
cfw1_lat = None
for code, cm_lat, cm_lon, cm_name in creepmeters:
    if code == "CFW1":
        cfw1_lat = cm_lat
assert cfw1_lat is not None, "CFW1 is not in the creepmeter list"
cfw1_crossing = x_fault.loc[(x_fault.ft_lat - cfw1_lat).abs().idxmin()]     # the crossing at CFW1's latitude
insar = read_insar_data(asc_dic)
points = get_window_points(insar, cfw1_crossing["easting"], cfw1_crossing["northing"], cfw1_crossing["az_smooth"])
w_subset, e_subset = get_fault_pairs(points, dist_min, dist_max)
last = insar["ts"][-1] * unit                                              # the last epoch, whole frame, mm
insar["ts_h5"].close()
# the rate mask, as loop 2 applies it
rate_cut_asc = mask_stats["ascending"]["rate_cut"]
w_keep = w_subset[np.abs(w_subset["vel"] - w_subset["vel"].median()) <= rate_cut_asc]
e_keep = e_subset[np.abs(e_subset["vel"] - e_subset["vel"].median()) <= rate_cut_asc]
east_median = np.nanmedian(last[e_keep["row"].to_numpy(), e_keep["col"].to_numpy()])
west_median = np.nanmedian(last[w_keep["row"].to_numpy(), w_keep["col"].to_numpy()])
by_hand = east_median - west_median
from_file = read_creep_ts(os.path.join(out_dir, "creep_ts_LOS_ascending.csv")).loc[cfw1_crossing["ft_lat"]].iloc[-1]
print(f"crossing at CFW1's latitude, {cfw1_crossing.ft_lat:.4f} N ({cfw1_crossing.km:.1f} km), ascending, last epoch:")
print(f"   {len(w_subset)} west px, {len(e_subset)} east px; after the rate mask {len(w_keep)} west, {len(e_keep)} east")
print(f"   east median - west median, recomputed = {by_hand:.3f} mm;  in creep_ts_LOS_ascending.csv = {from_file:.3f} mm")
assert abs(by_hand - from_file) < 1e-3, "the recomputed crossing differs from the extracted series"
print("Check passed: the extraction loop reproduces the step-by-step computation")

# %% [markdown]
# ## 7. Figure S3: the rate mask
#
# The pooled deviations $\delta_i$ of each track on a logarithmic count axis,
# with the cut at $\pm n_\sigma \sigma_r$. Pixels in the tails beyond the cut are
# rejected. Because creep is a difference between the two sides, it does not
# enter a within-side deviation, so the cut does not depend on the local creep
# rate.

# %%
utils.mpl_paper_style(base=8)          # Helvetica, 8 pt, editable PDF text
lim = 6.0              # mm/yr, histogram half-range
nbins = 181
bins = np.linspace(-lim, lim, nbins)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
for ax, dic, letter in zip(axes, insar_dic_list, "ab"):
    track = dic["ORBIT_DIRECTION"]
    track_stats = mask_stats[track]
    rate_dev = pd.read_csv(os.path.join(out_dir, f"rate_deviation_{track}.csv"))["rate_deviation_mm_yr"].to_numpy()
    ax.hist(np.clip(rate_dev, -lim, lim), bins=bins, color="0.4", log=True)     # clipped into the end bins, not dropped
    for sign in (-1, 1):
        ax.axvline(sign * track_stats["rate_cut"], color="#D55E00", lw=1.1, ls="--")
    ax.axvspan(-lim, -track_stats["rate_cut"], color="#D55E00", alpha=0.08)
    ax.axvspan(track_stats["rate_cut"], lim, color="#D55E00", alpha=0.08)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("LOS rate deviation from side median (mm/yr)")
    utils.panel_title(ax, letter.upper(), f"{track.capitalize()} (frame {dic['Frame']})")
    ax.text(0.03, 0.95, f"$\\sigma$ = {track_stats['sigma']:.2f} mm/yr\ncut $\\pm${track_stats['rate_cut']:.2f} ({rate_nsigma:g}$\\sigma$)\n"
            f"rejected {track_stats['rej_pct']:.1f}%\nlargest deviation {track_stats['max_dev']:.1f} mm/yr",
            transform=ax.transAxes, va="top", ha="left", fontsize=7,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))
axes[0].set_ylabel("Near-fault pixels")
fig.tight_layout()
figS3 = os.path.join(fig_dir, f"figS_ratemask_{params}_nsigma{rate_nsigma:g}.png")
fig.savefig(figS3, dpi=300)
fig.savefig(figS3.replace(".png", ".pdf"))
print("Saved figure to:", figS3)
plt.show()

# %% [markdown]
# ## 8. Common date grid
#
# The decomposition requires both tracks at the same epochs. The common grid is
# the union of the acquisition dates of both tracks between the later of their
# first dates and the earlier of their last dates. Each crossing is linearly
# interpolated from its acquisition dates onto this grid. An interpolated value
# at date $\tau$, with $t_l \le \tau < t_r$ the bracketing acquisitions, is kept
# only if $\tau$ is within half a day of an acquisition or
#
# $$ t_r - t_l \le \Delta t_{max} , $$
#
# where $\Delta t_{max}$ = `max_interp_gap_days` = 45 d. This bridges the 6, 12
# and 24 day sampling intervals and isolated missed acquisitions. Longer gaps
# are left as NaN, and no values are extrapolated. Only crossings present on
# both tracks are kept. An epoch counts as acquired on a track if that track
# has an acquisition within half a day of it.

# %%
def interp_to_dates(target_dates, dates, values):
    """Linearly interpolate one crossing's series onto new dates.

    Parameters
    ----------
    target_dates : numpy.ndarray
        Dates of the common grid (decimal years).
    dates : numpy.ndarray
        Acquisition dates of the track (decimal years).
    values : numpy.ndarray
        Series of the crossing at `dates` (mm).

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


asc_los = read_creep_ts(os.path.join(out_dir, "creep_ts_LOS_ascending.csv"))
desc_los = read_creep_ts(os.path.join(out_dir, "creep_ts_LOS_descending.csv"))
# the common grid runs over the overlap of the two tracks (registered as 03_window_start / _end)
grid_start = max(asc_los.columns.min(), desc_los.columns.min())
grid_end = min(asc_los.columns.max(), desc_los.columns.max())
common_dates = np.array(sorted(set(asc_los.columns) | set(desc_los.columns)))
common_dates = common_dates[(common_dates >= grid_start) & (common_dates <= grid_end)]
lats = np.array(sorted(set(asc_los.index) & set(desc_los.index)))
print(f"crossings on both tracks: {len(lats)}, {lats.min():.4f} to {lats.max():.4f} N")

los_interp = {}
for name, los in (("asc", asc_los), ("desc", desc_los)):
    dates = los.columns.values.astype(float)
    matrix = np.full((len(lats), len(common_dates)), np.nan)
    for i, lat in enumerate(lats):
        matrix[i] = interp_to_dates(common_dates, dates, los.loc[lat].values)
    los_interp[name] = pd.DataFrame(matrix, index=lats, columns=common_dates)
    los_interp[name].index.name = "lat"
print(f"common grid: {len(lats)} crossings x {len(common_dates)} dates, {grid_start:.3f} to {grid_end:.3f}")

# which epochs are acquired on each track (an acquisition within half a day) and which are interpolated
real = {}
for name, los in (("asc", asc_los), ("desc", desc_los)):
    dates = los.columns.values.astype(float)
    is_real = np.zeros(len(common_dates), dtype=bool)
    for k, epoch in enumerate(common_dates):
        is_real[k] = np.min(np.abs(dates - epoch)) < 0.5 / 365.25
    real[name] = is_real
both_real = real["asc"] & real["desc"]
print(f"both tracks real at {both_real.sum()} epochs; interpolated: asc {(~real['asc']).sum()} "
      f"({100 * (~real['asc']).mean():.0f}%), desc {(~real['desc']).sum()} ({100 * (~real['desc']).mean():.0f}%)")

# the exact UTC time of every acquisition of each track (the zero-Doppler start time of its OPERA granules), read
# once here; the onset windows (04), the rain days (12) and the forecast (13) take their times from this table
time_rows = []
for dic in insar_dic_list:
    track = dic["ORBIT_DIRECTION"]
    clock = utils.acquisition_times(frames[dic["Frame"]]["data_dir"], dic["Frame"])
    track_days = [utils.decimal_year_to_day(t) for t in read_creep_ts(os.path.join(out_dir, f"creep_ts_LOS_{track}.csv")).columns]
    missing = [day for day in track_days if day not in clock.index]
    assert not missing, f"{track}: {len(missing)} acquisition dates have no frame {dic['Frame']} granule, e.g. {missing[:3]}"
    for day in track_days:
        time_rows.append(dict(track=track, frame=dic["Frame"], day=day.strftime("%Y-%m-%d"),
                              utc_time=clock[day].strftime("%Y-%m-%d %H:%M:%S"),
                              decimal_year=round(utils.date_to_decimal_year(clock[day]), 8)))
acquisition_times = pd.DataFrame(time_rows).sort_values(["track", "day"]).reset_index(drop=True)
acquisition_times.to_csv(os.path.join(out_dir, "acquisition_times.csv"), index=False)
for track, rows in acquisition_times.groupby("track"):
    hours = pd.to_datetime(rows.utc_time).dt.hour + pd.to_datetime(rows.utc_time).dt.minute / 60
    print(f"{track}: {len(rows)} acquisitions at {hours.min():.2f} to {hours.max():.2f} h UTC (acquisition_times.csv)")

# %% [markdown]
# ## 9. Decompose each crossing into fault-parallel and vertical
#
# Each crossing $k$ is decomposed with its own geometry. At every epoch $t$ at
# which both tracks have a value,
#
# $$ \begin{bmatrix} \Delta d_{a}(t) \\ \Delta d_{d}(t) \end{bmatrix} =
#    \begin{bmatrix} \cos\bar\theta_a & \sin\bar\theta_a \cos(\bar\alpha_a + \Psi_k) \\
#                    \cos\bar\theta_d & \sin\bar\theta_d \cos(\bar\alpha_d + \Psi_k) \end{bmatrix}
#    \begin{bmatrix} U(t) \\ FP(t) \end{bmatrix}, $$
#
# where $\Delta d_a$ and $\Delta d_d$ are the ascending and descending
# cross-fault LOS series of section 5 on the common grid (mm),
# $\bar\theta_a, \bar\alpha_a$ and $\bar\theta_d, \bar\alpha_d$ the mean
# incidence and azimuth of the pixels used at that crossing on each track (deg;
# `creep_geom_<track>.csv`; MintPy convention, azimuth of the ground-to-satellite
# vector, anticlockwise from north), $\Psi_k$ the local strike of section 3,
# and $U$ and $FP$ the vertical and fault-parallel cross-fault displacements
# (mm). Each row is
# $[\hat{\mathbf{l}}\cdot\hat{\mathbf{z}},\ \hat{\mathbf{l}}\cdot\hat{\mathbf{s}}_k]$,
# with $\hat{\mathbf{l}} = (-\sin\theta\sin\alpha, \sin\theta\cos\alpha, \cos\theta)$
# the unit vector from the ground to the satellite and
# $\hat{\mathbf{s}}_k = (\sin\Psi_k, \cos\Psi_k, 0)$ the strike direction (east,
# north, up; `utils.design_matrix_up_fp`, covered by the tests). The system is
# solved by least squares (`utils.decompose_asc_desc`), which for two tracks is
# the exact solution. FP is positive along the strike bearing (south-east), so
# right-lateral creep is positive. All horizontal motion is assumed to be
# fault-parallel.
#
# The local strike and geometry are used for all products derived from the time
# series; the mean strike is used only for the decomposition of the velocity
# fields (notebook 02). The printed output lists the range of incidence and
# azimuth along the fault on each track.

# %%
# the incidence and azimuth of every kept crossing, on each track (the crossings must match)
crossing_inc = {}
crossing_azi = {}
for dic, key in zip(insar_dic_list, ("asc", "desc")):
    geometry = pd.read_csv(os.path.join(out_dir, f"creep_geom_{dic['ORBIT_DIRECTION']}.csv"), index_col=0)
    crossing_inc[key] = np.full(len(lats), np.nan)
    crossing_azi[key] = np.full(len(lats), np.nan)
    for i, lat in enumerate(lats):
        nearest = int(np.argmin(np.abs(geometry.index.values - lat)))
        assert np.isclose(geometry.index.values[nearest], lat, rtol=0, atol=1e-9), \
            f"{key}: crossing {lat} has no geometry row"
        crossing_inc[key][i] = geometry["inc_deg"].values[nearest]
        crossing_azi[key][i] = geometry["azi_deg"].values[nearest]
    assert np.isfinite(crossing_inc[key]).all() and np.isfinite(crossing_azi[key]).all(), f"{key}: missing geometry"
    print(f"{dic['ORBIT_DIRECTION']}: incidence {crossing_inc[key].min():.2f} to {crossing_inc[key].max():.2f}, "
          f"azimuth {crossing_azi[key].min():.2f} to {crossing_azi[key].max():.2f} deg")

# local strike of every kept crossing: the smoothed cross-fault azimuth of its trace point + 90
trace_lats = x_fault["ft_lat"].values
strike = np.full(len(lats), np.nan)
for i, lat in enumerate(lats):
    nearest = int(np.argmin(np.abs(trace_lats - lat)))
    assert np.isclose(trace_lats[nearest], lat, rtol=0, atol=1e-9), f"crossing {lat} is not a trace point"
    strike[i] = x_fault["az_smooth"].values[nearest] + 90.0

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
mid = len(lats) // 2
print(f"decomposed {np.isfinite(fp_matrix).any(axis=1).sum()} crossings; G for the mid-fault crossing ({lats[mid]:.3f} N):")
print(utils.design_matrix_up_fp([crossing_inc["asc"][mid], crossing_inc["desc"][mid]],
                                [crossing_azi["asc"][mid], crossing_azi["desc"][mid]], strike[mid]).round(3), "  (columns U, FP)")

# %% [markdown]
# ## 10. Along-fault creep rate and the crossing-level landslide flag
#
# **Rate.** At every crossing a straight line is fitted by least squares
# (`np.polyfit`) to the FP series,
#
# $$ FP_k(t_m) = c_k + r_k\,t_m , $$
#
# over the epochs $t_m$ (decimal years) with a finite value, where $c_k$ is an
# intercept (mm) and $r_k$ the creep rate (mm/yr). At least 6 epochs are
# required.
#
# **Landslide flag.** The pixel rate mask of section 5 removes pixels whose rate
# departs from the median of their side. A landslide that covers most of one
# side defines that side's median and is not removed by it. Whole crossings are
# therefore compared with a running median of the rate,
#
# $$ \tilde r_k = \operatorname{median}\left(r_{k-10}, \dots, r_{k+10}\right), \qquad e_k = r_k - \tilde r_k , $$
#
# taken over 21 consecutive common crossings (`trend_window`; about 2.1 km where
# no crossing is missing, at least 5 finite values). With the robust standard
# deviation of the residuals [citation needed],
#
# $$ \sigma_e = 1.4826\,\operatorname{median}_k \left| e_k - \operatorname{median}_j e_j \right| , $$
#
# a crossing is flagged if $|e_k| > n_e\,\sigma_e$ ($n_e$ = `trend_nsigma` = 3).
# Flagged crossings are set to NaN in every product.
#
# **Smoothing.** The remaining rate profile is smoothed along the fault with a
# Gaussian (`utils.gaussian_smooth_km`),
#
# $$ \bar r_k = \frac{\sum_j G(x_k - x_j)\,r_j}{\sum_j G(x_k - x_j)}, \qquad G(u) = \exp\left(-\frac{u^2}{2\sigma_s^2}\right), $$
#
# where $x_k$ is the along-trace distance of crossing $k$ (km from Point Pinole),
# $\sigma_s$ = `rate_smooth_km` = 0.5 km, and the sums run over the crossings
# with a finite rate. The crossings are placed on their regular 100 m grid, so a
# missing crossing is a gap, not a closer neighbour. The kernel is truncated at
# 4 $\sigma_s$ and the grid is extended at its ends with the nearest value
# (`scipy.ndimage.gaussian_filter1d`). $\bar r_k$ is NaN where less than 15 % of
# the kernel weight falls on crossings with data, and wherever $r_k$ is NaN.

# %%
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


fp_rate_raw = rate_per_crossing(fp_ts)
trend = pd.Series(fp_rate_raw).rolling(trend_window, center=True, min_periods=5).median().to_numpy()
resid = fp_rate_raw - trend
trend_sigma = utils.robust_sigma(resid)
landslide_flag = (np.abs(resid) > trend_nsigma * trend_sigma) & np.isfinite(fp_rate_raw)
print(f"along-fault rejection: {landslide_flag.sum()}/{len(fp_rate_raw)} crossings dropped "
      f"({100 * landslide_flag.mean():.1f}%), residual sigma {trend_sigma:.2f} mm/yr")
fp_ts.iloc[landslide_flag, :] = np.nan
fp_rate = np.where(landslide_flag, np.nan, fp_rate_raw)
crossing_km = km_of(lats)                                                       # along-trace km of each crossing
fp_rate_smooth = utils.gaussian_smooth_km(fp_rate, crossing_km, rate_smooth_km)  # NaN-aware, gap-preserving, along-trace km

valid = fp_rate[np.isfinite(fp_rate)]
print(f"FP creep rate: {valid.size} valid crossings, median {np.median(valid):.2f} mm/yr, "
      f"5-95% {np.percentile(valid, 5):.2f} to {np.percentile(valid, 95):.2f}, min-max {valid.min():.2f} to {valid.max():.2f}")

# %% [markdown]
# ## 11. Save the common-grid products
#
# `fp_ts.csv` (latitude x decimal year, landslide-flagged crossings NaN) and the
# rate profile `creep_rate_fp.csv`.

# %%
fp_ts.to_csv(os.path.join(out_dir, "fp_ts.csv"))
creep_rate_fp = pd.DataFrame({"lat": lats, "fp_creep_rate_mm_yr": fp_rate, "fp_creep_rate_raw": fp_rate_raw,
                              "landslide_flag": landslide_flag.astype(int), "fp_creep_rate_smooth": fp_rate_smooth,
                              "strike_deg": strike})
creep_rate_fp.to_csv(os.path.join(out_dir, "creep_rate_fp.csv"), index=False)
print("Saved fp_ts.csv and the rate profile to:", out_dir)

# %% [markdown]
# ## 12. Check: the FP rate at CFW1
#
# The crossing at the latitude of CFW1 is compared with the creepmeter in
# notebook 05. Check: it has a finite FP rate.

# %%
j = int(np.argmin(np.abs(lats - cfw1_lat)))      # the crossing at CFW1's latitude
assert np.isfinite(fp_rate[j]), "the crossing at CFW1 has no FP rate (dropped or flagged)"

# %% [markdown]
# ## 13. Figure S4: acquired versus interpolated epochs
#
# For each epoch of the common grid, whether each track is acquired or
# interpolated. Three dates are computed from the acquisition lists:
#
# - the start of sustained 6-day sampling on each track: the first date of the
#   first run of 10 acquisition intervals whose median is at most 8 days
#   (`sixday_start`);
# - the end of the early block of interpolated epochs on each track: the last
#   interpolated epoch before the first gap of more than 180 days between
#   interpolated epochs (`main_interp_block_end`);
# - the start of the same-day window: the first common date after the later of
#   those four dates. The window ends with the failure of Sentinel-1B
#   (`s1b_death`).
#
# The share of all interpolated epochs (both tracks together) that fall before
# the same-day window starts is $100\,n_{before}/n_{interp}$ (registered).

# %%
def month_year(dec_year):
    """Format a decimal year as a month and year.

    Parameters
    ----------
    dec_year : float
        Decimal year.

    Returns
    -------
    str
        Date as 'Mon YYYY'.
    """
    return utils.decimal_year_to_date(dec_year).strftime("%b %Y")


def sixday_start(dates, max_days=8.0, window=10):
    """First date from which the sampling is sustained at about 6 days.

    Parameters
    ----------
    dates : numpy.ndarray
        Acquisition dates (decimal years), sorted.
    max_days : float, optional
        Largest rolling median interval (days) that counts as 6-day sampling.
    window : int, optional
        Number of intervals in the rolling median.

    Returns
    -------
    float
        First date of the first run of `window` intervals whose median is at
        most `max_days`.
    """
    gaps = np.diff(dates) * 365.25
    rolling = pd.Series(gaps).rolling(window, min_periods=window).median().to_numpy()
    ok = np.where(rolling <= max_days)[0]
    assert len(ok) > 0, "sampling never reaches a sustained 6-day interval"
    return dates[ok[0] - window + 1]


def main_interp_block_end(dates, is_real, quiet_days=180.0):
    """End of the early block of interpolated epochs on one track.

    Parameters
    ----------
    dates : numpy.ndarray
        Dates of the common grid (decimal years).
    is_real : numpy.ndarray of bool
        True where the track is acquired at that date.
    quiet_days : float, optional
        A gap longer than this (days) between interpolated epochs closes the
        block.

    Returns
    -------
    float
        Last interpolated date before the first such gap (the first date if
        no epoch is interpolated, the last interpolated date if there is no
        such gap).
    """
    interp_dates = dates[~is_real]
    if len(interp_dates) == 0:
        return dates[0]
    breaks = np.where(np.diff(interp_dates) * 365.25 > quiet_days)[0]
    if len(breaks) == 0:
        return interp_dates[-1]
    return interp_dates[breaks[0]]


acq = {"asc": mask_stats["ascending"]["ts_dates"], "desc": mask_stats["descending"]["ts_dates"]}
sixday_asc = sixday_start(acq["asc"])
sixday_desc = sixday_start(acq["desc"])
# the same-day window starts at the first common date after both tracks reach 6-day sampling
# and after both early runs of interpolated epochs, and ends when Sentinel-1B failed
sameday_start = max(sixday_asc, sixday_desc, main_interp_block_end(common_dates, real["asc"]),
                    main_interp_block_end(common_dates, real["desc"]))
sameday_start = float(common_dates[common_dates > sameday_start].min())
in_window = (common_dates >= sameday_start) & (common_dates <= s1b_death)
# the share of all interpolated epochs (both tracks together) before the same-day window starts
# (quoted in Methods M3 and the Fig. S4 caption)
n_interp = int((~real["asc"]).sum() + (~real["desc"]).sum())
n_interp_before = int((~real["asc"] & (common_dates < sameday_start)).sum() + (~real["desc"] & (common_dates < sameday_start)).sum())
assert n_interp > 0, "no interpolated epoch on either track"
interp_before_pct = 100 * n_interp_before / n_interp

s4_lines = [f"common grid: {len(common_dates)} dates, {month_year(common_dates[0])} to {month_year(common_dates[-1])}",
      f"both orbits real       : {both_real.sum()} epochs",
      f"ascending interpolated : {(~real['asc']).sum()} ({100 * (~real['asc']).mean():.0f}%)",
      f"descending interpolated: {(~real['desc']).sum()} ({100 * (~real['desc']).mean():.0f}%)",
      f"acquisitions           : asc {acq['asc'].size}, desc {acq['desc'].size}",
      f"6-day sampling starts  : asc {month_year(sixday_asc)}, desc {month_year(sixday_desc)}",
      f"same-day window        : {month_year(sameday_start)} -> {month_year(s1b_death)} ({sameday_start:.3f}-{s1b_death:g}), "
      f"{in_window.sum()} epochs, both real {both_real[in_window].mean():.0%}, interpolated inside it: "
      f"asc {(~real['asc'])[in_window].sum()}, desc {(~real['desc'])[in_window].sum()}",
      f"interpolated epochs before the same-day window ({month_year(sameday_start)}): {n_interp_before} of {n_interp} "
      f"(asc + desc), {interp_before_pct:.1f}%"]
print("\n".join(s4_lines))

fig, ax = plt.subplots(figsize=(7.2, 2.2))
ax.plot(common_dates, real["asc"].astype(int) + 2.1, "|", ms=5, color="#E69F00")
ax.plot(common_dates, real["desc"].astype(int), "|", ms=5, color="#0072B2")
ax.set_yticks([0, 1, 2.1, 3.1])
ax.set_yticklabels(["Desc interpolated", "Desc real", "Asc interpolated", "Asc real"])
ax.set_ylim(-0.6, 3.7)
ax.set_xlabel("Time (decimal years)")
fig.tight_layout()
figS4 = os.path.join(fig_dir, f"figS_acquisitions_{params}_gap{max_interp_gap_days:.0f}d.png")
fig.savefig(figS4, dpi=300)
fig.savefig(figS4.replace(".png", ".pdf"))
print("Saved figure to:", figS4)
plt.show()

# %% [markdown]
# ## 14. Register
#
# The sidecar `03_creep_stats.txt` and the registry rows `03_*`.

# %%
lines = ["03 CROSS-FAULT CREEP TIME SERIES",
         f"aperture {dist_min * 1000:.0f}-{dist_max * 1000:.0f} m, along-strike +/-{x_width * 1000:.0f} m, "
         f"min {min_points} px/side, coherence {coh_timeseries}", ""]
for track, track_stats in mask_stats.items():
    lines.append(f"{track:<11} {track_stats['n_crossings']} crossings, {track_stats['n_dates']} dates; rate mask n={track_stats['n_px']} px, "
                 f"sigma {track_stats['sigma']:.3f} mm/yr, cut +/-{track_stats['rate_cut']:.3f}, rejected {track_stats['rej_pct']:.1f}%, "
                 f"max|dev| {track_stats['max_dev']:.1f}")
lines += ["", *s4_lines, "",
          f"crossings analysed {len(fp_rate_raw)}, landslide-flagged {int(landslide_flag.sum())} (trend residual sigma {trend_sigma:.2f} mm/yr)",
          f"FP creep rate (raw, cleaned): n {valid.size}, median {np.median(valid):.2f}, 5-95% {np.percentile(valid, 5):.2f} to "
          f"{np.percentile(valid, 95):.2f}, min-max {valid.min():.2f} to {valid.max():.2f} mm/yr",
          f"FP creep rate (smoothed {rate_smooth_km} km): median {np.nanmedian(fp_rate_smooth):.2f}, "
          f"5-95% {np.nanpercentile(fp_rate_smooth, 5):.2f} to {np.nanpercentile(fp_rate_smooth, 95):.2f} mm/yr"]
sidecar = REG.sidecar("03_creep", lines)

for track, track_stats in mask_stats.items():
    tag = track[:3]
    REG.register(f"03_n_crossings_{tag}", track_stats["n_crossings"], "count", 0, where="M3", cell="extract", sidecar=sidecar)
    REG.register(f"03_n_acquisitions_{tag}", track_stats["n_dates"], "count", 0, where="M1, M3", cell="extract", sidecar=sidecar)
    REG.register(f"03_rate_mask_sigma_{tag}", track_stats["sigma"], "mm/yr", 2, where="M3, Fig S3", cell="extract", sidecar=sidecar)
    REG.register(f"03_rate_mask_cut_{tag}", track_stats["rate_cut"], "mm/yr", 2, where="M3, Fig S3", cell="extract", sidecar=sidecar)
    REG.register(f"03_rate_mask_rejected_pct_{tag}", track_stats["rej_pct"], "pct", 1, where="M3, Fig S3", cell="extract", sidecar=sidecar)
    REG.register(f"03_rate_mask_maxdev_{tag}", track_stats["max_dev"], "mm/yr", 1, where="M3, Fig S3", cell="extract", sidecar=sidecar)
REG.register("03_n_common_dates", len(common_dates), "count", 0, where="M3, Fig S4", cell="common-grid", sidecar=sidecar)
REG.register("03_n_both_real", int(both_real.sum()), "count", 0, where="M3, Fig S4", cell="common-grid", sidecar=sidecar)
REG.register("03_interp_pct_asc", 100 * (~real["asc"]).mean(), "pct", 0, where="M3, Fig S4", cell="common-grid", sidecar=sidecar)
REG.register("03_interp_pct_desc", 100 * (~real["desc"]).mean(), "pct", 0, where="M3, Fig S4", cell="common-grid", sidecar=sidecar)
REG.register("03_window_start", grid_start, "decyear", 3, where="M3", cell="common-grid", sidecar=sidecar)
REG.register("03_window_end", grid_end, "decyear", 3, where="M3", cell="common-grid", sidecar=sidecar)
REG.register("03_sixday_start_asc", month_year(sixday_asc), "date", 0, where="M3", cell="figS4", sidecar=sidecar)
REG.register("03_sixday_start_desc", month_year(sixday_desc), "date", 0, where="M3", cell="figS4", sidecar=sidecar)
REG.register("03_sameday_window_epochs", int(in_window.sum()), "count", 0, where="M3", cell="figS4", sidecar=sidecar)
REG.register("03_sameday_start", month_year(sameday_start), "date", 0, where="M3", cell="figS4", sidecar=sidecar)
REG.register("03_interp_before_sameday_pct", interp_before_pct, "pct", 0, where="M3, Fig S4", cell="figS4", sidecar=sidecar)
REG.register("03_n_crossings_common", len(fp_rate_raw), "count", 0, where="M3", cell="decompose", sidecar=sidecar)
REG.register("03_n_landslide_flagged", int(landslide_flag.sum()), "count", 0, where="M3", cell="landslide-flag", sidecar=sidecar)
REG.register("03_landslide_flagged_pct", 100 * landslide_flag.mean(), "pct", 0, where="M3", cell="landslide-flag", sidecar=sidecar)
REG.register("03_rate_median_mm_yr", np.median(valid), "mm/yr", 2, where="Results", cell="rate-profile", sidecar=sidecar)
REG.register("03_rate_p5_mm_yr", np.percentile(valid, 5), "mm/yr", 2, where="Results", cell="rate-profile", sidecar=sidecar)
REG.register("03_rate_p95_mm_yr", np.percentile(valid, 95), "mm/yr", 2, where="Results", cell="rate-profile", sidecar=sidecar)
REG.render()
