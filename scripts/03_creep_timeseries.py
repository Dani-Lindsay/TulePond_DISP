# %% [markdown]
# # [1] 03 Cross-fault creep time series
#
# ```
# Created on Wed Sep 23 2026
# @author: daniellelindsay
# Run in opera_disp_feb13. ~15 min per track for the time-series read, the rest seconds.
# ```
#
# **The method.** At every 100 m crossing of the mapped Hayward trace:
#
# 1. take the InSAR pixels in a box perpendicular to the LOCAL strike:
#    within `x_width` along the fault and between `dist_min` and `dist_max`
#    from the trace, on each side (`w_subset`, `e_subset`)
# 2. median of the west pixels and median of the east pixels at every date
#    (`w_avg`, `e_avg`)
# 3. `west_east_ts = e_avg - w_avg`, in LOS mm
#
# That is the whole thing. The difference cancels the reference datum and the
# per-date common-mode noise (Methods M3). It is done separately for the
# ascending (09157) and descending (11115) frames. Only crossings that both
# tracks cover go on to the common grid; the south end is where the ascending
# frame stops (~37.486 N), the north end is Point Pinole.
#
# PREPUB: (Old_WD2TB_PhD/Scripts/fault_creeprate_itterate_segments.py and the AGU_extract_creep_rates_*.py scripts)
#
# **The four additions the paper needs**, each in its own section below:
#
# - a per-pixel RATE MASK: a pixel whose LOS rate is > 3 robust sigma from the
#   median rate of ITS OWN SIDE at that crossing is a landslide, quarry or edge
#   and is dropped (Fig S3)
# - a COMMON DATE GRID: each track interpolated onto the union of dates, only
#   across gaps shorter than 45 days (Fig S4)
# - the asc + desc DECOMPOSITION into fault-parallel (FP) and vertical (Up)
#   with the local strike and the local incidence / azimuth of each crossing;
#   `fp_ts.csv` is THE input to 04, 05, 09 and 11
# - a crossing-level LANDSLIDE FLAG: a crossing whose FP rate is > 3 robust
#   sigma from a 21-crossing running median is dropped from every product
#
# **Geometry** (definitions from MintPy `utils0.enu2los`; check there, not here):
#
# ```
# inc_angle  - incidence angle of the LOS vector, measured from vertical
# az_angle   - azimuth of the LOS vector FROM THE GROUND TO THE SATELLITE, measured
#              from north, ANTI-CLOCKWISE positive.  head_angle = 90 - az_angle.
#              Ours: ascending  ~100 (flies N10W, looks east, sits WEST of the pixel)
#                    descending ~260 (flies S10E, looks west, sits EAST of the pixel)
# profile_az - the cross-fault direction of each crossing, compass bearing clockwise
#              from north, pointing to the EAST side (~54).  strike = profile_az + 90 (~144, SE)
# ```
#
# **Inputs**: per track `timeseries_tropo.h5`, `velocity_masked.h5`,
# `geometryGeo.h5` and `maskTempCoh_07.h5`; `hayward_fault_profiles.csv`; the
# USGS creepmeter records (the creepmeter CHECK only).
#
# **Outputs** (`outputs/03_creep_timeseries/`):
# - per track: `creep_ts_LOS_<track>.csv` (lat x decimal-year matrix, mm LOS),
#   `creep_geom_<track>.csv` (mean incidence / azimuth of the pixels used at
#   each crossing; the decomposition uses them crossing by crossing),
#   `creep_params_<track>.json` (the reuse key), `rate_deviation_<track>.csv`
# - common grid: `asc_los_interp.csv`, `desc_los_interp.csv`, `fp_ts.csv`, `up_ts.csv`
# - along-fault rates: `creep_rate_fp.csv`, `creep_rate_up.csv`; `decompose_params.json`
# - `03_creep_stats.txt` and the registry rows `03_*`
#
# **Figures**: S3 `figS_ratemask_<params>_nsigma3` (the rate mask), S4
# `figS_acquisitions_<params>_gap45d` (real versus interpolated epochs), and the
# QC plots `qc03_profile_azimuth_smooth21` (trace azimuth) and
# `qc03_creep_rate_profile_<params>` (along-fault creep rate).
#
# **What to check when you run it.** Rate-mask sigma ~0.55 mm/yr on both
# tracks, ~6 % of pixels rejected; 439 common dates, 291 with both tracks real;
# 71 of 610 crossings landslide-flagged; median FP creep rate ~4.7 mm/yr. The
# crossing at CFW1 redone by hand must equal the extracted series (asserted).
# CFW1's crossing rate (~6.8 mm/yr) beside the creepmeter's (~7.0 mm/yr) is
# printed for reading, not asserted.

# %%
# [2]
import os
import json

import insar_utils as utils
from TulePond_filepaths import (common_paths, insar_dic_list, creepmeters, unit, params,
                                dist_min, dist_max, x_width, min_points, coh_timeseries,
                                rate_nsigma, az_smooth_n, max_interp_gap_days, rate_smooth_km,
                                repo_dir)

import numpy as np                               # Matrix calculations
import pandas as pd                              # Tables
import h5py                                      # MintPy HDF5 products
from pyproj import Transformer                   # lon/lat -> UTM
import matplotlib.pyplot as plt

# %% [markdown]
# ## [3] Paths and parameters
#
# The aperture, mask and smoothing choices come from `TulePond_filepaths.py`
# so every notebook uses the same values (reasons in the assumptions
# register). The few below are used only here.

# %%
# [4]
notebook = "03_creep_timeseries"
out_dir = os.path.join(common_paths["out_dir"], notebook)
fig_dir = common_paths["fig_dir"]
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

utm_epsg = 32610         # UTM 10N, the grid the frames live on
trend_window = 21        # crossings (~2 km) in the running median used to flag landslide crossings
trend_nsigma = 3.0       # a crossing whose FP rate departs from that trend by more than this is dropped
s1b_death = 2021.98      # Sentinel-1B failed 2021-12-23 (decimal year 2021.975, kept here as 2021.98): 6-day sampling ends (Fig S4)

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
# ## [5] Fault trace
#
# `hayward_fault_profiles.csv` is the mapped trace resampled every 100 m
# (built in notebook 01 from the KML): `ft_lon, ft_lat, profile_az`, south to
# north, `profile_az` the cross-fault azimuth pointing to the east side.
#
# The raw azimuth carries the 100 m jitter of hand digitising (a few kinks of
# 10-20 degrees). A 21-crossing (2.1 km) rolling median removes the kinks and
# keeps the real bends of the trace; the same smoothing is used in notebook 02.
# At Tule Pond it changes the band rates by ~0.2 mm/yr and the event steps by
# ~0.1 mm and does not move the section ends (checked 2026-09-23).

# %%
# [6]
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

utils.mpl_paper_style(base=8)          # Helvetica, 8 pt, editable PDF text
fig, ax = plt.subplots(figsize=(7.2, 2.2))
ax.plot(x_fault.km, x_fault.profile_az, ".", color="0.6", ms=3, label="Local azimuth at each 100 m crossing")
ax.plot(x_fault.km, x_fault.az_smooth, "-", color="k", lw=1.2, label=f"Rolling median over {az_smooth_n} crossings")
for code, cm_lat, cm_lon, cm_name in creepmeters:
    ax.axvline(km_of([cm_lat])[0], color="#0072B2", lw=0.8, ls=":")
    ax.text(km_of([cm_lat])[0], ax.get_ylim()[1], code, ha="center", va="bottom", fontsize=6, color="#0072B2")
ax.set_xlabel("Distance from Point Pinole (km)")
ax.set_ylabel("Cross-fault azimuth (deg)")
ax.legend(fontsize=7)
fig.tight_layout()
f_qc = os.path.join(fig_dir, f"qc03_profile_azimuth_smooth{az_smooth_n}.png")
fig.savefig(f_qc, dpi=300)
fig.savefig(f_qc.replace(".png", ".pdf"))
plt.show()

# %% [markdown]
# ## [7] Functions
#
# Only what is used for both tracks. `get_window_points` returns, for the
# coherent pixels in a small window around one crossing, `p` (across the
# fault, positive to the east side) and `q` (along the fault) in km, computed
# in UTM. The window is small because the displacement cube is 13 GB per track
# and is read one window at a time. Its half-width is the smallest number of
# pixels that holds the whole `dist_max` x `x_width` box at any orientation:
# the box corners are hypot(dist_max, x_width) = 250 m from the crossing, so
# 10 pixels (300 m) each way, 21 x 21 pixels per crossing. This is NOT a
# selection; any window that contains the box gives the same answer.
# `get_fault_pairs` splits the pixels into the west and east sides.
#
# PREPUB: `get_fault_pairs` is unchanged from the 2025 scripts. `get_window_points` replaces the
# PREPUB: `pygmt.project` call of those scripts and returns the same `p` and `q`; at the CFW1
# PREPUB: crossing the two routes pick the same pixels (checked 2026-09-23).

# %%
# [8]
def read_insar_data(dic):
    """Geometry, coherence mask, velocity (mm/yr), dates and the grid of one
    track. The displacement cube is NOT loaded: `ts` is the open h5py dataset
    and is read per crossing window. Close `ts_h5` when done."""
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
    """The coherent pixels in a square window around one crossing, as a table:
    row, col (pixel), p = signed distance across the fault (km, + toward az,
    the east side), q = distance along the fault (km), vel, inc, azi.
    Pixels further than x_width along the fault are dropped. None if the
    crossing is off the frame."""
    # The frame is a regular UTM grid: pixel (row, col) sits at
    #     easting  = x_first + col * x_step        (x_step = +30 m, col 0 is the west edge)
    #     northing = y_first + row * y_step        (y_step = -30 m, row 0 is the north edge)
    # so the pixel under the crossing is found by inverting that.
    col_centre = int(round((easting_0 - insar["x_first"]) / insar["x_step"]))
    row_centre = int(round((northing_0 - insar["y_first"]) / insar["y_step"]))
    # The square of +/- win pixels around it (300 m, enough to hold the whole dist_max x x_width
    # box whatever its orientation), clipped at the frame edge. Nothing is selected here; the
    # selection is the p, q filter below and get_fault_pairs. Fewer than 3 pixels a side means
    # the crossing is off the frame.
    row_min = max(0, row_centre - insar["win"])
    row_max = min(insar["length"], row_centre + insar["win"] + 1)
    col_min = max(0, col_centre - insar["win"])
    col_max = min(insar["width"], col_centre + insar["win"] + 1)
    if row_max - row_min < 3 or col_max - col_min < 3:
        return None
    # easting and northing of every pixel in the window, then the distance of each pixel from the
    # crossing along the cross-fault azimuth (p) and along the fault (q)
    # (the pygmt.project step of the 2025 scripts)  # PREPUB
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
    """West and east pixels of one crossing: dist_min < |p| < dist_max (km) on each side."""
    w_subset = points[(points.p > -dist_max) & (points.p < -dist_min)]
    e_subset = points[(points.p > dist_min) & (points.p < dist_max)]
    return w_subset, e_subset


def read_creep_ts(path):
    """A lat x decimal-year csv back as a DataFrame with float index and columns."""
    df = pd.read_csv(path, index_col=0)
    df.columns = df.columns.astype(float)
    df.index = df.index.astype(float)
    return df

# %% [markdown]
# ## [9] Extract the cross-fault time series, one track at a time
#
# Two loops over the crossings per track.
#
# **Loop 1: the pixel pairs and the rate mask.** At every crossing take the
# window, split it into `w_subset` and `e_subset`, and note every pixel's LOS
# rate as its deviation from the median rate of ITS OWN side. Fault creep is a
# difference BETWEEN sides so it does not enter, and the pooled deviations are
# centred on zero along the whole fault. One robust sigma per track sets the
# cut.
#
# **Loop 2: the difference.** Drop the pixels beyond the cut, read the
# displacement cube in the crossing's window, median each side per date, and
# difference east minus west. The mean incidence and azimuth of the pixels
# used are kept for the decomposition.
#
# The cube read takes ~15 min per track, so if the per-track files are already
# on disk with the same reuse key the track is skipped and the mask numbers are
# recomputed from the saved deviations. The key is the parameters plus the
# identity of the inputs: the dates of the time series and the fingerprints of
# the time series, velocity, geometry, coherence mask and fault crossings. A
# data refresh, a re-mask or a re-mapped trace therefore forces a re-run;
# delete the files to force one by hand.

# %%
# [10]
params_now = dict(coh=coh_timeseries, dist_min=dist_min, dist_max=dist_max, x_width=x_width,
                  min_points=min_points, rate_nsigma=rate_nsigma)

# PREPUB-START: history of why the reuse key includes the data
# A data refresh adds epochs to timeseries_tropo.h5 while every parameter stays the same, so
# keying on parameters alone silently reused the previous extraction: on 2026-09-28 the
# March-2026 run rebuilt fp_ts.csv from September LOS series and every product downstream
# inherited the old dates.
# PREPUB-END


def ts_data_key(dic):
    """Identity of this track's inputs, for the reuse test below: the dates of
    the time series and the fingerprints of every file the extraction reads
    (time series, velocity, geometry, coherence mask, fault crossings)."""
    with h5py.File(dic["ts_file"], "r") as h:
        dates = h["date"][:]
    key = dict(n_dates=len(dates), first_date=dates[0].decode(), last_date=dates[-1].decode())
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

    # ---------- already on disk with this reuse key? then only the mask numbers
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

    # ---------- Loop 2: drop the fast pixels, read the cube, median each side, east - west
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
# ## [11] CHECK: one crossing by hand
#
# The crossing at CFW1's latitude is redone on the ascending track at the last
# epoch, step by step, so the median difference is visible without the loop:
# take the window, split it into west and east, drop the pixels beyond the
# track's rate cut, and difference the side medians. The result must equal the
# value in `creep_ts_LOS_ascending.csv` (to 1e-3 mm).

# %%
# [12]
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
# the rate mask, exactly as loop 2 applies it
rate_cut_asc = mask_stats["ascending"]["rate_cut"]
w_keep = w_subset[np.abs(w_subset["vel"] - w_subset["vel"].median()) <= rate_cut_asc]
e_keep = e_subset[np.abs(e_subset["vel"] - e_subset["vel"].median()) <= rate_cut_asc]
east_median = np.nanmedian(last[e_keep["row"].to_numpy(), e_keep["col"].to_numpy()])
west_median = np.nanmedian(last[w_keep["row"].to_numpy(), w_keep["col"].to_numpy()])
by_hand = east_median - west_median
from_file = read_creep_ts(os.path.join(out_dir, "creep_ts_LOS_ascending.csv")).loc[cfw1_crossing["ft_lat"]].iloc[-1]
print(f"crossing at CFW1's latitude, {cfw1_crossing.ft_lat:.4f} N ({cfw1_crossing.km:.1f} km), ascending, last epoch:")
print(f"   {len(w_subset)} west px, {len(e_subset)} east px; after the rate mask {len(w_keep)} west, {len(e_keep)} east")
print(f"   east median - west median by hand = {by_hand:.3f} mm;  in creep_ts_LOS_ascending.csv = {from_file:.3f} mm")
assert abs(by_hand - from_file) < 1e-3, "the crossing redone by hand differs from the extracted series"
print("PASS: the extraction loop is the four steps above")
print(points.head())

# %% [markdown]
# ## [13] Figure S3: the rate mask
#
# The pooled deviation population per track on a log count axis, with the
# cut. The tail beyond the cut is what is rejected: landslides, quarries and
# their edges. The core is kept whatever the local creep rate, because creep is
# a difference between sides and never enters a within-side deviation.

# %%
# [14]
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
# ## [15] Common date grid
#
# The decomposition needs both tracks at the same epochs. Each crossing is
# interpolated against its REAL acquisition dates onto the union of both
# tracks' dates within their overlap, but only across gaps shorter than
# `max_interp_gap_days` (45 d): that bridges the 6, 12 and 24 day sampling and
# a few missed acquisitions, and leaves the two genuine 2018 ascending breaks
# (72 and 84 d) as NaN. Nothing is extrapolated. Only crossings on both
# tracks are kept (the ascending frame sets the south end).

# %%
# [16]
def interp_to_dates(target_dates, dates, values):
    """One crossing's series onto target_dates against its real dates. NaN
    inside gaps longer than max_interp_gap_days and outside the observed range."""
    out = np.full(len(target_dates), np.nan)
    ok = np.isfinite(values)
    if ok.sum() < 2:
        return out
    dates, values = dates[ok], values[ok]
    interp = np.interp(target_dates, dates, values)
    left = np.clip(np.searchsorted(dates, target_dates, side="right") - 1, 0, len(dates) - 1)   # real date at or before
    right = np.clip(left + 1, 0, len(dates) - 1)                                                # real date after
    is_real = np.isclose(target_dates, dates[left], atol=0.5 / 365.25)                          # a real acquisition
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
lats_all = np.array(sorted(set(asc_los.index) & set(desc_los.index)))
lats = lats_all
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

# which epochs are REAL on each track (an acquisition within half a day) and which are interpolated
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

# %% [markdown]
# ## [17] Decompose each crossing into fault-parallel and vertical
#
# Per crossing, with the LOCAL viewing geometry of that crossing: its own
# incidence and azimuth on each track (the mean over the pixels used at that
# crossing, saved in `creep_geom_<track>.csv`) and its LOCAL strike (smoothed
# cross-fault azimuth + 90). Along the fault the incidence varies by ~1.7 deg
# (ascending) and ~2.9 deg (descending); a single mean per track would change FP
# by less than 1 %, but every time-series product here uses local geometry.
# (Dani 2026-10-03: local strike and geometry for anything built on the time
# series; an averaged strike only where the velocity FIELDS are decomposed.)
#
# `utils.decompose_asc_desc` solves $[d_{asc}, d_{desc}]^T = G\,[U, FP]^T$ by
# least squares at every epoch (the design matrix is pinned by the tests). FP
# is positive along the strike bearing, so right-lateral creep is positive.

# %%
# [18]
# the incidence and azimuth of every kept crossing, on each track (the crossings must match exactly)
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

up_matrix = np.full((len(lats), len(common_dates)), np.nan)
fp_matrix = np.full((len(lats), len(common_dates)), np.nan)
for i, lat in enumerate(lats):
    d_asc = los_interp["asc"].values[i]
    d_desc = los_interp["desc"].values[i]
    ok = np.isfinite(d_asc) & np.isfinite(d_desc)
    if ok.sum() == 0:
        continue
    up_matrix[i, ok], fp_matrix[i, ok] = utils.decompose_asc_desc(
        d_asc[ok], d_desc[ok], crossing_inc["asc"][i], crossing_azi["asc"][i],
        crossing_inc["desc"][i], crossing_azi["desc"][i], strike[i])
up_ts = pd.DataFrame(up_matrix, index=lats, columns=common_dates)
fp_ts = pd.DataFrame(fp_matrix, index=lats, columns=common_dates)
up_ts.index.name = "lat"
fp_ts.index.name = "lat"
mid = len(lats) // 2
print(f"decomposed {np.isfinite(fp_matrix).any(axis=1).sum()} crossings; G for the mid-fault crossing ({lats[mid]:.3f} N):")
print(utils.design_matrix_up_fp([crossing_inc["asc"][mid], crossing_inc["desc"][mid]],
                                [crossing_azi["asc"][mid], crossing_azi["desc"][mid]], strike[mid]).round(3), "  (columns U, FP)")

# %% [markdown]
# ## [19] Along-fault creep rate and the crossing-level landslide flag
#
# A linear rate per crossing (`np.polyfit`). The pixel
# rate mask removes landslide EDGES, but a landslide that dominates one side IS
# that side's median; such crossings stand out as spikes against the smooth
# along-fault trend. Crossings more than `trend_nsigma` robust sigma from a
# 21-crossing running median are dropped from every product. The surviving
# profile is smoothed with a 0.5 km Gaussian that preserves the gaps
# (`utils.gaussian_smooth_km`).
#
# PREPUB: (`np.polyfit`, as in the 2025 scripts)

# %%
# [20]
def rate_per_crossing(ts):
    """Linear rate (mm/yr) of every row of a lat x decimal-year table; NaN if fewer than 6 dates."""
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
for ts in (los_interp["asc"], los_interp["desc"], up_ts, fp_ts):
    ts.iloc[landslide_flag, :] = np.nan
fp_rate = np.where(landslide_flag, np.nan, fp_rate_raw)
fp_rate_smooth = utils.gaussian_smooth_km(fp_rate, lats, rate_smooth_km)    # NaN-aware, gap-preserving
up_rate = rate_per_crossing(up_ts)
up_rate_smooth = utils.gaussian_smooth_km(up_rate, lats, rate_smooth_km)

valid = fp_rate[np.isfinite(fp_rate)]
print(f"FP creep rate: {valid.size} valid crossings, median {np.median(valid):.2f} mm/yr, "
      f"5-95% {np.percentile(valid, 5):.2f} to {np.percentile(valid, 95):.2f}, min-max {valid.min():.2f} to {valid.max():.2f}")
print(f"Up rate: median {np.nanmedian(up_rate):+.2f} mm/yr, 5-95% {np.nanpercentile(up_rate, 5):+.2f} to {np.nanpercentile(up_rate, 95):+.2f}")

# %% [markdown]
# ## [21] Save the common-grid products
#
# `asc_los_interp.csv`, `desc_los_interp.csv`, `fp_ts.csv`, `up_ts.csv` (lat x
# decimal year, landslide-flagged crossings NaN), the rate profiles
# `creep_rate_fp.csv` and `creep_rate_up.csv`, and `decompose_params.json`.

# %%
# [22]
los_interp["asc"].to_csv(os.path.join(out_dir, "asc_los_interp.csv"))
los_interp["desc"].to_csv(os.path.join(out_dir, "desc_los_interp.csv"))
fp_ts.to_csv(os.path.join(out_dir, "fp_ts.csv"))
up_ts.to_csv(os.path.join(out_dir, "up_ts.csv"))
creep_rate_fp = pd.DataFrame({"lat": lats, "fp_creep_rate_mm_yr": fp_rate, "fp_creep_rate_raw": fp_rate_raw,
                              "landslide_flag": landslide_flag.astype(int), "fp_creep_rate_smooth": fp_rate_smooth,
                              "strike_deg": strike})
creep_rate_fp.to_csv(os.path.join(out_dir, "creep_rate_fp.csv"), index=False)
pd.DataFrame({"lat": lats, "up_rate_mm_yr": up_rate, "up_rate_smooth": up_rate_smooth}
             ).to_csv(os.path.join(out_dir, "creep_rate_up.csv"), index=False)
decompose_params = dict(max_interp_gap_days=max_interp_gap_days, rate_smooth_km=rate_smooth_km,
                        trend_window=trend_window, trend_nsigma=trend_nsigma, **params_now)
with open(os.path.join(out_dir, "decompose_params.json"), "w") as fh:
    json.dump(decompose_params, fh, indent=1)
print("Saved fp_ts.csv, up_ts.csv, the interpolated LOS matrices and the rate profiles to:", out_dir)

# %% [markdown]
# ## [23] CHECK: the creep rate at the creepmeters
#
# The FP rate of the crossing at each creepmeter's latitude against the
# instrument's rate over the same window (the full comparison is notebook 05).
# A comparison only: the two rates are computed independently and nothing is
# fitted, scaled or offset to make one match the other (the InSAR is tied to
# GNSS in the preparation chain, never to the creepmeters).
# The rates are printed only, no tolerance is asserted: the agreement is read
# off the validation figure of notebook 05, where the values are written on.
# COZ1 is the known exception (the InSAR misses part of its two large creep
# events). The one assert is that the crossing at CFW1 has a rate at all.

# %%
# [24]
creepmeter_rate = {}
for code, cm_lat, cm_lon, cm_name in creepmeters:
    creepmeter_path = utils.creepmeter_file(common_paths["creepmeter_dir"], code)
    REG.inputs(creepmeter_path)
    creepmeter_record = utils.read_creepmeter_10min(creepmeter_path, date_start=grid_start, date_end=grid_end)
    creepmeter_rate[code] = utils.linear_rate(creepmeter_record["decimal_year"].values, creepmeter_record["slip_mm"].values)[0]
print("crossing FP rate at each creepmeter (mm/yr):")
insar_rate_at = {}
for code, cm_lat, cm_lon, cm_name in creepmeters:
    j = int(np.argmin(np.abs(lats - cm_lat)))
    insar_rate_at[code] = fp_rate[j]
    print(f"  {code} ({cm_name}, {km_of([cm_lat])[0]:.1f} km): InSAR {fp_rate[j]:5.2f}   creepmeter {creepmeter_rate[code]:5.2f}")
assert np.isfinite(insar_rate_at["CFW1"]), "the crossing at CFW1 has no FP rate (dropped or flagged)"

# %% [markdown]
# ## [25] QC figure: the along-fault creep-rate profile
#
# Raw per-crossing rates (grey), flagged crossings (vermillion), the smoothed
# profile (black) and the creepmeter rates (blue). North (Point Pinole) on the
# left.

# %%
# [26]
km = km_of(lats)
fig, ax = plt.subplots(figsize=(7.2, 2.4))
ax.plot(km, fp_rate_raw, ".", color="0.6", ms=3, label="per-crossing rate")
ax.plot(km[landslide_flag], fp_rate_raw[landslide_flag], "x", color="#D55E00", ms=4, label="flagged (dropped)")
ax.plot(km, fp_rate_smooth, "-", color="k", lw=1.2, label=f"{rate_smooth_km} km smooth")
for k, (code, cm_lat, cm_lon, cm_name) in enumerate(creepmeters):
    if k == 0:
        label = "creepmeter"          # one legend entry for the five creepmeters
    else:
        label = None
    ax.plot(km_of([cm_lat])[0], creepmeter_rate[code], "v", color="#0072B2", ms=6, label=label)
    ax.text(km_of([cm_lat])[0], 11.5, code, ha="center", fontsize=6, color="#0072B2")
ax.set_xlabel("Distance from Point Pinole (km)")
ax.set_ylabel("Fault-parallel creep rate (mm/yr)")
ax.set_ylim(0, 12)
ax.grid(True, ls="--", alpha=0.4)
ax.legend(fontsize=6, ncol=4)
fig.tight_layout()
f_qc = os.path.join(fig_dir, f"qc03_creep_rate_profile_{params}.png")
fig.savefig(f_qc, dpi=300)
fig.savefig(f_qc.replace(".png", ".pdf"))
plt.show()

# %% [markdown]
# ## [27] Figure S4: real versus interpolated epochs
#
# Which track is measured and which is filled in at each epoch of the common
# grid. Before both orbits reached 6-day sampling they shared few dates, so
# nearly every early epoch has one track interpolated. The 6-day start dates and
# the window when both were acquired on the same days are computed from the
# acquisition lists, not assumed.

# %%
# [28]
def month_year(dec_year):
    """Decimal year -> 'Mon YYYY'."""
    return utils.decimal_year_to_date(dec_year).strftime("%b %Y")


def sixday_start(dates, max_days=8.0, window=10):
    """First date from which sampling is sustained ~6-day (rolling median interval <= max_days)."""
    gaps = np.diff(dates) * 365.25
    rolling = pd.Series(gaps).rolling(window, min_periods=window).median().to_numpy()
    ok = np.where(rolling <= max_days)[0]
    assert len(ok) > 0, "sampling never reaches a sustained 6-day interval"
    return dates[ok[0] - window + 1]


def main_interp_block_end(dates, is_real, quiet_days=180.0):
    """End of the main early run of interpolated epochs (a quiet stretch of quiet_days closes it)."""
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

s4_lines = [f"common grid: {len(common_dates)} dates, {month_year(common_dates[0])} to {month_year(common_dates[-1])}",
      f"both orbits real       : {both_real.sum()} epochs",
      f"ascending interpolated : {(~real['asc']).sum()} ({100 * (~real['asc']).mean():.0f}%)",
      f"descending interpolated: {(~real['desc']).sum()} ({100 * (~real['desc']).mean():.0f}%)",
      f"acquisitions           : asc {acq['asc'].size}, desc {acq['desc'].size}",
      f"6-day sampling starts  : asc {month_year(sixday_asc)}, desc {month_year(sixday_desc)}",
      f"same-day window        : {month_year(sameday_start)} -> {month_year(s1b_death)} ({sameday_start:.3f}-{s1b_death:g}), "
      f"{in_window.sum()} epochs, both real {both_real[in_window].mean():.0%}, interpolated inside it: "
      f"asc {(~real['asc'])[in_window].sum()}, desc {(~real['desc'])[in_window].sum()}"]
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
# ## [29] REGISTER
#
# The sidecar `03_creep_stats.txt` and the registry rows `03_*`.

# %%
# [30]
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
REG.register("03_n_crossings_common", len(fp_rate_raw), "count", 0, where="M3", cell="decompose", sidecar=sidecar)
REG.register("03_n_landslide_flagged", int(landslide_flag.sum()), "count", 0, where="M3", cell="landslide-flag", sidecar=sidecar)
REG.register("03_landslide_flagged_pct", 100 * landslide_flag.mean(), "pct", 0, where="M3", cell="landslide-flag", sidecar=sidecar)
REG.register("03_rate_median_mm_yr", np.median(valid), "mm/yr", 2, where="Results", cell="rate-profile", sidecar=sidecar)
REG.register("03_rate_p5_mm_yr", np.percentile(valid, 5), "mm/yr", 2, where="Results", cell="rate-profile", sidecar=sidecar)
REG.register("03_rate_p95_mm_yr", np.percentile(valid, 95), "mm/yr", 2, where="Results", cell="rate-profile", sidecar=sidecar)
REG.render()
