#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Danielle Lindsay

The aperture sweep that notebook 05 scores (the aperture ladder and the coherence comparison of the chosen bin): the
cross-fault creep time series of notebook 03, re-run for every cross-fault bin (inner, outer distance from the trace)
and temporal-coherence threshold that 05 reads.

Every bin runs the code of notebook 03 (`creep_profiles.py`), with only the aperture and the coherence mask changed:
1) Per track: the coherent pixels of each 100 m crossing in the bin, the near-fault rate mask, and the east median
   minus the west median at every date (03 section 5)
2) Both tracks on the union of their dates, interpolated across gaps of at most `max_interp_gap_days` (03 section 8)
3) Fault-parallel (FP) decomposition with the geometry and strike of each crossing (03 section 9), the
   crossing-level landslide flag (03 section 10), and the FP rate profile (03 section 11)
4) The OLS rates of the five creepmeters against the FP stack at each one (the stack rule of notebook 05, section 3,
   with its defaults), summarised as the RMS and mean of (InSAR rate - creepmeter rate) over the stations

The other settings are those of notebook 03 (TulePond_filepaths.py: x_width, min_points, rate_nsigma, az_smooth_n,
max_interp_gap_days, rate_smooth_km; creep_profiles.py: the UTM grid and the landslide flag). The dates are decimal
years in the project convention (insar_utils.date_to_decimal_year, 2000 + days / 365.25), as in notebook 03.

The bins: the four sets of notebook 05's aperture ladder at coherence 0.9 (fixed 100 m width stepping out, growing
outer edge, the bins of the original grid, inner edge stepping out to 150 m), and the bin of the paper (dist_min to
dist_max, 30 to 150 m) at coherence 0.7, 0.8 and 0.9. The run of the paper's bin at coh_timeseries (0.7) is the same
computation as notebook 03, and the script asserts that its fp_ts.csv equals 03's when 03's product exists.

Coherence masks: 0.7 and 0.9 are the run04 masks (`maskTempCoh_07/09.h5`). The 0.8 mask is not written by run04;
it is built here by the same rule (unbroken record, `maskRecord.h5`, and average temporal coherence >= 0.8,
`avgSpatialCoh.h5`) into `<out_dir>/masks/<frame>/`. The rule is checked first: rebuilt at 0.7 and 0.9 it must
equal run04's masks.

Usage (from the repo root; minutes, it reads the displacement cube once per bin and track):
    python data_prep/run07_aperture_sweep.py                    # every bin, into data_dir/aperture_sweep (sweep_dir)
    python data_prep/run07_aperture_sweep.py --out-dir DIR      # elsewhere
    python data_prep/run07_aperture_sweep.py --tags ap30-150m_coh0.7_mp5 ap0-100m_coh0.9_mp5

Output (`--out-dir`, default sweep_dir = data_dir/aperture_sweep, the folder notebook 05 reads):
    <tag>/fp_ts.csv                     FP (mm), latitude x decimal year, landslide-flagged crossings NaN (03 layout)
    <tag>/creep_ts_LOS_<track>.csv      per-track east - west LOS (mm), latitude x decimal year
    <tag>/creep_geom_<track>.csv        mean incidence and azimuth of the pixels used at each crossing
    <tag>/creep_params_<track>.json     the bin, the settings and the input fingerprints (03's reuse key)
    <tag>/creep_rate_fp.csv             FP rate per crossing, the landslide flag, the smoothed rate, the strike
    sweep_summary_mp<min_points>.csv    one row per bin: tag, inner_m, outer_m, coh, min_points, n_stations,
                                        rate_rms, rate_bias, n_cross_asc, n_cross_desc, n_crossings, n_flagged
    sweep_results_mp<min_points>.csv    one row per bin and creepmeter: the two rates, their difference, the stack
    masks/<frame>/maskTempCoh_08.h5     the coherence-0.8 mask
    run07_aperture_sweep.log            what was run (git hash, settings, per-bin numbers)
where tag = ap<inner>-<outer>m_coh<coh>_mp<min_points> (inner, outer in m).
Source: written for this project; the method is that of notebook 03 (Li et al. 2023).
"""
import argparse
import os
import sys
import time
import warnings

import h5py
import numpy as np
import pandas as pd

import os as _os, sys as _sys        # path shim: import the repo root modules
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
import insar_utils as utils
import creep_profiles as cp
from TulePond_filepaths import (data_dir, common_paths, insar_dic_list, frames, creepmeters, creepmeter_glitches,
                                unit, dist_min, dist_max, x_width, min_points, coh_timeseries, rate_nsigma,
                                az_smooth_n, max_interp_gap_days, rate_smooth_km, repo_dir)
from run04_velocity import read_coherence, write_mask, grid_meta, coh_tag

# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------
LADDER_COH = 0.9                      # coherence of notebook 05's ladder
# the four sets of the ladder (notebook 05 section 2), (inner, outer) in m
FIXED_WIDTH = [(0, 100), (50, 150), (100, 200), (150, 250), (200, 300), (300, 400), (400, 500)]
GROWING = [(100, 200), (100, 300), (100, 400), (100, 500), (100, 700)]
ORIGINAL = [(0, 60), (30, 90), (60, 120), (0, 120), (30, 150), (60, 180), (200, 600), (500, 700)]
INNER_EDGE = [(0, 150), (10, 150), (20, 150), (30, 150), (40, 150), (50, 150), (60, 150), (70, 150), (80, 150)]
SELECTED = (int(round(dist_min * 1000)), int(round(dist_max * 1000)))   # the paper's bin (m)
SELECTED_COHS = [0.7, 0.8, 0.9]       # notebook 05 compares the paper's bin at these thresholds

# the creepmeter stack, the rule of notebook 05 section 3 (stack_insar with its defaults)
STACK_KM = 0.2                        # half-window of fp_ts rows along the trace (km)
STACK_WIDEN_KM = 1.0                  # widened half-window if fewer than 2 rows are usable (km)
STACK_MAX = 5                         # at most this many usable rows, nearest first

DISK_NEED_GB = 1.0                    # estimate: ~20 MB of csv per bin, ~30 bins, two 7 MB masks


def sweep_tag(inner_m, outer_m, coh):
    """Folder name of one bin: ap<inner>-<outer>m_coh<coh>_mp<min_points>, the names notebook 05 reads."""
    return f"ap{inner_m}-{outer_m}m_coh{coh:g}_mp{min_points}"


def sweep_bins():
    """Every (inner m, outer m, coherence) that notebook 05 reads, without repeats, in the ladder order."""
    bins = []
    for inner, outer in FIXED_WIDTH + GROWING + ORIGINAL + INNER_EDGE:
        if (inner, outer, LADDER_COH) not in bins:
            bins.append((inner, outer, LADDER_COH))
    for coh in SELECTED_COHS:
        if (SELECTED[0], SELECTED[1], coh) not in bins:
            bins.append((SELECTED[0], SELECTED[1], coh))
    return bins


def build_mask(dic, coh, mask_dir):
    """Coherence mask of one track by the run04 rule (unbroken record and average temporal coherence >= coh).

    Parameters
    ----------
    dic : dict
        Track entry of `insar_dic_list`.
    coh : float
        Threshold.
    mask_dir : str
        Folder for the new mask file.

    Returns
    -------
    str
        Path of `maskTempCoh_<tag>.h5`. Asserts first that the rule reproduces run04's 0.7 and 0.9 masks.
    """
    mdir = dic["mintpy_dir"]
    coherence = read_coherence(mdir)                                   # avgSpatialCoh.h5, as run04 reads it
    with h5py.File(os.path.join(mdir, "maskRecord.h5"), "r") as hf:
        record = hf["mask"][:].astype(bool)                            # valid at every epoch (run04)
    for c_check in (0.7, 0.9):
        with h5py.File(dic["mask_file"][c_check], "r") as hf:
            run04_mask = hf["mask"][:].astype(bool)
        rebuilt = record & np.isfinite(coherence) & (coherence >= c_check)
        assert np.array_equal(rebuilt, run04_mask), f"{dic['Frame']}: the mask rule does not reproduce run04's {c_check} mask"
    mask = record & np.isfinite(coherence) & (coherence >= coh)
    os.makedirs(mask_dir, exist_ok=True)
    path = os.path.join(mask_dir, f"maskTempCoh_{coh_tag(coh)}.h5")
    write_mask(path, mask, grid_meta(frames[dic["Frame"]]["timeseries"]))
    print(f"  {dic['Frame']}: coherence-{coh} mask (run04 rule, checked at 0.7 and 0.9) keeps {100 * mask.mean():.1f}% "
          f"of pixels -> {path}")
    return path


def stack_rows(fp_table, lat, to_km):
    """Mean of the fp_ts rows nearest a creepmeter along the trace (notebook 05's stack_insar, default arguments).

    Rows with more than 50 % finite epochs within STACK_KM along the trace are used, nearest first, at most
    STACK_MAX; with fewer than 2, the window widens to STACK_WIDEN_KM.

    Parameters
    ----------
    fp_table : pandas.DataFrame
        FP series, latitude x decimal year (mm).
    lat : float
        Latitude of the creepmeter (deg).
    to_km : callable
        latitude -> km from Point Pinole.

    Returns
    -------
    tuple or None
        (dates, mean, n_rows, half-width used in km), or None if no row is usable.
    """
    n_epochs = fp_table.shape[1]
    row_km = to_km(fp_table.index.values)
    centre_km = float(to_km([lat])[0])
    for half_width_km in (STACK_KM, STACK_WIDEN_KM):
        nearby = fp_table[np.abs(row_km - centre_km) <= half_width_km + 1e-6]
        usable = nearby[nearby.notna().sum(axis=1) > 0.5 * n_epochs]
        if len(usable) >= 2:
            break
    if len(usable) == 0:
        return None
    nearest_first = np.argsort(np.abs(to_km(usable.index.values) - centre_km), kind="stable")
    usable = usable.iloc[nearest_first[:STACK_MAX]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stack_mean = np.nanmean(usable.values, axis=0)
    return fp_table.columns.values.astype(float), stack_mean, len(usable), half_width_km


def station_rates(fp_ts, to_km):
    """OLS rates of each creepmeter and of the FP stack at it, over the window of fp_ts (no bootstrap).

    Parameters
    ----------
    fp_ts : pandas.DataFrame
        FP series of one bin.
    to_km : callable
        latitude -> km from Point Pinole.

    Returns
    -------
    pandas.DataFrame
        One row per creepmeter: code, cm_rate, cr_rate, d_cross = cr_rate - cm_rate (mm/yr), n_stack, stack_km.
    """
    t0 = float(fp_ts.columns.min())
    t1 = float(fp_ts.columns.max())
    rows = []
    for code, lat, lon, name in creepmeters:
        cm_dates, cm_slip = utils.daily_creepmeter(common_paths["creepmeter_dir"], code, t0, t1, creepmeter_glitches)
        stacked = stack_rows(fp_ts, lat, to_km)
        if stacked is None:
            rows.append(dict(code=code, cm_rate=np.nan, cr_rate=np.nan, d_cross=np.nan, n_stack=0, stack_km=np.nan))
            continue
        stack_dates, stack_mean, n_stack, half_width_km = stacked
        finite = np.isfinite(stack_mean)
        cm_rate = np.polyfit(cm_dates, cm_slip, 1)[0]
        cr_rate = np.polyfit(stack_dates[finite], stack_mean[finite], 1)[0]
        rows.append(dict(code=code, cm_rate=cm_rate, cr_rate=cr_rate, d_cross=cr_rate - cm_rate, n_stack=n_stack,
                         stack_km=half_width_km))
    return pd.DataFrame(rows)


def run_bin(inner_m, outer_m, coh, track_dics, x_fault, km_of, out_dir):
    """Notebook 03's chain for one bin into out_dir/<tag>/; returns the summary row and the per-station rows."""
    tag = sweep_tag(inner_m, outer_m, coh)
    bin_dir = os.path.join(out_dir, tag)
    os.makedirs(bin_dir, exist_ok=True)
    bin_min = inner_m / 1000.0                   # km
    bin_max = outer_m / 1000.0
    settings = dict(coh=coh, dist_min=bin_min, dist_max=bin_max, x_width=x_width, min_points=min_points,
                    rate_nsigma=rate_nsigma)

    # 1) per track: extraction (03 section 5)
    n_cross = {}
    for dic in track_dics:
        track = dic["ORBIT_DIRECTION"]
        insar = cp.read_insar_data(dic, coh, bin_max, x_width, unit)
        extracted = cp.extract_track(insar, x_fault, bin_min, bin_max, x_width, min_points, rate_nsigma, unit, track)
        insar["ts_h5"].close()
        params_key = dict(settings)
        params_key.update(cp.ts_data_key(dic, coh, common_paths["fault_profiles"]))
        params_key["mask_file"] = dic["mask_file"][coh]
        cp.write_track(extracted, os.path.join(bin_dir, f"creep_ts_LOS_{track}.csv"),
                       os.path.join(bin_dir, f"creep_geom_{track}.csv"),
                       os.path.join(bin_dir, f"creep_params_{track}.json"), None, params_key)
        n_cross[track] = len(extracted["creep_ts"])
        print(f"[{track}] {n_cross[track]}/{len(x_fault)} crossings; rate mask sigma {extracted['sigma']:.3f} mm/yr, "
              f"{100.0 * extracted['n_rejected'] / extracted['n_total']:.1f}% of pixels rejected, "
              f"{extracted['n_dropped']} crossings dropped")

    # 2) common date grid (03 section 8), from the files just written, as 03 reads them
    asc_los = cp.read_creep_ts(os.path.join(bin_dir, "creep_ts_LOS_ascending.csv"))
    desc_los = cp.read_creep_ts(os.path.join(bin_dir, "creep_ts_LOS_descending.csv"))
    grid = cp.common_grid(asc_los, desc_los, max_interp_gap_days)
    lats = grid["lats"]

    # 3) decomposition, rates and the landslide flag (03 sections 9 to 11)
    crossing_inc = {}
    crossing_azi = {}
    for dic, key in zip(track_dics, ("asc", "desc")):
        geom_file = os.path.join(bin_dir, f"creep_geom_{dic['ORBIT_DIRECTION']}.csv")
        crossing_inc[key], crossing_azi[key] = cp.crossing_geometry(geom_file, lats, key)
    strike = cp.crossing_strike(x_fault, lats)
    fp_ts = cp.decompose_fp(grid["los_interp"], crossing_inc, crossing_azi, strike, lats, grid["common_dates"])
    fp_rate_raw = cp.rate_per_crossing(fp_ts)
    landslide_flag, trend_sigma = cp.flag_landslide_crossings(fp_rate_raw, cp.TREND_WINDOW, cp.TREND_NSIGMA)
    fp_ts.iloc[landslide_flag, :] = np.nan
    fp_rate = np.where(landslide_flag, np.nan, fp_rate_raw)
    fp_rate_smooth = utils.gaussian_smooth_km(fp_rate, km_of(lats), rate_smooth_km)
    fp_ts.to_csv(os.path.join(bin_dir, "fp_ts.csv"))
    pd.DataFrame({"lat": lats, "fp_creep_rate_mm_yr": fp_rate, "fp_creep_rate_raw": fp_rate_raw,
                  "landslide_flag": landslide_flag.astype(int), "fp_creep_rate_smooth": fp_rate_smooth,
                  "strike_deg": strike}).to_csv(os.path.join(bin_dir, "creep_rate_fp.csv"), index=False)
    print(f"common grid {len(lats)} crossings x {len(grid['common_dates'])} dates; landslide-flagged "
          f"{int(landslide_flag.sum())} (residual sigma {trend_sigma:.2f} mm/yr)")

    # 4) the creepmeter rates (the scores of the summary table)
    stations = station_rates(cp.read_creep_ts(os.path.join(bin_dir, "fp_ts.csv")), km_of)
    ok = np.isfinite(stations["d_cross"])
    summary = dict(tag=tag, inner_m=inner_m, outer_m=outer_m, coh=coh, min_points=min_points,
                   n_stations=int(ok.sum()),
                   rate_rms=float(np.sqrt(np.mean(stations.loc[ok, "d_cross"] ** 2))),
                   rate_bias=float(stations.loc[ok, "d_cross"].mean()),
                   n_cross_asc=n_cross["ascending"], n_cross_desc=n_cross["descending"],
                   n_crossings=len(lats), n_flagged=int(landslide_flag.sum()))
    print(f"rate RMS vs the creepmeters {summary['rate_rms']:.3f} mm/yr, bias {summary['rate_bias']:+.3f} "
          f"({summary['n_stations']}/{len(creepmeters)} stations)")
    return summary, stations.assign(tag=tag, inner_m=inner_m, outer_m=outer_m, coh=coh)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Aperture sweep read by notebook 05 (notebook 03's chain per bin).")
    parser.add_argument("--out-dir", default=common_paths["sweep_dir"],
                        help="output folder (default: sweep_dir, data_dir/aperture_sweep)")
    parser.add_argument("--tags", nargs="*", default=None, help="run only these bins (folder names)")
    args = parser.parse_args()
    out_dir = args.out_dir

    bins = sweep_bins()
    if args.tags:
        unknown = sorted(set(args.tags) - {sweep_tag(*b) for b in bins})
        assert not unknown, f"not bins of the sweep: {unknown}"
        bins = [b for b in bins if sweep_tag(*b) in args.tags]
    os.makedirs(out_dir, exist_ok=True)
    utils.require_disk_space(out_dir, DISK_NEED_GB, "the aperture sweep")

    git, dirty = utils.git_hash(repo_dir)
    header = [f"run07_aperture_sweep  {time.strftime('%Y-%m-%d %H:%M')}  git {git}{'+' if dirty else ''}",
              f"data_dir {data_dir}", f"out_dir {out_dir}",
              f"x_width {x_width} km, min_points {min_points}, rate_nsigma {rate_nsigma}, az_smooth_n {az_smooth_n}, "
              f"max_interp_gap_days {max_interp_gap_days}, rate_smooth_km {rate_smooth_km}, "
              f"trend_window {cp.TREND_WINDOW}, trend_nsigma {cp.TREND_NSIGMA}",
              f"{len(bins)} bins: " + ", ".join(sweep_tag(*b) for b in bins)]
    print("\n".join(header))

    # the coherence masks of each track: run04's, plus the ones built here
    track_dics = {}
    for coh in sorted({b[2] for b in bins}):
        track_dics[coh] = []
        for dic in insar_dic_list:
            dic_coh = dict(dic, mask_file=dict(dic["mask_file"]))
            if coh not in dic_coh["mask_file"]:
                dic_coh["mask_file"][coh] = build_mask(dic, coh, os.path.join(out_dir, "masks", dic["Frame"]))
            track_dics[coh].append(dic_coh)
        assert [d["ORBIT_DIRECTION"] for d in track_dics[coh]] == ["ascending", "descending"], "track order"

    x_fault, km_of = cp.load_crossings(common_paths["fault_profiles"], az_smooth_n, cp.UTM_EPSG)
    summaries = []
    station_rows = []
    log_lines = list(header) + [""]
    t_start = time.time()
    for inner_m, outer_m, coh in bins:
        t_bin = time.time()
        print(f"\n=== {sweep_tag(inner_m, outer_m, coh)} ===")
        summary, stations = run_bin(inner_m, outer_m, coh, track_dics[coh], x_fault, km_of, out_dir)
        summaries.append(summary)
        station_rows.append(stations)
        log_lines.append(f"{summary['tag']:<24} asc {summary['n_cross_asc']:>4} desc {summary['n_cross_desc']:>4} "
                         f"common {summary['n_crossings']:>4} flagged {summary['n_flagged']:>3}  rate RMS "
                         f"{summary['rate_rms']:.3f} bias {summary['rate_bias']:+.3f} ({summary['n_stations']} st)  "
                         f"{time.time() - t_bin:.0f} s")
        # written after every bin, so an interrupted sweep keeps what it has
        pd.DataFrame(summaries).to_csv(os.path.join(out_dir, f"sweep_summary_mp{min_points}.csv"), index=False)
        pd.concat(station_rows, ignore_index=True).to_csv(os.path.join(out_dir, f"sweep_results_mp{min_points}.csv"),
                                                          index=False)

    # check: the paper's bin at coh_timeseries is notebook 03's computation, so its fp_ts must equal 03's
    nb03_fp = os.path.join(common_paths["out_dir"], "03_creep_timeseries", "fp_ts.csv")
    own_tag = sweep_tag(SELECTED[0], SELECTED[1], coh_timeseries)
    if os.path.exists(nb03_fp) and any(sweep_tag(*b) == own_tag for b in bins):
        mine = cp.read_creep_ts(os.path.join(out_dir, own_tag, "fp_ts.csv"))
        theirs = cp.read_creep_ts(nb03_fp)
        assert mine.index.equals(theirs.index) and mine.columns.equals(theirs.columns), \
            f"{own_tag}: crossings or dates differ from notebook 03's fp_ts.csv"
        assert np.array_equal(mine.to_numpy(), theirs.to_numpy(), equal_nan=True), \
            f"{own_tag}: values differ from notebook 03's fp_ts.csv (max |diff| {np.nanmax(np.abs(mine.values - theirs.values)):.3g} mm)"
        line = f"check: {own_tag}/fp_ts.csv equals {nb03_fp} ({mine.shape[0]} crossings x {mine.shape[1]} dates, identical)"
    else:
        line = f"check against notebook 03 not made ({nb03_fp} missing or {own_tag} not run)"
    print("\n" + line)
    log_lines += ["", line, f"total {time.time() - t_start:.0f} s"]
    with open(os.path.join(out_dir, "run07_aperture_sweep.log"), "w") as fh:
        fh.write("\n".join(log_lines) + "\n")
    print(f"-> {out_dir}")
