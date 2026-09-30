# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
# ---

# %% [markdown]
# # 06 The event field: per-pixel step maps, the mean event and its cross-fault shape
#
# ```
# Created on Mon Sep 21 2026
# @author: daniellelindsay
# Run in opera_disp_feb13. The per-pixel fits take ~3 min per track; the rest is seconds.
# ```
#
# **What this notebook does.** Notebook 04 measured the events on cross-fault
# differences (east minus west medians). Here we map them: every pixel in a box
# around Tule Pond gets its own time-function fit, so we see the SHAPE of each
# event across the fault, not just its size. Notebook 07 inverts the mean event
# for slip at depth; `07c_fig2` draws Fig 2 from both.
#
# 1. **Per-pixel model** on each track: MintPy's `timeseries2velocity` model
#    (intercept + linear rate + annual cycle + one Heaviside step per event, step
#    dates from `creep_events.csv`), solved per pixel with NaN-aware least
#    squares because `timeseries2velocity.py` refuses records with gaps.
# 2. **One data correction**, an explicit literal below: the descending
#    burst-boundary offsets (measured per event and removed). The old chain also
#    left the 2023 event out of the mean; here every event is kept (Dani,
#    2026-09-21: five events unless it is much noisier, and it is not).
# 3. **Mean event** = mean of the five step maps per track; written in the
#    layout notebook 07 (CSI) reads.
# 4. **Decomposition** of the inter-event rate and the mean event into
#    fault-parallel and vertical with MintPy's `asc_desc2horz_vert`, exactly as
#    notebook 02 did for the whole fault, here with the local strike of the box.
# 5. **Profiles** X-X' (the Tule Pond peak) and Y-Y' (CFW1) across the
#    fault-parallel maps (the numbers the text quotes) and across each track's
#    LOS maps: the pixels and the model-free near-fault step (east band minus
#    west band, the aperture of notebook 04). No fit here: the model is
#    notebook 07.
# 6. **Hand-off to Fig 2**: the maps and profiles, one file, drawn by
#    `07c_fig2` after the slip model.
#
# Ported from `Hayward_DISP-S1/modelling/m01_step_maps.py` (steps 1 to 4),
# `hayward_creep/analysis/event_map_profiles.py` (decomposition, profiles) and
# `hayward_creep/plotting/plot_tulepond_secular_profiles.py` (fits, Fig 2).
#
# **Inputs**: `timeseries_tropo.h5`, `maskTempCoh_07.h5`, `geometryGeo.h5` per
# track; `04_event_detection/creep_events.csv` and `section_lengths.csv`;
# `hayward_fault_profiles.csv`.
#
# **Outputs** (`outputs/06_event_field_geometry/`): `tulepond_steps_<track>.h5`
# (every per-event LOS step map, raw and burst-fixed, mean, scatter, rate,
# geometry), `csi_mean_event_<track>.npz` (07's input), `fp_maps.npz`
# (fault-parallel and vertical rate and mean event), `profile_<XX|YY>_<field>[_<track>].csv`,
# `fig2_inputs.npz` (for `07c_fig2`), `06_event_field_stats.txt`,
# registry rows `06_*`.
#
# **Figures**: the step-map grid (supplement).
#
# **What to check.** Near-fault LOS steps of opposite sign on the two tracks
# (ascending negative, descending positive, ratio near the geometric -0.6);
# burst offsets under 1 mm; the mean event's fault-parallel profile an arctangent that decays
# within a few hundred metres at X-X' and nothing at Y-Y'; the vertical
# component near zero.
#
# **Apertures.** Every near-fault step here is median(east band) minus
# median(west band) with the SAME band as notebook 04 (`dist_min` to
# `dist_max`, 30 to 150 m), so the two notebooks measure the same thing. A wider
# band (100 to 500 m) makes the 2023 ascending step vanish because that event
# carries a broad extra signal on the ascending west block within ~1 km; the
# step at the trace is normal.

# %%
import os
import warnings

import insar_utils as utils
from TulePond_filepaths import (common_paths, insar_dic_list, box_lonlat,
                                coh_timeseries, az_smooth_n, dist_min, dist_max, paths_out, repo_dir)

import numpy as np
import pandas as pd
import pyproj
import h5py
import pygmt
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
utils.mpl_paper_style(base=8)          # Helvetica, 8 pt, editable PDF text
from mintpy.utils import time_func                                    # the design matrix timeseries2velocity uses
from mintpy.asc_desc2horz_vert import asc_desc2horz_vert              # THE decomposition, MintPy's own

# %% [markdown]
# ## Constants

# %%
notebook = "06_event_field_geometry"
out_dir = paths_out[notebook]
fig_dir = common_paths["fig_dir"]
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

unit = 1000                     # 1000 = mm
periodic = [1.0]                # annual cycle in the per-pixel model (years)
min_obs = 60                    # a pixel needs this many finite epochs to be fitted
geom_step = 20                  # MintPy's decomposition window (pixels), as notebook 02

# Descending burst boundaries in the box (found on the per-date increment maps,
# 2026-08-19; RESULTS.md section 2): lat = b + s * (lon + 122), s = -0.14 deg/deg,
# perpendicular to the descending track as burst boundaries are. The offset
# across each is measured per event in +/- burst_band_deg bands (excluding
# +/- burst_far_m around the trace) and removed as a step, holding the north
# side (which contains the reference pixel ZOA1) fixed. Ascending has none.
burst_lines = {"descending": [(-0.14, 37.536), (-0.14, 37.502)]}
burst_band_deg = 0.0015
burst_far_m = 400.0

# Events left out of the mean (onset years, matched to 0.02 yr). The old chain
# excluded 2023.08 (RESULTS.md section 2); with the five-event mean no noisier
# than four (scatter 0.89 vs 0.95 mm about the running median at X-X'), all five
# are kept. Set [2023.08] to reproduce the old product; the checks adapt.
exclude_events = []

# profiles across the fault-parallel maps, perpendicular to the local strike
profile_pts = {"X": (-121.97816, 37.56270),      # X-X': the trace point 58.5 km from Point Pinole (Dani, 2026-09-23, from a 200 m sweep of the event profiles: clean in both tracks, inside the 90% band)
               "Y": (-121.95233, 37.53214)}      # Y-Y': CFW1
cross_m = 2000.0                # +/- profile length (2 km: beyond that the east block is noisy)
swath_m = 150.0                 # +/- along-strike half-width
median_bin_m = 150.0            # running-median bin (five pixels), taken on each side of the fault separately
coh_maps = 0.9                  # temporal-coherence mask for the per-pixel maps and the decomposed field the slip model inverts
                                # (Dani 2026-09-28: the 0.7 maps are too noisy); the crossings and the LOS checks stay at coh_timeseries
far_m = (1500.0, 2000.0)        # same-block far field the near-fault amplitude of each block is measured against
median_col, grey, ann_col = "#A50F15", "gray55", "black"       # per-side running median (dark red); pixels; block annotation
rng_blk = np.random.default_rng(20260923)
event_far_m = 1500.0            # the event profile is zeroed on |x| >= this (far field, both blocks)
xfault_step_m = (dist_min * 1e3, dist_max * 1e3)   # the near-fault step band, the aperture of notebook 04 (m)

events_file = os.path.join(paths_out["04_event_detection"], "creep_events.csv")
lengths_file = os.path.join(paths_out["04_event_detection"], "section_lengths.csv")

# saved-products mode (2026-09-28): on a machine without the MintPy stacks,
# the per-track products this notebook wrote (`tulepond_steps_<track>.h5`:
# every step map, the rate, the geometry, the mask) and `burst_offsets.csv`
# stand in for the MintPy files; the time-function fit and the burst fix
# are not redone. (The check that this path reproduces the MintPy-mode
# `fp_maps.npz` exactly is kept in drafts/06/strike143_reference_check.py.)
saved_steps = {o: os.path.join(out_dir, f"tulepond_steps_{o}.h5") for o in ("ascending", "descending")}
from_saved_steps = not all(os.path.exists(dic["ts_file"]) for dic in insar_dic_list)
if from_saved_steps:
    assert all(os.path.exists(f) for f in saved_steps.values()), "no MintPy stacks and no saved step products"
    print("saved-products mode: reading tulepond_steps_<track>.h5 (MintPy stacks not on this machine)")

REG = utils.Registry(notebook, common_paths, repo_dir)
if from_saved_steps:
    REG.inputs(events_file, lengths_file, common_paths["fault_profiles"],
               *saved_steps.values(), os.path.join(out_dir, "burst_offsets.csv"))
else:
    REG.inputs(events_file, lengths_file, common_paths["fault_profiles"],
               *[dic[k] for dic in insar_dic_list for k in ("ts_file", "geo_file")],
               *[dic["mask_file"][coh_timeseries] for dic in insar_dic_list])

# %% [markdown]
# ## LOAD: the events, the trace and the box
#
# The box is `box_lonlat` (W, S, E, N) converted to a row/column slice of the
# product grid. Both tracks are on the SAME grid (asserted), so one slice
# serves both and the two LOS maps line up pixel for pixel.

# %%
ev = pd.read_csv(events_file)
tp = ev[ev.segment == "TULEPOND"].sort_values("onset_yr").reset_index(drop=True)
onsets = tp.onset_yr.to_numpy(float)
print(f"{len(tp)} Tule Pond events: {onsets}")

prof = pd.read_csv(common_paths["fault_profiles"])
prof["az_smooth"] = prof["profile_az"].rolling(az_smooth_n, center=True, min_periods=3).median()
lengths = pd.read_csv(lengths_file).set_index("name")
band = lengths.loc["tulepond_band90"]          # the band Fig 1 stacks (Dani, 2026-09-21: the 90% band)
in_box = (prof.ft_lat >= box_lonlat[1]) & (prof.ft_lat <= box_lonlat[3])
strike_box = float(np.nanmean(prof.loc[in_box, "az_smooth"] + 90.0))  # mean strike of the whole map box (used until 2026-09-28)
# the decomposition strike is the mean local strike of the SSE section (the
# creep-event section from 04, Dani 2026-09-28): the fault-parallel direction
# of the slipping patch, not of the whole 20 km map box
section = lengths.loc["tulepond_creep_length"]
in_section = (prof.ft_lat >= section.lat_lo) & (prof.ft_lat <= section.lat_hi)
strike = float(np.nanmean(prof.loc[in_section, "az_smooth"] + 90.0))
print(f"strike: SSE section (km {section.km_lo:.1f}-{section.km_hi:.1f}, {in_section.sum()} crossings) {strike:.2f} deg; "
      f"whole box {strike_box:.2f} deg (whole-fault mean in notebook 02: ~144)")
assert 130 < strike < 160

grid_files = list(saved_steps.values()) if from_saved_steps else [dic["ts_file"] for dic in insar_dic_list]
attrs = utils.load_h5_attrs(grid_files[0])     # saved products carry the box grid: offsets 0, full width
for f_grid_ in grid_files[1:]:
    a2 = utils.load_h5_attrs(f_grid_)
    for k in ("X_FIRST", "Y_FIRST", "X_STEP", "Y_STEP", "EPSG", "WIDTH", "LENGTH"):
        assert attrs[k] == a2[k], f"tracks are on different grids ({k}): {attrs[k]} vs {a2[k]}"
epsg = int(attrs["EPSG"])
to_utm = pyproj.Transformer.from_crs(4326, epsg, always_xy=True)
to_ll = pyproj.Transformer.from_crs(epsg, 4326, always_xy=True)

W_, S_, E_, N_ = box_lonlat
xs_, ys_ = to_utm.transform([W_, W_, E_, E_], [S_, N_, S_, N_])
x0, dx, y0, dy = (float(attrs[k]) for k in ("X_FIRST", "X_STEP", "Y_FIRST", "Y_STEP"))
c0, c1 = max(int(np.floor((min(xs_) - x0) / dx)), 0), min(int(np.ceil((max(xs_) - x0) / dx)), int(attrs["WIDTH"]))
r0, r1 = max(int(np.floor((max(ys_) - y0) / dy)), 0), min(int(np.ceil((min(ys_) - y0) / dy)), int(attrs["LENGTH"]))
nr, nc = r1 - r0, c1 - c0
print(f"box rows {r0}:{r1} cols {c0}:{c1} = {nr} x {nc} pixels of {abs(dx):.0f} m")

# pixel-centre coordinates of the box
Xg, Yg = np.meshgrid(x0 + (np.arange(c0, c1) + 0.5) * dx, y0 + (np.arange(r0, r1) + 0.5) * dy)
lon_g, lat_g = to_ll.transform(Xg, Yg)
# trace in UTM and the signed cross-fault distance of every pixel (+ = east / north-east side)
fx, fy = to_utm.transform(prof.ft_lon.values, prof.ft_lat.values)
tree = cKDTree(np.c_[fx, fy])
d_trace, j_near = tree.query(np.c_[Xg.ravel(), Yg.ravel()])
nrm = np.array([np.sin(np.deg2rad(strike - 90)), np.cos(np.deg2rad(strike - 90))])   # unit normal: strike - 90 is the NE side (+ east); strike + 90 would be SW
d_signed = ((np.c_[Xg.ravel(), Yg.ravel()] - np.c_[fx, fy][j_near]) @ nrm).reshape(nr, nc)
d_trace = d_trace.reshape(nr, nc)

# %% [markdown]
# ## RUN: the per-pixel time-function model on each track
#
# MintPy's Heaviside is $H(t > t_{step})$, so the step date is set one day
# BEFORE the acquisition nearest the onset: that acquisition is the first
# post-event sample, as in notebook 04. The design matrix comes from
# `mintpy.utils.time_func` (so it IS `timeseries2velocity`'s model); we only
# replace its batch solver with a per-pixel one that skips missing epochs.
# Pixels with complete records are solved in one matrix operation; the rest
# one by one.

# %%
def step_dates_for(dates_yyyymmdd, onsets_yr):
    """(step_date, onset_epoch, onset_yr) per event: the acquisition nearest the onset, step the day before."""
    dts = pd.to_datetime(dates_yyyymmdd, format="%Y%m%d")
    yrs = np.array([utils.date_to_decimal_year(d) for d in dates_yyyymmdd])
    out = []
    for oy in onsets_yr:
        i = int(np.argmin(np.abs(yrs - oy)))
        out.append(((dts[i] - pd.Timedelta(days=1)).strftime("%Y%m%d"), dts[i].strftime("%Y%m%d"), float(oy)))
    return out


def fit_time_func(cube, dates, keep, step_dates):
    """Per-pixel least squares of MintPy's model (polynomial 1 + periodic + steps).
    cube (nt, nr, nc) in mm with NaN gaps; keep (nr, nc) bool. Returns dict of
    coefficient maps (mm, mm/yr), their formal errors and the residual RMS."""
    model = dict(polynomial=1, periodic=periodic, stepDate=list(step_dates), polyline=[], exp={}, log={})
    G = time_func.get_design_matrix4time_func(dates, model).astype(float)
    names = ["intercept", "velocity"] + [f"period{p}{c}" for p in periodic for c in ("Cos", "Sin")] + [f"step{d}" for d in step_dates]
    assert G.shape[1] == len(names)
    nt, nr_, nc_ = cube.shape
    Y = cube.reshape(nt, -1)
    npar = G.shape[1]
    m = np.full((npar, nr_ * nc_), np.nan, np.float32)
    ms = np.full((npar, nr_ * nc_), np.nan, np.float32)
    rms = np.full(nr_ * nc_, np.nan, np.float32)
    idx = np.nonzero(keep.reshape(-1))[0]
    fin = np.isfinite(Y[:, idx])
    full = fin.all(0)
    print(f"  fitting {len(idx)} pixels x {nt} dates, {npar} parameters: {full.sum()} complete records, {(~full).sum()} with gaps")
    if full.any():                                              # complete records: one solve
        c = np.linalg.lstsq(G, Y[:, idx[full]], rcond=None)[0]
        r = Y[:, idx[full]] - G @ c
        s2 = (r ** 2).sum(0) / (nt - npar)
        cov = np.linalg.inv(G.T @ G)
        m[:, idx[full]] = c
        ms[:, idx[full]] = np.sqrt(np.outer(np.diag(cov), s2))
        rms[idx[full]] = np.sqrt((r ** 2).mean(0))
    for j in idx[~full]:                                        # records with gaps: one at a time
        y = Y[:, j]; f = np.isfinite(y)
        if f.sum() < min_obs:
            continue
        Gj = G[f]
        c = np.linalg.lstsq(Gj, y[f], rcond=None)[0]
        r = y[f] - Gj @ c
        s2 = (r ** 2).sum() / max(f.sum() - npar, 1)
        m[:, j] = c
        ms[:, j] = np.sqrt(np.diag(np.linalg.inv(Gj.T @ Gj)) * s2)
        rms[j] = np.sqrt((r ** 2).mean())
    out = {n: m[i].reshape(nr_, nc_) for i, n in enumerate(names)}
    out.update({n + "Std": ms[i].reshape(nr_, nc_) for i, n in enumerate(names)})
    out["residualRMS"] = rms.reshape(nr_, nc_)
    return out, G, names


# %%
fits = {}           # per track: the coefficient maps, the geometry, the step dates
for dic in insar_dic_list:
    orbit = dic["ORBIT_DIRECTION"]
    if from_saved_steps:
        with h5py.File(saved_steps[orbit], "r") as h:
            n_ev = len([k for k in h.keys() if k.startswith("event") and k[5:].isdigit()])
            steps = [(h[f"event{i}"].attrs["step_date"], h[f"event{i}"].attrs["onset_epoch"],
                      float(h[f"event{i}"].attrs["onset_yr"])) for i in range(n_ev)]
            fits[orbit] = dict(coef=None, keep=h["coh_mask"][:].astype(bool), inc=h["incidenceAngle"][:],
                               azi=h["azimuthAngle"][:], steps=steps, dates=None)
        print(f"  {orbit}: steps at {[s0 for s0, _, _ in steps]} (saved products)")
        continue
    f_fit = os.path.join(out_dir, f"timefunc_fit_{orbit}.npz")
    with h5py.File(dic["ts_file"], "r") as h:
        dates = [d.decode() for d in h["date"][:]]
    steps = step_dates_for(dates, onsets)
    with h5py.File(dic["mask_file"][coh_timeseries], "r") as h:
        keep = h[[k for k in h.keys() if k != "date"][0]][r0:r1, c0:c1].astype(bool)
    with h5py.File(dic["geo_file"], "r") as h:                  # loaded as notebook 02 does (float32, as MintPy wrote it)
        inc = h["incidenceAngle"][r0:r1, c0:c1].astype(np.float32)
        azi = h["azimuthAngle"][r0:r1, c0:c1].astype(np.float32)
    # The fit is the slow step, so it is cached -- but the cache is only reusable
    # if it was made from the SAME step dates AND the same input record. The step
    # dates sit in 2017-2024, so a data refresh that appends epochs leaves them
    # unchanged; keying on them alone reused a stale fit (2026-09-28). A cache
    # that does not match is recomputed, not an error.
    cache_ok = False
    if os.path.exists(f_fit):
        z = np.load(f_fit)
        same_steps = list(z["step_dates"]) == [s0 for s0, _, _ in steps]
        same_input = "input_dates" in z.files and list(z["input_dates"]) == dates
        cache_ok = same_steps and same_input
        if not cache_ok:
            why = "step dates differ" if not same_steps else (
                f"made from {len(z['input_dates']) if 'input_dates' in z.files else 'unknown'} "
                f"input dates, now {len(dates)}")
            print(f"[{orbit}] cached fit unusable ({why}); refitting")
    if cache_ok:
        coef = {k: z[k] for k in z.files if k not in ("step_dates", "names", "input_dates")}
        print(f"[{orbit}] per-pixel fit loaded from {os.path.basename(f_fit)}")
    else:
        with h5py.File(dic["ts_file"], "r") as h:
            cube = h["timeseries"][:, r0:r1, c0:c1].astype(np.float32) * unit     # m -> mm
        cube[cube == 0] = np.nan                                # MintPy no-data
        print(f"[{orbit}] {len(dates)} dates, coherent pixels {keep.mean():.0%} of the box")
        coef, G, names = fit_time_func(cube, dates, keep, [s0 for s0, _, _ in steps])
        np.savez(f_fit, step_dates=np.array([s0 for s0, _, _ in steps]), names=np.array(names),
                 input_dates=np.array(dates), **coef)
        del cube
    fits[orbit] = dict(coef=coef, keep=keep, inc=inc, azi=azi, steps=steps, dates=dates)
    print(f"  {orbit}: steps at {[s0 for s0, _, _ in steps]} (onset epochs {[o for _, o, _ in steps]})")

# %% [markdown]
# ## CHECK 1: MintPy's design matrix is the one we think it is
#
# Build the same matrix by hand (1, t, cos 2πt, sin 2πt, H(t > t_step)) on
# MintPy's own time axis and fit a synthetic series with a known 10 mm step
# through both. The step estimates must agree to 1e-6 mm (measured 1.5e-8 mm on
# the March-2026 record; it was under 1e-9 on the shorter one): the two matrices
# span the same model. MintPy's decimal year (`ptime.yyyymmdd2years`) differs
# from our exact-seconds one by up to a few hours; the effect on a step is
# printed and is far below the noise.

# %%
from mintpy.utils import ptime

# saved-products mode: the MintPy date axis is not on this machine; the
# check tests MintPy's design-matrix construction (not our data) and passed
# when the products were written
if from_saved_steps:
    print("CHECK 1 skipped in saved-products mode (needs the MintPy date axis)")
else:

    dates = fits["descending"]["dates"]
    step_dates = [s0 for s0, _, _ in fits["descending"]["steps"]]
    model = dict(polynomial=1, periodic=periodic, stepDate=step_dates, polyline=[], exp={}, log={})
    G_mp = time_func.get_design_matrix4time_func(dates, model).astype(float)
    t = np.array(ptime.yyyymmdd2years(dates), float)
    t_step = np.array(ptime.yyyymmdd2years(step_dates), float)
    # time from the first epoch, as MintPy does (absolute years make the matrix
    # ill-conditioned, cond ~1e7 against 62)
    G_hand = np.column_stack([np.ones_like(t), t - t[0], np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)] + [(t > ts).astype(float) for ts in t_step])
    rng = np.random.default_rng(0)
    y = 2.0 * (t - t[0]) + 1.5 * np.sin(2 * np.pi * t) + 10.0 * (t > t_step[1]) + rng.normal(0, 0.5, t.size)
    c_mp = np.linalg.lstsq(G_mp, y, rcond=None)[0]
    c_hand = np.linalg.lstsq(G_hand, y, rcond=None)[0]
    print("step coefficients, MintPy design:", c_mp[4:].round(6))
    print("step coefficients, hand design:  ", c_hand[4:].round(6))
    assert np.allclose(c_mp[4:], c_hand[4:], rtol=0, atol=1e-6) and abs(c_mp[1] - c_hand[1]) < 1e-6
    print("PASS: MintPy's time-function design is intercept + rate + annual + Heaviside(t > step date)")
    # the year convention: ours (exact seconds) against MintPy's
    t_ours = np.array([utils.date_to_decimal_year(d) for d in dates])
    G_ours = np.column_stack([np.ones_like(t), t_ours - t_ours[0], np.cos(2 * np.pi * t_ours), np.sin(2 * np.pi * t_ours)] + [(t > ts).astype(float) for ts in t_step])
    c_ours = np.linalg.lstsq(G_ours, y, rcond=None)[0]
    print(f"note: MintPy's decimal years differ from ours by up to {24 * 365.25 * np.abs(t - t_ours).max():.1f} h; "
          f"the steps move by at most {np.abs(c_ours[4:] - c_mp[4:]).max():.3f} mm (the fit uses MintPy's)")

# %% [markdown]
# ## RUN: read the step maps, mask, remove the box median, fix the descending bursts
#
# The box median of each step map is removed (the InSAR datum is ZOA1, an
# arbitrary constant here). The burst correction measures, per event, the
# median offset between two thin bands either side of each burst line, away
# from the fault, and shifts the south side onto the north side.

# %%
event_maps = {}
burst_log = []
for dic in insar_dic_list:
    orbit = dic["ORBIT_DIRECTION"]
    F = fits[orbit]
    if from_saved_steps:
        with h5py.File(saved_steps[orbit], "r") as h:
            n_ev = len(F["steps"])
            event_maps[orbit] = dict(ev=np.stack([h[f"event{i}"][:] for i in range(n_ev)]),
                                     ev_raw=np.stack([h[f"event{i}_raw"][:] for i in range(n_ev)]),
                                     ev_std=np.stack([h[f"event{i}Std"][:] for i in range(n_ev)]),
                                     rate=h["rate"][:])
        continue
    keys = [f"step{s0}" for s0, _, _ in F["steps"]]
    evm = np.stack([F["coef"][k] for k in keys]).astype(float)              # (n_ev, nr, nc), mm
    evs = np.stack([F["coef"][k + "Std"] for k in keys]).astype(float)
    rate = F["coef"]["velocity"].astype(float)                              # mm/yr
    evm[:, ~F["keep"]] = np.nan; rate[~F["keep"]] = np.nan
    evm -= np.nanmedian(evm, axis=(1, 2))[:, None, None]                    # box median per event
    evm_raw = evm.copy()
    if orbit in burst_lines:
        far = d_trace > burst_far_m
        for (sl, b) in burst_lines[orbit]:
            dl = lat_g - sl * (lon_g + 122.0)
            nb = (dl > b) & (dl < b + burst_band_deg) & far
            sb = (dl < b) & (dl > b - burst_band_deg) & far
            south = dl < b
            for i, (_, _, oy) in enumerate(F["steps"]):
                off = np.nanmedian(evm[i][nb]) - np.nanmedian(evm[i][sb])
                evm[i][south] += off
                burst_log.append(dict(track=orbit, line_lat=b, onset_yr=oy, offset_mm=off, n_north=int(nb.sum()), n_south=int(sb.sum())))
                print(f"  [{orbit}] event {oy:.2f}: offset across the lat {b:.3f} burst line {off:+.2f} mm removed")
    event_maps[orbit] = dict(ev=evm, ev_raw=evm_raw, ev_std=evs, rate=rate)
if from_saved_steps:
    burst_df = pd.read_csv(os.path.join(out_dir, "burst_offsets.csv"))
else:
    burst_df = pd.DataFrame(burst_log)
    burst_df.to_csv(os.path.join(out_dir, "burst_offsets.csv"), index=False)
assert (burst_df.offset_mm.abs() < 2.0).all(), "a burst offset above 2 mm is not the +/-1 mm scatter described; look at the maps"

# %% [markdown]
# ## CHECK 2: the near-fault LOS step of every event, both tracks
#
# Model-free: median of the east band minus median of the west band
# (30 to 150 m from the trace, notebook 04's aperture) within the stacked section. Fault-parallel
# creep projects with OPPOSITE signs into the two geometries (ascending looks
# west, descending east), so the ascending step must be negative, the descending
# positive, and their ratio near the geometric value from the two LOS vectors.

# %%
in_band = (lat_g >= band.lat_lo) & (lat_g <= band.lat_hi)
east = in_band & (d_signed > xfault_step_m[0]) & (d_signed < xfault_step_m[1])
west = in_band & (d_signed < -xfault_step_m[0]) & (d_signed > -xfault_step_m[1])
L = {o: utils.los_unit_vector(np.nanmedian(fits[o]["inc"][in_band]), np.nanmedian(fits[o]["azi"][in_band])) for o in event_maps}
s_hat = utils.horizontal_unit_vector(strike)
geom_ratio = float((L["ascending"] @ s_hat) / (L["descending"] @ s_hat))
print(f"LOS . strike: asc {L['ascending'] @ s_hat:+.3f}, desc {L['descending'] @ s_hat:+.3f}; geometric asc/desc ratio {geom_ratio:+.2f}")

step_rows = []
for i, (_, _, oy) in enumerate(fits["ascending"]["steps"]):
    row = dict(onset_yr=oy)
    for o in event_maps:
        m_ = event_maps[o]["ev"][i]
        row[f"{o}_step_mm"] = float(np.nanmedian(m_[east]) - np.nanmedian(m_[west]))
    row["ratio_asc_desc"] = row["ascending_step_mm"] / row["descending_step_mm"]
    step_rows.append(row)
steps_df = pd.DataFrame(step_rows)
print(steps_df.round(2).to_string(index=False))
kept = ~steps_df.onset_yr.apply(lambda oy: any(abs(oy - x) < 0.02 for x in exclude_events))
m_asc, m_desc = steps_df.ascending_step_mm[kept].mean(), steps_df.descending_step_mm[kept].mean()
print(f"mean of the {kept.sum()} kept events: asc {m_asc:+.2f} mm, desc {m_desc:+.2f} mm, ratio {m_asc / m_desc:+.2f} (geometric {geom_ratio:+.2f})")
assert m_asc < 0 < m_desc, "the mean event has the wrong LOS sign: the geometry or the fault side is wrong"
assert (steps_df.descending_step_mm > 0).all(), "a descending step with the wrong sign"
off = steps_df[steps_df.ascending_step_mm >= 0]
for _, r_ in off.iterrows():
    print(f"NOTE {r_.onset_yr:.2f}: ascending step {r_.ascending_step_mm:+.2f} mm, ratio {r_.ratio_asc_desc:+.2f}: its ascending step is cancelled by an east-block signal")
print("PASS: the mean event steps negative on ascending and positive on descending; every descending step is positive")

# %% [markdown]
# ## RUN: the mean event per track, written for the slip model
#
# Mean of the kept events per pixel (only where all of them are valid), the
# per-pixel scatter across events, and the formal error of the mean. Written
# to `tulepond_steps_<track>.h5` (everything) and `csi_mean_event_<track>.npz`
# (the flat table notebook 07 reads: lon, lat, LOS mm, error, incidence, azimuth).

# %%
mean_event = {}
for dic in insar_dic_list:
    orbit = dic["ORBIT_DIRECTION"]
    F, M = fits[orbit], event_maps[orbit]
    if from_saved_steps:
        with h5py.File(saved_steps[orbit], "r") as h:
            mean_event[orbit] = dict(mean=h["mean_event"][:], scatter=h["event_scatter"][:],
                                     err=h["mean_event_err"][:], n=int(h.attrs["n_events"]))
        print(f"[{orbit}] mean event read from {os.path.basename(saved_steps[orbit])} (not rewritten)")
        continue
    keep_ev = np.array([not any(abs(oy - x) < 0.02 for x in exclude_events) for _, _, oy in F["steps"]])
    ev, ev_std = M["ev"][keep_ev], M["ev_std"][keep_ev]
    n_ok = np.isfinite(ev).sum(0)
    mean = np.where(n_ok == keep_ev.sum(), np.nanmean(ev, 0), np.nan)
    scatter = np.where(n_ok == keep_ev.sum(), np.nanstd(ev, 0, ddof=1), np.nan)
    err_mean = np.sqrt(np.nanmean(ev_std ** 2, 0)) / np.sqrt(keep_ev.sum())
    good = np.isfinite(mean)
    mean_event[orbit] = dict(mean=mean, scatter=scatter, err=err_mean, n=int(keep_ev.sum()))
    f_h5 = os.path.join(out_dir, f"tulepond_steps_{orbit}.h5")
    with h5py.File(f_h5, "w") as h:
        for i, (sd, od, oy) in enumerate(F["steps"]):
            g = h.create_dataset(f"event{i}", data=M["ev"][i].astype(np.float32))
            g.attrs.update(dict(step_date=sd, onset_epoch=od, onset_yr=oy, in_mean=bool(keep_ev[i])))
            h.create_dataset(f"event{i}Std", data=M["ev_std"][i].astype(np.float32))
            h.create_dataset(f"event{i}_raw", data=M["ev_raw"][i].astype(np.float32))
        for k2, v in dict(mean_event=mean, event_scatter=scatter, mean_event_err=err_mean, rate=M["rate"],
                          incidenceAngle=F["inc"], azimuthAngle=F["azi"], lon=lon_g, lat=lat_g, coh_mask=F["keep"]).items():
            h.create_dataset(k2, data=np.asarray(v, dtype=np.float32))
        h.attrs.update({k: str(v) for k, v in attrs.items()})
        h.attrs.update(dict(X_FIRST=str(x0 + c0 * dx), Y_FIRST=str(y0 + r0 * dy), WIDTH=str(nc), LENGTH=str(nr), UNIT="mm",
                            n_events=int(keep_ev.sum()), excluded=str(exclude_events), coherence=str(coh_timeseries),
                            note="LOS + toward satellite; box median removed per event; azimuthAngle = LOS ground->sat, anticlockwise from N",
                            provenance=utils.provenance_stamp(notebook, repo_dir)))
    np.savez(os.path.join(out_dir, f"csi_mean_event_{orbit}.npz"), lon=lon_g[good], lat=lat_g[good], vel_mm=mean[good],
             err_mm=err_mean[good], incidence=F["inc"][good], azimuth=F["azi"][good], scatter_mm=scatter[good])
    print(f"[{orbit}] mean of {keep_ev.sum()} events: {good.sum()} pixels, 2-98% {np.nanpercentile(mean, 2):+.1f} to {np.nanpercentile(mean, 98):+.1f} mm, "
          f"median scatter {np.nanmedian(scatter):.2f} mm; wrote {os.path.basename(f_h5)} and csi_mean_event_{orbit}.npz")

# %% [markdown]
# ## RUN: fault-parallel and vertical maps
#
# `asc_desc2horz_vert` (MintPy) on the inter-event rate and on the mean event,
# with each track's own incidence and azimuth and `horz_az_angle = -strike`
# (MintPy counts anticlockwise), as notebook 02. The vertical component of
# creep is ~0, so the Up maps are the QC: they should be flat across the
# trace.

# %%
def decompose(v_asc, v_des, strike_deg=None):
    strike_deg = strike if strike_deg is None else strike_deg
    both_ = np.isfinite(v_asc) & np.isfinite(v_des)
    dlos = np.stack([v_asc, v_des]).astype(np.float32); dlos[:, ~both_] = np.nan
    los_inc = np.stack([fits["ascending"]["inc"], fits["descending"]["inc"]]).astype(np.float32)
    los_azi = np.stack([fits["ascending"]["azi"], fits["descending"]["azi"]]).astype(np.float32)
    fp_, up_ = asc_desc2horz_vert(dlos, los_inc, los_azi, horz_az_angle=-strike_deg, step=geom_step)
    return fp_.astype(float), up_.astype(float)


# the per-pixel maps from here on (decomposition, profiles, Fig 2) use the
# stricter coh_maps mask; it lies inside the coh_timeseries mask the step
# maps were fitted on, so it only removes pixels
if coh_maps != coh_timeseries:
    for dic in insar_dic_list:
        orbit = dic["ORBIT_DIRECTION"]
        with h5py.File(dic["mask_file"][coh_maps], "r") as h:
            mx0, my0 = float(h.attrs["X_FIRST"]), float(h.attrs["Y_FIRST"])
            col0 = int(round((x0 + c0 * dx - mx0) / dx)); row0 = int(round((y0 + r0 * dy - my0) / dy))
            mask_maps = h[[k for k in h.keys() if k != "date"][0]][row0:row0 + nr, col0:col0 + nc].astype(bool)
        assert not (mask_maps & ~fits[orbit]["keep"]).any(), f"{orbit}: coh_maps mask not inside the fit mask"
        for key_ in ("mean", "scatter", "err"):
            mean_event[orbit][key_] = np.where(mask_maps, mean_event[orbit][key_], np.nan)
        event_maps[orbit]["rate"] = np.where(mask_maps, event_maps[orbit]["rate"], np.nan)
        event_maps[orbit]["ev"] = np.where(mask_maps[None], event_maps[orbit]["ev"], np.nan)
        print(f"[{orbit}] maps masked at temporal coherence {coh_maps}: {mask_maps.mean():.0%} of the box "
              f"(fit mask {coh_timeseries}: {fits[orbit]['keep'].mean():.0%})")

fp_rate, up_rate = decompose(event_maps["ascending"]["rate"], event_maps["descending"]["rate"])
fp_evt, up_evt = decompose(mean_event["ascending"]["mean"], mean_event["descending"]["mean"])
fp_events = {}
for i, (_, _, oy) in enumerate(fits["ascending"]["steps"]):
    fp_events[oy], _ = decompose(event_maps["ascending"]["ev"][i], event_maps["descending"]["ev"][i])

def block_step(m_):
    return float(np.nanmedian(m_[east]) - np.nanmedian(m_[west]))

print(f"inter-event FP rate step east - west (100-500 m, stacked band): {block_step(fp_rate):+.2f} mm/yr; Up {block_step(up_rate):+.2f} mm/yr")
print(f"mean event FP step: {block_step(fp_evt):+.2f} mm; Up {block_step(up_evt):+.2f} mm (should be ~0: creep has no vertical part)")
for oy, m_ in fp_events.items():
    print(f"  event {oy:.2f}: FP step {block_step(m_):+.2f} mm")
np.savez(os.path.join(out_dir, "fp_maps.npz"), fp_rate=fp_rate, up_rate=up_rate, fp_event=fp_evt, up_event=up_evt,
         lon=lon_g, lat=lat_g, X=Xg, Y=Yg, strike=strike, n_events=mean_event["ascending"]["n"],
         **{f"fp_event_{oy:.2f}": m_ for oy, m_ in fp_events.items()})

# %% [markdown]
# ## RUN: the cross-fault profiles X-X' and Y-Y'
#
# Perpendicular to the LOCAL strike at each point (the smoothed strike of the
# mapped trace there, not the box mean), +/- `cross_m`, all pixels within
# +/- `swath_m` along strike, plus a per-side running median.
# The inter-event profile is zeroed at the fault (mid-point of the two block
# medians within 1 km); the event profile on its far field. The same machinery
# runs on the fault-parallel maps (the numbers the text quotes) and on each
# track's LOS maps (what Fig 2 shows).

# %%
def local_strike(lonlat):
    """Smoothed local strike of the mapped trace (compass bearing) at the trace point nearest lonlat."""
    cx, cy = (float(t_) for t_ in to_utm.transform(*lonlat))
    j = int(tree.query([cx, cy])[1])
    return float(prof.az_smooth.iloc[j] + 90.0)


def profile(m_, lonlat, strike_deg=None):
    """Cross-fault profile of map m_ through lonlat, perpendicular to the LOCAL strike
    there (or to strike_deg): d (m, + east), v, the line's end points (UTM), the strike used."""
    cx, cy = (float(t_) for t_ in to_utm.transform(*lonlat))
    if strike_deg is None:
        strike_deg = local_strike(lonlat)
    pe, pn = np.sin(np.deg2rad(strike_deg - 90.0)), np.cos(np.deg2rad(strike_deg - 90.0))   # cross-fault unit vector (+ east)
    se, sn = -pn, pe                                            # along-strike
    dc = (Xg - cx) * pe + (Yg - cy) * pn; da = (Xg - cx) * se + (Yg - cy) * sn
    sel = (np.abs(da) <= swath_m) & (np.abs(dc) <= cross_m) & np.isfinite(m_)
    d, v = dc[sel], m_[sel]
    o = np.argsort(d)
    ends = (cx - cross_m * pe, cy - cross_m * pn, cx + cross_m * pe, cy + cross_m * pn)
    return d[o], v[o], ends, strike_deg


def running_median(d, v, bin_m=median_bin_m):
    """Binned median on each side of the fault separately: the bin edges start at
    the trace and run outward, so no bin (and no drawn segment) straddles the
    fault. Returns a list of (x, median) arrays, one per side (west, east)."""
    out = []
    for sgn in (-1, 1):
        edges = np.sort(sgn * np.arange(0.0, cross_m + bin_m, bin_m))
        idx = np.digitize(d, edges)
        xc, med = [], []
        for i in range(1, len(edges)):
            m_ = (idx == i) & (sgn * d > 0)
            if m_.sum() >= 3:
                xc.append(0.5 * (edges[i - 1] + edges[i])); med.append(np.median(v[m_]))
        out.append((np.array(xc), np.array(med)))
    return out


def block_amplitudes(d, v, n_boot=2000):
    """Per-block near-fault amplitude of a profile: median of the near band
    (`xfault_step_m`, 30 to 150 m) minus the median of the SAME block's far field
    (`far_m`), with a bootstrap standard error over the pixels. Reads the peak
    value of each block off the profile without any model."""
    out = {}
    for blk, sgn in (("W", -1), ("E", 1)):
        near = (sgn * d >= xfault_step_m[0]) & (sgn * d < xfault_step_m[1])
        far = (sgn * d >= far_m[0]) & (sgn * d < far_m[1])
        assert near.sum() >= 10 and far.sum() >= 50, (blk, near.sum(), far.sum())
        vn, vf = v[near], v[far]
        boot = np.array([np.median(rng_blk.choice(vn, vn.size)) - np.median(rng_blk.choice(vf, vf.size)) for _ in range(n_boot)])
        out[blk] = dict(amp=float(np.median(vn) - np.median(vf)), se=float(boot.std()), far=float(np.median(vf)),
                        n_near=int(near.sum()), n_far=int(far.sum()))
    return out


def zeroed_profiles(m_rate, m_evt, lonlat):
    """The rate and event profiles through lonlat, zeroed (rate: mid-point of the
    block medians within 1 km; event: its far field), and their near-fault steps
    (east minus west medians, the 30 to 150 m aperture of notebook 04)."""
    d, v, ends, st = profile(m_rate, lonlat)
    near = np.abs(d) <= 1000
    v = v - 0.5 * (np.median(v[near & (d < 0)]) + np.median(v[near & (d > 0)]))
    de, ve, _, _ = profile(m_evt, lonlat, st)
    far = np.abs(de) >= event_far_m
    ve = ve - (np.median(ve[far]) if far.sum() >= 20 else np.median(ve))
    lo_, hi_ = xfault_step_m
    r_step = float(np.median(v[(d > lo_) & (d < hi_)]) - np.median(v[(d < -lo_) & (d > -hi_)]))
    e_step = float(np.median(ve[(de > lo_) & (de < hi_)]) - np.median(ve[(de < -lo_) & (de > -hi_)]))
    return dict(d=d, v=v, de=de, ve=ve, ends=ends, lonlat=lonlat, strike=st, rate_step=r_step, event_step=e_step)


# fault-parallel: the numbers the text quotes
profiles = {}
for name, lonlat in profile_pts.items():
    p = zeroed_profiles(fp_rate, fp_evt, lonlat)
    profiles[name] = p
    pd.DataFrame(dict(d_m=p["d"], fp_rate_mm_yr=p["v"])).to_csv(os.path.join(out_dir, f"profile_{name}{name}_rate.csv"), index=False)
    pd.DataFrame(dict(d_m=p["de"], fp_event_mm=p["ve"])).to_csv(os.path.join(out_dir, f"profile_{name}{name}_event.csv"), index=False)
    print(f"{name}-{name}' FP (local strike {p['strike']:.1f} deg): {p['d'].size} rate pixels, {p['de'].size} event pixels; near-fault steps ({xfault_step_m[0]:.0f}-{xfault_step_m[1]:.0f} m): "
          f"rate {p['rate_step']:+.2f} mm/yr, event {p['event_step']:+.2f} mm")

# line of sight, per track: what Fig 2 shows (LOS positive toward the satellite, so
# right-lateral slip steps NEGATIVE on ascending and POSITIVE on descending)
tracks = ["ascending", "descending"]
los_profiles = {o: {n_: zeroed_profiles(event_maps[o]["rate"], mean_event[o]["mean"], ll) for n_, ll in profile_pts.items()} for o in tracks}
for o in tracks:
    for n_, p in los_profiles[o].items():
        pd.DataFrame(dict(d_m=p["d"], los_rate_mm_yr=p["v"])).to_csv(os.path.join(out_dir, f"profile_{n_}{n_}_rate_{o}.csv"), index=False)
        pd.DataFrame(dict(d_m=p["de"], los_event_mm=p["ve"])).to_csv(os.path.join(out_dir, f"profile_{n_}{n_}_event_{o}.csv"), index=False)
        p["blocks"] = block_amplitudes(p["de"], p["ve"])
        blk = p["blocks"]
        print(f"[{o[:3]}] {n_}-{n_}' LOS: rate step {p['rate_step']:+.2f} mm/yr, event step {p['event_step']:+.2f} mm; "
              f"event per block W {blk['W']['amp']:+.2f} +/- {blk['W']['se']:.2f}, E {blk['E']['amp']:+.2f} +/- {blk['E']['se']:.2f} mm")
xx_ratio = los_profiles["ascending"]["X"]["event_step"] / los_profiles["descending"]["X"]["event_step"]
print(f"X-X' event asc/desc LOS step ratio {xx_ratio:+.2f} (geometric, pure strike-slip: {geom_ratio:+.2f})")
assert los_profiles["ascending"]["X"]["event_step"] < 0 < los_profiles["descending"]["X"]["event_step"], "the X-X' LOS event steps have the wrong signs"

# %% [markdown]
# ## RUN: write the maps and profiles Fig 2 draws
#
# Fig 2 is drawn in its own notebook (`07c_fig2`) after the slip model,
# so its model row is the real model. It needs the per-track LOS maps (masked
# at `coh_maps`), the fault-parallel maps and the profiles; they go into one
# file. The fault-parallel and vertical profiles get their block amplitudes
# here too (after the LOS ones, so the LOS bootstrap draws are unchanged).

# %%
names = list(profile_pts)
for n_ in names:
    profiles[n_]["blocks"] = block_amplitudes(profiles[n_]["de"], profiles[n_]["ve"])
# the vertical field through the same lines (Fig 2's second column)
up_profiles = {n_: zeroed_profiles(up_rate, up_evt, ll) for n_, ll in profile_pts.items()}
for n_ in names:
    up_profiles[n_]["blocks"] = block_amplitudes(up_profiles[n_]["de"], up_profiles[n_]["ve"])

fig2_in = dict(lon=lon_g, lat=lat_g, X=Xg, Y=Yg, epsg=epsg, strike=strike, coh_maps=coh_maps,
               n_events=mean_event["ascending"]["n"], cross_m=cross_m, names=np.array(names),
               fp_rate=fp_rate, fp_event=fp_evt, up_rate=up_rate, up_event=up_evt)
for o in tracks:
    fig2_in[f"rate_{o}"] = event_maps[o]["rate"]
    fig2_in[f"event_{o}"] = mean_event[o]["mean"]
for field, pset in [("fp", profiles), ("up", up_profiles)] + [(o, los_profiles[o]) for o in tracks]:
    for n_ in names:
        p = pset[n_]
        for k in ("d", "v", "de", "ve", "ends"):
            fig2_in[f"prof_{field}_{n_}_{k}"] = np.asarray(p[k])
        for blk in ("W", "E"):
            fig2_in[f"prof_{field}_{n_}_{blk}_amp"] = p["blocks"][blk]["amp"]
            fig2_in[f"prof_{field}_{n_}_{blk}_se"] = p["blocks"][blk]["se"]
f_fig2_in = os.path.join(out_dir, "fig2_inputs.npz")
np.savez(f_fig2_in, provenance=utils.provenance_stamp(notebook, repo_dir), **fig2_in)
print("wrote", f_fig2_in)

# %% [markdown]
# ## Supplement: the step maps, every event and track
#
# Rows ascending / descending; columns the five events (burst-fixed), the mean
# of the kept events, and the per-pixel scatter across them. LOS mm, positive
# toward the satellite, so the two rows have opposite colours across the fault.

# %%
clim = 4.0
scatter_lim = 2.0          # the scatter is ~0.5-1.5 mm near the fault; 2 mm keeps it readable
n_ev = len(onsets)
n_col = n_ev + 2
# pygmt, north up in lon/lat (Dani 2026-09-29: pygmt for anything with a map).
# 18.3 cm wide: seven maps with small gaps, lon annotated on the bottom row and
# lat on the left column only; two colour bars of the same height and font
# under the bottom row (step bar under the event + mean columns, scatter bar
# under the scatter column).
region_box = [box_lonlat[0], box_lonlat[2], box_lonlat[1], box_lonlat[3]]
x_corner, y_corner = x0 + c0 * dx, y0 + r0 * dy                 # corner of the box's first pixel
Wm, gap_x, gap_y = 2.3, 0.15, 0.55
w_, h_ = utils.gmt_map_size(region_box, f"M{Wm}c")
Hm = h_
proj_box = f"M{Wm}c"
cpt_dir = os.path.join(out_dir, "cpt"); os.makedirs(cpt_dir, exist_ok=True)
cpt_step = utils.write_cpt(os.path.join(cpt_dir, "step_maps.cpt"), "RdBu_r", -clim, clim)
cpt_scatter = utils.write_cpt(os.path.join(cpt_dir, "step_scatter.cpt"), "magma", 0, scatter_lim)   # magma, not viridis: no blue/green
pygmt.config(FONT="7p,Helvetica", FONT_ANNOT_PRIMARY="6p,Helvetica", FONT_LABEL="6p,Helvetica", MAP_FRAME_TYPE="plain",
             MAP_FRAME_PEN="0.5p", MAP_TICK_LENGTH_PRIMARY="2p", MAP_ANNOT_OFFSET="2p", MAP_LABEL_OFFSET="2p", FORMAT_GEO_MAP="D")
fig = pygmt.Figure()
letters = iter("ABCDEFGHIJKLMNOP")
for r_, orbit in enumerate(event_maps):
    panels = [(event_maps[orbit]["ev"][i], f"{oy:.2f}" + (" (excluded)" if any(abs(oy - x) < 0.02 for x in exclude_events) else ""), cpt_step)
              for i, (_, _, oy) in enumerate(fits[orbit]["steps"])]
    panels += [(mean_event[orbit]["mean"], f"Mean of {mean_event[orbit]['n']}", cpt_step),
               (mean_event[orbit]["scatter"], "Scatter", cpt_scatter)]
    if r_:
        fig.shift_origin(xshift=f"-{(n_col - 1) * (Wm + gap_x)}c", yshift=f"-{Hm + gap_y}c")
    for c_, (m_, ttl, cpt) in enumerate(panels):
        if c_:
            fig.shift_origin(xshift=f"{Wm + gap_x}c")
        grid = utils.utm_to_lonlat_grid(m_, x_corner, y_corner, dx, dy, epsg, region_box, 0.00015, 0.00012)   # ~13 m, half the 30 m pixel, so no pixel is dropped or doubled
        sides = ("W" if c_ == 0 else "w") + ("S" if r_ == 1 else "s") + "nr"
        fig.basemap(region=region_box, projection=proj_box, frame=[sides, "xa0.15f0.05", "ya0.05f0.025"])
        fig.grdimage(grid=grid, cmap=cpt, nan_transparent=True)
        fig.plot(x=prof.ft_lon.values, y=prof.ft_lat.values, pen="0.5p," + ("black" if cpt == cpt_step else "white"))   # white on the dark scatter map
        fig.basemap(region=region_box, projection=proj_box, frame=[sides, "xa0.15f0.05", "ya0.05f0.025"])
        fig.text(position="TL", text=f"@%Helvetica-Bold%{next(letters)}@%%  {orbit[:3].capitalize()} {ttl}", justify="BL",
                 offset="0c/0.12c", font="6.5p,Helvetica", no_clip=True)
# colour bars, 0.25 cm tall, 0.7 cm under the bottom row
fig.shift_origin(xshift=f"-{(n_col - 1) * (Wm + gap_x)}c")
span_step = n_ev * (Wm + gap_x) + Wm                 # the event + mean columns
w_step = 2 * Wm + gap_x                              # bar two maps wide (Dani 2026-09-29), centred under those columns
cb_pt = 6.0 / min(1.0, np.sqrt(w_step / 9.0))        # GMT shrinks a short bar's fonts by ~sqrt(length / 9 cm): this prints ~6p
with pygmt.config(FONT_ANNOT_PRIMARY=f"{cb_pt:.1f}p,Helvetica", FONT_LABEL=f"{cb_pt:.1f}p,Helvetica"):
    fig.colorbar(cmap=cpt_step, position=f"x{0.5 * (span_step - w_step)}c/-0.75c+w{w_step}c/0.22c+h+jTL+e",
                 frame=["xa2f1+lLOS step (mm), positive toward the satellite"])
with pygmt.config(FONT_ANNOT_PRIMARY="12p,Helvetica", FONT_LABEL="12p,Helvetica"):    # GMT halves the fonts of a bar this short: 12p prints as ~6p
    fig.colorbar(cmap=cpt_scatter, position=f"x{(n_col - 1) * (Wm + gap_x)}c/-0.75c+w{Wm}c/0.22c+h+jTL+ef",
                 frame=["xa1f0.5+lScatter, 1 sd (mm)"])
# the caption carries what the old title said: per-pixel steps, box median removed, descending burst offsets removed, coherence coh_maps
f_grid = os.path.join(fig_dir, f"figS_event_step_maps_coh{coh_maps}")
fig.savefig(f_grid + ".png", dpi=600, transparent=False, crop="+gwhite", anti_alias=True, show=False)
fig.savefig(f_grid + ".pdf", transparent=False, crop="+gwhite", anti_alias=True, show=False)
print("Saved figure to:", f_grid + ".png")

# %% [markdown]
# ## REGISTER

# %%
L_ = [f"06 EVENT FIELD GEOMETRY  (coherence {coh_timeseries}, box {box_lonlat}, {nr} x {nc} px)",
      f"local strike {strike:.2f} deg; LOS.strike asc {L['ascending'] @ s_hat:+.3f} desc {L['descending'] @ s_hat:+.3f}, geometric asc/desc ratio {geom_ratio:+.2f}",
      f"per-event near-fault LOS steps (east - west, {xfault_step_m[0]:.0f}-{xfault_step_m[1]:.0f} m, stacked band):"]
for _, r_ in steps_df.iterrows():
    L_.append(f"   {r_.onset_yr:.2f}: asc {r_.ascending_step_mm:+.2f} mm, desc {r_.descending_step_mm:+.2f} mm, ratio {r_.ratio_asc_desc:+.2f}")
L_.append(f"excluded from the mean: {exclude_events}; mean event = {mean_event['ascending']['n']} events")
for _, r_ in burst_df.iterrows():
    L_.append(f"   burst fix {r_.track} lat {r_.line_lat:.3f} event {r_.onset_yr:.2f}: {r_.offset_mm:+.2f} mm")
L_.append(f"FP inter-event step (stacked band) {block_step(fp_rate):+.2f} mm/yr, Up {block_step(up_rate):+.2f}; FP mean event {block_step(fp_evt):+.2f} mm, Up {block_step(up_evt):+.2f}")
L_.append(f"profiles, near-fault steps east - west ({xfault_step_m[0]:.0f}-{xfault_step_m[1]:.0f} m), +/-{cross_m / 1e3:g} km, swath +/-{swath_m:.0f} m:")
for n_ in names:
    L_.append(f"   {n_}-{n_}' ({profiles[n_]['lonlat'][1]:.4f} N, local strike {profiles[n_]['strike']:.1f} deg): inter-event {profiles[n_]['rate_step']:+.2f} mm/yr, event {profiles[n_]['event_step']:+.2f} mm (n {profiles[n_]['d'].size})")
L_.append("LOS profiles per track (Fig 2), near-fault steps east - west, LOS positive toward the satellite:")
for o in tracks:
    for n_ in names:
        b_ = los_profiles[o][n_]["blocks"]
        L_.append(f"   {o[:3]} {n_}-{n_}': inter-event {los_profiles[o][n_]['rate_step']:+.2f} mm/yr, event {los_profiles[o][n_]['event_step']:+.2f} mm; "
                  f"event per block (near {xfault_step_m[0]:.0f}-{xfault_step_m[1]:.0f} m over far {far_m[0]:.0f}-{far_m[1]:.0f} m) "
                  f"W {b_['W']['amp']:+.2f} +/- {b_['W']['se']:.2f}, E {b_['E']['amp']:+.2f} +/- {b_['E']['se']:.2f} mm")
L_.append(f"X-X' event asc/desc LOS step ratio {xx_ratio:+.2f} (geometric {geom_ratio:+.2f})")
print("\n".join(L_))
sidecar = REG.sidecar("06_event_field", L_)

REG.register("06_strike_local_deg", strike, "deg", 2, where="M4", cell="strike", sidecar=sidecar)
REG.register("06_asc_desc_geom_ratio", geom_ratio, "", 2, where="M4", cell="los-steps", sidecar=sidecar)
REG.register("06_n_events_mean", mean_event["ascending"]["n"], "count", 0, where="M4", cell="mean-event", sidecar=sidecar)
for _, r_ in steps_df.iterrows():
    tag = f"{int(r_.onset_yr)}"
    REG.register(f"06_los_step_asc_{tag}_mm", r_.ascending_step_mm, "mm", 2, where="S", cell="los-steps", sidecar=sidecar)
    REG.register(f"06_los_step_desc_{tag}_mm", r_.descending_step_mm, "mm", 2, where="S", cell="los-steps", sidecar=sidecar)
    REG.register(f"06_step_ratio_{tag}", r_.ratio_asc_desc, "", 2, where="S", cell="los-steps", sidecar=sidecar)
for _, r_ in burst_df.iterrows():
    REG.register(f"06_burst_{r_.line_lat:.3f}_{int(r_.onset_yr)}_mm", r_.offset_mm, "mm", 2, where="S", cell="burst-fix", sidecar=sidecar)
REG.register("06_fp_rate_step_mm_yr", block_step(fp_rate), "mm/yr", 2, where="P6", cell="decompose", sidecar=sidecar)
REG.register("06_up_rate_step_mm_yr", block_step(up_rate), "mm/yr", 2, where="S", cell="decompose", sidecar=sidecar)
REG.register("06_fp_event_step_mm", block_step(fp_evt), "mm", 2, where="P6", cell="decompose", sidecar=sidecar)
REG.register("06_up_event_step_mm", block_step(up_evt), "mm", 2, where="S", cell="decompose", sidecar=sidecar)
for n_ in names:
    REG.register(f"06_{n_}{n_}_strike_local_deg", profiles[n_]["strike"], "deg", 1, where="Fig2", cell="profiles", sidecar=sidecar)
    REG.register(f"06_{n_}{n_}_rate_step_mm_yr", profiles[n_]["rate_step"], "mm/yr", 2, where="P6", cell="profiles", sidecar=sidecar)
    REG.register(f"06_{n_}{n_}_event_step_mm", profiles[n_]["event_step"], "mm", 2, where="P6", cell="profiles", sidecar=sidecar)
for o in tracks:
    for n_ in names:
        REG.register(f"06_{n_}{n_}_rate_step_{o[:3]}_mm_yr", los_profiles[o][n_]["rate_step"], "mm/yr", 2, where="Fig2", cell="los-profiles", sidecar=sidecar)
        REG.register(f"06_{n_}{n_}_event_step_{o[:3]}_mm", los_profiles[o][n_]["event_step"], "mm", 2, where="Fig2", cell="los-profiles", sidecar=sidecar)
        for blk in ("W", "E"):
            REG.register(f"06_{n_}{n_}_event_{blk}_{o[:3]}_mm", los_profiles[o][n_]["blocks"][blk]["amp"], "mm", 2, where="Fig2", cell="los-profiles", sidecar=sidecar)
            REG.register(f"06_{n_}{n_}_event_{blk}_{o[:3]}_se_mm", los_profiles[o][n_]["blocks"][blk]["se"], "mm", 2, where="Fig2", cell="los-profiles", sidecar=sidecar)
REG.register("06_XX_event_ratio_asc_desc", xx_ratio, "", 2, where="S2", cell="los-profiles", sidecar=sidecar)
for orbit in mean_event:
    REG.register(f"06_n_px_csi_{orbit[:3]}", int(np.isfinite(mean_event[orbit]["mean"]).sum()), "count", 0, where="M4", cell="mean-event", sidecar=sidecar)
REG.render()
print("registered; see", common_paths["registry_txt"])
