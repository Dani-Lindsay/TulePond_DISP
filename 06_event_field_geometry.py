# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
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
# event across the fault, not just its size. This is the observation half of
# Fig 2; notebook 07 will invert the mean event for slip at depth.
#
# 1. **Per-pixel model** on each track: MintPy's `timeseries2velocity` model
#    (intercept + linear rate + annual cycle + one Heaviside step per event, step
#    dates from `creep_events.csv`), solved per pixel with NaN-aware least
#    squares because `timeseries2velocity.py` refuses records with gaps.
# 2. **One data correction**, an explicit literal below: the descending
#    burst-boundary offsets (measured per event and removed). The old chain also
#    left the 2023 event out of the mean; here every event is kept (Dani,
#    2026-09-21: five events unless it is much noisier, and it is not), and the
#    2023 anomaly (its ascending step is cancelled by an east-block signal) is
#    reported by the symmetric / antisymmetric split, not acted on.
# 3. **Mean event** = mean of the five step maps per track; written in the
#    layout notebook 07 (CSI) reads.
# 4. **Decomposition** of the inter-event rate and the mean event into
#    fault-parallel and vertical with MintPy's `asc_desc2horz_vert`, exactly as
#    notebook 02 did for the whole fault, here with the local strike of the box.
# 5. **Profiles** X-X' (the Tule Pond peak) and Y-Y' (CFW1) across the
#    fault-parallel maps: the pixels, a one-pixel running median, and the
#    model-free near-fault step (east band minus west band, the aperture of
#    notebook 04). No fit here: the model is notebook 07.
# 6. Fig 2 (observations): the two profiles and the two maps.
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
# (fault-parallel and vertical rate and mean event), `profile_<XX|YY>_<field>.csv`,
# `symmetry_table.csv`, `06_event_field_stats.txt`, registry rows `06_*`.
#
# **Figures**: Fig 2 observations (pygmt), the step-map grid (supplement), the
# symmetry check (supplement).
#
# **What to check.** Near-fault LOS steps of opposite sign on the two tracks
# (ascending negative, descending positive, ratio near the geometric -0.6);
# burst offsets under 1 mm; 2023 with a same-sign east-block patch on both
# tracks; the mean event's fault-parallel profile an arctangent that decays
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
from TulePond_filepaths import (common_paths, insar_dic_list, creepmeters, box_lonlat,
                                coh_timeseries, az_smooth_n, dist_min, dist_max, paths_out, repo_dir, data_dir,
                                fig2_map_centre_km, fig2_map_half_km, fig2_map_extra_km)

import numpy as np
import pandas as pd
import pyproj
import h5py
import xarray as xr
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
utils.mpl_paper_style(base=8)          # Helvetica, 8 pt, editable PDF text
import pygmt
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
profile_pts = {"X": (-121.97698, 37.56116),      # X-X': the Tule Pond peak (trace point 1.3 km x 3 north of CFW1)
               "Y": (-121.95233, 37.53214)}      # Y-Y': CFW1
cross_m = 2500.0                # +/- profile length
swath_m = 150.0                 # +/- along-strike half-width
median_bin_m = 30.0             # running median bin (one pixel wide, so the near field is followed)
event_far_m = 2500.0            # the event profile is zeroed on |x| >= this (far field)
xfault_step_m = (dist_min * 1e3, dist_max * 1e3)   # the near-fault step band, the aperture of notebook 04 (m)

# symmetry check: cross-fault LOS profiles of every event, in the along-strike
# band of the stacked crossings (section_lengths.csv, the band Fig 1 uses), 100 m bins
sym_xmax_m, sym_bin_m, sym_eval_km = 3000.0, 100.0, 1.0

# Fig 2 map window (unrotated, north up): centred on the trace at map_centre_km from Point Pinole,
# +/- map_half_km east-west, +/- (map_half_km + map_extra_km) north-south, as the draft (10.4 x 12 km).
# The values live in TulePond_filepaths so Fig 1 (notebook 04) draws the same box.
map_centre_km, map_half_km, map_extra_km = fig2_map_centre_km, fig2_map_half_km, fig2_map_extra_km
aa_labelled = ["HGIL", "HRKT"]  # the arrays named in the paper (notebook 08); the other arrays of the section are not drawn
aa_label_pos = {"HGIL": ("RM", "-0.15c/0c"), "HRKT": ("LM", "0.15c/0.12c")}   # (justify, offset) of each label about its symbol

# legacy products for the gated parity checks (the old chain, coherence 0.9)
legacy_steps = {o: os.path.join(data_dir, "outputs", "modelling", o, "parkmont_steps_ex2023.h5" if exclude_events else "parkmont_steps.h5") for o in ("ascending", "descending")}
legacy_rate = os.path.join(data_dir, "outputs", "event_maps", "south_maps_look1.npz")

aa_file = common_paths["aa_summary"]
events_file = os.path.join(paths_out["04_event_detection"], "creep_events.csv")
lengths_file = os.path.join(paths_out["04_event_detection"], "section_lengths.csv")

REG = utils.Registry(notebook, common_paths, repo_dir)
REG.inputs(events_file, lengths_file, common_paths["fault_profiles"], aa_file,
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
strike = float(np.nanmean(prof.loc[in_box, "az_smooth"] + 90.0))      # local strike of the box, compass bearing
print(f"local strike of the trace in the box: {strike:.2f} deg (whole-fault mean in notebook 02: ~144)")
assert 130 < strike < 160

attrs = utils.load_h5_attrs(insar_dic_list[0]["ts_file"])
for dic in insar_dic_list[1:]:
    a2 = utils.load_h5_attrs(dic["ts_file"])
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
    f_fit = os.path.join(out_dir, f"timefunc_fit_{orbit}.npz")
    with h5py.File(dic["ts_file"], "r") as h:
        dates = [d.decode() for d in h["date"][:]]
    steps = step_dates_for(dates, onsets)
    with h5py.File(dic["mask_file"][coh_timeseries], "r") as h:
        keep = h[[k for k in h.keys() if k != "date"][0]][r0:r1, c0:c1].astype(bool)
    with h5py.File(dic["geo_file"], "r") as h:                  # loaded as notebook 02 does (float32, as MintPy wrote it)
        inc = h["incidenceAngle"][r0:r1, c0:c1].astype(np.float32)
        azi = h["azimuthAngle"][r0:r1, c0:c1].astype(np.float32)
    if os.path.exists(f_fit):                                   # gated: the fit is the slow step
        z = np.load(f_fit)
        assert list(z["step_dates"]) == [s0 for s0, _, _ in steps], "cached fit was made with other step dates; delete it"
        coef = {k: z[k] for k in z.files if k not in ("step_dates", "names")}
        print(f"[{orbit}] per-pixel fit loaded from {os.path.basename(f_fit)}")
    else:
        with h5py.File(dic["ts_file"], "r") as h:
            cube = h["timeseries"][:, r0:r1, c0:c1].astype(np.float32) * unit     # m -> mm
        cube[cube == 0] = np.nan                                # MintPy no-data
        print(f"[{orbit}] {len(dates)} dates, coherent pixels {keep.mean():.0%} of the box")
        coef, G, names = fit_time_func(cube, dates, keep, [s0 for s0, _, _ in steps])
        np.savez(f_fit, step_dates=np.array([s0 for s0, _, _ in steps]), names=np.array(names), **coef)
        del cube
    fits[orbit] = dict(coef=coef, keep=keep, inc=inc, azi=azi, steps=steps, dates=dates)
    print(f"  {orbit}: steps at {[s0 for s0, _, _ in steps]} (onset epochs {[o for _, o, _ in steps]})")

# %% [markdown]
# ## CHECK 1: MintPy's design matrix is the one we think it is
#
# Build the same matrix by hand (1, t, cos 2πt, sin 2πt, H(t > t_step)) on
# MintPy's own time axis and fit a synthetic series with a known 10 mm step
# through both. The step estimates must agree to 1e-9 mm: the two matrices
# span the same model. MintPy's decimal year (`ptime.yyyymmdd2years`) differs
# from our exact-seconds one by up to a few hours; the effect on a step is
# printed and is far below the noise.

# %%
from mintpy.utils import ptime

dates = fits["descending"]["dates"]
step_dates = [s0 for s0, _, _ in fits["descending"]["steps"]]
model = dict(polynomial=1, periodic=periodic, stepDate=step_dates, polyline=[], exp={}, log={})
G_mp = time_func.get_design_matrix4time_func(dates, model).astype(float)
t = np.array(ptime.yyyymmdd2years(dates), float)
t_step = np.array(ptime.yyyymmdd2years(step_dates), float)
G_hand = np.column_stack([np.ones_like(t), t, np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)] + [(t > ts).astype(float) for ts in t_step])
rng = np.random.default_rng(0)
y = 2.0 * (t - t[0]) + 1.5 * np.sin(2 * np.pi * t) + 10.0 * (t > t_step[1]) + rng.normal(0, 0.5, t.size)
c_mp = np.linalg.lstsq(G_mp, y, rcond=None)[0]
c_hand = np.linalg.lstsq(G_hand, y, rcond=None)[0]
print("step coefficients, MintPy design:", c_mp[4:].round(6))
print("step coefficients, hand design:  ", c_hand[4:].round(6))
assert np.allclose(c_mp[4:], c_hand[4:], atol=1e-9) and abs(c_mp[1] - c_hand[1]) < 1e-9
print("PASS: MintPy's time-function design is intercept + rate + annual + Heaviside(t > step date)")
# the year convention: ours (exact seconds) against MintPy's
t_ours = np.array([utils.date_to_decimal_year(d) for d in dates])
G_ours = np.column_stack([np.ones_like(t), t_ours, np.cos(2 * np.pi * t_ours), np.sin(2 * np.pi * t_ours)] + [(t > ts).astype(float) for ts in t_step])
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
    print(f"NOTE {r_.onset_yr:.2f}: ascending step {r_.ascending_step_mm:+.2f} mm, ratio {r_.ratio_asc_desc:+.2f}: its ascending step is cancelled by an east-block signal (see the symmetry check)")
print("PASS: the mean event steps negative on ascending and positive on descending; every descending step is positive")

# %% [markdown]
# ## CHECK 3: symmetric and antisymmetric parts of every event
#
# A cross-fault LOS profile $p(x)$ of pure strike-slip is antisymmetric about
# the trace: $p(-x) = -p(x)$. We split each event's profile (100 m bins, zeroed
# at the trace) into $A = (p(x) - p(-x))/2$ (the slip) and
# $S = (p(x) + p(-x))/2$ (whatever is the same on both blocks: vertical,
# atmosphere), read at 1 km. 2023 should stand out with a large S of the SAME
# sign on both tracks; that is the reason it is left out of the mean.

# %%
bins = np.arange(-sym_xmax_m, sym_xmax_m + sym_bin_m, sym_bin_m); bc = 0.5 * (bins[1:] + bins[:-1])
j_eval = int(np.argmin(np.abs(bc - sym_eval_km * 1e3)))
sym_rows, sym_prof = [], {}
for o in event_maps:
    base = in_band & (np.abs(d_signed) < sym_xmax_m)
    idx = np.digitize(d_signed[base], bins) - 1
    for i, (_, _, oy) in enumerate(fits[o]["steps"]):
        v = event_maps[o]["ev"][i][base]
        p = np.array([np.nanmedian(v[idx == j]) if (idx == j).sum() >= 15 else np.nan for j in range(bc.size)])
        p = p - np.nanmean(p[np.abs(bc) <= 150])                    # zero at the trace
        A, S = 0.5 * (p - p[::-1]), 0.5 * (p + p[::-1])
        sym_prof[(o, oy)] = (p, A, S)
        sym_rows.append(dict(track=o, onset_yr=oy, antisym_1km_mm=A[j_eval], sym_1km_mm=S[j_eval]))
sym_df = pd.DataFrame(sym_rows)
sym_wide = sym_df.pivot(index="onset_yr", columns="track", values=["antisym_1km_mm", "sym_1km_mm"])
print(sym_wide.round(2).to_string())
sym_df.to_csv(os.path.join(out_dir, "symmetry_table.csv"), index=False)

# The event that stands out: the largest ascending symmetric (non-slip) part and the asc/desc
# near-fault step ratio furthest from the geometric value (an east-block signal cancels its
# ascending step). Note the symmetric parts have OPPOSITE signs on the two tracks for every
# event at 1 km, i.e. the common-mode signal is mostly horizontal, not vertical. Reported
# always; asserted to be the excluded event only when one is excluded.
S_asc = sym_df[sym_df.track == "ascending"].set_index("onset_yr").sym_1km_mm
ratio_dev = (steps_df.set_index("onset_yr").ratio_asc_desc - geom_ratio).abs()
oy_sym, oy_ratio = float(S_asc.abs().idxmax()), float(ratio_dev.idxmax())
print(f"largest ascending symmetric part: {oy_sym:.2f} ({S_asc[oy_sym]:+.2f} mm; others at most {S_asc.drop(oy_sym).abs().max():.2f}); "
      f"most anomalous step ratio: {oy_ratio:.2f} ({steps_df.set_index('onset_yr').ratio_asc_desc[oy_ratio]:+.2f} against the geometric {geom_ratio:+.2f})")
for x in exclude_events:
    assert abs(oy_sym - x) < 0.02 and abs(oy_ratio - x) < 0.02, f"the excluded event {x} is not the anomalous one ({oy_sym:.2f}, {oy_ratio:.2f})"
print("PASS" if exclude_events else "no event excluded: the anomaly is reported, not acted on")

# %%
# QC figure: every event's LOS profile on both tracks; the descending scaled by the geometric ratio should sit on the ascending
fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
for ax, (i, (_, _, oy)) in zip(axes.flat, enumerate(fits["ascending"]["steps"])):
    pa, _, _ = sym_prof[("ascending", oy)]; pd_, _, _ = sym_prof[("descending", oy)]
    ax.plot(bc / 1e3, pa, color="#0072B2", lw=1.5, label="ascending LOS")
    ax.plot(bc / 1e3, pd_ * geom_ratio, color="#E69F00", lw=1.5, label=f"descending LOS x {geom_ratio:+.2f}")
    ax.axvline(0, color="k", lw=0.6); ax.axhline(0, color="k", lw=0.4)
    ax.set_title(f"{oy:.2f}" + ("  (excluded)" if any(abs(oy - x) < 0.02 for x in exclude_events) else ""), fontsize=9)
axes[0, 0].legend(fontsize=7)
axes[-1, -1].axis("off")
for ax in axes[-1]: ax.set_xlabel("cross-fault distance (km, + east)")
for ax in axes[:, 0]: ax.set_ylabel("LOS step (mm, zero at the trace)")
fig.suptitle(f"Event LOS profiles in the stacked band ({band.lat_lo:.3f} to {band.lat_hi:.3f} N): pure strike-slip puts orange on blue", fontsize=9)
fig.tight_layout()
f_sym = os.path.join(fig_dir, "figS_event_symmetry.png")
fig.savefig(f_sym, dpi=300); print("Saved figure to:", f_sym); plt.show()

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
# ## CHECK 4 (gated): parity with the legacy step maps
#
# `outputs/modelling/<track>/parkmont_steps.h5` (m01, coherence 0.9) holds
# the same fit on the same data; per-pixel fits do not depend on the mask, so on
# the pixels both have, the mean events must agree to a constant (the box
# median differs) apart from the burst offsets (measured on different pixel
# sets). Robust scatter of the difference under 0.1 mm = the same product.

# %%
for orbit in mean_event:
    f_leg = legacy_steps[orbit]
    if not os.path.exists(f_leg):
        print(f"[{orbit}] no legacy file, parity check skipped"); continue
    with h5py.File(f_leg, "r") as h:
        m_leg = h["mean_event"][:].astype(float)
        lx0, ly0 = float(h.attrs["X_FIRST"]), float(h.attrs["Y_FIRST"])
    assert abs(lx0 - (x0 + c0 * dx)) < 1 and abs(ly0 - (y0 + r0 * dy)) < 1 and m_leg.shape == (nr, nc), "legacy box differs; compare by coordinates"
    diff = mean_event[orbit]["mean"] - m_leg
    both = np.isfinite(diff)
    sig = utils.robust_sigma(diff[both])
    print(f"[{orbit}] {both.sum()} common pixels: median difference {np.nanmedian(diff[both]):+.3f} mm (datum), robust scatter {sig:.3f} mm")
    assert sig < 0.1, f"{orbit}: the new mean event differs from the legacy product by more than a constant"
print("PASS: mean events equal the legacy m01 products up to the datum")

# %% [markdown]
# ## RUN: fault-parallel and vertical maps
#
# `asc_desc2horz_vert` (MintPy) on the inter-event rate and on the mean event,
# with each track's own incidence and azimuth and `horz_az_angle = -strike`
# (MintPy counts anticlockwise), as notebook 02. The vertical component of
# creep is ~0, so the Up maps are the QC: they should be flat across the
# trace.

# %%
def decompose(v_asc, v_des):
    both_ = np.isfinite(v_asc) & np.isfinite(v_des)
    dlos = np.stack([v_asc, v_des]).astype(np.float32); dlos[:, ~both_] = np.nan
    los_inc = np.stack([fits["ascending"]["inc"], fits["descending"]["inc"]]).astype(np.float32)
    los_azi = np.stack([fits["ascending"]["azi"], fits["descending"]["azi"]]).astype(np.float32)
    fp_, up_ = asc_desc2horz_vert(dlos, los_inc, los_azi, horz_az_angle=-strike, step=geom_step)
    return fp_.astype(float), up_.astype(float)


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
# ## CHECK 5 (gated): the inter-event rate map against the legacy one
#
# `outputs/event_maps/south_maps_look1.npz` (event_map_profiles.py) solved the
# rate with ramps for the events instead of steps, coherence 0.9, and a
# west-block reference, then decomposed the same way. Same field, different
# solver: correlation above 0.95 on the common pixels.

# %%
f_leg = legacy_rate
if os.path.exists(f_leg):
    z = np.load(f_leg)
    lx0, ly0, lxs, lys = (float(z[k]) for k in ("geo_X_FIRST", "geo_Y_FIRST", "geo_X_STEP", "geo_Y_STEP"))
    m_leg = z["map_rate"]
    # sample the legacy grid at our pixel centres (both 30 m; offsets are whole pixels)
    ci = np.round((Xg - lx0) / lxs - 0.5).astype(int); ri = np.round((Yg - ly0) / lys - 0.5).astype(int)
    okp = (ci >= 0) & (ci < m_leg.shape[1]) & (ri >= 0) & (ri < m_leg.shape[0])
    leg = np.full(fp_rate.shape, np.nan); leg[okp] = m_leg[ri[okp], ci[okp]]
    both = np.isfinite(leg) & np.isfinite(fp_rate)
    corr = float(np.corrcoef(leg[both], fp_rate[both])[0, 1])
    print(f"inter-event FP rate vs legacy: {both.sum()} common pixels, correlation {corr:.3f}, "
          f"block step {block_step(fp_rate):+.2f} vs {block_step(leg):+.2f} mm/yr")
    assert corr > 0.95
    print("PASS")
else:
    print("no legacy rate map, check skipped")

# %% [markdown]
# ## RUN: the cross-fault profiles X-X' and Y-Y'
#
# Perpendicular to the local strike through each point, +/- `cross_m`, all
# pixels within +/- `swath_m` along strike, plus a running 100 m median. The
# inter-event profile is zeroed at the fault (mid-point of the two block
# medians within 1 km); the event profile on its far field.

# %%
def profile(m_, lonlat):
    """Cross-fault profile of map m_ through lonlat: d (m, + east), v, and the line's end points (UTM)."""
    cx, cy = (float(t_) for t_ in to_utm.transform(*lonlat))
    pe, pn = nrm                                                # cross-fault unit vector (+ east)
    se, sn = -pn, pe                                            # along-strike
    dc = (Xg - cx) * pe + (Yg - cy) * pn; da = (Xg - cx) * se + (Yg - cy) * sn
    sel = (np.abs(da) <= swath_m) & (np.abs(dc) <= cross_m) & np.isfinite(m_)
    d, v = dc[sel], m_[sel]
    o = np.argsort(d)
    ends = (cx - cross_m * pe, cy - cross_m * pn, cx + cross_m * pe, cy + cross_m * pn)
    return d[o], v[o], ends


def running_median(d, v, bin_m=median_bin_m):
    edges = np.arange(-cross_m, cross_m + bin_m, bin_m)
    idx = np.digitize(d, edges)
    xc, med = [], []
    for i in range(1, len(edges)):
        m_ = idx == i
        if m_.sum() >= 3:
            xc.append(0.5 * (edges[i - 1] + edges[i])); med.append(np.median(v[m_]))
    return np.array(xc), np.array(med)


profiles = {}
for name, lonlat in profile_pts.items():
    d, v, ends = profile(fp_rate, lonlat)
    near = np.abs(d) <= 1000
    v = v - 0.5 * (np.median(v[near & (d < 0)]) + np.median(v[near & (d > 0)]))     # zero at the fault
    de, ve, _ = profile(fp_evt, lonlat)
    far = np.abs(de) >= event_far_m
    ve = ve - (np.median(ve[far]) if far.sum() >= 20 else np.median(ve))
    lo_, hi_ = xfault_step_m
    r_step = float(np.median(v[(d > lo_) & (d < hi_)]) - np.median(v[(d < -lo_) & (d > -hi_)]))
    e_step = float(np.median(ve[(de > lo_) & (de < hi_)]) - np.median(ve[(de < -lo_) & (de > -hi_)]))
    profiles[name] = dict(d=d, v=v, de=de, ve=ve, ends=ends, lonlat=lonlat, rate_step=r_step, event_step=e_step)
    pd.DataFrame(dict(d_m=d, fp_rate_mm_yr=v)).to_csv(os.path.join(out_dir, f"profile_{name}{name}_rate.csv"), index=False)
    pd.DataFrame(dict(d_m=de, fp_event_mm=ve)).to_csv(os.path.join(out_dir, f"profile_{name}{name}_event.csv"), index=False)
    print(f"{name}-{name}': {d.size} rate pixels, {de.size} event pixels; near-fault steps ({lo_:.0f}-{hi_:.0f} m): rate {r_step:+.2f} mm/yr, event {e_step:+.2f} mm")
print("legacy (coherence 0.9, 2026-08-30, 1-D fits): X-X' event 9.12 mm, Y-Y' 2.68 mm")

# %% [markdown]
# ## Fig 2, observations: profiles and maps (pygmt)
#
# Left: X-X' and Y-Y' through the inter-event rate (top pair) and the mean
# event (bottom pair): grey pixels, orange one-pixel running median; no model
# (the slip model is 07's and shares this figure). Right: the
# two fault-parallel maps, north up, with the profile lines, the creepmeter
# and the alignment arrays of the section. Labels A to D; notebook 07 adds the
# model panels. The bottom of each profile pair sits on the bottom of its map.

# %%
# --- the map window: a lon/lat box about the trace point map_centre_km from Point Pinole
km_of = utils.lat_to_km_map(common_paths["fault_profiles"])
j_c = int(np.argmin(np.abs(km_of(prof.ft_lat.values) - map_centre_km)))
xc, yc = (float(t_) for t_ in to_utm.transform(prof.ft_lon[j_c], prof.ft_lat[j_c]))
lon_w, lat_s = to_ll.transform(xc - map_half_km * 1e3, yc - (map_half_km + map_extra_km) * 1e3)
lon_e, lat_n = to_ll.transform(xc + map_half_km * 1e3, yc + (map_half_km + map_extra_km) * 1e3)
region_map = [lon_w, lon_e, lat_s, lat_n]
print(f"map window: {2 * map_half_km:g} x {2 * (map_half_km + map_extra_km):g} km about {map_centre_km:g} km ({prof.ft_lat[j_c]:.4f} N)")

# the maps on a regular lon/lat grid (nearest 30 m pixel)
lat_q = np.arange(lat_s - 0.002, lat_n + 0.002, 0.00025)
lon_q = np.arange(lon_w - 0.002, lon_e + 0.002, 0.0003)
LONq, LATq = np.meshgrid(lon_q, lat_q)
Eq, Nq = to_utm.transform(LONq, LATq)
ci = np.round((Eq - (x0 + c0 * dx)) / dx - 0.5).astype(int); ri = np.round((Nq - (y0 + r0 * dy)) / dy - 0.5).astype(int)
okq = (ci >= 0) & (ci < nc) & (ri >= 0) & (ri < nr)
_, jq = tree.query(np.c_[Eq.ravel(), Nq.ravel()])
side_q = ((np.c_[Eq.ravel(), Nq.ravel()] - np.c_[fx, fy][jq]) @ nrm).reshape(Eq.shape)    # + east of the trace


def to_grid(m_, balance):
    z = np.full(LONq.shape, np.nan); z[okq] = m_[ri[okq], ci[okq]]
    if balance:                                                 # fault at zero: mid-point of the two block medians in the window
        z -= 0.5 * (np.nanmedian(z[side_q < 0]) + np.nanmedian(z[side_q > 0]))
    lim = max(float(np.nanpercentile(np.abs(z[np.isfinite(z)]), 98)), 0.3)
    return xr.DataArray(z, coords={"lat": lat_q, "lon": lon_q}, dims=("lat", "lon")), lim


def gmt_map_size(region, projection):
    """Width and height (cm) GMT gives a region/projection pair (mapproject -W)."""
    with pygmt.clib.Session() as lib:
        f_ = os.path.join(out_dir, "cpt", "dims.txt")
        lib.call_module("mapproject", f"-R{'/'.join(str(r) for r in region)} -J{projection} -W ->{f_}")
    return tuple(float(x) for x in open(f_).read().split())


cpt_dir = os.path.join(out_dir, "cpt"); os.makedirs(cpt_dir, exist_ok=True)
g_rate, lim_rate = to_grid(fp_rate, True)
g_evt, lim_evt = to_grid(fp_evt, False)
cpt_rate = utils.write_cpt(os.path.join(cpt_dir, "rate.cpt"), "RdBu_r", -lim_rate, lim_rate)
cpt_evt = utils.write_cpt(os.path.join(cpt_dir, "event.cpt"), "RdBu_r", -lim_evt, lim_evt)
lines_ll = {n_: to_ll.transform(np.array([p["ends"][0], p["ends"][2]]), np.array([p["ends"][1], p["ends"][3]])) for n_, p in profiles.items()}
cfw = [c for c in creepmeters if c[0] == "CFW1"][0]
aa = utils.load_aa_summary(aa_file)
aa = aa[(aa.long_deg > lon_w) & (aa.long_deg < lon_e) & (aa.lat_deg > lat_s) & (aa.lat_deg < lat_n)]
print("alignment arrays in the window:", ", ".join(aa.site_code))

# --- sizes (cm): the observation half of a double-column figure
Wp, Hp = 4.4, 2.0            # one profile panel
gap_x, gap_pair, gap_row = 1.2, 0.25, 1.4
Hpair = 2 * Hp + gap_pair    # a stacked pair; the map is made exactly this tall
w_, h_ = gmt_map_size(region_map, "M4c")
Wm, Hm = 4.0 * Hpair / h_, Hpair
proj_map = f"M{Wm:.3f}c"
print(f"map panels {Wm:.2f} x {Hm:.2f} cm")
orange, grey = "#E69F00", "gray30"
names = list(profiles)

fig = pygmt.Figure()
pygmt.config(FONT="8p,Helvetica", FONT_ANNOT_PRIMARY="7p,Helvetica", FONT_LABEL="7p,Helvetica", MAP_FRAME_TYPE="plain",
             MAP_FRAME_PEN="0.6p", MAP_TICK_LENGTH_PRIMARY="2p", MAP_LABEL_OFFSET="4p", MAP_ANNOT_OFFSET="3p", FORMAT_GEO_MAP="ddd.xx")


def profile_pair(ylab, lim, label, vkey, dkey):
    """Two stacked panels (X above Y) for one field, drawn from the current origin
    (the top panel's bottom-left). Leaves the origin at the LOWER panel."""
    for j, n_ in enumerate(names):
        p = profiles[n_]
        d, v = p[dkey], p[vkey]                                  # zeroed above: the rate at the fault, the event on its far field
        if j > 0:
            fig.shift_origin(yshift=f"-{Hp + gap_pair}c")
        xfr = "xa1000f500+lCross-fault distance (m)" if j == 1 else "xa1000f500"
        fig.basemap(region=[-cross_m, cross_m, -lim, lim], projection=f"X{Wp}c/{Hp}c", frame=["WSrt" if j == 1 else "Wsrt", xfr, f"ya{lim / 2:g}"])
        fig.plot(x=d, y=v, style="c0.03c", fill=grey, transparency=30)      # the data
        xm, vm = running_median(d, v)
        fig.plot(x=xm, y=vm, pen="1.0p," + orange)                           # the running median, one pixel wide
        fig.plot(x=[0, 0], y=[-lim, lim], pen="0.5p,gray50")
        fig.text(position="TL", text=n_, font="9p,Helvetica-Bold", offset="0.1c/-0.08c")
        fig.text(position="TR", text=n_ + "'", font="9p,Helvetica-Bold", offset="-0.1c/-0.08c")
        if j == 0:
            fig.text(x=-cross_m, y=lim, text=label, justify="BL", offset="0c/0.12c", font="9p,Helvetica", no_clip=True)
    # one y label for the pair, centred on the pair (drawn from the lower panel)
    fig.text(position="ML", text=ylab, angle=90, justify="BC", offset=f"-0.95c/{0.5 * (Hp + gap_pair)}c", font="9p,Helvetica", no_clip=True)


def map_panel(grid, cpt, lim, label, cb_label):
    """One map, north up, in lon/lat: the field, the trace, the profile lines, CFW1 and the alignment arrays."""
    fig.basemap(region=region_map, projection=proj_map, frame=["lrtb"])
    fig.grdimage(grid=grid, cmap=cpt, nan_transparent=True)
    fig.plot(x=prof.ft_lon.values, y=prof.ft_lat.values, pen="0.7p,black")
    for n_, (llon, llat) in lines_ll.items():
        fig.plot(x=llon, y=llat, pen="1.0p,black")
        fig.text(x=llon[0], y=llat[0], text=n_, justify="RM", offset="-0.08c/0c", font="8p,Helvetica-Bold")
        fig.text(x=llon[1], y=llat[1], text=n_ + "'", justify="LM", offset="0.08c/0c", font="8p,Helvetica-Bold")
    used = aa.site_code.isin(aa_labelled)                                        # only the arrays the paper uses are drawn
    fig.plot(x=aa.long_deg.values[used], y=aa.lat_deg.values[used], style="s0.18c", fill="black", pen="0.4p,white")
    for _, r_ in aa[used].iterrows():                                              # HGIL left of its symbol, HRKT right and nudged up off the Y-Y' line
        just, off = aa_label_pos[r_.site_code]
        utils.text_backed(fig, x=r_.long_deg, y=r_.lat_deg, text=r_.site_code, justify=just, offset=off, font="7p,Helvetica")
    fig.plot(x=[cfw[2]], y=[cfw[1]], style="i0.22c", fill="#F5B400", pen="0.5p,black")
    utils.text_backed(fig, x=cfw[2], y=cfw[1], text="CFW1", justify="RM", offset="-0.15c/0c", font="7p,Helvetica")
    fig.text(position="TL", text=label, justify="BL", offset="0c/0.12c", font="9p,Helvetica", no_clip=True)
    fig.basemap(region=region_map, projection=proj_map, frame=["lrtb"], map_scale="jBL+w2k+o0.3c/0.55c")      # frame on top, scale bar
    with pygmt.config(FONT_ANNOT_PRIMARY="13p", FONT_LABEL="13p"):
        fig.colorbar(cmap=cpt, position=f"x{Wm + 0.25}c/0.3c+w{Hm - 0.6}c/0.3c+jBL", frame=f"xa{max(1, round(lim / 2)):g}+l{cb_label}")


# top row: A the inter-event profiles, B the inter-event map (the map is Hpair tall, so its bottom is the pair's bottom)
profile_pair("Fault parallel (mm/yr)", 10.0, "@%Helvetica-Bold%A@%% @:7p:Inter-event rate@::", "v", "d")
fig.shift_origin(xshift=f"{Wp + gap_x}c")
map_panel(g_rate, cpt_rate, lim_rate, "@%Helvetica-Bold%B@%% @:7p:Inter-event rate (FP)@::", "mm/yr")
# bottom row: C the mean-event profiles, D the mean-event map
fig.shift_origin(xshift=f"-{Wp + gap_x}c", yshift=f"-{gap_row + Hp}c")
profile_pair("Fault parallel (mm)", 10.0, f"@%Helvetica-Bold%C@%% @:7p:Mean event ({mean_event['ascending']['n']} events)@::", "ve", "de")
fig.shift_origin(xshift=f"{Wp + gap_x}c")
map_panel(g_evt, cpt_evt, lim_evt, "@%Helvetica-Bold%D@%% @:7p:Mean event (FP)@::", "mm")

f_fig2 = os.path.join(fig_dir, f"fig2_observations_coh{coh_timeseries}_{mean_event['ascending']['n']}ev")
fig.savefig(f_fig2 + ".png", dpi=300, transparent=False, crop="+gwhite", anti_alias=True, show=False)   # +gwhite: opaque page
fig.savefig(f_fig2 + ".pdf", transparent=False, crop="+gwhite", anti_alias=True, show=False)
print("Saved Fig 2 (observations) to:", f_fig2 + ".png")
fig.show()

# %% [markdown]
# ## Supplement: the step maps, every event and track
#
# Rows ascending / descending; columns the five events (burst-fixed), the mean
# of the kept events, and the per-pixel scatter across them. LOS mm, positive
# toward the satellite, so the two rows have opposite colours across the fault.

# %%
clim = 4.0
n_ev = len(onsets)
fig, axes = plt.subplots(2, n_ev + 2, figsize=(2.1 * (n_ev + 2), 5.0), sharex=True, sharey=True)
for r_, orbit in enumerate(event_maps):
    panels = [(event_maps[orbit]["ev"][i], f"{oy:.2f}" + (" (excl.)" if any(abs(oy - x) < 0.02 for x in exclude_events) else ""), "RdBu_r", (-clim, clim))
              for i, (_, _, oy) in enumerate(fits[orbit]["steps"])]
    panels += [(mean_event[orbit]["mean"], f"mean of {mean_event[orbit]['n']}", "RdBu_r", (-clim, clim)),
               (mean_event[orbit]["scatter"], "scatter (sd)", "viridis", (0, clim))]
    for c_, (m_, ttl, cmap, cl) in enumerate(panels):
        ax = axes[r_, c_]
        im = ax.pcolormesh(lon_g, lat_g, m_, cmap=cmap, vmin=cl[0], vmax=cl[1], shading="nearest", rasterized=True)
        ax.plot(prof.ft_lon, prof.ft_lat, "k-", lw=0.5)
        ax.set_xlim(box_lonlat[0], box_lonlat[2]); ax.set_ylim(box_lonlat[1], box_lonlat[3]); ax.set_aspect(1 / np.cos(np.deg2rad(37.55)))
        ax.set_title(f"{orbit[:3]} {ttl}", fontsize=8, loc="left")
        ax.tick_params(labelsize=6)
        if r_ == 1 and c_ in (0, n_ev + 1):
            fig.colorbar(im, ax=axes[:, c_] if c_ else axes[:, :n_ev + 1], shrink=0.5, orientation="horizontal", pad=0.06, label="LOS mm" if c_ == 0 else "mm")
fig.suptitle(f"Per-pixel step maps, coherence {coh_timeseries}, box median removed, descending burst offsets removed", fontsize=9)
f_grid = os.path.join(fig_dir, f"figS_event_step_maps_coh{coh_timeseries}.png")
fig.savefig(f_grid, dpi=300, bbox_inches="tight"); print("Saved figure to:", f_grid); plt.show()

# %% [markdown]
# ## REGISTER

# %%
L_ = [f"06 EVENT FIELD GEOMETRY  (coherence {coh_timeseries}, box {box_lonlat}, {nr} x {nc} px)",
      f"local strike {strike:.2f} deg; LOS.strike asc {L['ascending'] @ s_hat:+.3f} desc {L['descending'] @ s_hat:+.3f}, geometric asc/desc ratio {geom_ratio:+.2f}",
      f"per-event near-fault LOS steps (east - west, {xfault_step_m[0]:.0f}-{xfault_step_m[1]:.0f} m, stacked band):"]
for _, r_ in steps_df.iterrows():
    L_.append(f"   {r_.onset_yr:.2f}: asc {r_.ascending_step_mm:+.2f} mm, desc {r_.descending_step_mm:+.2f} mm, ratio {r_.ratio_asc_desc:+.2f}")
L_.append("symmetric / antisymmetric parts at 1 km (mm):")
for oy, r_ in sym_wide.iterrows():
    L_.append(f"   {oy:.2f}: asc A {r_[('antisym_1km_mm', 'ascending')]:+.2f} S {r_[('sym_1km_mm', 'ascending')]:+.2f} | "
              f"desc A {r_[('antisym_1km_mm', 'descending')]:+.2f} S {r_[('sym_1km_mm', 'descending')]:+.2f}")
L_.append(f"excluded from the mean: {exclude_events}; mean event = {mean_event['ascending']['n']} events")
for _, r_ in burst_df.iterrows():
    L_.append(f"   burst fix {r_.track} lat {r_.line_lat:.3f} event {r_.onset_yr:.2f}: {r_.offset_mm:+.2f} mm")
L_.append(f"FP inter-event step (stacked band) {block_step(fp_rate):+.2f} mm/yr, Up {block_step(up_rate):+.2f}; FP mean event {block_step(fp_evt):+.2f} mm, Up {block_step(up_evt):+.2f}")
L_.append(f"profiles, near-fault steps east - west ({xfault_step_m[0]:.0f}-{xfault_step_m[1]:.0f} m), +/-{cross_m / 1e3:g} km, swath +/-{swath_m:.0f} m:")
for n_ in names:
    L_.append(f"   {n_}-{n_}' ({profiles[n_]['lonlat'][1]:.4f} N): inter-event {profiles[n_]['rate_step']:+.2f} mm/yr, event {profiles[n_]['event_step']:+.2f} mm (n {profiles[n_]['d'].size})")
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
for oy, r_ in sym_wide.iterrows():
    tag = f"{int(oy)}"
    REG.register(f"06_sym_asc_{tag}_mm", r_[("sym_1km_mm", "ascending")], "mm", 2, where="S", cell="symmetry", sidecar=sidecar)
    REG.register(f"06_sym_desc_{tag}_mm", r_[("sym_1km_mm", "descending")], "mm", 2, where="S", cell="symmetry", sidecar=sidecar)
for _, r_ in burst_df.iterrows():
    REG.register(f"06_burst_{r_.line_lat:.3f}_{int(r_.onset_yr)}_mm", r_.offset_mm, "mm", 2, where="S", cell="burst-fix", sidecar=sidecar)
REG.register("06_fp_rate_step_mm_yr", block_step(fp_rate), "mm/yr", 2, where="P6", cell="decompose", sidecar=sidecar)
REG.register("06_up_rate_step_mm_yr", block_step(up_rate), "mm/yr", 2, where="S", cell="decompose", sidecar=sidecar)
REG.register("06_fp_event_step_mm", block_step(fp_evt), "mm", 2, where="P6", cell="decompose", sidecar=sidecar)
REG.register("06_up_event_step_mm", block_step(up_evt), "mm", 2, where="S", cell="decompose", sidecar=sidecar)
for n_ in names:
    REG.register(f"06_{n_}{n_}_rate_step_mm_yr", profiles[n_]["rate_step"], "mm/yr", 2, where="P6", cell="profiles", sidecar=sidecar)
    REG.register(f"06_{n_}{n_}_event_step_mm", profiles[n_]["event_step"], "mm", 2, where="P6", cell="profiles", sidecar=sidecar)
for orbit in mean_event:
    REG.register(f"06_n_px_csi_{orbit[:3]}", int(np.isfinite(mean_event[orbit]["mean"]).sum()), "count", 0, where="M4", cell="mean-event", sidecar=sidecar)
REG.render()
print("registered; see", common_paths["registry_txt"])
