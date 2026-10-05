#!/usr/bin/env python3
"""
QC 05x, version 3 (exploratory, NOT a paper figure; registers nothing, edits no notebook).
v1 (event_zoom.py) and v2 (event_zoom_v2.py) and their outputs are kept unchanged.

Purpose: v2 showed what looks like event DURATION (up to ~30 d) and post-event DECAY. Two likely artefacts:
  (1) detrending -- v2 extrapolated a 120-d pre-window line (no annual term);
  (2) interpolation -- fp_ts sits on a common date grid where one track is often interpolated (across gaps
      up to 45 d), turning a step on one track into a ramp.
v3 therefore:
  * detrends every series with the paper's INTER-EVENT MODEL: per crossing, over the whole record,
    constant + rate + annual sin/cos + one Heaviside step H(t > onset_lo) per catalogue event (all seven
    rows of 04's creep_events.csv, Tule Pond and COZ1, as 04's per-crossing joint fit). 04's `design()` is
    not importable (04 is a script that runs on import), so it is REPLICATED here exactly
    (columns 1, t - mean(t), sin 2 pi t, cos 2 pi t, H(t > step epoch)). Displayed value for event k =
    data - (constant + rate + annual) - (the fitted steps of every event BEFORE event k): zero is the
    inter-event model just before the event, and the event's own step stays in.
  * row 1 (fault-parallel, common grid): filled markers = epochs where BOTH tracks acquired that day,
    open markers = epochs where either track is interpolated (03's rule: a real acquisition within half a day
    of the epoch, on the track's raw dates = the columns of creep_ts_LOS_<track>.csv).
  * row 2 (new): per-track LOS cross-fault series for the 58.0 and 60.2 km stacks on each track's own raw
    acquisition dates (no interpolation), converted to FP-equivalent by dividing by the LOS sensitivity to
    fault-parallel motion (local geometry from creep_geom_<track>.csv and the local strike az_smooth + 90,
    exactly the G of utils.design_matrix_up_fp); this assumes no vertical motion in the step. Same detrending.
  * location stacks narrowed to +/-0.1 km (3 crossings). Each crossing already averages pixels within
    +/-200 m along strike (x_width), so neighbouring crossings share pixels and the effective along-strike
    width is ~0.4-0.6 km regardless.

Outputs: fig_dir/qc05x_event_zoom_v3.png (600 dpi) + .pdf, _steps.csv (per-interval increments on each
track's real dates, plus decay test), _caption.txt
Run: ~/miniforge3/envs/opera_disp_feb13/bin/python drafts/05x_event_zoom/event_zoom_v3.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
import insar_utils as utils                                           # noqa: E402
from TulePond_filepaths import (common_paths, paths_out, creepmeters,  # noqa: E402
                                creepmeter_glitches, profile_smooth_km, x_width,
                                az_smooth_n)

# ------------------------------------------------------------------ parameters
HALF_DAYS = 60
STACK_HALF_KM = 0.1                 # 3 crossings at 100 m spacing
MIN_EPOCH_FRAC = 0.5                # 04's min_epoch_frac: a crossing needs >50 % finite epochs to be fitted
HOV_KM = (54.0, 65.0)
CLIM_PERCENTILE = 98
REAL_TOL = 0.5 / 365.25             # 03's rule for a real acquisition
NSIG_RISE = 2.0                     # an interval "rises" if its increment exceeds this x the single-interval noise
DECAY_EARLY = (0, 18)               # days after onset_hi: early post-event level
DECAY_LATE = (30, 60)               # days after onset_lo: late post-event level
CM_STYLE = dict(color="#000000", lw=0.6, ls=(0, (3, 1.5)))
CM_CODE = "CFW1"

fig_path = os.path.join(common_paths["fig_dir"], "qc05x_event_zoom_v3")
stamp = utils.provenance_stamp("drafts/05x_event_zoom/event_zoom_v3.py", REPO)
out03 = paths_out["03_creep_timeseries"]


def read_ts(path):
    df = pd.read_csv(path, index_col=0)
    df.columns = df.columns.astype(float)
    df.index = df.index.astype(float)
    return df


# ------------------------------------------------------------------ data
fp_ts = read_ts(os.path.join(out03, "fp_ts.csv"))
epochs = fp_ts.columns.values
epoch_days = pd.DatetimeIndex([utils.decimal_year_to_day(t) for t in epochs])
assert epoch_days.is_monotonic_increasing and epoch_days.is_unique
to_km = utils.lat_to_km_map(common_paths["fault_profiles"])
km = to_km(fp_ts.index.values)
fp = fp_ts.values

los = {"asc": read_ts(os.path.join(out03, "creep_ts_LOS_ascending.csv")),
       "desc": read_ts(os.path.join(out03, "creep_ts_LOS_descending.csv"))}
geom = {"asc": pd.read_csv(os.path.join(out03, "creep_geom_ascending.csv"), index_col=0),
        "desc": pd.read_csv(os.path.join(out03, "creep_geom_descending.csv"), index_col=0)}

# real epochs of the common grid (03's rule, on the raw per-track dates)
real = {}
for trk in ("asc", "desc"):
    d = los[trk].columns.values
    real[trk] = np.array([np.min(np.abs(d - t)) < REAL_TOL for t in epochs])
both_real = real["asc"] & real["desc"]
print(f"fp_ts epochs {len(epochs)}: both real {both_real.sum()}, asc interpolated {(~real['asc']).sum()} "
      f"({100 * (~real['asc']).mean():.0f} %), desc interpolated {(~real['desc']).sum()} ({100 * (~real['desc']).mean():.0f} %)")
assert both_real.sum() == 291, "03 reports 291 epochs with both tracks real"

events_all = pd.read_csv(os.path.join(paths_out["04_event_detection"], "creep_events.csv"))
step_epochs_all = np.sort(events_all["onset_lo"].to_numpy(float))
coz1_events = events_all[events_all.segment == "COZ1"].reset_index(drop=True)
events = events_all[events_all.segment == "TULEPOND"].reset_index(drop=True)
assert len(events) == 5 and len(coz1_events) == 2


def design(dates, step_epochs):
    """REPLICA of notebook 04's design(): constant, rate (t - mean t), annual sin and cos, then one Heaviside
    step H(t > step epoch) per event."""
    columns = [np.ones_like(dates), dates - dates.mean(), np.sin(2 * np.pi * dates), np.cos(2 * np.pi * dates)]
    for step_epoch in step_epochs:
        columns.append((dates > step_epoch).astype(float))
    return np.column_stack(columns)


def inter_event_fit(dates, values, min_frac=MIN_EPOCH_FRAC):
    """Joint fit over the whole record. Returns (background at every date, step amplitudes, full-model residual),
    or None if too few finite values."""
    ok = np.isfinite(values)
    if ok.sum() < min_frac * len(dates):
        return None
    G = design(dates, step_epochs_all)
    coef, *_ = np.linalg.lstsq(G[ok], values[ok], rcond=None)
    background = G[:, :4] @ coef[:4]
    resid = values - G @ coef
    return background, coef[4:], resid


def display_series(values, fit, onset_lo):
    """data - background - steps of every event before this one."""
    background, amps, _ = fit
    earlier = step_epochs_all < onset_lo - 1e-9
    return values - background - amps[earlier].sum()


# per-crossing fits on the common grid (rows 1 and 3)
fp_fits = [inter_event_fit(epochs, fp[r]) for r in range(fp.shape[0])]
print(f"fp crossings fitted: {sum(f is not None for f in fp_fits)} of {len(fp_fits)}")

# CFW1
prof = pd.read_csv(common_paths["fault_profiles"])
_, cfw1_lat, cfw1_lon, _ = [c for c in creepmeters if c[0] == CM_CODE][0]
dist_m = np.hypot((prof.ft_lat.values - cfw1_lat) * 111190.0,
                  (prof.ft_lon.values - cfw1_lon) * 111190.0 * np.cos(np.radians(cfw1_lat)))
cfw1_km = round((len(prof) - 1 - int(np.argmin(dist_m))) * 0.1, 1)
assert abs(cfw1_km - 62.6) < 0.05

LOCS = [("km56.5", "56.5 km (N extent)", 56.5, "#000000", "o"),
        ("km58.0", "58.0 km (peak)", 58.0, "#D55E00", "s"),
        ("km60.2", "60.2 km (peak)", 60.2, "#0072B2", "^"),
        ("cfw1", f"{cfw1_km} km CFW1 (S extent)", cfw1_km, "#7F7F7F", "D")]
TRACK_LOCS = [("km58.0", 58.0, "#D55E00"), ("km60.2", 60.2, "#0072B2")]
TRACK_STYLE = {("km58.0", "asc"): dict(marker="s", ls="-"), ("km58.0", "desc"): dict(marker="D", ls=(0, (2.5, 1.2))),
               ("km60.2", "asc"): dict(marker="^", ls="-"), ("km60.2", "desc"): dict(marker="v", ls=(0, (2.5, 1.2)))}

# per-track per-crossing FP-equivalent LOS fits for the two peak stacks
# local strike exactly as 03: rolling median of profile_az over az_smooth_n crossings, + 90
prof["az_smooth"] = prof["profile_az"].rolling(az_smooth_n, center=True, min_periods=3).median()
trace_lat = prof.ft_lat.values
track_rows = {}
for trk in ("asc", "desc"):
    lk = to_km(los[trk].index.values)
    dates_t = los[trk].columns.values
    for key, c, _ in TRACK_LOCS:
        sel = np.where(np.abs(lk - c) <= STACK_HALF_KM + 1e-6)[0]
        assert sel.size == 3, f"{trk} {c} km: {sel.size} crossings"
        crossings = []
        for r in sel:
            lat = los[trk].index.values[r]
            g = geom[trk].iloc[int(np.argmin(np.abs(geom[trk].index.values - lat)))]
            assert np.isclose(geom[trk].index.values[int(np.argmin(np.abs(geom[trk].index.values - lat)))], lat, atol=1e-9)
            nearest = int(np.argmin(np.abs(trace_lat - lat)))
            assert np.isclose(trace_lat[nearest], lat, atol=1e-9), "crossing is not a trace point"
            strike = prof["az_smooth"].values[nearest] + 90.0
            g_fp = utils.design_matrix_up_fp(g.inc_deg, g.azi_deg, strike)[0, 1]
            v = los[trk].values[r] / g_fp
            fit = inter_event_fit(dates_t, v)
            assert fit is not None, f"{trk} {lat}: too few epochs"
            crossings.append(dict(lat=lat, g_fp=g_fp, v=v, fit=fit))
        track_rows[(key, trk)] = dict(dates=dates_t, crossings=crossings)
        print(f"{trk} {c} km: G_fp = {', '.join(f'{x['g_fp']:+.3f}' for x in crossings)} (LOS per mm of FP)")

# 04's per-crossing joint-model steps (row 4) and their empirical noise from the COZ1 events
ef = pd.read_csv(os.path.join(paths_out["04_event_detection"], "event_fraction_by_crossing.csv")).sort_values("km")
ef_km = ef.km.to_numpy(float)
def amp_col(onset_lo):
    col = f"amp_{onset_lo:.3f}"
    assert col in ef.columns, col
    return col
in_tp = (ef_km >= HOV_KM[0]) & (ef_km <= HOV_KM[1])
null_single = np.concatenate([ef[amp_col(t)].to_numpy(float)[in_tp] for t in coz1_events.onset_lo])
null_stack = []
for t in coz1_events.onset_lo:
    a = ef[amp_col(t)].to_numpy(float)
    for c in np.arange(HOV_KM[0] + 0.1, HOV_KM[1] - 0.1 + 1e-9, 0.3):     # non-overlapping 3-crossing windows
        sel = np.abs(ef_km - c) <= STACK_HALF_KM + 1e-6
        if np.isfinite(a[sel]).sum() >= 2:
            null_stack.append(np.nanmean(a[sel]))
noise_single, noise_stack = utils.robust_sigma(null_single), utils.robust_sigma(np.array(null_stack))
print(f"joint-model step noise (COZ1 events at 54-65 km): single {noise_single:.2f} mm, "
      f"+/-{STACK_HALF_KM} km stack {noise_stack:.2f} mm (n={len(null_stack)})")

# Gittins
gittins = pd.read_csv(common_paths["gittins_catalogue"], parse_dates=["Start Time", "End Time"])
g_cfw1 = gittins[gittins["Creepmeter abbreviation"] == CM_CODE].copy()
g_cfw1_end = g_cfw1["End Time"].max()
GITTINS_COLOUR, GITTINS_TEXT = "#E69F00", "#9A6400"

# creepmeter, whole record, same inter-event model on its daily means
cm_t, cm_slip = utils.daily_creepmeter(common_paths["creepmeter_dir"], CM_CODE, epochs[0] - 0.1, epochs[-1] + 0.3,
                                       creepmeter_glitches)
cm_days = pd.DatetimeIndex([pd.Timestamp(utils.decimal_year_to_date(t)).floor("D") for t in cm_t])
cm_fit = inter_event_fit(cm_t, cm_slip)


def is_event_interval(t_a, t_b):
    """True if the interval (t_a, t_b] overlaps any catalogue onset window."""
    return any((t_b > lo) and (t_a < hi) for lo, hi in zip(events_all.onset_lo, events_all.onset_hi))


def interval_noise_whole(dates, resid):
    """Robust sigma of single-interval differences of the full-model residual over the whole record, intervals
    overlapping an onset window excluded."""
    ok = np.isfinite(resid)
    d, r = dates[ok], resid[ok]
    inc = np.diff(r)
    keep = np.array([not is_event_interval(d[i], d[i + 1]) for i in range(len(d) - 1)])
    return utils.robust_sigma(inc[keep])


def stack_display(dates, crossings, onset_lo):
    """Mean over crossings of the displayed series, the mean full-model residual, and the mean step amp."""
    disp = np.array([display_series(c["v"], c["fit"], onset_lo) for c in crossings])
    res = np.array([c["fit"][2] for c in crossings])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(disp, axis=0), np.nanmean(res, axis=0)


# ------------------------------------------------------------------ per event
inc_rows, decay_rows, per_event = [], [], []
k_this = {round(t, 4): i for i, t in enumerate(step_epochs_all)}
order = np.argsort(km)
hov_idx = order[(km[order] >= HOV_KM[0] - 1.0) & (km[order] <= HOV_KM[1] + 1.0)]

for _, ev in events.iterrows():
    lo_day, hi_day = utils.decimal_year_to_day(ev.onset_lo), utils.decimal_year_to_day(ev.onset_hi)
    k_ev = k_this[round(ev.onset_lo, 4)]
    rel = np.asarray((epoch_days - lo_day).days, float)
    # row 1 stacks
    series = {}
    for key, label, c, colour, marker in LOCS:
        sel = np.where(np.abs(km - c) <= STACK_HALF_KM + 1e-6)[0]
        crossings = [dict(v=fp[r], fit=fp_fits[r]) for r in sel if fp_fits[r] is not None]
        assert len(crossings) == 3, f"{c} km: {len(crossings)} fitted crossings"
        y, res = stack_display(epochs, crossings, ev.onset_lo)
        series[key] = dict(y=y, res=res, amp=float(np.mean([cc["fit"][1][k_ev] for cc in crossings])))
    # row 2 per-track stacks
    tracks = {}
    for (key, trk), tr in track_rows.items():
        y, res = stack_display(tr["dates"], tr["crossings"], ev.onset_lo)
        d_days = pd.DatetimeIndex([utils.decimal_year_to_day(t) for t in tr["dates"]])
        tracks[(key, trk)] = dict(rel=np.asarray((d_days - lo_day).days, float), y=y, res=res, dates=tr["dates"],
                                  amp=float(np.mean([cc["fit"][1][k_ev] for cc in tr["crossings"]])),
                                  noise=interval_noise_whole(tr["dates"], res))
    # creepmeter
    cm_rel = np.asarray((cm_days - lo_day).days, float)
    cm_y = display_series(cm_slip, cm_fit, ev.onset_lo)
    # hovmoller
    hov = np.full((hov_idx.size, len(epochs)), np.nan)
    for j, r in enumerate(hov_idx):
        if fp_fits[r] is not None:
            hov[j] = display_series(fp[r], fp_fits[r], ev.onset_lo)
    for j in range(hov.shape[1]):
        hov[:, j] = utils.gaussian_smooth_km(hov[:, j], fp_ts.index.values[hov_idx], profile_smooth_km)
    amp = ef[amp_col(ev.onset_lo)].to_numpy(float)
    g_near = g_cfw1.assign(rel_start=(g_cfw1["Start Time"] - lo_day).dt.total_seconds() / 86400.0,
                           rel_end=(g_cfw1["End Time"] - lo_day).dt.total_seconds() / 86400.0)
    g_near = g_near[(g_near.rel_end >= -HALF_DAYS) & (g_near.rel_start <= HALF_DAYS)].sort_values("Start Time")
    covered = lo_day + pd.Timedelta(days=HALF_DAYS) <= g_cfw1_end

    # ---- per-interval increments on real dates: fp (both-real epochs only) and each track
    def increments(name, trk, dates, y, noise, amp_fit):
        ok = np.isfinite(y)
        d, v = dates[ok], y[ok]
        i_lo = int(np.searchsorted(d, ev.onset_lo + 1e-6)) - 1      # last date at or before onset_lo
        i_hi = int(np.searchsorted(d, ev.onset_hi - 1e-6))          # first date at or after onset_hi
        for i in range(max(i_lo - 3, 0), min(i_hi + 4, len(d) - 1)):
            a, b = d[i], d[i + 1]
            inc = v[i + 1] - v[i]
            inc_rows.append(dict(event=lo_day.year, series=name, track=trk,
                                 date_a=utils.decimal_year_to_day(a).date(), date_b=utils.decimal_year_to_day(b).date(),
                                 days=int(round((b - a) * 365.25)),
                                 rel_a=int((utils.decimal_year_to_day(a) - lo_day).days),
                                 rel_b=int((utils.decimal_year_to_day(b) - lo_day).days),
                                 increment_mm=inc, interval_noise_mm=noise, ratio=inc / noise,
                                 rises=bool(inc > NSIG_RISE * noise), joint_step_mm=amp_fit))
    for key, *_ in LOCS[1:3]:
        s = series[key]
        y_real = np.where(both_real, s["y"], np.nan)
        res_real = np.where(both_real, s["res"], np.nan)
        increments(key, "fp_both_real", epochs, y_real, interval_noise_whole(epochs, res_real), s["amp"])
    for (key, trk), t in tracks.items():
        increments(key, trk, t["dates"], t["y"], t["noise"], t["amp"])

    # ---- decay test: early post-event level vs late, on each series' own (real) dates
    def decay(name, trk, rel_d, y, res, dates):
        w_hi = (hi_day - lo_day).days
        early = (rel_d >= w_hi + DECAY_EARLY[0]) & (rel_d <= w_hi + DECAY_EARLY[1]) & np.isfinite(y)
        late = (rel_d >= DECAY_LATE[0]) & (rel_d <= DECAY_LATE[1]) & np.isfinite(y)
        sig = utils.robust_sigma(res)                              # full-model residual scatter, whole record
        if early.sum() == 0 or late.sum() == 0:                    # e.g. no both-real epochs (2017 common grid)
            decay_rows.append(dict(event=lo_day.year, series=name, track=trk, n_early=int(early.sum()),
                                   n_late=int(late.sum()), resid_sigma_mm=sig))
            return
        diff = np.mean(y[late]) - np.mean(y[early])
        se = sig * np.sqrt(1 / early.sum() + 1 / late.sum())       # assumes independent residuals (they are not)
        decay_rows.append(dict(event=lo_day.year, series=name, track=trk, n_early=int(early.sum()), n_late=int(late.sum()),
                               early_mean_mm=np.mean(y[early]), late_mean_mm=np.mean(y[late]),
                               late_minus_early_mm=diff, resid_sigma_mm=sig, se_independent_mm=se))
    for key, *_ in LOCS[1:3]:
        s = series[key]
        decay(key, "fp_both_real", rel, np.where(both_real, s["y"], np.nan), s["res"], epochs)
    for (key, trk), t in tracks.items():
        decay(key, trk, t["rel"], t["y"], t["res"], t["dates"])

    per_event.append(dict(lo_day=lo_day, hi_day=hi_day, rel=rel, series=series, tracks=tracks, cm_rel=cm_rel,
                          cm_y=cm_y, hov=hov, amp=amp, amp_smooth=utils.gaussian_smooth_km(amp, ef.lat.values, profile_smooth_km),
                          g_near=g_near, covered=covered))

inc_tab, decay_tab = pd.DataFrame(inc_rows), pd.DataFrame(decay_rows)
with open(fig_path + "_steps.csv", "w") as f:
    f.write(f"# {stamp}\n# QC only, not registered. Per-interval increments (mm, FP-equivalent) of the +/-{STACK_HALF_KM} km "
            f"stacks at 58.0 and 60.2 km on REAL dates: track=asc/desc = that track's own raw acquisitions (LOS / G_fp); "
            f"fp_both_real = common-grid FP at epochs where both tracks acquired. Detrended by the inter-event model "
            f"(04 design replicated; 7 events). interval_noise_mm = robust sigma of single-interval differences of the "
            f"full-model residual over the whole record, intervals overlapping an onset window excluded. "
            f"rises = increment > {NSIG_RISE} x noise.\n")
    inc_tab.to_csv(f, index=False, float_format="%.2f")
with open(fig_path + "_decay.csv", "w") as f:
    f.write(f"# {stamp}\n# QC only. late_minus_early = mean level {DECAY_LATE[0]}-{DECAY_LATE[1]} d after onset_lo minus mean "
            f"level {DECAY_EARLY[0]}-{DECAY_EARLY[1]} d after onset_hi (real dates only). se_independent assumes independent "
            f"residuals (they are correlated over weeks), so it is a LOWER bound.\n")
    decay_tab.to_csv(f, index=False, float_format="%.2f")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print(inc_tab.drop(columns=["date_a", "date_b"]).to_string(index=False, float_format=lambda x: f"{x:6.2f}"))
print(decay_tab.to_string(index=False, float_format=lambda x: f"{x:6.2f}"))

# hovmoller limit
vals = []
for pe in per_event:
    vis_t = (pe["rel"] >= -HALF_DAYS) & (pe["rel"] <= HALF_DAYS)
    vis_k = (km[hov_idx] >= HOV_KM[0]) & (km[hov_idx] <= HOV_KM[1])
    vals.append(np.abs(pe["hov"][vis_k][:, vis_t]).ravel())
clim = float(np.ceil(np.nanpercentile(np.concatenate(vals), CLIM_PERCENTILE)))
print(f"hovmoller limit +/-{clim:.0f} mm")

# ------------------------------------------------------------------ figure
utils.mpl_paper_style(base=7)
plt.rcParams.update({"axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "figure.facecolor": "white",
                     "savefig.facecolor": "white"})
W = 18.3 / 2.54
fig = plt.figure(figsize=(W, 8.4))
outer = fig.add_gridspec(2, 1, height_ratios=[3.25, 0.95], left=0.075, right=0.92, top=0.905, bottom=0.06, hspace=0.22)
top = outer[0].subgridspec(3, 5, height_ratios=[1.0, 0.85, 1.15], wspace=0.10, hspace=0.10)
bot = outer[1].subgridspec(1, 5, wspace=0.10)
ax_ts = [fig.add_subplot(top[0, c]) for c in range(5)]
ax_tr = [fig.add_subplot(top[1, c], sharex=ax_ts[c]) for c in range(5)]
ax_hv = [fig.add_subplot(top[2, c], sharex=ax_ts[c]) for c in range(5)]
ax_pr = [fig.add_subplot(bot[0, c]) for c in range(5)]
LETTERS = "ABCDEFGHIJKLMNOPQRST"
norm = Normalize(vmin=-clim, vmax=clim)
cmap = plt.get_cmap("PuOr_r")


def vis_range(rel_d, y):
    v = (rel_d >= -HALF_DAYS) & (rel_d <= HALF_DAYS) & np.isfinite(y)
    return (np.min(y[v]), np.max(y[v])) if v.any() else (0, 0)


lo1 = hi1 = lo2 = hi2 = pr_lo = pr_hi = 0.0
for pe in per_event:
    for s in pe["series"].values():
        a, b = vis_range(pe["rel"], s["y"]); lo1, hi1 = min(lo1, a), max(hi1, b)
    a, b = vis_range(pe["cm_rel"], pe["cm_y"]); lo1, hi1 = min(lo1, a), max(hi1, b)
    for t in pe["tracks"].values():
        a, b = vis_range(t["rel"], t["y"]); lo2, hi2 = min(lo2, a), max(hi2, b)
    kv = (ef_km >= HOV_KM[0]) & (ef_km <= HOV_KM[1])
    pr_lo, pr_hi = min(pr_lo, np.nanmin(pe["amp"][kv])), max(pr_hi, np.nanmax(pe["amp"][kv]))
N_LEVELS, LEVEL_MM = 3, 2.9
ts_ylim = (np.floor(lo1) - 0.5, np.ceil(hi1) + 0.5 + N_LEVELS * LEVEL_MM)
tr_ylim = (np.floor(lo2) - 0.5, np.ceil(hi2) + 0.5)
pr_ylim = (np.floor(pr_lo) - 0.5, np.ceil(pr_hi) + 0.5)


def letter(ax, k):
    ax.text(0.03, 0.97, LETTERS[k], transform=ax.transAxes, fontsize=8, fontweight="bold", va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.6))


for col, pe in enumerate(per_event):
    w_hi = (pe["hi_day"] - pe["lo_day"]).days
    # ---- row 1
    ax = ax_ts[col]
    ax.axvspan(0, w_hi, color="#E69F00", alpha=0.22, lw=0, zorder=0)
    ax.axhline(0, color="0.75", lw=0.4, zorder=1)
    cv = (pe["cm_rel"] >= -HALF_DAYS - 2) & (pe["cm_rel"] <= HALF_DAYS + 2)
    cx, cy = pe["cm_rel"][cv], pe["cm_y"][cv]
    gap = np.where(np.diff(cx) > 2)[0]
    ax.plot(np.insert(cx, gap + 1, np.nan), np.insert(cy, gap + 1, np.nan), zorder=4, **CM_STYLE)
    vis = (pe["rel"] >= -HALF_DAYS - 7) & (pe["rel"] <= HALF_DAYS + 7)
    for key, label, c, colour, marker in LOCS:
        y = pe["series"][key]["y"]; ok = vis & np.isfinite(y)
        ax.plot(pe["rel"][ok], y[ok], "-", color=colour, lw=0.6, zorder=3)
        r_ok, i_ok = ok & both_real, ok & ~both_real
        ax.plot(pe["rel"][r_ok], y[r_ok], ls="", marker=marker, ms=2.4, mew=0.4, mfc=colour,
                mec="white" if key != "km56.5" else colour, zorder=3.5)
        ax.plot(pe["rel"][i_ok], y[i_ok], ls="", marker=marker, ms=2.4, mew=0.5, mfc="white", mec=colour, zorder=3.5)
    days_per_inch = 2 * HALF_DAYS / (ax.get_position().width * fig.get_figwidth())
    label_w = 4 * 0.55 * 5.5 / 72 * days_per_inch
    occupied = [[] for _ in range(N_LEVELS)]
    for _, g in pe["g_near"].iterrows():
        x0, x1 = max(g.rel_start, -HALF_DAYS), min(g.rel_end, HALF_DAYS)
        right = x0 + 1 + label_w > HALF_DAYS
        span = (x0 - 1 - label_w, x1) if right else (x0, max(x1, x0 + 1 + label_w))
        level = next((k for k in range(N_LEVELS)
                      if all(span[1] + 1.5 < a or span[0] - 1.5 > b for a, b in occupied[k])), None)
        assert level is not None
        occupied[level].append(span)
        yb = ts_ylim[1] - 2.6 - LEVEL_MM * level
        ax.plot([x0, x1], [yb, yb], color=GITTINS_COLOUR, lw=1.1, solid_capstyle="butt", zorder=5)
        if g.rel_start >= -HALF_DAYS:
            ax.plot([g.rel_start] * 2, [yb - 0.45, yb + 0.45], color=GITTINS_COLOUR, lw=1.1, zorder=5)
        ax.text(x0 - 1 if right else x0 + 1, yb + 0.2, f"{g['Displacement, mm']:.2f}", color=GITTINS_TEXT,
                fontsize=5.5, ha="right" if right else "left", va="bottom", zorder=6, clip_on=True)
    if not pe["covered"]:
        ax.text(0.97, 0.97, f"catalogue ends\n{g_cfw1_end:%-d %b %Y}", transform=ax.transAxes, fontsize=5.5,
                color=GITTINS_TEXT, ha="right", va="top")
    ax.set_xlim(-HALF_DAYS, HALF_DAYS); ax.set_ylim(ts_ylim)
    ax.set_xticks([-30, 0, 30]); ax.set_xticks(np.arange(-60, 61, 15), minor=True); ax.tick_params(labelbottom=False)
    same_year = pe["lo_day"].year == pe["hi_day"].year
    dates_txt = (f"{pe['lo_day']:%-d %b} to {pe['hi_day']:%-d %b %Y}" if same_year
                 else f"{pe['lo_day']:%-d %b %Y} to {pe['hi_day']:%-d %b %Y}")
    utils.panel_title(ax, LETTERS[col], dates_txt, size=7)
    if col == 0:
        ax.set_ylabel("FP slip above\ninter-event model (mm)")
    else:
        ax.tick_params(labelleft=False)
    # ---- row 2: per-track, real dates only
    ax1 = ax_tr[col]
    ax1.axvspan(0, w_hi, color="#E69F00", alpha=0.22, lw=0, zorder=0)
    ax1.axhline(0, color="0.75", lw=0.4, zorder=1)
    for (key, trk), t in pe["tracks"].items():
        colour = dict((k, c) for k, _, c in TRACK_LOCS)[key]
        st = TRACK_STYLE[(key, trk)]
        ok = (t["rel"] >= -HALF_DAYS - 13) & (t["rel"] <= HALF_DAYS + 13) & np.isfinite(t["y"])
        ax1.plot(t["rel"][ok], t["y"][ok], color=colour, lw=0.6, ls=st["ls"], marker=st["marker"], ms=2.3,
                 mew=0.3, mfc=colour, mec="white", zorder=3)
    ax1.set_xlim(-HALF_DAYS, HALF_DAYS); ax1.set_ylim(tr_ylim)
    ax1.set_xticks([-30, 0, 30]); ax1.set_xticks(np.arange(-60, 61, 15), minor=True); ax1.tick_params(labelbottom=False)
    if col == 0:
        ax1.set_ylabel("Per-track LOS / G$_{FP}$\n(FP-equivalent, mm)")
    else:
        ax1.tick_params(labelleft=False)
    letter(ax1, 5 + col)
    # ---- row 3: hovmoller
    ax2 = ax_hv[col]
    x_edges = np.concatenate([[pe["rel"][0] - 3], 0.5 * (pe["rel"][1:] + pe["rel"][:-1]), [pe["rel"][-1] + 3]])
    kk = km[hov_idx]
    k_edges = np.concatenate([[kk[0] - 0.05], 0.5 * (kk[1:] + kk[:-1]), [kk[-1] + 0.05]])
    vis_cols = (pe["rel"] >= -HALF_DAYS - 15) & (pe["rel"] <= HALF_DAYS + 15)
    i0, i1 = np.where(vis_cols)[0][[0, -1]]
    mesh = ax2.pcolormesh(x_edges[i0:i1 + 2], k_edges, np.ma.masked_invalid(pe["hov"][:, i0:i1 + 1]), cmap=cmap,
                          norm=norm, shading="flat", rasterized=True)
    for xv in (0, w_hi):
        ax2.axvline(xv, color="k", ls="--", lw=0.6)
    ax2.set_ylim(HOV_KM[1], HOV_KM[0]); ax2.set_xlim(-HALF_DAYS, HALF_DAYS)
    ax2.set_yticks([l[2] for l in LOCS]); ax2.set_yticks(np.arange(54, 66, 1), minor=True)
    ax2.tick_params(axis="y", which="major", length=4, width=1.0)
    for tick, (_, _, _, colour, _) in zip(ax2.get_yticklines()[::2], LOCS):
        tick.set_color(colour if colour != "#7F7F7F" else "#555555")
    if col == 0:
        ax2.set_yticklabels([f"{l[2]:.1f}" for l in LOCS])
        for lab, (_, _, _, colour, _) in zip(ax2.get_yticklabels(), LOCS):
            lab.set_color(colour if colour != "#7F7F7F" else "#555555")
        ax2.set_ylabel("km from Point Pinole (N at top)")
    else:
        ax2.tick_params(labelleft=False)
    ax2.set_xlabel("Days from window start")
    letter(ax2, 10 + col)
    # ---- row 4: per-event profile
    ax3 = ax_pr[col]
    ax3.axhline(0, color="0.75", lw=0.4)
    for key, label, c, colour, marker in LOCS:
        ax3.axvline(c, color=colour if colour != "#7F7F7F" else "#555555", ls=":", lw=0.8)
    ax3.plot(ef_km, pe["amp"], "o", color="0.6", ms=1.3, mew=0, zorder=2)
    ax3.plot(ef_km, pe["amp_smooth"], "-", color="#000000", lw=0.9, zorder=3)
    ax3.set_xlim(HOV_KM); ax3.set_ylim(pr_ylim); ax3.set_xticks([54, 56, 58, 60, 62, 64])
    ax3.set_xlabel("km from Point Pinole")
    if col == 0:
        ax3.set_ylabel("Event slip (mm)")
    else:
        ax3.tick_params(labelleft=False)
    letter(ax3, 15 + col)

p = ax_hv[-1].get_position()
cax = fig.add_axes([0.932, p.y0, 0.011, p.height])
cb = fig.colorbar(mesh, cax=cax, extend="both")
cb_step = 5 if clim >= 10 else 2
cb.set_ticks(np.arange(-np.floor(clim / cb_step) * cb_step, clim + 0.01, cb_step))
cb.ax.tick_params(labelsize=6.5, width=0.5, length=2)
cb.set_label("FP slip above inter-event model (mm)", fontsize=7)

handles = [Line2D([], [], color=l[3], marker=l[4], ms=3, lw=0.6, mfc=l[3],
                  mec="white" if l[0] != "km56.5" else l[3], mew=0.4) for l in LOCS]
labels = [f"{l[2]:.1f} km FP" + (" (CFW1)" if l[0] == "cfw1" else "") for l in LOCS]
handles += [Line2D([], [], color="0.3", marker="o", ls="", ms=3, mfc="white", mec="0.3", mew=0.5),
            Line2D([], [], **CM_STYLE), Patch(color="#E69F00", alpha=0.22, lw=0),
            Line2D([], [], color=GITTINS_COLOUR, lw=1.1)]
labels += ["Open: one track interpolated", "CFW1 creepmeter (daily)", "InSAR onset window",
           "Gittins CFW1 event (slip, mm)"]
for key, c, colour in TRACK_LOCS:
    for trk in ("asc", "desc"):
        st = TRACK_STYLE[(key, trk)]
        handles.append(Line2D([], [], color=colour, lw=0.6, ls=st["ls"], marker=st["marker"], ms=3, mfc=colour,
                              mec="white", mew=0.3))
        labels.append(f"{c:.1f} km {'asc' if trk == 'asc' else 'desc'} LOS (F-J)")
handles += [Line2D([], [], color="0.6", marker="o", ls="", ms=2), Line2D([], [], color="k", lw=0.9)]
labels += ["Step per crossing (04)", f"Smoothed (σ {profile_smooth_km} km)"]
fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=6.0, frameon=False, bbox_to_anchor=(0.5, 0.997),
           handlelength=2.2, columnspacing=1.0, handletextpad=0.5)
fig.text(0.005, 0.003, stamp + "  |  QC only, not a paper figure; caption in qc05x_event_zoom_v3_caption.txt",
         fontsize=5, color="0.5", ha="left", va="bottom")
fig.savefig(fig_path + ".png", dpi=600, metadata={"Description": stamp})
fig.savefig(fig_path + ".pdf", metadata={"Subject": stamp})

caption = f"""qc05x_event_zoom_v3 -- QC only, NOT a paper figure; no numbers registered.
{stamp}

Five Tule Pond slow slip events, one column each; x = days from the window start (onset_lo, the last acquisition
before the step), -{HALF_DAYS} to +{HALF_DAYS} d. Epochs are 6-24 d apart: differences in onset or duration shorter
than one acquisition interval are NOT resolvable.

DETRENDING (all rows): the paper's inter-event model, fitted per crossing (and to the creepmeter's daily record) over
the whole record: constant + rate + annual sin/cos + one Heaviside step H(t > onset_lo) for each of the seven events of
notebook 04's creep_events.csv (five Tule Pond, two COZ1). 04's design() is replicated, not imported (04 runs on
import). Plotted = data - (constant + rate + annual) - the fitted steps of all EARLIER events, so zero is the model just
before this event and the event's own step remains. Stacks = mean of the per-crossing displayed series.

STACKS: crossings within +/-{STACK_HALF_KM} km along the trace (lat_to_km_map), 3 crossings. Each crossing already
averages pixels within +/-{x_width * 1000:.0f} m along strike, so neighbouring crossings share pixels and the effective
along-strike width of a stack is ~0.4-0.6 km regardless of the stack half-width.

A-E: fault-parallel slip at 56.5 km (N extent; black circles), 58.0 km (peak; vermillion squares), 60.2 km (peak; blue
triangles), {cfw1_km} km (CFW1, S extent; grey diamonds), on 03's common date grid. FILLED markers: both tracks acquired
that day; OPEN markers: one track is linearly interpolated there (03: across gaps up to 45 d; {(~real['asc']).mean() * 100:.0f} % of
epochs ascending, {(~real['desc']).mean() * 100:.0f} % descending, mostly before 2020). Thin black dashed: CFW1 creepmeter daily
mean, same model (line broken at gaps > 2 d). Shaded: InSAR onset window. Orange bars: Gittins et al. (2026) CFW1
catalogue events, start (tick) to end, slip in mm; catalogue ends {g_cfw1_end.date()}.

F-J: per-track LOS cross-fault series of the 58.0 and 60.2 km stacks on each track's OWN acquisition dates (no
interpolation): ascending track 35 solid (squares 58.0, up-triangles 60.2), descending track 42 dashed (diamonds 58.0,
down-triangles 60.2). LOS divided by G_FP, the LOS sensitivity to fault-parallel motion at each crossing (local incidence
and azimuth, local strike; utils.design_matrix_up_fp), i.e. FP-equivalent assuming no vertical motion; noise is
amplified by 1/|G_FP|. Same detrending.

K-O: hovmoller of the per-crossing displayed series, smoothed along strike (Gaussian sigma {profile_smooth_km} km) at each
epoch; linear symmetric scale +/-{clim:.0f} mm (= {CLIM_PERCENTILE}th percentile of |value|, rounded up). Dashed: window bounds.

P-T: along-strike profile of this event's slip: 04's per-crossing joint-model step (event_fraction_by_crossing.csv; grey
dots) and its smoothing (sigma {profile_smooth_km} km). Joint-model step noise (the COZ1 events' steps at the Tule Pond
crossings): {noise_single:.2f} mm single crossing, {noise_stack:.2f} mm for a +/-{STACK_HALF_KM} km stack.

Tables: qc05x_event_zoom_v3_steps.csv (per-interval increments on real dates, each track and the both-real FP epochs),
qc05x_event_zoom_v3_decay.csv (late minus early post-event level).
"""
with open(fig_path + "_caption.txt", "w") as f:
    f.write(caption)
print(f"wrote {fig_path}.png/.pdf/_steps.csv/_decay.csv/_caption.txt")
