#!/usr/bin/env python3
"""
QC 05x, version 2 (exploratory, NOT a paper figure; registers nothing, edits no notebook).
v1 (event_zoom.py, qc05x_event_zoom.*) is kept unchanged.

Zoom on the five Tule Pond slow slip events, +/-60 d about each InSAR onset window, Science style.
  Row 1  time series at four along-trace positions (stack of crossings within +/-0.2 km of each):
         56.5 km (northern extent), 58.0 and 60.2 km (the two slip peaks), CFW1 at 62.6 km (southern
         extent), plus the CFW1 creepmeter daily record. Shaded = InSAR onset window.
  Row 2  hovmoller, km 54-65 vs days, symmetric linear diverging colour scale centred on 0.
  Row 3  along-strike profile of THIS event's slip: notebook 04's per-crossing joint-model step
         (outputs/04_event_detection/event_fraction_by_crossing.csv, amp_<onset_lo>; the fit behind Fig 1C:
         constant + rate + annual + one Heaviside step per event of both sections, step at onset_lo).
         Dots per crossing, line = Gaussian along strike (sigma profile_smooth_km, as Fig 1C).

Detrending (rows 1-2 only): 04 does not save its per-crossing inter-event model, so each series is detrended
by a line fitted to its own values over the 120 d up to and including onset_lo (as v1).

Noise for the joint-model steps: the same fit carries the two Oakland (COZ1) events, which have no slip at
Tule Pond; their per-crossing steps over 54-65 km are an empirical noise sample (single crossing and
+/-0.2 km stack). Printed and written to the sidecar.

Outputs: fig_dir/qc05x_event_zoom_v2.png (600 dpi) + .pdf, _steps.csv, _caption.txt
Run: ~/miniforge3/envs/opera_disp_feb13/bin/python drafts/05x_event_zoom/event_zoom_v2.py
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
                                creepmeter_glitches, profile_smooth_km)

# ------------------------------------------------------------------ parameters
HALF_DAYS = 60            # x range: +/- this about onset_lo
PRE_FIT_DAYS = 120        # detrending line fitted over [onset_lo - this, onset_lo]
MIN_PRE_EPOCHS = 6
STACK_HALF_KM = 0.2       # each location = crossings within +/- this along the trace
HOV_KM = (54.0, 65.0)
CLIM_PERCENTILE = 98      # hovmoller limit = this percentile of |values| over all five panels, rounded up to 1 mm
CM_STYLE = dict(color="#000000", lw=0.6, ls=(0, (3, 1.5)))   # creepmeter: thin black dashed
CM_CODE = "CFW1"

fig_path = os.path.join(common_paths["fig_dir"], "qc05x_event_zoom_v2")
stamp = utils.provenance_stamp("drafts/05x_event_zoom/event_zoom_v2.py", REPO)

# ------------------------------------------------------------------ data
fp_ts = pd.read_csv(os.path.join(paths_out["03_creep_timeseries"], "fp_ts.csv"), index_col=0)
epochs = fp_ts.columns.values.astype(float)
epoch_days = pd.DatetimeIndex([utils.decimal_year_to_day(t) for t in epochs])
assert epoch_days.is_monotonic_increasing and epoch_days.is_unique
to_km = utils.lat_to_km_map(common_paths["fault_profiles"])
km = to_km(fp_ts.index.values)
fp = fp_ts.values

events = pd.read_csv(os.path.join(paths_out["04_event_detection"], "creep_events.csv"))
coz1_events = events[events.segment == "COZ1"].reset_index(drop=True)
events = events[events.segment == "TULEPOND"].reset_index(drop=True)
assert len(events) == 5 and len(coz1_events) == 2

# CFW1 along-trace position: nearest trace vertex to the instrument
prof = pd.read_csv(common_paths["fault_profiles"])
_, cfw1_lat, cfw1_lon, _ = [c for c in creepmeters if c[0] == CM_CODE][0]
dist_m = np.hypot((prof.ft_lat.values - cfw1_lat) * 111190.0,
                  (prof.ft_lon.values - cfw1_lon) * 111190.0 * np.cos(np.radians(cfw1_lat)))
cfw1_km = round((len(prof) - 1 - int(np.argmin(dist_m))) * 0.1, 1)
assert abs(cfw1_km - 62.6) < 0.05, f"CFW1 at {cfw1_km} km, expected 62.6"

# (key, label, km, colour, marker)
LOCS = [("km56.5", "56.5 km (N extent)",  56.5,    "#000000", "o"),
        ("km58.0", "58.0 km (peak)",      58.0,    "#D55E00", "s"),
        ("km60.2", "60.2 km (peak)",      60.2,    "#0072B2", "^"),
        ("cfw1",   f"{cfw1_km} km CFW1 (S extent)", cfw1_km, "#7F7F7F", "D")]

# 04's per-crossing joint-model steps
ef = pd.read_csv(os.path.join(paths_out["04_event_detection"], "event_fraction_by_crossing.csv")).sort_values("km")
ef_km = ef.km.to_numpy(float)
assert np.allclose(np.sort(to_km(ef.lat.values)), ef_km, atol=0.051), "04's km column disagrees with lat_to_km_map"
def amp_col(onset_lo):
    col = f"amp_{onset_lo:.3f}"
    assert col in ef.columns, f"no per-crossing step column {col} in event_fraction_by_crossing.csv"
    return col

# empirical noise of a joint-model step: the two COZ1 events' steps at the Tule Pond crossings (54-65 km)
in_tp = (ef_km >= HOV_KM[0]) & (ef_km <= HOV_KM[1])
null_single = np.concatenate([ef[amp_col(t)].to_numpy(float)[in_tp] for t in coz1_events.onset_lo])
null_stack = []
for t in coz1_events.onset_lo:
    a = ef[amp_col(t)].to_numpy(float)
    for c in np.arange(HOV_KM[0] + 0.2, HOV_KM[1] - 0.2 + 1e-9, 0.4):      # independent 0.4 km windows
        sel = np.abs(ef_km - c) <= STACK_HALF_KM + 1e-6
        if np.isfinite(a[sel]).sum() >= 3:
            null_stack.append(np.nanmean(a[sel]))
noise_single = utils.robust_sigma(null_single)
noise_stack = utils.robust_sigma(np.array(null_stack))
print(f"joint-model step noise (COZ1 events at 54-65 km): single crossing {noise_single:.2f} mm "
      f"(n={np.isfinite(null_single).sum()}), +/-{STACK_HALF_KM} km stack {noise_stack:.2f} mm (n={len(null_stack)})")

# Gittins et al. (2026) catalogue, CFW1 rows (drawn as in v1: start-to-end bars with slip labels)
gittins = pd.read_csv(common_paths["gittins_catalogue"], parse_dates=["Start Time", "End Time"])
g_cfw1 = gittins[gittins["Creepmeter abbreviation"] == CM_CODE].copy()
assert len(g_cfw1) > 0
g_cfw1_end = g_cfw1["End Time"].max()
GITTINS_COLOUR, GITTINS_TEXT = "#E69F00", "#9A6400"

# creepmeter
t_cm0 = utils.date_to_decimal_year((epoch_days[0] - pd.Timedelta(days=200)).strftime("%Y-%m-%d"))
t_cm1 = utils.date_to_decimal_year((epoch_days[-1] + pd.Timedelta(days=100)).strftime("%Y-%m-%d"))
cm_t, cm_slip = utils.daily_creepmeter(common_paths["creepmeter_dir"], CM_CODE, t_cm0, t_cm1, creepmeter_glitches)
cm_days = pd.DatetimeIndex([pd.Timestamp(utils.decimal_year_to_date(t)).floor("D") for t in cm_t])


def detrend_pre(rel_days, y):
    pre = (rel_days >= -PRE_FIT_DAYS) & (rel_days <= 0) & np.isfinite(y)
    if pre.sum() < MIN_PRE_EPOCHS:
        return np.full_like(y, np.nan, dtype=float)
    b = np.polyfit(rel_days[pre], y[pre], 1)
    return y - np.polyval(b, rel_days)


def interval_noise(rel_days, y):
    pre = (rel_days >= -PRE_FIT_DAYS) & (rel_days <= 0) & np.isfinite(y)
    return utils.robust_sigma(np.diff(y[pre]))


# ------------------------------------------------------------------ per event
rows, per_event = [], []
for _, ev in events.iterrows():
    lo_day, hi_day = utils.decimal_year_to_day(ev.onset_lo), utils.decimal_year_to_day(ev.onset_hi)
    i_lo = int(np.where(epoch_days == lo_day)[0][0]); i_hi = int(np.where(epoch_days == hi_day)[0][0])
    rel = np.asarray((epoch_days - lo_day).days, float)
    in_x = (rel >= -PRE_FIT_DAYS) & (rel <= HALF_DAYS)
    det = np.full_like(fp, np.nan)
    for r in range(fp.shape[0]):
        det[r, in_x] = detrend_pre(rel[in_x], fp[r, in_x])
    amp = ef[amp_col(ev.onset_lo)].to_numpy(float)
    series = {}
    for key, label, c, colour, marker in LOCS:
        sel = (np.abs(km - c) <= STACK_HALF_KM + 1e-6) & np.isfinite(det[:, i_lo])
        assert sel.sum() >= 3, f"{lo_day.date()}: only {sel.sum()} crossings at {c} km"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            y = np.nanmean(det[sel], axis=0)
        sel_ef = np.abs(ef_km - c) <= STACK_HALF_KM + 1e-6
        a = amp[sel_ef]
        series[key] = dict(y=y, n=int(sel.sum()), joint=float(np.nanmean(a)), joint_n=int(np.isfinite(a).sum()),
                           joint_sd=float(np.nanstd(a, ddof=1)))
    cm_rel = np.asarray((cm_days - lo_day).days, float)
    cm_in = (cm_rel >= -PRE_FIT_DAYS) & (cm_rel <= HALF_DAYS)
    cm_det = detrend_pre(cm_rel[cm_in], cm_slip[cm_in])
    order = np.argsort(km)
    hov_idx = order[(km[order] >= HOV_KM[0] - 1.0) & (km[order] <= HOV_KM[1] + 1.0)]
    hov = det[hov_idx][:, in_x].copy()
    for j in range(hov.shape[1]):
        hov[:, j] = utils.gaussian_smooth_km(hov[:, j], fp_ts.index.values[hov_idx], profile_smooth_km)
    amp_smooth = utils.gaussian_smooth_km(amp, ef.lat.values, profile_smooth_km)
    g_near = g_cfw1.assign(rel_start=(g_cfw1["Start Time"] - lo_day).dt.total_seconds() / 86400.0,
                           rel_end=(g_cfw1["End Time"] - lo_day).dt.total_seconds() / 86400.0)
    g_near = g_near[(g_near.rel_end >= -HALF_DAYS) & (g_near.rel_start <= HALF_DAYS)].sort_values("Start Time")
    assert g_near["Event_id"].notna().all()
    covered = lo_day + pd.Timedelta(days=HALF_DAYS) <= g_cfw1_end
    print(f"{lo_day.date()}: {len(g_near)} Gittins CFW1 events within +/-{HALF_DAYS} d"
          + ("" if covered else f" (catalogue ends {g_cfw1_end.date()})"))

    e_idx = {"Em2": i_lo - 2, "Em1": i_lo - 1, "E0": i_lo, "E1": i_hi, "E2": i_hi + 1, "E3": i_hi + 2}
    e_day = {k: epoch_days[v] for k, v in e_idx.items()}
    for key, label, c, *_ in LOCS:
        y = series[key]["y"]; v = {k: y[i] for k, i in e_idx.items()}
        rows.append(dict(event=lo_day.year, series=key, label=label, km=c, n_crossings=series[key]["n"],
                         **{k: e_day[k].date() for k in e_idx},
                         step_before2_mm=v["Em1"] - v["Em2"], step_before_mm=v["E0"] - v["Em1"],
                         step_window_mm=v["E1"] - v["E0"], step_after_mm=v["E2"] - v["E1"],
                         step_after2_mm=v["E3"] - v["E2"], interval_noise_mm=interval_noise(rel[in_x], y[in_x]),
                         joint_model_step_mm=series[key]["joint"], joint_model_n=series[key]["joint_n"],
                         joint_model_sd_across_crossings_mm=series[key]["joint_sd"]))
    cm_val = {}
    for k, d in e_day.items():
        m = np.where(cm_days[cm_in] == d)[0]
        cm_val[k] = cm_det[m[0]] if m.size else np.nan
    rows.append(dict(event=lo_day.year, series="cfw1_creepmeter", label="CFW1 creepmeter (daily mean)", km=cfw1_km,
                     n_crossings=0, **{k: e_day[k].date() for k in e_idx},
                     step_before2_mm=cm_val["Em1"] - cm_val["Em2"], step_before_mm=cm_val["E0"] - cm_val["Em1"],
                     step_window_mm=cm_val["E1"] - cm_val["E0"], step_after_mm=cm_val["E2"] - cm_val["E1"],
                     step_after2_mm=cm_val["E3"] - cm_val["E2"], interval_noise_mm=np.nan,
                     joint_model_step_mm=np.nan, joint_model_n=0, joint_model_sd_across_crossings_mm=np.nan))
    per_event.append(dict(lo_day=lo_day, hi_day=hi_day, rel=rel, in_x=in_x, series=series, cm_rel=cm_rel[cm_in],
                          cm_det=cm_det, hov=hov, hov_km=km[hov_idx], amp=amp, amp_smooth=amp_smooth,
                          g_near=g_near, covered=covered))

table = pd.DataFrame(rows)
with open(fig_path + "_steps.csv", "w") as f:
    f.write(f"# {stamp}\n# QC only, not registered. step_* (mm): detrended series (line over the {PRE_FIT_DAYS} d up to "
            f"onset_lo) between fp_ts epochs E-2..E3 (E0 = onset_lo, E1 = onset_hi). joint_model_step_mm: mean over the "
            f"crossings within +/-{STACK_HALF_KM} km of notebook 04's per-crossing joint-model step "
            f"(event_fraction_by_crossing.csv). Joint-model step noise (COZ1 events at 54-65 km): single crossing "
            f"{noise_single:.2f} mm, +/-{STACK_HALF_KM} km stack {noise_stack:.2f} mm.\n")
    table.to_csv(f, index=False, float_format="%.2f")
pd.set_option("display.width", 250)
print(table[["event", "series", "n_crossings", "step_before_mm", "step_window_mm", "step_after_mm",
             "interval_noise_mm", "joint_model_step_mm", "joint_model_sd_across_crossings_mm"]]
      .to_string(index=False, float_format=lambda x: f"{x:6.2f}"))

# hovmoller limit from the data: CLIM_PERCENTILE of |values| in the plotted region, rounded up to 1 mm
vals = []
for pe in per_event:
    xs = pe["rel"][pe["in_x"]]
    vis_t = (xs >= -HALF_DAYS) & (xs <= HALF_DAYS)
    vis_k = (pe["hov_km"] >= HOV_KM[0]) & (pe["hov_km"] <= HOV_KM[1])
    vals.append(np.abs(pe["hov"][vis_k][:, vis_t]).ravel())
vals = np.concatenate(vals)
clim = float(np.ceil(np.nanpercentile(vals, CLIM_PERCENTILE)))
print(f"hovmoller |value| {CLIM_PERCENTILE}th percentile -> limit +/-{clim:.0f} mm")

# ------------------------------------------------------------------ figure (Science: 18.3 cm, 7 pt)
utils.mpl_paper_style(base=7)
plt.rcParams.update({"axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "figure.facecolor": "white",
                     "savefig.facecolor": "white"})
W = 18.3 / 2.54
fig = plt.figure(figsize=(W, 6.3))
outer = fig.add_gridspec(2, 1, height_ratios=[2.3, 0.95], left=0.075, right=0.92, top=0.905, bottom=0.075,
                         hspace=0.30)
top = outer[0].subgridspec(2, 5, height_ratios=[1.0, 1.25], wspace=0.10, hspace=0.10)
bot = outer[1].subgridspec(1, 5, wspace=0.10)
ax_ts = [fig.add_subplot(top[0, c]) for c in range(5)]
ax_hv = [fig.add_subplot(top[1, c], sharex=ax_ts[c]) for c in range(5)]
ax_pr = [fig.add_subplot(bot[0, c]) for c in range(5)]
LETTERS = "ABCDEFGHIJKLMNO"                                 # row-major: A-E row 1, F-J row 2, K-O row 3
norm = Normalize(vmin=-clim, vmax=clim)
cmap = plt.get_cmap("PuOr_r")

# common y ranges
ts_lo, ts_hi, pr_lo, pr_hi = 0.0, 0.0, 0.0, 0.0
for pe in per_event:
    vis = pe["in_x"] & (pe["rel"] >= -HALF_DAYS) & (pe["rel"] <= HALF_DAYS)
    for s in pe["series"].values():
        ts_lo, ts_hi = min(ts_lo, np.nanmin(s["y"][vis])), max(ts_hi, np.nanmax(s["y"][vis]))
    cv = (pe["cm_rel"] >= -HALF_DAYS) & (pe["cm_rel"] <= HALF_DAYS)
    ts_lo, ts_hi = min(ts_lo, np.nanmin(pe["cm_det"][cv])), max(ts_hi, np.nanmax(pe["cm_det"][cv]))
    kv = (ef_km >= HOV_KM[0]) & (ef_km <= HOV_KM[1])
    pr_lo, pr_hi = min(pr_lo, np.nanmin(pe["amp"][kv])), max(pr_hi, np.nanmax(pe["amp"][kv]))
N_LEVELS, LEVEL_MM = 3, 2.9                                  # catalogue strip above the data
ts_ylim = (np.floor(ts_lo) - 0.5, np.ceil(ts_hi) + 0.5 + N_LEVELS * LEVEL_MM)
pr_ylim = (np.floor(pr_lo) - 0.5, np.ceil(pr_hi) + 0.5)

def letter(ax, k):
    ax.text(0.03, 0.97, LETTERS[k], transform=ax.transAxes, fontsize=8, fontweight="bold", va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.6))

for col, pe in enumerate(per_event):
    w_hi = (pe["hi_day"] - pe["lo_day"]).days
    # row 1
    ax = ax_ts[col]
    ax.axvspan(0, w_hi, color="#E69F00", alpha=0.22, lw=0, zorder=0)
    ax.axhline(0, color="0.75", lw=0.4, zorder=1)
    cm_x, cm_y = pe["cm_rel"].astype(float), pe["cm_det"].astype(float)
    gap = np.where(np.diff(cm_x) > 2)[0]                     # break the line at record gaps > 2 d
    ax.plot(np.insert(cm_x, gap + 1, np.nan), np.insert(cm_y, gap + 1, np.nan), zorder=4, **CM_STYLE)
    vis = pe["in_x"] & (pe["rel"] >= -HALF_DAYS - 7) & (pe["rel"] <= HALF_DAYS + 7)
    for key, label, c, colour, marker in LOCS:
        y = pe["series"][key]["y"]; ok = vis & np.isfinite(y)
        ax.plot(pe["rel"][ok], y[ok], "-", color=colour, lw=0.7, marker=marker, ms=2.4, mew=0.4,
                mfc=colour, mec="white" if key != "km56.5" else colour, zorder=3)
    # Gittins CFW1 events: tick at start, bar start -> end, slip label; greedy stagger over N_LEVELS rows
    # so no two labels/bars overlap (label width estimated in days from the panel width and font size)
    days_per_inch = 2 * HALF_DAYS / (ax.get_position().width * fig.get_figwidth())
    label_w = 4 * 0.55 * 5.5 / 72 * days_per_inch            # 4 characters at 5.5 pt
    occupied = [[] for _ in range(N_LEVELS)]
    for _, g in pe["g_near"].iterrows():
        x0, x1 = max(g.rel_start, -HALF_DAYS), min(g.rel_end, HALF_DAYS)
        right = x0 + 1 + label_w > HALF_DAYS                  # label would leave the panel: put it left of the tick
        span = (x0 - 1 - label_w, x1) if right else (x0, max(x1, x0 + 1 + label_w))
        level = next((k for k in range(N_LEVELS)
                      if all(span[1] + 1.5 < a or span[0] - 1.5 > b for a, b in occupied[k])), None)
        assert level is not None, f"{pe['lo_day'].date()}: Gittins labels need more than {N_LEVELS} rows"
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
        ax.set_ylabel("Slip above pre-window\ntrend (mm)")
    else:
        ax.tick_params(labelleft=False)
    # row 2
    ax2 = ax_hv[col]
    rel_x = pe["rel"][pe["in_x"]]
    x_edges = np.concatenate([[rel_x[0] - 3], 0.5 * (rel_x[1:] + rel_x[:-1]), [rel_x[-1] + 3]])
    kk = pe["hov_km"]
    k_edges = np.concatenate([[kk[0] - 0.05], 0.5 * (kk[1:] + kk[:-1]), [kk[-1] + 0.05]])
    mesh = ax2.pcolormesh(x_edges, k_edges, np.ma.masked_invalid(pe["hov"]), cmap=cmap, norm=norm,
                          shading="flat", rasterized=True)
    for xv in (0, w_hi):
        ax2.axvline(xv, color="k", ls="--", lw=0.6)
    ax2.set_ylim(HOV_KM[1], HOV_KM[0])
    ax2.set_xlim(-HALF_DAYS, HALF_DAYS)
    ax2.set_yticks([l[2] for l in LOCS])
    ax2.set_yticks(np.arange(54, 66, 1), minor=True)
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
    letter(ax2, 5 + col)
    # row 3
    ax3 = ax_pr[col]
    ax3.axhline(0, color="0.75", lw=0.4)
    for key, label, c, colour, marker in LOCS:
        ax3.axvline(c, color=colour if colour != "#7F7F7F" else "#555555", ls=":", lw=0.8)
    ax3.plot(ef_km, pe["amp"], "o", color="0.6", ms=1.3, mew=0, zorder=2)
    ax3.plot(ef_km, pe["amp_smooth"], "-", color="#000000", lw=0.9, zorder=3)
    ax3.set_xlim(HOV_KM); ax3.set_ylim(pr_ylim)
    ax3.set_xticks([54, 56, 58, 60, 62, 64])
    ax3.set_xlabel("km from Point Pinole")
    if col == 0:
        ax3.set_ylabel("Event slip (mm)")
    else:
        ax3.tick_params(labelleft=False)
    letter(ax3, 10 + col)

# colour bar right of row 2
p = ax_hv[-1].get_position()
cax = fig.add_axes([0.932, p.y0, 0.011, p.height])
cb = fig.colorbar(mesh, cax=cax, extend="both")
cb_step = 5 if clim >= 10 else 2
cb.set_ticks(np.arange(-np.floor(clim / cb_step) * cb_step, clim + 0.01, cb_step))
cb.ax.tick_params(labelsize=6.5, width=0.5, length=2)
cb.set_label("Slip above pre-window trend (mm)", fontsize=7)

handles = [Line2D([], [], color=l[3], marker=l[4], ms=3, lw=0.7, mfc=l[3],
                  mec="white" if l[0] != "km56.5" else l[3], mew=0.4) for l in LOCS]
labels = [l[1] for l in LOCS]
handles += [Line2D([], [], **CM_STYLE), Patch(color="#E69F00", alpha=0.22, lw=0),
            Line2D([], [], color=GITTINS_COLOUR, lw=1.1),
            Line2D([], [], color="0.6", marker="o", ls="", ms=2), Line2D([], [], color="k", lw=0.9)]
labels += ["CFW1 creepmeter (daily)", "InSAR onset window", "Gittins et al. (2026) CFW1 event (slip, mm)", "Event step per crossing (04 joint model)",
           f"Smoothed (σ {profile_smooth_km} km)"]
fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 0.995),
           handlelength=2.2, columnspacing=1.2, handletextpad=0.5)
fig.text(0.005, 0.004, stamp + "  |  QC only, not a paper figure; caption in qc05x_event_zoom_v2_caption.txt",
         fontsize=5, color="0.5", ha="left", va="bottom")

fig.savefig(fig_path + ".png", dpi=600, metadata={"Description": stamp})
fig.savefig(fig_path + ".pdf", metadata={"Subject": stamp})

caption = f"""qc05x_event_zoom_v2 -- QC only, NOT a paper figure; no numbers registered.
{stamp}

Five Tule Pond slow slip events, one column each, x = days from the window start (onset_lo, the last acquisition
before the step), -{HALF_DAYS} to +{HALF_DAYS} d. fp_ts epochs are 6 or 12 d apart: differences in onset shorter
than one acquisition interval are NOT resolvable; locations that step in the same interval are simultaneous at
this sampling.

A-E: fault-parallel slip, mean of the crossings within +/-{STACK_HALF_KM} km along the trace (lat_to_km_map) of
56.5 km (northern extent; black circles), 58.0 km (peak; vermillion squares), 60.2 km (peak; blue triangles) and
{cfw1_km} km (CFW1, southern extent; grey diamonds), and the CFW1 creepmeter daily mean (thin black dashed; line
broken at record gaps > 2 d; days in the reader's day convention, pending day-of-year check may shift by 1 d).
Each series is detrended by a straight line fitted over the {PRE_FIT_DAYS} d up to and including onset_lo (04 does
not save its per-crossing inter-event model); zero = that line extrapolated, so levels drift late in the window.
No offsets between curves. Shaded: InSAR onset window (onset_lo to onset_hi). Orange bars: CFW1 events of the
Gittins et al. (2026) catalogue overlapping the x range, start (tick) to end, labelled with their slip (mm),
staggered over up to {N_LEVELS} rows; the catalogue ends {g_cfw1_end.date()} (no entries for the 2024 window).

F-J: hovmoller, km 54-65 (north at top) vs days; each crossing detrended as above, then smoothed along strike
(Gaussian sigma {profile_smooth_km} km) at each epoch; cells span half-way to the neighbouring epochs. Linear,
symmetric colour scale +/-{clim:.0f} mm (= the {CLIM_PERCENTILE}th percentile of |value| over the five panels,
rounded up to 1 mm). Dashed: window bounds. Coloured ticks on the y axis: the four locations of A-E.

K-O: along-strike profile of THIS event's slip (not cumulative): notebook 04's per-crossing joint-model step
(event_fraction_by_crossing.csv, the fit behind Fig 1C: constant + rate + annual + one step per event of both
sections, step at onset_lo; grey dots), and its Gaussian smoothing along strike (sigma {profile_smooth_km} km,
black). Same y scale in all five. Dotted: the four locations. Noise of a joint-model step, from the two Oakland
(COZ1) events' steps at the Tule Pond crossings (no slip expected there): {noise_single:.2f} mm for a single
crossing, {noise_stack:.2f} mm for a +/-{STACK_HALF_KM} km stack (robust sigma).

Table: qc05x_event_zoom_v2_steps.csv (window and neighbouring-interval steps of A-E's series, joint-model step at
each location).
"""
with open(fig_path + "_caption.txt", "w") as f:
    f.write(caption)
print(f"wrote {fig_path}.png/.pdf/_steps.csv/_caption.txt")
