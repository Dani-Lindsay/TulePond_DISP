#!/usr/bin/env python3
"""
QC 05x (exploratory, NOT a paper figure; registers nothing, edits no notebook):
zoom on the five Tule Pond slow slip events, +/-90 d about each InSAR onset window.

Question: did the events start in one place and propagate along strike (e.g. north -> south toward
CFW1), or are the small CFW1 creepmeter events near the windows unrelated?

Inputs (read-only, from the repo's outputs):
  outputs/03_creep_timeseries/fp_ts.csv           fault-parallel cross-fault slip, crossing (lat) x epoch (decyear), mm
  outputs/04_event_detection/creep_events.csv     onset_lo (last epoch before the step), onset_hi (first epoch after)
  CFW1 creepmeter daily record                    utils.daily_creepmeter + creepmeter_glitches
  Gittins et al. (2026) catalogue                 every CFW1 event starting within +/-90 d of onset_lo

Detrending: notebook 04 does not save its per-crossing inter-event model, so every series (each crossing,
and the creepmeter) is detrended here by a straight line fitted to its own values in the 120 d up to and
including onset_lo. Zero on every panel = that pre-window line extrapolated forward. The extrapolation
drifts with time after the window, so late-window levels (> ~+30 d) carry the error of the line and of the
annual cycle the line does not model.

Along-strike km: utils.lat_to_km_map on the fault-profile trace (km from Point Pinole, + south).
CFW1's km: the nearest trace vertex to the instrument location (distance, not latitude).

Outputs (fig_dir from TulePond_filepaths):
  qc05x_event_zoom.png (600 dpi) + .pdf, qc05x_event_zoom_steps.csv

Run: ~/miniforge3/envs/opera_disp_feb13/bin/python drafts/05x_event_zoom/event_zoom.py
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
import insar_utils as utils                                           # noqa: E402
from TulePond_filepaths import (common_paths, paths_out, creepmeters,  # noqa: E402
                                creepmeter_glitches, profile_smooth_km)

# ------------------------------------------------------------------ parameters
HALF_DAYS = 90            # x range: +/- this about onset_lo
PRE_FIT_DAYS = 120        # detrending line fitted over [onset_lo - this, onset_lo]
MIN_PRE_EPOCHS = 6        # a crossing needs this many finite epochs in the pre-window fit
BLOCKS = [                # (key, label, km_lo, km_hi, colour); CFW1 block filled in below from CFW1's km
    ("north", "N end 56.2-57.5 km", 56.2, 57.5, "#000000"),
    ("peak",  "peak 58.2-59.8 km",  58.2, 59.8, "#D55E00"),
    ("south", "S end 61.0-62.4 km", 61.0, 62.4, "#0072B2"),
]
CFW1_HALF_KM = 0.2
HOV_KM = (54.0, 65.0)
CM_COLOUR = "#7F7F7F"     # creepmeter: grey
GITTINS_COLOUR = "#E69F00"
WINDOW_SHADE = "#E69F00"
SINGLE_SITE_NOISE_MM = 1.7  # quoted single-crossing noise level (caller)

fig_path = os.path.join(common_paths["fig_dir"], "qc05x_event_zoom")
table_path = fig_path + "_steps.csv"
stamp = utils.provenance_stamp("drafts/05x_event_zoom/event_zoom.py", REPO)

# ------------------------------------------------------------------ data
fp_ts = pd.read_csv(os.path.join(paths_out["03_creep_timeseries"], "fp_ts.csv"), index_col=0)
epochs = fp_ts.columns.values.astype(float)
epoch_days = pd.DatetimeIndex([utils.decimal_year_to_day(t) for t in epochs])
assert epoch_days.is_monotonic_increasing and epoch_days.is_unique, "fp_ts epochs must be unique, increasing days"
to_km = utils.lat_to_km_map(common_paths["fault_profiles"])
km = to_km(fp_ts.index.values)
fp = fp_ts.values

events = pd.read_csv(os.path.join(paths_out["04_event_detection"], "creep_events.csv"))
events = events[events.segment == "TULEPOND"].reset_index(drop=True)
assert len(events) == 5, f"expected 5 Tule Pond events, got {len(events)}"

# CFW1 along-strike position: nearest trace vertex to the instrument, km = (n-1-index) * 0.1 (lat_to_km_map's rule)
prof = pd.read_csv(common_paths["fault_profiles"])
cfw1 = [c for c in creepmeters if c[0] == "CFW1"][0]
_, cfw1_lat, cfw1_lon, _ = cfw1
dist_m = np.hypot((prof.ft_lat.values - cfw1_lat) * 111190.0,
                  (prof.ft_lon.values - cfw1_lon) * 111190.0 * np.cos(np.radians(cfw1_lat)))
i_near = int(np.argmin(dist_m))
cfw1_km = (len(prof) - 1 - i_near) * 0.1
print(f"CFW1: {cfw1_km:.1f} km from Point Pinole, {dist_m[i_near]:.0f} m from the nearest trace vertex")
assert abs(cfw1_km - 62.6) < 0.3, f"CFW1 at {cfw1_km:.1f} km, caller expected ~62.6"
BLOCKS.append(("cfw1", f"CFW1 crossings {cfw1_km - CFW1_HALF_KM:.1f}-{cfw1_km + CFW1_HALF_KM:.1f} km",
               cfw1_km - CFW1_HALF_KM, cfw1_km + CFW1_HALF_KM, "#CC79A7"))

# CFW1 creepmeter, daily, over the whole span needed
t_cm0 = utils.date_to_decimal_year((epoch_days[0] - pd.Timedelta(days=200)).strftime("%Y-%m-%d"))
t_cm1 = utils.date_to_decimal_year((epoch_days[-1] + pd.Timedelta(days=100)).strftime("%Y-%m-%d"))
cm_t, cm_slip = utils.daily_creepmeter(common_paths["creepmeter_dir"], "CFW1", t_cm0, t_cm1, creepmeter_glitches)
cm_days = pd.DatetimeIndex([pd.Timestamp(utils.decimal_year_to_date(t)).floor("D") for t in cm_t])   # reader's day convention

# Gittins et al. (2026) catalogue, CFW1 rows
gittins = pd.read_csv(common_paths["gittins_catalogue"], parse_dates=["Start Time", "End Time"])
g_cfw1 = gittins[gittins["Creepmeter abbreviation"] == "CFW1"].copy()
assert len(g_cfw1) > 0, "no CFW1 events in the Gittins catalogue"
g_cfw1_end = g_cfw1["End Time"].max()
print(f"Gittins CFW1: {len(g_cfw1)} events, last ends {g_cfw1_end}")
near_file = os.path.join(paths_out["05_validation"], "cfw1_gittins_near_windows.csv")
near_05 = pd.read_csv(near_file)


# ------------------------------------------------------------------ helpers
def detrend_pre(rel_days, y):
    """Subtract the line fitted to y over rel_days in [-PRE_FIT_DAYS, 0]. NaN if too few points."""
    pre = (rel_days >= -PRE_FIT_DAYS) & (rel_days <= 0) & np.isfinite(y)
    if pre.sum() < MIN_PRE_EPOCHS:
        return np.full_like(y, np.nan, dtype=float)
    b = np.polyfit(rel_days[pre], y[pre], 1)
    return y - np.polyval(b, rel_days)


def interval_noise(rel_days, y):
    """Robust sigma of single-interval differences of the detrended series in the pre-window period."""
    pre = (rel_days >= -PRE_FIT_DAYS) & (rel_days <= 0) & np.isfinite(y)
    return utils.robust_sigma(np.diff(y[pre]))


# ------------------------------------------------------------------ per event
rows = []
per_event = []
for ev_i, ev in events.iterrows():
    lo_day = utils.decimal_year_to_day(ev.onset_lo)
    hi_day = utils.decimal_year_to_day(ev.onset_hi)
    i_lo = int(np.where(epoch_days == lo_day)[0][0])
    i_hi = int(np.where(epoch_days == hi_day)[0][0])
    assert i_hi > i_lo, f"event {ev_i}: onset_hi epoch not after onset_lo"
    rel = np.asarray((epoch_days - lo_day).days, float)
    in_x = (rel >= -PRE_FIT_DAYS) & (rel <= HALF_DAYS)
    # detrend every crossing
    det = np.full_like(fp, np.nan)
    for r in range(fp.shape[0]):
        det[r, in_x] = detrend_pre(rel[in_x], fp[r, in_x])
    # blocks: mean of detrended crossings
    blocks = {}
    for key, label, k0, k1, colour in BLOCKS:
        sel = (km >= k0 - 1e-6) & (km <= k1 + 1e-6) & np.isfinite(det[:, i_lo])
        assert sel.sum() > 0, f"{ev.onset_lo}: no crossings in block {key}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(det[sel], axis=0)
        blocks[key] = dict(y=mean, n=int(sel.sum()), label=label, colour=colour)
    # creepmeter
    cm_rel = np.asarray((cm_days - lo_day).days, float)
    cm_in = (cm_rel >= -PRE_FIT_DAYS) & (cm_rel <= HALF_DAYS)
    cm_det = detrend_pre(cm_rel[cm_in], cm_slip[cm_in])
    # hovmoller rows: crossings in HOV_KM, smoothed along strike at each epoch (gap-preserving)
    order = np.argsort(km)
    in_hov = (km[order] >= HOV_KM[0] - 1.0) & (km[order] <= HOV_KM[1] + 1.0)
    hov_idx = order[in_hov]
    hov = det[hov_idx][:, in_x].copy()
    for j in range(hov.shape[1]):
        hov[:, j] = utils.gaussian_smooth_km(hov[:, j], fp_ts.index.values[hov_idx], profile_smooth_km)
    # Gittins CFW1 events within +/-HALF_DAYS of onset_lo (start or end inside the x range)
    g_rel_start = (g_cfw1["Start Time"] - lo_day).dt.total_seconds() / 86400.0
    g_rel_end = (g_cfw1["End Time"] - lo_day).dt.total_seconds() / 86400.0
    # assign BEFORE filtering: assigning a Series to an empty frame would adopt the Series' whole index
    g_near = g_cfw1.assign(rel_start=g_rel_start, rel_end=g_rel_end)
    g_near = g_near[(g_near.rel_end >= -HALF_DAYS) & (g_near.rel_start <= HALF_DAYS)]
    assert g_near["Event_id"].notna().all(), "Gittins rows lost their ids"
    assert ((g_near.rel_end >= -HALF_DAYS) & (g_near.rel_start <= HALF_DAYS)).all()
    covered = lo_day + pd.Timedelta(days=HALF_DAYS) <= g_cfw1_end
    print(f"\n{lo_day.date()} to {hi_day.date()}: Gittins CFW1 events within +/-{HALF_DAYS} d: {len(g_near)}"
          + ("" if covered else f" (catalogue ends {g_cfw1_end.date()}: window not fully covered)"))
    for _, g in g_near.iterrows():
        print(f"   {g['Event_id']}: {g['Start Time']} to {g['End Time']} ({g.rel_start:+.1f} to {g.rel_end:+.1f} d), "
              f"{g['Displacement, mm']:.2f} mm")

    # the step table: E-2 -> E-1 and E-1 -> E0 (the two intervals before), E0 -> E1 (the InSAR window),
    # E1 -> E2 and E2 -> E3 (the two intervals after)
    e_idx = {"Em2": i_lo - 2, "Em1": i_lo - 1, "E0": i_lo, "E1": i_hi, "E2": i_hi + 1, "E3": i_hi + 2}
    e_day = {k: epoch_days[v] for k, v in e_idx.items()}
    for key, label, *_ in BLOCKS:
        y = blocks[key]["y"]
        v = {k: y[i] for k, i in e_idx.items()}
        rows.append(dict(event=lo_day.year, series=key, label=label, n_crossings=blocks[key]["n"],
                         **{k: e_day[k].date() for k in e_idx},
                         step_before2_mm=v["Em1"] - v["Em2"], step_before_mm=v["E0"] - v["Em1"],
                         step_window_mm=v["E1"] - v["E0"],
                         step_after_mm=v["E2"] - v["E1"], step_after2_mm=v["E3"] - v["E2"],
                         interval_noise_mm=interval_noise(rel[in_x], y[in_x])))
    cm_val = {}
    for k, d in e_day.items():
        match = np.where(cm_days[cm_in] == d)[0]
        cm_val[k] = cm_det[match[0]] if match.size else np.nan
    rows.append(dict(event=lo_day.year, series="cfw1_creepmeter", label="CFW1 creepmeter (daily mean)", n_crossings=0,
                     **{k: e_day[k].date() for k in e_idx},
                     step_before2_mm=cm_val["Em1"] - cm_val["Em2"], step_before_mm=cm_val["E0"] - cm_val["Em1"],
                     step_window_mm=cm_val["E1"] - cm_val["E0"],
                     step_after_mm=cm_val["E2"] - cm_val["E1"], step_after2_mm=cm_val["E3"] - cm_val["E2"],
                     interval_noise_mm=np.nan))
    per_event.append(dict(lo_day=lo_day, hi_day=hi_day, rel=rel, in_x=in_x, blocks=blocks,
                          cm_rel=cm_rel[cm_in], cm_det=cm_det, hov=hov, hov_km=km[hov_idx],
                          g_near=g_near, covered=covered, e_day=e_day))

table = pd.DataFrame(rows)
with open(table_path, "w") as f:
    f.write(f"# {stamp}\n# QC only, not registered. Steps (mm) of each detrended series (line fitted over the "
            f"{PRE_FIT_DAYS} d up to onset_lo) between fp_ts epochs: before2 = E-2 -> E-1, before = E-1 -> E0, window = E0 (onset_lo) -> "
            f"E1 (onset_hi), after = E1 -> E2, after2 = E2 -> E3. Creepmeter: daily means on the same days (reader day convention). "
            f"interval_noise_mm = robust sigma of single-interval differences of that series over the pre-window period.\n")
    table.to_csv(f, index=False, float_format="%.2f")
pd.set_option("display.width", 220)
print("\n", table.drop(columns=["label"]).to_string(index=False, float_format=lambda x: f"{x:6.2f}"))
print(f"\nwrote {table_path}")

# ------------------------------------------------------------------ figure
utils.mpl_paper_style(base=11)
plt.rcParams.update({"axes.titlesize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.labelsize": 11})
fig, axes = plt.subplots(2, 5, figsize=(14, 8.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.15]))
fig.subplots_adjust(left=0.06, right=0.985, top=0.905, bottom=0.235, wspace=0.12, hspace=0.12)
letters = "abcdefghij"

# common y range for row 1
ymin, ymax = 0.0, 0.0
for pe in per_event:
    vis = (pe["rel"] >= -HALF_DAYS) & pe["in_x"]
    for b in pe["blocks"].values():
        ymin = min(ymin, np.nanmin(b["y"][vis])); ymax = max(ymax, np.nanmax(b["y"][vis]))
    cvis = pe["cm_rel"] >= -HALF_DAYS
    ymin = min(ymin, np.nanmin(pe["cm_det"][cvis])); ymax = max(ymax, np.nanmax(pe["cm_det"][cvis]))
ylim1 = (np.floor(ymin) - 0.5, np.ceil(ymax) + 6.0)   # headroom: the catalogue strip
norm = TwoSlopeNorm(vmin=-4.0, vcenter=0.0, vmax=12.0)
cmap = plt.get_cmap("PuOr_r")

for col, pe in enumerate(per_event):
    ax = axes[0, col]
    w_hi = (pe["hi_day"] - pe["lo_day"]).days
    ax.axvspan(0, w_hi, color=WINDOW_SHADE, alpha=0.25, lw=0, zorder=0)
    ax.axhline(0, color="0.6", lw=0.6, zorder=1)
    # creepmeter line broken at record gaps longer than 2 d (no line drawn across missing days)
    cm_x, cm_y = pe["cm_rel"].astype(float), pe["cm_det"].astype(float)
    gap = np.where(np.diff(cm_x) > 2)[0]
    cm_x, cm_y = np.insert(cm_x, gap + 1, np.nan), np.insert(cm_y, gap + 1, np.nan)
    for g0 in gap:
        print(f"  {pe['lo_day'].year}: creepmeter gap {pe['cm_rel'][g0]:+.0f} to {pe['cm_rel'][g0 + 1]:+.0f} d")
    ax.plot(cm_x, cm_y, color=CM_COLOUR, lw=0.9, zorder=2,
            label="CFW1 creepmeter (daily)" if col == 0 else None)
    vis = pe["in_x"] & (pe["rel"] >= -HALF_DAYS)
    for key, b in pe["blocks"].items():
        ok = vis & np.isfinite(b["y"])
        ax.plot(pe["rel"][ok], b["y"][ok], "-o", color=b["colour"], ms=3.2, lw=1.1, zorder=3,
                label=f"{b['label']}" if col == 0 else None)
    # Gittins CFW1 events: tick at start, bar start -> end, slip label
    ytop = ylim1[1]
    for k, (_, g) in enumerate(pe["g_near"].sort_values("Start Time").iterrows()):
        yb = ytop - 1.4 - 1.8 * (k % 3)                     # three staggered levels so labels do not collide
        x0, x1 = max(g.rel_start, -HALF_DAYS), min(g.rel_end, HALF_DAYS)
        ax.plot([x0, x1], [yb, yb], color=GITTINS_COLOUR, lw=1.8, solid_capstyle="butt", zorder=4)
        if g.rel_start >= -HALF_DAYS:
            ax.plot([g.rel_start, g.rel_start], [yb - 0.5, yb + 0.5], color=GITTINS_COLOUR, lw=1.8, zorder=4)
        near_right = x0 > HALF_DAYS - 20                    # keep labels near the right edge inside the axis
        ax.text(x0 - 1.5 if near_right else x0 + 1.5, yb + 0.25, f"{g['Displacement, mm']:.2f}", color="#9A6400",
                fontsize=8.5, ha="right" if near_right else "left", va="bottom", zorder=5, clip_on=True)
    if not pe["covered"]:
        ax.text(0.03, 0.97, f"Gittins catalogue ends\n{g_cfw1_end.date()}", transform=ax.transAxes,
                fontsize=8.5, va="top", color="#A86E00")
    ax.set_ylim(ylim1)
    ax.set_title(f"{pe['lo_day']:%Y-%m-%d} to {pe['hi_day']:%Y-%m-%d} ({w_hi} d)", fontsize=11)
    ax.text(0.02, 0.03, rf"$\bf{{{letters[col]}}}$", transform=ax.transAxes, fontsize=13, va="bottom")
    if col == 0:
        ax.set_ylabel("Fault-parallel slip\nabove pre-window trend (mm)")
    else:
        ax.tick_params(labelleft=False)
    ax.grid(axis="y", color="0.9", lw=0.5)

    # row 2: hovmoller with cells bounded half-way between epochs
    ax2 = axes[1, col]
    rel_x = pe["rel"][pe["in_x"]]
    x_edges = np.concatenate([[rel_x[0] - 3], 0.5 * (rel_x[1:] + rel_x[:-1]), [rel_x[-1] + 3]])
    kk = pe["hov_km"]
    k_edges = np.concatenate([[kk[0] - 0.05], 0.5 * (kk[1:] + kk[:-1]), [kk[-1] + 0.05]])
    mesh = ax2.pcolormesh(x_edges, k_edges, np.ma.masked_invalid(pe["hov"]), cmap=cmap, norm=norm,
                          shading="flat", rasterized=True)
    for xv in (0, w_hi):
        ax2.axvline(xv, color="k", ls="--", lw=1.0)
    ax2.axhline(cfw1_km, color="#CC79A7", lw=1.6)
    if col == 4:
        ax2.text(HALF_DAYS - 2, cfw1_km - 0.12, "CFW1", color="#CC79A7", fontsize=10, ha="right", va="bottom",
                 fontweight="bold")
    for key, label, k0, k1, colour in BLOCKS[:3]:
        ax2.plot([-HALF_DAYS + 1] * 2, [k0, k1], color=colour, lw=4, solid_capstyle="butt", clip_on=False)
    ax2.set_ylim(HOV_KM[1], HOV_KM[0])            # north (small km) at top
    ax2.set_xlim(-HALF_DAYS, HALF_DAYS)
    ax2.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    ax2.set_xlabel("Days from onset_lo")
    ax2.text(0.02, 0.03, rf"$\bf{{{letters[col + 5]}}}$", transform=ax2.transAxes, fontsize=13, va="bottom",
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
    if col == 0:
        ax2.set_ylabel("km from Point Pinole (N at top)")
    else:
        ax2.tick_params(labelleft=False)

h, l = axes[0, 0].get_legend_handles_labels()
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
h += [Patch(color=WINDOW_SHADE, alpha=0.25), Line2D([], [], color=GITTINS_COLOUR, lw=1.6)]
l += ["InSAR onset window (onset_lo to onset_hi)", "Gittins et al. (2026) CFW1 event, start to end (slip, mm)"]
fig.legend(h, l, loc="upper center", ncol=4, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 0.995))
cax = fig.add_axes([0.30, 0.145, 0.40, 0.016])
cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="both")
cb.set_label("Fault-parallel slip above pre-window trend (mm), rows f-j")
cb.set_ticks([-4, -2, 0, 2, 4, 6, 8, 10, 12])

notes = (
    "QC only, not a paper figure. x = days from onset_lo (last acquisition before the step). fp_ts epochs are 6 or 12 d apart: differences in onset "
    "smaller than one acquisition interval are NOT resolvable,\n"
    f"so blocks that step in the same interval are simultaneous at this sampling. Every series (each crossing, the creepmeter) is detrended by a line "
    f"fitted over the {PRE_FIT_DAYS} d up to onset_lo; zero = that line extrapolated,\n"
    f"so levels drift late in the window. Row 1: block means of the detrended crossings, no offsets between curves; single-crossing noise ~{SINGLE_SITE_NOISE_MM} mm, "
    f"block means lower (pre-window interval scatter in qc05x_event_zoom_steps.csv).\n"
    f"Rows f-j: each crossing detrended, then smoothed along strike (Gaussian sigma {profile_smooth_km} km) at each epoch; cells span half-way to the neighbouring "
    f"epochs. Bars at left: N end, peak, S end blocks. CFW1 at {cfw1_km:.1f} km (nearest trace vertex).\n"
    "Creepmeter line broken at record gaps > 2 d; days in the reader day convention (pending day-of-year check may shift by 1 d). Gittins catalogue (CFW1) ends "
    f"{g_cfw1_end.date()}.\n{stamp}")
fig.text(0.01, 0.005, notes, fontsize=7.6, va="bottom", ha="left", color="0.25")

os.makedirs(common_paths["fig_dir"], exist_ok=True)
fig.savefig(fig_path + ".png", dpi=600, metadata={"Description": stamp})
fig.savefig(fig_path + ".pdf", metadata={"Subject": stamp})
print(f"wrote {fig_path}.png / .pdf")
