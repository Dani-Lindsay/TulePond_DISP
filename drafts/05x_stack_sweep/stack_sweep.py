#!/usr/bin/env python3
"""
QC 05x stack sweep (exploratory, NOT a paper figure; registers nothing, edits no notebook, writes nothing
under outputs/).

Question: how do the creepmeter scores of notebook 05 depend on the along-trace half-width of the InSAR
stack at each creepmeter? Swept: the single nearest usable crossing, then +/- 0.1, 0.2, ..., 1.0 km ALONG
THE TRACE with NO cap on the number of rows and NO widening; plus the current 05 setting (0.2 km, at most 5
rows, widen to 1 km if < 2 rows) for reference.

Scores per width and creepmeter, computed by the code of the notebooks themselves (the function and
parameter definitions are pulled out of scripts/05_validation.py and scripts/04_event_detection.py with
`ast` and exec'd, because both scripts run on import):
  * rate difference = InSAR crossing rate - creepmeter rate over the common window (05 [8]:
    block_bootstrap_slope point estimate = np.polyfit slope; the 95% CI is kept for context)
  * detrended RMSE and r (05 [8]: creepmeter interpolated onto the SAR epochs, utils.detrend of each)
  * COZ1: recovered fraction of the January 2019 and February 2022 creepmeter steps, the method of the
    paper's ~60% (05 [14c] = notebook 04 joint model: constant + rate + annual + a Heaviside step at the
    onset_lo of EVERY COZ1 event of creep_events.csv, fitted to the stack; amplitude / 04's
    creepmeter_step_mm). The 05 ladder's event_step fraction (local collars) is also written to the csv.
Anchor checks (asserted): 04's +/-2 km section stack reproduces 04's COZ1 amplitudes, and the joint model on
the daily creepmeter reproduces 04's creepmeter steps.

Inputs (read only): outputs/03_creep_timeseries/fp_ts.csv, outputs/04_event_detection/creep_events.csv,
the five creepmeter records (utils.daily_creepmeter with creepmeter_glitches).
Outputs: <fig_dir>/qc05x_stack_sweep.png + .csv (+ _aggregates.csv)
Run: nice -n 10 ~/miniforge3/envs/opera_disp_feb13/bin/python drafts/05x_stack_sweep/stack_sweep.py
"""
import ast
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
import insar_utils as utils                                           # noqa: E402
from TulePond_filepaths import (common_paths, paths_out, creepmeters,  # noqa: E402
                                creepmeter_glitches, segments)

# ------------------------------------------------------------------ the sweep
SWEEP_KM = [round(0.1 * i, 1) for i in range(1, 11)]      # half-widths along the trace, no row cap, no widening
NEAREST_SEARCH_KM = 1.0                                   # the single nearest usable crossing, within this
NO_CAP = 10 ** 6

# ------------------------------------------------------------------ the notebooks' own definitions
def pull(script, function_names, assign_names):
    """Source of the named top-level functions and simple assignments of a notebook script."""
    tree = ast.parse(open(script).read())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in function_names:
            found[node.name] = ast.get_source_segment(open(script).read(), node)
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in assign_names:
            found[node.targets[0].id] = ast.get_source_segment(open(script).read(), node)
    missing = set(function_names) | set(assign_names)
    missing -= set(found)
    assert not missing, f"{os.path.basename(script)}: not found {sorted(missing)}"
    return found


script05 = os.path.join(REPO, "scripts", "05_validation.py")
script04 = os.path.join(REPO, "scripts", "04_event_detection.py")
src05 = pull(script05,
             ["read_fp_ts", "stack_insar", "slope", "block_indices", "block_bootstrap_slope", "event_step"],
             ["block_years", "nboot", "seed", "stack_km", "stack_widen_km", "stack_max",
              "collar_yr", "min_collar_epochs", "min_fit_epochs"])
src04 = pull(script04, ["design", "fit_joint", "load_stack"], ["stack_half_km"])

ns05 = dict(np=np, pd=pd, warnings=warnings, utils=utils)
for name in ["block_years", "nboot", "seed", "stack_km", "stack_widen_km", "stack_max",
             "collar_yr", "min_collar_epochs", "min_fit_epochs"]:
    exec(src05[name], ns05)
ns05["to_km"] = utils.lat_to_km_map(common_paths["fault_profiles"])          # as 05 [6]
ns05["rng"] = np.random.default_rng(ns05["seed"])                             # as 05 [8]
for name in ["read_fp_ts", "stack_insar", "slope", "block_indices", "block_bootstrap_slope", "event_step"]:
    exec(src05[name], ns05)
assert (ns05["stack_km"], ns05["stack_max"], ns05["stack_widen_km"]) == (0.2, 5, 1.0), \
    f"05's stack setting changed: {ns05['stack_km']}, {ns05['stack_max']}, {ns05['stack_widen_km']}"
stack_insar = ns05["stack_insar"]
block_bootstrap_slope = ns05["block_bootstrap_slope"]
event_step = ns05["event_step"]

ns04 = dict(np=np, pd=pd, warnings=warnings, utils=utils)
exec(src04["stack_half_km"], ns04)
for name in ["design", "fit_joint"]:
    exec(src04[name], ns04)
fit_joint = ns04["fit_joint"]

# ------------------------------------------------------------------ inputs, as 05 reads them
fp_ts_file = os.path.join(paths_out["03_creep_timeseries"], "fp_ts.csv")
events_file = os.path.join(paths_out["04_event_detection"], "creep_events.csv")
fp_ts = ns05["read_fp_ts"](fp_ts_file)
t0 = float(fp_ts.columns.min())
t1 = float(fp_ts.columns.max())
events_all = pd.read_csv(events_file)
events_coz1 = events_all.query("segment == 'COZ1'").sort_values("onset_lo").reset_index(drop=True)
events_coz1_large = events_coz1.nlargest(2, "creepmeter_step_mm").sort_values("onset_lo")       # 05 [12]
large_days = events_coz1_large.onset_day.tolist()
assert large_days[0].startswith("2019-01") and large_days[1].startswith("2022-02"), f"large COZ1 events: {large_days}"
print(f"fp_ts {fp_ts.shape[0]} crossings x {fp_ts.shape[1]} epochs ({t0:.3f}-{t1:.3f}); "
      f"fp_ts mtime {datetime.fromtimestamp(os.path.getmtime(fp_ts_file))}, "
      f"creep_events mtime {datetime.fromtimestamp(os.path.getmtime(events_file))}")
print(f"COZ1 events (step epochs of the joint model): {events_coz1.onset_day.tolist()}; large: {large_days}")

cm_data = {}
cm_rates = {}
for code, lat, lon, name in creepmeters:
    cm_data[code] = utils.daily_creepmeter(common_paths["creepmeter_dir"], code, t0, t1, creepmeter_glitches)
    cm_dates, cm_slip = cm_data[code]
    cm_rates[code] = block_bootstrap_slope(cm_dates, cm_slip, round(ns05["block_years"] / np.median(np.diff(cm_dates))))

# ------------------------------------------------------------------ anchor checks against 04
coz1_step_epochs = events_coz1.onset_lo.to_numpy(float)
fp_sorted = fp_ts.sort_index()
ns04.update(fp_ts=fp_sorted, km=ns05["to_km"](fp_sorted.index.values), dates=fp_sorted.columns.values.astype(float), to_km=ns05["to_km"])
exec(src04["load_stack"], ns04)
section_dates, section_stack, section_n = ns04["load_stack"](segments["COZ1"])
coef_section, _ = fit_joint(section_dates, section_stack, coz1_step_epochs)
assert np.allclose(np.round(coef_section[4:], 2), events_coz1.amplitude_mm.values, atol=0.011), \
    f"04's section stack does not reproduce 04's COZ1 amplitudes: {coef_section[4:]} vs {events_coz1.amplitude_mm.values}"
coef_cm, _ = fit_joint(*cm_data["COZ1"], coz1_step_epochs)
assert np.allclose(np.round(coef_cm[4:], 2), events_coz1.creepmeter_step_mm.values, atol=0.011), \
    f"the joint model on the creepmeter does not reproduce 04's steps: {coef_cm[4:]} vs {events_coz1.creepmeter_step_mm.values}"
print(f"anchor OK: 04 section stack ({section_n} crossings, +/-{ns04['stack_half_km']} km) amplitudes {np.round(coef_section[4:], 2)}, "
      f"creepmeter steps {np.round(coef_cm[4:], 2)}")


# ------------------------------------------------------------------ the scores
def score(code, stacked):
    """05 [8] rate and detrended scores for one station stack; at COZ1 also the 04 joint-model and 05
    event_step recoveries of the two large events."""
    cm_dates, cm_slip = cm_data[code]
    cm_rate, cm_lo, cm_hi = cm_rates[code]
    stack_dates, stack_mean, stack_std, n_rows, half_used = stacked
    finite = np.isfinite(stack_mean)
    insar_dates = stack_dates[finite]
    insar_slip = stack_mean[finite]
    cr_rate, cr_lo, cr_hi = block_bootstrap_slope(insar_dates, insar_slip,
                                                  round(ns05["block_years"] / np.median(np.diff(insar_dates))))
    cm_on_insar_dates = np.interp(insar_dates, cm_dates, cm_slip)
    insar_detrended = utils.detrend(insar_dates, insar_slip)
    cm_detrended = utils.detrend(insar_dates, cm_on_insar_dates)
    both_finite = np.isfinite(insar_detrended) & np.isfinite(cm_detrended)
    det_r = float(np.corrcoef(insar_detrended[both_finite], cm_detrended[both_finite])[0, 1])
    det_rmse = float(np.sqrt(np.mean((insar_detrended[both_finite] - cm_detrended[both_finite]) ** 2)))
    row = dict(code=code, n_rows=n_rows, half_used_km=half_used, n_epochs=int(finite.sum()),
               cm_rate=cm_rate, cr_rate=cr_rate, cr_lo=cr_lo, cr_hi=cr_hi, d_rate=cr_rate - cm_rate,
               det_rmse=det_rmse, det_r=det_r)
    if code == "COZ1":
        coef, _ = fit_joint(insar_dates, insar_slip, coz1_step_epochs)
        for i, event in events_coz1.iterrows():
            if event.onset_day not in large_days:
                continue
            tag = "2019" if event.onset_day.startswith("2019") else "2022"
            row[f"amp_{tag}_mm"] = float(coef[4 + i])
            row[f"rec_joint_{tag}_pct"] = 100 * float(coef[4 + i]) / event.creepmeter_step_mm
            insar_step = event_step(insar_dates, insar_slip, event.onset_lo, event.onset_hi)
            cm_step = event_step(insar_dates, cm_on_insar_dates, event.onset_lo, event.onset_hi)
            row[f"rec_eventstep_{tag}_pct"] = 100 * insar_step / cm_step
        row["rec_joint_mean_pct"] = 0.5 * (row["rec_joint_2019_pct"] + row["rec_joint_2022_pct"])
    return row


settings = [("nearest", 0.0, dict(half_km=NEAREST_SEARCH_KM, widen_km=NEAREST_SEARCH_KM, n_stack=1))]
for width in SWEEP_KM:
    settings.append((f"{width:.1f}", width, dict(half_km=width, widen_km=width, n_stack=NO_CAP)))
settings.append(("05 current (0.2 km, max 5)", 0.2, dict()))                    # 05's defaults

rows = []
for setting, width, kwargs in settings:
    for code, lat, lon, name in creepmeters:
        stacked = stack_insar(fp_ts, lat, **kwargs)
        if stacked is None:
            rows.append(dict(setting=setting, width_km=width, code=code, n_rows=0))
            continue
        rows.append(dict(setting=setting, width_km=width, **score(code, stacked)))
table = pd.DataFrame(rows)

agg_rows = []
for (setting, width), group in table.groupby(["setting", "width_km"], sort=False):
    ok = group.dropna(subset=["d_rate"])
    ex = ok[ok.code != "COZ1"]
    coz1 = group[group.code == "COZ1"].iloc[0]
    agg_rows.append(dict(setting=setting, width_km=width, n_stations=len(ok),
                         rows_per_station=" ".join(f"{r.code}:{int(r.n_rows)}" for r in group.itertuples()),
                         rate_rms_all=float(np.sqrt(np.mean(ok.d_rate ** 2))),
                         rate_rms_excl_coz1=float(np.sqrt(np.mean(ex.d_rate ** 2))),
                         rate_sd_across_stations=float(np.std(ok.d_rate, ddof=1)),
                         det_rmse_median=float(ok.det_rmse.median()), det_rmse_mean=float(ok.det_rmse.mean()),
                         det_r_median=float(ok.det_r.median()),
                         coz1_rec_2019=coz1.get("rec_joint_2019_pct", np.nan), coz1_rec_2022=coz1.get("rec_joint_2022_pct", np.nan),
                         coz1_rec_mean=coz1.get("rec_joint_mean_pct", np.nan),
                         coz1_eventstep_2019=coz1.get("rec_eventstep_2019_pct", np.nan),
                         coz1_eventstep_2022=coz1.get("rec_eventstep_2022_pct", np.nan)))
aggregates = pd.DataFrame(agg_rows)

fig_dir = common_paths["fig_dir"]
os.makedirs(fig_dir, exist_ok=True)
out_base = os.path.join(fig_dir, "qc05x_stack_sweep")
provenance = (f"# qc05x_stack_sweep ({datetime.now():%Y-%m-%d %H:%M}); drafts/05x_stack_sweep/stack_sweep.py; "
              f"fp_ts {fp_ts_file}; creep_events {events_file}; rate/detrended = 05 [8], COZ1 recovery = 04 joint model "
              f"(05 [14c]) / creepmeter_step_mm; event_step = 05 ladder method")
with open(out_base + ".csv", "w") as handle:
    handle.write(provenance + "\n")
    table.to_csv(handle, index=False, float_format="%.4f")
with open(out_base + "_aggregates.csv", "w") as handle:
    handle.write(provenance + "\n")
    aggregates.to_csv(handle, index=False, float_format="%.4f")

pd.set_option("display.width", 250)
print("\nPER STATION")
print(table[["setting", "code", "n_rows", "half_used_km", "d_rate", "cr_lo", "cr_hi", "det_rmse", "det_r"]].round(2).to_string(index=False))
print("\nCOZ1 RECOVERY (%)")
print(table[table.code == "COZ1"][["setting", "n_rows", "amp_2019_mm", "rec_joint_2019_pct", "amp_2022_mm", "rec_joint_2022_pct",
                                   "rec_joint_mean_pct", "rec_eventstep_2019_pct", "rec_eventstep_2022_pct"]].round(1).to_string(index=False))
print("\nAGGREGATES")
print(aggregates.drop(columns=["rows_per_station"]).round(2).to_string(index=False))
print("\nROWS PER STATION")
print(aggregates[["setting", "rows_per_station"]].to_string(index=False))

# ------------------------------------------------------------------ figure (QC, slide-readable)
utils.mpl_paper_style(base=14)
colours = {"CPP1": "#000000", "CTM1": "#E69F00", "COZ1": "#D55E00", "CHP1": "#CC79A7", "CFW1": "#0072B2"}
markers = {"CPP1": "o", "CTM1": "s", "COZ1": "D", "CHP1": "^", "CFW1": "v"}
sweep = table[~table.setting.str.startswith("05")]
current = table[table.setting.str.startswith("05")]
agg_sweep = aggregates[~aggregates.setting.str.startswith("05")]
x_ticks = [0.0] + SWEEP_KM
x_labels = ["1 row"] + [f"{w:g}" for w in SWEEP_KM]

fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), sharex=True)
panels = [(axes[0, 0], "d_rate", "InSAR - creepmeter rate (mm/yr)", "a"),
          (axes[0, 1], "det_rmse", "Detrended RMSE (mm)", "b"),
          (axes[1, 0], "det_r", "Detrended correlation r", "c")]
for ax, column, ylabel, letter in panels:
    for code, lat, lon, name in creepmeters:
        here = sweep[sweep.code == code]
        ax.plot(here.width_km, here[column], marker=markers[code], color=colours[code], lw=1.8, ms=8, label=code)
        now = current[current.code == code]
        ax.plot(now.width_km, now[column], marker=markers[code], mfc="none", mec=colours[code], mew=2, ms=16, ls="none")
    if column == "d_rate":
        ax.plot(agg_sweep.width_km, agg_sweep.rate_rms_all, color="0.45", lw=3, ls="--", label="RMS, 5 stations")
        ax.plot(agg_sweep.width_km, agg_sweep.rate_rms_excl_coz1, color="0.45", lw=3, ls=":", label="RMS, excl. COZ1")
        ax.axhline(0, color="0.6", lw=0.8)
    if column == "det_rmse":
        ax.plot(agg_sweep.width_km, agg_sweep.det_rmse_median, color="0.45", lw=3, ls="--", label="median")
    ax.set_ylabel(ylabel)
    ax.set_title(f"({letter})", loc="left", fontweight="bold")
ax = axes[1, 1]
coz1_sweep = sweep[sweep.code == "COZ1"]
ax.plot(coz1_sweep.width_km, coz1_sweep.rec_joint_2019_pct, "o-", color="#E69F00", lw=2, ms=8, label="Jan 2019 (joint model)")
ax.plot(coz1_sweep.width_km, coz1_sweep.rec_joint_2022_pct, "s-", color="#0072B2", lw=2, ms=8, label="Feb 2022 (joint model)")
ax.plot(coz1_sweep.width_km, coz1_sweep.rec_joint_mean_pct, color="0.45", lw=3, ls="--", label="mean of the two")
ax.plot(coz1_sweep.width_km, coz1_sweep.rec_eventstep_2019_pct, "o:", color="#E69F00", mfc="none", lw=1.2, ms=6, label="Jan 2019 (05 event_step)")
ax.plot(coz1_sweep.width_km, coz1_sweep.rec_eventstep_2022_pct, "s:", color="#0072B2", mfc="none", lw=1.2, ms=6, label="Feb 2022 (05 event_step)")
now = current[current.code == "COZ1"]
for column, colour, marker in (("rec_joint_2019_pct", "#E69F00", "o"), ("rec_joint_2022_pct", "#0072B2", "s")):
    ax.plot(now.width_km, now[column], marker=marker, mfc="none", mec=colour, mew=2, ms=16, ls="none")
for i, event in events_coz1_large.reset_index(drop=True).iterrows():
    paper_pct = 100 * event.amplitude_mm / event.creepmeter_step_mm
    ax.axhline(paper_pct, color=("#E69F00", "#0072B2")[i], lw=1, ls=(0, (6, 3)), alpha=0.7)
ax.text(0.99, 0.02, "thin dashed: 04 +/-2 km section stack (the paper's numbers)", transform=ax.transAxes, ha="right", va="bottom", fontsize=10)
ax.set_ylabel("COZ1 creepmeter step recovered (%)")
ax.set_title("(d)", loc="left", fontweight="bold")
ax.set_ylim(10, 118)
ax.legend(fontsize=10, loc="upper center", ncol=2)
for ax in axes.ravel():
    ax.axvspan(0.15, 0.25, color="0.85", zorder=0)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.grid(alpha=0.3)
for ax in axes[1]:
    ax.set_xlabel("Stack half-width along the trace (km), no row cap")
axes[0, 0].legend(fontsize=11, ncol=2, loc="best")
fig.suptitle("QC 05x: InSAR stack width at the creepmeters (grey band / large open markers = 05 now: 0.2 km, max 5 rows)",
             fontsize=14)
fig.tight_layout()
fig.savefig(out_base + ".png", dpi=200)
print("Saved", out_base + ".png", out_base + ".csv", out_base + "_aggregates.csv")
