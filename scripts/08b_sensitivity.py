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
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 08b Sensitivity of the slip model: the supplementary figures and table (plots only)
#
# ```
# Created on Fri Oct 30 2026 (the rewire, code_audit_2026-10-28/REWIRE_BRIEF.md); replaces 08b_3d_model_tests and
# 08c_3d_model_sensitivity (drafts/retired_2026-10-28/).
# @author: daniellelindsay
# Run in opera_disp_feb13, after 07a, 08a and the scripts in scripts/sensitivity/. Seconds: nothing is inverted here.
# ```
#
# **What this notebook does.** It draws the archived results of the sensitivity scripts (`scripts/sensitivity/*.py`, run
# once; their csvs are archived on Zenodo) and of 07a / 08a: the sensitivity table, the sweeps figure (dip, sigma,
# lambda), the fixed-depth scan with its noise test, buried top and the SFSVM basement depth, the Green's-function error
# against the analytic solution, and the semivariogram of the noise model. Every csv says which script made it ([2]).
#
# **Inputs** (`model_config.sensitivity_dir`): `sensitivity_rows.csv`, `dip_rows.csv`, `depth_scan.csv`,
# `depth_scan_noise.csv`, `depth_scan_buried_top.csv`, `basement_depth.csv`, `gf_check.csv`; 07a `noise_model_5km.csv`,
# `semivariogram_bins_5km.csv`; 08a `selected_model.npz`.
#
# **Outputs** (`outputs/08b_sensitivity/`): `sensitivity_table.csv`, `08b_sensitivity_stats.txt`. **Figures**:
# `figS_sweeps`, `figS_depth_scan`, `figS_gf_error_vs_depth`, `figS_semivariogram` (Science style, png + pdf).
# **Registry keys** `08b_*`.
#
# **What to check**
# - [2] every csv's provenance names its script and a clean git hash.
# - [3] the baseline row equals 08a's selected model (same method, two routes).

# %% [markdown]
# ## [1] Imports, inputs, style

# %%
## [1]
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from TulePond_filepaths import common_paths, repo_dir, paths_out
import insar_utils as utils
import model_config as cfg
import model_utils

notebook = "08b_sensitivity"
out_dir = paths_out[notebook]
fig_dir = common_paths["fig_dir"]
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)
sens = cfg.sensitivity_dir
csv_files = {name: os.path.join(sens, name) for name in ("sensitivity_rows.csv", "dip_rows.csv", "depth_scan.csv", "depth_scan_noise.csv",
                                                          "depth_scan_buried_top.csv", "basement_depth.csv", "gf_check.csv")}
noise_file = os.path.join(paths_out["07a_uniform_model"], "noise_model_5km.csv")
bins_file = os.path.join(paths_out["07a_uniform_model"], "semivariogram_bins_5km.csv")
selected_file = os.path.join(paths_out["08a_3d_model"], "selected_model.npz")
REG = utils.Registry(notebook, common_paths, repo_dir)
REG.inputs(*csv_files.values(), noise_file, bins_file, selected_file)
tables = {name: pd.read_csv(path) for name, path in csv_files.items()}
utils.mpl_paper_style(base=7)
plt.rcParams.update({"axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
cm = 1 / 2.54
OKABE = dict(blue="#0072B2", orange="#E69F00", vermillion="#D55E00", purple="#CC79A7", black="#000000", grey="#999999")
band_lo_km, band_hi_km = cfg.band90_km()


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"), dpi=300, facecolor="white")
    print("saved", os.path.join(fig_dir, name + ".png"))

# %% [markdown]
# ## [2] Which script made each csv
#
# Every row of every csv carries the script and the git hash of the run that made it (`provenance`).

# %%
## [2]
for name, table in tables.items():
    sources = sorted(set(table.provenance))
    assert len(sources) == 1, f"{name}: rows from {len(sources)} runs: {sources}"
    print(f"{name:28s} {len(table):4d} rows  <- {sources[0]}")

# %% [markdown]
# ## [3] The sensitivity table (SM)
#
# One row per choice, each at its own L-curve corner (except the sigma rows), all scored on the same fitted samples
# (60 m cut): RMS (all, within 500 m west / east), Mw at 30 GPa and at the harmonic SFSVM rigidity, peak slip and depth,
# depth50 / depth90 of the potency, the Fig 2K band step. The inversion choices are measured on the selected (60 m)
# mesh against its baseline; the forward-model choices on the default (250 m) mesh against the default-mesh baseline.
# CHECK: the fine-mesh baseline row is 08a's selected model.

# %%
## [3]
rows = tables["sensitivity_rows.csv"]
selected = np.load(selected_file)
baseline = rows[(rows.row == "Baseline (fine mesh)")].iloc[0]
for key in ("sigma", "Mw_30gpa", "Mw_sfcvm", "rms_mm", "rms_west_mm", "rms_east_mm", "peak_slip_mm", "depth50_m", "depth90_m"):
    assert np.isclose(baseline[key], float(selected[key]), rtol=1e-4), f"baseline row {key} {baseline[key]} vs 08a {float(selected[key])}"
print("PASS: the baseline row of sensitivity_rows.csv is 08a's selected model")
columns = dict(row="Choice", variant="Variant", sigma="Sigma", Mw_30gpa="Mw (30 GPa)", Mw_sfcvm="Mw (harmonic SFSVM)",
               rms_mm="RMS (mm)", rms_west_mm="RMS 500 m west (mm)", rms_east_mm="RMS 500 m east (mm)", peak_slip_mm="Peak slip (mm)",
               peak_depth_m="Peak depth (m)", depth50_m="Depth50 (m)", depth90_m="Depth90 (m)", band_step_mean_mm="Band step (mm)")
sensitivity_table = rows[list(columns)].rename(columns=columns)
sensitivity_table.to_csv(os.path.join(out_dir, "sensitivity_table.csv"), index=False)
print(sensitivity_table.round({"Sigma": 1, "Mw (30 GPa)": 3, "Mw (harmonic SFSVM)": 3, "RMS (mm)": 3, "RMS 500 m west (mm)": 3,
                               "RMS 500 m east (mm)": 3, "Peak slip (mm)": 1, "Peak depth (m)": 0, "Depth50 (m)": 0,
                               "Depth90 (m)": 0, "Band step (mm)": 2}).to_string(index=False))
not_sigma = rows[rows.row != "Regularization sigma"]
mw_range = (not_sigma.Mw_30gpa.min(), not_sigma.Mw_30gpa.max())
sigma_rows = rows[rows.row == "Regularization sigma"]
print(f"Mw (30 GPa) over every choice except the sigma sweep: {mw_range[0]:.2f}-{mw_range[1]:.2f}; sigma sweep "
      f"{sigma_rows.Mw_30gpa.min():.2f}-{sigma_rows.Mw_30gpa.max():.2f}")
print(f"depth90 over sigma x0.25-x4 and the corner: {min(sigma_rows.depth90_m.min(), baseline.depth90_m):.0f}-"
      f"{max(sigma_rows.depth90_m.max(), baseline.depth90_m):.0f} m; over lambda: "
      f"{rows[rows.row == 'Prior lengths'].depth90_m.min():.0f}-{rows[rows.row == 'Prior lengths'].depth90_m.max():.0f} m")

# %% [markdown]
# ## [4] The sweeps figure (SM)
#
# Columns: the fault dip, the regularization sigma, the prior lengths lambda (the order the tests were made); rows: RMS
# within 500 m (west solid, east dashed), RMS of all fitted samples, Mw (30 GPa), peak slip, depth90 of the potency.
# Dip: the default (250 m) mesh with PyLith 3D (equal weights) and the analytic homogeneous solution (to 50 W, where no
# PyLith run exists), and the selected 60 m mesh at 70 W / vertical / 70 E (the mesh cross-check). Grey: the selected value.

# %%
## [4]
dips = tables["dip_rows.csv"]
fine = rows[rows.run.str.startswith("gf_shallow")]


def pick(row, variant):
    found = rows[(rows.row == row) & (rows.variant == variant)]
    assert len(found) == 1, f"{row} {variant}: {len(found)} rows"
    return found.iloc[0]


dip_series = []
for family, colour, marker, label in (("3D N^0", OKABE["vermillion"], "o", "3D SFSVM, 250 m surface cells"),
                                      ("Homogeneous analytic N^0", OKABE["black"], "D", "Uniform half-space, analytic")):
    series = dips[dips.family == family].copy()
    series["x"] = series.dip_from_vertical_deg
    dip_series.append((label, colour, marker, series.sort_values("x")))
fine_dips = pd.DataFrame([pick("Fault dip, fine mesh", "70 W"), baseline, pick("Fault dip, fine mesh", "70 E")])
fine_dips["x"] = [-20, 0, 20]
dip_series.append(("3D SFSVM, 60 m surface cells (selected)", OKABE["blue"], "s", fine_dips))
sigma_labels = ["x0.25", "x0.5", "Corner", "x2", "x4"]
sigma_frame = pd.DataFrame([baseline if lab == "Corner" else pick("Regularization sigma", lab) for lab in sigma_labels])
sigma_frame["x"] = np.arange(len(sigma_labels))
lambda_labels = ["x0.5", "x1", "x2"]
lambda_frame = pd.DataFrame([baseline if lab == "x1" else pick("Prior lengths", lab) for lab in lambda_labels])
lambda_frame["x"] = np.arange(len(lambda_labels))
selected_series = ("3D SFSVM, 60 m surface cells (selected)", OKABE["blue"], "s")
metrics = [("near", "RMS within\n500 m (mm)"), ("rms_mm", "RMS, all\nsamples (mm)"), ("Mw_30gpa", "Mw (30 GPa)"),
           ("peak_slip_mm", "Peak slip (mm)"), ("depth90_m", "Depth extent (m)\n90 % of potency")]
sweeps = [("Fault dip", dip_series, [-40, -30, -20, -10, 0, 10, 20], ["50 W", "60 W", "70 W", "80 W", "90", "80 E", "70 E"], 4),
          ("Regularization $\\sigma$", [selected_series + (sigma_frame,)], list(range(5)), sigma_labels, 2),
          ("Prior lengths $\\lambda$", [selected_series + (lambda_frame,)], list(range(3)), lambda_labels, 1)]
lw, ms = 0.8, 2.8
fig = plt.figure(figsize=(15.0 * cm, 14.0 * cm), constrained_layout=True)
fig.get_layout_engine().set(h_pad=0.01, hspace=0.0)
grid = fig.add_gridspec(len(metrics) + 1, len(sweeps), height_ratios=[1] * len(metrics) + [0.10], width_ratios=[7, 5, 3])
axes = np.empty((len(metrics), len(sweeps)), dtype=object)
for j in range(len(sweeps)):
    for i in range(len(metrics)):
        axes[i, j] = fig.add_subplot(grid[i, j], sharex=axes[0, j] if i else None, sharey=axes[i, 0] if j else None)
legend_ax = fig.add_subplot(grid[len(metrics), :])
legend_ax.axis("off")
handles = {}
letters = "ABCDEFGHIJKLMNO"
for j, (name, series_list, ticks, tick_labels, selected_at) in enumerate(sweeps):
    for i, (metric, metric_label) in enumerate(metrics):
        ax = axes[i, j]
        half = 0.4 * (ticks[1] - ticks[0])
        ax.set_xlim(ticks[0] - 2 * half, ticks[-1] + 2 * half)
        ax.axvspan(ticks[selected_at] - half, ticks[selected_at] + half, color="0.85", lw=0, zorder=0)
        utils.panel_title(ax, letters[len(sweeps) * i + j], name if i == 0 else "")
        if j == 0:
            ax.set_ylabel(metric_label)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_xticks(ticks)
        if i == len(metrics) - 1:
            rotation = 45 if len(tick_labels) > 3 else 0
            ax.set_xticklabels(tick_labels, rotation=rotation, ha="right" if rotation else "center", rotation_mode="anchor")
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
        for label, colour, marker, frame in series_list:
            if metric == "near":
                ax.plot(frame.x, frame.rms_west_mm, "-", marker=marker, color=colour, lw=lw, ms=ms)
                ax.plot(frame.x, frame.rms_east_mm, "--", marker=marker, color=colour, lw=lw, ms=ms, mfc="white")
            else:
                ax.plot(frame.x, frame[metric], "-", marker=marker, color=colour, lw=lw, ms=ms)
            handles.setdefault(label, Line2D([], [], color=colour, marker=marker, lw=lw, ms=ms, label=label))
handles["West"] = Line2D([], [], color="0.3", ls="-", marker="o", lw=lw, ms=ms, label="Within 500 m: west")
handles["East"] = Line2D([], [], color="0.3", ls="--", marker="o", mfc="white", lw=lw, ms=ms, label="Within 500 m: east")
legend_ax.legend(handles=list(handles.values()), loc="center", ncol=3, frameon=False)
save(fig, "figS_sweeps")

# %% [markdown]
# ## [5] The fixed-depth scan (SM): can the data tell how deep the slip goes?
#
# Uniform slip from the surface to D (`scripts/sensitivity/depth_scan.py`): RMS within 500 m west / east, RMS of all
# fitted samples, and the best-fit slip with its Mw, against D. Dotted: the selected (distributed-slip) model. Grey band:
# the base of the Cenozoic sediment west of the fault in the SFSVM (`basement_depth.py`, 90 % band). The noise test
# (the spread of the best D under correlated noise) and the buried-top test are printed, for the caption.

# %%
## [5]
scan = tables["depth_scan.csv"]
noise_test = tables["depth_scan_noise.csv"]
buried_top = tables["depth_scan_buried_top.csv"]
basement = tables["basement_depth.csv"]
west_basement = basement[basement.km.between(band_lo_km, band_hi_km) & (basement.side == "west")].basement_top_m / 1e3
b_median, b_lo, b_hi = west_basement.median(), west_basement.min(), west_basement.max()
best = scan.loc[scan.rms_all_mm.idxmin()]
print(noise_test.round(2).to_string(index=False))
print(buried_top.round(3).to_string(index=False))
fig, axes = plt.subplots(1, 3, figsize=(15.0 * cm, 5.6 * cm), constrained_layout=True, sharex=True)
ax = axes[0]
ax.plot(scan.D_km, scan.rms_W500_mm, "o-", color=OKABE["vermillion"], lw=lw, ms=ms, label="West")
ax.plot(scan.D_km, scan.rms_E500_mm, "s--", color=OKABE["blue"], lw=lw, ms=ms, mfc="white", label="East")
ax.axhline(baseline.rms_west_mm, color=OKABE["vermillion"], ls=":", lw=lw)
ax.axhline(baseline.rms_east_mm, color=OKABE["blue"], ls=":", lw=lw)
ax.legend(loc="upper center", title="Dotted: distributed slip\n(selected model)", frameon=False)
ax = axes[1]
ax.plot(scan.D_km, scan.rms_all_mm, "o-", color="k", lw=lw, ms=ms, label="Uniform slip to D")
ax.axhline(baseline.rms_mm, color="0.4", ls=":", lw=lw, label="Distributed slip (selected)")
ax.legend(loc="upper center", frameon=False)
ax = axes[2]
ax.plot(scan.D_km, scan.slip_mm, "o-", color="k", lw=lw, ms=ms)
twin = ax.twinx()
twin.plot(scan.D_km, scan.Mw_30gpa, "s--", color=OKABE["orange"], lw=lw, ms=ms)
twin.set_ylabel("Mw (30 GPa)", color=OKABE["orange"])
for ax, letter, title, ylabel in ((axes[0], "A", "Near-field misfit, west vs east", "RMS within 500 m (mm)"),
                                  (axes[1], "B", "Overall misfit", "RMS, all samples (mm)"),
                                  (axes[2], "C", "Slip amplitude and Mw", "Best-fit uniform slip (mm)")):
    utils.panel_title(ax, letter, title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Slip from the surface to D (km)")
    ax.axvspan(b_lo, b_hi, color="0.88", lw=0, zorder=0)
    ax.axvline(b_median, color="0.5", lw=lw * 0.6, zorder=0)
    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 3])
    ax.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "3"])
    ax.grid(True, lw=0.3, color="0.85")
axes[2].text(0.5, 0.97, f"Grey: base of the Cenozoic\nsediment, west (SFSVM)\n{1e3 * b_median:.0f} m ({1e3 * b_lo:.0f}-{1e3 * b_hi:.0f} m)",
             transform=axes[2].transAxes, ha="center", va="top", fontsize=6.5, color="0.3")
for axis, fraction in ((axes[0], 0.6), (axes[1], 0.35), (axes[2], 0.45), (twin, 0.45)):
    lo, hi = axis.get_ylim()
    axis.set_ylim(lo, hi + fraction * (hi - lo))
save(fig, "figS_depth_scan")

# %% [markdown]
# ## [6] The Green's functions against the analytic solution (SM)
#
# Per impulse, the relative RMS error of the homogeneous PyLith Green's function against the analytic solution for the
# same tent (`scripts/sensitivity/gf_check.py`). A: element order 1 vs 2 on the default mesh. B: the three meshes at order 2
# in the strip km 58.85-59.25; the selected mesh (60 m cells within 1 km of the fault) in blue.

# %%
## [6]
gf = tables["gf_check.csv"]
# a PyLith impulse whose sign disagrees with the analytic solution is a PyLith error (the analytic sign is fixed and tested):
# shown, listed, never flipped. Only order-1 runs may have one (2026-10-30: impulse 1030, km 60.23, 87 m, at Lake Elizabeth).
disagree = gf[~gf.sign_agrees]
print(f"impulses whose PyLith sign disagrees with the analytic solution: {len(disagree)}")
print(disagree[["panel", "run", "impulse", "km", "depth_m", "rel_rms"]].to_string(index=False))
assert (disagree.order == 1).all(), "an order-2 PyLith impulse disagrees in sign with the analytic solution"
fig, axes = plt.subplots(1, 2, figsize=(15.0 * cm, 6.0 * cm), constrained_layout=True, sharey=True)
panel_a = gf[gf.panel == "A"]
for run, colour, marker, label in (("gf_default_homog", "0.45", "^", "Order 1"), ("gf_default_homog_order2", OKABE["vermillion"], "o", "Order 2 (selected)")):
    part = panel_a[panel_a.run == run]
    axes[0].plot(part.depth_m, 100 * part.rel_rms, ls="none", marker=marker, color=colour, ms=3.0, label=label)
panel_b = gf[gf.panel == "B"]
for run, colour, marker, label in (("gf_default_homog_order2", OKABE["vermillion"], "o", "100 m fault / 250 m surface cells"),
                                   ("gf_fine_homog_order2_strip", OKABE["purple"], "D", "60 m fault / 200 m surface cells"),
                                   ("gf_shallow_homog_order2_strip", OKABE["blue"], "s", "60 m cells within 1 km (selected)")):
    part = panel_b[panel_b.run == run]
    axes[1].plot(part.depth_m, 100 * part.rel_rms, ls="none", marker=marker, color=colour, ms=3.0, label=label)
for ax, letter, title in ((axes[0], "A", "Element order (100 m fault / 250 m surface cells)"), (axes[1], "B", "Mesh refinement (order 2)")):
    ax.set_yscale("log")
    ax.set_ylim(0.08, 400)
    ax.set_xlim(-30, 1150)
    ax.set_xlabel("Impulse depth (m)")
    ax.grid(True, which="both", lw=0.3, color="0.85")
    ax.legend(loc="upper right", frameon=False)
    utils.panel_title(ax, letter, title)
axes[0].set_ylabel("RMS error vs analytic solution (%)")
gf_medians = gf.groupby(["panel", "run", "depth_group"], observed=True).rel_rms.median().unstack() * 100
print(gf_medians.round(1).to_string())
save(fig, "figS_gf_error_vs_depth")

# %% [markdown]
# ## [7] The noise model: semivariogram of the far field (Methods / SM)
#
# 07a's products: the binned semivariogram of the pixels farther than 5 km from the trace (plane removed) and the fitted
# model (`model_utils.semivariogram_model`).

# %%
## [7]
noise = pd.read_csv(noise_file).iloc[0]
semivariogram = pd.read_csv(bins_file)
params = (noise.nugget2_mm2, noise.sill_short_mm2, noise.range_short_km, noise.sill_long_mm2, noise.range_long_km)
fitted = semivariogram.gamma_mm2.notna()
fig, ax = plt.subplots(figsize=(8.0 * cm, 6.0 * cm), constrained_layout=True)
ax.plot(semivariogram.distance_km[fitted], semivariogram.gamma_mm2[fitted], "o", color="0.35", ms=2.2, mec="none",
        label="Far-field pixels (binned)")
distance = np.linspace(0, noise.distmax_km, 400)
ax.plot(distance, model_utils.semivariogram_model(distance, *params), "-", color=OKABE["vermillion"], lw=1.0, label="Fitted model")
ax.axhline(noise.nugget2_mm2, color=OKABE["blue"], ls=":", lw=1.0, label=f"Nugget ({noise.nugget_mm:.2f} mm)$^2$")
ax.set_xlim(0, noise.distmax_km)
ax.set_ylim(0, None)
ax.set_xlabel("Distance between pixels (km)")
ax.set_ylabel("Semivariance (mm$^2$)")
ax.legend(loc="lower right", frameon=False)
print(f"{int(noise.n_pixels):,} pixels beyond {noise.mask_km:g} km; nugget {noise.nugget_mm:.3f} mm; exponential "
      f"{noise.short_sd_mm:.3f} mm / {noise.range_short_km:.2f} km; total sd {noise.total_sd_mm:.3f} mm")
save(fig, "figS_semivariogram")

# %% [markdown]
# ## [8] Registry

# %%
## [8]
stats = REG.sidecar("08b_sensitivity", [f"{name}: {sorted(set(table.provenance))[0]}" for name, table in tables.items()]
                    + [sensitivity_table.to_string(index=False), noise_test.to_string(index=False), buried_top.to_string(index=False),
                       gf_medians.to_string()])
where = "SM"
REG.register("08b_mw_range_lo", mw_range[0], "Mw", 2, where, "[3]", stats)
REG.register("08b_mw_range_hi", mw_range[1], "Mw", 2, where, "[3]", stats)
REG.register("08b_mw_sigma_sweep_lo", sigma_rows.Mw_30gpa.min(), "Mw", 2, where, "[3]", stats)
REG.register("08b_mw_sigma_sweep_hi", sigma_rows.Mw_30gpa.max(), "Mw", 2, where, "[3]", stats)
REG.register("08b_depth90_sigma_lo_m", min(sigma_rows.depth90_m.min(), baseline.depth90_m), "m", 0, where, "[3]", stats)
REG.register("08b_depth90_sigma_hi_m", max(sigma_rows.depth90_m.max(), baseline.depth90_m), "m", 0, where, "[3]", stats)
REG.register("08b_depth90_lambda_lo_m", rows[rows.row == "Prior lengths"].depth90_m.min(), "m", 0, where, "[3]", stats)
REG.register("08b_depth90_lambda_hi_m", rows[rows.row == "Prior lengths"].depth90_m.max(), "m", 0, where, "[3]", stats)
REG.register("08b_full_cd_mw_30gpa", pick("Data weighting", "Full Cd").Mw_30gpa, "Mw", 2, where, "[3]", stats)
REG.register("08b_cut30_mw_30gpa", pick("Near-trace exclusion", "30 m").Mw_30gpa, "Mw", 2, where, "[3]", stats)
REG.register("08b_depth_scan_best_d_m", 1e3 * best.D_km, "m", 0, where, "[5]", stats)
REG.register("08b_depth_scan_best_slip_mm", best.slip_mm, "mm", 1, where, "[5]", stats)
all_samples = noise_test[noise_test.metric == "All samples"].iloc[0]
REG.register("08b_depth_scan_noise_p16_m", 1e3 * all_samples.noise_p16, "m", 0, where, "[5]", stats)
REG.register("08b_depth_scan_noise_p84_m", 1e3 * all_samples.noise_p84, "m", 0, where, "[5]", stats)
REG.register("08b_depth_scan_noise_p97_5_m", 1e3 * all_samples.noise_p97_5, "m", 0, where, "[5]", stats)
REG.register("08b_basement_west_band_median_m", 1e3 * b_median, "m", 0, where, "[5]", stats)
REG.render()
