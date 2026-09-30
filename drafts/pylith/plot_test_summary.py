#!/usr/bin/env python3
"""
Summary of every PyLith + CSI inversion run (2026-09-30): where does the selected model sit among the tests?
Reads gf/results_comparison*.csv (one per run tag; both media), decodes the run settings from the tag, and draws
(a) west vs east near-fault rms, (b) west rms vs Mw (30 GPa), (c) a run-by-run ladder of the west rms, homog vs
SFCVM. Selected model: default_nuggetcd / sfcvm; primary (paper's covariance): default / sfcvm; reference: 07 CSI.
Output: figures/summary_all_tests.png
"""
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for f in sorted(glob.glob(os.path.join(HERE, "gf", "results_comparison*.csv"))):
    tag = os.path.basename(f).replace("results_comparison", "").replace(".csv", "").lstrip("_") or "default"
    d = pd.read_csv(f)
    ref = d[d.model == "CSI homogeneous (07)"]
    d = d[d.model != "CSI homogeneous (07)"].copy()
    d["tag"] = tag
    d["medium"] = d.model.str.replace("PyLith ", "")
    rows.append(d)
    if tag == "default" and len(ref):
        ref_row = ref.iloc[0]
ref = ref_row
runs = pd.concat(rows, ignore_index=True)
runs = runs[runs.medium != "sfcvmasis"]                                   # aligned SFCVM only (Dani)
runs["cd"] = np.select([runs.tag.str.contains("nuggetcd"), runs.tag.str.contains("diagcd"), runs.tag.str.contains("floor")],
                       ["nugget (diagonal)", "equal weights", "full, nugget floor"], "full (correlated)")
runs["surface"] = np.where(runs.tag.str.contains("nosurf"), "off", "on")
runs["dip"] = np.select([runs.tag.str.contains("dip80e"), runs.tag.str.contains("dip70e")], ["80 E", "70 E"], "90")
runs["order"] = np.where(runs.tag.str.contains("order2"), 2, 1)
runs["label"] = runs.tag.str.replace("_", " ")
assert len(runs) >= 20, "fewer runs than expected"

selected = runs[(runs.tag == "nuggetcd") & (runs.medium == "sfcvm")].iloc[0]
primary = runs[(runs.tag == "default") & (runs.medium == "sfcvm")].iloc[0]
c_cd = {"full (correlated)": "#0072B2", "full, nugget floor": "#56B4E9", "nugget (diagonal)": "#D55E00", "equal weights": "#E69F00"}
m_med = {"homog": "o", "sfcvm": "s", "sfcvm_e50": "D", "sfcvm_w50": "D"}   # D: SFCVM boundary 50 m E / W of the trace (alignment sensitivity)

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial"], "font.size": 8})
fig = plt.figure(figsize=(7.2, 7.4), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25])


def scatter(ax, xcol, ycol):
    for _, r in runs.iterrows():
        ax.scatter(r[xcol], r[ycol], marker=m_med[r.medium], s=34 if r.order == 1 else 60, facecolors=c_cd[r.cd] if r.surface == "on" else "none",
                   edgecolors=c_cd[r.cd], linewidths=1.0, zorder=3)
        if r.dip != "90":
            ax.annotate(r.dip, (r[xcol], r[ycol]), xytext=(3, 3), textcoords="offset points", fontsize=5.5, color="0.35")
    ax.scatter(ref[xcol], ref[ycol], marker="*", s=110, color="#000000", zorder=4, label="07 CSI homogeneous (paper)")
    for r, name, col in ((primary, "primary: SFCVM, full Cd", "#0072B2"), (selected, "selected: SFCVM, nugget Cd", "#D55E00")):
        ax.scatter(r[xcol], r[ycol], marker="s", s=170, facecolors="none", edgecolors=col, linewidths=2.0, zorder=5, label=name)


ax = fig.add_subplot(gs[0, 0])
scatter(ax, "rms_west300_mm", "rms_east300_mm")
ax.set_xlabel("rms within 300 m WEST of the trace (mm)"); ax.set_ylabel("rms within 300 m EAST (mm)")
ax.set_title("A  Near-fault fit, both sides", loc="left", fontweight="bold")
ax.legend(fontsize=6, loc="upper left")

ax = fig.add_subplot(gs[0, 1])
scatter(ax, "rms_west300_mm", "Mw_30gpa")
ax.set_xlabel("rms within 300 m WEST of the trace (mm)"); ax.set_ylabel("Mw (30 GPa)")
ax.set_title("B  West-side fit against event size", loc="left", fontweight="bold")
handles = [plt.Line2D([], [], marker="o", ls="", color="0.3", label="homogeneous"), plt.Line2D([], [], marker="s", ls="", color="0.3", label="SFCVM aligned"),
           plt.Line2D([], [], marker="s", ls="", mfc="none", color="0.3", label="surface constraint off"),
           plt.Line2D([], [], marker="s", ls="", color="0.3", ms=8, label="order-2 elements"),
           plt.Line2D([], [], marker="D", ls="", color="0.3", label="SFCVM boundary 50 m E / W")]
handles += [plt.Line2D([], [], marker="s", ls="", color=c, label=k) for k, c in c_cd.items()]
ax.legend(handles=handles, fontsize=6, loc="lower left", ncol=2)

ax = fig.add_subplot(gs[1, :])
order = ["default", "shallow", "floor15", "floor4", "swath7", "dip80e", "dip70e", "order2", "nosurf",
         "diagcd", "nosurf_diagcd", "nuggetcd", "swath7_nuggetcd", "dip80e_nuggetcd", "dip70e_nuggetcd", "order2_nuggetcd",
         "e50", "w50", "e50_nuggetcd", "w50_nuggetcd", "coh07", "coh07_nuggetcd"]
order = [t for t in order if t in set(runs.tag)]
y = np.arange(len(order))
for i, tag in enumerate(order):
    sub = runs[runs.tag == tag]
    h = sub[sub.medium == "homog"]; s = sub[sub.medium == "sfcvm"]
    if len(h) and len(s):
        ax.plot([h.rms_west300_mm.iloc[0], s.rms_west300_mm.iloc[0]], [i, i], "-", color="0.75", lw=1.2, zorder=2)
    for r in sub.itertuples():
        ax.scatter(r.rms_west300_mm, i, marker=m_med[r.medium], s=40, facecolors=c_cd[r.cd] if r.surface == "on" else "none",
                   edgecolors=c_cd[r.cd], linewidths=1.0, zorder=3)
ax.axvline(ref.rms_west300_mm, color="#000000", lw=1.0, ls="--", label="07 CSI homogeneous")
ax.scatter(selected.rms_west300_mm, order.index("nuggetcd"), marker="s", s=170, facecolors="none", edgecolors="#D55E00", linewidths=2.0, zorder=5)
ax.scatter(primary.rms_west300_mm, order.index("default"), marker="s", s=170, facecolors="none", edgecolors="#0072B2", linewidths=2.0, zorder=5)
ax.set_yticks(y); ax.set_yticklabels([t.replace("_", " + ") for t in order]); ax.invert_yaxis()
ax.set_xlabel("rms within 300 m WEST of the trace (mm); circle = homogeneous, square = SFCVM aligned")
ax.set_title("C  Every test: west-side fit, homogeneous vs SFCVM on the same settings", loc="left", fontweight="bold")
ax.grid(axis="x", color="0.9", lw=0.6)
ax.legend(fontsize=6, loc="lower right")
fig.suptitle("PyLith Green's functions + CSI inversion: all runs, 2026-09-30 (vertical fault unless labelled; default mesh unless 'shallow')", fontsize=8)
out = os.path.join(HERE, "figures", "summary_all_tests.png")
fig.savefig(out, dpi=300, facecolor="white")
print("wrote", out, len(runs), "runs")
