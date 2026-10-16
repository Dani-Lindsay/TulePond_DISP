#!/usr/bin/env python3
"""
Summary of every PyLith + CSI inversion run (2026-09-30): where does the selected model sit among the tests?
Reads <gf_dir>/results_comparison*.csv (one per run tag; both media), decodes the run settings from the tag, and draws
(a) west vs east near-fault rms, (b) west rms vs Mw (30 GPa), (c) a run-by-run ladder of the west rms, homog vs
SFCVM. Selected model: default_nuggetcd / sfcvm; companion (07a's covariance): default / sfcvm; reference: 07a CSI.

Ported from drafts/pylith (2026-09-30) for notebook 08b: `main(gf_dir, out_path)` returns the run table it drew;
gf_dir = 08c's products (outputs/08c_3d_model_sensitivity).
Run in opera_disp_feb13:  python scripts/model_plots/plot_test_summary.py   -> figS_pylith_all_tests.png
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
from TulePond_filepaths import paths_out, common_paths, model_use_surface   # noqa: E402
from insar_utils import mpl_paper_style                     # noqa: E402

LADDER = ["default", "shallow", "floor15", "floor4", "swath7", "dip80e", "dip70e", "order2", "nosurf",
          "diagcd", "nosurf_diagcd", "nuggetcd", "swath7_nuggetcd", "dip80e_nuggetcd", "dip70e_nuggetcd", "order2_nuggetcd",
          "e50", "w50", "e50_nuggetcd", "w50_nuggetcd", "coh07", "coh07_nuggetcd", "surf", "surf_diagcd", "surf_nuggetcd",
          "asis", "asis_nuggetcd"]


def load_runs(gf_dir):
    """Every PyLith row of every results_comparison*.csv in gf_dir (the aligned SFCVM and homogeneous media), with the
    settings decoded from the tag, plus the 07a CSI baseline row of the default run."""
    rows, ref_row = [], None
    for f in sorted(glob.glob(os.path.join(gf_dir, "results_comparison*.csv"))):
        tag = os.path.basename(f).replace("results_comparison", "").replace(".csv", "").lstrip("_") or "default"
        d = pd.read_csv(f)
        ref = d[d.model == "CSI homogeneous (07)"]
        d = d[d.model != "CSI homogeneous (07)"].copy()
        d["tag"] = tag
        d["medium"] = d.model.str.replace("PyLith ", "")
        rows.append(d)
        if tag == "default" and len(ref):
            ref_row = ref.iloc[0]
    assert ref_row is not None, "results_comparison.csv (the default run) must carry the 07a CSI baseline row"
    runs = pd.concat(rows, ignore_index=True)
    runs = runs[runs.medium != "sfcvmasis"]                                   # aligned SFCVM only (Dani)
    runs["cd"] = np.select([runs.tag.str.contains("nuggetcd"), runs.tag.str.contains("diagcd"), runs.tag.str.contains("floor")],
                           ["nugget (diagonal)", "equal weights", "full, nugget floor"], "full (correlated)")
    default_surface = "on" if model_use_surface else "off"                   # tags name the deviation from the default (2026-10-16)
    other_surface = "off" if model_use_surface else "on"
    switched = runs.tag.str.contains("nosurf") | (runs.tag.str.contains("surf") & ~runs.tag.str.contains("nosurf"))
    runs["surface"] = np.where(switched, other_surface, default_surface)
    runs["dip"] = np.select([runs.tag.str.contains("dip80e"), runs.tag.str.contains("dip70e")], ["80 E", "70 E"], "90")
    runs["order"] = np.where(runs.tag.str.contains("order2"), 2, 1)
    runs["label"] = runs.tag.str.replace("_", " ")
    return runs, ref_row


def main(gf_dir=None, out_path=None):
    gf_dir = gf_dir or paths_out["08c_3d_model_sensitivity"]
    out_path = out_path or os.path.join(common_paths["fig_dir"], "figS_pylith_all_tests.png")
    runs, ref = load_runs(gf_dir)
    assert len(runs) >= 20, f"fewer runs than expected ({len(runs)})"
    selected = runs[(runs.tag == "nuggetcd") & (runs.medium == "sfcvm")].iloc[0]
    primary = runs[(runs.tag == "default") & (runs.medium == "sfcvm")].iloc[0]
    c_cd = {"full (correlated)": "#0072B2", "full, nugget floor": "#56B4E9", "nugget (diagonal)": "#D55E00", "equal weights": "#E69F00"}
    m_med = {"homog": "o", "sfcvm": "s", "sfcvm_e50": "D", "sfcvm_w50": "D",   # D: SFCVM boundary 50 m E / W of the trace
             "sfcvm_asis": "^"}                                               # ^: the SFCVM not shifted onto the trace (2026-10-14)
    unknown_media = sorted(set(runs.medium) - set(m_med))
    assert not unknown_media, f"no marker for media {unknown_media}"

    mpl_paper_style(base=7)
    plt.rcParams.update({"axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6})   # journal size whatever ran before
    fig = plt.figure(figsize=(7.2, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25])

    def scatter(ax, xcol, ycol):
        for _, r in runs.iterrows():
            ax.scatter(r[xcol], r[ycol], marker=m_med[r.medium], s=34 if r.order == 1 else 60,
                       facecolors=c_cd[r.cd] if r.surface == "on" else "none", edgecolors=c_cd[r.cd], linewidths=1.0, zorder=3)
            if r.dip != "90":
                ax.annotate(r.dip, (r[xcol], r[ycol]), xytext=(3, 3), textcoords="offset points", fontsize=5.5, color="0.35")
        ax.scatter(ref[xcol], ref[ycol], marker="*", s=110, color="#000000", zorder=4, label="07a CSI homogeneous")
        for r, name, col in ((primary, "companion: SFSVM, full Cd", "#0072B2"), (selected, "selected: SFSVM, nugget Cd", "#D55E00")):
            ax.scatter(r[xcol], r[ycol], marker="s", s=170, facecolors="none", edgecolors=col, linewidths=2.0, zorder=5, label=name)

    ax = fig.add_subplot(gs[0, 0])
    scatter(ax, "rms_west300_mm", "rms_east300_mm")
    ax.set_xlabel("rms within 300 m WEST of the trace (mm)"); ax.set_ylabel("rms within 300 m EAST (mm)")
    ax.set_title(r"$\bf{A}$  Near-fault fit, both sides", loc="left", fontsize=7)
    ax.legend(fontsize=6, loc="upper left")

    ax = fig.add_subplot(gs[0, 1])
    scatter(ax, "rms_west300_mm", "Mw_30gpa")
    ax.set_xlabel("rms within 300 m WEST of the trace (mm)"); ax.set_ylabel("Mw (30 GPa)")
    ax.set_title(r"$\bf{B}$  West-side fit against event size", loc="left", fontsize=7)
    handles = [plt.Line2D([], [], marker="o", ls="", color="0.3", label="homogeneous"), plt.Line2D([], [], marker="s", ls="", color="0.3", label="SFSVM aligned"),
               plt.Line2D([], [], marker="s", ls="", mfc="none", color="0.3", label="surface constraint off"),
               plt.Line2D([], [], marker="s", ls="", color="0.3", ms=8, label="order-2 elements"),
               plt.Line2D([], [], marker="D", ls="", color="0.3", label="SFSVM boundary 50 m E / W"),
               plt.Line2D([], [], marker="^", ls="", color="0.3", label="SFSVM not shifted (as published)")]
    handles += [plt.Line2D([], [], marker="s", ls="", color=c, label=k) for k, c in c_cd.items()]
    ax.legend(handles=handles, fontsize=6, loc="lower left", ncol=2)

    ax = fig.add_subplot(gs[1, :])
    order = [t for t in LADDER if t in set(runs.tag)]
    y = np.arange(len(order))
    for i, tag in enumerate(order):
        sub = runs[runs.tag == tag]
        h = sub[sub.medium == "homog"]; s = sub[sub.medium == "sfcvm"]
        if len(h) and len(s):
            ax.plot([h.rms_west300_mm.iloc[0], s.rms_west300_mm.iloc[0]], [i, i], "-", color="0.75", lw=1.2, zorder=2)
        for r in sub.itertuples():
            ax.scatter(r.rms_west300_mm, i, marker=m_med[r.medium], s=40, facecolors=c_cd[r.cd] if r.surface == "on" else "none",
                       edgecolors=c_cd[r.cd], linewidths=1.0, zorder=3)
    ax.axvline(ref.rms_west300_mm, color="#000000", lw=1.0, ls="--", label="07a CSI homogeneous")
    ax.scatter(selected.rms_west300_mm, order.index("nuggetcd"), marker="s", s=170, facecolors="none", edgecolors="#D55E00", linewidths=2.0, zorder=5)
    ax.scatter(primary.rms_west300_mm, order.index("default"), marker="s", s=170, facecolors="none", edgecolors="#0072B2", linewidths=2.0, zorder=5)
    ax.set_yticks(y); ax.set_yticklabels([t.replace("_", " + ") for t in order]); ax.invert_yaxis()
    ax.set_xlabel("rms within 300 m WEST of the trace (mm); circle = homogeneous, square = SFSVM aligned")
    ax.set_title(r"$\bf{C}$  Every test: west-side fit, homogeneous vs SFSVM on the same settings", loc="left", fontsize=7)
    ax.grid(axis="x", color="0.9", lw=0.6)
    ax.legend(fontsize=6, loc="lower right")
    fig.suptitle("PyLith Green's functions + CSI inversion: all runs (vertical fault unless labelled; default mesh unless 'shallow')", fontsize=7)
    fig.savefig(out_path, dpi=300, facecolor="white")
    plt.close(fig)
    print("wrote", out_path, len(runs), "runs")
    return runs


if __name__ == "__main__":
    main()
