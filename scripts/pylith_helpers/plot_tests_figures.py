#!/usr/bin/env python3
"""
Synthetic-recovery figure of notebook 08b, drawn from its saved products alone
(<gf_dir>/recovery_<run>_<medium>.csv, <gf_dir>/recovery_profiles.csv), so it can be restyled without re-running the tests.
Panel (b)'s input is drawn as an exact step (10 mm from the surface to D, 0 below).

Ported from drafts/pylith (2026-09-30): `main(gf_dir, out_path)`; gf_dir defaults to 08b's outputs folder.
Run in opera_disp_feb13:  python scripts/pylith_helpers/plot_tests_figures.py   -> figS_pylith_recovery.png
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
from TulePond_filepaths import paths_out, common_paths   # noqa: E402
from insar_utils import mpl_paper_style                   # noqa: E402

OK = dict(blue="#0072B2", orange="#E69F00", vermillion="#D55E00", black="#000000")
RUNS = [("sfcvm", "nugget", OK["vermillion"]), ("sfcvm", "full", OK["blue"]), ("homog", "nugget", OK["orange"]), ("homog", "full", OK["black"])]
RUN_NAME = {"nugget": "default_nuggetcd", "full": "default"}


def main(gf_dir=None, out_path=None):
    gf_dir = gf_dir or paths_out["08b_3d_model_tests"]
    out_path = out_path or os.path.join(common_paths["fig_dir"], "figS_pylith_recovery.png")
    mpl_paper_style(base=7)
    plt.rcParams.update({"axes.titlesize": 7, "axes.labelsize": 7, "legend.fontsize": 6})   # journal size whatever ran before
    tables = {(m, cd): pd.read_csv(os.path.join(gf_dir, f"recovery_{RUN_NAME[cd]}_{m}.csv")) for m, cd, _ in RUNS}
    prof = pd.read_csv(os.path.join(gf_dir, "recovery_profiles.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), constrained_layout=True)
    ax = axes[0]
    for m, cd, colour in RUNS:
        t = tables[(m, cd)]
        ax.plot(t.D_km, t.moment_recovery, "o-", color=colour, ms=3, lw=1, label=f"{m}, Cd {cd}")
    ax.axhline(1, color="0.5", lw=0.6)
    ax.set_xlabel("input depth D (km)"); ax.set_ylabel("potency recovered / input"); ax.legend(fontsize=5)
    ax.set_title(r"$\bf{A}$  Moment recovery", loc="left", fontsize=7)
    ax = axes[1]
    sel = prof[(prof.run == "default_nuggetcd") & (prof.medium == "sfcvm")]
    depths = sorted(sel.D_km.unique())
    for D, colour in zip(depths, plt.cm.magma(np.linspace(0.15, 0.85, len(depths)))):
        p = sel[sel.D_km == D].sort_values("z_top_km")
        ax.plot(p.slip_mm, -(p.z_top_km + 0.05), "-", color=colour, lw=1.2, label=f"D {D:g} km")   # bin centres
        ax.plot([10, 10, 0, 0], [0, -D, -D, -2.5], ":", color=colour, lw=0.8)                        # input: exact step
    ax.set_xlabel("section-mean slip (mm); dotted = input"); ax.set_ylabel("depth (km)")
    ax.set_ylim(-2.5, 0); ax.set_xlim(-0.3, 13); ax.legend(fontsize=5, loc="lower right")
    ax.set_title(r"$\bf{B}$  Recovered depth profile, selected model", loc="left", fontsize=7)
    ax = axes[2]
    for m, cd, colour in RUNS:
        t = tables[(m, cd)]
        ax.plot(t.D_km, t.mean_res_west300_mm, "v-", color=colour, ms=3, lw=1, label=f"{m}, Cd {cd}: west")
        ax.plot(t.D_km, t.mean_res_east300_mm, "^--", color=colour, ms=3, lw=1, label=f"{m}, Cd {cd}: east")
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xlabel("input depth D (km)"); ax.set_ylabel("mean residual within 300 m (mm)")
    ax.set_title(r"$\bf{C}$  Recovery residual bias, west vs east", loc="left", fontsize=7)
    ax.legend(fontsize=4.5, ncol=2)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


if __name__ == "__main__":
    main()
