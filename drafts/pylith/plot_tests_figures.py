#!/usr/bin/env python3
"""
Synthetic-recovery figure of pylith_sfcvm_tests.py, redrawn from its saved products alone
(gf/recovery_<run>_<medium>.csv, gf/recovery_profiles.csv), so it can be restyled without re-running the tests.
Panel (b)'s input is drawn as an exact step (10 mm from the surface to D, 0 below).
Run in opera_disp_feb13:  python drafts/pylith/plot_tests_figures.py   -> figures/recovery_synthetic.png
"""
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OK = dict(blue="#0072B2", orange="#E69F00", vermillion="#D55E00", black="#000000")
RUNS = [("sfcvm", "nugget", OK["vermillion"]), ("sfcvm", "full", OK["blue"]), ("homog", "nugget", OK["orange"]), ("homog", "full", OK["black"])]
RUN_NAME = {"nugget": "default_nuggetcd", "full": "default"}


def main():
    matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial"], "font.size": 12,
                                "savefig.facecolor": "white", "figure.facecolor": "white"})
    tables = {(m, cd): pd.read_csv(os.path.join(HERE, "gf", f"recovery_{RUN_NAME[cd]}_{m}.csv")) for m, cd, _ in RUNS}
    prof = pd.read_csv(os.path.join(HERE, "gf", "recovery_profiles.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    ax = axes[0]
    for m, cd, colour in RUNS:
        t = tables[(m, cd)]
        ax.plot(t.D_km, t.moment_recovery, "o-", color=colour, label=f"{m}, Cd {cd}")
    ax.axhline(1, color="0.5", lw=0.8)
    ax.set_xlabel("input depth D (km)"); ax.set_ylabel("potency recovered / input"); ax.legend(fontsize=9)
    ax.set_title("(a) moment recovery", loc="left")
    ax = axes[1]
    sel = prof[(prof.run == "default_nuggetcd") & (prof.medium == "sfcvm")]
    depths = sorted(sel.D_km.unique())
    for D, colour in zip(depths, plt.cm.magma(np.linspace(0.15, 0.85, len(depths)))):
        p = sel[sel.D_km == D].sort_values("z_top_km")
        ax.plot(p.slip_mm, -(p.z_top_km + 0.05), "-", color=colour, lw=2, label=f"D {D:g} km")   # bin centres
        ax.plot([10, 10, 0, 0], [0, -D, -D, -2.5], ":", color=colour, lw=1.2)                        # input: exact step
    ax.set_xlabel("section-mean slip (mm); dotted = input"); ax.set_ylabel("depth (km)")
    ax.set_ylim(-2.5, 0); ax.set_xlim(-0.3, 13); ax.legend(fontsize=9, loc="lower right")
    ax.set_title("(b) recovered depth profile, selected model (sfcvm, nugget Cd)", loc="left", fontsize=11)
    ax = axes[2]
    for m, cd, colour in RUNS:
        t = tables[(m, cd)]
        ax.plot(t.D_km, t.mean_res_west300_mm, "v-", color=colour, label=f"{m}, Cd {cd}: west")
        ax.plot(t.D_km, t.mean_res_east300_mm, "^--", color=colour, label=f"{m}, Cd {cd}: east")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("input depth D (km)"); ax.set_ylabel("mean residual within 300 m (mm)")
    ax.set_title("(c) recovery residual bias, west vs east (artefact flag)", loc="left", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    out = os.path.join(HERE, "figures", "recovery_synthetic.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    main()
