#!/usr/bin/env python3
"""
DRAFT (2026-09-29): what drove the top tie. The final configuration without
the tie at sigma 12, 24 and 48, against sigma 12 with the 0.2 km tie. The tie
was adopted on 2026-09-28 at sigma 24 on the 0.7 field (drafts/07/top_tie_test.py),
where the untied model peaked below the surface.

Runs the machinery cells [1]-[7] of scripts/07_slip_model_tests.py unchanged.
Run from the repo root in opera_disp_feb13 (PROJ_LIB pointed at the env).
Output: Figures_<date>/test_tie_motivation.png, outputs/07_slip_model_tests/tie_motivation.csv
"""
import os

source = open(os.path.join("scripts", "07_slip_model_tests.py")).read()
exec(source.split("# %% [markdown]\n# ## [8]")[0])

cases = [("sigma 12, no tie", 12, 0.0), ("sigma 24, no tie", 24, 0.0),
         ("sigma 48, no tie", 48, 0.0), ("sigma 12, tie 0.2 km (model)", 12, 0.2)]
rows, tables = [], {}
for label, sigma, tie in cases:
    row, flt, _, _ = run_case("tie_motivation", label, sigma=sigma, tie=tie, record=False)
    rows.append(dict(label=label, sigma=sigma, tie_km=tie, **{k: row[k] for k in
                     ("Mw", "rms_mm", "surface_pct", "peak_slip_mm", "peak_depth_m")}))
    tables[label] = slip_table(flt)
summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(out_dir, "tie_motivation.csv"), index=False)
print(summary.round(3).to_string(index=False))

mpl_paper_style(base=7)
fig = plt.figure(figsize=(7.25, 5.2), constrained_layout=True)
grid = fig.add_gridspec(3, 2, width_ratios=[2.2, 1])
nodes_, faces_ = unrolled["smooth12"]
s0 = float(lat_to_km_pp([ref_fault.lati[0]])[0])
vmax = max(t.slip_mm.max() for t in tables.values())
for k, label in enumerate([cases[0][0], cases[1][0], cases[3][0]]):
    ax = fig.add_subplot(grid[k, 0])
    pc = ax.tripcolor(s0 + nodes_[:, 0], nodes_[:, 1], faces_, facecolors=tables[label].slip_mm.values,
                      cmap="viridis", vmin=0, vmax=vmax, edgecolors="none")
    ax.set_xlim(54, 65); ax.set_ylim(1.0, 0)
    ax.set_ylabel("Depth (km)")
    ax.set_title(f"{'ABC'[k]}  {label}", loc="left", fontsize=7)
    if k == 2:
        ax.set_xlabel("km from Point Pinole")
fig.colorbar(pc, ax=fig.axes, location="left", shrink=0.5, label="Slip (mm)", pad=0.08)
ax = fig.add_subplot(grid[:, 1])
for label, table in tables.items():
    peak_s = table.loc[table.slip_mm.idxmax(), "s_km"]
    column = table[(table.s_km - peak_s).abs() < 0.3]
    by_depth = column.groupby(np.round(column.z_c, 3)).slip_mm.max()
    ax.plot(by_depth.values, by_depth.index, "o-", ms=2, lw=1, label=label)
ax.set_ylim(1.0, 0); ax.set_xlabel("Slip at the peak column (mm)"); ax.set_ylabel("Depth (km)")
ax.legend(fontsize=5.5)
ax.set_title("D  Slip with depth at the peak", loc="left", fontsize=7)
f_fig = os.path.join(fig_dir, "test_tie_motivation.png")
fig.savefig(f_fig, dpi=300)
print("wrote", f_fig)
