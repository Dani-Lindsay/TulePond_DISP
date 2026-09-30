#!/usr/bin/env python3
"""
DRAFT (2026-09-29): keep or drop the top tie, judged on the models, not only
on Mw (Dani: "we added the uniform constraint because of the weird bulge").
Each option at the sigma its own L-curve picks (drafts/07/lcurve_no_tie.py:
no tie -> sigma 8, tie 0.2 km -> sigma 12), plus the untied model at sigma 12.

Figure rows: the three slip sections (top 1 km); then slip with depth at the
peak column, surface slip along strike against the crossings, and the
near-fault residual against distance from the trace.
Runs the machinery cells [1]-[7] of scripts/07_slip_model_tests.py unchanged.
Output: Figures_<date>/test_tie_decision.png, outputs/07_slip_model_tests/tie_decision.csv
"""
import os

source = open(os.path.join("scripts", "07_slip_model_tests.py")).read()
exec(source.split("# %% [markdown]\n# ## [8]")[0])

cases = [("no tie, sigma 8 (its L-curve pick)", 8, 0.0, "#E69F00"),
         ("no tie, sigma 12", 12, 0.0, "#999999"),
         ("tie 0.2 km, sigma 12 (current model)", 12, 0.2, "#0072B2")]
rows, results = [], {}
for label, sigma, tie, colour in cases:
    row, flt, ds, surface = run_case("tie_decision", label, sigma=sigma, tie=tie, record=False)
    table = slip_table(flt)
    resid = ds.vel - ds.synth
    dist = dist_to_trace(ds, flt)
    near = dist <= 0.5
    rows.append(dict(label=label, sigma=sigma, tie_km=tie, Mw=row["Mw"], rms_mm=row["rms_mm"],
                     rms_within_200m_mm=row["rms_near_mm"], rms_within_500m_mm=float(np.sqrt(np.mean(resid[near] ** 2))),
                     surface_rms_mm=row["surf_rms_mm"], surface_pct=row["surface_pct"],
                     peak_slip_mm=row["peak_slip_mm"], peak_depth_m=row["peak_depth_m"]))
    results[label] = dict(table=table, resid=resid, dist=dist, surface=surface, colour=colour)
summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(out_dir, "tie_decision.csv"), index=False)
print(summary.round(3).to_string(index=False))

mpl_paper_style(base=7)
fig = plt.figure(figsize=(7.25, 6.4), constrained_layout=True)
grid = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1.6])
nodes_, faces_ = unrolled["smooth12"]
s0 = float(lat_to_km_pp([ref_fault.lati[0]])[0])
vmax = max(r["table"].slip_mm.max() for r in results.values())
for k, (label, res) in enumerate(results.items()):
    ax = fig.add_subplot(grid[k, :])
    pc = ax.tripcolor(s0 + nodes_[:, 0], nodes_[:, 1], faces_, facecolors=res["table"].slip_mm.values,
                      cmap="viridis", vmin=0, vmax=vmax, edgecolors="none")
    ax.set_xlim(54, 65); ax.set_ylim(1.0, 0); ax.set_ylabel("Depth (km)")
    ax.set_title(f"{'ABC'[k]}  {label}", loc="left", fontsize=7, color=res["colour"])
fig.colorbar(pc, ax=fig.axes[:3], location="right", shrink=0.8, label="Slip (mm)")

ax = fig.add_subplot(grid[3, 0])                                  # D: slip with depth at the peak column
for label, res in results.items():
    table = res["table"]
    peak_s = table.loc[table.slip_mm.idxmax(), "s_km"]
    column = table[(table.s_km - peak_s).abs() < 0.3]
    by_depth = column.groupby(np.round(column.z_c, 3)).slip_mm.max()
    ax.plot(by_depth.values, by_depth.index, "o-", ms=2, lw=1, color=res["colour"])
ax.set_ylim(0.8, 0); ax.set_xlabel("Slip at the peak column (mm)"); ax.set_ylabel("Depth (km)")
ax.set_title("D  Slip with depth", loc="left", fontsize=7)

ax = fig.add_subplot(grid[3, 1])                                  # E: surface slip along strike vs crossings
ax.plot(lat_to_km_pp(crossing_profile.lat), crossing_profile.vel, "-", color="#A50F15", lw=1.4, label="crossings")
for label, res in results.items():
    surface_row = res["table"][res["table"].z_top <= 1e-6].sort_values("km_pp")
    ax.plot(surface_row.km_pp, surface_row.slip_mm, "-", color=res["colour"], lw=0.9)
ax.set_xlim(54, 65); ax.set_xlabel("km from Point Pinole"); ax.set_ylabel("Surface slip (mm)")
ax.legend(fontsize=6)
ax.set_title("E  Surface slip", loc="left", fontsize=7)

ax = fig.add_subplot(grid[3, 2])                                  # F: residual rms against distance from the trace
bins = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])
for label, res in results.items():
    idx = np.digitize(res["dist"], bins) - 1
    rms_bin = [np.sqrt(np.mean(res["resid"][idx == j] ** 2)) if (idx == j).sum() >= 5 else np.nan for j in range(len(bins) - 1)]
    ax.plot(0.5 * (bins[1:] + bins[:-1]), rms_bin, "o-", ms=2, lw=1, color=res["colour"])
ax.set_xscale("log"); ax.set_xlabel("Distance from the trace (km)"); ax.set_ylabel("Residual rms (mm)")
ax.set_title("F  Fit against distance", loc="left", fontsize=7)
f_fig = os.path.join(fig_dir, "test_tie_decision.png")
fig.savefig(f_fig, dpi=300)
print("wrote", f_fig)
