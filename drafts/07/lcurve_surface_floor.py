#!/usr/bin/env python3
"""
DRAFT (2026-09-29): does the surface-constraint error floor move the L-curve
corner? The L-curve is driven mostly by the surface chi2, so the floor (0.5 mm,
set before the L-curve rule was adopted) could set the sigma pick the way the
top tie did. The L-curve of 07_slip_model [8] without the tie, at floors 0.25,
0.5 and 1.0 mm. Machinery cells [1]-[7] of
scripts/07_slip_model_tests.py, unchanged.
Output: outputs/07_slip_model_tests/lcurve_surface_floor.csv, Figures_<date>/test_lcurve_surface_floor.png
"""
import os

source = open(os.path.join("scripts", "07_slip_model_tests.py")).read()
exec(source.split("# %% [markdown]\n# ## [8]")[0])

sigma_grid = [4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128]   # 07_slip_model [1]


def corner_of(table):
    """07_slip_model [8], verbatim: maximum curvature of the log L-curve."""
    log_sigma = np.log10(table.sigma.values)
    spline_misfit = CubicSpline(log_sigma, np.log10(np.sqrt(table.chi2_total.values)))
    spline_rough = CubicSpline(log_sigma, np.log10(table.prior_norm.values))
    fine_log_sigma = np.linspace(log_sigma[1], log_sigma[-2], 400)
    d_misfit, dd_misfit = spline_misfit(fine_log_sigma, 1), spline_misfit(fine_log_sigma, 2)
    d_rough, dd_rough = spline_rough(fine_log_sigma, 1), spline_rough(fine_log_sigma, 2)
    curvature = np.abs(d_misfit * dd_rough - d_rough * dd_misfit) / (d_misfit ** 2 + d_rough ** 2) ** 1.5
    corner = float(10 ** fine_log_sigma[np.argmax(curvature)])
    return corner, float(table.sigma[table.sigma >= corner].min())


rows = []
for floor in (0.25, 0.5, 1.0):
    for sigma in sigma_grid:
        row, flt, _, _ = run_case("lcurve_floor", f"floor{floor:g} sigma{sigma:g}", sigma=sigma, tie=0.0, floor=floor, record=False)
        slip_vec = flt.slip[:, 0]
        row["prior_norm"] = float(np.sqrt(np.dot(slip_vec, np.linalg.solve(flt.Cm / sigma ** 2, slip_vec))))
        row["chi2_total"] = row["chi2"] + row["chi2_surf"]
        row["floor_mm"] = floor
        rows.append(row)
curves = pd.DataFrame(rows)
summary = []
for floor, table in curves.groupby("floor_mm"):
    corner, pick = corner_of(table.reset_index(drop=True))
    at_pick = table[table.sigma == pick].iloc[0]
    summary.append(dict(floor_mm=floor, corner=corner, pick=pick, Mw=at_pick.Mw, peak_slip_mm=at_pick.peak_slip_mm,
                        peak_depth_m=at_pick.peak_depth_m, surface_pct=at_pick.surface_pct, rms_mm=at_pick.rms_mm))
    curves.loc[curves.floor_mm == floor, "corner"] = corner
summary = pd.DataFrame(summary)
curves.to_csv(os.path.join(out_dir, "lcurve_surface_floor.csv"), index=False)
print(summary.round(3).to_string(index=False))

mpl_paper_style(base=7)
fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.8), constrained_layout=True)
for (floor, table), colour in zip(curves.groupby("floor_mm"), ("#E69F00", "#000000", "#0072B2")):
    label = f"floor {floor:g} mm" + (" (model)" if floor == 0.5 else "")
    pick = float(summary.loc[summary.floor_mm == floor, "pick"].iloc[0])
    axes[0].plot(np.sqrt(table.chi2_total), table.prior_norm, "o-", ms=2.5, lw=0.8, color=colour, label=f"{label}: pick sigma {pick:g}")
    chosen = table[table.sigma == pick]
    axes[0].plot(np.sqrt(chosen.chi2_total), chosen.prior_norm, "*", ms=9, color=colour)
    axes[1].plot(table.sigma, table.Mw, "o-", ms=2.5, lw=0.8, color=colour, label=label)
axes[0].set_xscale("log"); axes[0].set_yscale("log")
axes[0].set_xlabel(r"Weighted misfit $\sqrt{\chi^2}$"); axes[0].set_ylabel(r"Roughness $\sqrt{m^T C^{-1} m}$")
axes[0].legend(fontsize=6); axes[0].set_title("A  L-curve at three surface error floors (no tie)", loc="left", fontsize=7)
axes[1].set_xscale("log"); axes[1].set_xlabel("sigma"); axes[1].set_ylabel("Mw (30 GPa)")
axes[1].legend(fontsize=6); axes[1].set_title("B  Mw against sigma", loc="left", fontsize=7)
f_fig = os.path.join(fig_dir, "test_lcurve_surface_floor.png")
fig.savefig(f_fig, dpi=300)
print("wrote", f_fig)
