#!/usr/bin/env python3
"""
DRAFT (2026-09-29): relax the along-strike smoothing (Dani: the model misses
the sharp ends of the creeping section in Fig 2 K). The individual crossings
step from 3.7 to 9.3 mm over km 57.4-57.8; the surface constraint (a running
median, 0.4 km window every 0.2 km) and the along-strike correlation length
lam_s (0.3 km) both smooth it. Here lam_s 0.3, 0.15, 0.1 and 0.05 km, with the
current constraint and a finer one (0.2 km window every 0.1 km, the crossing
spacing), no top tie (Dani 2026-09-29), each at the sigma its own L-curve picks
(lam enters the prior's amplitude as well as its shape, 07_slip_model_tests [14]).

Judged on the surface slip against the individual crossings: the steepest
change over 0.3 km at each end of the section, the surface misfit, and Mw.

Runs the machinery cells [1]-[7] of scripts/07_slip_model_tests.py unchanged.
Output: outputs/07_slip_model_tests/along_strike_smoothing.csv, Figures_<date>/test_along_strike_smoothing.png
"""
import os

source = open(os.path.join("scripts", "07_slip_model_tests.py")).read()
exec(source.split("# %% [markdown]\n# ## [8]")[0])

lam_s_values = [0.3, 0.15, 0.1, 0.05]
constraints = {"current (0.4 km window, 0.2 km step)": (0.4, 0.2, 4), "fine (0.3 km window, 0.1 km step)": (0.3, 0.1, 3)}
surf_min_n = 4                    # crossings needed in a window (07: 4; the fine window holds 3)


def make_surface(floor_mm):
    """07_slip_model_tests [5] make_surface, verbatim except the minimum
    number of crossings per window, surf_min_n (4 in the notebook)."""
    rows = []
    for k_centre in np.arange(np.ceil(crossings_all.km.min()),
                              crossings_all.km.max(), surf_step_km):
        in_win = np.abs(crossings_all.km - k_centre) <= surf_win_km / 2
        amps = crossings_all.mean_event_mm[in_win].dropna()
        if len(amps) < surf_min_n:
            continue
        med = float(np.median(amps))
        err = max(1.4826 * float(np.median(np.abs(amps - med))) / np.sqrt(len(amps)),
                  floor_mm)
        lat_c = float(np.median(crossings_all.lat[in_win]))
        lon_c = float(np.interp(lat_c, trace_south_north[:, 1], trace_south_north[:, 0]))
        rows.append((lon_c, lat_c, k_centre, med, err))
    rows = np.array(rows)
    rows = rows[(rows[:, 2] >= km_window[0]) & (rows[:, 2] <= km_window[1])]
    ss = surfaceslip(f"crossing stacks f{floor_mm:g} w{surf_win_km:g}", lon0=frame_lon0,
                     lat0=frame_lat0, verbose=False)
    ss.read_from_binary(rows[:, 3], rows[:, 0], rows[:, 1], err=rows[:, 4])
    ss.buildCd()
    return ss
sigma_grid = [4, 6, 8, 12, 16, 24, 32, 64]
edge_span_km = 0.3                 # the steepest change over this along-strike span


def corner_pick(table):
    """07_slip_model [8]: maximum curvature of the log L-curve, then the grid sigma at or above it."""
    log_sigma = np.log10(table.sigma.values)
    spline_misfit = CubicSpline(log_sigma, np.log10(np.sqrt(table.chi2_total.values)))
    spline_rough = CubicSpline(log_sigma, np.log10(table.prior_norm.values))
    fine_log_sigma = np.linspace(log_sigma[1], log_sigma[-2], 400)
    d_misfit, dd_misfit = spline_misfit(fine_log_sigma, 1), spline_misfit(fine_log_sigma, 2)
    d_rough, dd_rough = spline_rough(fine_log_sigma, 1), spline_rough(fine_log_sigma, 2)
    curvature = np.abs(d_misfit * dd_rough - d_rough * dd_misfit) / (d_misfit ** 2 + d_rough ** 2) ** 1.5
    corner = float(10 ** fine_log_sigma[np.argmax(curvature)])
    return corner, float(table.sigma[table.sigma >= corner].min())


def steepest_steps(km, slip):
    """Largest slip change over edge_span_km at the north end (km < 60) and the
    south end (km >= 60) of the section."""
    order = np.argsort(km); km, slip = km[order], slip[order]
    steps = []
    for part in (km < 60.0, km >= 60.0):
        k_, s_ = km[part], slip[part]
        best = 0.0
        for i in range(len(k_)):
            window = (k_ >= k_[i]) & (k_ <= k_[i] + edge_span_km)
            best = max(best, float(np.ptp(s_[window])))
        steps.append(best)
    return steps


crossings_ok = np.isfinite(crossings_all.mean_event_mm.values)
cross_north, cross_south = steepest_steps(crossings_all.km.values[crossings_ok], crossings_all.mean_event_mm.values[crossings_ok])
print(f"individual crossings: steepest change over {edge_span_km} km: north {cross_north:.1f} mm, south {cross_south:.1f} mm")

rows, surfaces = [], {}
for constraint_name, (win_km, step_km, min_n) in constraints.items():
    surf_win_km, surf_step_km, surf_min_n = win_km, step_km, min_n   # make_surface reads these
    for lam_s in lam_s_values:
        lam = (lam_s, lam_xz[1])
        curve = []
        for sigma in sigma_grid:
            row, flt, _, _ = run_case("lam_s", f"{constraint_name} lam{lam_s} s{sigma}", sigma=sigma, lam=lam, tie=0.0, record=False)
            slip_vec = flt.slip[:, 0]
            row["prior_norm"] = float(np.sqrt(np.dot(slip_vec, np.linalg.solve(flt.Cm / sigma ** 2, slip_vec))))
            row["chi2_total"] = row["chi2"] + row["chi2_surf"]
            curve.append((row, slip_table(flt)))
        table = pd.DataFrame([r for r, _ in curve])
        corner, pick = corner_pick(table)
        row, slips = curve[int(np.argmax(table.sigma.values == pick))]
        surface_row = slips[slips.z_top <= 1e-6].sort_values("km_pp")
        north, south = steepest_steps(surface_row.km_pp.values, surface_row.slip_mm.values)
        label = f"lam_s {lam_s:g} km, {constraint_name.split(' (')[0]} constraint"
        rows.append(dict(label=label, constraint=constraint_name, lam_s_km=lam_s, corner=corner, sigma=pick, Mw=row["Mw"],
                         rms_mm=row["rms_mm"], rms_near_mm=row["rms_near_mm"], surface_rms_mm=row["surf_rms_mm"],
                         surface_pct=row["surface_pct"], peak_slip_mm=row["peak_slip_mm"], peak_depth_m=row["peak_depth_m"],
                         north_step_mm=north, south_step_mm=south))
        surfaces[label] = surface_row
        print(f"{label}: corner {corner:.1f} -> sigma {pick:g}; Mw {row['Mw']:.3f}; steepest step N {north:.1f} S {south:.1f} mm "
              f"(crossings {cross_north:.1f} / {cross_south:.1f}); surface rms {row['surf_rms_mm']:.2f} mm; rms {row['rms_mm']:.3f}")
summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(out_dir, "along_strike_smoothing.csv"), index=False)
print(summary.round(3).to_string(index=False))

mpl_paper_style(base=7)
fig, axes = plt.subplots(2, 1, figsize=(7.25, 4.6), sharex=True, constrained_layout=True)
shades = {0.3: 1.0, 0.15: 0.75, 0.1: 0.5, 0.05: 0.3}
for ax, (k, constraint_name) in zip(axes, enumerate(constraints)):
    ax.plot(crossings_all.km, crossings_all.mean_event_mm, "o", ms=1.8, color="0.55", label="individual crossings")
    for lam_s in lam_s_values:
        label = f"lam_s {lam_s:g} km, {constraint_name.split(' (')[0]} constraint"
        s_ = summary.set_index("label").loc[label]
        ax.plot(surfaces[label].km_pp, surfaces[label].slip_mm, "-", lw=1.0, color="#0072B2", alpha=shades[lam_s],
                label=f"lam_s {lam_s:g} km (sigma {s_.sigma:g}, Mw {s_.Mw:.2f})")
    ax.set_xlim(54, 65); ax.set_ylabel("Surface slip (mm)")
    ax.legend(fontsize=6, loc="upper right")
    ax.set_title(f"{'AB'[k]}  No tie, {constraint_name} surface constraint", loc="left", fontsize=7)
axes[-1].set_xlabel("km from Point Pinole")
f_fig = os.path.join(fig_dir, "test_along_strike_smoothing.png")
fig.savefig(f_fig, dpi=300)
print("wrote", f_fig)
