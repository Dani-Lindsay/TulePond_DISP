# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
# ---

# %% [markdown]
# # Uncertainty and resolution of the PyLith + SFCVM slip model (companion to pylith_sfcvm_gf.py)
#
# ```
# Created on Wed Sep 30 2026 (daytime agent run, DRAFT)
# @author: daniellelindsay
# ```
#
# For the SELECTED model (vertical fault, aligned SFCVM, nugget-only Cd, surface constraint ON; Dani 2026-09-30),
# its primary companion (07's full correlated Cd) and the homogeneous PyLith equivalents, on the cached order-1
# Green's functions:
#
# 1. **Bootstrap** (07 [11]): noise drawn from the inversion's own Cd (Cholesky) added to the model prediction and
#    re-inverted at the selected sigma, N = 07's `n_boot`; mean and 2-sigma of Mw30, Mw(SFCVM mu), potency,
#    surface %, peak slip and its depth. Plus the same numbers at the neighbouring grid sigmas.
# 2. **Synthetic recovery** (07_slip_model_tests [11]): 10 mm uniform slip from the surface to D over the SSE section
#    (tapered), forward through the PyLith G, noise from the same Cd, inverted with the same setup: moment and depth
#    recovery, and the west/east near-field residual (an artefact flag).
# 3. **Fixed-depth scan** (07_slip_model_tests [9]): uniform slip to D, amplitude + constant by whitened least
#    squares with the run's Cd, chi2(D), bracket delta chi2 <= 4, Mw per D.
#
# The notebook `pylith_sfcvm_gf.py` is exec'd through cell [8] (mesh, materials, cached PyLith G, 07's data and
# machinery, `invert_tents`), so nothing is re-implemented here beyond the loops; the depth-scan least squares and
# the window/shape definitions are those of the tests notebook, applied to the tent (vertex) parameters.
# Products: gf/uncertainty_<run>_<medium>.csv, gf/recovery_<run>_<medium>.csv (+ _profiles), gf/depth_scan_<run>_<medium>.csv,
# figures/uncertainty_*.png, figures/recovery_*.png, figures/depth_scan_*.png (run = default_nuggetcd | default).

# %%
## [1]
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
os.environ.setdefault("PYLITH_MEDIA", "homog,sfcvm")       # both media, default mesh, vertical, order 1, default points
for k in ("PYLITH_DIP", "PYLITH_POINTS", "PYLITH_ORDER", "PYLITH_CD", "PYLITH_SURFACE", "PYLITH_MESH_PRESET"):
    os.environ.pop(k, None)
t_setup = time.time()
notebook_src = open(os.path.join(HERE, "pylith_sfcvm_gf.py")).read()
head = notebook_src.split("# %% [markdown]\n# ## [9]")[0]     # cells [1]-[8]: G, 07's machinery, invert_tents, potency, moment
assert "def invert_tents" in head and "def potency_m3" in head, "notebook layout changed: cell [8] not found"
exec(head, globals())
print(f"notebook cells [1]-[8] exec'd in {time.time() - t_setup:.0f} s; media {list(media)}; cd_mode {cd_mode!r}")
assert use_surface and dip == "90" and points_set == "default" and elem_order == "1"

tests_src = open(os.path.join(REPO, "scripts", "07b_uniform_model_tests.py")).read()
def param(name):
    """A parameter list/number from the tests notebook source (derived, not re-typed)."""
    m = re.search(rf"^{name}\s*=\s*(\[[^\]]*\]|[0-9.]+)", tests_src, re.S | re.M)
    assert m, f"{name} not found in 07b_uniform_model_tests.py"
    return eval(m.group(1))
recovery_depths, scan_taper, scan_depths = param("recovery_depths"), param("scan_taper"), param("scan_depths")
scan_km = (creep_lo_km + scan_taper, creep_hi_km - scan_taper)      # tests notebook [1]: SSE section tapered at its ends
print(f"tests parameters: n_boot {n_boot}, recovery depths {recovery_depths}, scan window km {scan_km[0]:.1f}-{scan_km[1]:.1f} "
      f"(taper {scan_taper} km), {len(scan_depths)} scan depths")
matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial"], "font.size": 12,
                            "savefig.facecolor": "white", "figure.facecolor": "white"})
OK = dict(blue="#0072B2", orange="#E69F00", vermillion="#D55E00", purple="#CC79A7", black="#000000", grey="#999999")

# the runs: (medium, Cd mode); the selected sigma of each from its results csv
RUNS = [("sfcvm", "nugget"), ("sfcvm", "full"), ("homog", "nugget"), ("homog", "full")]
CD_TAG = {"full": "", "nugget": "_nuggetcd"}
def run_name(cd):
    return "default" + CD_TAG[cd]
def selected_sigma(medium, cd):
    res = pd.read_csv(os.path.join(HERE, "gf", f"results_comparison{CD_TAG[cd]}.csv")).set_index("model")
    return float(res.loc[f"PyLith {medium}", "sigma"])
def set_cd(cd):
    """invert_tents reads the notebook's global cd_mode."""
    globals()["cd_mode"] = cd
imp_depth_km = -imp_xyz[:, 2] / 1e3
in_window = np.clip(np.minimum(imp_km - scan_km[0], scan_km[1] - imp_km) / scan_taper + 1, 0, 1)   # tests notebook [9]
west300 = (signed_d_km < 0) & (signed_d_km >= -0.3)
east300 = (signed_d_km > 0) & (signed_d_km <= 0.3)
in_sec_surface = np.isin(np.arange(n_imp), surface_imp) & (imp_km >= creep_lo_km) & (imp_km <= creep_hi_km)


def numbers(medium, flt, ds, surface):
    """The notebook's [10] numbers for one solution."""
    slip = flt.slip[:, 0]
    res = ds.vel - ds.synth
    pot = potency_m3(slip)
    return dict(rms_mm=float(np.sqrt(np.mean(res ** 2))), rms_west300_mm=float(np.sqrt(np.mean(res[west300] ** 2))),
                rms_east300_mm=float(np.sqrt(np.mean(res[east300] ** 2))), chi2=float(np.dot(res, np.linalg.solve(ds.Cd, res))),
                surface_pct=float(100 * np.median(slip[in_sec_surface]) / crossing_median), potency_m3=pot,
                Mw_30gpa=float(mw(mu_ref * pot)), Mw_sfcvm=float(mw(moment_sfcvm(slip, medium))),
                peak_slip_mm=float(slip.max()), peak_depth_m=float(imp_depth_km[np.argmax(slip)] * 1e3))


REPORT = ["Mw_30gpa", "Mw_sfcvm", "potency_m3", "surface_pct", "peak_slip_mm", "peak_depth_m"]

# %% [markdown]
# ## [2] Bootstrap and sigma sensitivity (07 [11])

# %%
## [2]
rng = np.random.default_rng(11)                                   # 07 [11]'s seed
uncertainty = {}
for medium, cd in RUNS:
    set_cd(cd)
    sigma = selected_sigma(medium, cd)
    ref_flt, ref_ds, ref_surf = invert_tents(medium, sigma)
    ref = numbers(medium, ref_flt, ref_ds, ref_surf)
    chol_insar = np.linalg.cholesky(ref_ds.Cd + 1e-9 * np.eye(ref_ds.Cd.shape[0]))
    chol_surface = np.linalg.cholesky(ref_surf.Cd + 1e-9 * np.eye(ref_surf.Cd.shape[0]))
    synth_insar, synth_surface = ref_ds.synth.copy(), ref_surf.synth.copy()
    rows = []
    t0 = time.time()
    for k in range(n_boot):
        vel_i = synth_insar + np.dot(chol_insar, rng.standard_normal(synth_insar.size))
        vel_s = synth_surface + np.dot(chol_surface, rng.standard_normal(synth_surface.size))
        flt, ds, surf = invert_tents(medium, sigma, vel_insar=vel_i, vel_surface=vel_s)
        rows.append(dict(kind="boot", k=k, sigma=sigma, **numbers(medium, flt, ds, surf)))
        print(f"  {medium} {cd} boot {k + 1}/{n_boot}: Mw30 {rows[-1]['Mw_30gpa']:.3f}, surface {rows[-1]['surface_pct']:.0f} %")
    boot = pd.DataFrame(rows)
    summary = [dict(kind="reference", k=-1, sigma=sigma, **ref),
               dict(kind="boot_mean", k=-1, sigma=sigma, **{c: float(boot[c].mean()) for c in REPORT}),
               dict(kind="boot_2sd", k=-1, sigma=sigma, **{c: float(2 * boot[c].std(ddof=1)) for c in REPORT})]
    # sigma sensitivity: the neighbouring grid sigmas (same data, no noise)
    i_sig = sigma_grid.index(sigma)
    for s_nb in [sigma_grid[j] for j in (i_sig - 1, i_sig + 1) if 0 <= j < len(sigma_grid)]:
        flt, ds, surf = invert_tents(medium, s_nb)
        summary.append(dict(kind="sigma_neighbour", k=-1, sigma=s_nb, **numbers(medium, flt, ds, surf)))
    table = pd.concat([pd.DataFrame(summary), boot], ignore_index=True)
    table["provenance"] = f"pylith_sfcvm_tests.py {time.strftime('%Y-%m-%d %H:%M')}; G {run_names[medium]}; Cd {cd}; seed 11"
    out = os.path.join(HERE, "gf", f"uncertainty_{run_name(cd)}_{medium}.csv")
    table.to_csv(out, index=False)
    uncertainty[(medium, cd)] = table
    m2 = table[table.kind == "boot_mean"].iloc[0]; s2 = table[table.kind == "boot_2sd"].iloc[0]
    print(f"{medium} {cd} (sigma {sigma:g}, {n_boot} realisations, {time.time() - t0:.0f} s): "
          + ", ".join(f"{c} {m2[c]:.3g} +/- {s2[c]:.2g}" for c in REPORT) + f"; wrote {os.path.basename(out)}")

fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
for ax, (medium, cd) in zip(axes.ravel(), RUNS):
    t = uncertainty[(medium, cd)]
    b = t[t.kind == "boot"]; ref = t[t.kind == "reference"].iloc[0]
    ax.hist(b.Mw_30gpa, bins=12, color=OK["blue"] if cd == "full" else OK["vermillion"], alpha=0.8)
    ax.axvline(ref.Mw_30gpa, color="k", lw=1.5, label=f"reference {ref.Mw_30gpa:.2f}")
    m, s = b.Mw_30gpa.mean(), 2 * b.Mw_30gpa.std(ddof=1)
    ax.axvspan(m - s, m + s, color=OK["grey"], alpha=0.25, label=f"mean {m:.2f} +/- {s:.2f} (2 sd)")
    for _, r in t[t.kind == "sigma_neighbour"].iterrows():
        ax.axvline(r.Mw_30gpa, color=OK["orange"], ls="--", lw=1.2, label=f"sigma {r.sigma:g}: {r.Mw_30gpa:.2f}")
    ax.set_xlabel("Mw (30 GPa)"); ax.set_ylabel("realisations"); ax.legend(fontsize=9)
    ax.set_title(f"{medium}, Cd {cd}: bootstrap ({n_boot}) and neighbouring sigmas", loc="left", fontsize=12)
fig.savefig(os.path.join(FIG, "uncertainty_bootstrap.png"), dpi=300)

# %% [markdown]
# ## [3] Synthetic recovery (07_slip_model_tests [11])

# %%
## [3]
recovery_rng = np.random.default_rng(7)                          # tests notebook's seed
recovery_tables, profiles = {}, []
for medium, cd in RUNS:
    set_cd(cd)
    sigma = selected_sigma(medium, cd)
    _, cd_ds, cd_surf = invert_tents(medium, sigma)               # the run's Cd (InSAR) and the surface-constraint Cd
    chol = np.linalg.cholesky(cd_ds.Cd + 1e-9 * np.eye(cd_ds.Cd.shape[0]))
    surf_ref, G_surf = surface_constraint()
    noise_w = float(np.sqrt(np.mean(np.diag(cd_ds.Cd)[west300]))); noise_e = float(np.sqrt(np.mean(np.diag(cd_ds.Cd)[east300])))
    rows = []
    for D in recovery_depths:
        slip_in = 10.0 * in_window * (imp_depth_km < D)
        pot_in = potency_m3(slip_in)
        vel_i = G[medium].dot(slip_in) + np.dot(chol, recovery_rng.standard_normal(cd_ds.vel.size))
        vel_s = G_surf.dot(slip_in) + np.sqrt(np.diag(surf_ref.Cd)) * recovery_rng.standard_normal(surf_ref.vel.size)
        flt, ds, surf = invert_tents(medium, sigma, vel_insar=vel_i, vel_surface=vel_s)
        num = numbers(medium, flt, ds, surf)
        res = ds.vel - ds.synth
        slip_out = flt.slip[:, 0]
        # depth extent recovered: shallowest depth below which the section-mean slip falls under 10 % of its surface value
        sec = (imp_km >= scan_km[0]) & (imp_km <= scan_km[1])
        zbins = np.arange(0, 3.01, 0.1)
        prof = np.array([slip_out[sec & (imp_depth_km >= z0) & (imp_depth_km < z0 + 0.1)].mean() if np.any(sec & (imp_depth_km >= z0) & (imp_depth_km < z0 + 0.1)) else np.nan for z0 in zbins[:-1]])
        below = np.nonzero(prof < 0.1 * np.nanmax(prof))[0]
        d_rec = float(zbins[below[0]]) if below.size else np.nan
        rows.append(dict(D_km=D, potency_in_m3=pot_in, potency_out_m3=num["potency_m3"], moment_recovery=num["potency_m3"] / pot_in,
                         Mw_in=float(mw(mu_ref * pot_in)), Mw_out=num["Mw_30gpa"], depth_10pct_km=d_rec,
                         peak_out_mm=num["peak_slip_mm"], peak_depth_m=num["peak_depth_m"], surface_pct=num["surface_pct"],
                         rms_mm=num["rms_mm"], rms_west300_mm=num["rms_west300_mm"], rms_east300_mm=num["rms_east300_mm"],
                         mean_res_west300_mm=float(res[west300].mean()), mean_res_east300_mm=float(res[east300].mean()),
                         noise_sd_west300_mm=noise_w, noise_sd_east300_mm=noise_e))
        profiles.append(pd.DataFrame(dict(run=run_name(cd), medium=medium, D_km=D, z_top_km=zbins[:-1], slip_mm=prof,
                                          input_mm=[10.0 * float(z0 + 0.05 < D) for z0 in zbins[:-1]])))
        print(f"  {medium} {cd} input 10 mm to {D:g} km: moment recovered {rows[-1]['moment_recovery']:.2f}, Mw {rows[-1]['Mw_in']:.2f} -> "
              f"{rows[-1]['Mw_out']:.2f}, depth(10 %) {d_rec:.1f} km, W/E rms {rows[-1]['rms_west300_mm']:.2f}/{rows[-1]['rms_east300_mm']:.2f} "
              f"(noise sd {noise_w:.2f}/{noise_e:.2f}), W/E mean residual {rows[-1]['mean_res_west300_mm']:+.2f}/{rows[-1]['mean_res_east300_mm']:+.2f} mm")
    table = pd.DataFrame(rows)
    table["provenance"] = f"pylith_sfcvm_tests.py {time.strftime('%Y-%m-%d %H:%M')}; G {run_names[medium]}; Cd {cd}; sigma {sigma:g}; seed 7"
    table.to_csv(os.path.join(HERE, "gf", f"recovery_{run_name(cd)}_{medium}.csv"), index=False)
    recovery_tables[(medium, cd)] = table
pd.concat(profiles).to_csv(os.path.join(HERE, "gf", "recovery_profiles.csv"), index=False)

import plot_tests_figures                                  # the recovery figure, drawn from the products just written
plot_tests_figures.main()

# %% [markdown]
# ## [4] Fixed-depth scan (07_slip_model_tests [9])

# %%
## [4]
scans = {}
for medium, cd in RUNS:
    set_cd(cd)
    _, cd_ds, _ = invert_tents(medium, selected_sigma(medium, cd))
    chol = np.linalg.cholesky(cd_ds.Cd + 1e-9 * np.eye(cd_ds.Cd.shape[0]))
    data_w = np.linalg.solve(chol, cd_ds.vel)                     # the ramp-removed data of the reference solution
    rows = []
    for D in [d for d in scan_depths if d <= imp_depth_km.max() + 1e-6]:
        shape = in_window * (imp_depth_km < D)
        design = np.c_[G[medium].dot(shape), np.ones(cd_ds.vel.size)]
        design_w = np.linalg.solve(chol, design)
        coef, *_ = np.linalg.lstsq(design_w, data_w, rcond=None)
        resid_w = data_w - design_w.dot(coef)
        pot = potency_m3(shape * coef[0])
        rows.append(dict(D_km=D, slip_mm=float(coef[0]), constant_mm=float(coef[1]), chi2=float(np.dot(resid_w, resid_w)),
                         potency_m3=pot, Mw=float(mw(mu_ref * max(pot, 1e-9)))))
    scan = pd.DataFrame(rows)
    scan["dchi2"] = scan.chi2 - scan.chi2.min()
    scan["provenance"] = f"pylith_sfcvm_tests.py {time.strftime('%Y-%m-%d %H:%M')}; G {run_names[medium]}; Cd {cd}"
    scan.to_csv(os.path.join(HERE, "gf", f"depth_scan_{run_name(cd)}_{medium}.csv"), index=False)
    scans[(medium, cd)] = scan
    best, bracket = scan.loc[scan.chi2.idxmin()], scan[scan.dchi2 <= 4]
    print(f"depth scan {medium} {cd}: best D {best.D_km:g} km ({best.slip_mm:.1f} mm, Mw {best.Mw:.2f}); delta chi2 <= 4: "
          f"{bracket.D_km.min():g}-{bracket.D_km.max():g} km, Mw {bracket.Mw.min():.2f}-{bracket.Mw.max():.2f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
for ax, cd in zip(axes, ("full", "nugget")):
    for medium, colour in (("sfcvm", OK["vermillion"]), ("homog", OK["blue"])):
        s = scans[(medium, cd)]
        ax.plot(s.D_km, s.dchi2, "o-", color=colour, label=f"{medium}")
        br = s[s.dchi2 <= 4]
        ax.axvspan(br.D_km.min(), br.D_km.max(), color=colour, alpha=0.12)
    ax.axhline(4, color="0.5", lw=0.8, ls="--"); ax.set_xscale("log"); ax.set_ylim(-1, 60)
    ax.set_xlabel("uniform slip to depth D (km)"); ax.set_ylabel("delta chi2"); ax.legend()
    ax.set_title(f"fixed-depth scan, Cd {cd} (shaded: delta chi2 <= 4)", loc="left", fontsize=12)
fig.savefig(os.path.join(FIG, "depth_scan.png"), dpi=300)

# %% [markdown]
# ## [5] Summary

# %%
## [5]
lines = []
for medium, cd in RUNS:
    t = uncertainty[(medium, cd)]
    m2, s2, ref = (t[t.kind == k].iloc[0] for k in ("boot_mean", "boot_2sd", "reference"))
    nb = t[t.kind == "sigma_neighbour"]
    rec = recovery_tables[(medium, cd)]
    sc = scans[(medium, cd)]; br = sc[sc.dchi2 <= 4]
    lines.append(f"{run_name(cd)} / {medium} (sigma {ref.sigma:g}): Mw30 {ref.Mw_30gpa:.2f} (boot {m2.Mw_30gpa:.2f} +/- {s2.Mw_30gpa:.2f}; "
                 f"sigma {', '.join(f'{r.sigma:g}: {r.Mw_30gpa:.2f}' for _, r in nb.iterrows())}), Mw_sfcvm {ref.Mw_sfcvm:.2f} +/- {s2.Mw_sfcvm:.2f}, "
                 f"surface {ref.surface_pct:.0f} +/- {s2.surface_pct:.0f} %, peak {ref.peak_slip_mm:.1f} +/- {s2.peak_slip_mm:.1f} mm; "
                 f"recovery: moment {rec.moment_recovery.min():.2f}-{rec.moment_recovery.max():.2f} of input over D {rec.D_km.min():g}-{rec.D_km.max():g} km, "
                 f"W mean residual {rec.mean_res_west300_mm.min():+.2f}..{rec.mean_res_west300_mm.max():+.2f} mm; "
                 f"depth scan: best {sc.loc[sc.chi2.idxmin()].D_km:g} km, bracket {br.D_km.min():g}-{br.D_km.max():g} km (Mw {br.Mw.min():.2f}-{br.Mw.max():.2f})")
print("\n".join(lines))
with open(os.path.join(HERE, "gf", "tests_summary.txt"), "w") as fh:
    fh.write(f"# pylith_sfcvm_tests.py {time.strftime('%Y-%m-%d %H:%M')}\n" + "\n".join(lines) + "\n")
print("done")
