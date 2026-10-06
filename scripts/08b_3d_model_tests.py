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
# ---

# %% [markdown]
# # 08b The 3D slip model: uncertainty, resolution and sensitivity
#
# ```
# Created on Wed Sep 30 2026
# PREPUB: promoted from drafts/retired_2026-10-06/pylith/pylith_sfcvm_tests.py and the 2026-09-30 sensitivity runs; 2026-10-06: no exec
# PREPUB: of 08a (model_utils.py), the bootstrap and the sensitivity runs read from 08c.
# @author: daniellelindsay
# Run in opera_disp_feb13, after 08c and 08a. About 15 min (about 70 tent inversions of ~10 s; no PyLith run).
# ```
#
# **What this notebook does.** For the SELECTED model of 08a (aligned SFCVM, nugget Cd), its
# companion (07a's full correlated Cd) and the homogeneous PyLith twins of both, on the cached
# order-1 Green's functions:
#
# 1. **Bootstrap** (made by 08c [6], read here): noise drawn from the inversion's own Cd (Cholesky) added to the
#    model prediction and re-inverted at the selected sigma, N = 07a's `n_boot`; mean and 2-sigma of Mw30,
#    Mw(SFCVM mu), potency, surface %, peak slip and its depth; plus the same numbers at the neighbouring grid
#    sigmas (sigma sensitivity). Drawn and registered here.
# 2. **Synthetic recovery** (07b [11]): 10 mm uniform slip from the surface to D over the SSE
#    section (tapered), forward through the PyLith G, noise from the same Cd, inverted with the
#    same setup: moment and depth recovery, and the west/east near-field residual (an artefact flag).
# 3. **Fixed-depth scan** (07b [9]): uniform slip to D, amplitude + constant by whitened least
#    squares with the run's Cd, chi2(D), bracket delta chi2 <= 4, Mw per D.
# 4. **Sensitivity runs** (made by 08c, read here): every other PyLith + CSI inversion of the model selection --
#    mesh `shallow` (60 m near-surface band), Cd nugget floors 15 and 4 pixels, equal weights (`diagcd`), no surface
#    constraint (`nosurf`), swath 7 km, dip 80 E and 70 E, order-2 elements, the SFCVM boundary 50 m east / west of
#    the trace (`e50`, `w50`), and the coherence-0.7 field (`coh07`), each with both media and, where run, both Cd
#    treatments: one-page summary per run, the all-tests figure, the table and its range.
# Every run is solved at the locked sigma (`TulePond_filepaths.model_sigma_locked`, Dani 2026-10-03), not at its own
# L-curve pick (which each run still computes and records as `sigma_lcurve_pick`), so the spread of the
# table is the spread of the factor tested, not of sigma. Each row carries the fingerprint of the 06
# field it fits; [5] refuses a table made before the current 06 field or on another field.
#
# Surface % is the median modelled slip on the top-edge vertices of the SSE section over the median
# of the measured crossings in the SSE section (`model_utils.surface_pct_denominator_mm`, the one denominator of
# 07a, 08a, 08c and this notebook), so 08c's and fresh surface % are compared and registered as they are.
#
# The inversion setup is 08a's, built here from the same products with the same functions
# (`model_utils.tent_model_setup`, `invert_tents`, `tent_numbers`); the depth-scan least squares and the
# window/shape definitions are those of 07b, applied to the tent (vertex) parameters.
#
# **Inputs** (made by): 07a `model_data_07a.npz` and the sample points (07a); the G matrices, the SFCVM grid and the
# GreensFns run outputs (pylith/, model_utils.build_G); 08a `results_comparison<tag>.csv` (the selected sigmas and
# the models 08c's default rows must equal); 08c `uncertainty_<run>_<medium>.csv` (the bootstrap),
# `final_comparison.csv`, `results_comparison<tag>.csv`, `inversion_products<tag>.npz`, `lcurve_<medium><tag>.csv`
# (the sensitivity runs); the recovery depths and depth-scan grid of 07b (read from its source text).
# **Outputs** (`outputs/08b_3d_model_tests/`): `uncertainty_<run>_<medium>.csv` (08c's, copied with provenance),
# `recovery_<run>_<medium>.csv`, `recovery_profiles.csv`, `depth_scan_<run>_<medium>.csv`, `sensitivity_runs.csv`,
# `tests_summary.txt`, the stats sidecar `08b_3d_model_tests_stats.txt`.
# **Figures**: `qc08b_bootstrap`, `figS_pylith_recovery`, `figS_pylith_depth_scan`,
# `qc08b_summary_<run>_<medium>` (one page per run), `figS_pylith_all_tests`.
# **Registry keys** `08b_*`.
#
# **What to check.** The selected model's bootstrap Mw30 2-sigma is small next to the sensitivity
# range (`08b_sens_*`); the synthetic recovery shows no one-sided west/east mean residual larger
# than the noise sd; every depth-scan amplitude is positive (asserted); the all-tests figure draws
# as many runs as the table has (asserted).

# %%
## [1]
# ------------------------
# Paths & constants, and 08a's inversion setup rebuilt from the products
# ------------------------
import ast
import glob
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)   # library noise only; numerical warnings stay visible
warnings.filterwarnings("ignore", category=FutureWarning)
from TulePond_filepaths import (common_paths, repo_dir, paths_out, paths_sfcvm, medium_paths, model_crs, model_sigma_locked)
import insar_utils as utils
from insar_utils import mpl_paper_style
import model_utils                                  # every modelling function, once (with its source)

notebook = "08b_3d_model_tests"
out_dir = paths_out[notebook]
fig_dir = common_paths["fig_dir"]
os.makedirs(out_dir, exist_ok=True)
dir_07a = paths_out["07a_uniform_model"]
dir_08a = paths_out["08a_3d_model"]
dir_08c = paths_out["08c_3d_model_sensitivity"]
field_dir = paths_out["06_event_field_geometry"]
sys.path.insert(0, os.path.join(repo_dir, "scripts", "model_plots"))
REG = utils.Registry(notebook, common_paths, repo_dir)
OK = dict(blue="#0072B2", orange="#E69F00", vermillion="#D55E00", purple="#CC79A7", black="#000000", grey="#999999")
mu_ref = 3.0e10                                     # 30 GPa reference rigidity (07a)
sigma_locked = model_sigma_locked
RUNS = [("sfcvm", "nugget"), ("sfcvm", "full"), ("homog", "nugget"), ("homog", "full")]   # (medium, Cd); first = SELECTED (08a)
SELECTED = ("sfcvm", "nugget")
CD_TAG = {"nugget": "_nuggetcd", "full": ""}
media = ("homog", "sfcvm")
run_names = {m: f"gf_default_{m}" for m in media}   # 08a's PyLith GreensFns runs


def run_tag(cd):
    """The product tag of one data-covariance treatment: default (full Cd) or default_nuggetcd."""
    return "default" + CD_TAG[cd]


t_setup = time.time()
model_data = np.load(os.path.join(dir_07a, "model_data_07a.npz"))
frame_lonlat = tuple(model_data["frame_lonlat"])
look = model_data["look"]
lam_xz = tuple(model_data["lam_xz"])
sigma_grid = [float(s) for s in model_data["sigma_grid"]]
n_boot = int(model_data["n_boot"])
creep_lo_km, creep_hi_km = float(model_data["creep_lo_km"]), float(model_data["creep_hi_km"])
cov_p = model_data["cov_p"]
nug_power = float(model_data["nug_power"])
field_fingerprint = model_utils.field_fingerprint(os.path.join(field_dir, "fp_maps.npz"))
assert str(model_data["field_fingerprint"]) == field_fingerprint, "07a's model data were made from another 06 field: re-run 07a"
profile_lats = pd.read_csv(common_paths["fault_profiles"]).ft_lat.values
trace = model_utils.model_trace(model_data["trace_lonlat"], -90.0, 90.0, frame_lonlat, profile_lats,
                                every_km=float(model_data["trace_every_km"]))
samples = model_utils.samples_from_arrays("Downsampled fault-parallel", model_data["sample_lon"], model_data["sample_lat"],
                                          model_data["sample_x_km"], model_data["sample_y_km"], model_data["data_mm"],
                                          model_data["wgt"], look, frame_lonlat, cov_p, nug_power)
points_npz = str(model_data["points_default"]) + ".npz"
setup = model_utils.tent_model_setup(run_names, points_npz, float(model_data["strike_deg"]),
                                     {m: (medium_paths(m)["pylith_output"], medium_paths(m)["gf"]) for m in media},
                                     dict(km=model_data["crossing_km"], lat=model_data["crossing_lat"], mean_mm=model_data["crossing_mean_mm"]),
                                     tuple(model_data["km_window"]), creep_lo_km, creep_hi_km, float(model_data["surf_floor_mm"]),
                                     trace["lonlat"][np.argsort(trace["lonlat"][:, 1])], (trace["x_km"] * 1e3, trace["y_km"] * 1e3),
                                     frame_lonlat, model_crs, profile_lats)
tents = setup["tents"]
G = setup["G"]
imp_km, imp_depth_km = tents.imp_km, tents.imp_depth_km
west300, east300 = setup["west300"], setup["east300"]
mu_sfcvm = model_utils.mu_beside_fault(os.path.join(paths_sfcvm["pylith_output"], f"{run_names['sfcvm']}-elastic_info.h5"),
                                       os.path.join(paths_sfcvm["spatialdb"], "sfcvm_v26-0_aligned_grid.npz"), tents, "sfcvm")[0]
data_covariance = {cd: model_utils.data_cd(cd, samples, cov_p, nug_power) for cd in CD_TAG}


def invert_tents(medium, sigma, cd, vel_insar=None, vel_surface=None):
    """08a's inversion (model_utils.invert_tents) with its data, G and Cd; returns (fault, InSAR data, surface data)."""
    return model_utils.invert_tents(setup["template"], samples, G[medium], data_covariance[cd], setup["surface"],
                                    setup["G_surface"], sigma, lam_xz, frame_lonlat, use_surface=True,
                                    vel_insar=vel_insar, vel_surface=vel_surface, name=f"{medium} {cd} sigma {sigma:g}")


def numbers(fault_model, insar_data, surface):
    """08a's numbers of one solution (model_utils.tent_numbers)."""
    return model_utils.tent_numbers(fault_model.slip[:, 0], insar_data, surface, tents, west300, east300,
                                    setup["in_section_surface"], setup["denominator_mm"], mu_ref, mu_sfcvm)


print(f"08a's setup rebuilt in {time.time() - t_setup:.0f} s: {samples.vel.size} samples x {tents.n_imp} impulses")

tests_src = open(os.path.join(repo_dir, "scripts", "07b_uniform_model_tests.py")).read()


def param(name):
    """A parameter list/number from 07b's source (derived, not re-typed): the literal on the line `<name> = ...`."""
    match = re.search(rf"^{name}\s*=\s*(\[[^\]]*\]|[0-9.]+)", tests_src, re.S | re.M)
    assert match, f"{name} not found in 07b_uniform_model_tests.py"
    return ast.literal_eval(match.group(1))


recovery_depths = param("recovery_depths")
scan_taper = param("scan_taper")
scan_depths = param("scan_depths")
scan_km = (creep_lo_km + scan_taper, creep_hi_km - scan_taper)      # 07b [1]: SSE section tapered at its ends
print(f"07b parameters: n_boot {n_boot}, recovery depths {recovery_depths}, scan window km {scan_km[0]:.1f}-{scan_km[1]:.1f} "
      f"(taper {scan_taper} km), {len(scan_depths)} scan depths")


def selected_sigma(medium, cd):
    """The sigma 08a solved this model at (its results csv; 08a must have run): the locked sigma."""
    results_file = os.path.join(dir_08a, f"results_comparison{CD_TAG[cd]}.csv")
    assert os.path.exists(results_file), f"missing {results_file}: run 08a_3d_model first"
    sigma = float(pd.read_csv(results_file).set_index("model").loc[f"PyLith {medium}", "sigma"])
    assert sigma == sigma_locked, f"08a solved {medium} {cd} at sigma {sigma:g}, not the locked {sigma_locked:g}"
    return sigma


in_window = np.clip(np.minimum(imp_km - scan_km[0], scan_km[1] - imp_km) / scan_taper + 1, 0, 1)   # 07b [9]
REPORT = ["Mw_30gpa", "Mw_sfcvm", "potency_m3", "surface_pct", "peak_slip_mm", "peak_depth_m"]

# %% [markdown]
# ## [2] Bootstrap and sigma sensitivity (made by 08c [6])
#
# n_boot noise realisations from each run's own data and surface covariances (Cholesky) added
# to the model's own synthetics, re-inverted at the selected sigma, each on a fresh fault and
# fresh copies of the data (seed 11, as 07a). The spread is the formal error of THIS
# configuration; the structural range is the sensitivity table of [5]. Then the neighbouring grid
# sigmas on the real data. Read from 08c (made there because 08a quotes the 2-sigma); CHECK: each table's reference
# row is 08a's model (Mw30, surface %) and fits the current 06 field.

# %%
## [2]
uncertainty = {}
results_08a = {cd: pd.read_csv(os.path.join(dir_08a, f"results_comparison{CD_TAG[cd]}.csv")).set_index("model") for cd in CD_TAG}
for medium, cd in RUNS:
    unc_file = os.path.join(dir_08c, f"uncertainty_{run_tag(cd)}_{medium}.csv")
    assert os.path.exists(unc_file), f"missing {unc_file}: run 08c_3d_model_sensitivity"
    table = pd.read_csv(unc_file)
    assert table.provenance.str.contains(field_fingerprint, regex=False).all(), f"{unc_file} fits another 06 field: re-run 08c"
    reference = table[table.kind == "reference"].iloc[0]
    assert float(reference.sigma) == selected_sigma(medium, cd)
    for col in ("Mw_30gpa", "surface_pct", "potency_m3"):
        mine = float(results_08a[cd].loc[f"PyLith {medium}", col])
        assert abs(float(reference[col]) - mine) <= 1e-9 * abs(mine), f"{unc_file}: reference {col} is not 08a's model"
    table["provenance"] = table.provenance + f"; copied by {notebook} {time.strftime('%Y-%m-%d %H:%M')}"
    table.to_csv(os.path.join(out_dir, f"uncertainty_{run_tag(cd)}_{medium}.csv"), index=False)
    uncertainty[(medium, cd)] = table
    boot_mean_row = table[table.kind == "boot_mean"].iloc[0]
    boot_2sd_row = table[table.kind == "boot_2sd"].iloc[0]
    print(f"{medium} {cd} (sigma {reference.sigma:g}, {int((table.kind == 'boot').sum())} realisations): "
          + ", ".join(f"{col} {boot_mean_row[col]:.3g} +/- {boot_2sd_row[col]:.2g}" for col in REPORT))

plt.rcParams.update({"font.size": 12, "savefig.facecolor": "white", "figure.facecolor": "white"})
fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
for ax, (medium, cd) in zip(axes.ravel(), RUNS):
    table = uncertainty[(medium, cd)]
    boot = table[table.kind == "boot"]
    reference = table[table.kind == "reference"].iloc[0]
    ax.hist(boot.Mw_30gpa, bins=12, color=OK["blue"] if cd == "full" else OK["vermillion"], alpha=0.8)
    ax.axvline(reference.Mw_30gpa, color="k", lw=1.5, label=f"reference {reference.Mw_30gpa:.2f}")
    boot_mw_mean = boot.Mw_30gpa.mean()
    boot_mw_2sd = 2 * boot.Mw_30gpa.std(ddof=1)
    ax.axvspan(boot_mw_mean - boot_mw_2sd, boot_mw_mean + boot_mw_2sd, color=OK["grey"], alpha=0.25,
               label=f"mean {boot_mw_mean:.2f} +/- {boot_mw_2sd:.2f} (2 sd)")
    for _, row in table[table.kind == "sigma_neighbour"].iterrows():
        ax.axvline(row.Mw_30gpa, color=OK["orange"], ls="--", lw=1.2, label=f"sigma {row.sigma:g}: {row.Mw_30gpa:.2f}")
    ax.set_xlabel("Mw (30 GPa)")
    ax.set_ylabel("realisations")
    ax.legend(fontsize=9)
    ax.set_title(f"{medium}, Cd {cd}: bootstrap ({n_boot}) and neighbouring sigmas", loc="left", fontsize=12)
fig.savefig(os.path.join(fig_dir, "qc08b_bootstrap.png"), dpi=300)
plt.close(fig)
print("wrote qc08b_bootstrap.png")

# %% [markdown]
# ## [3] Synthetic recovery (07b [11])
#
# 10 mm uniform right-lateral slip from the surface to D over the SSE section (07b's tapered
# window), forward through the PyLith G of the medium, noise from the run's Cd (seed 7), the
# surface constraint's own error on each measured crossing, inverted with the same setup at the
# selected sigma. Reported: potency recovered / input, the depth below which the section-mean slip
# falls under 10 % of its largest value over depth, and the mean residual within 300 m west / east
# of the trace (an artefact flag: a one-sided bias means the medium cannot fit that side).

# %%
## [3]
recovery_rng = np.random.default_rng(7)                          # 07b's seed
in_scan_section = (imp_km >= scan_km[0]) & (imp_km <= scan_km[1])
zbins = np.arange(0, 3.01, 0.1)                                  # depth bins of the recovered profile (km)
recovery_tables = {}
profiles = []
surf_ref = setup["surface"]
G_surf = setup["G_surface"]
for medium, cd in RUNS:
    sigma = selected_sigma(medium, cd)
    _, cd_data, _ = invert_tents(medium, sigma, cd)              # the run's Cd (InSAR)
    chol = np.linalg.cholesky(cd_data.Cd + 1e-9 * np.eye(cd_data.Cd.shape[0]))
    noise_w = float(np.sqrt(np.mean(np.diag(cd_data.Cd)[west300])))
    noise_e = float(np.sqrt(np.mean(np.diag(cd_data.Cd)[east300])))
    rows = []
    for D in recovery_depths:
        slip_in = 10.0 * in_window * (imp_depth_km < D)
        pot_in = tents.potency_m3(slip_in)
        vel_i = G[medium].dot(slip_in) + np.dot(chol, recovery_rng.standard_normal(cd_data.vel.size))
        vel_s = G_surf.dot(slip_in) + np.sqrt(np.diag(surf_ref.Cd)) * recovery_rng.standard_normal(surf_ref.vel.size)
        fault_model, insar_data, surface = invert_tents(medium, sigma, cd, vel_insar=vel_i, vel_surface=vel_s)
        num = numbers(fault_model, insar_data, surface)
        res = insar_data.vel - insar_data.synth
        slip_out = fault_model.slip[:, 0]
        prof = []                                                # section-mean recovered slip per 0.1 km depth bin
        for z0 in zbins[:-1]:
            in_bin = in_scan_section & (imp_depth_km >= z0) & (imp_depth_km < z0 + 0.1)
            if np.any(in_bin):
                prof.append(slip_out[in_bin].mean())
            else:
                prof.append(np.nan)
        prof = np.array(prof)
        below = np.nonzero(prof < 0.1 * np.nanmax(prof))[0]
        d_rec = float(zbins[below[0]]) if below.size else np.nan
        rows.append(dict(D_km=D, potency_in_m3=pot_in, potency_out_m3=num["potency_m3"], moment_recovery=num["potency_m3"] / pot_in,
                         Mw_in=float(model_utils.moment_magnitude(mu_ref * pot_in)), Mw_out=num["Mw_30gpa"], depth_10pct_km=d_rec,
                         peak_out_mm=num["peak_slip_mm"], peak_depth_m=num["peak_depth_m"], surface_pct=num["surface_pct"],
                         rms_mm=num["rms_mm"], rms_west300_mm=num["rms_west300_mm"], rms_east300_mm=num["rms_east300_mm"],
                         mean_res_west300_mm=float(res[west300].mean()), mean_res_east300_mm=float(res[east300].mean()),
                         noise_sd_west300_mm=noise_w, noise_sd_east300_mm=noise_e))
        input_profile = []
        for z0 in zbins[:-1]:
            input_profile.append(10.0 * float(z0 + 0.05 < D))
        profiles.append(pd.DataFrame(dict(run=run_tag(cd), medium=medium, D_km=D, z_top_km=zbins[:-1], slip_mm=prof,
                                          input_mm=input_profile)))
        print(f"  {medium} {cd} input 10 mm to {D:g} km: moment recovered {rows[-1]['moment_recovery']:.2f}, Mw {rows[-1]['Mw_in']:.2f} -> "
              f"{rows[-1]['Mw_out']:.2f}, depth(10 %) {d_rec:.1f} km, W/E rms {rows[-1]['rms_west300_mm']:.2f}/{rows[-1]['rms_east300_mm']:.2f} "
              f"(noise sd {noise_w:.2f}/{noise_e:.2f}), W/E mean residual {rows[-1]['mean_res_west300_mm']:+.2f}/{rows[-1]['mean_res_east300_mm']:+.2f} mm")
    table = pd.DataFrame(rows)
    table["provenance"] = f"08b_3d_model_tests {time.strftime('%Y-%m-%d %H:%M')}; G {run_names[medium]}; Cd {cd}; sigma {sigma:g}; seed 7"
    table.to_csv(os.path.join(out_dir, f"recovery_{run_tag(cd)}_{medium}.csv"), index=False)
    recovery_tables[(medium, cd)] = table
pd.concat(profiles).to_csv(os.path.join(out_dir, "recovery_profiles.csv"), index=False)

import plot_tests_figures                                  # the recovery figure, drawn from the products just written
plot_tests_figures.main(out_dir, os.path.join(fig_dir, "figS_pylith_recovery.png"))

# %% [markdown]
# ## [4] Fixed-depth scan (07b [9])
#
# Uniform slip to depth D over the tapered SSE window, amplitude and a constant by whitened least
# squares with the run's Cd on the ramp-removed data of the reference solution, chi2(D), the
# bracket delta chi2 <= 4 and the Mw of each D. No smoothing, no sigma: the depth the data alone
# prefer under each covariance. Every amplitude must be positive (asserted).
#
# NB (2026-10-06, investigated, method NOT changed): under the nugget-only (diagonal) Cd the bracket collapses to the
# best grid depth (`08b_depth_bracket_lo_km` = `_hi_km` = 0.40 km). The nugget Cd leaves out the correlated noise, so
# chi2 is far above the number of samples (reduced chi2 printed and registered, `*_depth_scan_reduced_chi2`) and the
# neighbouring grid depths (0.3, 0.5 km) lie hundreds of chi2 units above the minimum: delta chi2 <= 4 assumes a
# calibrated chi2 and is not met by any other depth. Even scaled by the reduced chi2 (delta chi2 <= 4 x chi2_nu, printed)
# the bracket stays one grid point. The full-Cd bracket (`08b_full_depth_bracket_*`) is the calibrated one.

# %%
## [4]
scans = {}
scan_diagnosis = {}               # (medium, Cd) -> reduced chi2 at the best depth and the chi2_nu-scaled bracket (printed, see above)
for medium, cd in RUNS:
    _, cd_data, _ = invert_tents(medium, selected_sigma(medium, cd), cd)
    chol = np.linalg.cholesky(cd_data.Cd + 1e-9 * np.eye(cd_data.Cd.shape[0]))
    data_w = np.linalg.solve(chol, cd_data.vel)                   # the ramp-removed data of the reference solution
    rows = []
    for D in scan_depths:
        if D > imp_depth_km.max() + 1e-6:
            continue                                              # deeper than the fault
        shape = in_window * (imp_depth_km < D)
        design = np.c_[G[medium].dot(shape), np.ones(cd_data.vel.size)]
        design_w = np.linalg.solve(chol, design)
        coef, *_ = np.linalg.lstsq(design_w, data_w, rcond=None)
        resid_w = data_w - design_w.dot(coef)
        pot = tents.potency_m3(shape * coef[0])
        assert pot > 0, f"depth scan {medium} {cd} D={D}: non-positive amplitude {coef[0]:.2f} mm"
        rows.append(dict(D_km=D, slip_mm=float(coef[0]), constant_mm=float(coef[1]), chi2=float(np.dot(resid_w, resid_w)),
                         potency_m3=pot, Mw=float(model_utils.moment_magnitude(mu_ref * pot))))
    scan = pd.DataFrame(rows)
    scan["dchi2"] = scan.chi2 - scan.chi2.min()
    scan["provenance"] = f"08b_3d_model_tests {time.strftime('%Y-%m-%d %H:%M')}; G {run_names[medium]}; Cd {cd}"
    scan.to_csv(os.path.join(out_dir, f"depth_scan_{run_tag(cd)}_{medium}.csv"), index=False)
    scans[(medium, cd)] = scan
    best = scan.loc[scan.chi2.idxmin()]
    bracket = scan[scan.dchi2 <= 4]
    print(f"depth scan {medium} {cd}: best D {best.D_km:g} km ({best.slip_mm:.1f} mm, Mw {best.Mw:.2f}); delta chi2 <= 4: "
          f"{bracket.D_km.min():g}-{bracket.D_km.max():g} km, Mw {bracket.Mw.min():.2f}-{bracket.Mw.max():.2f}")
    # the bracket diagnosis (markdown above): how calibrated chi2 is, and the bracket with delta chi2 scaled by chi2_nu
    reduced_chi2 = float(best.chi2 / (cd_data.vel.size - 2))                  # two parameters: amplitude and constant
    scaled_bracket = scan[scan.dchi2 <= 4 * reduced_chi2]
    best_index = int(scan.chi2.idxmin())
    neighbour_text = []
    for neighbour in (best_index - 1, best_index + 1):
        if 0 <= neighbour < len(scan):
            neighbour_text.append(f"D {scan.D_km.iloc[neighbour]:g} km delta chi2 {scan.dchi2.iloc[neighbour]:.1f}")
    scan_diagnosis[(medium, cd)] = dict(reduced_chi2=reduced_chi2, scaled_lo=float(scaled_bracket.D_km.min()), scaled_hi=float(scaled_bracket.D_km.max()))
    print(f"    reduced chi2 at the best depth {reduced_chi2:.2f} ({cd_data.vel.size} samples); grid neighbours: {'; '.join(neighbour_text)}; "
          f"bracket with delta chi2 <= 4 x chi2_nu ({4 * reduced_chi2:.1f}): {scaled_bracket.D_km.min():g}-{scaled_bracket.D_km.max():g} km")

mpl_paper_style(base=7)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True)
for ax, cd, letter in zip(axes, ("nugget", "full"), "AB"):
    for medium, colour in (("sfcvm", OK["vermillion"]), ("homog", OK["blue"])):
        scan = scans[(medium, cd)]
        ax.plot(scan.D_km, scan.dchi2, "o-", color=colour, ms=3, lw=1, label=f"{medium}")
        bracket = scan[scan.dchi2 <= 4]
        ax.axvspan(bracket.D_km.min(), bracket.D_km.max(), color=colour, alpha=0.12)
    ax.axhline(4, color="0.5", lw=0.6, ls="--")
    ax.set_xscale("log")
    ax.set_ylim(-1, 60)
    ax.set_xlabel("uniform slip to depth D (km)")
    ax.set_ylabel("delta chi2")
    ax.legend(fontsize=6)
    ax.set_title(r"$\bf{" + letter + r"}$  " + f"fixed-depth scan, Cd {cd} (shaded: delta chi2 <= 4)", loc="left", fontsize=7)
fig.savefig(os.path.join(fig_dir, "figS_pylith_depth_scan.png"), dpi=300)
plt.close(fig)
print("wrote figS_pylith_depth_scan.png")

# %% [markdown]
# ## [5] The sensitivity runs: one page per run, and all runs on one figure
#
# The table `final_comparison.csv` (every run: settings, sigma, rms, near-fault rms west / east,
# chi2, surface %, potency, Mw at 30 GPa and with the SFCVM rigidity, the bootstrap 2-sigma where
# made) and the per-run products are read from 08c (`outputs/08c_3d_model_sensitivity/`); each run needs its own
# PyLith Green's-function run (mesh `shallow`, the dipping meshes, order-2 elements, the swath-7 and coherence-0.7
# sample sets, the +/- 50 m SFCVM grids; pylith/README.md).
# CHECKS (fail loud on a stale table): the table is newer than 06's `fig2_inputs.npz` and `fp_maps.npz`;
# every PyLith row (and every `results_comparison*.csv` the figures read) carries the current 06 field's
# fingerprint and the locked sigma; the `default` and `default_nuggetcd` rows equal 08a's
# `results_comparison*.csv`. One-page diagnostic per run and medium (`qc08b_summary_<run>_<medium>`), then the
# all-tests figure (`figS_pylith_all_tests`): near-fault fit both sides, west fit against Mw, and
# the run-by-run ladder of the west-side rms, homogeneous vs SFCVM on the same settings.

# %%
## [5]
def two_decimals(value):
    """A table cell to two decimals ('' for a missing value)."""
    return "" if pd.isna(value) else f"{value:.2f}"


def no_decimals(value):
    """A table cell to the nearest unit ('' for a missing value)."""
    return "" if pd.isna(value) else f"{value:.0f}"


final_file = os.path.join(dir_08c, "final_comparison.csv")
assert os.path.exists(final_file), f"missing {final_file}: run 08c_3d_model_sensitivity"
# stale table = made from another 06 field: every row carries the fingerprint (sha256) of the fp_maps.npz it fits and
# must carry the current one (asserted below). Until 2026-10-06 the table also had to be NEWER than 06's fp_maps.npz and
# fig2_inputs.npz; a rerun of 06 that writes the same field (same fingerprint; np.savez is byte-reproducible) then failed
# that check without any change to the field, and 08c never reads fig2_inputs.npz. The content check is the one kept.
sensitivity = pd.read_csv(final_file)
assert {"run", "medium", "sigma", "sigma_lcurve_pick", "field_fingerprint", "rms_W300_mm", "Mw_30gpa", "surface_pct"} <= set(sensitivity.columns), \
    f"{final_file} lacks the sigma / field provenance columns"
pylith_rows = sensitivity[sensitivity.run != "07 CSI baseline"]
stale = pylith_rows[pylith_rows.field_fingerprint.astype(str) != field_fingerprint]
assert stale.empty, f"{len(stale)} sensitivity rows fit another 06 field (not {field_fingerprint}): {sorted(set(stale.run))}"
off_sigma = pylith_rows[~np.isclose(pylith_rows.sigma.astype(float), sigma_locked)]
assert off_sigma.empty, f"{len(off_sigma)} sensitivity rows not at the locked sigma {sigma_locked:g}: {sorted(set(off_sigma.run))}"
for cd in CD_TAG:                                                           # the default rows ARE 08a's models
    for medium in media:
        row = pylith_rows[(pylith_rows.run == run_tag(cd)) & (pylith_rows.medium == medium)]
        assert len(row) == 1, f"{run_tag(cd)} / {medium}: {len(row)} rows in {final_file}"
        for col in ("Mw_30gpa", "potency_m3", "surface_pct", "rms_mm"):
            mine = float(results_08a[cd].loc[f"PyLith {medium}", col])
            assert abs(float(row[col].iloc[0]) - mine) <= 1e-6 * abs(mine), f"{final_file} {run_tag(cd)} / {medium} {col} differs from 08a's"
print(f"PASS: {final_file} is newer than the 06 field, every PyLith row fits field {field_fingerprint} at sigma {sigma_locked:g}, "
      f"and its default rows equal 08a's")
sensitivity["provenance"] = f"copied by 08b_3d_model_tests {time.strftime('%Y-%m-%d %H:%M')} from {final_file}"
sensitivity.to_csv(os.path.join(out_dir, "sensitivity_runs.csv"), index=False)
show = sensitivity.drop(columns=["summary_png", "provenance", "field_fingerprint", "made"]).copy()
for col in ("rms_mm", "rms_W300_mm", "rms_E300_mm", "Mw_30gpa", "Mw_sfcvm", "Mw_30gpa_2sd", "Mw_sfcvm_2sd", "surface_pct_2sd"):
    show[col] = show[col].map(two_decimals)
for col in ("chi2", "surface_pct", "potency_m3"):
    show[col] = show[col].map(no_decimals)
show["sigma"] = show["sigma"].map("{:g}".format)
show["sigma_lcurve_pick"] = show["sigma_lcurve_pick"].map(lambda v: "" if pd.isna(v) else f"{v:g}")
show["sigma_corner"] = show["sigma_corner"].map(lambda v: "" if pd.isna(v) else f"{v:.1f}")
cols = list(show.columns)
print("| " + " | ".join(cols) + " |")
print("|" + "---|" * len(cols))
for row in show.itertuples(index=False):
    cells = []
    for value in row:
        cells.append(str(value))
    print("| " + " | ".join(cells) + " |")
print(f"\n{len(pylith_rows)} PyLith inversions on {pylith_rows.run.nunique()} run tags (+ the 07a CSI baseline row)")

import plot_run_summary
import plot_test_summary
tags = []
for results_file in glob.glob(os.path.join(dir_08c, "results_comparison*.csv")):
    tags.append(os.path.basename(results_file).replace("results_comparison", "").replace(".csv", ""))
tags = sorted(tags)
for tag in tags:                                                            # the figures read every table: none may be stale
    table_tag = pd.read_csv(os.path.join(dir_08c, f"results_comparison{tag}.csv"))
    pylith_tag = table_tag[table_tag.model.str.startswith("PyLith ")]
    assert (pylith_tag.field_fingerprint.astype(str) == field_fingerprint).all(), f"results_comparison{tag}.csv fits another 06 field"
    assert np.allclose(pylith_tag.sigma.astype(float), sigma_locked), f"results_comparison{tag}.csv is not at the locked sigma"
summary_pages = []
for tag in tags:
    media_tag = []
    for run_medium in plot_run_summary.load_run(tag, dir_08c)["media"]:
        if run_medium != "sfcvmasis":                                            # aligned SFCVM only (Dani)
            media_tag.append(run_medium)
    summary_pages += plot_run_summary.main(tag, media_tag, gf_dir=dir_08c, fig_dir=fig_dir, prefix="qc08b_summary")
all_runs = plot_test_summary.main(dir_08c, os.path.join(fig_dir, "figS_pylith_all_tests.png"))
assert len(all_runs) == len(pylith_rows), f"all-tests figure drew {len(all_runs)} runs, the table has {len(pylith_rows)}"
print(f"{len(summary_pages)} one-page summaries; all-tests figure on {len(all_runs)} runs")

# %% [markdown]
# ## [6] Summary and registry
#
# `08b_*` = the selected model; `08b_hom_*`, `08b_full_*`, `08b_full_hom_*` as in 08a. The
# structural range keys (`08b_sens_*`) are over every SFCVM run WITH the surface constraint in the
# sensitivity table (mesh, Cd treatments, swath, dip, order, alignment, coherence), all at the locked
# sigma; surface % on the one measured-crossing denominator. NB: `08b_*_2sig` and 08a's `08a_*_2sig` are the same
# bootstrap (08c's).

# %%
## [6]
KEY_08B = {("sfcvm", "nugget"): "08b", ("homog", "nugget"): "08b_hom", ("sfcvm", "full"): "08b_full", ("homog", "full"): "08b_full_hom"}
lines = []
for medium, cd in RUNS:
    table = uncertainty[(medium, cd)]
    boot_mean_row = table[table.kind == "boot_mean"].iloc[0]
    boot_2sd_row = table[table.kind == "boot_2sd"].iloc[0]
    reference = table[table.kind == "reference"].iloc[0]
    neighbours = table[table.kind == "sigma_neighbour"]
    rec = recovery_tables[(medium, cd)]
    scan = scans[(medium, cd)]
    bracket = scan[scan.dchi2 <= 4]
    neighbour_text = []
    for _, row in neighbours.iterrows():
        neighbour_text.append(f"{row.sigma:g}: {row.Mw_30gpa:.2f}")
    lines.append(f"{run_tag(cd)} / {medium} (sigma {reference.sigma:g}): Mw30 {reference.Mw_30gpa:.2f} "
                 f"(boot {boot_mean_row.Mw_30gpa:.2f} +/- {boot_2sd_row.Mw_30gpa:.2f}; sigma {', '.join(neighbour_text)}), "
                 f"Mw_sfcvm {reference.Mw_sfcvm:.2f} +/- {boot_2sd_row.Mw_sfcvm:.2f}, "
                 f"surface {reference.surface_pct:.0f} +/- {boot_2sd_row.surface_pct:.0f} %, "
                 f"peak {reference.peak_slip_mm:.1f} +/- {boot_2sd_row.peak_slip_mm:.1f} mm; "
                 f"recovery: moment {rec.moment_recovery.min():.2f}-{rec.moment_recovery.max():.2f} of input over D {rec.D_km.min():g}-{rec.D_km.max():g} km, "
                 f"W mean residual {rec.mean_res_west300_mm.min():+.2f}..{rec.mean_res_west300_mm.max():+.2f} mm; "
                 f"depth scan: best {scan.loc[scan.chi2.idxmin()].D_km:g} km, bracket {bracket.D_km.min():g}-{bracket.D_km.max():g} km "
                 f"(Mw {bracket.Mw.min():.2f}-{bracket.Mw.max():.2f})")
    depth_error = rec.depth_10pct_km - rec.D_km
    lines.append(f"   recovery depth error (10 % depth minus D, 0.1 km bins) {depth_error.min():+.1f} to {depth_error.max():+.1f} km over D "
                 f"{rec.D_km.min():g}-{rec.D_km.max():g} km; depth scan reduced chi2 {scan_diagnosis[(medium, cd)]['reduced_chi2']:.2f}, "
                 f"bracket with delta chi2 <= 4 x chi2_nu {scan_diagnosis[(medium, cd)]['scaled_lo']:g}-{scan_diagnosis[(medium, cd)]['scaled_hi']:g} km")
    dip_text = []
    for run_suffix, dip_name in (("", "vertical"), ("_dip80e", "80 E"), ("_dip70e", "70 E")):
        dip_row = pylith_rows[(pylith_rows.run == f"default{run_suffix}{CD_TAG[cd]}") & (pylith_rows.medium == medium)]
        dip_text.append(f"{dip_name} {float(dip_row.rms_mm.iloc[0]):.2f} mm")
    lines.append("   rms by fault dip: " + ", ".join(dip_text))
sens_sfcvm = pylith_rows[(pylith_rows.medium.str.startswith("sfcvm")) & (pylith_rows.surface == "on")]
lines.append(f"sensitivity table: {len(pylith_rows)} PyLith inversions at sigma {sigma_locked:g} on 06 field {field_fingerprint}; "
             f"SFSVM with surface constraint ({len(sens_sfcvm)} runs): "
             f"Mw30 {sens_sfcvm.Mw_30gpa.min():.2f}-{sens_sfcvm.Mw_30gpa.max():.2f}, rms W300 {sens_sfcvm.rms_W300_mm.min():.2f}-{sens_sfcvm.rms_W300_mm.max():.2f} mm, "
             f"surface {sens_sfcvm.surface_pct.min():.0f}-{sens_sfcvm.surface_pct.max():.0f} %; "
             f"own L-curve picks {', '.join(f'{v:g}' for v in sorted(set(sens_sfcvm.sigma_lcurve_pick)))} (not used)")
print("\n".join(lines))
with open(os.path.join(out_dir, "tests_summary.txt"), "w") as fh:
    fh.write(f"# 08b_3d_model_tests {time.strftime('%Y-%m-%d %H:%M')}\n" + "\n".join(lines) + "\n")
input_files = [os.path.join(dir_07a, "model_data_07a.npz"), final_file]
for cd in CD_TAG:
    input_files.append(os.path.join(dir_08a, f"results_comparison{CD_TAG[cd]}.csv"))
for medium in media:
    input_files.append(os.path.join(medium_paths(medium)["gf"], f"G_{run_names[medium]}.npz"))
for medium, cd in RUNS:
    input_files.append(os.path.join(dir_08c, f"uncertainty_{run_tag(cd)}_{medium}.csv"))
REG.inputs(*input_files)
sidecar = REG.sidecar("08b_3d_model_tests", lines)
for medium, cd in RUNS:
    key = KEY_08B[(medium, cd)]
    table = uncertainty[(medium, cd)]
    boot_mean_row = table[table.kind == "boot_mean"].iloc[0]
    boot_2sd_row = table[table.kind == "boot_2sd"].iloc[0]
    neighbours = table[table.kind == "sigma_neighbour"]
    rec = recovery_tables[(medium, cd)]
    scan = scans[(medium, cd)]
    bracket = scan[scan.dchi2 <= 4]
    cell = "selected" if (medium, cd) == SELECTED else "comparison"
    REG.register(f"{key}_mw_30gpa_boot_mean", boot_mean_row.Mw_30gpa, "Mw", 2, where="M8", cell=cell, sidecar=sidecar)
    REG.register(f"{key}_mw_30gpa_2sig", boot_2sd_row.Mw_30gpa, "Mw", 2, where="M8", cell=cell, sidecar=sidecar)
    REG.register(f"{key}_mw_sfcvm_2sig", boot_2sd_row.Mw_sfcvm, "Mw", 2, where="M8", cell=cell, sidecar=sidecar)
    REG.register(f"{key}_surface_pct_2sig", boot_2sd_row.surface_pct, "%", 0, where="M8", cell=cell, sidecar=sidecar)
    REG.register(f"{key}_potency_2sig_m3", boot_2sd_row.potency_m3, "m3", 0, where="M8", cell=cell, sidecar=sidecar)
    REG.register(f"{key}_mw_30gpa_sigma_lo", float(neighbours.Mw_30gpa.min()), "Mw", 2, where="M8", cell="sigma neighbours", sidecar=sidecar)
    REG.register(f"{key}_mw_30gpa_sigma_hi", float(neighbours.Mw_30gpa.max()), "Mw", 2, where="M8", cell="sigma neighbours", sidecar=sidecar)
    REG.register(f"{key}_recovery_ratio_min", float(rec.moment_recovery.min()), "", 2, where="M8", cell="recovery", sidecar=sidecar)
    REG.register(f"{key}_recovery_ratio_max", float(rec.moment_recovery.max()), "", 2, where="M8", cell="recovery", sidecar=sidecar)
    REG.register(f"{key}_recovery_west_bias_max_mm", float(rec.mean_res_west300_mm.abs().max()), "mm", 2, where="M8", cell="recovery", sidecar=sidecar)
    REG.register(f"{key}_depth_best_km", float(scan.loc[scan.chi2.idxmin()].D_km), "km", 2, where="M8", cell="depth scan", sidecar=sidecar)
    REG.register(f"{key}_depth_bracket_lo_km", float(bracket.D_km.min()), "km", 2, where="M8", cell="depth scan", sidecar=sidecar)
    REG.register(f"{key}_depth_bracket_hi_km", float(bracket.D_km.max()), "km", 2, where="M8", cell="depth scan", sidecar=sidecar)
    REG.register(f"{key}_depth_bracket_mw_lo", float(bracket.Mw.min()), "Mw", 2, where="M8", cell="depth scan", sidecar=sidecar)
    REG.register(f"{key}_depth_bracket_mw_hi", float(bracket.Mw.max()), "Mw", 2, where="M8", cell="depth scan", sidecar=sidecar)
    # (2026-10-06) the depth error of the synthetic recovery (depth where the recovered section-mean slip falls under 10 % of
    # its peak, minus the input depth D; 0.1 km bins) and the depth scan's chi2 calibration
    depth_error = rec.depth_10pct_km - rec.D_km
    REG.register(f"{key}_recovery_depth_err_min_km", float(depth_error.min()), "km", 1, where="M8", cell="recovery", sidecar=sidecar)
    REG.register(f"{key}_recovery_depth_err_max_km", float(depth_error.max()), "km", 1, where="M8", cell="recovery", sidecar=sidecar)
    REG.register(f"{key}_depth_scan_reduced_chi2", scan_diagnosis[(medium, cd)]["reduced_chi2"], "", 2, where="M8", cell="depth scan", sidecar=sidecar)
    # the fault dip: rms of the vertical fault and of the 80 E / 70 E runs (08c's table, this medium and Cd)
    for dip_tag, run_suffix in (("90", ""), ("80e", "_dip80e"), ("70e", "_dip70e")):
        dip_row = pylith_rows[(pylith_rows.run == f"default{run_suffix}{CD_TAG[cd]}") & (pylith_rows.medium == medium)]
        assert len(dip_row) == 1, f"default{run_suffix}{CD_TAG[cd]} / {medium}: {len(dip_row)} rows in {final_file}"
        REG.register(f"{key}_rms_dip{dip_tag}_mm", float(dip_row.rms_mm.iloc[0]), "mm", 2, where="P7, M8", cell="dip", sidecar=sidecar)
REG.register("08b_recovery_depth_min_km", float(min(recovery_depths)), "km", 1, where="M8", cell="recovery", sidecar=sidecar)
REG.register("08b_recovery_depth_max_km", float(max(recovery_depths)), "km", 1, where="M8", cell="recovery", sidecar=sidecar)
REG.register("08b_n_sensitivity_runs", int(len(pylith_rows)), "count", 0, where="M8", cell="sensitivity", sidecar=sidecar)
REG.register("08b_sens_mw_30gpa_min", float(sens_sfcvm.Mw_30gpa.min()), "Mw", 2, where="M8", cell="sensitivity", sidecar=sidecar)
REG.register("08b_sens_mw_30gpa_max", float(sens_sfcvm.Mw_30gpa.max()), "Mw", 2, where="M8", cell="sensitivity", sidecar=sidecar)
REG.register("08b_sens_surface_pct_min", float(sens_sfcvm.surface_pct.min()), "%", 0, where="M8", cell="sensitivity", sidecar=sidecar)
REG.register("08b_sens_surface_pct_max", float(sens_sfcvm.surface_pct.max()), "%", 0, where="M8", cell="sensitivity", sidecar=sidecar)
REG.render()
print("08b_3d_model_tests complete")
