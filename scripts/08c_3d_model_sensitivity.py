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
# # 08c The 3D slip model: the sensitivity runs and the bootstrap
#
# ```
# Created on Tue Oct 06 2026 (from drafts/retired_2026-10-06/pylith/pylith_sfcvm_gf.py, the PyLith driver run 22 times by
# drafts/retired_2026-10-06/pylith/models_overnight_2026-10-05.sh with PYLITH_* switches, and the bootstrap of drafts/retired_2026-10-06/pylith/pylith_sfcvm_tests.py)
# @author: daniellelindsay
# Run in opera_disp_feb13, after 07a and BEFORE 08a (08a reads the +/- 50 m alignment runs and the bootstrap made
# here; run_all.sh runs 07a, 08c, 08a, 08b in that order). About 110 min: 22 runs x (13-sigma L-curve + the model)
# per medium, ~4 min per run, then 4 x 26 bootstrap inversions. No PyLith run: every Green's function is read.
# ```
#
# **What this notebook does.** Every inversion of the model selection on the cached PyLith Green's functions, as ONE
# loop over an explicit table of settings ([2]): for each run and each medium, the L-curve over 07a's sigma grid (its
# corner and pick are recorded, not used), the model at the locked sigma (`TulePond_filepaths.model_sigma_locked`,
# Dani 2026-10-03; so the spread of the table is the spread of the factor tested, not of sigma), and its numbers.
# Then the bootstrap of the four main models ([6]; the selected model of 08a, its companion and their homogeneous
# twins), and the one table of all runs, `final_comparison.csv` ([7]). The selected model itself and its checks are
# 08a; the uncertainty and resolution tests and the figures of these runs are 08b.
#
# The settings each run changes (one row each, [2]):
#
# | Setting | Values | What it tests |
# |---|---|---|
# | media | `homog` (30 GPa) and `sfcvm` (SFCVM v26.0 aligned on the trace); `sfcvm_e50` / `sfcvm_w50` (the block boundary 50 m east / west); `sfcvm_asis` (not shifted onto the trace, as published; 2026-10-14) | the elastic structure |
# | Cd | `full` (07a's correlated), `nugget` (its diagonal: the selected model), `diag` (equal weights), `floor` (n_px floored at 15 or 4) | the data covariance |
# | surface | `model_use_surface` (OFF since 2026-10-16) for every run; the `_surf` rows (full, equal, nugget Cd) turn the measured-crossing constraint ON | the surface data |
# | mesh | `default` (100 m fault cells in the patch), `shallow` (60 m near-surface band) | the fault discretisation |
# | dip | 90, 80 E, 70 E, 80 W, 70 W (top edge on the trace; the SFCVM is the paper's grid, laterally aligned, for every dip, 2026-10-25) | the fault dip |
# | dip scan | 50 E, 60 E, 60 W, 50 W, homogeneous only (2026-10-24, Dani): with the dip rows above the scan 50 W-50 E; reported on its own (08a, 08b), not in the sensitivity range | can a dipping fault in a homogeneous medium fit the near-field asymmetry |
# | order | 1, 2 (quadratic solution, linear materials, no Poisson cap; Charles Williams 2026-09-30) | the element order |
# | points | `default` (07a's samples), `swath7`, `coh07` (07a's sensitivity sample sets) | the data selection |
#
# Each run reads its own Green's functions: `G_<run>.npz` from the medium's `gf/` folder, rebuilt from the cached
# PyLith output (`<run>-points.h5`, `-fault.h5`) when it was projected on another strike than the current field's
# (`model_utils.load_G`). The PyLith runs themselves (one GreensFns run per mesh x medium x point set; 33 min each on
# 8 ranks) are listed with their cfg stacks in `pylith/README.md`.
#
# **Inputs** (made by): 07a `model_data_07a.npz` (strike, crossings, prior, sigma grid, the trace) and
# `fig2_model_row.npz` (the CSI baseline row); `modeling_inputs/points/insar_points{,_swath7,_coh07}.npz` (07a: the
# sample sets with their data and noise model); `modeling_uniform/` and `modeling_SFCVM/` `pylith_output/` (the
# GreensFns runs and the forward runs of the superposition check, pylith/run_gf.sh, run_fwd.sh) and `gf/` (G
# matrices); `modeling_SFCVM/spatialdb/sfcvm_v26-0_aligned*_grid.npz` (the SFCVM grids, pylith/build/sfcvm_grid.py).
#
# **Outputs** (`outputs/08c_3d_model_sensitivity/`): per run `lcurve_<medium><tag>.csv`, `results_comparison<tag>.csv`,
# `inversion_products<tag>.npz`; the bootstrap `uncertainty_<run>_<medium>.csv` ([6]); `final_comparison.csv` ([7]);
# `sensitivity_summary.txt`. Registers nothing: 08a and 08b register the numbers taken from these tables.
#
# **What to check.** Every run at the locked sigma (asserted); the superposition check of each run with a forward run
# (default mesh and points: G x slip = the forward run to 1e-4 mm at order 1, within 5 % at order 2); every row on the
# same 06 field (one fingerprint, asserted in [7]).

# %%
## [1]
# ------------------------
# Paths & constants
# ------------------------
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)   # library noise only; numerical warnings stay visible
warnings.filterwarnings("ignore", category=FutureWarning)
from TulePond_filepaths import (common_paths, repo_dir, paths_out, paths_inputs, paths_sfcvm, medium_paths,
                                model_sigma_locked, model_crs, model_sigma_grid, model_use_surface, model_near_trace_cut_km)
import insar_utils as utils
import model_utils                                  # every modelling function, once (with its source)

notebook = "08c_3d_model_sensitivity"
out_dir = paths_out[notebook]
dir_07a = paths_out["07a_uniform_model"]
os.makedirs(out_dir, exist_ok=True)
mu_ref = 3.0e10                                      # 30 GPa reference rigidity (07a)
sigma_locked = model_sigma_locked                    # every run is solved at this sigma (Dani 2026-10-03)
n_boot_seed = 11                                     # 07a [12]'s bootstrap seed
made = time.strftime("%Y-%m-%d %H:%M")
print(f"{notebook}: products to {out_dir}; locked sigma {sigma_locked:g}")

# %% [markdown]
# ## [2] The runs: one row per inversion setting
#
# `tag` = the product suffix (the driver's naming: mesh, dip, points, order, alignment, surface, Cd, in that order;
# asserted below so the names cannot drift from the settings). The first two rows are the companion (`default`,
# 07a's full Cd) and the SELECTED model (`nuggetcd`) of 08a, both media; 08a re-makes them and checks them against
# this table.

# %%
## [2]
RUN_COLUMNS = ["tag", "media", "cd", "cd_floor", "surface", "mesh", "dip", "order", "points"]
# the surface constraint of the default and of every run is model_use_surface (TulePond_filepaths; OFF since 2026-10-16);
# the `_surf` rows switch it the other way (until 2026-10-16 the default was ON and these rows were `_nosurf`)
SURF = model_use_surface
ALT = not model_use_surface
RUN_TABLE = [
    # tag                    media                 Cd        floor  surface  mesh       dip    order  points
    ("",                     ("homog", "sfcvm"),   "full",   None,  SURF,    "default", "90",  "1",   "default"),
    ("_nuggetcd",            ("homog", "sfcvm"),   "nugget", None,  SURF,    "default", "90",  "1",   "default"),
    ("_e50",                 ("sfcvm_e50",),       "full",   None,  SURF,    "default", "90",  "1",   "default"),
    ("_e50_nuggetcd",        ("sfcvm_e50",),       "nugget", None,  SURF,    "default", "90",  "1",   "default"),
    ("_w50",                 ("sfcvm_w50",),       "full",   None,  SURF,    "default", "90",  "1",   "default"),
    ("_w50_nuggetcd",        ("sfcvm_w50",),       "nugget", None,  SURF,    "default", "90",  "1",   "default"),
    # the SFCVM as published, NOT shifted onto the trace (GF run gf_default_sfcvmasis, 2026-09-30; dropped then, back
    # 2026-10-14 so the paper has numbers for rejecting it, Dani): reported on its own, not in the sensitivity range
    ("_asis",                ("sfcvm_asis",),      "full",   None,  SURF,    "default", "90",  "1",   "default"),
    ("_asis_nuggetcd",       ("sfcvm_asis",),      "nugget", None,  SURF,    "default", "90",  "1",   "default"),
    ("_diagcd",              ("homog", "sfcvm"),   "diag",   None,  SURF,    "default", "90",  "1",   "default"),
    ("_floor15",             ("homog", "sfcvm"),   "floor",  15,    SURF,    "default", "90",  "1",   "default"),
    ("_floor4",              ("homog", "sfcvm"),   "floor",  4,     SURF,    "default", "90",  "1",   "default"),
    ("_surf",                ("homog", "sfcvm"),   "full",   None,  ALT,     "default", "90",  "1",   "default"),
    ("_surf_diagcd",         ("homog", "sfcvm"),   "diag",   None,  ALT,     "default", "90",  "1",   "default"),
    ("_surf_nuggetcd",       ("homog", "sfcvm"),   "nugget", None,  ALT,     "default", "90",  "1",   "default"),   # the selected model WITH the crossing constraint
    ("_shallow",             ("homog", "sfcvm"),   "full",   None,  SURF,    "shallow", "90",  "1",   "default"),
    ("_dip80e",              ("homog", "sfcvm"),   "full",   None,  SURF,    "default", "80e", "1",   "default"),
    ("_dip80e_nuggetcd",     ("homog", "sfcvm"),   "nugget", None,  SURF,    "default", "80e", "1",   "default"),
    ("_dip70e",              ("homog", "sfcvm"),   "full",   None,  SURF,    "default", "70e", "1",   "default"),
    ("_dip70e_nuggetcd",     ("homog", "sfcvm"),   "nugget", None,  SURF,    "default", "70e", "1",   "default"),
    ("_dip80w",              ("homog", "sfcvm"),   "full",   None,  SURF,    "default", "80w", "1",   "default"),   # west dips, 2026-10-25
    ("_dip80w_nuggetcd",     ("homog", "sfcvm"),   "nugget", None,  SURF,    "default", "80w", "1",   "default"),
    ("_dip70w",              ("homog", "sfcvm"),   "full",   None,  SURF,    "default", "70w", "1",   "default"),
    ("_dip70w_nuggetcd",     ("homog", "sfcvm"),   "nugget", None,  SURF,    "default", "70w", "1",   "default"),
    # ("_swath7", ...) and ("_swath7_nuggetcd", ...) DROPPED 2026-10-25 (review of the 08 tests for Dani): the 7 km swath adds
    # only 10 samples to the 2386 (all far west), so it does not test the data selection; the coh07 rows do
    ("_order2",              ("homog", "sfcvm"),   "full",   None,  SURF,    "default", "90",  "2",   "default"),
    ("_order2_nuggetcd",     ("homog", "sfcvm"),   "nugget", None,  SURF,    "default", "90",  "2",   "default"),
    ("_coh07",               ("homog", "sfcvm"),   "full",   None,  SURF,    "default", "90",  "1",   "coh07"),
    ("_coh07_nuggetcd",      ("homog", "sfcvm"),   "nugget", None,  SURF,    "default", "90",  "1",   "coh07"),
    # the homogeneous dip scan (2026-10-24, Dani): with 90 / 80 E / 70 E above, 50 W to 50 E in 10 deg steps; homogeneous
    # only (the SFCVM aligned to a dipping fault is warped at depth, folding at 60-50 W): reported on its own, not in the
    # sensitivity range; no Mw(SFSVM) (no SFCVM run of these meshes)
    ("_dip50e",              ("homog",),           "full",   None,  SURF,    "default", "50e", "1",   "default"),
    ("_dip50e_nuggetcd",     ("homog",),           "nugget", None,  SURF,    "default", "50e", "1",   "default"),
    ("_dip60e",              ("homog",),           "full",   None,  SURF,    "default", "60e", "1",   "default"),
    ("_dip60e_nuggetcd",     ("homog",),           "nugget", None,  SURF,    "default", "60e", "1",   "default"),
    ("_dip60w",              ("homog",),           "full",   None,  SURF,    "default", "60w", "1",   "default"),
    ("_dip60w_nuggetcd",     ("homog",),           "nugget", None,  SURF,    "default", "60w", "1",   "default"),
    ("_dip50w",              ("homog",),           "full",   None,  SURF,    "default", "50w", "1",   "default"),
    ("_dip50w_nuggetcd",     ("homog",),           "nugget", None,  SURF,    "default", "50w", "1",   "default"),
]
DIP_SCAN = ("50e", "60e", "60w", "50w")   # the homogeneous-only dip-scan rows (no SFCVM medium)
runs = pd.DataFrame(RUN_TABLE, columns=RUN_COLUMNS, dtype=object)   # object: keeps the floors as integers (None elsewhere)
CD_TAG = {"full": "", "nugget": "_nuggetcd", "diag": "_diagcd"}


def expected_tag(row):
    """The product suffix the settings imply (the driver's out_tag)."""
    tag = "" if row.mesh == "default" else f"_{row.mesh}"
    tag += "" if row.dip == "90" else f"_dip{row.dip}"
    tag += "" if row.points == "default" else f"_{row.points}"
    tag += "" if row.order == "1" else "_order2"
    for medium in row.media:
        if medium.startswith("sfcvm_"):
            tag += f"_{medium[6:]}"
    tag += "" if row.surface == SURF else ("_surf" if row.surface else "_nosurf")   # relative to the default (2026-10-16)
    tag += f"_floor{row.cd_floor}" if row.cd == "floor" else CD_TAG[row.cd]
    return tag


for row in runs.itertuples():
    assert row.tag == expected_tag(row), f"run {row.tag!r}: its settings imply the tag {expected_tag(row)!r}"
assert runs.tag.is_unique
print(f"{len(runs)} runs, {sum(len(m) for m in runs.media)} inversions at sigma {sigma_locked:g}")
print(runs.to_string(index=False))


def gf_run_name(row, medium):
    """The PyLith GreensFns run of one medium of a run (pylith/README.md): gf_<mesh>[_dip<d>]_<medium>[_<points>][_order2]."""
    dip_tag = "" if row.dip == "90" else f"_dip{row.dip}"
    points_tag = "" if row.points == "default" else f"_{row.points}"
    order_tag = "" if row.order == "1" else "_order2"
    medium_name = "sfcvmasis" if medium == "sfcvm_asis" else medium      # the unshifted run's name (pylith/README.md)
    return f"gf_{row.mesh}{dip_tag}_{medium_name}{points_tag}{order_tag}"


def sfcvm_grid_file(row, medium):
    """The SFCVM grid a medium's PyLith run read its rigidity from (pylith/build/sfcvm_grid.py)."""
    if medium == "sfcvm_asis":                                            # not aligned: the grid as published
        return os.path.join(paths_sfcvm["spatialdb"], "sfcvm_v26-0_asis_grid.npz")
    if medium.startswith("sfcvm_"):
        return os.path.join(paths_sfcvm["spatialdb"], f"sfcvm_v26-0_aligned_{medium[6:]}_grid.npz")
    # the dipping faults sit in the paper's grid (the lateral alignment, model's own dip kept), 2026-10-25 (Dani): until then
    # each dip had its own grid with the block boundary moved onto the dipping fault at every depth (a depth-varying warp
    # of the SFCVM that folds it for west dips; retired)
    nocap_tag = "_nocap" if row.order == "2" else ""
    return os.path.join(paths_sfcvm["spatialdb"], f"sfcvm_v26-0_aligned{nocap_tag}_grid.npz")


def settings_json(row, n_samples, run_names, n_imp, field_fingerprint):
    """The run's settings, as every product carries them (the driver's format, read by 08b's one-page summaries)."""
    mesh_text = row.mesh + ("" if row.order == "1" else " order2 (P2 solution, P1 materials, no nu cap)")
    dip_text = "90 (vertical)" if row.dip == "90" else f"{row.dip[:-1]} {row.dip[-1].upper()} (top edge on the trace)"
    points_text = {"default": "07 quadtree swath 4 km", "swath7": "swath 7 km", "coh07": "coh 0.7 field, swath 4 km"}[row.points]
    mesh_file = f"mesh_vertical_tet_{row.mesh}.msh" if row.dip == "90" else f"mesh_dip{row.dip}_tet_{row.mesh}.msh"
    return json.dumps(dict(mesh=mesh_text, dip=dip_text, cd_mode=row.cd, cd_floor=row.cd_floor, use_surface=bool(row.surface),
                           points=f"{row.points} ({n_samples} samples, {points_text})", run_names=run_names,
                           n_impulses=int(n_imp), mesh_file=mesh_file, sigma_rule=f"fixed {sigma_locked:g} (model_sigma_locked)",
                           field_fingerprint=field_fingerprint))

# %% [markdown]
# ## [3] 07a's model data and sample sets
#
# The strike, the measured crossings, the SSE section, the prior and the sigma grid from 07a's `model_data_07a.npz`;
# each run's samples from its point set (07a: positions, data, block pixel counts, noise model; the full Cd is rebuilt
# with `model_utils.build_cd`). 07a's trace (rebuilt from the saved trace points, `model_utils.model_trace`) gives the
# signed distance of the samples.

# %%
## [3]
model_data = np.load(os.path.join(dir_07a, "model_data_07a.npz"))
frame_lonlat = tuple(model_data["frame_lonlat"])
strike_deg = float(model_data["strike_deg"])
look = model_data["look"]
lam_xz = tuple(model_data["lam_xz"])
sigma_grid = sorted(set(float(s) for s in model_sigma_grid) | {float(model_sigma_locked)})   # the 3D-model grid (2026-10-16), not 07a's
n_boot = int(model_data["n_boot"])
creep_lo_km, creep_hi_km = float(model_data["creep_lo_km"]), float(model_data["creep_hi_km"])
km_window = tuple(model_data["km_window"])
surf_floor_mm = float(model_data["surf_floor_mm"])
field_fingerprint = str(model_data["field_fingerprint"])
assert field_fingerprint == model_utils.field_fingerprint(os.path.join(paths_out["06_event_field_geometry"], "fp_maps.npz")), \
    "07a's model data were made from another 06 field: re-run 07a"
assert sigma_locked in sigma_grid, f"locked sigma {sigma_locked:g} is not on the grid {sigma_grid} (08b needs its neighbours)"
crossings = dict(km=model_data["crossing_km"], lat=model_data["crossing_lat"], mean_mm=model_data["crossing_mean_mm"])
profile_lats = pd.read_csv(common_paths["fault_profiles"]).ft_lat.values
trace = model_utils.model_trace(model_data["trace_lonlat"], -90.0, 90.0, frame_lonlat, profile_lats,
                                every_km=float(model_data["trace_every_km"]))
trace_south_north = trace["lonlat"][np.argsort(trace["lonlat"][:, 1])]
trace_xy_m = (trace["x_km"] * 1e3, trace["y_km"] * 1e3)
row07 = np.load(os.path.join(dir_07a, "fig2_model_row.npz"))
points_npz = {}
samples_of = {}
for points_set in ("default", "swath7", "coh07"):
    points_npz[points_set] = str(model_data[f"points_{points_set}"]) + ".npz"
    p = np.load(points_npz[points_set])
    assert str(p["field_fingerprint"]) == field_fingerprint, f"{points_npz[points_set]} was made from another 06 field: re-run 07a"
    samples_of[points_set] = model_utils.samples_from_arrays(f"fault-parallel {points_set}", p["lon"], p["lat"], p["x_km"], p["y_km"],
                                                             p["vel_mm"], p["wgt"], look, frame_lonlat, p["cov_p"],
                                                             float(p["nug_power"]))
    print(f"points {points_set}: {p['lon'].size} samples, noise nugget {np.sqrt(p['cov_p'][0]):.3f} mm ({points_npz[points_set]})")
default_points = np.load(points_npz["default"])
assert np.array_equal(default_points["vel_mm"], model_data["data_mm"]) and np.array_equal(default_points["lon"], model_data["sample_lon"]), \
    "07a's default point set differs from its model data product"
assert np.array_equal(row07["sample_lon"], model_data["sample_lon"]), "07a's fig2_model_row.npz is not on its samples: re-run 07a"

# %% [markdown]
# ## [4] One run: the L-curve per medium, the model at the locked sigma, its numbers
#
# `model_utils.tent_model_setup` reads the run's Green's functions (one per medium; same fault mesh and impulse
# order asserted), builds the tent fault, the surface constraint and the masks. The rigidity beside the fault for
# Mw(SFSVM) comes from the run's SFCVM medium (`model_utils.mu_beside_fault`; the homogeneous model is rescaled by
# it, as 08a). The L-curve (`model_utils.tent_lcurve`) is the 07a [9] rule; its corner and pick are recorded
# (`sigma_corner`, `sigma_lcurve_pick`) and the model is solved at the locked sigma. The CSI baseline of 07a is
# re-scored on the default samples under the run's Cd. Where a forward run of the same mesh, medium, order and
# points exists, G x its slip must reproduce it (superposition, `model_utils.superposition_check`).
# The L-curve misfit is the InSAR chi2, plus the surface chi2 only in the runs that fit the surface data (until 2026-10-16
# the predicted, unfitted surface chi2 was added in the runs without the constraint too).

# %%
## [4]
def run_one(row):
    """All inversions of one run (each medium): returns the results table, the L-curves and the products."""
    t_run = time.time()
    media = list(row.media)
    run_names = {medium: gf_run_name(row, medium) for medium in media}
    medium_dirs = {medium: (medium_paths(medium)["pylith_output"], medium_paths(medium)["gf"]) for medium in media}
    samples = samples_of[row.points]
    p = np.load(points_npz[row.points])
    setup = model_utils.tent_model_setup(run_names, points_npz[row.points], strike_deg, medium_dirs, crossings, km_window,
                                         creep_lo_km, creep_hi_km, surf_floor_mm, trace_south_north, trace_xy_m,
                                         frame_lonlat, model_crs, profile_lats,
                                         near_trace_cut_km=model_near_trace_cut_km, trace_km_pp=trace["km_pp"])
    tents = setup["tents"]
    print(f"run {row.tag or '(default)'}: {', '.join(run_names.values())}; {samples.vel.size} samples x {tents.n_imp} impulses; "
          f"surface constraint {setup['constraint']['vel'].size} crossings")
    # the rigidity beside the fault: the SFCVM medium of this run (the homogeneous model is rescaled by it)
    mu_mean = {}
    for medium in media:
        if medium.startswith("sfcvm"):
            elastic_info = os.path.join(medium_paths(medium)["pylith_output"], f"{run_names[medium]}-elastic_info.h5")
            mu_mean[medium] = model_utils.mu_beside_fault(elastic_info, sfcvm_grid_file(row, medium), tents, medium)[0]
    sfcvm_media = [m for m in media if m.startswith("sfcvm")]
    assert sfcvm_media or row.dip in DIP_SCAN, f"run {row.tag}: no SFCVM medium for Mw(SFSVM)"

    def mu_for(medium):
        """Per-triangle rigidity of Mw(SFSVM): the medium's own, or the run's aligned SFCVM for the homogeneous model;
        None (Mw(SFSVM) NaN) for the homogeneous-only dip-scan rows."""
        if medium in mu_mean:
            return mu_mean[medium]
        if not sfcvm_media:
            return None
        return mu_mean["sfcvm"]

    Cd = model_utils.data_cd(row.cd, samples, p["cov_p"], float(p["nug_power"]), floor_px=row.cd_floor, keep=setup["keep"])
    lcurves = {}
    models = {}
    table = []
    if row.points == "default":                       # 07a's CSI baseline on the same samples, under this run's Cd
        res07 = row07["residual_mm"]
        table.append(dict(model="CSI homogeneous (07)", sigma=float(row07["sigma"]), rms_mm=float(np.sqrt(np.mean(res07 ** 2))),
                          rms_west300_mm=float(np.sqrt(np.mean(res07[setup["west300"]] ** 2))),
                          rms_east300_mm=float(np.sqrt(np.mean(res07[setup["east300"]] ** 2))),
                          chi2=float(np.dot(res07, np.linalg.solve(Cd, res07))), surface_pct=float(row07["surf_ratio_pct"]),
                          potency_m3=np.nan, Mw_30gpa=float(row07["mw_30gpa"]), Mw_sfcvm=np.nan, sigma_lcurve_pick=np.nan,
                          sigma_corner=np.nan, sigma_rule="07a L-curve pick", field_fingerprint=field_fingerprint, made=made))
    for medium in media:
        def solve(sigma, medium=medium):
            return model_utils.invert_tents(setup["template"], samples, setup["G"][medium], Cd, setup["surface"],
                                            setup["G_surface"], sigma, lam_xz, frame_lonlat, use_surface=bool(row.surface),
                                            name=f"{medium} sigma {sigma:g}")
        lcurve = model_utils.tent_lcurve(solve, sigma_grid, tents, mu_ref, f"{row.tag or 'default'} {medium}", use_surface=bool(row.surface))
        corner, _, _ = model_utils.lcurve_corner(lcurve.sigma.values, np.sqrt(lcurve.chi2_total.values), lcurve.prior_norm.values)
        pick = model_utils.lcurve_pick(lcurve.sigma.values, corner)
        lcurve["corner"] = corner
        lcurve["lcurve_pick"] = pick
        lcurve["selected"] = sigma_locked
        lcurve["sigma_rule"] = f"fixed {sigma_locked:g} (model_sigma_locked)"
        lcurve.to_csv(os.path.join(out_dir, f"lcurve_{medium}{row.tag}.csv"), index=False)
        lcurves[medium] = lcurve
        fault_model, insar_data, surface = solve(sigma_locked)
        slip = fault_model.slip[:, 0].copy()
        num = model_utils.tent_numbers(slip, insar_data, surface, tents, setup["west300"], setup["east300"],
                                       setup["in_section_surface"], setup["denominator_mm"], mu_ref, mu_for(medium), keep=setup["keep"])
        models[medium] = dict(slip=slip, data=insar_data.vel.copy(), model=insar_data.synth.copy(),
                              residual=insar_data.vel - insar_data.synth)
        table.append(dict(model=f"PyLith {medium}", sigma=sigma_locked, rms_mm=num["rms_mm"], rms_west300_mm=num["rms_west300_mm"],
                          rms_east300_mm=num["rms_east300_mm"], chi2=num["chi2"], surface_pct=num["surface_pct"],
                          potency_m3=num["potency_m3"], Mw_30gpa=num["Mw_30gpa"], Mw_sfcvm=num["Mw_sfcvm"],
                          peak_slip_mm=num["peak_slip_mm"], peak_depth_m=num["peak_depth_m"], sigma_lcurve_pick=pick,
                          sigma_corner=corner, sigma_rule=f"fixed {sigma_locked:g} (model_sigma_locked)",
                          field_fingerprint=field_fingerprint, made=made))
        print(f"  {medium}: L-curve corner {corner:.1f} (pick {pick:g}); at sigma {sigma_locked:g}: rms {num['rms_mm']:.3f} mm, "
              f"W/E {num['rms_west300_mm']:.2f}/{num['rms_east300_mm']:.2f}, surface {num['surface_pct']:.0f} %, "
              f"Mw30 {num['Mw_30gpa']:.3f}, Mw(SFSVM) {num['Mw_sfcvm']:.3f}")
    results = pd.DataFrame(table).set_index("model")
    results.to_csv(os.path.join(out_dir, f"results_comparison{row.tag}.csv"))

    # superposition: G x the slip of a forward run of the same mesh, medium, order and points (pylith/README.md)
    validation_c = None
    if row.mesh == "default" and row.points == "default" and row.dip == "90":
        validation_c = {}
        fwd_tag = "" if row.order == "1" else "_o2"
        for medium in media:
            fwd_run = f"fwd_default{fwd_tag}_{medium}"
            fwd_dir = medium_paths(medium)["pylith_output"]
            if not os.path.exists(os.path.join(fwd_dir, f"{fwd_run}-points.h5")):
                print(f"  superposition: no forward run {fwd_run} for {medium}; skipped")
                continue
            check = model_utils.superposition_check(setup["G"][medium], tents.imp_xyz, os.path.join(fwd_dir, f"{fwd_run}-points.h5"),
                                                    os.path.join(fwd_dir, f"{fwd_run}-fault.h5"), np.c_[p["x"], p["y"]], look,
                                                    float(setup["gnpz"][medium]["sign_rl"]))
            validation_c[medium] = check
            rel = check["rms_diff_mm"] / check["rms_signal_mm"]
            print(f"  superposition G[{medium}] x slip vs {fwd_run}: rms diff {check['rms_diff_mm']:.2e} mm ({100 * rel:.2f} % of the signal)")
            if row.order == "1":
                assert check["rms_diff_mm"] < 1e-4, f"superposition failed for {medium}: G is not consistent with {fwd_run}"
            else:                # order 2: NOT exact -- the P2 kinematic forward run evaluates its "nearest" slip db at the P2
                assert rel < 0.05, f"order-2 superposition off by {100 * rel:.1f} % for {medium}"   # edge nodes (sharper patch edge)

    products = dict(impulse_vertex_xyz=tents.imp_xyz, impulse_km=tents.imp_km, sub_faces=tents.sub_faces)
    for medium in media:
        products[f"slip_{medium}_mm"] = models[medium]["slip"]
        products[f"model_{medium}_mm"] = models[medium]["model"]
        products[f"residual_{medium}_mm"] = models[medium]["residual"]
    np.savez(os.path.join(out_dir, f"inversion_products{row.tag}.npz"), **products,
             sample_lon=samples.lon, sample_lat=samples.lat, data_mm=models[media[0]]["data"], signed_d_km=setup["signed_d_km"],
             block_lon=p["block_lon"], block_lat=p["block_lat"], trace_lon=trace["lonlat"][:, 0], trace_lat=trace["lonlat"][:, 1],
             settings=settings_json(row, samples.vel.size, run_names, tents.n_imp, field_fingerprint),
             validation_c=json.dumps(validation_c), results=results.to_json(),
             provenance=f"{notebook} {made}; runs {run_names}; 06 field {field_fingerprint}; G from "
                        + ", ".join(str(setup["gnpz"][m]["provenance"]) for m in media))
    print(f"  run {row.tag or '(default)'} done in {time.time() - t_run:.0f} s")
    return dict(results=results, lcurves=lcurves, setup=setup, run_names=run_names, Cd=Cd, mu_for=mu_for, samples=samples)

# %% [markdown]
# ## [5] All runs

# %%
## [5]
t_all = time.time()
run_outputs = {}
for row in runs.itertuples():
    run_outputs[row.tag] = run_one(row)
print(f"{len(runs)} runs in {(time.time() - t_all) / 60:.0f} min")

# %% [markdown]
# ## [6] Bootstrap of the four main models (07a [12])
#
# For the selected model of 08a (`nuggetcd` / sfcvm), its companion (`default` / sfcvm) and their homogeneous twins:
# n_boot noise realisations from each model's own data and surface covariances (Cholesky) added to its own
# synthetics, re-inverted at the locked sigma, each on fresh copies (one generator, seed 11 as 07a, in this order of
# models). The spread is the formal error of THIS configuration; the structural range is the table of [7]. Then the
# same numbers at the neighbouring grid sigmas on the real data (the sigma sensitivity; 08a's Mw ranges).
# `uncertainty_<run>_<medium>.csv`: rows `reference`, `boot_mean`, `boot_2sd`, `sigma_neighbour`, then every `boot`.
# PREPUB: Moved here from drafts/retired_2026-10-06/pylith/pylith_sfcvm_tests.py (and 08b, which re-made and checked it) on 2026-10-06:
# PREPUB: 08a quotes the bootstrap 2-sigma, so it is made before 08a.

# %%
## [6]
BOOT_RUNS = [("sfcvm", "nugget"), ("sfcvm", "full"), ("homog", "nugget"), ("homog", "full")]   # (medium, Cd); the order fixes the draws
REPORT = ["Mw_30gpa", "Mw_sfcvm", "potency_m3", "surface_pct", "peak_slip_mm", "peak_depth_m"]
rng = np.random.default_rng(n_boot_seed)
uncertainty = {}
for medium, cd in BOOT_RUNS:
    tag = CD_TAG[cd]
    run = run_outputs[tag]
    setup = run["setup"]
    tents = setup["tents"]

    def solve(sigma, vel_insar=None, vel_surface=None):
        return model_utils.invert_tents(setup["template"], run["samples"], setup["G"][medium], run["Cd"], setup["surface"],
                                        setup["G_surface"], sigma, lam_xz, frame_lonlat, use_surface=model_use_surface,
                                        vel_insar=vel_insar, vel_surface=vel_surface, name=f"{medium} {cd} sigma {sigma:g}")

    def numbers(fault_model, insar_data, surface):
        return model_utils.tent_numbers(fault_model.slip[:, 0], insar_data, surface, tents, setup["west300"], setup["east300"],
                                        setup["in_section_surface"], setup["denominator_mm"], mu_ref, run["mu_for"](medium), keep=setup["keep"])

    sigma = sigma_locked
    ref_fault, ref_data, ref_surface = solve(sigma)
    ref = numbers(ref_fault, ref_data, ref_surface)
    assert abs(ref["Mw_30gpa"] - float(run["results"].loc[f"PyLith {medium}", "Mw_30gpa"])) < 1e-12, "bootstrap reference is not the run's model"
    chol_insar = np.linalg.cholesky(ref_data.Cd + 1e-9 * np.eye(ref_data.Cd.shape[0]))
    chol_surface = np.linalg.cholesky(ref_surface.Cd + 1e-9 * np.eye(ref_surface.Cd.shape[0]))
    synth_insar = ref_data.synth.copy()
    synth_surface = ref_surface.synth.copy()
    rows = []
    t0 = time.time()
    for k in range(n_boot):
        vel_i = synth_insar + np.dot(chol_insar, rng.standard_normal(synth_insar.size))
        vel_s = synth_surface + np.dot(chol_surface, rng.standard_normal(synth_surface.size))
        fault_model, insar_data, surface = solve(sigma, vel_insar=vel_i, vel_surface=vel_s)
        rows.append(dict(kind="boot", k=k, sigma=sigma, **numbers(fault_model, insar_data, surface)))
        print(f"  {medium} {cd} boot {k + 1}/{n_boot}: Mw30 {rows[-1]['Mw_30gpa']:.3f}, surface {rows[-1]['surface_pct']:.0f} %")
    boot = pd.DataFrame(rows)
    boot_mean = {}
    boot_2sd = {}
    for col in REPORT:
        boot_mean[col] = float(boot[col].mean())
        boot_2sd[col] = float(2 * boot[col].std(ddof=1))
    summary = [dict(kind="reference", k=-1, sigma=sigma, **ref),
               dict(kind="boot_mean", k=-1, sigma=sigma, **boot_mean),
               dict(kind="boot_2sd", k=-1, sigma=sigma, **boot_2sd)]
    i_sig = sigma_grid.index(sigma)
    for j in (i_sig - 1, i_sig + 1):
        if 0 <= j < len(sigma_grid):
            fault_model, insar_data, surface = solve(sigma_grid[j])
            summary.append(dict(kind="sigma_neighbour", k=-1, sigma=sigma_grid[j], **numbers(fault_model, insar_data, surface)))
    table = pd.concat([pd.DataFrame(summary), boot], ignore_index=True)
    table["provenance"] = f"{notebook} {made}; G {run['run_names'][medium]}; Cd {cd}; seed {n_boot_seed}; 06 field {field_fingerprint}"
    out = os.path.join(out_dir, f"uncertainty_default{tag}_{medium}.csv")
    table.to_csv(out, index=False)
    uncertainty[(medium, cd)] = table
    print(f"{medium} {cd} (sigma {sigma:g}, {n_boot} realisations, {time.time() - t0:.0f} s): "
          + ", ".join(f"{col} {boot_mean[col]:.3g} +/- {boot_2sd[col]:.2g}" for col in REPORT) + f"; wrote {os.path.basename(out)}")

# %% [markdown]
# ## [7] The table of every run: `final_comparison.csv`
#
# One row per run and medium (settings, sigma, rms, near-fault rms west / east, chi2, surface %, potency, Mw at 30 GPa
# and with the SFSVM rigidity, the L-curve pick and corner, the 06 field fingerprint, the bootstrap 2-sigma where
# made), the 07a CSI baseline first. All rows must fit the same 06 field (asserted). 08b draws it and registers its
# range (`08b_sens_*`); 08a reads the alignment rows.

# %%
## [7]
PROVENANCE = ("sigma_lcurve_pick", "sigma_corner", "sigma_rule", "field_fingerprint", "made")
rows = []
for row in runs.itertuples():
    results = run_outputs[row.tag]["results"]
    settings = json.loads(str(np.load(os.path.join(out_dir, f"inversion_products{row.tag}.npz"))["settings"]))
    cd_text = row.cd + (f"{row.cd_floor}" if row.cd == "floor" else "")
    for model_name, r in results.iterrows():
        if not model_name.startswith("PyLith "):
            continue
        medium = model_name[7:]
        rows.append(dict(run="default" + row.tag, medium=medium, mesh=settings["mesh"], dip=settings["dip"].split(" (")[0],
                         Cd=cd_text, surface="on" if row.surface else "off", points=settings["points"].split(" (")[0],
                         sigma=r.sigma, rms_mm=r.rms_mm, rms_W300_mm=r.rms_west300_mm, rms_E300_mm=r.rms_east300_mm,
                         chi2=r.chi2, surface_pct=r.surface_pct, potency_m3=r.potency_m3, Mw_30gpa=r.Mw_30gpa,
                         Mw_sfcvm=r.Mw_sfcvm, summary_png=f"qc08b_summary_default{row.tag}_{medium}.png",
                         **{c: r[c] for c in PROVENANCE}))
final = pd.DataFrame(rows)
for col in ("Mw_30gpa", "Mw_sfcvm", "surface_pct"):
    final[col + "_2sd"] = np.nan
for (medium, cd), table in uncertainty.items():
    is_row = (final.run == "default" + CD_TAG[cd]) & (final.medium == medium)
    assert is_row.sum() == 1
    boot_2sd_row = table[table.kind == "boot_2sd"].iloc[0]
    for col in ("Mw_30gpa", "Mw_sfcvm", "surface_pct"):
        final.loc[is_row, col + "_2sd"] = boot_2sd_row[col]
base = run_outputs[""]["results"].loc["CSI homogeneous (07)"]
final = pd.concat([pd.DataFrame([dict(run="07 CSI baseline", medium="homog (Meade TDE)", mesh="07 smooth12", dip="90", Cd="full",
                                      surface="on", points="default", sigma=base.sigma, rms_mm=base.rms_mm,
                                      rms_W300_mm=base.rms_west300_mm, rms_E300_mm=base.rms_east300_mm, chi2=base.chi2,
                                      surface_pct=base.surface_pct, potency_m3=np.nan, Mw_30gpa=base.Mw_30gpa,
                                      Mw_sfcvm=base.Mw_sfcvm, summary_png="", **{c: base[c] for c in PROVENANCE})]), final],
                  ignore_index=True)
fingerprints = sorted(set(final.field_fingerprint.astype(str)))
assert fingerprints == [field_fingerprint], f"the runs fit different 06 fields {fingerprints}"
pylith_rows = final[final.run != "07 CSI baseline"]
assert np.allclose(pylith_rows.sigma.astype(float), sigma_locked), "a run is not at the locked sigma"
final.to_csv(os.path.join(out_dir, "final_comparison.csv"), index=False)
show = final.drop(columns=["summary_png", "field_fingerprint", "made"]).copy()
for col in ("rms_mm", "rms_W300_mm", "rms_E300_mm", "Mw_30gpa", "Mw_sfcvm"):
    show[col] = show[col].map(lambda v: f"{v:.2f}")
for col in ("chi2", "surface_pct", "potency_m3"):
    show[col] = show[col].map(lambda v: "" if pd.isna(v) else f"{v:.0f}")
for col in ("Mw_30gpa_2sd", "Mw_sfcvm_2sd", "surface_pct_2sd"):
    show[col] = show[col].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
print(show.to_string(index=False))
lines = [f"# {utils.provenance_stamp(notebook, repo_dir)}",
         f"{len(pylith_rows)} PyLith inversions on {len(runs)} runs at sigma {sigma_locked:g}, 06 field {field_fingerprint}",
         f"bootstrap: n_boot {n_boot}, seed {n_boot_seed}, models {BOOT_RUNS}"]
with open(os.path.join(out_dir, "sensitivity_summary.txt"), "w") as fh:
    fh.write("\n".join(lines) + "\n" + show.to_string(index=False) + "\n")
print(f"wrote final_comparison.csv ({len(final)} rows) and sensitivity_summary.txt")
print("08c_3d_model_sensitivity complete")
