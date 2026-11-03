"""common.py: the set-up and the scoring shared by every sensitivity script (scripts/sensitivity/*.py).

Author: Danielle Lindsay

The settings come from model_config.py, the data from the products of notebook 07, the modelling functions from
model_utils.py. One set-up (`make_setup`), one solve (`solve`), one corner finder (`own_corner`, the rule of
model_config) and one scoring (`score`: RMS of the fitted samples, near-field RMS, Mw, peak slip, depth50 / depth90)
are shared by all scripts.

The outputs of every script go to model_config.sensitivity_dir (archived on Zenodo); notebook 10_model_sensitivity draws them.
Run a script from the repository root in the environment of the notebooks, at low priority:
    PYTHONPATH=. nice -n 15 python scripts/sensitivity/<script>.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from TulePond_filepaths import common_paths, paths_out, paths_sfcvm, model_crs, repo_dir   # noqa: E402
import insar_utils as utils                                                                   # noqa: E402
import model_config as cfg                                                                    # noqa: E402
import model_utils                                                                            # noqa: E402

OUT = cfg.sensitivity_dir
MARK = os.path.join(os.path.dirname(paths_sfcvm["root"]), "pylith_runs_done")   # done markers of the PyLith runs


def stamp(script):
    """The provenance string every output row carries: script, git hash, time."""
    return f"{script} | {utils.provenance_stamp(os.path.basename(script), repo_dir)}"


def load_inputs():
    """The data products of 07 and everything the set-ups need."""
    model_data_file = os.path.join(paths_out["07_model_data"], "model_data.npz")
    model_data = np.load(model_data_file)
    assert str(model_data["field_fingerprint"]) == model_utils.field_fingerprint(
        os.path.join(paths_out["06_event_field_geometry"], "fp_maps.npz")), "07's model data are from another 06 field"
    frame_lonlat = tuple(model_data["frame_lonlat"])
    profile_lats = pd.read_csv(common_paths["fault_profiles"]).ft_lat.values
    trace = model_utils.model_trace(model_data["trace_lonlat"], -90.0, 90.0, frame_lonlat, profile_lats,
                                    every_km=float(model_data["trace_every_km"]))
    crossings = pd.read_csv(os.path.join(paths_out["04_event_detection"], "event_fraction_by_crossing.csv"))
    return dict(model_data=model_data, frame_lonlat=frame_lonlat, strike_deg=float(model_data["strike_deg"]),
                look=model_data["look"], cov_p=np.asarray(model_data["cov_p"], float), profile_lats=profile_lats, trace=trace,
                trace_xy_m=(trace["x_km"] * 1e3, trace["y_km"] * 1e3), section_km=cfg.sse_section_km(), band_km=cfg.band90_km(),
                crossing_km=crossings.km.values,
                points={"default": str(model_data["points_default"]), "coh07": str(model_data["points_coh07"])})


def samples_for(inputs, points_set):
    """A point set (npz of 07) as a csi.insar data set (positions, data, block pixel counts)."""
    p = np.load(inputs["points"][points_set])
    return model_utils.samples_from_arrays(f"fault-parallel {points_set}", p["lon"], p["lat"], p["x_km"], p["y_km"], p["vel_mm"],
                                           p["wgt"], inputs["look"], inputs["frame_lonlat"], inputs["cov_p"], cfg.nugget_power)


def make_setup(inputs, run, points_set="default", cut_km=cfg.near_trace_cut_km, run_dir=None):
    """model_utils.tent_model_setup for one Green's-function run (PyLith: its done marker is asserted; or an analytic G
    stored in the same npz layout, run_dir = (output dir, gf dir))."""
    if run_dir is None:
        assert os.path.exists(os.path.join(MARK, run)), f"{run}: no done marker in {MARK}"
        run_dir = (paths_sfcvm["pylith_output"], paths_sfcvm["gf"])
    lo, hi = inputs["section_km"]
    return model_utils.tent_model_setup({"m": run}, inputs["points"][points_set], inputs["strike_deg"], {"m": run_dir},
                                        lo, hi, inputs["trace_xy_m"], inputs["frame_lonlat"], model_crs, inputs["profile_lats"],
                                        cut_km, cfg.near_field_km, inputs["trace"]["km_pp"])


def solve(inputs, setup, G, samples, sigma, weighting=cfg.weighting, nug_power=cfg.nugget_power, lam_xz=cfg.lam_xz_km,
          ramp_spec=cfg.ramp_spec, keep_fit=None, label="", vel=None):
    """One inversion by the method of model_config (model_utils.invert_tents), with the data covariance of `weighting` /
    `nug_power` (model_utils.data_cd) over the fitted samples keep_fit (default: the set-up's keep)."""
    keep_fit = setup["keep"] if keep_fit is None else keep_fit
    Cd = model_utils.data_cd(weighting, samples, inputs["cov_p"], nug_power, keep=keep_fit)
    return model_utils.invert_tents(setup["template"], samples, G, Cd, sigma, lam_xz, inputs["frame_lonlat"], vel_insar=vel,
                                    name=label, ramp_spec=ramp_spec)


def own_corner(inputs, setup, G, samples, label, keep_fit=None, **settings):
    """The model's own L-curve corner by the rule of model_config (model_utils.own_corner); the L-curve is saved as
    lcurve_<label>.csv in OUT. Returns (corner, base-grid corner)."""
    keep_fit = setup["keep"] if keep_fit is None else keep_fit

    def solve_at(sigma):
        return solve(inputs, setup, G, samples, sigma, keep_fit=keep_fit, label=f"{label} sigma {sigma:.3f}", **settings)
    corner, base_corner, table = model_utils.own_corner(solve_at, cfg.sigma_grid_base, cfg.sigma_refine_factors, setup["tents"],
                                                        cfg.mu_ref_pa, label, keep_fit)
    os.makedirs(OUT, exist_ok=True)
    table.assign(corner=corner, base_corner=base_corner).to_csv(
        os.path.join(OUT, "lcurve_" + "".join(ch if ch.isalnum() else "_" for ch in label) + ".csv"), index=False)
    return corner, base_corner


def mu_harmonic_of(run, setup):
    """Per-triangle harmonic SFSVM rigidity beside the fault of a PyLith SFSVM run (model_utils.mu_beside_fault on the
    run's elastic_info and the grid it read), or None for a run that is not SFSVM."""
    grid_of = {"gf_default_sfcvmasis_order2": "sfcvm_v26-0_asis_nocap_grid.npz",
               "gf_default_sfcvm_e50_order2": "sfcvm_v26-0_aligned_nocap_e50_grid.npz",
               "gf_default_sfcvm_w50_order2": "sfcvm_v26-0_aligned_nocap_w50_grid.npz"}
    if "sfcvm" not in run:
        return None
    grid = os.path.join(paths_sfcvm["spatialdb"], grid_of.get(run, "sfcvm_v26-0_aligned_nocap_grid.npz"))
    elastic_info = os.path.join(paths_sfcvm["pylith_output"], f"{run}-elastic_info.h5")
    _, mu_west, mu_east, rel = model_utils.mu_beside_fault(elastic_info, grid, setup["tents"], run)
    assert rel < 0.05, f"{run}: PyLith's rigidity differs from {os.path.basename(grid)} by {rel:.3f} (median)"
    return model_utils.harmonic_mean_rigidity(mu_west, mu_east)


def score(inputs, setup, fault_model, insar_data, score_setup=None, mu_per_triangle=None):
    """The metrics of one model: model_utils.tent_numbers over the fitted samples and the near-field masks of
    score_setup (default: the model's own set-up; the 30 m row is scored on the 60 m set), plus the band step of Fig 2K:
    data minus model step 60-150 m at the crossings of 04 in the 90 % band (mean, RMS). Residuals are against the
    plane-removed data (CSI removes the fitted frame from vel)."""
    score_setup = setup if score_setup is None else score_setup
    slip = fault_model.slip[:, 0].copy()
    numbers = model_utils.tent_numbers(slip, insar_data, setup["tents"], score_setup["keep"], score_setup["west_near"],
                                       score_setup["east_near"], cfg.mu_ref_pa, mu_per_triangle)
    steps = []
    for values in (insar_data.vel, insar_data.synth):
        steps.append(model_utils.crossing_steps_like(values, setup["signed_d_km"], setup["sample_km_pp"], inputs["crossing_km"],
                                                     cfg.fig2k_inner_km, cfg.fig2k_outer_km))
    band = (inputs["crossing_km"] >= inputs["band_km"][0]) & (inputs["crossing_km"] <= inputs["band_km"][1])
    step_residual = (steps[0] - steps[1])[band]
    step_residual = step_residual[np.isfinite(step_residual)]
    numbers.update(band_step_mean_mm=float(step_residual.mean()), band_step_rms_mm=float(np.sqrt(np.mean(step_residual ** 2))))
    return numbers, slip


def write(rows, name, script):
    """The table, with the provenance column, to OUT/name (rewritten after every row, so that an interrupted run keeps
    the rows already computed)."""
    os.makedirs(OUT, exist_ok=True)
    table = pd.DataFrame(rows)
    table["provenance"] = stamp(script)
    path = os.path.join(OUT, name)
    table.to_csv(path, index=False)
    return path


def timer():
    # returns a function that gives the elapsed time since this call, in minutes, as text
    start = time.time()
    return lambda: f"{(time.time() - start) / 60:.1f} min"
