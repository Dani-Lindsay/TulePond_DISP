"""sensitivity_rows.py: the sensitivity rows of the preferred slip model, with the settings of model_config. Each case is
solved at its own L-curve corner (the rule of model_config), except the rows of the sigma sweep.

Author: Danielle Lindsay

Rows. Inversion choices and fault dip, on the preferred run (model_config.gf_run: 3D SFSVM, fault dipping 70 deg W,
fine mesh): baseline; near-trace exclusion 30 m (scored on the same 60 m sample set as every other row); reference
frame constant; prior lengths x0.5 / x2; regularization sigma x0.25 / x0.5 / x2 / x4 (times the baseline corner); data
weighting full Cd (nugget / N^0.5 plus the exponential term, as in Jolivet et al. 2012, Neyrinck et al. 2024 and Vavra &
Fialko 2026); fault dip vertical / 70 E / 60 W / 50 W (PyLith runs on the same fine mesh). Forward-model choices, on the
vertical fault on the default (250 m) mesh, compared with the default-mesh vertical baseline: SFSVM as delivered
(unshifted), SFSVM alignment +50 / -50 m, coherence-0.7 data (its own samples, the same noise model).
Metrics (common.score): RMS of the fitted samples, within 500 m west / east in the SSE section; Mw at 30 GPa and at the
harmonic SFSVM rigidity of the run; potency; peak slip and depth; depth50 / depth90 of the potency; Fig 2K band step.

Output (model_config.sensitivity_dir): sensitivity_rows.csv, lcurve_<row>.csv, sensitivity_models.npz (slip of each row).
Run (repo root): PYTHONPATH=. nice -n 15 python scripts/sensitivity/sensitivity_rows.py      (~2.5 h)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common                                         # noqa: E402
from common import cfg                                # noqa: E402

SH, DF = cfg.gf_run, "gf_default_sfcvm_order2"     # SH = the preferred run (70 W, fine mesh); DF = vertical, 250 m mesh
# The settings of model_config, used by every case except where the case sets its own value
LOCKED = dict(points="default", cut=cfg.near_trace_cut_km, ramp=cfg.ramp_spec, lam=1.0, weighting=cfg.weighting, power=cfg.nugget_power,
              sigma="corner")
# The cases: (row, variant, baseline it belongs to, run, points, cut km, ramp, lambda factor, weighting, nugget power,
# sigma rule); sigma rule "corner" = own L-curve corner; ("x", f) = f x the corner of the row's baseline.
# The inversion-choice rows vary one setting of the preferred run; the fine-mesh dip rows are the other fault
# geometries on the same mesh; the forward-model rows use the vertical fault on the default (250 m) mesh and are compared
# with their own baseline.
CASES = [
    dict(row=f"Baseline (fine mesh, {cfg.selected_dip_label})", variant="Selected", base=SH, run=SH),
    dict(row="Near-trace exclusion", variant="30 m", base=SH, run=SH, cut=0.03),
    dict(row="Reference frame", variant="Constant", base=SH, run=SH, ramp="1"),
    dict(row="Prior lengths", variant="x0.5", base=SH, run=SH, lam=0.5),
    dict(row="Prior lengths", variant="x2", base=SH, run=SH, lam=2.0),
    dict(row="Regularization sigma", variant="x0.25", base=SH, run=SH, sigma=("x", 0.25)),
    dict(row="Regularization sigma", variant="x0.5", base=SH, run=SH, sigma=("x", 0.5)),
    dict(row="Regularization sigma", variant="x2", base=SH, run=SH, sigma=("x", 2.0)),
    dict(row="Regularization sigma", variant="x4", base=SH, run=SH, sigma=("x", 4.0)),
    dict(row="Data weighting", variant="Full Cd", base=SH, run=SH, weighting="full", power=0.5),
    dict(row="Fault dip, fine mesh", variant="Vertical", base=SH, run=cfg.gf_run_vertical),
    dict(row="Fault dip, fine mesh", variant="70 E", base=SH, run="gf_shallow_dip70e_sfcvm_order2"),
    dict(row="Fault dip, fine mesh", variant="60 W", base=SH, run="gf_shallow_dip60w_sfcvm_order2"),
    dict(row="Fault dip, fine mesh", variant="50 W", base=SH, run="gf_shallow_dip50w_sfcvm_order2"),
    dict(row="Baseline (default mesh, vertical)", variant="Selected", base=DF, run=DF),
    dict(row="Elastic structure (vertical)", variant="SFSVM as is", base=DF, run="gf_default_sfcvmasis_order2"),
    dict(row="SFSVM alignment (vertical)", variant="+50 m (e50)", base=DF, run="gf_default_sfcvm_e50_order2"),
    dict(row="SFSVM alignment (vertical)", variant="-50 m (w50)", base=DF, run="gf_default_sfcvm_w50_order2"),
    dict(row="Data coherence (vertical)", variant="0.7", base=DF, run="gf_default_sfcvm_coh07_order2", points="coh07"),
]
assert SH != DF and cfg.gf_run_vertical != SH
ONLY = os.environ.get("ONLY")                         # a comma-separated subset of rows (e.g. a re-run of one row)

if __name__ == "__main__":
    elapsed = common.timer()
    inputs = common.load_inputs()
    samples = {}
    setups = {}
    mu_harmonic = {}
    base_corner = {}
    rows, slips = [], {}
    for case in CASES:
        c = dict(LOCKED, **case)
        if ONLY and c["row"] not in ONLY.split(","):
            continue
        label = f"{c['row']} {c['variant']}"
        key = (c["run"], c["points"], c["cut"])
        if key not in setups:
            setups[key] = common.make_setup(inputs, c["run"], c["points"], c["cut"])
        setup = setups[key]
        score_key = (c["run"], c["points"], cfg.near_trace_cut_km)       # every row is scored on the 60 m set of its points
        if score_key not in setups:
            setups[score_key] = common.make_setup(inputs, c["run"], c["points"], cfg.near_trace_cut_km)
        if c["points"] not in samples:
            samples[c["points"]] = common.samples_for(inputs, c["points"])
        if c["run"] not in mu_harmonic:
            mu_harmonic[c["run"]] = common.mu_harmonic_of(c["run"], setup)
        G = setup["G"]["m"]
        settings = dict(weighting=c["weighting"], nug_power=c["power"], lam_xz=(cfg.lam_xz_km[0] * c["lam"], cfg.lam_xz_km[1] * c["lam"]),
                        ramp_spec=c["ramp"])
        if c["sigma"] == "corner":
            sigma, sigma_base = common.own_corner(inputs, setup, G, samples[c["points"]], label, **settings)
            if c["run"] == c["base"] and c["variant"] == "Selected":                   # the baseline of its group
                base_corner[c["base"]] = sigma
        else:
            sigma, sigma_base = c["sigma"][1] * base_corner[c["base"]], np.nan
        fault_model, insar_data = common.solve(inputs, setup, G, samples[c["points"]], sigma, label=label, **settings)
        numbers, slip = common.score(inputs, setup, fault_model, insar_data, score_setup=setups[score_key],
                                     mu_per_triangle=mu_harmonic[c["run"]])
        rows.append(dict(row=c["row"], variant=c["variant"], baseline=c["base"], run=c["run"], points=c["points"], cut_m=1e3 * c["cut"],
                         scored_cut_m=1e3 * cfg.near_trace_cut_km, ramp=c["ramp"], lam_x_km=settings["lam_xz"][0],
                         lam_z_km=settings["lam_xz"][1], weighting=c["weighting"], nugget_power=c["power"], sigma=sigma,
                         sigma_base_corner=sigma_base, **numbers))
        slips[label] = slip
        common.write(rows, "sensitivity_rows.csv", __file__)
        print(f"done: {label}: sigma {sigma:.2f}, Mw {numbers['Mw_30gpa']:.3f} / {numbers['Mw_sfcvm']:.3f}, RMS {numbers['rms_mm']:.3f}, "
              f"W500 {numbers['rms_west_mm']:.3f}, E500 {numbers['rms_east_mm']:.3f}, depth90 {numbers['depth90_m']:.0f} m ({elapsed()})",
              flush=True)
    np.savez(os.path.join(common.OUT, "sensitivity_models.npz"), labels=np.array(list(slips)),
             slip_mm=np.array([slips[k] for k in slips], dtype=object), allow_pickle=True)
    print(f"wrote {common.OUT}/sensitivity_rows.csv ({len(rows)} rows) in {elapsed()}")
