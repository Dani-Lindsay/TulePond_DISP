"""sensitivity_rows.py: every sensitivity row of the selected model at the locked settings (model_config), each case at its
OWN L-curve corner (model_config rule) unless the row is the sigma sweep. Ported 2026-10-30 from
sensitivity_2026-10-27/sensitivity_rows.py (which exec'd 07a / 08c) onto scripts/sensitivity/common.py, with the rows the
rewire brief added: the full-Cd weighting on the selected (shallow) mesh, and the 30 m cut scored on the same 1787
samples as every other row.

Rows (2026-10-30: the selected model dips 70 W, model_config.gf_run). Inversion choices on the selected run
(gf_shallow_dip70w_sfcvm_order2, fine mesh): baseline; near-trace exclusion 30 m;
reference frame constant; prior lengths x0.5 / x2; regularization sigma x0.25 / x0.5 / x2 / x4 (x the baseline corner);
data weighting full Cd (nugget / N^0.5 + exponential, the Jolivet 2012 / Neyrinck 2024 / Vavra & Fialko 2026 choice);
fault dip vertical / 70 E / 60 W / 50 W (the fine-mesh dip runs). Forward-model choices on the VERTICAL fault on the DEFAULT
mesh against the default-mesh vertical baseline: SFSVM unshifted, alignment +50 / -50 m, coherence-0.7 data (its own
samples, the same noise model).
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

SH, DF = cfg.gf_run, "gf_default_sfcvm_order2"     # SH = the SELECTED run (70 W, fine mesh, since 2026-10-30); DF = vertical, 250 m mesh
# (row, variant, baseline it belongs to, run, points, cut km, ramp, lambda factor, weighting, nugget power, sigma rule)
#   sigma rule "corner" = own L-curve corner; ("x", f) = f x the corner of the row's baseline
LOCKED = dict(points="default", cut=cfg.near_trace_cut_km, ramp=cfg.ramp_spec, lam=1.0, weighting=cfg.weighting, power=cfg.nugget_power,
              sigma="corner")
# 2026-10-30 (Dani): the selected model dips 70 W (model_config.gf_run). The inversion-choice rows are around it; the fine-mesh
# dip rows are the other geometries; the forward-model rows stay on the vertical default (250 m) mesh against their own baseline.
CASES = [
    dict(row="Baseline (fine mesh, 70 W)", variant="Selected", base=SH, run=SH),
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
    # Dani 2026-10-30: does the 3D fit plateau toward shallower west dips, as the homogeneous one does? (weekend runs)
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
        fault_model, insar_data, _ = common.solve(inputs, setup, G, samples[c["points"]], sigma, label=label, **settings)
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
