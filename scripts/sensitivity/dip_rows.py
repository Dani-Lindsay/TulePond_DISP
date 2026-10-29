"""dip_rows.py: the fault-dip sweep on the DEFAULT mesh (the dip column of the sweeps figure; register #12). Ported 2026-10-30
from dip_checks_2026-10-26/dip_inversions.py + analytic_west_scan.py + rescore_dips_depth90.py (which exec'd 07a / 08c)
onto scripts/sensitivity/common.py: one solve, one corner rule (model_config), one scoring.

Geometries: vertical, 80 E / 80 W, 70 E / 70 W (PyLith order-2 SFSVM runs on the default mesh, `gf_default[_dip..]_sfcvm_order2`)
and 60 W / 50 W (no PyLith run: analytic only). Families: "3D N^0" (the locked weighting), "3D full Cd" (nugget / N^0.5 +
exponential) and "Homogeneous analytic N^0" (CSI Meade triangular dislocations on the SAME tents, model_utils.analytic_tent_G).
The analytic matrices are the caches of 26 Oct (n_sub 4; dip_checks_2026-10-26/G_analytic_<run>_nsub4.npy in each run's impulse
order, gf_analytic/G_analytic_dip{60w,50w}_nsub4.npz): n_sub 4 vs 8 differs by up to 1.5 % for surface tents on this mesh
(validate_analytic.py), recomputing at n_sub 8 would take ~2.5 h per geometry. Each analytic matrix is checked against its
PyLith run (same-sign columns) before use.
Output (model_config.sensitivity_dir): dip_rows.csv, lcurve_<geometry>_<family>.csv.
Run (repo root): PYTHONPATH=. nice -n 15 python scripts/sensitivity/dip_rows.py      (~2 h)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common                                         # noqa: E402
from common import cfg                                # noqa: E402
from TulePond_filepaths import data_dir               # noqa: E402

DIP_CHECKS = os.path.join(data_dir, "dip_checks_2026-10-26")
PYLITH_RUNS = {"vertical": "gf_default_sfcvm_order2", "dip80e": "gf_default_dip80e_sfcvm_order2", "dip80w": "gf_default_dip80w_sfcvm_order2",
               "dip70e": "gf_default_dip70e_sfcvm_order2", "dip70w": "gf_default_dip70w_sfcvm_order2"}
ANALYTIC_ONLY = {"dip60w": "analytic_dip60w_nsub4", "dip50w": "analytic_dip50w_nsub4"}     # G_<name>.npz in gf_analytic/
DIP_DEG = {"dip50w": -40, "dip60w": -30, "dip70w": -20, "dip80w": -10, "vertical": 0, "dip80e": 10, "dip70e": 20}   # from vertical, W < 0
FAMILIES = {"3D N^0": ("pylith", cfg.weighting, cfg.nugget_power), "3D full Cd": ("pylith", "full", 0.5),
            "Homogeneous analytic N^0": ("analytic", cfg.weighting, cfg.nugget_power)}

if __name__ == "__main__":
    elapsed = common.timer()
    inputs = common.load_inputs()
    samples = common.samples_for(inputs, "default")
    rows = []
    for geometry in list(PYLITH_RUNS) + list(ANALYTIC_ONLY):
        if geometry in PYLITH_RUNS:
            run = PYLITH_RUNS[geometry]
            setup = common.make_setup(inputs, run)
            G_pylith = setup["G"]["m"]
            G_analytic = np.load(os.path.join(DIP_CHECKS, f"G_analytic_{run}_nsub4.npy"))
            assert G_analytic.shape == G_pylith.shape, f"{geometry}: analytic G {G_analytic.shape} vs PyLith {G_pylith.shape}"
            same_sign = np.einsum("ij,ij->j", G_analytic, G_pylith) > 0
            assert same_sign.mean() > 0.99, f"{geometry}: analytic and PyLith columns disagree in sign for {(~same_sign).sum()} tents"
            matrices = {"pylith": G_pylith, "analytic": G_analytic}
            mu_harmonic = common.mu_harmonic_of(run, setup)
        else:
            run = ANALYTIC_ONLY[geometry]
            gf_analytic = os.path.join(DIP_CHECKS, "gf_analytic")
            setup = common.make_setup(inputs, run, run_dir=(gf_analytic, gf_analytic))
            matrices = {"analytic": setup["G"]["m"]}
            mu_harmonic = None
        for family, (medium, weighting, power) in FAMILIES.items():
            if medium not in matrices:
                continue
            label = f"{geometry} {family}"
            settings = dict(weighting=weighting, nug_power=power)
            sigma, sigma_base = common.own_corner(inputs, setup, matrices[medium], samples, label, **settings)
            fault_model, insar_data, _ = common.solve(inputs, setup, matrices[medium], samples, sigma, label=label, **settings)
            numbers, _ = common.score(inputs, setup, fault_model, insar_data, mu_per_triangle=mu_harmonic if medium == "pylith" else None)
            rows.append(dict(geometry=geometry, dip_from_vertical_deg=DIP_DEG[geometry], family=family, run=run, weighting=weighting,
                             nugget_power=power, sigma=sigma, sigma_base_corner=sigma_base, **numbers))
            common.write(rows, "dip_rows.csv", __file__)
            print(f"done: {label}: sigma {sigma:.2f}, Mw {numbers['Mw_30gpa']:.3f}, RMS {numbers['rms_mm']:.3f}, W500 "
                  f"{numbers['rms_west_mm']:.3f}, E500 {numbers['rms_east_mm']:.3f} ({elapsed()})", flush=True)
    print(f"wrote {common.OUT}/dip_rows.csv ({len(rows)} rows) in {elapsed()}")
