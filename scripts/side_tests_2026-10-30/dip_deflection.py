"""dip_deflection.py: the M7 near-fault deflection (west / east) for the fault-dip sweep models (Dani 2026-10-30:
"which are not improved with tests of fault dip" - is that true for the homogeneous model?).

Re-solves each dip model of scripts/sensitivity/dip_rows.py at the sigma that dip_rows.csv recorded (no new corner
search), with the same set-up, solve and Green's functions (scripts/sensitivity/common.py), then measures
model_utils.side_deflection_mm (mean 60-250 m from the trace minus mean 1.5-3 km, SSE section, fitted samples) on
the plane-removed data and on the model. Asserts that the re-solve reproduces dip_rows.csv's RMS.
Run: cd ~/tools/internal/TulePond_DISP && PYTHONPATH=. OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 \
     nice -n 15 ~/miniforge3/envs/opera_disp_feb13/bin/python ~/work/proj_TulePond_DISP/dip_deflection_2026-10-30/dip_deflection.py
"""
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.expanduser("~/tools/internal/TulePond_DISP")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "sensitivity"))
import common                                   # noqa: E402
import dip_rows                                 # noqa: E402  (its run tables only; its __main__ does not run)
import model_utils                              # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
NEAR_KM, FAR_KM = (0.06, 0.25), (1.5, 3.0)       # as 08a's M7 metric
CASES = [("vertical", "Homogeneous analytic N^0"), ("dip80w", "Homogeneous analytic N^0"), ("dip70w", "Homogeneous analytic N^0"),
         ("dip60w", "Homogeneous analytic N^0"), ("dip50w", "Homogeneous analytic N^0"),
         ("vertical", "3D N^0"), ("dip70w", "3D N^0")]

recorded = pd.read_csv(os.path.join(common.OUT, "dip_rows.csv"))
inputs = common.load_inputs()
samples = common.samples_for(inputs, "default")
lo, hi = inputs["section_km"]
rows = []
for geometry, family in CASES:
    row = recorded[(recorded.geometry == geometry) & (recorded.family == family)]
    assert len(row) == 1, (geometry, family)
    row = row.iloc[0]
    medium, weighting, power = dip_rows.FAMILIES[family]
    if geometry in dip_rows.PYLITH_RUNS:
        run = dip_rows.PYLITH_RUNS[geometry]
        setup = common.make_setup(inputs, run)
        G = setup["G"]["m"] if medium == "pylith" else np.load(os.path.join(dip_rows.DIP_CHECKS, f"G_analytic_{run}_nsub4.npy"))
    else:
        run = dip_rows.ANALYTIC_ONLY[geometry]
        gf_dir = os.path.join(dip_rows.DIP_CHECKS, "gf_analytic")
        setup = common.make_setup(inputs, run, run_dir=(gf_dir, gf_dir))
        G = setup["G"]["m"]
    fault_model, insar_data, _ = common.solve(inputs, setup, G, samples, float(row.sigma), weighting=weighting, nug_power=power,
                                              label=f"{geometry} {family}")
    numbers, _ = common.score(inputs, setup, fault_model, insar_data)
    assert abs(numbers["rms_mm"] - row.rms_mm) < 1e-3, f"{geometry} {family}: RMS {numbers['rms_mm']:.4f} vs recorded {row.rms_mm:.4f}"
    in_section = (setup["sample_km_pp"] >= lo) & (setup["sample_km_pp"] <= hi)
    out = dict(geometry=geometry, family=family, sigma=float(row.sigma), rms_mm=numbers["rms_mm"],
               rms_west_mm=numbers["rms_west_mm"], rms_east_mm=numbers["rms_east_mm"], Mw_30gpa=numbers["Mw_30gpa"])
    for name, values in (("data", np.asarray(insar_data.vel, float)), ("model", np.asarray(insar_data.synth, float))):
        for side, tag in ((-1, "west"), (1, "east")):
            value, n_near, n_far = model_utils.side_deflection_mm(values, setup["signed_d_km"], in_section, setup["keep"], side, NEAR_KM, FAR_KM)
            out[f"{name}_{tag}_mm"] = value
            out[f"n_{tag}"] = f"{n_near}/{n_far}"
    rows.append(out)
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()}, flush=True)
    table = pd.DataFrame(rows)
    table["provenance"] = common.stamp(__file__)
    table.to_csv(os.path.join(OUT, "dip_deflection.csv"), index=False)
print("wrote", os.path.join(OUT, "dip_deflection.csv"))
