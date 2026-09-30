#!/usr/bin/env python3
"""
DRAFT (2026-09-30): extend 07_slip_model_tests' dip family towards the WEST.
The suite (60 E .. 75 W, sigma 8) improves monotonically towards west dips and
is still improving at 75 W, so the minimum was never reached (Dani: "InSAR
with fault dip, have we tried that?"). Homogeneous CSI model, 07's machinery
(run_case, record=False), sigma fixed at the model's pick. Dips 90 (vertical)
then 80, 75, 70, 60, 50, 45 W (= 100..135 in the suite's convention, 105 = 75 W).
Output: outputs/07_slip_model_tests/dip_west_scan.csv
"""
import os

source = open(os.path.join("scripts", "07_slip_model_tests.py")).read()
exec(source.split("# %% [markdown]\n# ## [8]")[0])

rows = []
for dip in (90, 100, 105, 110, 120, 130, 135):
    row, flt, ds, surface = run_case("dip_west", f"dip{dip}", dip=dip, record=False)
    resid = ds.vel - ds.synth
    d_signed = dist_to_trace(ds, flt, signed=True)
    near_w, near_e = (np.abs(d_signed) <= 0.3) & (d_signed < 0), (np.abs(d_signed) <= 0.3) & (d_signed > 0)
    row.update(dip_west_deg=180 - dip if dip > 90 else (90 if dip == 90 else np.nan),
               rms_west300_mm=float(np.sqrt(np.mean(resid[near_w] ** 2))), rms_east300_mm=float(np.sqrt(np.mean(resid[near_e] ** 2))))
    rows.append(row)
    print(f"dip {dip:3d} ({row['dip_west_deg']:g} W): chi2 {row['chi2']:.0f} + surf {row['chi2_surf']:.0f}; rms {row['rms_mm']:.3f} "
          f"(near {row['rms_near_mm']:.3f}; W {row['rms_west300_mm']:.2f} E {row['rms_east300_mm']:.2f}); Mw {row['Mw']:.3f}; "
          f"surface {row['surface_pct']:.0f}%")
scan = pd.DataFrame(rows)
scan.to_csv(os.path.join(out_dir, "dip_west_scan.csv"), index=False)
