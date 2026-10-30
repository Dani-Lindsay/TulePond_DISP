"""stress_drop.py: LOCKED 2026-10-30 (Dani): Kanamori & Anderson (1975) Eq. 1 with C = 2/pi and L = w for a
shallow, long strike-slip fault including the free surface (Knopoff 1958): delta-sigma = (2/pi) mu D / w, D the average
(uniform) slip, w the down-dip width. Materna et al. (2024 SI Eq. 2: mu d / x, x = half the depth, a dimensional
estimate) is kept as a second column for comparison.
Original description: stress drop of the Tule Pond mean SSE for the vertical and the 70 W fault (Dani 2026-10-30: lock the
stress-drop calculation; dip still open). Formula as Materna et al. 2024 (GRL, doi 10.1029/2023GL107244 ... see README):
delta-tau = mu d / (W / 2), d = best uniform slip of the fixed-depth scan, W = its down-dip width (= best vertical depth D
for the vertical fault, D / sin(70) for 70 W). Check: Materna's Superstition Hills numbers (2 km, 20 mm, 4 GPa) give
80 kPa with this formula, as quoted in their main text. mu at 30 GPa and at the harmonic mean (Vavrycuk 2013) of the
area-weighted SFSVM rigidity west and east of the fault triangles in the SSE section from 0 to D (as 08a [11]).
Inputs: the depth scans (vertical: sensitivity_2026-10-30/retired_2026-10-30_vertical/depth_scan.csv; 70 W:
sensitivity_2026-10-30/depth_scan.csv), the PyLith elastic_info of each run and the aligned SFSVM grid.
Run: cd ~/tools/internal/TulePond_DISP && PYTHONPATH=. nice -n 15 <env python> <this file>"""
import os, sys
import numpy as np
import pandas as pd
REPO = os.path.expanduser("~/tools/internal/TulePond_DISP")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts", "sensitivity"))
import common
from common import cfg, model_utils
from TulePond_filepaths import data_dir, paths_sfcvm

OUT = os.path.dirname(os.path.abspath(__file__))
assert abs(model_utils.materna_stress_drop_pa(4e9, 0.020, 1000.0) - 80e3) < 1, "formula does not reproduce Materna's 80 kPa"
sens = os.path.join(data_dir, "sensitivity_2026-10-30")
cases = [("Vertical", "gf_shallow_sfcvm_order2", 90.0, os.path.join(sens, "retired_2026-10-30_vertical", "depth_scan.csv")),
         ("70 W", "gf_shallow_dip70w_sfcvm_order2", 70.0, os.path.join(sens, "depth_scan.csv"))]
inputs = common.load_inputs()
lo_km, hi_km = inputs["section_km"]
grid = os.path.join(paths_sfcvm["spatialdb"], "sfcvm_v26-0_aligned_nocap_grid.npz")
rows = []
for label, run, dip_deg, scan_file in cases:
    scan = pd.read_csv(scan_file)
    scan = scan[scan.top_km == 0.0] if "top_km" in scan else scan
    best = scan.loc[scan.rms_all_mm.idxmin()]
    depth_km, slip_mm = float(best.D_km), float(best.slip_mm)
    width_m = depth_km * 1e3 / np.sin(np.radians(dip_deg))
    setup = common.make_setup(inputs, run)
    tents = setup["tents"]
    elastic_info = os.path.join(paths_sfcvm["pylith_output"], f"{run}-elastic_info.h5")
    _, mu_west, mu_east, rel = model_utils.mu_beside_fault(elastic_info, grid, tents, run)
    upper = (tents.tri_depth_m <= depth_km * 1e3) & (tents.tri_km >= lo_km) & (tents.tri_km <= hi_km)
    mu_w = float(np.average(mu_west[upper], weights=tents.tri_area[upper]))
    mu_e = float(np.average(mu_east[upper], weights=tents.tri_area[upper]))
    mu_h = float(model_utils.harmonic_mean_rigidity(mu_w, mu_e))
    for rig_label, mu in (("30 GPa", 30e9), ("SFSVM harmonic west/east, 0 to D", mu_h)):
        rows.append(dict(dip=label, run=run, best_depth_km=depth_km, slip_mm=slip_mm, downdip_width_m=width_m, rigidity=rig_label,
                         mu_west_gpa=mu_w / 1e9, mu_east_gpa=mu_e / 1e9, mu_gpa=mu / 1e9,
                         stress_drop_kpa=(2 / np.pi) * mu * slip_mm * 1e-3 / width_m / 1e3,     # LOCKED: Knopoff 1958 / Kanamori & Anderson 1975 Eq. 1, C = 2/pi
                         stress_drop_materna_kpa=model_utils.materna_stress_drop_pa(mu, slip_mm * 1e-3, width_m / 2) / 1e3))
table = pd.DataFrame(rows)
table.to_csv(os.path.join(OUT, "stress_drop.csv"), index=False)
print(table.round(3).to_string(index=False))
