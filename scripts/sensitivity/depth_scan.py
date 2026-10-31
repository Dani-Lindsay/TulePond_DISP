"""depth_scan.py: the fixed-depth scan of the preferred forward model (model_config.gf_run), used for the depth extent of
slip and the stress drop (notebook 09, section 12). The depth-scan functions are model_utils.uniform_slip_shape and
model_utils.uniform_slip_fit.

Author: Danielle Lindsay

For each D: uniform right-lateral slip from the surface to D over the SSE section (the creep length of notebook 04,
tapered to zero over 0.5 km at each end), on the fault of model_config.gf_run (dipping 70 deg W; D is the vertical
depth; the section ends are along strike on the unrolled fault, TentFault.imp_km); one slip amplitude and a plane
(1, x, y) solved by least squares with the weighting of model_config (equal weights), samples within 60 m of the trace
left out. No smoothing, no sigma. Tents: full slip at the vertices down to D, zero at the next vertex row (one mesh
cell, 60 m above 600 m depth).
Metrics against the plane-removed data: RMS of the fitted samples; RMS within 500 m west / east in the SSE section.
Noise test: the best uniform-slip model (smallest RMS) predicts synthetic data; 100 correlated-noise realisations (full
covariance of the 5 km noise model, nugget / N^0.5, seed model_config.mc_seed) are scanned in the same way as the
data; the best D of each realisation gives the spread.
Buried top: the scan repeated with zero slip above 50 / 100 / 200 m.
Outputs (model_config.sensitivity_dir): depth_scan.csv, depth_scan_noise.csv, depth_scan_buried_top.csv.
Run (repo root): PYTHONPATH=. nice -n 15 python scripts/sensitivity/depth_scan.py      (~5 min)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common                                         # noqa: E402
from common import cfg, model_utils                   # noqa: E402

BURIED_TOPS_KM = (0.0, 0.05, 0.1, 0.2)


def scan(G, tents, data_mm, x_km, y_km, setup, section_km, top_km=0.0):
    """One scan over model_config.depth_scan_depths_km: amplitude, potency, Mw (30 GPa) and the RMS metrics per D."""
    keep, west, east = setup["keep"], setup["west_near"], setup["east_near"]
    rows = []
    for depth_km in cfg.depth_scan_depths_km:
        if depth_km <= top_km or depth_km > tents.imp_depth_km.max() + 1e-6:
            continue
        shape = model_utils.uniform_slip_shape(tents, section_km, cfg.depth_scan_taper_km, depth_km, top_km)
        amplitude_mm, residual = model_utils.uniform_slip_fit(G, shape, data_mm, x_km, y_km, keep)
        potency = tents.potency_m3(shape * amplitude_mm)
        rows.append(dict(top_km=top_km, D_km=depth_km, slip_mm=amplitude_mm, potency_m3=potency,
                         Mw_30gpa=float(model_utils.moment_magnitude(cfg.mu_ref_pa * potency)),
                         rms_all_mm=float(np.sqrt(np.mean(residual[keep] ** 2))), rms_W500_mm=float(np.sqrt(np.mean(residual[west] ** 2))),
                         rms_E500_mm=float(np.sqrt(np.mean(residual[east] ** 2))), mean_res_W500_mm=float(np.mean(residual[west])),
                         mean_res_E500_mm=float(np.mean(residual[east]))))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    elapsed = common.timer()
    inputs = common.load_inputs()
    setup = common.make_setup(inputs, cfg.gf_run)
    G, tents = setup["G"]["m"], setup["tents"]
    p = np.load(inputs["points"]["default"])
    data = p["vel_mm"].astype(float)
    x_km, y_km = p["x_km"].astype(float), p["y_km"].astype(float)
    section = inputs["section_km"]

    table = scan(G, tents, data, x_km, y_km, setup, section)
    path = common.write(table.to_dict("records"), "depth_scan.csv", __file__)
    best = table.loc[table.rms_all_mm.idxmin()]
    print(table.round(3).to_string(index=False))
    print(f"best D {best.D_km:.2f} km (RMS {best.rms_all_mm:.3f} mm, slip {best.slip_mm:.2f} mm); wrote {path} ({elapsed()})", flush=True)

    # noise test: the best uniform-slip model plus its plane predicts the data; correlated noise is added; each
    # realisation is scanned
    shape = model_utils.uniform_slip_shape(tents, section, cfg.depth_scan_taper_km, best.D_km)
    _, residual = model_utils.uniform_slip_fit(G, shape, data, x_km, y_km, setup["keep"])
    prediction = data - residual
    noise_cd = model_utils.covariance_matrix(x_km, y_km, p["wgt"].astype(float), inputs["cov_p"], cfg.mc_noise_nugget_power)
    noise = model_utils.correlated_noise(noise_cd, cfg.mc_n_realisations, cfg.mc_seed)
    best_d = []
    for k in range(cfg.mc_n_realisations):
        noisy = scan(G, tents, prediction + noise[k], x_km, y_km, setup, section)
        best_d.append([noisy.D_km[noisy[metric].idxmin()] for metric in ("rms_all_mm", "rms_W500_mm", "rms_E500_mm")])
    best_d = np.array(best_d)
    noise_rows = []
    for i, (metric, name) in enumerate((("rms_all_mm", "All samples"), ("rms_W500_mm", "West 500 m"), ("rms_E500_mm", "East 500 m"))):
        q = np.percentile(best_d[:, i], [2.5, 16, 50, 84, 97.5])
        noise_rows.append(dict(metric=name, true_D_km=best.D_km, best_D_data_km=table.D_km[table[metric].idxmin()], noise_p2_5=q[0],
                               noise_p16=q[1], noise_median=q[2], noise_p84=q[3], noise_p97_5=q[4]))
    print(pd.DataFrame(noise_rows).round(3).to_string(index=False))
    common.write(noise_rows, "depth_scan_noise.csv", __file__)

    top_rows = []
    for top_km in BURIED_TOPS_KM:
        buried = scan(G, tents, data, x_km, y_km, setup, section, top_km)
        row = buried.loc[buried.rms_all_mm.idxmin()]
        top_rows.append(dict(top_km=top_km, best_D_km=row.D_km, slip_mm=row.slip_mm, rms_all_mm=row.rms_all_mm,
                             rms_W500_mm=row.rms_W500_mm, rms_E500_mm=row.rms_E500_mm))
    print(pd.DataFrame(top_rows).round(3).to_string(index=False))
    common.write(top_rows, "depth_scan_buried_top.csv", __file__)
    print(f"done ({elapsed()})")
