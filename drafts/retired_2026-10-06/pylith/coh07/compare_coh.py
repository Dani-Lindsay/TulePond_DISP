#!/usr/bin/env python3
"""
Coherence 0.9 vs 0.7 for the selected setup (and its companions): per run and medium, the model's surface slip at
km 60.6-61.2 (top-edge impulse vertices) against the crossing stacks there (07's crossings, fig2_model_row.npz),
the samples east / west of the trace within 300 m in km 60.2-60.8 (Lake Elizabeth), rms all / W / E, Mw30.
Run in opera_disp_feb13 from the repo root:  python drafts/pylith/coh07/compare_coh.py  -> coh07/compare_coh.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PYL = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(PYL, "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, PYL)
SPIKE_KM, GAP_KM = (60.6, 61.2), (60.2, 60.8)
row07 = np.load(os.path.join(REPO, "outputs", "07_slip_model_tests", "fig2_model_row.npz"))
ck, cm = row07["crossing_km"], row07["crossing_mm"]
in_spike = (ck >= SPIKE_KM[0]) & (ck <= SPIKE_KM[1]) & np.isfinite(cm)
pk, pm = row07["crossing_profile_km"], row07["crossing_profile_mm"]
prof_spike = (pk >= SPIKE_KM[0] - 1e-9) & (pk <= SPIKE_KM[1] + 1e-9)
below, above = pk <= SPIKE_KM[0], pk >= SPIKE_KM[1]                 # the running-median profile points bracketing the stretch
i_lo = int(np.nonzero(below)[0][np.argmax(pk[below])]); i_hi = int(np.nonzero(above)[0][np.argmin(pk[above])])
cross_txt = (f"crossings km {SPIKE_KM[0]}-{SPIKE_KM[1]}: {in_spike.sum()} crossings, median {np.median(cm[in_spike]):.1f} mm "
             f"(range {cm[in_spike].min():.1f}-{cm[in_spike].max():.1f}); running-median profile (the surface constraint): "
             f"{prof_spike.sum()} points inside; nearest km {pk[i_lo]:.1f} ({pm[i_lo]:.1f} mm) and km {pk[i_hi]:.1f} ({pm[i_hi]:.1f} mm), "
             f"interpolated across the gap")

rows = []
for tag, label in (("", "coh 0.9, full Cd"), ("_nuggetcd", "coh 0.9, nugget Cd (SELECTED)"),
                   ("_coh07", "coh 0.7, full Cd"), ("_coh07_nuggetcd", "coh 0.7, nugget Cd")):
    f = os.path.join(PYL, "gf", f"inversion_products{tag}.npz")
    if not os.path.exists(f):
        print(f"(missing {os.path.basename(f)}: not run yet)")
        continue
    z = np.load(f, allow_pickle=True)
    res = pd.read_csv(os.path.join(PYL, "gf", f"results_comparison{tag}.csv")).set_index("model")
    xyz, km = z["impulse_vertex_xyz"], z["impulse_km"]
    top = np.abs(xyz[:, 2]) < 1e-6
    spike = top & (km >= SPIKE_KM[0]) & (km <= SPIKE_KM[1])
    for medium in [k[5:-3] for k in z.files if k.startswith("slip_") and k.endswith("_mm")]:
        slip = z[f"slip_{medium}_mm"]
        r = res.loc[f"PyLith {medium}"]
        rows.append(dict(run="default" + tag, label=label, medium=medium, sigma=r.sigma,
                         surface_slip_spike_mean_mm=float(slip[spike].mean()), surface_slip_spike_max_mm=float(slip[spike].max()),
                         n_top_vertices=int(spike.sum()), crossings_spike_median_mm=float(np.median(cm[in_spike])),
                         rms_mm=r.rms_mm, rms_W300_mm=r.rms_west300_mm, rms_E300_mm=r.rms_east300_mm, chi2=r.chi2,
                         surface_pct=r.surface_pct, Mw_30gpa=r.Mw_30gpa, Mw_sfcvm=r.Mw_sfcvm))
table = pd.DataFrame(rows)

# apples to apples: every model scored on the SAME 2386 coh-0.9 samples (07's data, the default products' data_mm),
# prediction = G at those samples (gf/G_gf_default_<medium>.npz, same fault vertices asserted) x the model's slip,
# with one constant removed (the samples' mean residual); W / E within 300 m as everywhere else
ref = np.load(os.path.join(PYL, "gf", "inversion_products_nuggetcd.npz"), allow_pickle=True)   # made 2026-09-30: has signed_d_km
assert np.array_equal(ref["sample_lon"], row07["sample_lon"])
d09, sd09 = row07["data_mm"], ref["signed_d_km"]             # 07's data at the 2386 samples (any per-run constant cancels below)
w09, e09 = (sd09 < 0) & (sd09 >= -0.3), (sd09 > 0) & (sd09 <= 0.3)
for c in ("rms_on09_mm", "rms_W300_on09_mm", "rms_E300_on09_mm"):
    table[c] = np.nan
for i, r in table.iterrows():
    z = np.load(os.path.join(PYL, "gf", f"inversion_products{r.run[len('default'):]}.npz"), allow_pickle=True)
    g = np.load(os.path.join(PYL, "gf", f"G_gf_default_{r.medium}.npz"))
    assert np.allclose(g["fault_xyz"][g["vertex_index"]], z["impulse_vertex_xyz"]), "different fault vertices"
    res = d09 - g["G_rl_m_per_m"].dot(z[f"slip_{r.medium}_mm"])
    res = res - res.mean()
    table.loc[i, ["rms_on09_mm", "rms_W300_on09_mm", "rms_E300_on09_mm"]] = [
        float(np.sqrt(np.mean(res ** 2))), float(np.sqrt(np.mean(res[w09] ** 2))), float(np.sqrt(np.mean(res[e09] ** 2)))]
table.to_csv(os.path.join(HERE, "compare_coh.csv"), index=False)
print(cross_txt)
print(table.drop(columns=["label"]).round(2).to_string(index=False))
