"""alongstrike_dip_preference.py: where along the SSE section does the InSAR prefer a west or east dip? (Dani 2026-10-30:
CFW1 (km 62.6, borehole 70.5 SW at 21 m, Bilham & Whitehead 1997) and the Alameda Creek reflection line (km ~57.3,
70 E at 100-650 m, Williams et al. 2005) bound the SSE section.) For the 3D SFSVM fine-mesh models (vertical =
08a selected; 70 W / 70 E = product_3d_dip70w.py, each at its own corner), the RMS of the residual (each model's own
plane-removed data minus model) of the fitted samples within 500 m of the trace, west and east, in 1 km bins.
Writes alongstrike_dip_preference.csv. Run: PYTHONPATH=~/tools/internal/TulePond_DISP <env python> <this file>"""
import os
import numpy as np
import pandas as pd
from TulePond_filepaths import paths_out, data_dir

HERE = os.path.dirname(os.path.abspath(__file__))
models = {"vertical": np.load(os.path.join(paths_out["08a_3d_model"], "selected_model.npz")),
          "70 W": np.load(os.path.join(HERE, "dip70w_3d_model.npz")),
          "70 E": np.load(os.path.join(HERE, "dip70e_3d_model.npz"))}
ref = models["vertical"]
keep, d_km, km = ref["keep"], ref["signed_d_km"], ref["sample_km_pp"]
for m in models.values():
    assert np.array_equal(m["keep"], keep)
edges = [56.3, 57.3, 58.3, 59.3, 60.3, 61.3, 62.2]
rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    for side, sign in (("west", -1), ("east", 1)):
        sel = keep & (km >= lo) & (km < hi) & (sign * d_km > 0) & (sign * d_km <= 0.5)
        row = dict(km_lo=lo, km_hi=hi, side=side, n=int(sel.sum()))
        for name, m in models.items():
            res = np.asarray(m["data_plane_removed_mm"], float) - np.asarray(m["model_mm"], float)
            row[f"rms_{name.replace(' ', '')}_mm"] = float(np.sqrt(np.mean(res[sel] ** 2)))
        rows.append(row)
table = pd.DataFrame(rows)
table.to_csv(os.path.join(HERE, "alongstrike_dip_preference.csv"), index=False)
print(table.round(2).to_string(index=False))
