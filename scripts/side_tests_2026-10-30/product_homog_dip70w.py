"""product_homog_dip70w.py: the homogeneous (analytic) model with a 70 W fault, default-mesh dip sweep of
scripts/sensitivity/dip_rows.py, re-solved at its recorded corner and saved with selected_model.npz's layout for the
comparison figure (Dani 2026-10-30: bottom two rows at 70 W). Run as product_3d_dip70w.py."""
import os, sys
import numpy as np
import pandas as pd
REPO = os.path.expanduser("~/tools/internal/TulePond_DISP")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts", "sensitivity"))
import common
import dip_rows
from TulePond_filepaths import paths_out

OUT = os.path.dirname(os.path.abspath(__file__))
recorded = pd.read_csv(os.path.join(common.OUT, "dip_rows.csv"))
row = recorded[(recorded.geometry == "dip70w") & (recorded.family == "Homogeneous analytic N^0")].iloc[0]
inputs = common.load_inputs()
samples = common.samples_for(inputs, "default")
run = dip_rows.PYLITH_RUNS["dip70w"]
setup = common.make_setup(inputs, run)
G = np.load(os.path.join(dip_rows.DIP_CHECKS, f"G_analytic_{run}_nsub4.npy"))
fault_model, data_sel, _ = common.solve(inputs, setup, G, samples, float(row.sigma), label="homogeneous 70 W")
numbers, slip = common.score(inputs, setup, fault_model, data_sel)
assert abs(numbers["rms_mm"] - row.rms_mm) < 1e-3, f"RMS {numbers['rms_mm']:.4f} vs recorded {row.rms_mm:.4f}"
selected = np.load(os.path.join(paths_out["08a_3d_model"], "selected_model.npz"))
assert np.array_equal(selected["keep"], setup["keep"])
tents = setup["tents"]
data_plane_removed_mm = np.asarray(data_sel.vel, float).copy()
np.savez(os.path.join(OUT, "homog70w_model.npz"), sample_lon=selected["sample_lon"], sample_lat=selected["sample_lat"],
         keep=setup["keep"], model_mm=np.asarray(data_sel.synth, float).copy(), data_plane_removed_mm=data_plane_removed_mm,
         plane_mm=samples.vel - data_plane_removed_mm, slip_mm=slip, imp_km=tents.imp_km, imp_depth_km=tents.imp_depth_km,
         sub_faces=tents.sub_faces, imp_xyz=tents.imp_xyz, sigma=float(row.sigma), run=run,
         **{k: v for k, v in numbers.items()}, provenance=f"product_homog_dip70w.py: analytic on {run} tents, n_sub 4, sigma {float(row.sigma):.3f}")
print({k: round(float(v), 3) for k, v in numbers.items() if np.isscalar(v)})
