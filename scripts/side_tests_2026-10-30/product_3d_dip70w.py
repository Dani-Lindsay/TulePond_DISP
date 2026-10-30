"""product_3d_dip70w.py: the 3D SFSVM model with a 70 W fault on the fine (shallow) mesh, re-solved at the corner
recorded in sensitivity_2026-10-30/sensitivity_rows.csv ("Fault dip, fine mesh", "70 W"), saved with
selected_model.npz's layout for the comparison figure (Dani 2026-10-30: "what does the dipping 3D one look like?").
Run: cd ~/tools/internal/TulePond_DISP && PYTHONPATH=. OMP_NUM_THREADS=3 nice -n 15 ~/miniforge3/envs/opera_disp_feb13/bin/python <this file>"""
import os, sys
import numpy as np
import pandas as pd
REPO = os.path.expanduser("~/tools/internal/TulePond_DISP")
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts", "sensitivity"))
import common
from common import cfg
import model_utils
from TulePond_filepaths import paths_out

OUT = os.path.dirname(os.path.abspath(__file__))
rows = pd.read_csv(os.path.join(common.OUT, "sensitivity_rows.csv"))
DIP = os.environ.get("DIP", "70 W")                     # "70 W" or "70 E"
row = rows[(rows.row == "Fault dip, fine mesh") & (rows.variant == DIP)].iloc[0]
inputs = common.load_inputs()
samples = common.samples_for(inputs, "default")
setup = common.make_setup(inputs, row.run)
G = setup["G"]["m"]
mu = common.mu_harmonic_of(row.run, setup)
fault_model, data_sel, _ = common.solve(inputs, setup, G, samples, float(row.sigma), label="3D 70 W fine")
numbers, slip = common.score(inputs, setup, fault_model, data_sel, mu_per_triangle=mu)
assert abs(numbers["rms_mm"] - row.rms_mm) < 1e-3, f"RMS {numbers['rms_mm']:.4f} vs recorded {row.rms_mm:.4f}"
selected = np.load(os.path.join(paths_out["08a_3d_model"], "selected_model.npz"))
assert np.array_equal(selected["keep"], setup["keep"])
model_mm = np.asarray(data_sel.synth, float).copy()
data_plane_removed_mm = np.asarray(data_sel.vel, float).copy()
tents = setup["tents"]
np.savez(os.path.join(OUT, f"dip{DIP.replace(' ', '').lower()}_3d_model.npz"), sample_lon=selected["sample_lon"], sample_lat=selected["sample_lat"],
         keep=setup["keep"], model_mm=model_mm, data_plane_removed_mm=data_plane_removed_mm,
         plane_mm=samples.vel - data_plane_removed_mm, slip_mm=slip, imp_km=tents.imp_km, imp_depth_km=tents.imp_depth_km,
         sub_faces=tents.sub_faces, imp_xyz=tents.imp_xyz, sigma=float(row.sigma), run=row.run,
         **{k: v for k, v in numbers.items()}, provenance=f"product_3d_dip70w.py: {row.run}, sigma {float(row.sigma):.3f}")
print({k: round(float(v), 3) for k, v in numbers.items() if np.isscalar(v)})
