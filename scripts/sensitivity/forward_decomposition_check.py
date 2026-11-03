"""forward_decomposition_check.py: does the decomposition itself make the near-fault asymmetry? A forward test with the
preferred slip model.

Author: Danielle Lindsay

The preferred 3D model (notebook 09: SFSVM, 70 W fault, right-lateral slip only) predicts a full 3D surface
displacement, with its own vertical and fault-normal parts. This script takes that 3D displacement at the inversion
samples, projects it into the ascending and descending lines of sight (each sample's LOS vector from 06's incidence
and azimuth), decomposes the two LOS values with no fault-normal motion (FP, U), as the data are decomposed (Methods
M2, M7), and compares the near-fault deflections (the M7 definition, model_config.near/far_deflection_km) of the
apparent fault-parallel field with those of the model's true fault-parallel field. If they agree, the geometry and
the decomposition do not create the asymmetry from the model's own vertical and fault-normal motion. The model's
fault-normal and vertical deflections are reported too.

Gates (asserted): the 3D displacements projected on the fault-parallel look vector reproduce the run's G (07/09), and
that G times the preferred slip reproduces 09's fault-parallel prediction.

Inputs: 07 model_data.npz (samples, points file), 09 selected_model.npz (slip, prediction, sample distances), the
PyLith output of model_config.gf_run, 06 tulepond_steps_<track>.h5 and fp_maps.npz (LOS geometry, pixel positions).
Output (model_config.sensitivity_dir): forward_decomposition_check.csv. Notebook 10 draws and registers it.
Run from the repository root (about a minute, reading the PyLith output):
    PYTHONPATH=. python scripts/sensitivity/forward_decomposition_check.py
"""
import os

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree

import common
from common import cfg, model_utils, utils
from TulePond_filepaths import insar_dic_list, paths_out

SCRIPT = os.path.abspath(__file__)

# ---- 1. the preferred model and its G
model_data = np.load(os.path.join(paths_out["07_model_data"], "model_data.npz"))
selected = np.load(os.path.join(paths_out["09_model_3d"], "selected_model.npz"))
strike_deg = float(model_data["strike_deg"])
points_npz = str(model_data["points_default"])
gnpz = model_utils.load_G(cfg.gf_run, points_npz, strike_deg, cfg.gf_pylith_output_dir, cfg.gf_gf_dir)
slip_mm = np.asarray(selected["slip_mm"], float)

# ---- 2. the run's 3D displacements at the samples, in the impulse order and right-lateral sign of G (as build_G)
fault_xyz, _, slip, point_xyz, disp = model_utils.read_gf_run(cfg.gf_pylith_output_dir, cfg.gf_run)
assert np.array_equal(fault_xyz, gnpz["fault_xyz"]), "the fault mesh differs from the G product"
left_lateral = slip[:, :, 1]
k_vertex = np.argmax(np.abs(left_lateral), axis=1)
assert np.array_equal(k_vertex, gnpz["vertex_index"]), "the impulse order differs from the G product"
peak_sign = np.sign(left_lateral[np.arange(left_lateral.shape[0]), k_vertex])
points = np.load(points_npz)
dist, order = cKDTree(point_xyz[:, :2]).query(np.c_[points["x"], points["y"]])
assert dist.max() < 1e-3 and np.unique(order).size == order.size, "the samples do not match the output points"
scale = float(gnpz["sign_rl"]) * peak_sign
G_enu = disp[:, order, :] * scale[:, None, None]                    # (impulses, samples, 3): east, north, up (mesh CRS)
G_fp = np.einsum("kpc,c->pk", G_enu, np.asarray(gnpz["look"], float))
assert np.allclose(G_fp, gnpz["G_rl_m_per_m"], rtol=0, atol=1e-12), "the 3D displacements do not reproduce the fault-parallel G"
assert G_fp.shape[1] == slip_mm.size, "the slip is not on the impulses of G"
assert np.allclose(G_fp @ slip_mm, selected["model_mm"], rtol=0, atol=1e-6), "G x slip does not reproduce 09's prediction"
model_enu = np.einsum("kpc,k->pc", G_enu, slip_mm)                 # mm, (samples, 3)

# ---- 3. each sample's LOS vectors (nearest pixel of 06's box) and the decomposition with no fault-normal motion
field_dir = paths_out["06_event_field_geometry"]
fp = np.load(os.path.join(field_dir, "fp_maps.npz"))
steps = {d["ORBIT_DIRECTION"]: os.path.join(field_dir, f"tulepond_steps_{d['ORBIT_DIRECTION']}.h5") for d in insar_dic_list}
geometry = model_utils.los_from_steps(steps)
to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
sx, sy = to_utm.transform(np.asarray(model_data["sample_lon"]), np.asarray(model_data["sample_lat"]))
has_geometry = np.isfinite(geometry["ascending"]["inc"]) & np.isfinite(geometry["descending"]["inc"])
_, pixel = cKDTree(np.c_[fp["X"][has_geometry], fp["Y"][has_geometry]]).query(np.c_[sx, sy])
los = {}
for track in ("ascending", "descending"):
    inc = geometry[track]["inc"][has_geometry][pixel]
    azi = geometry[track]["azi"][has_geometry][pixel]
    los[track] = utils.los_unit_vector(inc, azi).T                    # (samples, 3)
s = np.deg2rad(strike_deg)
fp_dir = np.array([np.sin(s), np.cos(s), 0.0])
fn_dir = np.array([np.sin(s - np.pi / 2), np.cos(s - np.pi / 2), 0.0])   # the east side away from the fault (NE)
up_dir = np.array([0.0, 0.0, 1.0])
d_asc = (model_enu * los["ascending"]).sum(axis=1)
d_desc = (model_enu * los["descending"]).sum(axis=1)
apparent_fp = np.empty(d_asc.size)
apparent_up = np.empty(d_asc.size)
for i in range(d_asc.size):
    G2 = np.array([[los["ascending"][i] @ fp_dir, los["ascending"][i] @ up_dir],
                   [los["descending"][i] @ fp_dir, los["descending"][i] @ up_dir]])
    apparent_fp[i], apparent_up[i] = np.linalg.solve(G2, [d_asc[i], d_desc[i]])

# ---- 4. the M7 deflections of each field
section = cfg.sse_section_km()
in_section = (selected["sample_km_pp"] >= section[0]) & (selected["sample_km_pp"] <= section[1])
keep = np.asarray(selected["keep"], bool)
fields = {"model fault-parallel (true)": model_enu @ fp_dir, "apparent fault-parallel (decomposed, FN = 0)": apparent_fp,
          "model fault-normal": model_enu @ fn_dir, "model vertical": model_enu @ up_dir, "apparent vertical": apparent_up}
assert np.allclose(fields["model fault-parallel (true)"], selected["model_mm"], atol=1e-6)
stamp = common.stamp(SCRIPT)
rows = []
for name, values in fields.items():
    west, n_near_w, n_far_w = model_utils.side_deflection_mm(values, selected["signed_d_km"], in_section, keep, -1,
                                                             cfg.near_deflection_km, cfg.far_deflection_km)
    east, n_near_e, n_far_e = model_utils.side_deflection_mm(values, selected["signed_d_km"], in_section, keep, 1,
                                                             cfg.near_deflection_km, cfg.far_deflection_km)
    rows.append(dict(field=name, west_mm=west, east_mm=east, ratio_we=abs(west / east) if east else np.nan,
                     n_near_west=n_near_w, n_near_east=n_near_e, run=cfg.gf_run, provenance=stamp))
table = pd.DataFrame(rows)
os.makedirs(common.OUT, exist_ok=True)
table.to_csv(os.path.join(common.OUT, "forward_decomposition_check.csv"), index=False, float_format="%.4f")
print(f"run {cfg.gf_run}; strike {strike_deg:.2f}; {keep.sum()} fitted samples, {in_section.sum()} in the SSE section")
print(table.drop(columns=["provenance", "run"]).round(3).to_string(index=False))
