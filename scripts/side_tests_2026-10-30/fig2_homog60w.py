"""fig2_homog60w.py: Fig 2 with the best-fitting dipping HOMOGENEOUS model (60 W) as the model (Dani 2026-10-30:
"give me Fig 2 with the best homogeneous model with that deflection"). Not a paper figure: a comparison.

Step 1: re-solve the default-mesh homogeneous 60 W model of scripts/sensitivity/dip_rows.py at its recorded sigma
(dip_rows.csv; RMS reproduced, asserted) and save a product with selected_model.npz's layout (as 08a [6]/[12]).
Step 2: run a COPY of scripts/07c_fig2.py with only these changes: the model = this product, the comparison line on G
= the selected 3D model, the labels, the Mw text (no SFSVM rigidity for a homogeneous model), and the output folder
(this folder). The paper's Fig 2 in Figures_2026-10-28 is not touched.
Run: cd ~/tools/internal/TulePond_DISP && PYTHONPATH=. OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 nice -n 15 \
     ~/miniforge3/envs/opera_disp_feb13/bin/python ~/work/proj_TulePond_DISP/dip_deflection_2026-10-30/fig2_homog60w.py
"""
import os
import sys
import subprocess

import numpy as np
import pandas as pd

REPO = os.path.expanduser("~/tools/internal/TulePond_DISP")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "sensitivity"))
import common                                   # noqa: E402
from common import cfg                          # noqa: E402
import dip_rows                                 # noqa: E402
import model_utils                              # noqa: E402
from TulePond_filepaths import paths_out        # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
GEOMETRY, FAMILY = "dip60w", "Homogeneous analytic N^0"

# ---- step 1: the product ------------------------------------------------------------------------------------------
recorded = pd.read_csv(os.path.join(common.OUT, "dip_rows.csv"))
row = recorded[(recorded.geometry == GEOMETRY) & (recorded.family == FAMILY)].iloc[0]
inputs = common.load_inputs()
samples = common.samples_for(inputs, "default")
run = dip_rows.ANALYTIC_ONLY[GEOMETRY]
gf_dir = os.path.join(dip_rows.DIP_CHECKS, "gf_analytic")
setup = common.make_setup(inputs, run, run_dir=(gf_dir, gf_dir))
G = setup["G"]["m"]
fault_model, data_sel, _ = common.solve(inputs, setup, G, samples, float(row.sigma), label=f"{GEOMETRY} {FAMILY}")
numbers, slip = common.score(inputs, setup, fault_model, data_sel)
assert abs(numbers["rms_mm"] - row.rms_mm) < 1e-3, f"RMS {numbers['rms_mm']:.4f} vs recorded {row.rms_mm:.4f}"

selected = np.load(os.path.join(paths_out["08a_3d_model"], "selected_model.npz"))
model_data = np.load(os.path.join(paths_out["07a_uniform_model"], "model_data.npz"))
assert np.array_equal(selected["keep"], setup["keep"]), "the 60 W model fits different samples from the selected model"
model_mm = np.asarray(data_sel.synth, float).copy()
data_plane_removed_mm = np.asarray(data_sel.vel, float).copy()
plane_mm = samples.vel - data_plane_removed_mm
residual_mm = data_plane_removed_mm - model_mm
design = np.c_[np.ones(samples.vel.size), samples.x, samples.y]
plane_coef = np.linalg.lstsq(design, plane_mm, rcond=None)[0]
assert np.allclose(design @ plane_coef, plane_mm, atol=1e-6), "the removed reference frame is not a plane"
assert np.allclose(G @ slip, model_mm, atol=1e-6), "CSI's prediction is not G x slip"
signed_d_km, sample_km_pp = setup["signed_d_km"], setup["sample_km_pp"]
assert np.allclose(signed_d_km, selected["signed_d_km"]) and np.allclose(sample_km_pp, selected["sample_km_pp"])
crossing_km = np.asarray(selected["crossing_km"], float)
data_step = model_utils.crossing_steps_like(data_plane_removed_mm, signed_d_km, sample_km_pp, crossing_km, cfg.fig2k_inner_km, cfg.fig2k_outer_km)
model_step = model_utils.crossing_steps_like(model_mm, signed_d_km, sample_km_pp, crossing_km, cfg.fig2k_inner_km, cfg.fig2k_outer_km)
tents = setup["tents"]
product = os.path.join(OUT, "homog60w_model.npz")
np.savez(product, sample_lon=model_data["sample_lon"], sample_lat=model_data["sample_lat"], block_lon=model_data["block_lon"],
         block_lat=model_data["block_lat"], data_mm=samples.vel.copy(), plane_mm=plane_mm, model_mm=model_mm,
         data_plane_removed_mm=data_plane_removed_mm, residual_mm=residual_mm, keep=setup["keep"], signed_d_km=signed_d_km,
         sample_km_pp=sample_km_pp, crossing_km=crossing_km, data_step_mm=data_step, model_step_mm=model_step,
         step_inner_km=cfg.fig2k_inner_km, step_outer_km=cfg.fig2k_outer_km, slip_mm=slip, imp_km=tents.imp_km,
         imp_depth_km=tents.imp_depth_km, sub_faces=tents.sub_faces, imp_xyz=tents.imp_xyz,
         **{k: v for k, v in numbers.items()}, sigma=float(row.sigma),
         plane_offset_mm=float(plane_coef[0]), plane_dx_mm_per_km=float(plane_coef[1]), plane_dy_mm_per_km=float(plane_coef[2]),
         run=run, provenance=f"fig2_homog60w.py: {run}, {FAMILY}, sigma {float(row.sigma):.3f}")
print({k: round(float(v), 3) for k, v in numbers.items() if np.isscalar(v)})

# ---- step 2: a copy of 07c with the swaps ------------------------------------------------------------------------
src = open(os.path.join(REPO, "scripts", "07c_fig2.py")).read()
swaps = [
    ('selected_file = os.path.join(paths_out["08a_3d_model"], "selected_model.npz")', f'selected_file = {product!r}'),
    ('homogeneous_file = os.path.join(paths_out["07a_uniform_model"], "homogeneous_model.npz")',
     'homogeneous_file = os.path.join(paths_out["08a_3d_model"], "selected_model.npz")'),
    ('model_tag = "_selected3d"', 'model_tag = "_homog60w_COMPARISON"'),
    ('(profile_dots(compare), homog_col, "Homogeneous")', '(profile_dots(compare), hetero_col, "3D (SFSVM) vertical")'),
    ('(profile_dots(model), hetero_col, "3D (SFSVM)")', '(profile_dots(model), homog_col, "Homogeneous 60@.W")'),
]
for old, new in swaps:
    assert src.count(old) == 1, f"07c changed: {old!r} found {src.count(old)} times"
    src = src.replace(old, new)
# the Mw text: 30 GPa only (a homogeneous model has no SFSVM rigidity)
mw_old = src[src.index('fig.text(position="TR", text=f"M@-w@-'):]
mw_old = mw_old[:mw_old.index(")\n") + 1]
src = src.replace(mw_old, 'fig.text(position="TR", text=f"M@-w@- {float(product[\'Mw_30gpa\']):.2f} (30 GPa), homogeneous 60\\260W", '
                          'justify="BR", offset="0c/0.12c", font=annot_font, no_clip=True)')
# K legend label (the model is the homogeneous one) and the output folder
assert src.count('0.35c 3D (SFSVM)') == 1
src = src.replace('0.35c 3D (SFSVM)', '0.35c Homogeneous 60@.W')
assert src.count('model_col = hetero_col') == 1
src = src.replace('model_col = hetero_col', 'model_col = homog_col')   # K line in the homogeneous colour, as on G
assert src.count('fig_dir = common_paths["fig_dir"]') == 1
src = src.replace('fig_dir = common_paths["fig_dir"]', f'fig_dir = {OUT!r}')
copy = os.path.join(OUT, "_07c_fig2_homog60w_copy.py")
open(copy, "w").write(src)
subprocess.run([sys.executable, copy], cwd=REPO, check=True, env={**os.environ, "PYTHONPATH": REPO})
print("wrote Fig 2 comparison in", OUT)
