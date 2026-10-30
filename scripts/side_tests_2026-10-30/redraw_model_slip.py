"""redraw_model_slip.py: redraw figS_model_slip from the saved 08a / 07a products without re-running 08a (its Monte
Carlo takes ~40 min). The plotting lines are a copy of 08a [figures] as edited 2026-10-30 (commit after 9ed55c5): dashed
section ends instead of the bar that hid the top ~45 m, labels with structure, dip, Mw, RMS and depth extent.
The next full 08a run draws the same figure. Run:
cd ~/tools/internal/TulePond_DISP && PYTHONPATH=. ~/miniforge3/envs/opera_disp_feb13/bin/python <this file>"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from cmcrameri import cm as cmc
import insar_utils as utils
import model_config as cfg
from TulePond_filepaths import paths_out, common_paths

utils.mpl_paper_style(base=7)
cm = 1 / 2.54
model = np.load(os.path.join(paths_out["08a_3d_model"], "retired_2026-10-30", "selected_model.npz"))   # VERTICAL (retired 2026-10-30 late)
homogeneous = np.load(os.path.join(paths_out["07a_uniform_model"], "retired_2026-10-30", "homogeneous_model.npz"))
assert np.allclose(model["imp_xyz"], homogeneous["imp_xyz"]), "the two models are not on the same fault vertices"
creep_lo_km, creep_hi_km = cfg.sse_section_km()
numbers = {k: float(model[k]) for k in ("Mw_30gpa", "Mw_sfcvm", "rms_mm", "depth90_m")}
slip = model["slip_mm"]

triangulation = mtri.Triangulation(model["imp_km"], model["imp_depth_km"], model["sub_faces"])
fig, axes = plt.subplots(2, 1, figsize=(18.3 * cm, 9.0 * cm), sharex=True, constrained_layout=True)
vmax = float(max(slip.max(), homogeneous["slip_mm"].max()))
for ax, letter, values, title in ((axes[0], "A", slip,
                                   f"3D SFSVM, vertical fault, Mw {numbers['Mw_30gpa']:.2f} (30 GPa) and {numbers['Mw_sfcvm']:.2f} "
                                   f"(SFSVM rigidity), RMS {numbers['rms_mm']:.2f} mm, depth extent {numbers['depth90_m'] / 1e3:.2f} km"),
                                  (axes[1], "B", homogeneous["slip_mm"],
                                   f"Homogeneous half-space, vertical fault, Mw {float(homogeneous['Mw_30gpa']):.2f} (30 GPa), "
                                   f"RMS {float(homogeneous['rms_mm']):.2f} mm, depth extent {float(homogeneous['depth90_m']) / 1e3:.2f} km")):
    image = ax.tripcolor(triangulation, values, shading="gouraud", cmap=cmc.lajolla, vmin=0, vmax=vmax)
    for section_end_km in (creep_lo_km, creep_hi_km):
        ax.axvline(section_end_km, color="0.85", ls="--", lw=0.6)
    ax.set_ylim(1.5, 0)
    ax.set_xlim(54.5, 64.5)
    ax.set_ylabel("Depth (km)")
    utils.panel_title(ax, letter, title)
axes[1].set_xlabel("Distance along the fault from Point Pinole (km)")
fig.colorbar(image, ax=axes, label="Right-lateral slip (mm)", shrink=0.8)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(common_paths["fig_dir"], f"figS_model_slip.{ext}"), dpi=300, facecolor="white")
print("saved", os.path.join(common_paths["fig_dir"], "figS_model_slip.png"))
