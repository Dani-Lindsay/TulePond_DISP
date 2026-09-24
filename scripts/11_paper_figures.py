# %% [markdown]
# # 11 The paper figures, collected
#
# ```
# Created on Mon Sep 21 2026
# @author: daniellelindsay
# Run in opera_disp_feb13. Seconds.
# ```
#
# **What this notebook does.** Each paper figure is drawn by the notebook that
# computes its content, so the numbers on the figure and in the registry come
# from one place. This notebook only collects the current versions under the
# names the manuscript uses (`Figures_<date>/paper/Fig_N.png`) and lists which
# notebook made each, so the Overleaf figure folder can be refreshed in one
# step and every file is traceable.
#
# | Paper figure | Made by | File stem |
# |---|---|---|
# | Fig 1 | 04 | `fig1_tulepond_*` |
# | Fig 2 observations (LOS maps and profiles per track; model row as placeholders) | 06 | `fig2_observations_los_*` |
# | Fig 2 model row (descending: quadtree data, model, residual, slip with depth) | 07 (next round) | |
# | Fig 3 top | 08 | `fig3_aa_history_*` |
# | Fig 3 bottom | 09 | `fig3b_onset_seasonality_*` |
# | Fig 4 | 09 | `fig4_recurrence_forecast_*` |
# | S: processing chain, GNSS tie | Hayward_DISP-S1 figS1, figS2 (notebook 01 documents them) | |
# | S: rate mask, acquisitions | 03 | `figS_ratemask_*`, `figS_acquisitions_*` |
# | S: event detection, slip budget | 04 | `figS_event_detection_*`, `figS_slip_budget_*` |
# | S: fault-parallel maps and profiles, per-event step maps, symmetry check | 06 | `figS_fp_maps_*`, `figS_event_step_maps_*`, `figS_event_symmetry` |
# | S: creepmeter validation, aperture ladder | 05 | `figS_validation_*`, `figS_binning_ladder_*` |
# | S: arrays and InSAR, threshold sweep | 08 | `figS_tulepond_arrays_insar_*`, `figS_hrkt_threshold_sweep_*` |
# | S: rainfall conditioning | 09 | `figS_precip_conditioning_*` |

# %%
import glob
import os
import shutil

from TulePond_filepaths import common_paths

fig_dir = common_paths["fig_dir"]
paper = os.path.join(fig_dir, "paper")
os.makedirs(paper, exist_ok=True)
wanted = {"Fig_1": "fig1_tulepond_*.png", "Fig_2_observations": "fig2_observations_los_*ev.png",
          "Fig_3a": "fig3_aa_history_*.png", "Fig_3b": "fig3b_onset_seasonality_*.png",
          "Fig_4": "fig4_recurrence_forecast_*.png",
          "Fig_S_ratemask": "figS_ratemask_*.png", "Fig_S_acquisitions": "figS_acquisitions_*.png",
          "Fig_S_event_detection": "figS_event_detection_*.png", "Fig_S_slip_budget": "figS_slip_budget_*.png",
          "Fig_S_fp_maps": "figS_fp_maps_*.png",
          "Fig_S_event_step_maps": "figS_event_step_maps_*.png", "Fig_S_event_symmetry": "figS_event_symmetry*.png",
          "Fig_S_validation": "figS_validation_*.png", "Fig_S_binning_ladder": "figS_binning_ladder_*.png",
          "Fig_S_arrays_insar": "figS_tulepond_arrays_insar_*.png", "Fig_S_hrkt_threshold": "figS_hrkt_threshold_sweep_*.png",
          "Fig_S_precip_conditioning": "figS_precip_conditioning_*.png"}
for name, pattern in wanted.items():
    hits = sorted(glob.glob(os.path.join(fig_dir, pattern)), key=os.path.getmtime)
    if not hits:
        print(f"{name:26s} MISSING ({pattern})"); continue
    src = hits[-1]
    for ext in (".png", ".pdf"):
        s_ = src.replace(".png", ext)
        if os.path.exists(s_):
            shutil.copy(s_, os.path.join(paper, name + ext))
    print(f"{name:26s} <- {os.path.basename(src)}")
print("collected in", paper)
