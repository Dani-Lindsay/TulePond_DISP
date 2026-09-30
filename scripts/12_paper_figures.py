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
# | Fig 2 (fault-parallel and vertical maps and profiles, 3D model row) | 07c_fig2 with the PyLith product | `fig2_fault_parallel_vertical_*_pylith_nuggetcd` |
# | Fig 3 (array history, repeat intervals, onsets over rainfall) | 09 | `fig3_history_rain_*` |
# | Fig 4 | 09 | `fig4_recurrence_forecast_*` |
# | S: velocity chain, GNSS tie | 01_prep_figures | `figS1_velocity_chain_*`, `figS2_gnss_tie_*` |
# | S: rate mask, acquisitions | 03 | `figS_ratemask_*`, `figS_acquisitions_*` |
# | S: event detection, slip budget | 04 | `figS_event_detection_*`, `figS_slip_budget_*` |
# | S: LOS maps and profiles per track, with the model row | 07c_fig2 | `figS_los_maps_profiles_*` |
# | S: per-event step maps | 06 | `figS_event_step_maps_*` |
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
# the earlier Fig 3 panels (08 fig3_aa_history, 09 fig3b) are working figures, not paper figures (Dani 2026-09-29)
for stale in glob.glob(os.path.join(paper, "Fig_3[ab].*")):
    os.remove(stale)
wanted = {"Fig_1": "fig1_tulepond_*.png", "Fig_2": "fig2_fault_parallel_vertical_*ev_pylith_nuggetcd.png",    # the 3D (SFCVM) model, PyLith GFs (2026-09-30) "Fig_S_los_maps_profiles": "figS_los_maps_profiles_*ev.png",
          "Fig_3": "fig3_history_rain_*.png",
          "Fig_4": "fig4_recurrence_forecast_*.png",
          "Fig_S_velocity_chain": "figS1_velocity_chain_*.png", "Fig_S_gnss_tie": "figS2_gnss_tie_*.png",
          "Fig_S_ratemask": "figS_ratemask_*.png", "Fig_S_acquisitions": "figS_acquisitions_*.png",
          "Fig_S_event_detection": "figS_event_detection_*.png", "Fig_S_slip_budget": "figS_slip_budget_*.png",
          "Fig_S_event_step_maps": "figS_event_step_maps_*.png",
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
