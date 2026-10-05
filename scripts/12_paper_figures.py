# %% [markdown]
# # [1] 12 The paper figures, collected
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
# | Fig 1 | 04_event_detection | `fig1_tulepond_*` |
# | Fig 2 (fault-parallel and vertical maps and profiles, 3D model row) | 07c_fig2, from the 3D model of 08a | `fig2_fault_parallel_vertical_*_pylith_nuggetcd` |
# | Fig 3 (HRKT history, onsets over rainfall, repeat intervals, forecast) | 11_forecast (panels from 09, 10) | `fig3_history_forecast_*` |
# | S: velocity chain, GNSS tie | 01_prep_figures | `figS1_velocity_chain_*`, `figS2_gnss_tie_*` |
# | S: rate mask, acquisitions | 03_creep_timeseries | `figS_ratemask_*`, `figS_acquisitions_*` |
# | S: event detection, slip budget | 04_event_detection | `figS_event_detection_*`, `figS_slip_budget_*` |
# | S: LOS maps and profiles per track, with the model row | 07c_fig2 | `figS_los_maps_profiles_*` |
# | S: per-event step maps | 06_event_field_geometry | `figS_event_step_maps_*` |
# | S: creepmeter validation, aperture ladder | 05_validation | `figS_validation_*`, `figS_binning_ladder_*` |
# | S: arrays and InSAR, threshold sweep | 09_alignment_arrays | `figS_tulepond_arrays_insar_*`, `figS_hrkt_threshold_sweep_*` |
# | S: rainfall conditioning | 11_forecast | `figS_precip_conditioning_*` |
#
# Not collected here (supplement candidates, copied by hand if used): the PyLith model figures
# `figS_pylith_*` of 08a and 08b.
#
# **Outputs**: `Figures_<date>/paper/Fig_<name>.png` and `.pdf`. No registry keys.
#
# **What to check.** Every row prints `<- <source file>`. The cell stops if a figure is
# missing, or if a pattern matches more than one file (so a stale variant is never picked).

# %%
# [2]
import glob
import os
import shutil

from TulePond_filepaths import common_paths

fig_dir = common_paths["fig_dir"]
paper = os.path.join(fig_dir, "paper")
os.makedirs(paper, exist_ok=True)

# paper name -> the file pattern in fig_dir (one entry per line)
wanted = {
    "Fig_1": "fig1_tulepond_*.png",
    "Fig_2": "fig2_fault_parallel_vertical_*ev_pylith_nuggetcd.png",     # the 3D (SFSVM) model, PyLith Green's functions
    "Fig_3": "fig3_history_forecast_*.png",
    "Fig_S_velocity_chain": "figS1_velocity_chain_*.png",
    "Fig_S_gnss_tie": "figS2_gnss_tie_*.png",
    "Fig_S_ratemask": "figS_ratemask_*.png",
    "Fig_S_acquisitions": "figS_acquisitions_*.png",
    "Fig_S_event_detection": "figS_event_detection_*.png",
    "Fig_S_slip_budget": "figS_slip_budget_*.png",
    "Fig_S_los_maps_profiles": "figS_los_maps_profiles_*ev.png",
    "Fig_S_event_step_maps": "figS_event_step_maps_*.png",
    "Fig_S_validation": "figS_validation_*.png",
    "Fig_S_binning_ladder": "figS_binning_ladder_*.png",
    "Fig_S_arrays_insar": "figS_tulepond_arrays_insar_*.png",
    "Fig_S_hrkt_threshold": "figS_hrkt_threshold_sweep_*.png",
    "Fig_S_precip_conditioning": "figS_precip_conditioning_*.png",
}
missing = []
for name, pattern in wanted.items():
    hits = glob.glob(os.path.join(fig_dir, pattern))
    if len(hits) == 0:
        missing.append(f"{name} ({pattern})")
        continue
    assert len(hits) == 1, f"{name}: {len(hits)} files match {pattern}: {hits}"
    source = hits[0]
    for extension in (".png", ".pdf"):
        source_file = source.replace(".png", extension)
        if os.path.exists(source_file):
            shutil.copy(source_file, os.path.join(paper, name + extension))
    print(f"{name:26s} <- {os.path.basename(source)}")
assert not missing, f"paper figures missing in {fig_dir}: {missing}"
print("collected in", paper)
