# %% [markdown]
# # 14 Paper figures
#
# ```
# Author: Danielle Lindsay
# Environment: opera_disp_feb13. Run time: seconds.
# ```
#
# **What this notebook does.** Each paper figure is drawn by the notebook that computes its content, so that the
# numbers on the figure and in the registry come from the same run. This notebook gathers the figures for the
# manuscript.
#
# 1. Find the current version of each paper figure in the figure folder, by its file pattern, and the notebook
#    that drew it (table below).
# 2. Check that no figure is missing, that each pattern matches a single file, and that no figure is stale
#    (section 1).
# 3. Copy each figure, as .png and .pdf, to the name used in the manuscript (`Figures_<date>/paper/Fig_<name>`).
#
# | Paper figure | Made by | File stem |
# |---|---|---|
# | Fig 1 | 04_event_detection | `fig1_tulepond_*` |
# | Fig 2 (fault-parallel and vertical maps and profiles, 3D model row) | 09_fig2, from the selected model of 08 | `fig2_fault_parallel_vertical_*_selected3d` |
# | Fig 3 (HRKT history, onsets over rainfall, repeat intervals, forecast) | 13_forecast (panels from 11 and 12) | `fig3_history_forecast_*` |
# | S: velocity chain, GNSS tie | 01_prep_figures | `figS1_velocity_chain_*`, `figS2_gnss_tie_*` |
# | S: rate mask, acquisitions | 03_creep_timeseries | `figS_ratemask_*`, `figS_acquisitions_*` |
# | S: event detection, creep rate along the fault | 04_event_detection | `figS_event_detection_*`, `figS_creep_rate_along_fault_*` |
# | S: LOS maps and profiles per track, with the model row | 09_fig2 | `figS_los_maps_profiles_*` |
# | S: per-event step maps | 06_event_field_geometry | `figS_event_step_maps_*` |
# | S: creepmeter validation, aperture ladder | 05_validation | `figS_validation_*`, `figS_binning_ladder_*` |
# | S: arrays and InSAR, threshold sweep, HRKT and HAPP histories | 11_alignment_arrays | `figS_tulepond_arrays_insar_*`, `figS_hrkt_threshold_sweep_*`, `qc09_aa_history_HRKT_*`, `qc09_aa_history_HAPP_*` |
# | S: rainfall conditioning | 12_rain_seasonality | `figS_precip_conditioning_*` |
# | S: renewal models, single fits | 13_forecast | `qc11_renewal_models_single_fits` |
# | S: slip model (L-curves, slip, fit maps, profiles, Monte Carlo, materials) | 09_model_3d | `figS_model_*`, `figS_pylith_materials_sfcvm_v26` |
# | S: sensitivity (sweeps, fixed-depth scan, Green's-function check, semivariogram) | 10_model_sensitivity | `figS_sweeps`, `figS_depth_scan`, `figS_gf_error_vs_depth`, `figS_semivariogram` |
#
# The dictionary `wanted` holds the same table and names the source notebook of each figure for the staleness
# check; `reads_from` lists the notebooks whose products each source notebook reads.
#
# **Inputs**: the figures in `Figures_<date>/` and the executed notebooks (`<notebook>.ipynb` in the repository
# root), whose cell execution records give the time of each notebook's latest run.
#
# **Outputs**: `Figures_<date>/paper/Fig_<name>.png` and `.pdf`. No registry keys.
#
# **Figures**: none drawn here.
#
# **What to check.** Each row prints `<- <source file>`, the source notebook and the start of its latest run. A
# figure written after the recorded run of its notebook (a run of the .py outside nbconvert) is accepted and
# flagged in the printout.

# %% [markdown]
# ## 1. Collect, check and copy the paper figures
#
# The latest run of a notebook starts at the earliest cell start and ends at the last cell reply recorded in the
# executed `<notebook>.ipynb` (written by `run_all.sh` through nbconvert); the cells must have run top to bottom.
# Let $t_0(n)$ and $t_1(n)$ be the start and end of the latest run of notebook $n$, and $t_w(f)$ the time a figure
# file $f$ was written. A figure $f$ drawn by notebook $n$ is stale if
#
# $$t_w(f) < t_0(n) \quad \text{or} \quad t_1(m) > t_0(n) \ \text{for any notebook } m \text{ that } n \text{ reads},$$
#
# that is, if the latest run of $n$ did not write it, or if $n$ last started before a notebook it reads had
# finished (for example when 08 has been rerun but 09 has not, or when 09 failed: nbconvert does not save a
# failed notebook, so its record is not updated). All times are in UTC. The cell stops if a figure is missing, if
# a pattern matches more than one file (so that an outdated variant is never selected), or if a figure is stale.

# %%
import glob
import json
import os
import shutil

import pandas as pd
from TulePond_filepaths import common_paths, repo_dir

fig_dir = common_paths["fig_dir"]
paper = os.path.join(fig_dir, "paper")
os.makedirs(paper, exist_ok=True)

# paper name -> (file pattern in fig_dir, notebook that draws it)
wanted = {
    "Fig_1": ("fig1_tulepond_*.png", "04_event_detection"),
    "Fig_2": ("fig2_fault_parallel_vertical_*ev_selected3d.png", "09_fig2"),         # the selected 3D (SFSVM) model of 08
    "Fig_3": ("fig3_history_forecast_*.png", "13_forecast"),
    "Fig_S_velocity_chain": ("figS1_velocity_chain_*.png", "01_prep_figures"),
    "Fig_S_gnss_tie": ("figS2_gnss_tie_*.png", "01_prep_figures"),
    "Fig_S_ratemask": ("figS_ratemask_*.png", "03_creep_timeseries"),
    "Fig_S_acquisitions": ("figS_acquisitions_*.png", "03_creep_timeseries"),
    "Fig_S_event_detection": ("figS_event_detection_*.png", "04_event_detection"),
    "Fig_S_creep_rate_along_fault": ("figS_creep_rate_along_fault_*.png", "04_event_detection"),
    "Fig_S_los_maps_profiles": ("figS_los_maps_profiles_*ev.png", "09_fig2"),
    "Fig_S_event_step_maps": ("figS_event_step_maps_*.png", "06_event_field_geometry"),
    "Fig_S_validation": ("figS_validation_*.png", "05_validation"),
    "Fig_S_binning_ladder": ("figS_binning_ladder_*.png", "05_validation"),
    "Fig_S_arrays_insar": ("figS_tulepond_arrays_insar_*.png", "11_alignment_arrays"),
    "Fig_S_hrkt_threshold": ("figS_hrkt_threshold_sweep_*.png", "11_alignment_arrays"),
    "Fig_S_aa_history_HRKT": ("qc09_aa_history_HRKT_*.png", "11_alignment_arrays"),
    "Fig_S_aa_history_HAPP": ("qc09_aa_history_HAPP_*.png", "11_alignment_arrays"),
    "Fig_S_precip_conditioning": ("figS_precip_conditioning_*.png", "12_rain_seasonality"),
    "Fig_S_renewal_single_fits": ("qc11_renewal_models_single_fits.png", "13_forecast"),
    "Fig_S_model_lcurve": ("figS_model_lcurve.png", "09_model_3d"),
    "Fig_S_model_slip": ("figS_model_slip.png", "09_model_3d"),
    "Fig_S_model_fit_maps": ("figS_model_fit_maps.png", "09_model_3d"),
    "Fig_S_model_profiles": ("figS_model_profiles.png", "09_model_3d"),
    "Fig_S_model_montecarlo": ("figS_model_montecarlo.png", "09_model_3d"),
    "Fig_S_pylith_materials": ("figS_pylith_materials_sfcvm_v26.png", "09_model_3d"),       # scripts/model_plots/plot_materials.py
    "Fig_S_sweeps": ("figS_sweeps.png", "10_model_sensitivity"),
    "Fig_S_depth_scan": ("figS_depth_scan.png", "10_model_sensitivity"),
    "Fig_S_gf_error_vs_depth": ("figS_gf_error_vs_depth.png", "10_model_sensitivity"),
    "Fig_S_semivariogram": ("figS_semivariogram.png", "10_model_sensitivity"),
}
# the notebooks whose products each source notebook reads (01_prep_figures and 03 read only the prepared data)
reads_from = {
    "01_prep_figures": [],
    "03_creep_timeseries": [],
    "04_event_detection": ["02_velocity_decomposition", "03_creep_timeseries"],
    "05_validation": ["02_velocity_decomposition", "03_creep_timeseries", "04_event_detection"],
    "06_event_field_geometry": ["04_event_detection"],
    "07_model_data": ["04_event_detection", "06_event_field_geometry"],
    "09_model_3d": ["04_event_detection", "06_event_field_geometry", "07_model_data"],
    "09_fig2": ["04_event_detection", "06_event_field_geometry", "07_model_data", "09_model_3d"],
    "10_model_sensitivity": ["07_model_data", "09_model_3d"],
    "11_alignment_arrays": ["03_creep_timeseries", "04_event_detection", "06_event_field_geometry"],
    "12_rain_seasonality": ["03_creep_timeseries", "04_event_detection", "11_alignment_arrays"],
    "13_forecast": ["03_creep_timeseries", "04_event_detection", "11_alignment_arrays", "12_rain_seasonality"],
}


def latest_run(notebook):
    """Start and end of a notebook's latest recorded run.

    nbconvert records the start and the reply of every code cell in the executed .ipynb. The run starts at the
    first cell's start and ends at the last cell's reply. Stops if a record is missing or if the cells did not
    run top to bottom.

    Parameters
    ----------
    notebook : str
        Notebook name without extension, e.g. "04_event_detection".

    Returns
    -------
    start, end : pandas.Timestamp
        Start and end of the run (UTC).
    """
    path = os.path.join(repo_dir, f"{notebook}.ipynb")
    with open(path) as f:
        cells = json.load(f)["cells"]
    starts = []
    ends = []
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        timing = cell.get("metadata", {}).get("execution", {})
        assert "iopub.execute_input" in timing and "shell.execute_reply" in timing, \
            f"{path}: a code cell has no execution record; run the notebook with run_all.sh"
        starts.append(pd.Timestamp(timing["iopub.execute_input"]))
        ends.append(pd.Timestamp(timing["shell.execute_reply"]))
    assert starts, f"{path} has no code cells"
    assert all(a <= b for a, b in zip(starts, starts[1:])), f"{path}: the cells did not run top to bottom; rerun it with run_all.sh"
    return starts[0], ends[-1]


def mtime_utc(path):
    """Modification time of a file.

    Parameters
    ----------
    path : str
        File path.

    Returns
    -------
    pandas.Timestamp
        Modification time (UTC).
    """
    return pd.Timestamp(os.path.getmtime(path), unit="s", tz="UTC")


runs = {}
for notebook in set(source for _, source in wanted.values()) | {n for upstream in reads_from.values() for n in upstream}:
    runs[notebook] = latest_run(notebook)

missing = []
stale = []
for name, (pattern, source_notebook) in wanted.items():
    hits = glob.glob(os.path.join(fig_dir, pattern))
    if len(hits) == 0:
        missing.append(f"{name} ({pattern})")
        continue
    assert len(hits) == 1, f"{name}: {len(hits)} files match {pattern}: {hits}"
    source = hits[0]
    run_start, run_end = runs[source_notebook]
    # check that the source notebook ran after every notebook it reads, and that the figure was written by that run
    for upstream in reads_from[source_notebook]:
        if runs[upstream][1] > run_start:
            stale.append(f"{name}: {source_notebook} last ran {run_start:%Y-%m-%d %H:%M}Z, before {upstream} finished "
                         f"{runs[upstream][1]:%Y-%m-%d %H:%M}Z; rerun {source_notebook}")
    note = ""
    for extension in (".png", ".pdf"):
        source_file = source.replace(".png", extension)
        if not os.path.exists(source_file):
            continue
        written = mtime_utc(source_file)
        if written < run_start:
            stale.append(f"{name}: {os.path.basename(source_file)} written {written:%Y-%m-%d %H:%M}Z, before {source_notebook}'s "
                         f"latest run started {run_start:%Y-%m-%d %H:%M}Z (that run did not write it)")
        elif written > run_end:
            note = f"  [written after the recorded run of {source_notebook}: a run outside nbconvert]"
    print(f"{name:26s} <- {os.path.basename(source)}  ({source_notebook}, run {run_start:%Y-%m-%d %H:%M}Z){note}")
assert not missing, f"paper figures missing in {fig_dir}: {missing}"
assert not stale, "Stale paper figures, not collected:\n  " + "\n  ".join(stale)
for name, (pattern, source_notebook) in wanted.items():
    source = glob.glob(os.path.join(fig_dir, pattern))[0]
    for extension in (".png", ".pdf"):
        source_file = source.replace(".png", extension)
        if os.path.exists(source_file):
            shutil.copy(source_file, os.path.join(paper, name + extension))
print("collected in", paper)
