# TulePond_DISP
# Predictable Surface Slow Slip Events on the Hayward Fault

This repository contains the notebooks that reproduce every figure and every
number in *Predictable Surface Slow Slip Events on the Hayward Fault*
(Lindsay and Bürgmann, in preparation for Science) from the OPERA DISP-S1
Sentinel-1 displacement products.

## Layout

The layout follows the NC_ALOS-2 repository: two importable modules and a
numbered chain of notebooks run in order.

| File | Role |
|---|---|
| `TulePond_filepaths.py` | WHERE: every path and shared constant. Edit `data_dir` (one line) to point at the data. |
| `insar_utils.py` | HOW: short shared helpers (MintPy readers, LOS projections, along-fault axis, statistics, creepmeter readers at daily and raw 10-min sampling, the figure style helpers `mpl_paper_style` / `text_backed`, the number registry). Each has a test. |
| `data_prep/` | HOW THE INPUTS WERE MADE: the download and preparation chain (`run00`-`run05`, `run20`) that turns the public archives into the products the notebooks read -- OPERA DISP-S1, CSLC static layers and DEM, OPERA TROPO, MintPy conversion and mosaics, the GNSS tie, and the GHCN rain gauges. Run once per data refresh, by hand, not by `run_all.sh`; see `data_prep/README.md`. |
| `NN_*.ipynb` + `scripts/NN_*.py` | WHAT, in what order. Notebooks are jupytext-paired (`jupytext.toml`: `ipynb,scripts//py:percent`): the `.py` (percent format, in `scripts/`) is the readable source, the `.ipynb` carries the executed outputs. |
| `tests/` | pytest: projection conventions, paths, the registry, and `test_anchors.py` that pins every number quoted in the paper. |
| `outputs/` | products, `*_stats.txt` sidecars and the number registry (git-ignored; regenerated). |
| `Figures_<date>/` | figures, one dated directory per round of figure work (git-ignored). |
| `environment.yml` | the packages the notebooks need, pinned to the versions the paper numbers were made with; `environment_full_2026-09-25.yml` is the exact export of the working env (`opera_disp_feb13`). |
| `assumptions_register.md` | every analysis choice a reviewer could ask about, where it is set, and what justifies it. |

## The number registry

Every number that appears in the manuscript is registered by the notebook
that computes it into `outputs/registry.csv` (append-only) and rendered to
`outputs/registry.txt`, grouped by notebook and by manuscript location. The
manuscript is filled from `registry.txt` by hand, so each number can be traced
to a notebook, a cell, the input files (fingerprinted) and the git hash.
Re-running a notebook that changes a registered value raises
`RegistryConflict` unless `TULEPOND_SUPERSEDE=1` is set.

## Run order

| Notebook | Does | Figures | Sidecars / registry keys |
|---|---|---|---|
| `01_prep_inputs` | Documents the OPERA DISP-S1 preparation chain (download, rebase, tropo, MintPy conversion, GNSS tie -- the scripts are in `data_prep/`) and fingerprints its products. Not re-run here. Rebuilds the 100 m crossings of the mapped fault trace (`hayward_fault_profiles.csv`) from the KML and checks them against the `Inputs/` copy. | S1, S2 | `01_*` |
| `02_velocity_decomposition` | Ascending + descending velocity mosaics -> fault-parallel and vertical maps (MintPy `asc_desc2horz_vert.py`), creepmeter transects | Fig 1a, Fig 2 map | `02_*` |
| `03_creep_timeseries` | Cross-fault LOS time series at every 100 m crossing, near-fault rate mask, common date grid, fault-parallel / vertical decomposition, along-fault rate profile | S3, S4 | `03_*` |
| `04_event_detection` | Event detector on the Tule Pond stack (COZ1 as control), joint model, per-crossing event slip, section lengths, band stack, slip budget with the alignment arrays, recurrence intervals, Bo Xie's relocated seismicity | Fig 1, S detection, S budget | `04_*` |
| `05_validation` | Agreement with the five USGS creepmeters (section brackets from 04); the aperture ladder from the frozen sweep and the coherence comparison | S validation, S ladder | `05_*` |
| `06_event_field_geometry` | Per-pixel step maps for each event on both tracks, burst fix, mean event, fault-parallel / vertical maps, cross-fault profiles X-X' and Y-Y' on each track, symmetric / antisymmetric check, the ascending-deficit end members | Fig 2 rows 1-2 (LOS per track), S | `06_*` |
| `07a_uniform_model` | The SELECTED CSI slip inversion of the five-event mean, both tracks, InSAR only: smooth-growth mesh (50 m x1.2) on the hand-drawn trace, dip 75 E (seismicity-motivated; InSAR-indifferent), quadtree to one 30 m pixel, ascending covariance inflated by the MEASURED off-fault vertical (fp_maps), sigma by the rms-plateau rule from the suite; bootstrap Mw error, held-out surface validation against the crossing stacks | Fig 2 row 3 (panels I, J, K), S fit maps The **3D model of 08a supersedes this uniform model in the paper**; 07a stays as the homogeneous baseline it is measured against. | `07a_hom_*` |
| `07b_uniform_model_tests` | Every modelling choice for the mean-event slip inversion, tested: mesh suite (4 parameterizations, zero-difference port gate vs `csi_io.load_fault_tri`), dip 60-90, smoothing sigma, sampling, ascending covariance inflation, free rake, parametric depth scan (both covariance treatments), synthetic depth recovery; machinery parity CHECK against the legacy m02 chain. Writes `suite_results.csv`; runs BEFORE `07a_uniform_model`. | S sensitivity (collated) | `outputs/07b_uniform_model_tests/` (no registry keys) |
| `07c_fig2` | Draws Fig 2 from 06's maps and profiles (`fig2_inputs.npz`) and the model row: by default 08a's `fig2_model_row_pylith_nuggetcd.npz` (the 3D model, with the homogeneous PyLith model of the same inversion as the comparison line), 07a's `fig2_model_row.npz` before 08a has run; `TULEPOND_FIG2_MODEL` / `_COMPARE` / `_TAG` override. Computes no numbers. | Fig 2 (`fig2_fault_parallel_vertical_*_pylith_nuggetcd`), S LOS version | none |
| `08a_3d_model` | The SELECTED slip model: 07a's data, samples, noise model, surface constraint and CSI machinery (cells [1]-[6] exec'd unchanged) inverted on 3D finite-element Green's functions (PyLith 5.0.2, `GreensFns` + `FaultCohesiveImpulses`) for the vertical fault under the trace in the USGS SFCVM v26.0 aligned on the trace; nugget-only diagonal Cd (companion: 07a's full correlated Cd), both media (homogeneous 30 GPa and SFCVM) in each; material CHECK, validation against CSI Meade and the 2D screw dislocation, forward-run superposition and mesh convergence, L-curves, the four models' numbers, the Fig-2 products. Reads the PyLith products from `modeling_SFCVM/` (never re-runs PyLith unless `TULEPOND_RUN_PYLITH=1`); the parameter files are in `scripts/pylith_cfg/`, the helpers in `scripts/pylith_helpers/`. | Fig 2 row 3 (via 07c), `figS_pylith_*` (materials, validation, L-curves, fit maps, slip, X-X'), `qc08a_*` | `08a_*` (selected), `08a_hom_*`, `08a_full_*`, `08a_full_hom_*` |
| `08b_3d_model_tests` | Uncertainty and resolution of the 3D model (exec's 08a's cells [1]-[10]): bootstrap and sigma sensitivity, synthetic depth recovery, fixed-depth scan for the selected model, its companion and their homogeneous twins (each CHECKED against the draft run's cached tables); then the sensitivity runs (mesh `shallow`, Cd floors, equal weights, no surface constraint, swath 7 km, dip 80/70 E, order-2 elements, SFCVM boundary +/- 50 m, coherence 0.7) read from `modeling_SFCVM/gf/` and NOT recomputed: one-page summary per run, the all-tests figure, the table. | `figS_pylith_recovery`, `figS_pylith_depth_scan`, `figS_pylith_all_tests`, `qc08b_bootstrap`, `qc08b_summary_<run>_<medium>` | `08b_*`, `08b_hom_*`, `08b_full_*`, `08b_full_hom_*`, `08b_sens_*` |
| `09_alignment_arrays` | The six Tule Pond arrays against the InSAR, the 45-year HRKT event detector, repeat intervals, threshold sweep | Fig 3 top, S arrays, S sweep | `09_*` |
| `10_rain_seasonality` | Daily rain composite at Tule Pond from the three nearest GHCN gauges (COOP shift checked on the 11 Dec 2014 storm), water-year totals, onsets on the water year | Fig 3 bottom | `10_*` |
| `11_forecast` | The next-event forecast: renewal models on four catalogues (issue date pinned), the closed-form InSAR clock, rainfall conditioning (rules A, B, C), the prediction statement | Fig 3 (with panels from 09, 10), S conditioning | `11_*` |
| `12_paper_figures` | Collects the current paper figures under their manuscript names in `Figures_<date>/paper/` | Figs 1-4, S | |

## Data layout

`TulePond_filepaths.data_dir` (`TULEPOND_DATA_DIR` if set, else `proj_TulePond_DISP`, else the repo itself) holds
ONLY what the notebooks read, in the layout the Zenodo deposit will carry (the comment block at the top of
`TulePond_filepaths.py`):

| Directory | Contents |
|---|---|
| `09156/ 09157/ 11115/` | the DISP-S1 frames (data, stack, tropo, mintpy) from `data_prep/run01-04` |
| `mosaic_velfield/` | the tied same-orbit velocity + geometry mosaics (run04 step 6, run05) |
| `external_inputs/` | fault trace, arrays, creepmeters, GHCN, catalogues, qfaults, USGS SFCVM v26 |
| `GPS/` | the screened UNR series and the run05 tie table |
| `prep_validation/` | run05's GNSS-tie record and run20's rain tables |
| `aperture_sweep/` | the frozen aperture sweep notebook 04 loads (Hayward_DISP-S1/analysis/run11) |
| `modeling_uniform/` | products of 07a/07b that are data-sized (Green's functions, sweeps) |
| `modeling_SFCVM/` | PyLith: `cfg/` (copies of `scripts/pylith_cfg/`), `mesh/`, `spatialdb/`, `points/`, `gf/` (G matrices, the draft run's inversion products = 08b's sensitivity table), `pylith_output/` (the Green's-function and forward runs read by 08a/08b) |

Everything else of the older `proj_HaywardCreep` working tree stays there, untouched. The PyLith runs
(33 min each, 8 ranks, PyLith 5.0.2) are not part of `run_all.sh`: 08a reads their products and stops with the
command that makes a missing one; `TULEPOND_RUN_PYLITH=1` lets it shell out to PyLith
(`scripts/pylith_helpers/pylith_env.sh`, `run_gf.sh`).

## Running

Everything runs in the conda environment `opera_disp_feb13` (MintPy, opera-utils,
pygmt, CSI, jupyter). Call the environment's binaries by full path in scripts;
`conda activate` is unreliable in non-interactive shells on this machine.

```bash
PATH=/Users/dlindsay/miniforge3/envs/opera_disp_feb13/bin:$PATH
python -m pytest -q          # conventions, paths, registry, and the anchors (every quoted number vs the registry)
./run_all.sh                 # every notebook, restart-and-run-all, in order
./run_all.sh 03 05           # a subset
jupyter lab                  # to work interactively
```

Every number in the manuscript comes from a `run_all.sh` execution, never from
an interactively edited kernel.

## Data

- One creep stream everywhere: 30 to 150 m from the trace, coherence 0.7 (chosen 2026-09-21 over 0.9 for coverage; the frozen sweep shows the threshold is second order).
- OPERA DISP-S1 v1.0 products, frames 09157 (ascending, track 35) and 11115
  (descending, track 42), plus 09156 (ascending, northern half) which is merged
  with 09157 into the ascending mosaic. Prepared by the scripts in `data_prep/`
  and documented in notebook 01, so the repository is self-contained from the
  public archives through to the figures. The granules themselves are
  open-access from ASF and are not redistributed; the prepared products and the
  inputs the figures are built from will be deposited on Zenodo [DOI TBD].
- USGS creepmeters (Langbein et al., 2024), USGS Bay Area alignment arrays
  v3.0, NOAA GHCN-daily, the Johnson et al. (2022) creep-rate compilation,
  T. Taira's repeating-earthquake catalogue (GEN3) and the Waldhauser and
  Schaff double-difference catalogue, all under `data_dir/Inputs`.

MintPy: editable install of the OPERA fork at commit 054c6010 (`mintpy.__version__`
1.6.2, pip 1.6.3.post10). CSI at commit 1277bb4, okada4py 1.0.1.
