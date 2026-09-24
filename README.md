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
| `01_prep_inputs` | Documents the OPERA DISP-S1 preparation chain (download, rebase, tropo, MintPy conversion, GNSS tie) and fingerprints its products. Not re-run here. Rebuilds the 100 m crossings of the mapped fault trace (`hayward_fault_profiles.csv`) from the KML and checks them against the `Inputs/` copy. | S1, S2 | `01_*` |
| `02_velocity_decomposition` | Ascending + descending velocity mosaics -> fault-parallel and vertical maps (MintPy `asc_desc2horz_vert.py`), creepmeter transects | Fig 1a, Fig 2 map | `02_*` |
| `03_creep_timeseries` | Cross-fault LOS time series at every 100 m crossing, near-fault rate mask, common date grid, fault-parallel / vertical decomposition, along-fault rate profile | S3, S4 | `03_*` |
| `04_event_detection` | Event detector on the Tule Pond stack (COZ1 as control), joint model, per-crossing event slip, section lengths, band stack, slip budget with the alignment arrays, recurrence intervals, Bo Xie's relocated seismicity | Fig 1, S detection, S budget | `04_*` |
| `05_validation` | Agreement with the five USGS creepmeters (section brackets from 04); the aperture ladder from the frozen sweep and the coherence comparison | S validation, S ladder | `05_*` |
| `06_event_field_geometry` | Per-pixel step maps for each event on both tracks, burst fix, mean event, fault-parallel / vertical maps, cross-fault profiles X-X' and Y-Y' on each track, symmetric / antisymmetric check, the ascending-deficit end members | Fig 2 rows 1-2 (LOS per track), S | `06_*` |
| `07_slip_model` | (next round) CSI slip inversion of the mean event, descending track only | Fig 2 row 3 | `07_*` |
| `08_alignment_arrays` | The six Tule Pond arrays against the InSAR, the 45-year HRKT event detector, repeat intervals, threshold sweep | Fig 3 top, S arrays, S sweep | `08_*` |
| `09_rain_seasonality` | Daily rain composite at Tule Pond from the three nearest GHCN gauges (COOP shift checked on the 11 Dec 2014 storm), water-year totals, onsets on the water year | Fig 3 bottom | `09_*` |
| `10_forecast` | The next-event forecast: renewal models on four catalogues (issue date pinned), the closed-form InSAR clock, rainfall conditioning (rules A, B, C), the prediction statement | Fig 4, S conditioning | `10_*` |
| `11_paper_figures` | Collects the current paper figures under their manuscript names in `Figures_<date>/paper/` | Figs 1-4, S | |

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
  (descending, track 42), prepared as described in notebook 01; the prepared
  MintPy products will be deposited on Zenodo [DOI TBD].
- USGS creepmeters (Langbein et al., 2024), USGS Bay Area alignment arrays
  v3.0, NOAA GHCN-daily, the Johnson et al. (2022) creep-rate compilation,
  T. Taira's repeating-earthquake catalogue (GEN3) and the Waldhauser and
  Schaff double-difference catalogue, all under `data_dir/Inputs`.

MintPy: editable install of the OPERA fork at commit 054c6010 (`mintpy.__version__`
1.6.2, pip 1.6.3.post10). CSI at commit 1277bb4, okada4py 1.0.1.
