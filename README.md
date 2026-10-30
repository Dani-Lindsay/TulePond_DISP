# TulePond_DISP
# Predictable Slow Slip Events on the Hayward Fault

This repository contains the code that reproduces every figure and every number in *Predictable Slow Slip Events on
the Hayward Fault* (Lindsay and Bürgmann, in preparation for Science) from the OPERA DISP-S1 Sentinel-1 displacement
products.

## Overview

The analysis runs as a numbered chain of Jupyter notebooks. Each notebook reads only the products of the notebooks
before it, never their code, and writes its own products, figures and a stats file to the results directory.

- **Preparing the InSAR inputs** from the public archives (`data_prep/`, run once by hand)
- **Velocity decomposition and creep time series** at 100 m crossings of the mapped trace (02, 03)
- **Slow slip event detection** and the along-fault extent of the Tule Pond section (04)
- **Validation** against the USGS creepmeters (05)
- **The mean event field** in fault-parallel and vertical components (06)
- **Slip models**: homogeneous half-space and 3D elastic structure (USGS SFSVM v26.0) with PyLith 5.0.2 Green's
  functions, the fault dipping 70 deg west (07-10; `pylith/`)
- **Event history** from the alignment arrays (11), **rainfall** (12) and the **forecast** (13)
- **Paper figures**: Figs 1-3 drawn, and every figure collected under its manuscript name (14)

## Repository layout

| File | Role |
|---|---|
| `TulePond_filepaths.py` | Every path and shared constant. Set `data_dir` (or the environment variable `TULEPOND_DATA_DIR`) to the results directory. |
| `insar_utils.py` | Shared helpers: MintPy readers, line-of-sight projections, the along-fault axis, statistics, creepmeter and alignment-array readers, figure style, the number registry. |
| `model_config.py` | The settings of the slip models, each with its unit and source. |
| `model_utils.py` | The slip-model functions: data sets and quadtree samples, noise model, CSI inversions on the PyLith fault, Green's functions from the PyLith output, L-curve, rigidity, moment and stress drop. |
| `NN_*.ipynb`, `scripts/NN_*.py` | The notebooks, paired with jupytext (`jupytext.toml`): the `.py` in `scripts/` is the readable source, the `.ipynb` holds the executed outputs. |
| `scripts/sensitivity/` | The sensitivity tests of the slip model (inversion choices, fault dip, elastic structure, depth scan), run by hand; 10 draws their results. |
| `scripts/model_plots/` | The SFSVM materials figure (called by 08). |
| `pylith/` | PyLith parameter files, run wrappers and the mesh and material build scripts; `pylith/README.md` lists every run. |
| `data_prep/` | The download and preparation chain from the public archives; `data_prep/README.md`. |
| `tests/` | pytest: projection conventions, paths, the slip-model functions, the registry, and `test_anchors.py`, which pins the numbers quoted in the paper. |
| `run_all.sh` | Runs the notebook chain in order, then the tests. |
| `environment.yml` | The packages and versions used for the paper; `environment_full_2026-09-25.yml` is the full export of the working environment. |

## Notebooks

`run_all.sh` runs them in name order, except that `01_prep_figures` runs right after `01_prep_inputs`.

| Notebook | Does | Paper figures |
|---|---|---|
| `01_prep_inputs` | Fingerprints the prepared inputs (`data_prep/`); rebuilds the 100 m crossings of the mapped trace from the KML. | |
| `01_prep_figures` | Velocity processing chain and the GNSS tie of the frames. | Figs S1, S2 |
| `02_velocity_decomposition` | Ascending and descending velocity mosaics decomposed into fault-parallel and vertical fields (MintPy `asc_desc2horz_vert`); creepmeter transects. | Fig 1 (map data) |
| `03_creep_timeseries` | Cross-fault time series at every 100 m crossing (30-150 m from the trace), common date grid, fault-parallel decomposition, along-fault rate profile and the flag of anomalous crossings. | Rate mask, acquisitions |
| `04_event_detection` | Event detector on the Tule Pond stack (COZ1 as control), joint model of each crossing, per-crossing event slip, section extent and 90 % band, slip budget, seismicity near the section; the Fig 1 inputs. | Event detection, creep rate along the fault |
| `05_validation` | Comparison with the five USGS creepmeters; the aperture ladder. | Validation, aperture ladder |
| `06_event_field_geometry` | Per-pixel step maps of each event on both tracks, burst-seam correction, mean event, fault-parallel and vertical maps, cross-fault profiles X-X' and Y-Y'. | Step maps |
| `07_model_data` | The data of the slip models: quadtree samples (the PyLith output points), the noise model, the coherence-0.7 sample set. | |
| `08_model_homogeneous` | The homogeneous slip model on the same 70 deg W fault (analytic half-space Green's functions). | |
| `09_model_3d` | The preferred slip model (3D SFSVM, PyLith Green's functions, fault dipping 70 deg west): fit, moment, depth extent, Monte Carlo of correlated noise, stress drop. | Model figures, SFSVM materials |
| `10_model_sensitivity` | Draws the sensitivity tests of `scripts/sensitivity/`: sweeps, depth scan, semivariogram. | Sensitivity figures |
| `11_alignment_arrays` | The Tule Pond alignment arrays against the InSAR; the 45-year HRKT event history and its threshold sweep. | Fig 3 (arrays), array history |
| `12_rain_seasonality` | Daily rain composite at Tule Pond (GHCN-daily), onsets by water year, rainfall conditioning of the onsets. | Fig 3 (rain), rainfall conditioning |
| `13_forecast` | Repeat-interval statistics, renewal models (Brownian passage time and alternatives) and the next-event forecast; the Fig 3 inputs. | Renewal fits |
| `14_paper_figures` | Draws Figs 1, 2 and 3 from the products of 04, 06-09 and 11-13, then collects every paper and supplement figure under its manuscript name in `Figures_<date>/paper/`. | Figs 1-3, LOS maps and profiles |

## The number registry

Every number quoted in the paper is registered by the notebook that computes it. `outputs/paper_numbers.md` lists the
current value of every number, one table per notebook, with the cell that computes it. `outputs/registry.csv` is the
append-only log behind it (full value, input-file fingerprints, git hash). A re-run that changes a registered value
stops with `RegistryConflict` unless `TULEPOND_SUPERSEDE=1` is set.

## Data

The results directory (`data_dir`) holds only what the notebooks read, in the layout of the Zenodo deposit [DOI TBD]:

| Directory | Contents |
|---|---|
| `09156/ 09157/ 11115/` | The DISP-S1 frames (data, stack, troposphere, MintPy) from `data_prep/run01`-`run04` |
| `mosaic_velfield/` | The tied same-orbit velocity and geometry mosaics |
| `external_inputs/` | Fault trace, alignment arrays, creepmeters, GHCN-daily, catalogues, the Johnson et al. (2022) creep-rate compilation, USGS SFSVM v26.0 |
| `GPS/` | The screened UNR GNSS series and the tie table |
| `prep_validation/` | The GNSS-tie record and the rain tables of the preparation chain |
| `aperture_sweep/` | The precomputed aperture sweep read by 05 |
| `modeling_inputs/` | PyLith meshes, the PyLith output points written by 07, impulse databases |
| `modeling_uniform/` | Homogeneous-medium Green's functions (analytic, cached by 07) |
| `modeling_SFCVM/` | SFSVM material grids and databases, PyLith runs, G matrices and run logs |
| `sensitivity/` | The outputs of `scripts/sensitivity/` |
| `outputs/`, `Figures_<date>/`, `executed/` | Notebook products, figures, and an HTML copy of every executed notebook |

Data sources: OPERA DISP-S1 v1.0 frames 09156 and 09157 (ascending, merged) and 11115 (descending); USGS creepmeters
(Langbein et al., 2024); USGS Bay Area alignment arrays v3.0; NOAA GHCN-daily; the Johnson et al. (2022) creep-rate
compilation; the Taira repeating-earthquake catalogue (GEN3) and the Waldhauser and Schaff double-difference catalogue;
the USGS San Francisco Bay region 3D seismic velocity model (SFSVM) v26.0. The DISP-S1 granules are open access from
ASF and are not redistributed.

## Running

```bash
conda env create -f environment.yml          # once
python -m pytest -q                          # conventions, paths, functions, registry and the quoted numbers
./run_all.sh                                 # every notebook, in order
./run_all.sh 07 08 09 10                 # a subset (each reads only its upstream products)
```

The PyLith runs are not part of `run_all.sh`; 08 and `scripts/sensitivity/` read their stored products
(`pylith/README.md` gives the command of every run). Every number in the paper comes from a `run_all.sh` execution.

Software: MintPy (OPERA fork, commit 054c6010; version 1.6.2), CSI (commit 1277bb4), PyLith 5.0.2.

## Authorship and use of AI tools

Code by Danielle Lindsay. The code was written with the assistance of Claude (Anthropic; Claude Opus 5, Claude Opus 5.5,
Claude Fable 5 and Claude Fable 5.1), used through Claude Code in 2026. Under the author's direction, Claude wrote and
revised code for the data preparation, the creep time series, the event detection, the slip-model inversions, the
sensitivity tests, the statistics and the forecast, and the figures. The authors designed the analysis, chose the
methods and parameters, reviewed the code, checked its outputs (including the tests in `tests/`), and take
responsibility for the results. Claude also suggested wording edits to the manuscript; the authors wrote and verified
all text. This use is declared in the Methods and Acknowledgements of the paper.
