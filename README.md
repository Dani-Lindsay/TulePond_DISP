# TulePond_DISP
# Periodic slow slip events on the Hayward Fault

This repository contains the code that reproduces every figure and every number in *Periodic slow slip events on the Hayward Fault*
(Lindsay and Bürgmann, in preparation) from the OPERA DISP-S1 Sentinel-1 displacement
products.

**Archive.** The input data, model files and results of the paper's run are in the Zenodo record
[doi:10.5281/zenodo.23850500](https://doi.org/10.5281/zenodo.23850500); this code is archived separately at the release
of that run, v1.0.0, [doi:10.5281/zenodo.23934127](https://doi.org/10.5281/zenodo.23934127) (all versions:
[doi:10.5281/zenodo.23934126](https://doi.org/10.5281/zenodo.23934126)). Please cite the paper, the data record and the code (`CITATION.cff`). The code is under the MIT licence (`LICENSE`); the data in the Zenodo record are CC BY 4.0.

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
| `TulePond_filepaths.py` | Every path and shared constant. Set the environment variable `TULEPOND_DATA_DIR` to the results directory. |
| `insar_utils.py` | Shared helpers: MintPy readers, line-of-sight projections, the along-fault axis, statistics, creepmeter and alignment-array readers, figure style, the number registry. |
| `model_config.py` | The settings of the slip models, each with its unit and source. |
| `model_utils.py` | The slip-model functions: data sets and quadtree samples, noise model, CSI inversions on the PyLith fault, Green's functions from the PyLith output, L-curve, rigidity, moment and stress drop. |
| `creep_profiles.py` | The cross-fault series at each trace crossing, their interpolation to the common dates and the fault-parallel decomposition (03; `data_prep/run07`). |
| `renewal.py` | The renewal statistics of the forecast (13): posterior draws of the Brownian passage time, lognormal and Weibull models, the posterior predictive conditioned on the open interval, the Poisson baseline and the calibration test. |
| `NN_*.ipynb` | The notebooks, the analysis code with its executed outputs (`run_all.sh` runs them in order). |
| `scripts/sensitivity/` | The sensitivity tests of the slip model (inversion choices, fault dip, elastic structure, depth scan), run by hand; 10 draws their results. |
| `scripts/model_plots/` | The SFSVM materials figure (called by 09). |
| `scripts/zenodo_package.py`, `scripts/read_audit/` | The files of the Zenodo deposit, built from a clean results directory; the list of files the notebooks read. |
| `pylith/` | PyLith parameter files, run wrappers and the mesh and material build scripts; `pylith/README.md` lists every run. |
| `data_prep/` | The download and preparation chain from the public archives; `data_prep/README.md`. |
| `original_2025_code/` | The first author's pre-repository script of February 2025 (verbatim), its surviving output and figure, and the SCEC 2025 poster reference; run and compared by notebook 15. |
| `tests/` | pytest: projection conventions, paths, the creep-profile extraction, the slip-model functions, the renewal statistics, the registry, and `test_anchors.py`, which pins the headline numbers quoted in the paper. |
| `run_all.sh` | Runs the notebook chain in order, then the tests. |
| `environment.yml` | The packages and versions used for the paper (CSI is installed from its repository; see the file's header). |

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
| `10_model_sensitivity` | Draws the sensitivity tests of `scripts/sensitivity/`: sweeps, depth scan, semivariogram, the fault-normal leak test and the forward check of the two-look decomposition. | Sensitivity figures |
| `11_alignment_arrays` | The Tule Pond alignment arrays against the InSAR; the 45-year HRKT event history and its threshold sweep. | Fig 3 (arrays), array history |
| `12_rain_seasonality` | Daily rain composite at Tule Pond (GHCN-daily), onsets by water year, rainfall conditioning of the onsets. | Fig 3 (rain), rainfall conditioning |
| `13_forecast` | Repeat-interval statistics, renewal models (Brownian passage time, lognormal, Weibull) and the next-event forecast, a Bayesian posterior predictive conditioned on the open interval (`renewal.py`), with Poisson and seasonal baselines and a calibration test on synthetic records; the Fig 3 inputs. | Renewal fits |
| `14_paper_figures` | Draws Figs 1, 2 and 3 from the products of 04, 06-09 and 11-13, the six-panel Hovmöller set of the cross-fault series along the whole fault, then collects every paper and supplement figure under its manuscript name in `Figures_<date>/paper/`. | Figs 1-3, LOS maps and profiles, six-panel Hovmöller |
| `15_check_original_2025_code` | Independent check: the first author's February 2025 script run verbatim (paths and frame crop only) on the repository's descending product, compared with the 03 series at Tule Pond; the surviving 2025 result compared too. | Check of the original 2025 code |

## The number registry

Every number quoted in the paper is registered by the notebook that computes it. `outputs/paper_numbers.md` lists the
current value of every number, one table per notebook, with the cell that computes it. `outputs/registry.csv` is the
append-only log behind it (full value, input-file fingerprints, git hash). A re-run that changes a registered value
stops with `RegistryConflict` unless `TULEPOND_SUPERSEDE=1` is set.

## Data

The results directory (`TULEPOND_DATA_DIR`) holds what the notebooks read and write:

| Directory | Contents |
|---|---|
| `09156/ 09157/ 11115/` | The DISP-S1 frames (data, stack, troposphere, MintPy) from `data_prep/run01`-`run04` |
| `mosaic_velfield/` | The tied same-orbit velocity and geometry mosaics |
| `external_inputs/` | Fault trace, alignment arrays, creepmeters, GHCN-daily, catalogues, the Johnson et al. (2022) creep-rate compilation, USGS SFSVM v26.0 |
| `GPS/` | The screened UNR GNSS series and the tie table |
| `prep_validation/` | The GNSS-tie record and the rain tables of the preparation chain |
| `aperture_sweep/` | The precomputed aperture sweep read by 05 |
| `modeling_inputs/` | PyLith meshes, the PyLith output points written by 07, impulse databases |
| `modeling_uniform/` | Homogeneous-medium Green's functions (analytic, cached by 08) |
| `modeling_SFCVM/` | SFSVM material grids and databases, PyLith runs, G matrices and run logs |
| `sensitivity/` | The outputs of `scripts/sensitivity/` |
| `outputs/`, `Figures_<date>/`, `executed/` | Notebook products, figures, and an HTML copy of every executed notebook |

Data sources: OPERA DISP-S1 v1.0 frames 09156 and 09157 (ascending, merged) and 11115 (descending); USGS creepmeters
(Langbein et al., 2024); USGS Bay Area alignment arrays v3.0; NOAA GHCN-daily; the Johnson et al. (2022) creep-rate
compilation; the Taira repeating-earthquake catalogue (GEN3) and the Waldhauser and Schaff double-difference catalogue;
the USGS San Francisco Bay region 3D seismic velocity model (SFSVM) v26.0 (file and folder names call it SFCVM); the
Gittins et al. (2026) creep-event catalogue; the UNR / Nevada Geodetic Laboratory GNSS series; the PB2002 plate
boundaries (Bird, 2003). The DISP-S1 granules are open access from ASF and are not redistributed.

**Reproducing from the Zenodo record** ([doi:10.5281/zenodo.23850500](https://doi.org/10.5281/zenodo.23850500), built
by `scripts/zenodo_package.py`). The record holds the three rebased DISP-S1 stacks, the external inputs, the GNSS tables
and tied velocity mosaics, the aperture sweep, the PyLith inputs and outputs with their G matrices, the sensitivity
tables, and every product and figure of the paper's run. The MintPy time series, masks and geometry are not deposited:
`data_prep/run02`-`run04` rebuild them from the stacks and public OPERA products (`data_prep/README.md`). Two
catalogues are not redistributed and are available from their authors: the relocated seismicity catalogue of B. Rong
(Rong et al. 2026) and the Taira GEN3 repeating-earthquake catalogue (notebooks 01 and 04 read them). The executed
notebooks, with all their outputs, are in this repository at the release commit.

## Running

```bash
conda env create -f environment.yml          # once
export TULEPOND_DATA_DIR=/path/to/results       # the results directory (run_all.sh requires it)
python -m pytest -q                          # conventions, paths, functions, statistics, registry and the quoted numbers
./run_all.sh                                 # every notebook, in order
./run_all.sh 07 08 09 10                     # a subset (each reads only its upstream products)
```

The PyLith runs and the sensitivity scripts are not part of `run_all.sh`; 08, 09, 10 and `scripts/sensitivity/` read
their stored products. To rebuild the slip models from scratch, run 07, then the PyLith runs (`pylith/README.md`), then
`scripts/sensitivity/` (`dip_rows_analytic_G.py` before `dip_rows.py`), then 08, 09 and 10: 09 checks its fit against the
sensitivity table. The numbers of the paper come from `run_all.sh` executions from clean commits: each registry row
records the commit and whether the tree was clean when run_all.sh started.

Software: MintPy 1.6.3 (the modules the notebooks use are identical in 1.6.2 and 1.6.3), CSI (commit 1277bb4), PyLith 5.0.2.

## Times

All times are UTC and are expressed as decimal years with one convention, $y = 2000 + (t - t_{2000}) / 365.25$ d,
where $t_{2000}$ is 1 January 2000, 00:00 UTC, so that a difference of decimal years times 365.25 is an exact number of
days. Every notebook converts through the functions of `insar_utils` (`decimal_years`, `date_to_decimal_year`,
`decimal_year_to_date`). The exact time of each InSAR acquisition (the zero-Doppler start time of its OPERA DISP-S1
granule, the `time` coordinate of the rebased stack `<frame>_stack.nc`) is read once, in notebook 03
(`acquisition_times.csv`), and the onset windows of notebook 04 run between these
times. The fault-parallel and vertical decomposition pairs the ascending and descending acquisitions of the same UTC
day (about 12 h apart) as one epoch; no onset window is shorter than 5 d. Creepmeter and earthquake times are UTC;
alignment-array surveys and rainfall are whole local days. External files written with calendar decimal years are
converted on reading (`insar_utils.from_calendar_decimal_years`).

## Authorship and use of AI tools

Code by Danielle Lindsay. The original analysis was written in February 2025 with occasional syntax help from ChatGPT
(OpenAI) and presented at the 2025 SCEC Annual Meeting (Lindsay, Taira and Bürgmann, 2025, SCEC Contribution 14951,
https://central.scec.org/publication/14951; the code is in `original_2025_code/`). The project was restarted in July
2026 and the current code was written with the assistance of Claude (Anthropic; Claude Opus 5, Claude Opus 5.5, Claude
Fable 5 and Claude Fable 5.1), used through Claude Code. Under the author's direction, Claude wrote and revised code for
the data preparation, the creep time series, the event detection, the slip-model inversions, the sensitivity tests, the
statistics and the forecast, and the figures. The authors designed the analysis, chose the methods and parameters,
reviewed the code, checked its outputs, and take responsibility for the results. The code is presented as notebooks
written to be read, and with tests that check the code reproduces expected outputs. As an independent check, the
February 2025 code was rerun with only path changes on the released descending data (notebook 15). The two Tule Pond
creep series agree (correlation 0.97, RMS 0.44 mm) and all five slip events appear in both. The surviving February 2025
output, which ends in August 2024, already shows the first four events (correlation 0.975, RMS 0.49 mm with the current
series; the fifth began in December 2024), so the signal was established before AI-assisted code was used. Claude also
suggested wording edits to the manuscript; the authors wrote and verified all text.
