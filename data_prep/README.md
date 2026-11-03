# `data_prep/`: from the public archives to the prepared inputs

These scripts turn open-access archives into the prepared products that the
analysis notebooks read. Every file under `data_dir` that the notebooks open is
written by one of them or listed under "Inputs that are not produced here", and
notebook 01 fingerprints those products so each number in the paper can be
traced back through this directory.

They are not part of `run_all.sh`. Between them they download tens of
gigabytes and run MintPy over ~360 acquisitions per frame, so they are run once
per data refresh, by hand, in the order below.

The `runNN` names belong to the data-preparation chain and are unrelated to the `NN_name` numbers of the analysis
notebooks.

## Run order

| Script | Fetches / does | Writes |
|---|---|---|
| `run00_prepUNR.py` | UNR / Nevada Geodetic Laboratory daily GNSS time series (`geodesy.unr.edu`), screened for steps and non-linearity, trends fitted over the InSAR window | `GPS/<frame>_enu_NA.csv`, `GPS/NA/*.tenv3`, station plots |
| `run01_download_opera.py` | OPERA DISP-S1 v1.0 granules for frames 09156, 09157, 11115, subset to the AOI, then rebased into one continuous stack per frame (`disp-s1-reformat`) | `<frame>/data/*.nc`, `<frame>/<frame>_stack.nc` |
| `run02_static_layers.py` | CSLC static layers for the bursts intersecting the AOI, and one ellipsoidal Copernicus GLO-30 DEM, resampled onto each frame's stack grid | `<frame>/geometry/*.tif`, `dem_glo30.tif` |
| `run03_tropo.py` | OPERA L4 TROPO zenith-delay granules bracketing each acquisition (found via CMR), cropped to the AOI and projected into line of sight | `<frame>/tropo/cropped/`, `<frame>/tropo/corrections/` |
| `run04_velocity.py` (+ `tropo_to_mintpy.py`) | MintPy conversion, common reference pixel at GNSS ZOA1, record and coherence masks, troposphere correction (`TROPO.h5` from run03's corrections), velocity fit, and the same-orbit mosaics | `<frame>/mintpy/timeseries*.h5`, `TROPO.h5`, `velocity*.h5`, `mask*.h5`, `mosaic_velfield/<orbit>_*.h5` |
| `run05_GPS_InSAR_validation.py` | Plate-frame tie of each velocity field to the screened GNSS, and the tied mosaics that notebook 02 decomposes | `<frame>/mintpy/velocity_NA.h5`, `mosaic_velfield/<orbit>_velocity_NA.h5`, `gps_insar_validation.csv` |
| `run06_precip_ghcn.py` | NOAA GHCN-daily records for the gauges nearest Tule Pond, with the COOP observation-time shift documented | `external_inputs/GHCN_daily/*.csv` |
| `run07_aperture_sweep.py` | The aperture sweep that notebook 05 scores: notebook 03's creep time series (`creep_profiles.py`) re-run for every cross-fault bin of 05's aperture ladder at coherence 0.9 and for the paper's 30-150 m bin at coherence 0.7, 0.8 and 0.9, each scored against the five creepmeters | `aperture_sweep/<tag>/fp_ts.csv` (and the per-track series), `aperture_sweep/sweep_summary_mp5.csv`, `sweep_results_mp5.csv`, `masks/<frame>/maskTempCoh_08.h5` |

`run05` imports `frame_grid` and `merge_orbit` from `run04`, so those two stay
together. `run07` reads the MintPy products of `run04` (and imports its mask
helpers to build the coherence-0.8 mask by the same rule), the fault crossings
and the creepmeter records; it runs in minutes. Its run of the paper's bin at
coherence 0.7 is the computation of notebook 03, and it asserts that this
`fp_ts.csv` equals notebook 03's. `run00` is independent of `run01`-`run04` and only has to precede
`run05`. `run02` depends on the stack grid written by `run01` and has no date
dependence, so a refresh that keeps the same AOI bbox does not need to re-run it.

## Running them

```bash
python data_prep/run01_download_opera.py          # from the repo root, in the environment.yml environment
```

Requirements: the `environment.yml` environment (`opera-utils`, MintPy, h5py, xarray, rasterio, pyproj), and
Earthdata credentials in `~/.netrc` for everything that accesses ASF or CMR. `run01`-`run03` and `run07` check free
disk space before starting (`insar_utils.require_disk_space`) and stop if the job would not fit.

The observation window comes from `TulePond_filepaths.py` (`date_start`,
`date_end`): edit those two and re-run, rather than editing any script here.

## Where these write

Prepared products go under `data_dir`, beside the frames they belong to, never
into the repository; the notebooks' `outputs/` and `Figures_<date>/` folders are
written only by the notebooks. The prep chain's own QC output is under
`data_dir/prep_validation/` (`validation/` for run05's tie statistics,
`precip/` for run06's prep-side rain tables); run05's per-station residuals are
written beside the GNSS series, `GPS/gps_insar_validation.csv`.
`run00` writes its GNSS station plots under `GPS/Figures/`, sorted into
`Uncorrected/`, `Corrected/`, `Noisy/` and `Manual_Exclude/` for the visual
review that sets the manual exclusion list. The notebooks do not read these QC
products; notebook 12 builds its own rain composite from
`external_inputs/GHCN_daily`, which is what `run06` downloads.

## Inputs that are not produced here

These are received or made by hand, so they have no script. They are under
`data_dir/external_inputs` and are fingerprinted by notebook 01:

| Input | Source |
|---|---|
| USGS creepmeter records (`*.10min.gz`) | Langbein et al. (2024), USGS; the `.gz` copies are the ones the notebooks read |
| Bay Area alignment arrays v3.0 | USGS |
| Repeating-earthquake catalogues | T. Taira (GEN3; not redistributed, available from its author); Waldhauser & Schaff (2021) |
| Relocated seismicity | B. Rong, from the statewide catalogue of Rong et al. (2026, SCEC poster 145); not redistributed, available from its authors |
| Mapped fault trace (`hayward_fault_trace_*.kml`) | mapped by hand in QGIS from the USGS Quaternary faults plus the steps visible in the velocity fields; notebook 01 rebuilds the 100 m crossings from it and asserts they match the archived copy |
| Creep-event catalogue | Gittins et al. (2026) |
| Creep-rate compilation | Johnson et al. (2022) |
| Creepmeter locations (`creepmeter_locations.csv`) | USGS creepmeter metadata |
| Plate boundaries (`pb2002/`, Fig 1F) | PB2002, Bird (2003), as a GMT file |
| SFSVM v26.0 (GeoModelGrids HDF5; `pylith/build/sfcvm_grid.py`) | USGS, doi:10.5066/P13AEJFI |

## Download details

- **The download window ends at `date_end` + 1 day.** CMR's `--end-datetime` is
  a datetime, so passing the bare date would drop that day's acquisition (the
  Bay Area passes are at ~02:07 UTC ascending and ~14:07 UTC descending, both
  after midnight).
- **`aoi_bbox_disp` is used instead of `aoi_bbox` for the 09157 and 11115
  downloads.** `opera-utils` subsets by row/column, converting the bbox corners
  with `lonlat_to_rowcol`, and the resulting window can differ by one pixel
  between downloads. Mixing two windows makes `disp-s1-reformat` fail, which
  `run01`'s grid check catches. `aoi_bbox_disp` is the science AOI nudged 34 m
  west and 32 m north so that new granules land on the window of the existing
  archive; the lattice is unchanged (x, y = 15 mod 30), so these are the same
  ground pixels. The science AOI, `aoi_bbox`, is unchanged.
