# `data_prep/` — from the public archives to the prepared inputs

These scripts turn open-access archives into the prepared products that
notebooks 01–11 read. They are the answer to "where did the input files come
from": every file under `data_dir` that the notebooks open is written by one of
them, and notebook 01 fingerprints those products so each number in the paper
traces back through this directory.

They are **not** part of `run_all.sh`. Between them they download tens of
gigabytes and run MintPy over ~360 acquisitions per frame, so they are run once
per data refresh, by hand, in the order below. `run_all.sh` re-runs only the
notebooks, which read the products these leave behind.

Numbering is inherited from the processing chain these were developed in
(`Hayward_DISP-S1/analysis/run00`–`run20`) and is kept so that the Methods,
`assumptions_register.md` and `style_review_2026-09-23.md` still refer to the
same script names. That is why the rain-gauge download is `run20` and there is
a gap after `run05`. These `runNN` names are the data-preparation chain; the
`NN_name.ipynb` numbers are the analysis notebooks, and the two are unrelated.

## Run order

| Script | Fetches / does | Writes |
|---|---|---|
| `run00_prepUNR.py` | UNR / Nevada Geodetic Laboratory daily GNSS time series (`geodesy.unr.edu`), screened for steps and non-linearity, trends fitted over the InSAR window | `GPS/<frame>_enu_NA.csv`, `GPS/NA/*.tenv3`, station plots |
| `run01_download_opera.py` | OPERA DISP-S1 v1.0 granules for frames 09156, 09157, 11115, subset to the AOI, then rebased into one continuous stack per frame (`disp-s1-reformat`) | `<frame>/data/*.nc`, `<frame>/<frame>_stack.nc` |
| `run02_static_layers.py` | CSLC static layers for the bursts intersecting the AOI, and one ellipsoidal Copernicus GLO-30 DEM, resampled onto each frame's stack grid | `<frame>/geometry/*.tif`, `dem_glo30.tif` |
| `run03_tropo.py` (+ `tropo_to_mintpy.py`) | OPERA L4 TROPO zenith-delay granules bracketing each acquisition (found via CMR), cropped to the AOI and projected into line of sight | `<frame>/tropo/cropped/`, `<frame>/tropo/corrections/`, `<frame>/mintpy/TROPO.h5` |
| `run04_velocity.py` | MintPy conversion, common reference pixel at GNSS ZOA1, record and coherence masks, troposphere correction, velocity fit, and the same-orbit mosaics (step 6) | `<frame>/mintpy/timeseries*.h5`, `velocity*.h5`, `mask*.h5`, `mosaic/<orbit>_*.h5` |
| `run05_GPS_InSAR_validation.py` | Plate-frame tie of each velocity field to the screened GNSS, and the tied mosaics that notebook 02 decomposes (step 5) | `<frame>/mintpy/velocity_NA.h5`, `mosaic/<orbit>_velocity_NA.h5`, `gps_insar_validation.csv` |
| `run20_precip_ghcn.py` | NOAA GHCN-daily records for the gauges nearest Tule Pond, with the COOP observation-time shift documented | `Inputs/GHCN_daily/*.csv` |

`run05` imports `frame_grid` and `merge_orbit` from `run04`, so those two stay
together. `run00` is independent of `run01`–`run04` and only has to precede
`run05`. `run02` depends on the stack grid written by `run01`, and has no date
dependence, so a refresh that keeps the same AOI bbox does not need to re-run it.

## Running them

```bash
PATH=/Users/dlindsay/miniforge3/envs/opera_disp_feb13/bin:$PATH
python data_prep/run01_download_opera.py          # from the repo root
```

Requirements: the `opera_disp_feb13` environment (`opera-utils`, MintPy, h5py,
xarray, rasterio, pyproj), and Earthdata credentials in `~/.netrc` for
everything that touches ASF or CMR. Wrap long runs in `caffeinate -is` — the
machine sleeping through a rebase looks like a hang. The scripts check free
disk space up front (`insar_utils.require_disk_space`) and refuse to start
rather than filling the disk halfway through.

The observation window comes from `TulePond_filepaths.py` (`date_start`,
`date_end`): edit those two and re-run, rather than editing any script here.

## Where these write

Prepared products go under `data_dir`, beside the frames they belong to, never
into the repository: the notebooks' `outputs/` and `Figures_<date>/` are theirs
alone. The prep chain's own QC output has one home, `data_dir/data_prep/`
(`validation/` for run05's per-station GNSS residuals, `precip/` for run20's
prep-side rain tables) -- deliberately not a second `outputs/` tree. `run00`
writes its GNSS station plots under `GPS/Figures/`, sorted into
`Uncorrected/`, `Corrected/`, `Noisy/` and `Manual_Exclude/` for the visual
review that sets the manual exclusion list. Nothing in notebooks 01-11 reads
any of that; notebook 09 builds its own rain composite from `Inputs/GHCN_daily`,
which is what `run20` downloads.

## Inputs that are not produced here

These are received or made by hand, so they have no script. They live under
`data_dir/Inputs` and are fingerprinted by notebook 01:

| Input | Source |
|---|---|
| USGS creepmeter records (`*.10min.gz`) | Langbein et al. (2024), USGS; the `.gz` copies are the refreshed ones and are what the notebooks read |
| Bay Area alignment arrays v3.0 | USGS |
| Repeating-earthquake catalogues | T. Taira (HFRC GEN3, received 2026-08-26); Waldhauser & Schaff double-difference |
| Relocated seismicity | B. Xie, GrowClust catalogue for the Hayward area |
| Mapped fault trace (`hayward_fault_trace_*.kml`) | re-mapped by hand in QGIS from USGS Quaternary faults plus steps visible in the velocity fields; notebook 01 rebuilds the 100 m crossings from it and asserts they match the copy under `Inputs/` |
| USGS Quaternary faults, double-difference catalogue | public USGS / NCEDC downloads, used for context figures |
| Aperture / coherence sweep (`outputs/sweep`) | frozen output of `Hayward_DISP-S1/analysis/run11b_sweep_ladder.py`; notebook 05 reads it and does not re-run the sweep (2026-09-21) |

## Provenance

Ported into this repo on 2026-09-28 from `Hayward_DISP-S1/analysis/`, unchanged
except for the imports (`TulePond_filepaths`, the repo-root `insar_utils`) and a
note in each docstring. Two corrections were made to `run01` during the
March-2026 refresh, both worth knowing if the numbers are ever rebuilt:

- **The download window is now `date_end` + 1 day.** CMR's `--end-datetime` is a
  datetime, so passing the bare date dropped that day's acquisition — the Bay
  Area passes are ~02:07 UTC ascending and ~14:07 UTC descending, both after
  midnight. This had silently cost the 2026-03-27 ascending acquisition.
- **`aoi_bbox_disp` replaces `aoi_bbox` for the 09157 and 11115 downloads.**
  `opera-utils` subsets by row/column, and its `lonlat_to_rowcol` shifted by one
  pixel between the 2026-07 download and the 2026-09 refresh, so the same bbox
  that produced a 2333 × 3048 window in July produced 2332 × 3049 afterwards.
  Mixing the two makes `disp-s1-reformat` fail, which `run01`'s grid check
  catches. `aoi_bbox_disp` is the same rectangle nudged 34 m west and 32 m north
  so that new granules land on the window the existing archive already uses; the
  lattice is unchanged (x, y = 15 mod 30), so these are the same ground pixels.
  The science AOI, `aoi_bbox`, is untouched.
