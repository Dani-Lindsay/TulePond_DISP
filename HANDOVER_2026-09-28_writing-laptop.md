# Handover, 2026-09-28 (writing laptop): the March-2026 data refresh

Companion to `HANDOVER_2026-09-28.md`, which was written on the modelling laptop
and still governs the two-laptop split. This one covers the data refresh and the
repository work done on the writing laptop. Nothing in it is speculative: every
number below was measured, and every code change named is committed and pushed.

State at the end of the day: **working tree clean, HEAD = origin/main =
`46a8410`.** The chain runs green from `01` to `05` on the refreshed data.

---

## 1. What the data is now

The refresh ran the whole preparation chain. OPERA has published nothing past
March 2026 (latest production date 2026-08-17), so this is the current archive.

| frame | epochs | window | note |
|---|---|---|---|
| 09157 asc, track 35 | 359 (+15) | 2016-07-29 → **2026-03-27** | creep time series use this frame |
| 11115 des, track 42 | 374 (+15) | 2016-09-03 → **2026-03-21** | |
| 09156 asc, track 35 | 359 (+15) | 2016-07-17 → **2026-02-13** | mosaicked with 09157 for velocities only |

The last acquisition common to both paper tracks is **2026-03-21**, three months
later than the 2025-12-15 the manuscript currently quotes. The common date grid
is now 439 epochs, was 423.

`date_end` in `TulePond_filepaths.py` is `2026-03-27`. `run02_static_layers` was
**not** re-run: the AOI bbox is unchanged, so July's geometry still aligns
pixel-for-pixel with the new stacks (checked against `los_enu`, `dem` and
`incidence_angle` on every frame).

## 2. The repository is now self-contained: `data_prep/`

The download and preparation chain lives in the repo, ported from
`Hayward_DISP-S1/analysis/` unchanged except for imports and a provenance note:
`run00` (UNR GNSS), `run01` (OPERA download + rebase), `run02` (CSLC static
layers + DEM), `run03` + `tropo_to_mintpy` (OPERA TROPO), `run04` (MintPy,
masks, velocity, mosaics), `run05` (GNSS tie), `run20` (GHCN rain).
`data_prep/README.md` has the run order, the requirements, the inputs that have
no script because they were received by hand, and where each writes.
`Hayward_DISP-S1` is now provenance only.

The legacy `runNN` names are kept deliberately so the Methods,
`assumptions_register.md` and `style_review_2026-09-23.md` cross-references still
resolve; that is why the rain download is `run20` and there is a gap after
`run05`. They cannot live in `scripts/`, where jupytext pairs every `.py` with a
notebook, and they are not in `run_all.sh` — hours of downloads, run once per
refresh.

**Two corrections were made to `run01` and both matter if the numbers are ever
rebuilt** (documented in `data_prep/README.md` under Provenance):

- The download window is now `date_end` + 1 day. CMR's `--end-datetime` is a
  datetime, so passing the bare date dropped that day's acquisition — Bay Area
  passes are ~02:07 UTC ascending and ~14:07 descending, both after midnight.
  This had silently cost the 2026-03-27 ascending acquisition.
- `aoi_bbox_disp` replaces `aoi_bbox` for the 09157 and 11115 downloads.
  `opera-utils` subsets by row/column and its `lonlat_to_rowcol` moved by one
  pixel between the July download and September, so the same bbox that gave a
  2333 x 3048 window in July gave 2332 x 3049 afterwards. Mixing windows makes
  `disp-s1-reformat` fail. `aoi_bbox_disp` is the same rectangle nudged 34 m
  west and 32 m north so new granules land on the window the archive already
  uses; the lattice is unchanged (x, y = 15 mod 30), so these are the same
  ground pixels. The science AOI is untouched.

## 3. The chain is single-track and is Dani's code

The `_v2` notebooks were promoted to the plain names (`46a8410`). `02`, `03`,
`04` and `08` take the canonical names; `09_rain_seasonality` and `10_forecast`
replace the pre-split `09_seasonality_forecast`; the superseded AI-written
notebooks moved to `drafts/originals/`, byte-identical to their old blobs.

If the direction is ever in doubt: the promoted scripts carry `# [n]` cell
numbering (`number_cells.py` was only ever run on the v2s) and the drafts copies
have none. **Do not trust git's rename lines** for this commit — its similarity
heuristic pairs the `.ipynb` files misleadingly.

This had to happen for the refresh to be coherent, not for tidiness: before the
swap the v2 notebooks read the *originals'* output folders, so the originals had
to run to feed them, and `run_all.sh` globs `NN_*.ipynb`, so `./run_all.sh 04`
ran both 04s and every number was computed twice into the registry.

The port-parity CHECK cells were removed from the promoted notebooks. They
compared each v2 product against the originals' output folder, and nothing
writes those folders now. The zero-difference proof stands in history at
`dea2f59` on the Dec-2025 data.

**Crossed the two-laptop split, with Dani's approval:** notebook 07 read
notebook 04's output folder by its `_v2` name, so the rename would have broken
it. Three strings changed to `04_event_detection`, in `scripts/07_slip_model.py`
(line 68), `scripts/07_slip_model_tests.py` (373) and
`drafts/07/interim_results_2026-09-28.py` (46). Nothing else in 07 was touched.
**The modelling laptop must pull before its next 07 run.**

For the record, because it caused confusion: 04 does not depend on 07. 07
consumes 04's output, which is the right order. Only the folder *name* coupled
them.

## 4. Three caching bugs that served stale data — read this before any refresh

The same bug appeared in three places: an expensive result was cached on *"does
the output exist"* or *"are the parameters unchanged"*, never on the identity of
the input **data**. A refresh adds epochs while every parameter stays the same,
so each cache hit returned the previous vintage and the script exited 0.

It was caught because notebook 04's `event_fraction_by_crossing.csv` came out
**byte-identical** after three months of new data, and because notebook 03 ran in
**21 seconds** for a step that has to read 20 GB.

- `data_prep/run04_velocity.ensure_mintpy` skipped `disp_nc_to_mintpy` whenever
  `timeseries.h5` existed, so the new epochs never entered MintPy and the masks,
  `TROPO.h5`, velocity and mosaics were rebuilt from July data. The unmasked
  `timeseries_tropo.h5` — exactly what 03 and 06 read — had the same guard.
- `scripts/03_creep_timeseries` cached the per-crossing extraction on
  `creep_params_<track>.json`; the aperture and coherence had not changed, so it
  reused September LOS series and rebuilt `fp_ts.csv` and everything downstream.
- `scripts/06_event_field_geometry` keyed its per-pixel fit cache on the *step*
  dates, which sit in 2017-2024 and do not change when 2026 epochs are appended,
  so the existing assert passed on a stale fit.

All three now compare the input date list and rebuild when it differs (`78f904b`).
06 recomputes a mismatched cache rather than raising.

Also fixed: 03's legacy parity check asserted equal shapes against the **frozen**
July/August sweep products, which cannot hold once the data moves on (654 x 359
now against 656 x 344). It is now vintage-aware — still asserts equality while
the shapes match, otherwise reports that it was skipped. Expect the same class of
problem in any other check that compares against frozen products.

**On any future refresh, check mtimes and date ranges of every product, not exit
codes.** Anything that runs suspiciously fast has probably skipped its work, and
no product should be byte-identical to the previous vintage.

## 5. Where the numbers stand

`pytest` is **29 failed, 86 passed**, and every failure is `test_anchors`. They
pin pre-refresh manuscript values, so they are the refresh landing correctly, not
a fault. Examples:

```
03_n_acquisitions_asc   registry 359   manuscript 344
03_n_acquisitions_des   registry 374   manuscript 359
03_n_common_dates       registry 439   manuscript 423
03_n_crossings_asc      registry 654   manuscript 656
02_xfault_step_CFW1     registry 6.62  manuscript 6.68
```

**Refitting the anchors is the highest-value next job.** Until it is done there
is no automated check that a later change has not moved a number it should not
have — which is exactly the protection that caught the three caching bugs.

Notebooks **06 and later have not been re-run on the refreshed data.** That is
the "combine", deliberately deferred while the modelling was in flight. Note
`outputs/06_event_field_geometry/fp_maps.npz` was regenerated at 15:44 from the
*old* MintPy products with the new 06 code: it is a valid Dec-2025 result, so do
not copy it to the modelling laptop expecting March-2026 data.

## 6. The aim for notebooks 06 and 07 — agreed, not started

**No code has been written for this.** It is recorded here so the intent is not
lost.

The principle Dani set: *the main notebooks should be clear recipes for how the
results in the Science publication are generated.* Explorations and tests do not
belong in them — clean checks move to `tests/`, everything else to `drafts/`.

The split, in her words, is **"mean events through to the quadtree"** as one
notebook that prepares the input data, and **"the model parts"** as the other:

- **Event field and inversion inputs** — per-pixel time-function fits, step maps,
  the descending burst correction, the mean event, the fault-parallel/vertical
  decomposition, then the noise model and quadtree (which move out of 07). Ends
  with an inversion-ready dataset.
- **Slip model** — mesh, Green's functions, surface constraint, inversion, the
  sigma L-curve, bootstrap, numbers.

This boundary is clean and needs no reordering: the quadtree section touches no
mesh objects (it is distance-to-trace only), while the surface constraint does
(`fault.patch`, `fault.patchll`) and so must stay on the model side. Once the
quadtree leaves 07, 07 simply starts at the mesh and loads a prepared dataset.

Specific decisions already taken:

- **Fig 2 moves out of 06** into its own notebook that runs *after* the model, so
  its row 3 is the real model rather than a placeholder.
- **Drop the ascending-deficit check plot** (06 CHECK 6) — not useful. The
  ascending test itself goes to `07_slip_model_tests`.
- The legacy parity checks (06 CHECK 4 and 5, both gated) and the symmetry
  material are exploration and belong in `drafts/` or `tests/`.
- Renumbering is permitted to get the run order right.

**Open question, deliberately unanswered.** Giving Fig 2 its own notebook after
07 needs a number that sorts after 07, which means renumbering 08→09, 09→10,
10→11, 11→12. That cascade reaches the **registry key prefixes**, which are
hardcoded strings and do not move with the notebook name: 26 distinct `08_*`
keys (3 anchor lines), 10 `09_*` keys, 26 `10_*` keys (2 anchor lines). Since the
anchors are being refitted anyway, the marginal cost is lower than usual — but it
is a real decision and was left to Dani. The alternative is to leave the numbers
alone and let `11_paper_figures` draw Fig 2, at the cost of turning a collector
into a drawer.

**Independent of all that, a live bug:** `run_all.sh` globs alphabetically, so
`07_slip_model` runs **before** `07_slip_model_tests`, which is the wrong order.
Renaming the tests notebook to `07_model_tests` sorts it first and needs no
renumbering.

## 7. Other open items

- **`coh_maps` lives in `scripts/06`**, while `coh_timeseries` and `coh_velocity`
  are in `TulePond_filepaths`. It is now a shared constant — the modelling
  depends on it and the Methods quote it — so it belongs with the other two.
  Not moved, because the modelling was running against 06's outputs.
- **Notebook 01 does not register every number M1 quotes.** It registers
  `01_n_acquisitions_{asc,des}`, `01_first_date_*` and `01_last_date_*` for the
  two paper frames only. M1 also states 09156's span and the mini-stack counts
  (24 ascending, 25 descending), which are in no registry. Now that
  `TulePond_filepaths.frames` covers all three frames, 01 can close this.
- **Supplement figures S1 and S2 are not in the repo.** They are still
  `Hayward_DISP-S1/plotting/figS1_processing.py` and `figS2_gnss_tie.py`, and they
  read exactly the products run04/run05 rebuild, so they are stale. Agreed plan:
  port into `data_prep/` and **restyle onto `insar_utils.mpl_paper_style`** rather
  than porting Hayward's `figstyle.py`, so the supplement is one visual system.
  The only real gap is `map_panel`, which should become a shared helper in
  `insar_utils`. Both figures will look different from the current draft, so Dani
  reviews them before they are accepted. S3 (notebook 03) and S4 (notebook 05)
  are in the repo and regenerate normally.
- **A manuscript numbering error.** M3 cites "(Figure S3)" for the rate-mask
  pixel rejection, but the figure captioned S3 is the acquisitions grid; the
  rate-mask figure (`figS_ratemask_*`, notebook 03) is generated but appears
  nowhere in the supplement. Notebook 03 still uses the legacy numbering
  internally (`figS3 = figS_ratemask...`, `figS4 = figS_acquisitions...`). Either
  give the rate-mask figure a slot or repoint the citation.
- **The data root move, approved but not done.** `proj_HaywardCreep` is a mixed
  legacy tree (177 GB, including 3.9 GB of Hayward-era `outputs/` and old
  `Figures_*`). Agreed plan: `mv` the three frames, `Inputs`, `GPS`, `mosaic` and
  the frozen sweep into a clean `/Users/dlindsay/work/TulePond_data`, point
  `TULEPOND_DATA_DIR` at it, and repoint `sweep_dir` at `frozen/sweep` so there
  is only one `outputs/` in the project. Same filesystem, so the move is instant.
- **Zenodo.** Data only; the code is cited from GitHub via the release
  integration. Archive the data the figures are built from plus the lowest-level
  data it came from; skip the big intermediates. On that rule the deposit is
  ~4.3 GB. Limits are 50 GB per record and **100 files**, so it must be a handful
  of zips, not a loose tree. `Inputs/` trims from 2.3 GB to ~0.75 GB by dropping
  1.67 GB of duplicate uncompressed `.10min` creepmeter files. If the time series
  is ever wanted, it is stored uncompressed and lossless gzip-4 measured 2.6x.

## 8. M1 has been checked line by line against the code

The counts, spans and mini-stack numbers in the current draft of M1 are correct
for the refreshed data. Two substantive problems remain in it:

- The sentence about the **recommended masking layer at 0.5** is wrong three
  ways: the code uses `reliability_threshold = 0.9`; the parameter is not a
  threshold on the mask's value but the *fraction of epochs* in which a pixel
  must be valid; and the resulting `recommended_mask_90thresh.h5` is **never read
  by run04 or any notebook**, so the sentence describes a step the analysis does
  not perform. Dropping it is the honest fix. The clause "retain only pixels with
  an unbroken record" in the same sentence *is* real (`maskRecord`) and must be
  kept.
- The **coherence story** is three-way and the draft collapses it. `0.7` is
  `coh_timeseries` and `coh_velocity`: creep time series, the per-pixel step
  fits, velocity fields, GNSS tie. `0.9` is `coh_maps`: notebook 06's per-pixel
  maps, the profiles and the decomposed field. The nuance worth stating is that
  06 **fits at 0.7** and applies the 0.9 mask afterwards, asserting it is a strict
  subset — so 0.9 changes which pixels are shown and inverted, never the fits.

Also worth reconciling: M1 says 09157's coverage "spans the full creeping section
north of 37.4 N". Measured within 500 m of the trace on the rebuilt field, near-
fault coverage is 0.0% between 37.40 and 37.45 (53 crossings with no data), 27%
between 37.45 and 37.50, and 78-86% north of 37.50. **37.49 was right**; 37.4
overstates it. M3 samples crossings south to 37.41, so the two numbers should be
made consistent.

Finally, the GNSS tie numbers moved: station residual RMSE is now **1.47 → 0.76**
mm/yr ascending and **1.01 → 0.66** descending (was 1.6 → 0.8 and 1.0 → 0.7), and
35 stations are used ascending (13 on 09156, 22 on 09157), 29 descending.

## 9. Housekeeping

Files moved aside rather than deleted, all with a README explaining why:

- `proj_HaywardCreep/MISC_2026-09-28/` — `maskTempCoh_08.h5` and
  `velocity_notropo.h5` per frame (neither is written or read any more, both
  stale from July), and the `_verify_patch` ROI patches (~1.8 GB, re-downloaded
  automatically if absent). Plus `TulePond_outputs/` holding the retired `_v2`
  output folders and the pre-split `09_seasonality_forecast`.
- `proj_HaywardCreep/snapshot_2026-09-28_pre_refresh/` — the pre-refresh state:
  small MintPy products, mosaics, GPS, the registry, and a full copy of
  `outputs/` as it was before any of this.

`TulePond_filepaths.mask_file` dropped its `0.8` entry (nothing selected it and
run04 stopped writing it, so the path resolved to a stale July file), and its
comment had 0.7 and 0.9 the wrong way round.
