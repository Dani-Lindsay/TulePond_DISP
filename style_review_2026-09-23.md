# Readability review of the notebooks against Dani's style (2026-09-23)

Reference for the accepted style: `scripts/03_creep_timeseries_v2.py`. Every
suggestion here is a rename, a re-layout or an inline: none changes a number,
a product, a filename or a registry key. Where a name appears in an output
filename or a `REG.register` key, that literal string stays.

Semicolon-joined lines per file (the quickest measure of "three things on one line"):
02: 12, 03 (old): 28, 03 v2: 19, 04: 94, 05: 62, 06: 79, 08: 46, 09: 96.

## The patterns that recur everywhere

1. **Semicolon set-up lines at the top of analysis cells**: load a table, then
   three or four one- or two-letter aliases of its columns (04:120-126,
   05:362-364, 06:606, 08:237, 09:372-375). Fix: one statement per line, and
   use the dataframe's own column names instead of aliases.
2. **The same helper copied into several notebooks under different names**:
   `gaussian_smooth_km` (03, 04), `step_fit` (02, 05), `daily_creepmeter`
   (04, 05), `design` / `fit_joint` (04, 08), the InSAR stacking function
   (04 `load_stack`, 05 `stack_insar`, 08 `insar_stack`), and the map-frame
   trio `to_uv` / `uv_to_lonlat` / `gmt_map_size` (04, 05, 06). One copy each
   in `insar_utils`, one name each, one test each.
3. **Figure cells that grew into option-heavy drawing functions** (06
   `draw_fig2` / `los_maps` / `los_profile_pairs` with a module-level `fig`
   and `global fig`; 05's five-creepmeter row loop; 02's enumerate-with-`if i`
   panel loop). Everywhere else the figures are flat pygmt / matplotlib
   calls, which is the reference form. The fix is to unroll, not to generalise.
4. **Panel sizes as `Wm/Hm/Wp/Hr/Wd`** and results carried in dicts of slices
   (`qc`, `solutions`, `los_profiles[o][n_]["blocks"]`) where a printable
   dataframe with named columns would do.

## Per notebook

### 02_velocity_decomposition (474 lines): light rename and comment pass
- 116-121 `ia, aa`, `id_, ad`, `L_asc` -> `inc_asc, azi_asc`, `inc_desc, azi_desc`, `los_vector_asc/desc`.
- 142-144 `prof` -> `x_fault` (the same table the reference calls `x_fault`; one name across notebooks). 268 `xf` -> `x_fault_transects`.
- 325-342 `strike_frame` returns a closure plus a six-tuple of one-letter offsets; used once. Write the two lines of arithmetic where they are needed, with the names `along_e, along_n, cross_e, cross_n, u_origin, v_origin`.
- 345-361 `rotated_map`: `ug/vg/U/V/E/N/ci/ri/ok` -> `u_km, v_km, easting, northing, col_index, row_index, on_grid`. Used twice, so it stays a function.
- 394-396 the two-iteration `enumerate` panel loop with `if i:` -> two explicit panel blocks.
- Inline `sample_xfault` (228-257) and `strike_frame` (325), each used once. `step_fit` (260) is used three times and stays.

### 04_event_detection (1045 lines): needs restructuring
- 120-126 the densest block in the repo: `F`, `t`, `o`, `ks`, `lat_s` and an assigned lambda. One statement per line; `F` -> `fp_matrix`, `t` -> `dates`, `o/ks/lat_s` -> `north_to_south, km_sorted, lat_sorted`; `lat_of_km` becomes a `def`.
- 769 `near = np.abs(t_reg[:, None] - t[None, :]).argmin(axis=1)` -> a three-line loop or `np.searchsorted` with the "nearest epoch within 9 days" rule spelled out. Same indices.
- 759-771 `show/osh/gres/gi/okg/hov` -> `in_window, south_to_north, hovmoller_grid, row_index, on_grid`.
- 784 `resb` -> `band_stack_90`; 574-582 `bud` -> `slip_budget` (it is already a printable dataframe with named columns).
- 556-557 the nested `join` one-liner for the CI overlap -> a small loop building `overlap_lines`.
- 290 `qc[key] = dict(tt=, y=, n=, fits=, events=, sig=, mid=, val=, cm_ev=, cm_series=, cmcode=)`, a dict of slices carried across three cells for two segments -> two explicit blocks.
- Inline `event_detrend` (426-439), used once. `daily_creepmeter` (132) is duplicated in 05: one copy in `insar_utils`.

### 05_validation (801 lines): the figure cell needs restructuring, light pass elsewhere
- 362-364, 378-381 `ae, an`, `pe, pn`, `x0, dx`, `y0, dy`, `ci, ri`, `ob_z` -> `along_e/along_n, cross_e/cross_n, col_index/row_index, map_rotated`, one per line.
- 485-487 `cig/rig/okg/zg`: the same lookup a second time 100 lines later. Do it once, keep `grid_ll`.
- 462-467 `gmt_map_size` (three call sites across 04/05/06) -> `insar_utils`.
- 429 `lev0, tl0 = cfw_level(lo, +1); lev1, tl1 = ...` with a comment saying the names `v0/v1` are taken by the map frame: the file is reporting its own name collision. -> `level_start, t_start`, `level_end, t_end`, and the map-frame `v0` -> `cross_origin`.
- 545-615 the five-creepmeter loop: 70 lines, three panels in one body, `Wp/Wc/Wd/Hr/Hm`, `a_, b_, r, s, d, v`. Keep the loop over `creepmeters` (that part is hers) but give the three panels three separated blocks with `profile_*`, `cumulative_*`, `detrended_*` names, and lift the sizes into the constants cell with words.
- 715-736 `.assign(_z=...)`, `panel_label`, `draw(ax, col, scale)`: keep `draw` (used four times), rename `col` -> `score_column`, `_z` -> `is_selected`.
- Inline `largest_event` (236) and `n_eff` (247), each used once. `step_fit` (215) duplicates 02's; `load_fp` (650) is the reference's `read_creep_ts`.

### 06_event_field_geometry (1161 lines): needs restructuring
- 403-406 `L = {o: los_unit_vector(...)}` while `L_` is the stats list (1093), and `L` is the stats list in 04/05/08. -> `los_vector` and `stats_lines`.
- 449 `A, S = 0.5 * (p - p[::-1]), 0.5 * (p + p[::-1])` -> `antisym_profile, sym_profile` (the csv columns already say so).
- 836, 948 `fig = None` then `global fig` inside `draw_fig2`, with six drawing functions mutating a module-level figure and `draw_fig2` taking eight options. Write Fig 2 once, top to bottom, in one cell (as 02 and 04 do); the FP/Up test figure (979-992) becomes a second cell, or is deleted (already agreed). The panels read from `fp_rate/up_rate/fp_evt/up_evt` and `fpup_profiles`, computed before the drawing, so the numbers are untouched.
- 717, 722, 1112 `los_profiles[o][n_]["blocks"]["W"]["amp"]`, four levels deep -> one flat dataframe `block_amps` with columns `track, profile, block, amp_mm, se_mm`, printed and written; the register loop at 1149-1151 reads named columns.
- 565-578 `decompose(v_asc, v_des)` closes over `fits` and `strike` and half the result is thrown away -> pass `fits` in, or inline the three-line MintPy call.
- `map_panel` (1034) and `local_strike` (630) are thin wrappers; `block_step` (580) is used about ten times and stays.

### 09_alignment_arrays (458 lines): light rename and comment pass
- 237-240 `t, cum` and `cat, excess, b, m, G = run_detector(...)` -> `survey_yr, cum_slip` and `hrkt_events, excess_mm, inter_event_rate, coef, design_G`. `b` and `m` reappear 30 lines later in print and register lines.
- 260-264 `interval_series` returning `ivt, iv, ive` -> `interval_midpoint_yr, interval_yr, interval_window_yr`.
- 272-277 and 380-383 `mv_t, mv, mv_err, mv_src, cc, med, tr_`: the same five merge lines twice -> one small `merge_catalogues(array_events, insar_events)` used in both places.
- 203-212 `detect_intervals` uses `b` for the rate and `b_` for a loop index in the same return -> `rate_mm_yr`, `i_end`.
- 449-453 rebuilds `pd.DataFrame(step_rows)` inside the register loop -> build `aa_steps` once at 158 (it is already written to csv there).
- 220-227 `while True` inside `for` inside `run_detector` with a `pop` -> a drop-one-at-a-time `for` loop with a printed line per dropped event.
- `design` (197) duplicates 04's exactly -> `insar_utils`.

### 09_seasonality_forecast (676 lines): needs restructuring
- 274 and 511 two assigned lambdas (`winter`, `dowy_label`) -> `def n_dec_to_mar(catalogue)` and `def dowy_label(day_of_water_year)`; `winter` is used in register lines at 625-626.
- 287-299 `fit(model, x)`: a five-branch string dispatch hiding five distributions -> five explicit two-line blocks in the loop (same parameterisations, so AICc and the quantiles are bit-identical).
- 322-331 `forecast(...)` with `cdf/n/means/covs` and a bootstrap nested in a timing loop: called 40 times, so it stays a function, but one `#` line per step and `d0` -> `fitted_interval_dist`, `xb` -> `bootstrap_intervals`, `tau` -> `time_since_last_yr`.
- 543-545 `A, B, C = fc[...]` -> `insar_forecast, array_forecast, merged_forecast`.
- 372-375 `iv, n_iv, tcrit, half, nxt, sd_ci`: the closed-form clock the abstract quotes, in six two-letter names on four lines -> `intervals_yr, t_critical, half_width_yr, next_onset_yr, sd_ci_days`.
- 588, 598, 601 `r._asdict()[f"lagY{Y}_hi"]` -> select the `lagY*` columns by name.
- 380, 539 `print("\n".join(L[L.index("RENEWAL FORECAST"):]))`, slicing the stats list by searching for a header -> build each section's lines in its own list.
- The merged-catalogue construction at 220-221 duplicates 08's (269-271).

### 10_paper_figures (63 lines): fine as is
Two `print(...); continue` lines to split (56, 62). Otherwise flat, named, printed.

## Suggested order

| # | Notebook | Effort | Why |
|---|---|---|---|
| 1 | 02 | half day | Closest to the reference; renaming `prof` -> `x_fault` sets the vocabulary every later notebook inherits; nothing downstream depends on its internals. |
| 2 | 08 | half day | Five functions, all reused; names and semicolons only. Feeds 09, so its vocabulary should settle first. |
| 3 | 04 | day | The most-read notebook (it defines `creep_events.csv` for 05, 06, 08, 09) and the worst first ten lines; the Fig 1 cell alone is 250 lines. |
| 4 | 09 | day | Two lambdas, the string-dispatch `fit`, and the abstract's forecast numbers hidden in two-letter names. After 08, so the catalogue merge is shared. |
| 5 | 05 | day | Mostly one 165-line figure cell; the analysis half is nearly hers already. |
| 6 | 06 | day, the longest | The `global fig` and six option-heavy drawing functions have to be unwound; do it after the shared map helpers have a home in `insar_utils`. |

Gate for every port: every csv in the notebook's output folder equal to the previous version's, no registry conflict, pytest green. Same as 03.

## Steps still done outside the repo (found the way the trace builder was)

The trace builder (`run07`) turned out to live only in `Hayward_DISP-S1`; it is now a cell of
notebook 01. These are the other steps the chain depends on that are not in this repo yet,
from the "ported from" and "frozen" notes in the scripts:

| # | Step | Where it lives | What reads it | Status |
|---|---|---|---|---|
| 1 | OPERA DISP-S1 preparation: download, rebase, tropo, MintPy conversion, masks, velocity fit, GNSS tie (`velocity_NA.h5`) | `Hayward_DISP-S1/analysis/run00` to `run05` | 01 fingerprints, 02, 03, 06 | described in 01 and Methods M1; products to go on Zenodo; not re-run here by design |
| 2 | The same-orbit velocity mosaics (`mosaic/<orbit>_velocity_NA.h5`, `_geometry.h5`) | not yet located (run04/run06 in `Hayward_DISP-S1`?) | 02 | FIND the script and either port it into 01 or document it there |
| 3 | The aperture / coherence sweep (`outputs/sweep/ap*_coh*`) | `Hayward_DISP-S1/analysis/run11b_sweep_ladder.py` (and run11, 11c, 11d) | 05 (the ladder, the coherence comparison) | frozen inputs, fingerprinted; Dani decided 2026-09-21 not to re-run; state that in Methods |
| 4 | The slip inversion, depth scan, bootstrap (M7) | `Hayward_DISP-S1/modelling/m02, m05, m13` | 07 (empty) | to be built descending-only; questions on the prior, surface constraint, five vs four events |
| 5 | The legacy step maps and rate map used only for parity checks | `proj_HaywardCreep/outputs/modelling`, `outputs/event_maps` | 06 CHECK 4 and 5 | checks only; can be deleted once 06 is ported with its own zero-difference gate |
| 6 | GHCN rain-gauge download and the COOP observation-time shift | `Hayward_DISP-S1/analysis/run20_precip_ghcn.py` | 09 (frozen csv in `Inputs/GHCN_daily`) | data download; document the query in 01 or 09 |
| 7 | The prediction-statement wording | legacy `run22` template, now inside 09 | 09 | in the repo already |

Rule going forward: anything that turns an input file into another input file belongs in
notebook 01 with a print and an assert that it reproduces the copy under `Inputs/`, as the
trace now does.
