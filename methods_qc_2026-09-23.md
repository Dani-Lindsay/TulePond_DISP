# Methods QC against the publication notebooks (2026-09-23)

Manuscript: `TulePondCreepEvents_Science_Template (2).pdf`. Each claim in
Materials and Methods M1 to M9, the supplementary figure captions, and the
main-text numbers they support was checked against `scripts/NN_*.py`,
`outputs/registry.txt`, the `outputs/*/*_stats.txt` sidecars and
`tests/test_anchors.py`. Where a claim is backed only by the legacy
`Hayward_DISP-S1` chain (not by this repo) it is marked **H-only**.

Verdicts: CORRECT, STALE (was true of an earlier run; the registry now says
otherwise), WRONG (does not describe what the code does), UNVERIFIABLE (no
code or registry entry), LEGACY-ONLY (Hayward_DISP-S1 modelling, not
reproduced here).

## The five things to fix first

1. **Coherence threshold.** M1 and M3 say the cross-fault series use 0.9 and
   the profile 0.7. The repo runs ONE stream at 0.7 everywhere
   (`TulePond_filepaths.py` `coh_timeseries = 0.7`, decided 2026-09-21).
   Every mask-dependent number in M3 is stale (see M3 below).
2. **Band definition.** M6 says 75 % of the local peak; the main text, Fig 1
   caption and every quoted band number are the 90 % band
   (`band_main = 0.90`, registry `04_band90_*`). The criterion is the smoothed
   event-driven FRACTION, not "event slip".
3. **M7 is the legacy model.** Both tracks, mean of FOUR events (2023
   excluded), `Hayward_DISP-S1/modelling`. Notebook 06 now keeps FIVE events
   and notebook 07 is a placeholder. Nothing in M7 is reproduced by this repo
   yet, and the plan is now a descending-only inversion, so M7 is rewritten
   after 07 is built.
4. **Small numbers that moved** (registry value in brackets): detection
   thresholds 1.56/2.59 mm -> 1.52/2.42 mm; cross-fault rate RMS 0.6 -> 0.7
   (0.66) mm/yr; step bias 0.5 -> 0.4 (0.43); CHP1 overshoot 1.1 -> 1.2 (1.19);
   detrended RMS 1.6-1.8 -> 1.5-1.9 mm; CFW1 2023 window 1.6 -> 1.7 (1.66) mm;
   InSAR step median 9.0 -> 8.8 mm; Fig S2 overlap RMS 1.12 -> 1.18 mm/yr;
   HRKT stack 19 -> 24 crossings; array rates refit over the InSAR window
   (2016-2025), not 2014-2024.
5. **Equations.** S4 as printed has two typos per row (`sin a` should be
   sin θ_a; the `· v_U cos θ` term is added, not multiplied). Delete "the two
   are summed because their senses are opposite" (M2): the terms are the dot
   product of the LOS unit vector with v_FP ŝ + v_U ẑ, no sign argument.

## M1. OPERA DISP-S1 time series and GNSS tie

| Claim | Verdict | Evidence |
|---|---|---|
| DISP-S1 v1.0, 30 m | CORRECT | product names; `02` asserts X_STEP 30 m |
| referenced to the first date of each 15-date mini-stack; 23 asc / 24 desc mini-stacks | CORRECT, H-only | product filenames (344/23, 359/24) |
| 6 to 12 day repeat | UNVERIFIABLE | not computed anywhere |
| frames 09156 + 09157 (track 35), 11115 (track 42) | CORRECT | `TulePond_filepaths.py` |
| 344 acquisitions per ascending frame, 359 descending | CORRECT (09157, 11115); 09156 H-only | `01_n_acquisitions_*` |
| 09157 2016-07-29 to 2025-12-21; 11115 2016-09-03 to 2025-12-15 | CORRECT | `01_first/last_date_*` |
| 09156 2016-07-17 to 2025-11-09 | CORRECT, H-only | `09156/mintpy/timeseries.h5`; not registered |
| velocity fields use both asc frames mosaicked; creep series use 09157 north of 37.49 N | CORRECT | `paths_mosaic`, `insar_dic_list`; 37.49 N from Hayward_filepaths (H-only) |
| rebase to a common reference; SET + ionosphere layers applied | CORRECT, H-only | `Hayward_DISP-S1/analysis/run01` (`disp-s1-reformat --apply-solid-earth-corrections --apply-ionospheric-corrections`). Omitted: the reformat also applies a per-date HIGH_COHERENCE (0.90) spatial reference before ZOA1 |
| TROPO-ZENITH v1.0, ECMWF HRES, 0.0703 deg, per-pixel incidence | CORRECT, H-only | run03; granule spacing 0.0703 deg; "ECMWF HRES" only in the H README |
| recommended mask at 0.5 during accumulation | CORRECT but implicit | opera-utils default (`quality_thresholds=[0.5]`); not set explicitly anywhere |
| only pixels with an unbroken record | CORRECT, H-only | run04 record_mask |
| coherence 0.7 for velocity fields and GNSS tie | CORRECT | `coh_velocity = 0.7` |
| "cross-fault creep time series use a stricter threshold of 0.9" | **WRONG** | `coh_timeseries = 0.7`; `03_n_crossings_common = 609` is the 0.7 stream |
| NGL daily, NA frame; >= 70 % coverage; UNR + visual steps, earthquake steps kept; manual exclusions | CORRECT, H-only | run00. Not stated: steps corrected only if > 3 mm; stations with std(Z) > 1 mm dropped |
| 49 stations, 30 asc, 29 desc, 29 common | CORRECT, H-only | `outputs/validation/gps_tie_stats.txt` |
| >= 10 pixels within 150 m; subsiding > 5 mm/yr excluded | CORRECT, H-only | run05 (the subsidence gate never fired) |
| Eq. S1 and the angle conventions | CORRECT | `insar_utils.los_unit_vector` |
| subtract ZOA1 LOS; shared quadratic + one constant per frame (Eq. S2) | CORRECT, H-only | run05. The sentence after S2 lost its subject ("... and k_f absorbs ..."). The per-frame constants are diagnostic; the output is re-zeroed at ZOA1 with one scalar per track |
| ramp gates: LOO >= 10 % and no overlap degradation, both met | CORRECT with caveat | gate 2 is "spread of per-frame constants <= 0.5 mm/yr" and exists only for the two-frame ascending track; LOO +36 % (asc), +15 % (desc) |
| RMSE 1.6 -> 0.8 asc, 1.0 -> 0.7 desc | CORRECT, H-only | gps_tie_stats: 1.55 -> 0.78, 1.03 -> 0.65 (in-sample; LOO after 0.99 / 0.87) |
| overlap 604,937 px, median -0.02, RMS 1.2 | CORRECT, H-only | figS2_stats: rms 1.175 |
| "applying the tie leaves this RMS unchanged" | UNVERIFIABLE | never computed; true by construction (shared quadratic; constants differ by 0.03 mm/yr) |

Provenance gap: the 09156 dates, mini-stack counts, tropo grid, station counts,
RMSEs and overlap statistics live only in `Hayward_DISP-S1` and
`proj_HaywardCreep/outputs/validation/`. Notebook 01 should register or
fingerprint `gps_tie_stats.txt` and `figS2_stats.txt`. `velocity_notropo.h5`
(Fig S1 D, J) is not produced by the current run04.

## M2. Velocity decomposition

| Claim | Verdict | Evidence |
|---|---|---|
| [NOTE: should not be locked] | Already true | `02` calls `asc_desc2horz_vert` with per-pixel incidence and azimuth (window medians over 20 x 20 px); only the strike is map-wide. CHECK 1 in 02 re-derives 200 pixels to 6e-6 mm/yr. Delete the note; say so |
| mean strike 144 deg | CORRECT | `02_strike_mean_deg = 144.10` |
| negligible fault-perpendicular motion | CORRECT | stated in 02 and `insar_utils` |
| Eq. S3, Psi clockwise from north | CORRECT | `design_matrix_up_fp`; equals MintPy's `sin(inc) cos(azi - horz_az)` with `horz_az = -Psi` |
| "summed because their senses are opposite" | **WRONG rationale** | delete |
| Eq. S4 | **typos** | `sin a` -> sin theta_a; `· v_U` -> `+ v_U`, both rows |
| solved together per pixel by least squares | CORRECT | 2 x 2 pinv per window |
| ZOA1 retained, Fremont stable | CORRECT for products | but the Fig 1A rotated map subtracts the CTM1-COZ1 patch mean for display (02); the caption must say so |
| asc much less sensitive to FP than desc; comparable in vertical | CORRECT, unregistered | FP coefficients -0.283 (asc) / +0.494 (desc); Up 0.762 / 0.720. Local box strike in 06 is 143.05 deg (register or quote) |

## M3. Cross-fault creep time series

| Claim | Verdict | Evidence |
|---|---|---|
| east minus west medians cancel common mode | CORRECT | `03` extract_track |
| 100 m, 79 km, 37.99 to 37.41 N, 791 crossings | CORRECT | executed 03 |
| azimuth smoothed with a 2.0 km rolling median | minor | 21 crossings = 2.1 km (the text later says 2.1 km); use 2.1 |
| 30-150 m, +/- 200 m along strike, >= 5 pixels per side, per-date medians | CORRECT | `dist_min/max`, `x_width`, `min_points` |
| series at coherence 0.9, profile at 0.7 | **STALE** | one stream at 0.7 |
| "0.7 carries 28 % more crossings" | STALE | frozen sweep: 609 vs 480 = 27 %; creepmeter rate RMS 0.72 vs 0.63 mm/yr (`05_coh0.7/0.9_*`) |
| "profile median changes by < 0.05 mm/yr" | UNVERIFIABLE | not computed |
| rate mask: own-side median, 3 robust sigma | CORRECT | `rate_nsigma = 3` |
| sigma 0.44 / 0.46 mm/yr | STALE | `03_rate_mask_sigma_asc/des` = 0.56 / 0.57 |
| cut +/- 1.30 / 1.37 | STALE | 1.68 / 1.72 |
| rejected 3.7 % / 3.6 % | STALE | 6.0 / 5.4 % |
| max rejected 8.1 / 15.6 mm/yr | STALE | 16.1 / 17.1 |
| "(Figure S3)" for the rate mask | WRONG cross-ref | the S3 caption is the acquisition figure; the rate-mask figure (`figS_ratemask_*`) has no caption |
| 344 / 359 acquisitions; 423 common dates; 2016.67-2025.95; 45 d gap limit; 2018 gaps kept | CORRECT | `03_*` keys; gaps 84 d and 72 d |
| 19 % / 15 % interpolated | CORRECT | `03_interp_pct_*` |
| > 90 % before March 2020 | CORRECT, unregistered | recomputed 91 % / 95 % |
| 6-day sampling from Oct 2018 / Jun 2019; 107 same-date epochs, one interpolated; 2020.22-2021.98 | CORRECT | `03_sixday_start_*`, `03_sameday_window_epochs` |
| ascending evening pass ~12 h before descending | UNVERIFIABLE | |
| decomposition per date and crossing with local strike | CORRECT | `design_matrix_up_fp` |
| linear rate; 0.5 km Gaussian | CORRECT | `rate_smooth_km` |
| plotting interpolates along-fault gaps of at most 1 km | UNVERIFIABLE | `04` `interp_nan` bridges all inner gaps with no 1 km limit |
| crossing rejection: 2.1 km running median, 3 robust sigma | CORRECT | `trend_window 21` |
| 54 of 480 (11 %) at 0.9 | STALE | no 0.9 stream |
| 63 of 609 (10 %) at 0.7; excluded from every product | CORRECT | `03_n_landslide_flagged` |

## M4. Creepmeter validation

| Claim | Verdict | Evidence |
|---|---|---|
| five creepmeters; 30 deg obliquity already removed in the published records | CORRECT (as coded) | `read_creepmeter_10min` |
| "robust median filter" for glitches | WRONG wording | `05` daily_creepmeter drops samples with abs deviation >= 6 x 1.4826 MAD + 10 mm (a global robust cut) |
| daily means; OLS slopes over the common window | CORRECT | |
| detrended correlation with the creepmeter sampled at SAR epochs | CORRECT | |
| step recovery over the 2019 and 2022 COZ1 events | CORRECT method, wrong stream | computed only in the frozen coherence-0.9 aperture ladder with a fixed 4-row stack (`05_ladder_selected_coz1_step_pct = 45`), not on the 0.7 validation series |
| velocity-field step within +/- 1 km and +/- 200 m; RMS 1.0 mm/yr; four of five low | CORRECT | `05_step_rms_mm_yr = 1.01` |
| average offset 0.5 mm/yr | STALE | `05_step_bias_mm_yr = -0.43` |
| CHP1 overshoot 1.1 mm/yr; maximum offset near 300 m | STALE / UNVERIFIABLE | +1.19; the 300 m is not computed |
| cross-fault series RMS 0.6 mm/yr | STALE | 0.66 |
| 0.3 excluding COZ1; none > 0.5; COZ1 1.3 low | CORRECT | `05_rate_rms_excl_coz1 = 0.32`; 1.33 |
| 2018 gaps before the first COZ1 event | CORRECT | gaps end 2018.79, event 2018.997 |
| detrended residuals 1.6-1.8 mm | STALE | 1.47-1.86 |
| COZ1 r = 0.6 | CORRECT | 0.58 |
| CHP1 annual modulation 1-2 mm in both records | UNVERIFIABLE | no key; `05_CHP1_det_r = -0.18` (anti-correlated) |
| noise floor ~1.7 mm | CORRECT | 1.68 |

## M5. Event detection and temporal model

| Claim | Verdict | Evidence |
|---|---|---|
| mean of following three minus preceding three | CORRECT | `step_statistic`, `win_n = 3` (evaluated at each inter-acquisition gap) |
| 18 d / 36 d; 12-day repeat after December 2021 | CORRECT | grid spacing |
| positive only; 3 robust sigma over the full record | CORRECT | `k_sigma = 3` |
| thresholds 1.56 mm Oakland, 2.59 mm Tule Pond | **STALE** | `04_threshold_mm = 2.42`; Oakland 3 x 0.505 = 1.52 (not registered) |
| Eq. S5 solved jointly | CORRECT | `fit_joint` |
| onset grid search over epochs; interval = preceding acquisition to latest epoch within one variance unit | CORRECT | two passes event by event (not stated) |
| moving-block bootstrap, one-year blocks, 2000 refits, 95 % CI, onsets fixed | CORRECT | block = 61 samples (1 yr at 6 d; ~2 yr at 12 d) |
| refit at every crossing | CORRECT | |
| per-pixel LOS solve and burst fix | CORRECT in 06, wording wrong | offsets are measured per EVENT step map in +/- 0.0015 deg bands beyond 400 m from the trace (06), not "at every epoch"; values -0.95 to +0.92 mm |

## M6. Section extent, band stack and slip budget

| Claim | Verdict | Evidence |
|---|---|---|
| band = 75 % of local peak | **WRONG** | `band_main = 0.90`; `section_lengths.csv` has both (band75 57.8-60.5 km n = 28; band90 58.2-59.8 km n = 17); the paper quotes band90 |
| criterion "event slip" (main text, Fig 1) | wording | the smoothed event-driven FRACTION (summed event slip / total) >= 90 % of its peak |
| contiguous run, short gaps bridged; refit with catalogue onsets | CORRECT | |
| total rate = inter-event + summed event slip / record length | CORRECT | span 9.28 yr |
| arrays refit over 2014 to 2024 | **WRONG** | refit over the InSAR window 2016.67-2025.95, >= 4 surveys spanning >= 5 yr |
| how the ~6 km length is defined | MISSING | cumulative event slip above flank median + 3 robust sigma (6.10 mm; flanks 52-56 / 64-68 km); 6.1 / 5.3 km at 10 / 20 % of peak |

Main-text numbers backed by 04: onset windows (all five) CORRECT; 10.3 mm,
7.6-11.8 CORRECT; inter-event < 1 mm/yr CORRECT (0.94 stack, 0.83
per-crossing); 6.5 vs 6.3 mm/yr CORRECT but say "whole-record array rate
averaged over the 6 km section (n = 4)"; window-refit array rates are
6.9-7.0; < 0.4 mm/yr N and S unregistered (block means 0.22 / 0.03, smoothed
profile reaches 0.8 at the section edges); ~6 km CORRECT; 1.6 km CORRECT;
CFW1 3.8 km south CORRECT (computed from the 58.8 km peak); CFW1 1.6 mm ->
1.7 mm (1.66, with 46 % data coverage in the window). Fig 1 caption
placeholders: band 58.2 to 59.8 km; record span 9.3 yr; n = 17; steps sum
52 mm; secular 0.9 mm/yr.

Housekeeping in 04: stale "75 % band" comments at lines ~469, 529, 671 and
the stats line "0 repeaters in the 75% band" (computed on the 90 % band).

## M7. Slip inversion, ensemble and depth (all LEGACY-ONLY)

Legacy = both tracks, mean of four events, `Hayward_DISP-S1/modelling`.
Notebook 07 is a placeholder; 06 now writes a FIVE-event mean. Everything
below is what the legacy code did, for when 07 is rebuilt.

| Claim | Verdict | Evidence |
|---|---|---|
| CSI (Jolivet et al., 2014) | citation | legacy README says cite the CSI Zenodo record and Jolivet et al. 2023 |
| meshing with XXx | fill | scipy Delaunay + Laplacian smoothing in `csi_io.load_fault_tri`; 1237 elements |
| quadtree 2 km to a single 30 m pixel | values off | START 2.0 km, MIN 0.04 km; the pixel within 30 m of the trace is dropped; realised blocks 31-2003 m |
| 2,594 / 2,563 samples | CORRECT for the legacy 4-event, swath 4 km, 52-67 km run | not valid for a rebuild |
| semivariogram nugget + two exponentials, nugget / sqrt(n) | CORRECT | asc nugget 0.52 mm, short 0.42 mm / 1.84 km; desc nugget 0.52, long 0.48 mm / 3.0 km. Rousset 2016 is cited in the legacy notes for one ramp per scene, not the semivariogram |
| mesh 100 m to 0.5 km, ~1 km at 3 km | CORRECT | h_top 0.1, const_to 0.5, growth 1.6 |
| 50 m tested | weak | tri50m runs exist only for an early configuration |
| Meade 2007, nu 0.25, vertical; dip-slip / opening / 70 deg no better | CORRECT | m03, m02 --dip 70 |
| prior lengths and sigma | AMBIGUOUS | README/m13/m14: lam_x 0.3, lam_z 0.05 km, sigma 16 mm; RESULTS section 13 and `fig_slip_model.py`: lam 0.3 x 0.3, sigma 16 (peak 10.7 mm, Mw 3.97). Bootstrap and stress drop exist only at lam_z 0.05 |
| surface constraint one point per 200 m, 0.5 mm floor, "same four events" | WRONG detail | `csi_io.load_surface_slip` averages all FIVE onsets |
| depth scan 57.5-61.5 km, dchi2 <= 4, 0.8 km (0.6-1.0) | 0.8 CORRECT; range WRONG for the canonical swath-4 run | dchi2 <= 4 only at 0.8 and 1.1 km; 0.6-1.0 holds for the older swath-2 run. Asc-only best 0.3 km, desc-only 1.25-1.5 km (relevant to a descending-only 07) |
| unchanged with constant / no ramp, 2-6 km swath, three sampling schemes | CORRECT | |
| 24 noise realisations; synthetic recovery; 115 m smoothing; null 2.3 mm | CORRECT | m13, m04, m15, m01 --null |
| mu 30 GPa; rigidity 3-10 GPa | CORRECT-ish | legacy notes say 3-8 GPa |
| stress drop slip-weighted (Ripperger and Mai 2004) | WRONG attribution | slip-weighted = Noda et al. 2013 (as in the assumptions register); Ripperger and Mai 2004 is the area-weighted >= 20 % rule. Legacy uncertainty is the smoothing sweep 0.15-0.6 MPa, not slab vs regularised |

## Off-fault deformation paragraph, Fig 2, Fig 5 and Fig S5 (notebook 06)

- Per-pixel linear + seasonal + one step per event: CORRECT.
- "mean strike of the Tule Pond segment": CORRECT, 143.05 deg (`06_strike_local_deg`); the whole-fault 144.1 is 02's.
- Smooth rate change at X-X' vs sharp step at CFW1; arctangent event at X-X', none at Y-Y': CORRECT (rate 1.49 vs 6.16 mm/yr; event 8.41 vs 0.75 mm, all fault-parallel).
- "slip confined to the uppermost kilometre (Fig ??G)": LEGACY-ONLY (depth scan); 06 fits no model.
- Stale registry rows `06_XX_event_depth_m`, `06_*_rate_DC_m`, `06_joint_*` ("git nogit", 2026-09-21) come from a removed profile-fit cell; purge them from `registry.csv`.
- Fig 5 caption "(B) Mean event displacement (four events)": STALE, five.
- Fig S5 caption: 3 x 7 layout described with six named columns; the current `figS_event_step_maps` is 2 rows (asc, desc burst-fixed) x 7 (five events, mean, scatter). The raw descending row is saved (`event{i}_raw`) but not drawn.
- Fig 2 panel references (??A-G) now map to the new layout: A, B rate maps; C, D rate profiles; E, F event maps; G, H event profiles; I, J, K the model row (07).
- New numbers from 06 for the asymmetry text (S2): X-X' LOS event steps asc -1.92 mm, desc +4.69 mm (ratio -0.41 against the geometric -0.60). The mean near-fault steps (asc -1.77, desc +4.56 mm) are fitted exactly by strike-slip 8.0 mm + east side up 0.8 mm, or by strike-slip 12.2 mm + east block contraction 3.2 mm (`06_asym_*`). The two-block forward model lives in `proj_HaywardCreep/TwoBlock_LOS_test` (figure `figS_twoblock_asymmetry_paper.png`), outside this repo.

## M8, page 6 and Fig 3 (notebook 08)

| Claim | Verdict | Evidence |
|---|---|---|
| HRKT 1979.7-2024.7; ~bimonthly to mid-2007, annual from 2008 | CORRECT | `08_hrkt_dense_era_end_yr = 2007.84` is a hard-coded literal in 08, not computed |
| 3.4 mm/yr trend; 1.7 mm seasonal | CORRECT | joint-model inter-event rate 3.41 (whole-record slope 5.73) |
| 6 mm (2.3 sigma) threshold; 14 events | CORRECT | |
| nine events 1979-2007 at 0.9-4.9 yr | CORRECT | |
| step sizes 5-11 mm | WRONG | 0.5-11.0 mm (1988.35 = 0.5 mm, 1993.27 = 4.6 mm); seven of nine are 5.4-11.0 |
| brackets 2017, 2021, 2023 | CORRECT | |
| misses 2019 and 2025, just below threshold | half WRONG | 2019 yes (excess 4.7 mm); 2024.98 is not bracketed at all (last survey 2024.7) |
| 2.8 yr (n = 13), 7.4 mm (n = 14), 1.9 yr (n = 4) | CORRECT | |
| 9.0 mm (n = 5) | WRONG | `08_insar_step_median_mm = 8.8` (not anchored; add) |
| no step > 6 mm 1988-1993; 1993.3 event | CORRECT | max 3.7 mm over 30 surveys |
| "Figure 5a-d" references | STALE numbering | these are notebook 08's Fig 3 a-d; the manuscript Fig 3 caption lacks panels c and d |
| Fig 3 caption "19 crossings within 1 km" | WRONG | HRKT stack = 24 (19 is HPIN); "Parkmont" -> Tule Pond |
| title "forced-period test" | NOT PORTED | run17/run19 comb tests are not in 08 (assumptions register row 36) |

## M9 and Fig 4 (notebook 09)

Fig 4 labels match 09 (A InSAR 2017-25, B array 1983-2006, B' excl. Loma
Prieta, C merged 1983-2025; wet band 2026-10-01 to 2027-04-30). The Fig 8
caption (onset day-of-water-year; renewal distributions; prediction
statement) does not match any single 09 figure. Draft M9 from what 09 does:

Daily precipitation is a day-by-day composite of the NOAA GHCN-daily gauges
Fremont (USC00043244), Newark (USC00046144) and Hayward Air Terminal
(USW00093228), first reporting gauge in that order, the morning-observer
Newark record shifted one day earlier. The series runs from 1978-10-01 to
the pinned issue date 2026-08-25; water years are named by the year they
end (45 complete years, median 362 mm, 10-90 % 219-567 mm). Onset
seasonality uses the season-resolved onsets: the five InSAR windows (5-12 d)
and the nine HRKT windows of 130 d or less. The renewal forecast fits
exponential, lognormal, Weibull, BPT and Gaussian intervals to four
catalogues (A InSAR, 4 intervals; B dense-era array 1983-2006, 8; B' without
the Loma Prieta interval, 7; C merged, 15), drawing onsets uniformly within
their windows (400 draws) with a parametric bootstrap (5 refits), conditioned
on no event by the reference date, families compared by AICc; reports the
5-50-95 % onset dates and the probability of onset in the WY2027 wet season
and by end-2027. The InSAR clock in closed form: mean 1.97 yr, sd 30 d (95 %
CI 17-113 d). Rainfall conditioning scores three rules over each onset
window (cumulative threshold Y 50-400 mm; first 3-day storm >= Z 15-75 mm;
a 3-day storm >= Z 20-50 mm within X 7-30 d) against the December-March base
rate and the chance of all five InSAR windows being hit (`09_ruleC_*`:
40 mm, 21 d, p = 0.02).

## Figure captions S1 to S4

- S1: matches `figS1_processing.py` (H-only); D, J need `velocity_notropo.h5`, which the current run04 does not write.
- S2: matches `figS2_gnss_tie.py`; RMS 1.12 -> 1.18 mm/yr.
- S3 (acquisitions): CORRECT; it is notebook 03's "Fig S4".
- S4 (validation): ONE bracket (Tule Pond), not two; gold bands are the InSAR detection windows drawn in I/N (COZ1 events) and K/P (Tule Pond events), not creepmeter transient times.
