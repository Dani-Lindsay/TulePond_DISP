# Assumptions register, Tule Pond letter (started 2026-09-16)

Every choice a reviewer could attack, where it is set, what justifies it, and
whether that justification is a test in the repo (T), a sweep (S), a
citation (C), or nothing yet (NONE). Values read from the Hayward_DISP-S1
scripts on 2026-09-16; they carry over to TulePond_DISP unless changed.

| # | Assumption | Value | Where set | Justification | Status |
|---|---|---|---|---|---|
| 1 | Creep chain frames | asc 09157 + desc 11115 only | paths | 09156 tops out north; south of 37.489 N the fault splays | C (v0.2 Methods) |
| 2 | Cross-fault aperture | 30 to 150 m each side | run08 DIST_MIN/DIST_MAX | scored on creepmeter RMS in the bin sweep | S (run11, Fig S5) |
| 3 | Along-strike half-width per crossing | 200 m | run08 X_WIDTH | NOT swept | NONE |
| 4 | Min pixels per side | 5 | run08 MIN_POINTS | a 60 m bin at 30 m pixels | NONE (state it) |
| 5 | Temporal-coherence mask | 0.9 (series), 0.7 (maps) | run08 COH_THRESH | same pixels valid at every epoch | S (coh sweep) |
| 6 | Near-fault rate mask | 3 robust sigma from own-side median | run08 RATE_NSIGMA | removes landslide edges without an absolute cap | C (Fig S3) |
| 7 | Strike smoothing | 21 crossings (2.1 km rolling median of `profile_az`) | `TulePond_filepaths.az_smooth_n`, used in 02 and 03 | removes the 100 m digitising kinks (26 crossings > 10 deg off), keeps the real bends. At Tule Pond, raw vs smoothed: band rates differ by ~0.2 mm/yr, event steps by ~0.1 mm, section ends unchanged (re-extraction 2026-09-23; figure `qc03_profile_azimuth_*`) | S |
| 8 | Interpolation gap limit | 45 d | run09 MAX_INTERP_GAP_DAYS | bridges 6/12/24 d sampling and the 2018 gaps | NONE (show gap map, Fig S4) |
| 9 | Along-fault crossing rejection | 3 robust sigma from 21-crossing running median | run09 TREND_NSIGMA | landslide-dominated crossings | NONE |
| 10 | Rate profile smoothing | 0.5 km Gaussian | run09 RATE_SMOOTH_KM | | NONE |
| 11 | LOS to FP/Up design matrix | product azimuth convention, asc/desc | insar_utils | verified against product docs | T (unit test exists? confirm) |
| 12 | Detection stream | narrow bin | run12 STREAM | robust sigma 0.86 vs 1.26 mm; 5/5 events for K 3.0 to 4.0 | S (documented in run12) |
| 13 | Stack half-width for detection | +/- 2 km of segment centre | run12 STACK_HALF_KM | | NONE |
| 14 | Matched filter window | 3 epochs each side | run12 WIN_N | 18 d at 6 d repeat | NONE (sweep it) |
| 15 | Detection threshold | 3 robust sigma (1.4826 MAD) | run12 K_SIGMA | 5/5 for K 3 to 4 | S |
| 16 | Events are instantaneous steps | ramp rejected | run12 docstring | residuals not independent; 6 to 12 d sampling | C (Methods) |
| 17 | Onset window | the acquisition interval, uniform inside it | run12/run21 | | C (feedback rule) |
| 18 | Secular model | linear + annual per crossing | run12/run14 | | NONE |
| 19 | Event-fraction definition | sum A / (bT + sum A), unbounded | run14 | stated as a regression ratio | C |
| 20 | Fraction smoothing | 0.4 km | run14 SMOOTH_KM | | NONE |
| 21 | Band definition | >= 75% of local peak event slip | fig_event_summary | Dani's decision 2026-09-16 | NONE (define once) |
| 22 | Section extent | where event slip departs from secular (threshold TBD) | task J | | NONE |
| 23 | Mean event | mean of 4 (2023 excluded) | m01 EXCLUDE / m02 | 2023 same-sign east-side signal in both tracks | C (Methods); T? |
| 24 | Descending burst offsets | removed as far-field steps | m01 BURST_FIX | | NONE (show before/after) |
| 25 | Fault geometry | vertical, mapped trace | m02 | 70 deg dip does not improve fit | S (m02) |
| 26 | Slip components | right-lateral only | m02 | dip-slip/opening no improvement; data vertical <10% | S |
| 27 | Elastic model | homogeneous half-space, nu 0.25, mu 30 GPa | m02/m14 | rigidity ladder reported | C; task E (contrast) |
| 28 | Downsampling | distance-based quadtree, 2 km to 30 m | m02 | Lohman & Simons 2005 | C |
| 29 | Data covariance | nugget + 2 exponentials, far field | m02 | Rousset 2016 | C |
| 30 | Prior | Gaussian, lam_x 0.3 km, lam_z 0.05 km, sigma 16 | m02 | chi2/N ~ 1.2; the honest range is the sweep | S (task A replaces) |
| 31 | Surface constraint | band profile every 200 m, 0.5 mm floor | m02 | near-fault, same pixels | NONE (test with/without: m15 says removing it deepens the peak) |
| 32 | Depth extent | uniform slab scan, dchi2 <= 4 | m05 | | S; task B extends |
| 33 | Top 300 m unresolved | 115 m product smoothing | m15 | | T (synthetic) |
| 34 | Stress drop definition | slip-weighted ds_E (Noda 2013) | m14 | | C |
| 35 | AA detector | 6 mm excess per interval | run18 | 2.3 robust sigma | S (threshold sweep figS) |
| 36 | AA forced period null | 300 comb-free bootstrap series | run17 | | T |
| 37 | Renewal models | exp/lognormal/Weibull/BPT/Gaussian, AICc | run21 | family irrelevant at n = 4 | C |
| 38 | Wet season | Oct 1 to Apr 30 | run21 WET | | NONE (say it) |
| 39 | Loma Prieta exclusion | intervals spanning 1989.80 dropped (variant B') | run21 | Lienkaemper 1997 | C |
| 40 | Rain gauge | composite Fremont/Newark/Hayward, COOP obs-time shift | run20 | no hills gauge exists | C (memory pitfall) |
| 41 | Storm rule | >= 40 mm/3 d within 21 d | run22 | tuned on 5 events; hypothesis | C (stated) |
| 42 | Fault trace | 698 hand-placed points (KML 2026-07-24, one strand, from Qfaults + the InSAR velocity steps), resampled every 100 m by arc length in UTM, cross-fault azimuth = tangent bearing - 90 | notebook 01 | the trace follows the velocity step, not the Qfaults line; the 100 m file is regenerated from the KML and asserted identical to the `Inputs/` copy | T (01) |
