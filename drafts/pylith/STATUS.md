# STATUS (daytime run, 2026-09-30)

Updated: 2026-09-30 20:25

## Now (20:25): ALL DONE, nothing running. Task 6 = coherence-0.7 sensitivity (the gap is Lake Elizabeth, Central Park)

Method: 06's field rebuilt at map coherence 0.7 with 06's own code (at 0.9 the same code reproduces 06's fp_event
exactly; outputs/06 and the registry untouched): 201,620 pixels vs 149,115. 07's quadtree on it: 3148 samples (2386).
PyLith GFs at those samples for SFCVM aligned and homog (vertical, default mesh, order 1), inverted with nugget and full Cd.

| run | medium | sigma | surface slip km 60.6-61.2 mean / max (mm) | rms / W / E, own samples | rms / W / E on the SAME 2386 coh-0.9 samples | surf % | Mw30 | Mw_sfcvm |
|---|---|---|---|---|---|---|---|---|
| coh 0.9, nugget (SELECTED) | sfcvm | 12 | 10.2 / 12.0 | 1.36 / 1.50 / 1.39 | 1.35 / 1.48 / 1.37 | 98 | 3.76 | 3.04 |
| coh 0.7, nugget | sfcvm | 12 | 10.1 / 11.4 | 1.71 / 1.80 / 1.83 | 1.35 / 1.48 / 1.35 | 95 | 3.77 | 3.12 |
| coh 0.9, full | sfcvm | 12 | 7.1 / 8.1 | 1.72 / 2.22 / 1.28 | 1.65 / 2.06 / 1.33 | 74 | 3.55 | 2.84 |
| coh 0.7, full | sfcvm | 12 | 6.4 / 7.6 | 2.03 / 2.48 / 1.70 | 1.74 / 2.20 / 1.35 | 69 | 3.49 | 2.75 |
| coh 0.9, nugget | homog | 12 | 9.5 / 11.2 | 1.66 / 1.84 / 1.73 | 1.62 / 1.77 / 1.66 | 90 | 3.70 | 2.96 |
| coh 0.7, nugget | homog | 8 | 8.6 / 9.6 | 1.99 / 2.19 / 2.05 | 1.67 / 1.91 / 1.60 | 86 | 3.68 | 3.00 |
| coh 0.9, full | homog | 12 | 7.2 / 8.2 | 1.98 / 2.52 / 1.62 | 1.83 / 2.24 / 1.54 | 75 | 3.51 | 2.69 |
| coh 0.7, full | homog | 12 | 6.4 / 7.8 | 2.24 / 2.75 / 1.92 | 1.90 / 2.39 / 1.52 | 68 | 3.45 | 2.61 |

Crossings at km 60.6-61.2: 6, median 8.0 mm (7.4-9.5). The surface constraint has NO running-median point there (it is
interpolated between km 60.5, 9.1 mm, and km 61.5, 5.0 mm). "Own samples" rms is not comparable across coherences
(different, noisier sample set); the common-sample columns are (one mean constant removed).

Answer, plainly:
- The spike is NOT caused by the east-side gap. Filling it (east samples within 300 m in km 60.2-60.8: 1 at coh 0.9,
  16 at coh 0.7) leaves the selected setup's surface slip there at 10.1 / 11.4 mm (was 10.2 / 12.0), still ~2-4 mm
  above the crossings. It goes with the nugget Cd, which weights the near-trace west samples strongly and has no
  surface-constraint point in that stretch; under the full Cd the surface slip there is 6.4-7.2 mm at either coherence.
- The coh-0.7 model is equivalent to the selected one on the common samples (1.35 / 1.48 / 1.35 vs 1.35 / 1.48 / 1.37),
  with Mw30 3.77 vs 3.76 and Mw_sfcvm 3.12 vs 3.04 (a little more slip at depth, where the SFCVM rigidity is higher).
  The selected model is robust to the coherence choice.
- Lake Elizabeth strip (east of the trace 0-200 m, km 60.2-60.8; coh07/strip_stats.csv): coh 0.9 asc 2 / desc 4
  pixels; coh 0.7 asc 27 / desc 35 / both-track 21. Median implied fault-parallel: asc -2.0 mm (16-84 %: -8.0 to +0.3),
  desc +4.8 mm (1.6 to 6.3), decomposed +2.0 mm (-3.0 to +3.8). The ascending pixels have the wrong sign for the east
  block, so the strip at 0.7 is partly noise (descending plausible). 3 of the 16 new east samples lie within 40 m of
  the trace and carry west-block values (-4.6 to -6.7 mm): trace location off by tens of metres there.
- Fig-2 product: gf/fig2_model_row_pylith_coh07_nuggetcd.npz: sigma 12, rms 1.71 (W 1.80 / E 1.83) on its 3148
  samples, surface 95 %, Mw30 3.77 (range 3.72-3.80 over the neighbouring sigmas; no bootstrap or alignment runs for
  this sample set), Mw_sfcvm 3.12 (mu_eff 3.12 GPa), X-X' fp west -7.67 / east +2.76 mm, forward = G x slip exactly.
- Table: gf/final_comparison.csv (41 rows); figures/summary_all_tests.png (40 runs; coh07 tags added to the ladder);
  one-pagers figures/summary_default_coh07{,_nuggetcd}_{homog,sfcvm}.png; coh07/compare_coh.csv.

## Earlier today: ALL DONE by 19:10 (tasks 1-5)
- SELECTED model (Dani): default_nuggetcd / sfcvm. Primary companion: default / sfcvm (07's full Cd).
- Done since 17:35: uncertainty + recovery + depth scan (section below), alignment sensitivity (below), Fig-2
  products for scripts/07_slip_model_fig2.py, final table refreshed (37 rows, gf/final_comparison.csv),
  figures/summary_all_tests.png redrawn (36 runs). 80W and vertical-control runs not done (cancelled).
- Fig-2 products (each from one PyLith forward run of the inverted slip; forward = G x slip at the samples, 0.000 mm rms):
  - gf/fig2_model_row_pylith_nuggetcd.npz (SELECTED): sigma 12, rms 1.36 (W 1.50 / E 1.39), surface 98 +/- 3 %,
    Mw30 3.76 +/- 0.02, Mw_sfcvm 3.04 +/- 0.07 (mu_eff 2.54 GPa), X-X' fp west -7.88 / east +2.77 mm.
  - gf/fig2_model_row_pylith_full.npz (primary): sigma 12, rms 1.72 (W 2.22 / E 1.28), surface 74 +/- 4 %,
    Mw30 3.55 +/- 0.03, Mw_sfcvm 2.84 +/- 0.11 (mu_eff 2.53 GPa), X-X' fp west -7.48 / east +1.89 mm.
  - NB for 07_slip_model_fig2.py: the fault is given per VERTEX (nodes_kmz, faces, slip_vertex_mm, s0_km_pp = 0),
    not per face as in 07's file; panel L needs a tripcolor(gouraud) on the vertices instead of face colours.

## Alignment sensitivity (SFCVM boundary moved 50 m east / west of the trace; vertical, default mesh; own L-curve pick)

| boundary | Cd | sigma | rms | W300 | E300 | chi2 | surf % | potency m3 | Mw30 | Mw_sfcvm |
|---|---|---|---|---|---|---|---|---|---|---|
| on trace | full | 12 | 1.72 | 2.22 | 1.28 | 3954 | 74 | 8,973 | 3.55 | 2.84 |
| 50 m E | full | 12 | 1.81 | 2.35 | 1.34 | 4051 | 74 | 9,712 | 3.58 | 2.85 |
| 50 m W | full | 12 | 1.67 | 2.09 | 1.33 | 3835 | 75 | 7,152 | 3.49 | 2.75 |
| on trace | nugget | 12 | 1.36 | 1.50 | 1.39 | 13,296 | 98 | 18,073 | 3.76 | 3.04 |
| 50 m E | nugget | 16 | 1.35 | 1.42 | 1.45 | 12,703 | 93 | 21,686 | 3.81 | 3.06 |
| 50 m W | nugget | 8 | 1.38 | 1.54 | 1.37 | 13,968 | 98 | 13,037 | 3.66 | 2.97 |

- A 50 m shift of the contrast moves Mw30 by -0.10 / +0.05 (selected) and -0.06 / +0.03 (primary), Mw_sfcvm by
  -0.07 / +0.02: as large as the sigma choice and 3-5 x the bootstrap spread. For the nugget runs most of it comes
  through the L-curve pick (sigma 8 / 12 / 16).
- The two covariances disagree on the best position (full Cd prefers 50 m W, chi2 -119; nugget prefers 50 m E,
  chi2 -593), so the data give no robust preference: consistent with treating the offset as mapping / CVM error and
  keeping the boundary on the trace. Fit to the west lobe changes by <= 0.13 mm.
- Suggested Mw quote for the selected model with this included: Mw30 3.76 (3.66-3.81 over sigma and +/- 50 m
  alignment; bootstrap +/- 0.02), Mw with SFCVM rigidity 3.04 (2.97-3.06).

## Uncertainty and resolution of the selected model (pylith_sfcvm_tests.py, 18:15)

Selected = default_nuggetcd / sfcvm (vertical, aligned SFCVM, nugget-only Cd, surface ON, sigma 12). Primary = default /
sfcvm (07's full correlated Cd). Tables: gf/uncertainty_*.csv, gf/recovery_*.csv, gf/depth_scan_*.csv, gf/tests_summary.txt.
Figures: figures/uncertainty_bootstrap.png, recovery_synthetic.png, depth_scan.png.

| | selected (sfcvm, nugget) | primary (sfcvm, full) | homog, nugget | homog, full |
|---|---|---|---|---|
| Mw30 (model) | 3.76 | 3.55 | 3.70 | 3.51 |
| bootstrap mean +/- 2 sd (24) | 3.74 +/- 0.02 | 3.41 +/- 0.03 | 3.72 +/- 0.02 | 3.40 +/- 0.02 |
| Mw30 at sigma 8 / 16 | 3.70 / 3.79 | 3.46 / 3.62 | 3.66 / 3.71 | 3.43 / 3.57 |
| Mw(SFCVM mu) +/- 2 sd | 3.04 +/- 0.07 | 2.84 +/- 0.11 | 2.96 +/- 0.07 | 2.69 +/- 0.06 |
| surface % +/- 2 sd | 98 +/- 3 | 74 +/- 4 | 90 +/- 3 | 75 +/- 4 |
| recovery: moment out/in, D 0.3 -> 2 km | 1.09 -> 0.68 | 0.56 -> 0.12 | 1.15 -> 0.54 | 0.52 -> 0.09 |
| recovery: Mw out for Mw in 3.74-4.29 | 3.77-4.18 | 3.58-3.67 | 3.78-4.11 | 3.55-3.61 |
| recovery: mean W / E residual within 300 m | -0.11..-0.02 / +0.16..+0.29 mm | -0.42..+0.06 / +0.57..+1.58 | -0.15..+0.02 / +0.07..+0.13 | -0.92..-0.01 / +0.47..+1.43 |
| depth scan (uniform slip to D): best, bracket | 0.4 km, 0.4 only (Mw 3.80) | 0.85 km, 0.7-1.25 km (Mw 3.84-4.01) | 0.25 km only (3.64) | 0.85 km, 0.6-1.5 km (3.79-4.05) |

What this says, plainly:
- Bootstrap spreads are tiny (Mw +/- 0.02-0.03). They are NOT the real uncertainty: the choice of sigma moves Mw by
  -0.06/+0.03 (selected), the choice of Cd by 0.21 (3.55 vs 3.76).
- Under 07's full Cd the setup cannot resolve depth: whatever the input (10 mm to 0.3-2 km), it returns slip in the
  top ~0.4-0.5 km and Mw 3.58-3.67, and its own bootstrap mean sits 0.14 below the model. 07's CSI tests show the same
  (recovery 0.70 -> 0.12, Mw 3.63-3.69; CSI bootstrap mean 3.49 vs 3.57). So the primary model's Mw 3.55 is a LOWER
  BOUND, and its fixed-depth scan (Mw 3.84-4.01, D 0.7-1.25 km; CSI: 3.82-4.00, 0.65-1.25 km) says the data allow more.
- Under the nugget Cd (selected) recovery is good: moment within -32/+9 % and Mw within 0.11 up to D = 2 km, depth
  extent tracked (smeared ~0.3-0.5 km deeper than the input), bootstrap unbiased (3.74 vs 3.76).
- Artefact check: the recovery residual shows NO west-side excess for the selected setup (west mean -0.02 to -0.11 mm
  against a noise sd of 0.71 mm; east slightly positive, +0.2 mm). The inversion does not manufacture a west lobe,
  so the west lobe in the data is real signal, fitted here by the SFCVM contrast.
- Caveat on the nugget depth scan: its delta chi2 <= 4 bracket collapses to a single depth even after rescaling by
  the reduced chi2 (7.1), because a diagonal Cd treats the 2386 samples as independent. Use recovery (above) for the
  selected model's depth resolution, not that bracket.
- Suggested quote for the selected model: Mw30 3.76 (3.70-3.79 over the neighbouring sigmas; bootstrap +/- 0.02),
  Mw 3.04 +/- 0.07 with the SFCVM rigidity, surface slip 98 +/- 3 % of the crossings; with 07's full Cd the same
  medium gives Mw30 3.55, a lower bound by the recovery test.
- Alignment sensitivity: see the section above (adds -0.10 / +0.05 to the selected Mw30).

## Order-2 (Charles Williams' advice; slip impulses stay order 1)
- Timing test (default mesh, P2 displacement + Lagrange, quadrature 2, homog, 2 ranks while the 8-rank dip job runs):
  12.7 s per impulse (order 1 with 4 ranks alone: 1.45 s; with 8 ranks: 0.95 s). Setup ~5 min (P2 point location).
  Estimate with 8 ranks alone: ~4-5 s per impulse -> 2105 impulses ~2.5-3 h PER MEDIUM.
- Plan: after the 70E runs (~11:35) the queue runs order-2 SFCVM (no nu cap, materials order 1) first (~14:30), then
  order-2 homog (~17:30), then the 80W sensitivity and the vertical control only if the day allows. If order-2 homog
  cannot finish, the order-2 SFCVM is compared against order-1 homog and flagged.
- 20-impulse production-setup test (order2.cfg + no-cap SFCVM): first attempt failed at setup, 'could not find values
  for density at y = -50000 m': with materials of order 1 the db is queried at boundary VERTICES, exactly on the
  grid's last node. Grid extended to +/-51 km (z to -41 km), db rebuilt; the relaunched test RUNS: GMRES 23-26
  iterations per impulse with the uncapped sediment (order 1 capped: 14-17), ~12 s per impulse on 2 contended ranks,
  the same as order-2 homog. Production runs stay queued (order-2 SFCVM no-cap first, then order-2 homog), BUT:
- CAUTION on the same 33 test impulses: ||G|| order-2/no-cap/materials-order-1 is 0.65 x the order-1/capped/order-0
  G (surface impulses 0.60 x, buried 0.90 x; rms difference 67 %). That is far too big for a locking correction of a
  strike-slip source (nu hardly matters in antiplane shear). The likely cause is MATERIALS ORDER 1: properties are
  queried at the vertices, so on the fault (where the aligned contrast sits) both sides get the mid value and the
  sharp contrast is smeared over ~150 m. Three 2-rank tests are running to separate the effects (P2 + materials
  order 0 + no cap; P1 + order 0 + no cap; P2 homog vs P1 homog).
  - 11:12 P2 + materials order 0 + no cap: ratio 0.62 (surface 0.59) -> the material order is NOT the cause.
    Remaining suspects: the nu cap itself (nu 0.40 -> ~0.5 changes the sediment response) or P1 under-resolving the
    thin soft layer. Charles' spec (materials order 1) stays for production.
  - 11:20 P1 + order 0 + no cap: 1.13 x the P1 capped G (the cap alone is a +13 % effect at P1). P2 vs P1 on the
    SAME no-cap medium: 0.55 (rms difference 71 %). So P1 and P2 DISAGREE by ~45 % for the SFCVM medium, and the P2
    SFCVM response is 0.8 x homogeneous where P1 gave 1.27 x: the sign of the "soft sediment amplifies" effect flips.
    Until this is understood the order-2 SFCVM result is NOT validated. Running: P2 homog vs P1 homog (P1 homog is
    within 4 % of the analytic/CSI checks), then P2 + order 0 + capped.
  - 11:30 P2 HOMOG vs P1 homog: ||G|| ratio 0.71 (surface impulses 0.63, buried 1.00; rms difference 59 %, mostly
    at samples 0.3-1 km from the trace). P1 homog is validated against CSI/analytic to 4 %, so with the present
    setup the P2 Green's functions disagree with a validated solution for a homogeneous half-space. The order-2
    PRODUCTION RUNS ARE ON HOLD (queue stopped) until a P2 forward (kinematic) run of the validation patch is
    compared with the P1 forward run and CSI (running now, ~5 min). If P2 cannot be validated today, the order-2
    rows will be reported as "not run: P2 GreensFns not validated", with these test numbers.
  - 11:38 all five variants on the 33 test impulses (||G|| relative to P1 capped | relative to HOMOG of the same order):
    P1 capped 1.00 | 1.27;  P1 no-cap 1.13 | 1.43;  P2 capped 0.61 | 1.09;  P2 no-cap 0.62 | 1.11;  P2 no-cap mat1 0.65 | 1.16.
    So RELATIVE to homogeneous of the same order the SFCVM amplification is present at both orders (P1 1.27-1.43,
    P2 1.09-1.16, surface impulses 1.18-1.25 at P2). The 30-40 % P1-vs-P2 gap sits in the homogeneous response of
    single surface tents; the forward-patch test (running) says which order agrees with CSI for that.
  - 11:40 P2 FORWARD (kinematic, 10 mm on km 57-61 top 1 km, homog) vs P1 forward: rms 0.092 mm on a 2.2 mm signal
    (4 %; P1 vs the fine P1 mesh 0.083, P2 vs fine 0.080) -> P2 kinematics agree with the validated P1 solution.
    The P1/P2 gap is confined to the single-tent GreensFns responses. DECISION: the order-2 production runs go
    ahead as specified (queue relaunched: order-2 SFCVM no-cap ~11:45-14:30, order-2 homog ~14:30-17:15), and the
    P2 GFs are validated by superposition against a P2 forward run of the same medium (running) before any
    order-2 inversion is reported.
  - 11:45 sum test: the 33 test tents SUMMED give P2/P1 = 0.98 (surface tents alone 0.70, buried alone 1.23): P1 and P2
    agree for smooth slip and differ only in how a single one-cell tent at/below the surface is resolved. So the
    order-2 run is a check on the slip-depth/surface-% numbers, not on the fitted field.
- Products: tag `_order2` (e.g. gf/results_comparison_order2.csv, figures/summary_default_order2_sfcvm.png); the
  final table flags them in the mesh column.
- Materials order 1 smears the fault contrast (as the earlier caution said): fault-adjacent mu in the top 100 m is
  0.67 GPa west / 1.09 east (ratio 1.6) against 0.50 / 1.32 (ratio 2.6) with order-0 materials; 90th-percentile
  cell-vs-grid mu mismatch 37 % (order 0: 0.3 %). The order-2 SFCVM run therefore also has a weaker contrast.
- Order-2 SFCVM GF validation (superposition against the P2 forward run of the same medium): 0.085 mm rms (4 %),
  but 0.022 mm (0.5 %) inside the patch span and all of the rest within 300 m of the patch ENDS, where the P2
  forward run imposes a sharper end than the tent sum. Order-1 superposition is exact. Accepted with a recorded
  warning; the order-2 G is consistent with the P2 solution.
- ORDER-2 SFCVM RESULT (full Cd, surface ON, sigma 12): rms 1.77 / W 2.27 / E 1.32 mm, chi2 4017, surface 72 %,
  potency 9,764 m3 (order-1 SFCVM 8,973; order-1 homog 7,823), Mw30 3.58, Mw(SFCVM mu) 2.85. Against order-1 SFCVM
  (1.72 / 2.22 / 1.28) the fit is the same within 0.05 mm and the potency 9 % higher. The order-2 treatment does
  not change the conclusion: the west lobe stays under-fit under the correlated Cd. One-pager:
  figures/summary_default_order2_sfcvm.png. Nugget-Cd version running; order-2 homog after its GF (~16:51).

## Done
- Step 1: `plot_run_summary.py`. One-pagers for every cached run are in `figures/summary_<run>_<medium>.png`
  (default, default_nosurf, default_nosurf_diagcd, default_shallow; homog + sfcvm). The notebook now saves what the
  plot needs (data, signed distance, block polygons, settings) and calls it at the end of every run.
- Step 3 prep: swath-7 sample set built (`make_points_swath.py`, 07's cells [1]-[4] with only swath_km changed;
  swath 4 reproduces 07's 2386 samples exactly). Swath 7 gives only 2396 samples (10 more): the quadtree starts
  from 2 km blocks, so the extra 32k far-field pixels (all west) fill ~10 large blocks. GFs running.

## Step 2 result (noise weighting; default mesh, vertical, surface ON, cached GFs)
- `floor 15` and `floor 4` (07's Cd with n_px floored): NO change in the fit at equal sigma (sfcvm rms 1.74/1.73 vs
  1.72 full; W lobe still missed). The floor only rescales chi2 and moves the L-curve corner (floor 15 -> sigma 8).
  So the per-block nugget rule is not what suppresses the west lobe; the result is robust to it (useful statement).
- `diag` (equal weights) + surface ON: sfcvm W 1.76 (homog 1.98), surface 94 %, Mw30 3.66. `nugget` (07's per-block
  nugget variances, no correlation) + surface ON: sfcvm rms 1.36, W 1.50, E 1.39, surface 98 %, Mw30 3.76, potency
  x2.0; homog W 1.84. -> it is the CORRELATED part of 07's Cd (0.82 mm at 1.2 km range, from the far-field
  variogram) that lets the inversion treat the broad west lobe as noise. Without it the west lobe is fitted by
  SFCVM (and partly by homog), but the potency doubles (far-field signal now interpreted as deeper slip) and the
  reduced chi2 under the nugget-only Cd is ~5.6 (its variances are too small), i.e. the correlated model is the
  statistically consistent one.
- Recommendation (step 2): keep 07's full correlated Cd as the primary weighting (data-derived, unchanged, and
  insensitive to the nugget floor); report the nugget-only diagonal Cd as the sensitivity that shows the SFCVM
  west-lobe explanation. The dips are inverted with both, so the dip answer does not hinge on this choice.

## Step 3 result (swath 7 km, full Cd): no change
- 2396 samples (vs 2386): homog rms 2.02 / W 2.57 / E 1.66, potency 7,894 m3 (+0.9 %), Mw30 3.52; sfcvm 1.75 / 2.27 /
  1.29, potency 9,061 m3 (+1.0 %), Mw30 3.56; sigma 12 both. The extra far field (32k pixels, all west, in ~10 large
  quadtree blocks) changes neither the slip total nor the west-lobe fit. (A smaller quadtree start block would be
  needed to give the far field real weight; not done, 07's logic kept.) Nugget-Cd version: also unchanged
  (sfcvm 1.37 / 1.51 / 1.39, potency +0.6 %). => the dips are run at the default 2386 points.
- Dip-grid check: the SFCVM contrast follows each dipping fault to within the 50 m grid at 0.4-2 km depth; the
  vertical 'aligned' grid keeps the SFCVM's own ~68 E boundary dip (contrast +385 m east of the fault at 1 km).

## Step 4 result: fault dip in PyLith (full Cd, surface ON, 2386 samples; SFCVM contrast ON the dipping fault)

| dip | medium | rms all | rms W300 | rms E300 | chi2 | surf % | potency m3 | Mw30 |
|---|---|---|---|---|---|---|---|---|
| 90 | homog | 1.98 | 2.52 | 1.62 | 3981 | 75 | 7823 | 3.51 |
| 90 | sfcvm | 1.72 | 2.22 | 1.28 | 3954 | 74 | 8973 | 3.55 |
| 80 E | homog | 2.09 | 2.65 | 1.73 | 3952 | 74 | 8091 | 3.52 |
| 80 E | sfcvm | 1.78 | 2.30 | 1.32 | 3959 | 74 | 8308 | 3.53 |
| 70 E | homog | 2.21 | 2.79 | 1.88 | 3954 | 74 | 8285 | 3.53 |
| 70 E | sfcvm | 1.87 | 2.42 | 1.39 | 3950 | 75 | 8038 | 3.52 |

- An east dip makes the west near-field fit WORSE, monotonically (homog W 2.52 -> 2.65 -> 2.79 mm; SFCVM 2.22 -> 2.30
  -> 2.42), as expected with the hanging wall to the east. Confirmed in PyLith with the material contrast following
  the dipping fault. Under the nugget-only Cd the same ordering holds (80E: sfcvm W 1.78 vs vertical 1.50).
- SFCVM beats homogeneous by ~0.3-0.4 mm in the west at every dip; chi2 (full Cd) is flat at 3950-3980 for all six.
- Nothing geologically realistic (70-90 E) improves the west lobe beyond the vertical SFCVM model. The 80 W
  sensitivity (implausible) is queued behind the order-2 runs and will only be reported as such.

## Step 5: final comparison and recommendation (table refreshed 19:10; recommendation text from 12:15, see note)

Every PyLith run (gf/final_comparison.csv; one-pagers figures/summary_<run>_<medium>.png):

| run | medium | mesh | dip | Cd | surface | points | sigma | rms_mm | rms_W300_mm | rms_E300_mm | chi2 | surface_pct | potency_m3 | Mw_30gpa | Mw_sfcvm | Mw_30gpa_2sd | Mw_sfcvm_2sd | surface_pct_2sd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 07 CSI baseline | homog (Meade TDE) | 07 smooth12 | 90 | full | on | default | 8 | 1.95 | 2.47 | 1.62 | 3890 | 79 |  | 3.57 | 2.71 |  |  |  |
| default | homog | default | 90 | full | on | default | 12 | 1.98 | 2.52 | 1.62 | 3981 | 75 | 7823 | 3.51 | 2.69 | 0.02 | 0.06 | 4.00 |
| default | sfcvm | default | 90 | full | on | default | 12 | 1.72 | 2.22 | 1.28 | 3954 | 74 | 8973 | 3.55 | 2.84 | 0.03 | 0.11 | 4.17 |
| default_diagcd | homog | default | 90 | diag | on | default | 12 | 1.70 | 1.98 | 1.60 | 6151 | 90 | 12746 | 3.65 | 3.03 |  |  |  |
| default_diagcd | sfcvm | default | 90 | diag | on | default | 12 | 1.48 | 1.76 | 1.29 | 4629 | 94 | 13116 | 3.66 | 3.01 |  |  |  |
| default_dip70e | homog | default | 70 E | full | on | default | 12 | 2.21 | 2.79 | 1.88 | 3954 | 74 | 8285 | 3.53 | 2.74 |  |  |  |
| default_dip70e | sfcvm | default | 70 E | full | on | default | 12 | 1.87 | 2.42 | 1.39 | 3950 | 75 | 8038 | 3.52 | 2.78 |  |  |  |
| default_dip70e_nuggetcd | homog | default | 70 E | nugget | on | default | 8 | 1.98 | 2.45 | 1.73 | 24672 | 82 | 11742 | 3.63 | 2.95 |  |  |  |
| default_dip70e_nuggetcd | sfcvm | default | 70 E | nugget | on | default | 8 | 1.60 | 1.92 | 1.41 | 18595 | 104 | 10666 | 3.60 | 2.96 |  |  |  |
| default_dip80e | homog | default | 80 E | full | on | default | 12 | 2.09 | 2.65 | 1.73 | 3952 | 74 | 8091 | 3.52 | 2.73 |  |  |  |
| default_dip80e | sfcvm | default | 80 E | full | on | default | 12 | 1.78 | 2.30 | 1.32 | 3959 | 74 | 8308 | 3.53 | 2.79 |  |  |  |
| default_dip80e_nuggetcd | homog | default | 80 E | nugget | on | default | 8 | 1.85 | 2.22 | 1.70 | 21668 | 86 | 12420 | 3.65 | 2.94 |  |  |  |
| default_dip80e_nuggetcd | sfcvm | default | 80 E | nugget | on | default | 8 | 1.51 | 1.78 | 1.35 | 16611 | 102 | 12440 | 3.65 | 2.97 |  |  |  |
| default_e50 | sfcvm_e50 | default | 90 | full | on | default | 12 | 1.81 | 2.35 | 1.34 | 4051 | 74 | 9712 | 3.58 | 2.85 |  |  |  |
| default_e50_nuggetcd | sfcvm_e50 | default | 90 | nugget | on | default | 16 | 1.35 | 1.42 | 1.45 | 12703 | 93 | 21686 | 3.81 | 3.06 |  |  |  |
| default_floor15 | homog | default | 90 | floor15 | on | default | 8 | 2.05 | 2.73 | 1.45 | 9565 | 68 | 6260 | 3.45 | 2.61 |  |  |  |
| default_floor15 | sfcvm | default | 90 | floor15 | on | default | 8 | 1.84 | 2.44 | 1.25 | 9490 | 65 | 6907 | 3.48 | 2.72 |  |  |  |
| default_floor4 | homog | default | 90 | floor4 | on | default | 12 | 1.97 | 2.52 | 1.58 | 5951 | 74 | 8150 | 3.53 | 2.70 |  |  |  |
| default_floor4 | sfcvm | default | 90 | floor4 | on | default | 12 | 1.73 | 2.23 | 1.27 | 5880 | 73 | 9290 | 3.56 | 2.84 |  |  |  |
| default_nosurf | homog | default | 90 | full | off | default | 24 | 1.84 | 2.30 | 1.58 | 3220 | 51 | 12928 | 3.66 | 2.85 |  |  |  |
| default_nosurf | sfcvm | default | 90 | full | off | default | 24 | 1.65 | 2.08 | 1.33 | 3202 | 53 | 14153 | 3.69 | 2.96 |  |  |  |
| default_nosurf_diagcd | homog | default | 90 | diag | off | default | 16 | 1.60 | 1.85 | 1.51 | 5423 | 66 | 15554 | 3.71 | 3.12 |  |  |  |
| default_nosurf_diagcd | sfcvm | default | 90 | diag | off | default | 16 | 1.36 | 1.53 | 1.30 | 3917 | 86 | 15714 | 3.72 | 3.08 |  |  |  |
| default_nuggetcd | homog | default | 90 | nugget | on | default | 12 | 1.66 | 1.84 | 1.73 | 17684 | 90 | 14803 | 3.70 | 2.96 | 0.02 | 0.07 | 3.28 |
| default_nuggetcd | sfcvm | default | 90 | nugget | on | default | 12 | 1.36 | 1.50 | 1.39 | 13296 | 98 | 18073 | 3.76 | 3.04 | 0.02 | 0.07 | 3.46 |
| default_order2 | homog | default order2 (P2 solution, P1 materials, no nu cap) | 90 | full | on | default | 12 | 1.99 | 2.54 | 1.60 | 4002 | 73 | 8607 | 3.54 | 2.72 |  |  |  |
| default_order2 | sfcvm | default order2 (P2 solution, P1 materials, no nu cap) | 90 | full | on | default | 12 | 1.77 | 2.27 | 1.32 | 4017 | 72 | 9764 | 3.58 | 2.85 |  |  |  |
| default_order2_nuggetcd | homog | default order2 (P2 solution, P1 materials, no nu cap) | 90 | nugget | on | default | 12 | 1.66 | 1.83 | 1.75 | 17744 | 88 | 15483 | 3.71 | 2.97 |  |  |  |
| default_order2_nuggetcd | sfcvm | default order2 (P2 solution, P1 materials, no nu cap) | 90 | nugget | on | default | 20 | 1.32 | 1.29 | 1.50 | 12495 | 83 | 22107 | 3.81 | 3.11 |  |  |  |
| default_shallow | homog | shallow | 90 | full | on | default | 16 | 2.02 | 2.62 | 1.55 | 4060 | 72 | 7152 | 3.49 | 2.64 |  |  |  |
| default_shallow | sfcvm | shallow | 90 | full | on | default | 16 | 1.81 | 2.36 | 1.27 | 4092 | 71 | 8170 | 3.53 | 2.79 |  |  |  |
| default_swath7 | homog | default | 90 | full | on | swath7 | 12 | 2.02 | 2.57 | 1.66 | 3923 | 75 | 7894 | 3.52 | 2.69 |  |  |  |
| default_swath7 | sfcvm | default | 90 | full | on | swath7 | 12 | 1.75 | 2.27 | 1.29 | 3897 | 74 | 9061 | 3.56 | 2.84 |  |  |  |
| default_swath7_nuggetcd | homog | default | 90 | nugget | on | swath7 | 12 | 1.66 | 1.86 | 1.73 | 17716 | 90 | 15036 | 3.70 | 2.97 |  |  |  |
| default_swath7_nuggetcd | sfcvm | default | 90 | nugget | on | swath7 | 12 | 1.37 | 1.51 | 1.39 | 13483 | 98 | 18175 | 3.76 | 3.05 |  |  |  |
| default_w50 | sfcvm_w50 | default | 90 | full | on | default | 12 | 1.67 | 2.09 | 1.33 | 3835 | 75 | 7152 | 3.49 | 2.75 |  |  |  |
| default_w50_nuggetcd | sfcvm_w50 | default | 90 | nugget | on | default | 8 | 1.38 | 1.54 | 1.37 | 13968 | 98 | 13037 | 3.66 | 2.97 |  |  |  |


### Recommendation (written 12:15, BEFORE Dani selected the nugget-Cd SFCVM model; facts unchanged, and the
### uncertainty/recovery tests since then show the full-Cd Mw is a lower bound, which supports her choice)
- The best model that is geologically realistic (vertical), uses the defensible noise model (07's full correlated Cd,
  unchanged) and fits the west near field better than homogeneous is the **vertical SFCVM-aligned model**:
  rms 1.72 mm (homogeneous 1.98), W<=300 m 2.22 (2.52), E<=300 m 1.28 (1.62), sigma 12, surface slip 74 % of the
  crossings, potency 8,973 m3 (1.15 x homogeneous), Mw 3.55 at 30 GPa / 2.84 with the SFCVM rigidity.
  One-pager: `drafts/pylith/figures/summary_default_sfcvm.png` (homogeneous: `summary_default_homog.png`).
- But the improvement is modest and NOT in the west lobe: the west rms drops 12 % (east 21 %, all 13 %) while chi2
  under the correlated Cd drops 0.7 % (3981 -> 3954), and the model still reaches only ~-4 mm at 100 m west of the
  trace where the data are -7 to -9 mm. Under 07's noise model no elastic model tested fits the west lobe.
- Fault dip does not rescue it: 80 E and 70 E make the west fit worse for both media (SFCVM W 2.30, 2.42 mm), as
  expected with the hanging wall to the east, and chi2 is flat (3950-3980) across all dips and media.
- The west lobe IS fitted when the correlated part of Cd is dropped (nugget-only diagonal Cd: SFCVM W 1.50 vs
  homogeneous 1.84, sigma 12, surface 98 %), but that weighting has a reduced chi2 of ~5.6 and doubles the potency
  (Mw30 3.76); it is a sensitivity, not a defensible primary. The nugget-floor tests (4, 15) show the result is
  insensitive to the per-block nugget rule. Extra far field (7 km swath) changes nothing.
- Plain answer for today: a PyLith + SFCVM model with a realistic dip does NOT fit the west-side asymmetry under
  the noise model we can defend. Publish the homogeneous model as the primary result; the vertical SFCVM model can
  be shown as the heterogeneous variant (13 % lower rms, same Mw within 0.04, same chi2) with the dip tests as
  evidence that the asymmetry is not a fault-geometry effect. The order-2 SFCVM run (in progress, ETA ~15:30) is a
  numerical check of the surface-slip / depth numbers and is not expected to change this conclusion.

## Key numbers so far (default mesh, 2386 samples; vertical unless stated)

| run | Cd | surface | sigma | rms all | rms W300 | rms E300 | chi2 | surf % | Mw30 |
|---|---|---|---|---|---|---|---|---|---|
| homog | full | on | 12 | 1.98 | 2.52 | 1.62 | 3981 | 75 | 3.51 |
| sfcvm | full | on | 12 | 1.72 | 2.22 | 1.28 | 3954 | 74 | 3.55 |
| homog | floor15 | on | 8 | 2.05 | 2.73 | 1.45 | 9565 | 68 | 3.45 |
| sfcvm | floor15 | on | 8 | 1.84 | 2.44 | 1.25 | 9490 | 65 | 3.48 |
| homog | floor4 | on | 12 | 1.97 | 2.52 | 1.58 | 5951 | 74 | 3.53 |
| sfcvm | floor4 | on | 12 | 1.73 | 2.23 | 1.27 | 5880 | 73 | 3.56 |
| homog | diag | on | 12 | 1.70 | 1.98 | 1.60 | 6151 | 90 | 3.65 |
| sfcvm | diag | on | 12 | 1.48 | 1.76 | 1.29 | 4629 | 94 | 3.66 |
| homog | nugget | on | 12 | 1.66 | 1.84 | 1.73 | 17684 | 90 | 3.70 |
| sfcvm | nugget | on | 12 | 1.36 | 1.50 | 1.39 | 13296 | 98 | 3.76 |
| homog | full | off | 24 | 1.84 | 2.30 | 1.58 | 3220 | 51 | 3.66 |
| sfcvm | full | off | 24 | 1.65 | 2.08 | 1.33 | 3202 | 53 | 3.69 |
| homog | diag | off | 16 | 1.60 | 1.85 | 1.51 | 5423 | 66 | 3.71 |
| sfcvm | diag | off | 16 | 1.36 | 1.53 | 1.30 | 3917 | 86 | 3.72 |
| homog dip80E | full | on | 12 | 2.09 | 2.65 | 1.73 | 3952 | 74 | 3.52 |
| sfcvm dip80E | full | on | 12 | 1.78 | 2.30 | 1.32 | 3959 | 74 | 3.53 |
| homog shallow | full | on | 16 | 2.02 | 2.62 | 1.55 | 4060 | 72 | 3.49 |
| sfcvm shallow | full | on | 16 | 1.81 | 2.36 | 1.27 | 4092 | 71 | 3.53 |

## ETA
- Step 2: done
- Step 3: done
- Step 4: done (70E nugget ~12:10)
- Step 5 (final table + recommendation): ~12:30; refreshed when the order-2 SFCVM inversion lands (~15:30)

## New custom code this session (all in drafts/pylith)
- `plot_run_summary.py`: one-page diagnostic from cached products (plotting + log parsing only).
- `make_points_swath.py`: exec of 07's cells [1]-[4] with swath_km replaced; writes the PyLith PointsList.
- `run_gf.sh`: optional points-file argument. `run_notebook.sh`: env-switch wrapper. `queue_swath7.sh`.
- `build_G.py`: `--points` argument (which point set the run used).
- notebook: env switches PYLITH_DIP, PYLITH_POINTS, PYLITH_CD=floor (+PYLITH_CD_FLOOR) | nugget; Cd floor = 07's build_cd
  on max(n_px, floor); nugget = diag of 07's Cd minus the zero-lag correlated terms.
- `generate_gmsh_dip.py` (extrusion vector + dip check), `sfcvm_grid.py --dip` (per-depth boundary search and shift),
  `sfcvm_grid.py --no-cap` and grid extent 51 km, `final_comparison.py` (table), queue scripts (`queue_dips*.sh`,
  `queue_inversions.sh`, `queue_after_dips.sh`, `build_dip_grids*.sh`, `build_nocap_grid.sh`, `run_o2_tests.sh`).
- `cfg/order2.cfg`, `cfg/order2_fwd.cfg`, `run_gf_o2.sh`, `cfg/mat_sfcvm_nocap.cfg`, `cfg/mat_sfcvm_dip*.cfg`;
  notebook switch PYLITH_ORDER=2 (P1 materials read from vertex fields; superposition against the P2 forward runs).
- `pylith_sfcvm_tests.py` (exec's the notebook through [8]; loops for bootstrap / recovery / depth scan; the
  whitened least squares and window shape of 07_slip_model_tests [9]/[11] applied to the tent parameters),
  `invert_tents(vel_insar=, vel_surface=)` overrides, `sfcvm_grid.py --offset`, `cfg/mat_sfcvm_{e50,w50}.cfg`,
  `queue_offset.sh`, `make_vertex_slip_db.py` (spatialdata SimpleIOAscii writer), `make_fig2_product.py`,
  `cfg/fwd_kin_simpledb.cfg`; `final_comparison.py` merges the bootstrap 2-sigma columns.
- Task 6: `coh07/make_fp_coh.py` (exec of 06 with asserted substitutions; reads 06's fit cache, writes only to coh07/),
  `coh07/strip_stats.py` (per-track implied fp with MintPy's own design matrix), `coh07/compare_coh.py` (spike and
  common-sample scores), `make_points_swath.py <swath> <fp_maps> <tag>`, notebook PYLITH_POINTS=coh07,
  `make_fig2_product.py <cd> _coh07` + `mw_30gpa_range`, `queue_coh07*.sh`, `plot_tests_figures.py`.
