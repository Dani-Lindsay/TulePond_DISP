# 07 slip model: decisions and sweep results (2026-09-28)

Every number below comes from a run that has been done, and carries a
source tag. Where no run exists yet the entry says **[planned]**. No number
here is in the registry yet; replace with registry values once the final
model runs.

**Source tags** (all files in `outputs/07_slip_model_tests/` unless noted):
- [suite] `suite_results.csv`, batch-2 run of 2026-09-28 (37 configurations,
  every CHECK passed). LOS input (asc + desc), surface constraint with CSI's
  single-triangle link, uniform 30 GPa, homogeneous Green's functions.
  Dip, sampling and some mesh rows are on the `plateau100` mesh; sigma,
  lambda and isolation rows on the selected `smooth12` mesh.
- [dip] `dip_on_selected_mesh.csv`: dip family re-run on `smooth12`,
  sigma 24, LOS input, interpolated surface constraint, 30 GPa.
- [sigma] `sigma_selection.csv`: sigma family on `smooth12`, LOS input,
  interpolated surface constraint, 30 GPa.
- [fp] `interim_logs_2026-09-28.txt`, fault-parallel inversion test
  (`smooth12`, sigma 24, 30 GPa).
- [scan] `depth_scan*.csv`: fixed-depth test (uniform slip 0-D).
- [recovery] `recovery_summary.csv`: batch-2 synthetic recovery (LOS,
  single-triangle surface link).
- [scanMw] Mw of the fixed-depth models, computed 2026-09-28 from
  `depth_scan.csv` on the `plateau100` mesh (`interim_logs_2026-09-28.txt`,
  section fixed_depth_model_mw; not a notebook product yet).
- [SFCVM] `sfcvm_creep_section_profiles.csv` (`drafts/07/sfcvm_extract.py`),
  and the depth-dependent-rigidity Mw computed from it on 2026-09-28
  (`interim_logs_2026-09-28.txt`, section rigidity_scaled_mw; not a
  notebook product yet). Noise-model numbers: same file, noise_model_fits.

## A. Decisions

1. **Fault-parallel input.** We invert the fault-parallel component of the
   mean event (06 decomposition) instead of the two LOS tracks, so the
   non-tectonic, off-fault vertical signal east of the fault is excluded
   rather than down-weighted. Assumption: no horizontal motion
   perpendicular to the fault. Check the model's predicted fault-normal
   motion is negligible: now a CHECK cell in `07_slip_model` (see G).
   - Consistency: the surface constraint (04 per-event crossing
     amplitudes) is also fault-parallel: the east - west step at each
     crossing (30-150 m aperture), decomposed from asc + desc in 03. From
     here on everything is fault-parallel. Difference to note: 03 uses the
     local trace azimuth at each crossing; 06 uses one box strike (143°)
     **[check planned]**.
   - Residual maps: the LOS inversion leaves a coherent residual patch
     just east of the fault in ascending; the fault-parallel inversion
     does not [fp; `../Figures_2026-09-25/draft_residuals_los_vs_fp.png`].
2. **Smoothed least squares.** Regularised least squares with positivity
   and an exponential model covariance (buildCmXY). **Superseded (G):**
   sigma is now chosen at the L-curve corner, leaning to under-smoothing:
   sigma = 12 [lcurve]. (The earlier "onset of the misfit plateau" rule
   picked 24; on the final configuration there is no plateau.)
3. **Parameter sweep, not Bayesian sampling**, to sample the uncertainty
   from model-design choices (section B). A small parametric Bayesian
   depth/Mw model stays optional **[planned, optional]**.
4. **Crossing constraint in the inversion.** The pixels straddling the
   trace are excluded (nearest data >= 30 m from the trace), so the InSAR
   alone does not constrain the uppermost cells. The crossing-stack surface
   slip is added as data, interpolated to every surface triangle (errors
   scaled so the total weight is unchanged) instead of CSI's
   single-nearest-triangle link.
5. **Vertical fault.** The InSAR does not resolve dip (section B).
6. **Mean of all five events**, nothing excluded.
7. **Equal weighting of the data** (asc/desc weighting and ascending-only
   tests dropped: they changed Mw by <= 0.01 [suite: asc_x2 3.82 vs sigma
   24 row 3.83]).
8. **Far-field levelling** of each dataset (plane fitted outside the creep
   section ± 1 km), then a per-dataset constant in the inversion.
9. **Prediction stated in observables** (location, timing, surface slip);
   Mw given as a range with the rigidity stated, plus potency
   **[potency planned]**.
10. **1D layered Green's functions with PyLith 5.0.2** from the USGS SFCVM
    profiles; 3D later with colleagues **[planned; PyLith 5.0.2 installed
    and its 3D Green's function example runs]**.

## B. What we tried and how far Mw moved (30 GPa unless stated)

| Choice | Values tried | Mw range | Source / notes |
|---|---|---|---|
| Smoothing sigma (with surface constraint) | 8, 16, 24, 32, 48, 64 | 3.59-3.96 | [sigma]; rms 1.16-1.19 mm, i.e. misfit nearly flat. The largest lever of any model-design choice |
| Smoothing sigma (InSAR only) | 8, 32, 64 | 3.59-3.96 | [suite] |
| Correlation lengths (km, strike x depth) | 0.15x0.05, 0.6x0.05, 0.3x0.15, 0.6x0.15 | 3.81-3.87 | [suite] |
| Dip | 60°E, 70, 75, 80, 85, 90, 95, 100, 105 (= 75°W) | 3.82-3.89 | [dip]; rms falls monotonically 1.24 -> 1.13 mm, descending near-field (<200 m) 1.18 -> 0.96 mm, no minimum |
| Dip (plateau100 mesh) | 60-95 | 3.94-4.07 | [suite]; different mesh, same trend |
| Mesh (surface cell, growth) | 100, 50, 30 m; plateau vs smooth x1.2; 1.5 km plateau | 3.86-3.99 | [suite]; each at its own sigma |
| Quadtree floor / sampling | 30 (base), 40, 60 m; uniform 100 m | 3.95-3.99 | [suite]; plateau100 mesh |
| Surface-constraint error floor | 0.25, 0.5, 1.0 mm | 3.83 (all) | [suite] |
| Surface constraint on/off | off / on | 3.83 / 3.83 | [suite]; surface slip recovered 44% -> 75% of the crossings (LOS), 33% -> 68% (fault-parallel) [fp] |
| Far-field levelling | off / on | 3.83 / 3.83 | [suite] |
| Input | LOS asc + desc / fault-parallel | 3.82 / 3.75 | [fp]; fault-parallel pixel noise 0.99 mm vs 0.6-0.7 mm per track |
| Free rake (strike + dip slip) | fixed / free | 3.83 / 3.67 | [suite]; fit worse with the rake free |
| Prior form (stress tests) | correlated (base); minimum-norm; effectively unregularised | 3.83; 4.19; 4.34 | [suite]. The prior form is the shape of the regularisation: correlated = neighbouring cells expected to be similar (the smoothing); minimum-norm = cells independent, only small slip preferred; unregularised = positivity only (unphysical peaks). Stress tests, not candidates |
| Rigidity | 30 GPa / SFCVM depth-dependent | LOS 3.82 -> 3.11; fault-parallel 3.75 -> 2.86 (2.90 W side, 3.03 E side) | [SFCVM]; slip from homogeneous Green's functions, so a lower bound. Layered Green's functions will raise the near-surface slip **[planned]** |
| Depth of slip (fixed-depth models) | D = 0.85 / 1.5 / 3 km (all within Δχ² 4) | 3.91 / 4.04 / 4.26 at 30 GPa; 3.24 / 3.68 / 4.10 with SFCVM rigidity | [scan] + [scanMw]. See section C |

## C. Depth

- Fits within Δχ² = 4 of the best: D = 0.85-3 km, best 1.5 km (current
  fields); 0.75-1.65 km, best 1.1 km (legacy fields). The same range
  without levelling and on a 5 km-deep mesh. Δχ² = 28 at 0.5 km: slip
  confined above ~0.8 km is rejected [scan].
- Data extend 4 km from the trace. By the ~1-2 D rule this covers depths
  of a few km, so coverage is not the limit: slip to 0.85 km and to 3 km
  differ by < 0.7 mm in descending LOS at 0.5-4 km from the fault, below
  the spatially correlated noise at those distances (arithmetic from the
  [scan] amplitudes and the fitted noise model, 2026-09-28; to be moved
  into the notebook **[planned]**).
- The smoothed model cannot tell depths apart: synthetic input to 0.3-2 km
  comes back with nearly the same shallow profile; moment recovered 1.25
  (0.3 km), 0.81 (0.5), 0.51 (0.8), 0.45 (1.0), 0.30 (1.5), 0.26 (2.0)
  [recovery]. The same test with the interpolated constraint and the
  fault-parallel input **[planned]**.
- **The moment is not robust to depth.** Across the equally fitting
  fixed-depth models Mw spans 3.91-4.26 at 30 GPa and 3.24-4.10 with the
  SFCVM rigidity [scanMw]. Depth and rigidity together are the largest
  magnitude uncertainty.

## D. Asymmetry and rigidity

- Fault-parallel profile X-X' of the mean event: west -5.48 ± 0.31 mm,
  east +2.10 ± 0.18 mm (Fig 2 G, notebook 06).
- Candidate sources: (1) vertical or fault-perpendicular motion leaking
  into LOS: ascending sees vertical strongly and fault-parallel weakly
  (|LOS·fault-parallel| asc 0.28, desc 0.49, from the 09:28 run of 07);
  (2) an off-fault vertical source east of the fault; (3) a rigidity
  contrast across the fault. Only (3) can produce asymmetry in the
  fault-parallel field itself.
- SFCVM v21.1, median across km 56.5-62.3 at 0.2-3 km from the trace [SFCVM]:

| Depth | Vs west | Vs east | Rigidity west | Rigidity east |
|---|---|---|---|---|
| 50 m | 419 m/s | 1008 m/s | 0.3 GPa | 2.3 GPa |
| 200 m | 681 m/s | 1074 m/s | 1.0 GPa | 2.7 GPa |
| 500 m | 903 m/s | 1018 m/s | 1.8 GPa | 2.3 GPa |
| 750 m | 2307 m/s | 1505 m/s | 14.1 GPa | 5.2 GPa |
| 1.5 km | 2851 m/s | 1823 m/s | 21.6 GPa | 7.9 GPa |
| 3 km | 3264 m/s | 3264 m/s | 28.3 GPa | 28.3 GPa |

  - West: Quaternary/Tertiary sediment over Franciscan (step at ~0.75 km).
    East: Great Valley Sequence.
  - The west side is softer in the top ~500 m; the contrast reverses below
    ~0.75 km. A softer west side deforms more for the same slip, in the
    same sense as the observed asymmetry. Testing that needs the 3D model
    **[planned]**.
  - The surface value on the west (~80 m/s) looks like soft mud at Tule
    Pond itself; values from ~50 m down are more trustworthy.
- Softer rock gives a lower moment for the same slip (moment = rigidity x
  area x slip).

## E. Planned, not yet run

- 1D layered Green's functions (PyLith 5.0.2): uniform-rock check against
  CSI first, then the SFCVM 1D profile.
- ~~Uniform slip tied over the top ~200 m~~ done (F); re-check the tie
  depth at sigma 12 (none / 0.1 / 0.2 km).
- ~~Fault-normal motion check~~ in the notebook (G); crossing vs field
  strike check.
- ~~L-curve for sigma~~ done (G).
- Re-test dip, mesh, lambda and sampling on the final configuration (they
  were decided on the LOS setup).
- **Depth vs noise, on the fault-parallel field.** The 2026-09-27 argument
  (u = (s/pi) arctan(D/x); D = 0.85 vs 3 km differ by 0.4-0.7 mm in desc LOS,
  0.9-1.4 mm fault-parallel, at 0.5-4 km [scan amplitudes]) compared this
  with "correlated noise ~0.6 mm over ~2 km", which does NOT match the noise
  fits recorded on 2026-09-28 (asc short 0.56 mm / 0.57 km, desc short ~0,
  long 1.4-1.6 mm / 14-19 km; fault-parallel long term at the 50 km bound).
  Do not quote it. Run the fixed-depth scan on the fault-parallel field
  (delta chi^2 with the full covariance is the rigorous version) and add a
  registered notebook check tabulating the arctan signal difference against
  the fitted noise at the same separations.
- **Correlation lengths lambda, each at its own L-curve corner.** In CSI
  (`Fault.buildCmXY`, Radiguet et al. 2010) Cm = sigma^2 (lam0^2 / (lamx
  lamz)) exp(-0.5 dx/lamx) exp(-0.5 dz/lamz): lambda sets the shape AND
  enters the amplitude, so the sigma corner (10.1) holds only for (0.3,
  0.05) km, and the old lambda sweep (Mw 3.81-3.87 [suite], sigma fixed, LOS
  setup) mixed shape with strength. Test: the 4-5 lambda pairs, an L-curve
  each, compare chi2 and Mw at each corner; if Mw stays within the sigma
  8-16 range (3.64-3.70) say so in one sentence. With the new data, as part
  of the final-configuration rerun of the tests.
  - Why lamz is short (Dani): a long down-dip length ties each cell to the
    cells below it, so the prior drags the well-constrained shallow slip to
    depth; that was the model "pushing slip deep". With lamz = 0.05 km
    (1/e at ~100 m) the prior imposes no depth, and the model returns the
    SHALLOWEST solution the data allow. That is not a measured depth: the
    fixed-depth scan fits D = 0.85-3 km equally [scan], and synthetic slip to
    1-2 km comes back shallow with 26-45% of its moment [recovery]. Hence the
    distributed-model Mw is a minimum. Paper line: "the down-dip correlation
    length is kept short so the prior does not impose slip at depth; the
    depth extent is not resolved (0.85-3 km fit equally), so the distributed
    model gives the shallowest consistent solution and a minimum moment."
- Recovery test with the interpolated constraint and fault-parallel input.
- Potency; event-to-event spread of surface slip and extent (from 04).
- Rerun of `07_slip_model_tests` on the final configuration; drop panels B
  and C from `fig07_slip_model`.

## F. Added later on 2026-09-28

- **Decomposition strike:** the field decomposition in 06 now uses the mean
  local strike of the SSE section (146.44°, km 56.3-62.3) instead of the
  whole-box mean (143.05°); the crossings (03) keep their local strike. Zero-
  difference CHECK: at 143.05° the new saved-products path reproduces the
  MintPy-mode field exactly. Effect: mean-event fault-parallel step 7.97 ->
  8.65 mm (registry `06_fp_event_step_mm`), vertical step 0.70 -> 0.57 mm;
  fault-parallel inversion surface slip 68% -> 75% of the crossings, Mw 3.75
  -> 3.76 [interim logs; 06 registry].
- **Coherence for the maps and the inversion field: 0.9** (Dani); crossings
  and LOS checks stay at 0.7. 30% of the box kept (0.7: 46% asc, 42% desc).
  Mean-event fault-parallel step 8.83 mm at 0.9 (`06_fp_event_step_mm`).
- **Top-layer tie** (slip uniform over the top of each along-strike column,
  after Jolivet et al. 2023) [`drafts/07/top_tie_test.py`, 0.7 field]:
  | Tie depth | Peak (depth) | Surface % of crossings | rms | Mw |
  |---|---|---|---|---|
  | none | 12.9 mm (140 m) | 75% | 1.96 mm | 3.76 |
  | 0.1 km | 11.6 mm (140 m) | 79% | 1.98 mm | 3.76 |
  | 0.2 km | 10.2 mm (30 m) | 80% | 1.99 mm | 3.76 |
  | 0.5 km | 10.0 mm (30 m) | 80% | 1.99 mm | 3.85 |
  Synthetic 10 mm to 0.3/0.8/1.5 km with a 0.2 km tie: surface cell 10.1-10.4
  mm and no sub-surface overshoot; recovered Mw 3.70/3.80/3.84 (the depth bias
  on the moment remains). **Adopted: 0.2 km.**
- **Current best model** (Fig 2 row 3, `fig2_model_row.npz`; draft, homogeneous
  Green's functions): fault-parallel, 0.9 field, SSE-section strike, 0.2 km
  tie, sigma 24: Mw 3.73 (30 GPa), 2.83 (SFCVM rigidity); rms 1.79 mm;
  surface slip 86% of the crossings; peak 10.2 mm at the surface. To be
  replaced by the 1D layered PyLith model **[planned]**.

## G. Smoothing by L-curve; the clean homogeneous baseline (2026-09-28, evening)

Source tag: [lcurve] `sigma_lcurve.csv` / `sigma_lcurve_slip.npz`
(`drafts/07/sigma_lcurve.py`, final configuration, 13 sigmas); figure
`Figures_2026-09-28/test_sigma_lcurve.png` (`drafts/07/sigma_lcurve_figure.py`).

- **Direction.** In CSI sigma is the prior standard deviation of slip
  (Cm = sigma^2 exp(-d/lambda)): larger sigma = WEAKER smoothing (roughly
  1/alpha in the damped-least-squares convention). Slip norm 6.2 at sigma 4,
  16.1 at 128 [lcurve].
- **The plateau rule fails** on the final configuration: rms keeps falling
  to sigma 128 (1.968 -> 1.744 mm), so "within 0.005 mm of the minimum" just
  returns the largest sigma tried [lcurve].
- **L-curve (textbook, Hansen 1992):** the full weighted misfit
  sqrt(chi2 InSAR + chi2 surface) against the prior-weighted roughness
  sqrt(m^T C_corr^-1 m). The InSAR-only misfit alone gives a false corner
  (it hardly changes, sqrt(chi2) 59-62); the surface chi2 changes most
  (1429 at sigma 4 -> 125 at sigma 32). **Corner at sigma 10.1; selected the
  next grid value above it, sigma 12** (Dani: lean to under-smoothing, and
  textbook) [lcurve].

| sigma | rms (mm) | surface % | peak (mm) | Mw 30 GPa | note |
|---|---|---|---|---|---|
| 4 | 1.968 | 58 | 6.9 | 3.56 | misses the crossings |
| 8 | 1.845 | 78 | 8.9 | 3.64 | |
| **12** | **1.819** | **83** | **9.6** | **3.67** | **selected (corner 10.1)** |
| 16 | 1.807 | 85 | 9.9 | 3.70 | |
| 24 | 1.792 | 86 | 10.2 | 3.73 | earlier choice |
| 48 | 1.770 | 85 | 12.3 | 3.80 | band of slip at ~250 m, just below the tie |
| 128 | 1.744 | 84 | 26.8 | 3.82 | band 27 mm; jagged surface |

- Over the high-curvature range (sigma 8-16) Mw is 3.64-3.70 at 30 GPa: the
  smoothing uncertainty to quote [lcurve].
- **Surface data have no free offset.** CSI's
  `surfaceslip.getTransformEstimator` returns zeros, so the "constant" for
  the surface dataset is a dead parameter (it sits at its bound, -1000) and
  the model fits the absolute surface slip. InSAR constant -0.35 mm [lcurve].
- **Clean notebook** `scripts/07_slip_model.py` / `07_slip_model.ipynb`: the
  homogeneous baseline, machinery ported verbatim from
  `07_slip_model_tests.py`. CHECKs: reproduces the sigma 24 draft
  (`fig2_model_row.npz`) with max |slip diff| 0.00e+00; its in-notebook
  L-curve equals [lcurve]. It runs the L-curve itself, reads the event list
  and the SSE section from 04's outputs, and skips the draft gates on new
  data (and says so). Registry keys `07_hom_*`. Baseline numbers at sigma
  12: see the notebook / registry once run.

### Rerunning the modelling on new data (e.g. through March 2026)

1. On the processing laptop: 01-05, then 06.
2. Copy to this laptop (outputs/ is not in git):
   `outputs/06_event_field_geometry/fp_maps.npz`;
   `outputs/04_event_detection_v2/{event_fraction_by_crossing.csv,
   creep_events.csv, section_lengths.csv}`.
3. Run `07_slip_model.ipynb` (about 1.5 h: 13-sigma L-curve + 24 bootstrap
   runs). Check the corner and the selected sigma: the rule is fixed, the
   value is not.
4. `fig2_model_row.npz` (Fig 2 row 3) is rewritten by the notebook; copy
   it back for 06.
5. `07_slip_model_tests` still hard-codes the five event columns, the SSE
   section and imports `csi_io` from Hayward_DISP-S1 for its mesh gate;
   make it data-driven before rerunning it **[planned]**.
