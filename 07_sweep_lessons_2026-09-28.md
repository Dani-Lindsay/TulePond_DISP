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
   motion is negligible **[planned]**.
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
   and an exponential model covariance (buildCmXY). Sigma chosen at the
   onset of the misfit plateau, balancing misfit against over-smoothing:
   sigma = 24 on `smooth12` [sigma]. A proper L-curve plot (misfit vs model
   roughness) **[planned]**.
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
- Uniform slip tied over the top ~200 m of the fault (test 100 and 500 m),
  after Jolivet et al. (2023).
- Fault-normal motion check; crossing vs field strike check.
- Proper L-curve figure for sigma.
- Recovery test with the interpolated constraint and fault-parallel input.
- Potency; event-to-event spread of surface slip and extent (from 04).
- Rerun of `07_slip_model_tests` on the final configuration; drop panels B
  and C from `fig07_slip_model`.
