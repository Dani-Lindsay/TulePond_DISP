# What the 07 slip-model sweep tests taught us (2026-09-28)

Written from Dani's draft of 2026-09-28, with the numbers filled in and
three corrections marked **[corrected]**. All numbers are provisional:
they come from the test products in `outputs/07_slip_model_tests/`, not
from the registry, and most use uniform 30 GPa rigidity, homogeneous
elastic Green's functions and CSI's single-triangle surface link. Replace
them with registry values once the final (1D layered, fault-parallel)
model runs.

Sources: `suite_results.csv` (batch 2, 37 configurations, every CHECK
passed), `dip_on_selected_mesh.csv`, `sigma_selection.csv`,
`depth_scan*.csv`, `recovery_summary.csv`, `fp_inversion_*.csv`,
`sfcvm_creep_section_profiles.csv`, `interim_logs_2026-09-28.txt`.
Figures: `interim_07_results.png` (this folder),
`../Figures_2026-09-25/fig07_sensitivity_suite.png`,
`../Figures_2026-09-25/draft_residuals_los_vs_fp.png`.

## 1. Depth of slip: not resolved, and it matters for Mw

- Rule of thumb: surface displacement from slip reaching depth D changes
  most at distances ~1-2 D from the fault. Our data extend 4 km from the
  trace, so coverage is enough in principle for depths of ~2-4 km.
- Coverage is not the limit; signal to noise is. Slip reaching 0.85 km and
  slip reaching 3 km differ by < 0.7 mm in descending LOS at 0.5-4 km from
  the fault, less than the spatially correlated noise at the same
  distances. Widening the window (2 -> 6 km, legacy test) adds nothing.
- Fixed-depth test: uniform slip from the surface to D (0.05-5 km,
  amplitude solved, ~7.7 mm for D >= 1 km). Fits within Δχ² = 4 of the
  best for **D = 0.85-3 km** (best 1.5 km) on the current fields, and
  0.75-1.65 km (best 1.1 km) on the legacy fields. Slip confined above
  ~0.8 km is rejected (Δχ² = 28 at 0.5 km, > 150 at 0.25 km). The same
  range holds on a 5 km-deep mesh and without the far-field levelling.
- The distributed (smoothed) model cannot tell depths apart: synthetic 10
  mm input to 0.3, 0.5, 0.8, 1.0, 1.5 or 2.0 km comes back with nearly
  the same shallow profile. Its moment recovery falls from 1.25 (0.3 km)
  to 0.30 (1.5 km) and 0.26 (2 km), i.e. Mw under-reported by up to 0.35-0.39
  if slip is deep.
- **[corrected] The moment is NOT robust to depth.** Across the equally
  fitting fixed-depth models, Mw is 3.91 (0.85 km), 4.04 (1.5 km) and 4.26
  (3 km) at 30 GPa; with the USGS rigidity it is 3.24, 3.68 and 4.10. Depth
  (together with rigidity) is the largest source of magnitude
  uncertainty. This is why the prediction is stated in surface slip, and
  Mw is given as a range with the rigidity stated.

## 2. Vertical motion and the fault-parallel input

- The decomposed mean event shows vertical motion on the east side of the
  fault (Fig 2 F, H). It is same-sign in both geometries, so it cannot be
  strike slip on the fault; freeing the rake does not fit it (χ² worse,
  strike-slip Mw 3.67 vs ~3.83). We treat it as a non-tectonic or
  off-fault signal outside the model's scope.
- So we invert the fault-parallel component (06 decomposition) instead of
  the two LOS tracks. The ascending-LOS inversion leaves a coherent
  residual patch just east of the fault; the fault-parallel inversion
  does not (residual maps). Mw differs by 0.07 (3.75 fault-parallel, 3.82
  LOS, both at 30 GPa, same settings). The fault-parallel field is
  noisier (pixel noise 0.99 mm vs 0.6-0.7 mm per track).
- Assumption: no horizontal motion perpendicular to the fault (the
  decomposition solves two components from two geometries). To check:
  the model's predicted fault-normal motion should be negligible, as
  Neyrinck et al. (2024) showed for their N-S component.

## 3. The asymmetry across the fault

- Profiles through the mean event are asymmetric in both tracks, and in
  the fault-parallel component itself (X-X': W -5.5 mm, E +2.1 mm).
- Three candidate sources:
  1. Vertical and/or fault-perpendicular motion leaking into LOS.
     Ascending sees vertical strongly (|LOS·up| 0.76) and fault-parallel
     weakly (0.28), so a small vertical or fault-normal signal cancels
     more of its fault-parallel signal; the asymmetry is stronger in
     ascending. This cannot explain the fault-parallel asymmetry, which
     has the vertical removed.
  2. An off-fault vertical source east of the fault (same reasoning;
     affects LOS, not the fault-parallel field).
  3. A rigidity contrast across the fault (section 5): the softer side
     deforms more for the same slip. This can produce the
     fault-parallel asymmetry, and the sense matches (softer west side,
     larger west-side motion).
- Dip: the InSAR is insensitive. On the selected mesh, rms falls
  monotonically from 60°E (1.24 mm) to 75°W (1.13 mm), descending
  near-field (< 200 m) from 1.18 to 0.96 mm; there is no minimum, so this
  is the model fitting the asymmetry, not a dip measurement. Mw 3.82-3.89
  across 60°E-75°W. Relocated seismicity dips ~75-80°E at 2-8 km north of
  the segment, below the slip. We adopt a vertical fault.

## 4. Smoothing, mesh, sampling, prior

- Smoothing is the strongest modelling lever on Mw: σ 8-64 (with the
  surface constraint) gives Mw 3.59-3.96 at nearly the same rms (1.15-1.19
  mm). σ = 24 by the rms-plateau rule. Neyrinck et al. report the same
  dependence of amplitude on smoothing.
- Correlation lengths (λ strike 0.15-0.6 km × λ depth 0.05-0.15 km): Mw
  3.81-3.87.
- Mesh: surface cells of 100, 50 and 30 m, plateau vs smooth growth, a
  deep plateau: Mw 3.86-3.99 across meshes at matched misfit. The
  30 m cells add nothing over 50 m. Prior strength must be re-chosen per
  mesh.
- Quadtree floor 30/40/60 m and a uniform 100 m grid: Mw 3.95-3.99 (same
  mesh), negligible.
- Prior form: minimum-norm (Mw 4.19) and effectively unregularised (4.34,
  unphysical 300 mm peaks) are stress tests, not candidates; they show
  why a least-squares distributed model needs regularisation.
- Far-field levelling: the ascending mean event had a ~2.5 mm tilt along
  strike; a plane fitted outside the creep section (± 1 km) removes it.
  Mw unchanged (3.83 with and without); it matters for the residuals.

## 5. Rigidity from the USGS SFCVM v21.1

- Creep section, 0.2-3 km either side of the trace (median):
  - top ~500 m: **west softer** (Quaternary/Tertiary sediment, Vs 400-900
    m/s, 0.3-1.8 GPa, from 50 m down) than east (Great Valley Sequence,
    Vs ~900-1100 m/s, 1.8-2.7 GPa); east ~1.3-5× stiffer.
  - 0.75-2 km: the contrast **reverses**: west stiffer (Franciscan, 14-24
    GPa) than east (5-17 GPa).
  - below ~3 km: both ~28-30 GPa.
  - **[corrected]** Dani's draft had west stiffer / east softer at the
    surface; it is the other way round in the top 500 m.
- Averages: ~4 GPa over 0-1 km, ~8 GPa over 0-3 km (vs 30 GPa assumed).
  Neyrinck et al. used 18 GPa over the top 4 km at Izmit.
- **[corrected] Softer rock gives a LOWER moment for the same slip**
  (moment = rigidity × area × slip). With uniform-rock Green's functions,
  the inversion's slip does not depend on rigidity, so applying the
  depth-dependent rigidity lowers Mw from 3.82 to ~3.11 (LOS) and 3.75 to
  ~2.86-3.03 (fault-parallel): the slip sits in the top ~500 m, where
  rigidity is 1-2 GPa. With layered Green's functions the inversion needs
  MORE slip near the surface to fit the same data (Dianala et al. 2020;
  Jolivet et al. 2015), which pushes Mw back up part of the way. The net
  effect is what the 1D layered PyLith model will tell us.
- The contrast is also the leading explanation for the fault-parallel
  asymmetry (section 3).

## 6. Surface constraint

- Without it, the InSAR-only model reproduces 44% (LOS) / 33%
  (fault-parallel) of the crossing-stack surface slip over the creep
  section; with it, 75% / 68%.
- CSI's default surface link ties each surface point to the single
  nearest triangle (27 of 227 creep-section surface triangles). We
  interpolate the crossing profile to every surface triangle, with errors
  scaled so the total weight is unchanged; in synthetics the top cell
  then recovers the input (10.6-11.1 mm for 10 mm), and the sub-surface
  overshoot drops from 17-23% to 12-16%.
- Next: tie slip uniform over the top ~200 m in each along-strike column
  (after Jolivet et al. 2023, who tie 0-1 km); test 100 and 500 m.

## Dropped (Dani, 2026-09-28)

Ascending-only tests and track weighting: explicit down-weighting or
masking the ascending east side changes Mw by < 0.02; no weighting is
applied.
