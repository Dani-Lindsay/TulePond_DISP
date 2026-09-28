# Slip model write-up draft: homogeneous baseline (2026-09-28)

Draft text for the manuscript, in case the homogeneous model stays the final
one. Every number carries a source tag. **[pending]** = from the sigma 12
notebook run that is executing now (registry keys `07_hom_*`); fill from the
registry once it finishes. All numbers are on the data through 2025 and move
with the March 2026 refresh.

Source tags: [nb] `07_slip_model.ipynb` / registry `07_hom_*`;
[lcurve] `sigma_lcurve.csv`; [scan] fixed-depth scan `depth_scan*.csv`;
[scanMw] Mw of the fixed-depth models; [recovery] `recovery_summary.csv`;
[dip] `dip_on_selected_mesh.csv`; [suite] `suite_results.csv`; [SFCVM]
`sfcvm_creep_section_profiles.csv`; [06] notebook 06 registry; [04] notebook
04 products. Details in `07_sweep_lessons_2026-09-28.md`.

---

## 1. Methods: slip inversion of the mean event

**Data.** We invert the fault-parallel component of the mean surface
displacement of the five Tule Pond slow slip events, obtained by combining
the ascending and descending OPERA DISP-S1 event offsets (notebook 06) with
the horizontal direction taken as the mean local strike of the slow-slip
section (146.44°, km 56.3-62.3 from Point Pinole) [06]. Inverting the
fault-parallel component rather than the two line-of-sight tracks excludes
the off-fault vertical signal east of the fault instead of down-weighting it;
it assumes no horizontal motion perpendicular to the fault, which we check
against the model's own prediction (below). We keep pixels with temporal
coherence >= 0.9, within 4 km of the mapped trace and between km 52 and 67,
and exclude the pixels straddling the trace (< 30 m). A plane fitted to the
far field (the slow-slip section +/- 1 km masked) is removed to level the
field, and a constant offset is estimated in the inversion.

**Noise model and sampling.** The data covariance follows from an empirical
semivariogram of pixels more than 1 km from the trace (plane removed),
fitted with a nugget plus two exponential terms. The field is downsampled
with a distance-based quadtree (CSI; Jolivet et al.) from 2 km blocks in the
far field to single 30 m pixels at the trace, and the full covariance of the
samples is built from the fitted model, with the nugget of each block
divided by the square root of its pixel count.

**Fault and Green's functions.** The fault is vertical and follows the
mapped surface trace (24.6 km modelled) to 3 km depth, discretised in
triangles 50 m high at the surface growing by a factor 1.2 per row (at most
1.5 km), with along-strike spacing equal to the row height between km 56 and
63 and coarsening outside (1510 nodes, 2610 triangles) [nb]. Surface
displacements are computed for strike slip on each triangle in a homogeneous
elastic half-space (Meade 2007; Poisson's ratio 0.25) and projected on the
fault-parallel direction. Dip is not resolved by the data: the misfit
decreases monotonically from 60°E to 75°W with Mw between 3.82 and 3.89
[dip, earlier line-of-sight configuration], so we use a vertical fault.

**Surface constraint.** Because the pixels closest to the trace are
excluded, the InSAR alone does not constrain slip in the uppermost cells.
We add the surface slip measured at the fault crossings (the fault-parallel
step across the trace, 30-150 m aperture, mean of the five events [04]) as a
second dataset: a running median every 0.2 km (+/- 0.2 km window), with
uncertainty 1.4826 x MAD / sqrt(n) and a floor of 0.5 mm. The profile is
interpolated to the centre of every surface triangle, with uncertainties
scaled by the square root of the number of triangles per 0.2 km so that its
total weight is unchanged (CSI's default links each point to a single
triangle only). No offset is estimated for the surface data, so the model
fits the absolute surface slip.

**Near-surface tie.** Following Jolivet et al. (2023), who hold slip uniform
over the top 1 km beneath a 1 km data gap, we hold slip uniform from the
surface to 0.2 km depth within each 0.1 km along-strike column (equality
constraints with 0.01 mm uncertainty). Without it the inversion places a
slip maximum ~140 m below the surface that the data do not require
(synthetic tests recover a uniform surface-reaching input with no such
overshoot when the tie is applied) [top-tie tests].

**Regularisation and solution.** We solve for right-lateral slip with
non-negative least squares, with an exponential model covariance (Radiguet
et al. 2010, as implemented in CSI): C_m(i,j) = sigma^2 (lambda_0^2 /
(lambda_x lambda_z)) exp(-d_x / 2 lambda_x) exp(-d_z / 2 lambda_z), with
correlation lengths lambda_x = 0.3 km along strike and lambda_z = 0.05 km
down dip. The short down-dip length means the prior does not impose slip at
depth: deeper slip must be required by the data. The prior standard deviation
sigma (larger sigma = weaker smoothing) is chosen at the corner of the
L-curve (Hansen 1992), the point of maximum curvature of the full weighted
misfit (InSAR + surface constraint) against the prior-weighted model
roughness, computed over 13 values from 4 to 128 mm; the corner lies at
10.1 mm and we use the next value, 12 mm, leaning towards under-smoothing rather than
over-smoothing [lcurve].

**Moment.** We report potency and the moment for a uniform shear modulus of
30 GPa, and a moment with depth-dependent rigidity from the USGS San
Francisco Bay region 3D seismic velocity model v21.1 (Aagaard & Hirakawa
2021): the median shear modulus of the model profiles across the slow-slip
section, applied at the depth of each cell (floored at 0.5 GPa). The slip is
from homogeneous Green's functions in both cases. Mw = 2/3 (log10 M0 - 9.1).
The formal uncertainty is estimated from 24 inversions of the model's own
predictions with noise drawn from the full data covariances.

**Model checks.** (i) The model's predicted fault-normal horizontal motion
is [pending] mm at most, [pending]% of its fault-parallel maximum, supporting
the decomposition. (ii) The notebook reproduces the tested draft model to
round-off (max slip difference 0 mm at sigma 24) [nb]. (iii) Sensitivity to
the model design (section B of the lessons file): smoothing is the largest
lever; correlation lengths moved Mw by 0.06 [suite, earlier configuration;
re-test at each L-curve corner planned]; mesh and sampling by <= 0.13
[suite]; error floor and levelling by < 0.01 [suite].

**Depth resolution.** A fixed-depth scan (uniform slip from the surface to
depth D) fits equally well (within delta chi^2 = 4) for D = 0.85-3 km and
rejects slip confined above ~0.8 km [scan]. In synthetic tests, slip to
1-2 km is recovered as shallow slip with 26-45% of its moment [recovery,
earlier configuration]. The distributed model therefore gives the shallowest
slip distribution consistent with the data, and its moment is a minimum.

Software: CSI (Jolivet et al.; commit 1277bb4).

---

## 2. Results statements

- The mean Tule Pond event is fit by right-lateral slip concentrated in the
  top few hundred metres of the fault between km ~57 and ~62 [figure
  `test_sigma_lcurve.png`], with a maximum of 9.6 mm [lcurve, sigma 12]
  (depth of the maximum [pending]; at the surface in the sigma 24 draft).
- The model reproduces 83% of the surface slip measured at the fault
  crossings over the slow-slip section [lcurve, sigma 12], with an rms
  misfit of 1.82 mm to the fault-parallel field [lcurve, sigma 12].
- The geodetic moment corresponds to Mw 3.67 for a uniform 30 GPa rigidity
  [lcurve, sigma 12] (formal uncertainty +/- [pending]), and Mw [pending] with
  the depth-dependent rigidity of the USGS velocity model, whose top ~500 m
  is soft (0.3-2.7 GPa at 50-500 m) [SFCVM]; potency [pending] m^3.
- Across the high-curvature part of the L-curve (sigma 8-16) Mw varies from
  3.64 to 3.70 [lcurve]; the depth extent is not resolved (0.85-3 km fit
  equally [scan]), and uniform slip over those depths gives Mw 3.91-4.26 at
  30 GPa and 3.24-4.10 with the USGS rigidity [scanMw]. Depth and rigidity,
  not the fit, dominate the magnitude uncertainty.
- Slip reaches the surface: the inversion without the crossing constraint
  already places the largest slip in the uppermost cells, and with the
  constraint the modelled surface slip matches the crossings to within
  [surface rms, pending] mm.
- The fault dip is not resolved (vertical to 75°W and to 60°E fit
  comparably, Mw 3.82-3.89 [dip, earlier configuration]).

### Wording to check before use
- "without the crossing constraint already places the largest slip in the
  uppermost cells": true for the earlier configurations (33-44% of the
  crossing slip at the surface without the constraint [fp; suite]); re-state
  from a run on this configuration, or drop.
- The dip, lambda, mesh and recovery numbers come from earlier
  configurations (LOS input or before the tie); re-run on the final
  configuration with the new data [planned].
- Jolivet et al. 2023 section 4.2 (p. 7 of 17) for the tie; Neyrinck et al.
  2024 (EPSL 648, 119104) used a similar smoothed least-squares approach
  with a layered model; cite where the approach is compared.
