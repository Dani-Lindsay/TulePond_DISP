# PyLith + SFCVM v26 Green's functions: morning summary (2026-09-30)

Overnight, unattended run. Full log with commands and timings: `drafts/pylith/NOTES.md`.
Nothing outside `drafts/pylith/` was modified. No commits.

## What ran

- **Mesh** (`generate_gmsh_vertical.py`, PyLith `gmsh_utils`): fixed. The hang came from the gmsh Box field, and also
  from any MathEval that references another MathEval (gmsh 4.15.2). Fix: three self-contained MathEval size fields
  combined by a gmsh Min field.
  - `default`: 192,634 tets, 100 m fault cells on km 56-63 in the top 1.5 km, 250 m elsewhere, min SICN 0.32, 11 s.
  - `fine`: 445k tets, 60 m cells; used for the convergence check.
  - `shallow`: 485k tets, with a 60 m band within 1 km of the fault in the top 600 m.
- **Materials** (`sfcvm_grid.py`, `write_sfcvm_griddb.py`): SFCVM v26 on a 2.6 M-point SimpleGridDB, written with
  spatialdata's own writer.
  - Alignment: the Hayward boundary is blocks {22 or 23} | 7. South of km ~65 the west block is 23 (Evergreen), not 22.
  - The boundary is shifted onto the trace. The shift is full within 3 km, tapered to 0 at 6 km, and the dip is kept.
  - vs floor 150 m/s. Poisson's ratio capped at 0.40 by lowering vp (vs kept).
  - PyLith's per-cell mu matches the grid (median relative difference 0.000).
- **Green's functions**: PyLith `GreensFns`, right-lateral = −1 × PyLith left-lateral (sign derived and asserted).
  - 2,105 tent impulses on km 54-65 over the full 3 km depth; displacement output at the 2,386 quadtree samples.
  - Dirichlet zero on the sides and bottom, free top.
  - About 33 min per medium with 8 MPI ranks.
  - Media run: homogeneous 30 GPa, SFCVM aligned, SFCVM as is.
- **Inversion**: 07's machinery exec'd unchanged, with CSI `TriangularTents` on the PyLith vertices and G passed in
  via `setGFs`.
  - Same `buildCmXY` prior (lambda 0.1, 0.05 km), `transformation` constant, NNLS, data covariance and crossing-stack
    surface constraint.
  - Sigma for each medium chosen by 07's L-curve rule.

## Validation (homogeneous PyLith)

- **Superposition**: G × slip equals a PyLith forward (kinematic) run to 1e-6 mm.
- **PyLith vs CSI Meade TDEs** (same 10 mm patch, km 57-61, top 1 km):
  - rms difference 0.09 mm on a 2.2 mm signal: 0.017 mm beyond 300 m from the trace, 0.10 mm within it.
  - Fine mesh: 0.077 mm.
  - 6 samples within 15 m of the trace fall on opposite sides of the mesh fault and 07's trace (spline vs polyline).
    Each gives a full-step difference (up to 10 mm). The notebook reports this separately.
- **2D screw dislocation**: 4.8% rms. The 3D end effect is expected.
- **Like for like**: PyLith-homogeneous reproduces the CSI baseline: rms 1.980 vs 1.952 mm, Mw 3.51 vs 3.57.

## Key numbers (default mesh, sigma at each L-curve corner: all corners 9.2-9.6, so sigma 12)

| model | rms (mm) | rms 300 m W | rms 300 m E | chi2 | surface % | Mw 30 GPa |
|---|---|---|---|---|---|---|
| CSI homogeneous (07) | 1.952 | 2.467 | 1.624 | 3890 | 79 | 3.57 |
| PyLith homogeneous | 1.980 | 2.518 | 1.619 | 3981 | 75 | 3.51 |
| PyLith SFCVM aligned | **1.724** | **2.220** | **1.276** | 3954 | 74 | 3.55 |
| PyLith SFCVM as is | 1.950 | 2.535 | 1.458 | 4141 | 74 | 3.58 |

- **Rms**: aligned SFCVM lowers it by 13% against PyLith-homogeneous. It fits both near fields better, west by 0.30 mm
  and east by 0.34 mm.
- **Chi2** (full correlated-noise covariance) barely moves (3954 vs 3981). The gain is mostly in directions the noise
  model down-weights.
- **Alignment matters**: SFCVM as is fits no better than homogeneous by rms, and worse by chi2.
- **Potency**: SFCVM aligned is 1.15× homogeneous (8,973 vs 7,823 m³). This is much smaller than the 1.8-2.4× from the
  2D antiplane test. See the mesh caveat below.
- **Mw with the fault-adjacent SFCVM rigidity** (mean of the two sides): 2.84 aligned, 2.69 for the homogeneous slip.
  Near the surface the rigidity is 0.5 GPa west and 1.3 GPa east (top 100 m).

## Caveats

- Materials are one value per cell. With 100-250 m cells, the SFCVM sediment layering (25 m steps, 200-550 m thick
  west of the trace) is only coarsely represented. The `shallow` mesh run tests this (see below).
- Slip lives almost entirely in the top ~0.5 km in all models. Below that the data do not constrain it.
- Sigma is not numerically comparable with 07's triangle sigma: CSI's lambda0 normalisation depends on the parameter
  spacing.

## Shallow-mesh sensitivity (60 m near-surface band)

- Both PyLith runs on the `shallow` mesh finished: `gf_shallow_sfcvm` in 2 h 11 min and `gf_shallow_homog` in
  2 h 05 min, each 3,373 impulses with 60 m fault cells in the top ~600 m.
- Both G matrices are built (`gf/G_gf_shallow_{homog,sfcvm}.npz`) and pass the sign checks.
- The inversion on these G matrices was not run at the time of writing, because Bash was blocked by a transient
  classifier error at about 05:45. If `gf/results_comparison_shallow.csv` does not exist, run:
  `bash drafts/pylith/run_notebook_shallow.sh`. It takes about 20 min, re-uses all PyLith outputs and writes
  `*_shallow` products plus `runs/notebook_shallow.log`.
- This comparison answers the open question: does resolving the sediment layers change the SFCVM-vs-homogeneous
  result, in particular the 1.15× potency ratio against 1.8-2.4× in 2D?

## Figures (`drafts/pylith/figures/`)

- `fig_i_mesh_fault.png`
- `fig_materials_sfcvm_v26.png`, `fig_materials_boundary_offset.png`
- `fig_iii_validation.png`, `fig_iii_validation_mesh.png`
- `fig_lcurves.png`
- `fig_iv_fit_maps.png`: Fig-2 style, 4 rows × data/model/residual
- `fig_v_slip.png`
- `fig_vi_profile_XX.png`

## Files

- Notebook: `pylith_sfcvm_gf.py` ↔ `pylith_sfcvm_gf.ipynb` (executed). It re-uses PyLith outputs when present.
  - Env vars `PYLITH_MESH_PRESET=shallow` and `PYLITH_MEDIA=homog,sfcvm` switch to the sensitivity run.
- Helper scripts: `generate_gmsh_vertical.py`, `make_points.py`, `sfcvm_grid.py`, `write_sfcvm_griddb.py`,
  `make_impulse_db.py`, `make_kin_slip_db.py`, `build_G.py`, `plot_materials.py`, `run_gf.sh`, `pylith_env.sh`,
  and `cfg/*.cfg`.
- Products:
  - `gf/G_gf_<mesh>_<medium>.npz`: G with provenance
  - `gf/results_comparison.csv`
  - `gf/lcurve_<medium>.csv`
  - `gf/inversion_products.npz`
  - `output/`: PyLith HDF5, about 1 GB. Delete when done.

## Custom code (not from a validated library)

- **`generate_gmsh_vertical.py`**: the size-field expressions and the mesh report. Meshing, marking and quality
  optimisation are gmsh / PyLith `gmsh_utils`.
- **`sfcvm_grid.py`**:
  - a vectorised re-implementation of `drafts/07/sfcvm_extract_dense.get_model_values` (slab reads, nearest point);
  - the block-boundary search;
  - the alignment shift and its taper;
  - the water/no-data fill, the vs floor and the Poisson cap;
  - the grid axes.
- **Spatial databases**: `make_impulse_db.py` / `make_kin_slip_db.py` choose the region values (y window × depth).
  The file format is spatialdata's writer.
- **`build_G.py`**:
  - impulse → vertex mapping (argmax, single unit tent asserted);
  - KD-tree point matching (< 1 mm asserted);
  - projection on 07's look vector;
  - right-lateral sign derivation (asserted).
- **Notebook**:
  - `signed_distance_m` (07's signed `dist_to_trace` flips sides within ~50 m of the trace, so it is not used for sides);
  - `tent_fault_template`: a CSI `TriangularTents` populated by hand with the PyLith vertices and faces;
  - `surface_constraint`: 07's profile mapped to top-edge vertices, plus a selection G;
  - `invert_tents`: 07's `invert` with `setGFs` in place of `buildGFs`; all solver and prior calls are CSI;
  - `potency_m3` / `moment_sfcvm`: exact tent integral, with mu from the PyLith cells either side;
  - `mu_beside_fault`;
  - validation glue: PyLith slip interpolated to CSI centroids, and the 2D screw formula;
  - L-curve corner code, copied from 07 [8].
- **Plotting**: `plot_materials.py` and the notebook figures.

## Next steps (suggested)

1. Look at `fig_iv_fit_maps.png` and `fig_vi_profile_XX.png`. All models under-fit the broad west-side negative
   lobe within ~0.5 km of the trace.
2. Decide whether the rms-vs-chi2 discrepancy matters. A bootstrap (07 [11]) with the SFCVM G would give a formal Mw
   error.
3. If the shallow mesh changes the SFCVM result materially, adopt it (or a finer band) as the reference.
4. Delete `output/*.h5` test runs (`test_*`, `fwd_*`) once satisfied.
