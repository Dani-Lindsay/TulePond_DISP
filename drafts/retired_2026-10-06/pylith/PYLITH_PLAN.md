# PyLith 3D heterogeneous Green's functions for the Tule Pond slow-slip inversion: plan

Research note, 2026-09-29. Read-only research: nothing is installed yet. Items I checked in the PyLith v5.0.2 source tree, release assets, Crossref or CIG forum threads are stated as fact. Items marked **[UNVERIFIED]** are my inference or rough estimates.

---

## (a) Verdict

**This is moderate work: most of the pieces already exist in PyLith. Plan on about 2 to 3 focused weeks (8 to 15 working days) to get a validated heterogeneous G into CSI. Compute time is hours, not days.**

- **What PyLith already provides.** PyLith v5.0.2 (released 2026-08-15, with a native macOS arm64 binary) has a dedicated Green's function problem type, `pylith.problems.GreensFns` with `pylith.faults.FaultCohesiveImpulses`. It writes surface displacement at an arbitrary list of points (`OutputSolnPoints`) to HDF5. There is a worked 3D Green's function plus inversion example: `examples/subduction-3d` step07 with `invert_slip.py`. There is also a Gmsh Python example for vertical faults extruded from a map trace: `examples/crustal-strikeslip-3d/generate_gmsh.py`.
- **The hard parts are not PyLith itself.** They are:
  1. meshing a curved, surface-breaking fault at 50 to 100 m resolution inside a 100 km box without distorted cells;
  2. converting SFCVM to a spatialdata grid, because spatialdata has no GeoModelGrids reader;
  3. deciding how to treat near-incompressible, very soft near-surface sediment;
  4. reconciling PyLith's per-vertex "tent" slip basis with CSI's constant-slip triangles.
- **Each run is cheap.** The Jacobian is assembled once and reused for every impulse: `GreensFns.cc` calls `SNESSetLagJacobian(_snes, -2)`. Each impulse is therefore one preconditioned iterative solve.
- **Rough compute estimate [UNVERIFIED].** About 0.3 to 0.6 M tets gives about 0.2 to 0.4 M DOF. That is about 1 to 5 s per impulse, so about 1 to 4 h serial for about 1500 to 2600 impulses. MPI or impulse-splitting across cores brings this under an hour.

---

## (b) Step-by-step implementation path

### Step 0. Install and smoke test (0.5 day)

- **Download.** Get the release asset `pylith-5.0.2-macOS-12.0-arm64.tar.gz` (1.10 GB compressed; needs macOS 12 or later) with `curl -L -O https://github.com/geodynamics/pylith/releases/download/v5.0.2/pylith-5.0.2-macOS-12.0-arm64.tar.gz`. Downloading with curl avoids the macOS quarantine flag. Otherwise run `xattr -r -d com.apple.quarantine`.
- **Set up the environment.** Run `source setup.sh` from a **bash** shell with a clean PATH and no conda environment active. `setup.sh` is bash-only and the user's shell is zsh.
- **Self-contained environment.** The binary carries its own Python, PETSc, MPI, h5py, spatialdata and (per the install docs) Gmsh and PyVista. Gmsh on macOS is **[UNVERIFIED]**: check `python -c "import gmsh"` inside the environment.
- **Other install routes do not work:**
  - there is no conda-forge `pylith` package (the anaconda.org API returns "not found");
  - the Docker Hub `geodynamics/pylith` image is v2.2.2 and amd64-only (2019);
  - a source build via `pylith_installer` is possible but unnecessary.
- **Smoke test.** Run `examples/crustal-strikeslip-3d` (`generate_gmsh.py --write`, then `pylith step01_slip.cfg`) and `examples/subduction-3d` `pylith step07a_leftlateral.cfg mat_elastic.cfg`.
- **Risks.**
  - A conda base environment leaking into PATH or PYTHONPATH.
  - The unpacked size is **[UNVERIFIED]** (several GB).
  - PETSc in the binary is built by default with metis, parmetis, f2cblaslapack and ml, per the `pylith_installer` `configure.ac` default `petsc_options`. So **no MUMPS or SuperLU_dist**: a reusable parallel direct factorization is not available out of the box.

### Step 1. Coordinate frame (0.5 day)

- **Choose one projected Cartesian CRS for everything:** mesh, points, spatial databases and impulse database.
- **Recommendation:** a local transverse Mercator centred on Tule Pond, e.g. `+proj=tmerc +datum=WGS84 +lon_0=<site> +lat_0=<site> +k=1` (the pattern used in `examples/subduction-3d/pylithapp.cfg`). This makes x≈east, y≈north, with negligible grid convergence. UTM 10N (`EPSG:32610`) also works; Fremont is about 1° east of the zone 10 central meridian.
- **Config:** `reader.coordsys = spatialdata.geocoords.CSGeo`, `coordsys.crs_string = ...`, `space_dim = 3`, set on the mesh reader and repeated on the `PointsList` reader.
- **Risk:** mismatched CRS between the mesh and the spatialdb or points. This is the most common forum failure (CIG thread 3629). Assert the round trip in Python before running.

### Step 2. Mesh in Gmsh via the PyLith `gmsh_utils` API (2 to 4 days; this is the hardest step)

- **Template:** `examples/crustal-strikeslip-3d/generate_gmsh.py`. It subclasses `pylith.meshio.gmsh_utils.GenerateMesh` and implements `create_geometry()`, `mark()` and `generate_mesh()`. Run it with `generate_gmsh.py --write [--gui]`.
- **Geometry.**
  - `occ.add_box` for a domain of about 100 × 100 × 50 km. Brad Aagaard's advice on the forum: domain edges should be "several fault lengths away" (thread 2645).
  - Build the fault from the projected trace points with `occ.add_spline` and `occ.extrude(0, 0, -Z_fault)`. Take Z_fault = 3 km to match the current CSI mesh (`mesh_depth_km = 3.0`, 24.6 km of trace).
  - `occ.fragment([(3, domain)], [(2, fault)])` so that cell faces conform to the fault.
- **Physical groups** via `MaterialGroup` and `BoundaryGroup`:
  - material tag 1;
  - `boundary_south/east/north/west/bottom/top` with tags 10 to 15;
  - `fault` with tag 20 and `dim=2`.
- **Buried edges.** The fault's bottom edge and its two lateral ends are buried; the top edge breaks the surface. **In v5.0.2 buried fault edges are marked automatically**; CHANGES.md says "Removed manual marking of buried edges (use automatic marking)". No `edge` label is needed, although `FaultCohesiveImpulses.edge` still exists. Slip is forced to zero on buried-edge vertices, so those vertices get no impulse.
- **Sizing.** Turn off `MeshSizeFromPoints`, `MeshSizeFromCurvature` and `MeshSizeExtendFromBoundary`. Then:
  - `Distance` field on the fault surface, combined through `MathEval` with `GenerateMesh.get_math_progression(field, min_dx, bias)`;
  - a second `Distance` field on the top surface over the data footprint;
  - take the `Min` of the two;
  - finish with `self.improve_quality()` (added in v5.0.0).
- **Suggested sizes:**
  - 50 m on the fault in the fine section (km 56 to 63) and down to about 1.5 km;
  - 100 to 200 m on the rest of the fault;
  - about 100 m on the free surface within the data area;
  - bias 1.10 to 1.15, growing to about 5 km at the boundaries.
- **Cell-count estimate [UNVERIFIED, back of envelope].** Tet volume is about 0.118 h³. For linear growth h = h0 + g·r around a fault patch of area A, N ≈ A/(0.118 g h0²).
  - fine fault patch (7 km × 1.5 km, h0 = 50 m, g = 0.15): about 0.24 M;
  - rest of fault (h0 = 200 m): about 0.1 M;
  - surface data area (8 × 8 km, 100 m, one-sided): about 0.2 M;
  - **total about 0.4 to 0.6 M tets, about 0.1 M vertices, about 0.3 M DOF.**
  - Going to 30 m everywhere near the fault multiplies the count by about 3.
- **Impulse count.** One impulse per non-edge fault vertex (basis order 1) or per fault cell (basis order 0) in the impulse region.
  - 50 m cells over 7 km × 1.5 km gives about 4,000 vertices: too many for the data.
  - 100 m cells gives about 1,000.
  - Size the fault mesh for the slip parameterisation you want, and refine the volume mesh separately (Gmsh allows smaller cells off the fault than on it only through the fields).
- **Risks:**
  - sliver or needle tets where a curved spline fault meets the top surface at a low angle;
  - surface-entity IDs after `fragment` must be found in the GUI (the example hard-codes them);
  - spline overshoot on a noisy trace, so smooth or decimate the trace first;
  - a solver-convergence warning from the developer: "The solver struggles when the fault cuts through the entire domain with a very fine grid" (thread 2645). Our fault does not cut through, but check `ksp_monitor_true_residual`.

### Step 3. Material spatial database from SFCVM v26.0 (1 to 2 days)

- **No direct GeoModelGrids support.** The PyLith development plan lists "Add support for GeoModelGrids implementation of spatial databases for 3D seismic velocity models" at **0%** (`docs/intro/development-plan.md`). spatialdata v4.0.1 offers `UniformDB`, `SimpleDB`, `SimpleGridDB`, `CompositeDB`, `AnalyticDB`, `UserFunctionDB` and `SCECCVMH`; there is no GeoModelGrids reader. GeoModelGrids itself is not on conda-forge either.
- **Route:** reuse the existing h5py reader in `drafts/07/sfcvm_extract_dense.py`, which already handles the CRS, origin, `y_azimuth` rotation and the four resolution blocks. Sample SFCVM onto an axis-aligned, **non-uniform** grid in the mesh CRS and write a `SimpleGridDB` ASCII file:
  - header `#SPATIAL_GRID.ascii 1`, with `num-x/y/z`, `value-names = density vs vp`, units `kg/m**3 m/s m/s`, `cs-data = geographic { crs-string = ... }`;
  - x/y spacing: 50 m within 2 km of the trace, growing to km;
  - z spacing: 25 m to 500 m depth, then coarser;
  - roughly 10⁶ to 3×10⁶ nodes, i.e. a 100 to 300 MB ASCII file (acceptable).
  - Make the grid cover the **whole** mesh domain. Pad with the deepest or edge values and assert there are no NaNs.
- **Config:**
  - `db_auxiliary_field = spatialdata.spatialdb.SimpleGridDB`, `filename = mat_sfcvm.spatialdb`, `query_type = linear` (default `nearest`);
  - `auxiliary_subfields.density.basis_order = 0` and `bulk_rheology.auxiliary_subfields.{bulk,shear}_modulus.basis_order = 0`, which gives one value per cell (as in the examples).
  - Required values: `density`, `vs`, `vp`, as in `crustal-strikeslip-3d/pylithapp.cfg`.
- **Low-Vs and high-Poisson risk (science decision).**
  - Saturated Quaternary sediment with Vs ≈ 150 m/s and Vp ≈ 1500 m/s has ν ≈ 0.495.
  - Linear tets lock volumetrically as ν → 0.5, which makes the response too stiff and worsens conditioning.
  - For a slow slip lasting weeks to months, drained moduli are arguably more appropriate than the undrained moduli seismic Vp implies. Options: cap ν (e.g. 0.40 to 0.45, i.e. derive Vp from Vs), or use `displacement.basis_order = 2` (about 8× the DOF).
  - The rigidity contrast spans about 40× near the surface and about 750× against deep crust. Set `problem.scales.shear_modulus` to about 1 GPa and `length_scale` to the fault cell size. The v5.0.2 notes say "the length scale should be set to the discretization size on the fault; this is related to the preconditioner".

### Step 4. Green's-function configuration (1 day)

**Template:** `examples/subduction-3d/step07a_leftlateral.cfg` plus `pylithapp.cfg`. The skeleton:

```cfg
[pylithapp.problem.mesh_initializer.phases.read_mesh]
reader = pylith.meshio.MeshIOPetsc
reader.filename = mesh/mesh_tet.msh
reader.coordsys = spatialdata.geocoords.CSGeo
reader.coordsys.crs_string = +proj=tmerc ...
reader.coordsys.space_dim = 3

[pylithapp]
problem = pylith.problems.GreensFns

[pylithapp.greensfns]
label = fault
label_value = 20
solution = pylith.problems.SolnDispLagrange
scales.length_scale = 0.1*km          ; ~fault cell size (v5.0.2 guidance)
solution_observers = [points]         ; NOT domain/groundsurf (GBs per impulse)
solution_observers.points = pylith.meshio.OutputSolnPoints
interfaces = [fault]
interfaces.fault = pylith.faults.FaultCohesiveImpulses
bc = [bc_west, bc_east, bc_south, bc_north, bc_bottom]   ; ZeroDB rollers as in examples

[pylithapp.problem.solution_observers.points]
label = insar
data_fields = [displacement]
reader.filename = insar_points.txt     ; "name x y z" per line
reader.coordsys = spatialdata.geocoords.CSGeo
reader.coordsys.crs_string = +proj=tmerc ...
reader.coordsys.space_dim = 3

[pylithapp.problem.interfaces.fault]
label = fault
label_value = 20
impulse_dof = [1]                      ; 1 = left-lateral (0 opening, 2 reverse)
threshold = 0.5*m
db_auxiliary_field = spatialdata.spatialdb.SimpleDB   ; 1 m inside impulse region, 0 outside
db_auxiliary_field.iohandler.filename = impulse_region.spatialdb
auxiliary_subfields.slip.basis_order = 1
observers.observer.data_fields = [slip, traction_change]
```

- **Outputs:** `output/<name>-insar.h5` and `output/<name>-fault.h5`.
- **HDF5 layout** (as read by `examples/subduction-3d/invert_slip.py`):
  - `geometry/vertices` (npts×3);
  - `vertex_fields/displacement` with shape (n_impulse, npts, 3) for points;
  - the fault file has `geometry/vertices`, `viz/topology/cells` and `vertex_fields/slip` with shape (n_impulse, nvert, 3); component 1 is left-lateral.
  - The "time" axis is the impulse index.
- **Build G** with `G[:, k] = e_fp · disp[k, :, :]`, i.e. project east/north onto the local fault-parallel unit vector exactly as the CSI data are defined. Right-lateral positive means `G_rl = -G_ll`; assert this in a test.
- **Impulse to vertex mapping:** use `argmax |slip[k,:,1]|`. Impulses are ordered "based on the processor and degree of freedom number" (Aagaard, thread 3152), **not** by vertex. Match output points back to the data by coordinates with a KD-tree (`invert_slip.py` `matchCoords`); do not assume input order.
- **Risks:**
  - `ref_dir_1 = [0,0,1]` defines the along-strike direction t1 = r1 × n. Check the sign on a known case. For a vertical fault this default is fine.
  - Points exactly on the trace are ambiguous across the split. Keep the current near-fault exclusion or test points within one cell of the trace.
  - Points must lie inside the domain. Use a flat top at z = 0 and z = 0 for points, consistent with the "squash negligible" note in `sfcvm_extract_dense.py`.

### Step 5. Basis-function reconciliation with CSI (1 to 2 days)

- **Default basis (`slip.basis_order = 1`).** Each impulse is a unit tent on one fault vertex. The `strikeslip-2d/step05_greensfns.cfg` comment says "Currently, only basis order = 1 results in appropriate output for inversion."
- **Conflicting developer advice.** In a Sept 2026 CIG thread (4625, PyLith v5), Aagaard says basis order 0 slip impulses with basis order 1 (or 2) displacement give "uniform slip within the fault cell", and the higher solution order affects "only the accuracy ... rather than introducing slip into the adjacent fault cells." **These two statements conflict; test basis order 0 before relying on it.**
- **Options:**
  - **A (recommended).** Invert on PyLith's own fault mesh with vertex (tent) parameters. Rebuild the Laplacian and damping from `viz/topology/cells`, then hand G to CSI via `Fault.setGFs(data, strikeslip=[G, None, None])` (this signature exists in the local `tools/external/csi/csi/Fault.py`, line 2137). The CSI fault object must be built on the same nodes; that is **[UNVERIFIED]** plumbing.
  - **B.** Basis order 0 on a PyLith fault mesh that reproduces the CSI triangles, giving one impulse per CSI triangle. This needs the Gmsh fault surface built from the CSI node rows (per-triangle plane surfaces), which is hard with OCC `fragment`.
  - **C.** Run a fine PyLith mesh with basis 0, then area-weight-aggregate cells into CSI triangles by superposition. This is exact only if the cells nest.

### Step 6. Runs and performance (0.5 to 1 day of setup; compute is hours)

- **Default solver** for elasticity with a fault (`share/settings/solver_elasticity_fault.cfg`): `pc_type = gamg`, `mg_fine_pc_type = vpbjacobi`, `dm_reorder_section_type = cohesive`, `ksp_gmres_restart = 100`, `ksp_atol = 1e-7`, `snes_atol = 5e-7`. Keep the defaults; the developer strongly advises against overriding them.
- **What is reused:** the Jacobian and the GAMG setup (lag −2). Each impulse costs GMRES iterations only. The subduction-3d log shows 10 to 11 KSP iterations per impulse on a coarse mesh; expect more with the 40 to 750× rigidity contrast.
- **Parallel options:**
  - (i) `pylith --nodes=8 step_gf.cfg` using the bundled MPI;
  - (ii) embarrassingly parallel: split the impulse-region spatialdb into K disjoint along-strike blocks and run K single-process jobs. Each block must own its boundary vertices exactly once (the threshold on amplitude decides).
  - The docs recommend the same split pattern for step07a/7b.
- **Benchmark first:** run 20 impulses with `--petsc.log_view` to measure s/impulse, then scale up. `progress_monitor` writes an ETA.
- **Output size:** 2400 points × 3 components × 2600 impulses × 8 B ≈ 150 MB (fine). Never enable the `domain` observer in the GF run.

### Step 7. Integration and science runs (1 to 2 days)

Rerun the L-curve and bootstrap in `07_slip_model` with G_pylith. Material variants (e.g. shifting the SFCVM contrast 175 m west onto the mapped trace) only need a new spatialdb and a rerun, not a new mesh.

---

## (c) Minimal layout (inside TulePond_DISP)

```
pylith/
  README.md                  # how to run; PyLith version; CRS string
  env.sh                     # bash; sources pylith-5.0.2 setup.sh with clean PATH
  mesh/generate_gmsh.py      # App(GenerateMesh): trace -> spline -> extrude -> fragment; fields
  mesh/faulttrace_local.txt  # projected trace (written by prep script, provenance header)
  spatialdb/make_sfcvm_griddb.py   # h5py SFCVM -> SimpleGridDB (reuses drafts/07 reader)
  spatialdb/make_impulse_db.py     # impulse region (1 m / 0 m) SimpleDB
  spatialdb/make_points.py         # quadtree lon/lat -> "name x y 0" PointsList
  cfg/pylithapp.cfg          # mesh reader, CRS, BCs, scales, material
  cfg/mat_homog.cfg          # UniformDB (for validation)
  cfg/mat_sfcvm.cfg          # SimpleGridDB
  cfg/gf_ll.cfg              # GreensFns + FaultCohesiveImpulses + OutputSolnPoints
  cfg/fwd_kin.cfg            # TimeDependent + FaultCohesiveKin (validation forward runs)
  build_G.py                 # HDF5 -> G (fault-parallel), vertex map, provenance -> .npz
  tests/test_pylith_G.py     # pytest anchor tests = validation gates below
  output/                    # PyLith HDF5 (git-ignored)
```

The `mesh/` and `spatialdb/` scripts run in the PyLith environment or the user's existing environment; `build_G.py` and the tests run in the CSI environment. The `.npz` holding G records the PyLith version, mesh hash, spatialdb hash and CRS.

---

## (d) Validation gates (each is a pytest with a numeric tolerance)

1. **Install gate.** `crustal-strikeslip-3d` step01 and `subduction-3d` step07a run on arm64.
2. **Half-space gate.** Homogeneous `UniformDB` (e.g. μ = 30 GPa, ν = 0.25), planar vertical fault. A forward `FaultCohesiveKin` uniform-slip rectangle, compared with Okada (or CSI Meade TDEs on the same rectangle) at the data points, should agree to 2 to 3% RMS of signal. This checks domain size, sign and CRS. Remember that PyLith slip tapers to zero at buried-edge vertices, so compare against an Okada rectangle shrunk by about half a cell, or compare away from the tips.
3. **Boundary gate.** Doubling the domain changes G by less than 1%.
4. **Mesh-convergence gate.** Halving the near-fault h changes predictions by less than 2 to 3% (it matters most for points close to the trace).
5. **Superposition gate.** Σ_k a_k G[:,k] equals a forward `FaultCohesiveKin` run with the same vertex slip a (tests assembly, ordering and point matching).
6. **CSI cross-check (homogeneous).** Predictions of the current best CSI slip model, interpolated to PyLith vertices: PyLith G against CSI G within about 3%. Inverting the same data with both Gs gives similar moment and slip pattern.
7. **Sign gate.** A unit right-lateral impulse (−LL) gives positive fault-parallel displacement on the side that matches the CSI convention.
8. **Material gate.** Output of the material info/aux fields (μ per cell) reproduces `outputs/07_slip_model_tests/sfcvm_creep_section_dense_v26-0.csv` on the profile lines. Assert that min μ and max ν equal the chosen caps.
9. **Optional layered benchmark.** A 1D layered model compared with an independent layered-half-space code, or with the existing 2D antiplane SFCVM result (`drafts/07/antiplane_sfcvm_2d*.py`) far from the fault ends.

---

## (e) Open decisions for you

1. **Fault extent and parameterisation.** Keep the 24.6 km × 3 km CSI extent, or model 6 to 8 km × 1.5 to 2 km with impulses only there? Choose tent-vertex parameters (option A, native, recommended) or CSI triangles (B or C). This sets the fault mesh size (about 100 m) and the impulse count.
2. **Drained or undrained near-surface moduli.** Cap ν (value?) or keep SFCVM Vp and pay for P2 elements.
3. **SFCVM contrast location.** Use as is (175 m east of the trace) or shift it to the mapped trace. This could be a sensitivity pair, since only the spatialdb changes.
4. **Topography.** A flat top is recommended (tens of m of relief).
5. **Near-trace data points.** Keep the current exclusion distance; PyLith output within about one cell of the fault is less accurate.
6. **Compute.** Laptop MPI is probably enough. Moving to a Linux box (the same binary exists for x86_64) is the fallback.

---

## (f) Sources

**PyLith (checked in the v5.0.2 tree or release metadata):**
- Releases: <https://github.com/geodynamics/pylith/releases>. v5.0.2 dated 2026-08-15; asset `pylith-5.0.2-macOS-12.0-arm64.tar.gz` (1,103,107,276 B). CHANGES.md notes that v5.0.2 is the last macOS x86_64 binary.
- Install guide: <https://github.com/geodynamics/pylith/blob/v5.0.2/docs/user/install/index.md> (also rendered at pylith.readthedocs.io).
- GreensFns and FaultCohesiveImpulses docs: `docs/user/components/problems/GreensFns.md`, `docs/user/components/faults/FaultCohesiveImpulses.md`, `docs/user/physics/faults/slip-impulses.md`, `docs/user/problems/problems.md`.
- Examples:
  - `examples/strikeslip-2d/step05_greensfns.cfg`, `invert_slip.py`, `slip_impulses.spatialdb`, `gnss_stations.txt`;
  - `examples/subduction-3d/step07a_leftlateral.cfg`, `step07b_reverse.cfg`, `invert_slip.py`, `invert_slip.cfg`, `generate_gmsh.py`, plus the doc page `docs/user/examples/subduction-3d/step07-greensfns.md`;
  - `examples/crustal-strikeslip-3d/generate_gmsh.py`, `pylithapp.cfg`. Tree: <https://github.com/geodynamics/pylith/tree/v5.0.2/examples>.
- Solver defaults and tolerances: `docs/user/run-pylith/petsc-options.md`. `SNESSetLagJacobian(-2)`: `libsrc/pylith/problems/GreensFns.cc`.
- Scales: `docs/user/components/scales/QuasistaticElasticity.md`. Buried edges: `docs/user/physics/faults/index.md` and CHANGES.md v5.0.2 / v4.0.0.
- Development plan (GeoModelGrids spatialdb at 0%): <https://github.com/geodynamics/pylith/blob/v5.0.2/docs/intro/development-plan.md>.
- PETSc build defaults: <https://github.com/geodynamics/pylith_installer/blob/main/configure.ac>.
- spatialdata v4.0.1 (SimpleGridDB format and `query_type`): <https://github.com/geodynamics/spatialdata>, `docs/user/spatialdb/simplegriddb.md`.

**CIG forum threads (read via the Discourse JSON API):**
- Basis order 0 vs 1 impulses (2026-09-16): <https://community.geodynamics.org/t/how-to-calculate-greens-function-in-fault-cell/4625>
- Impulse ordering (2023-10-23): <https://community.geodynamics.org/t/how-to-know-the-order-of-the-greens-function/3152>
- Domain size, fine grids and PETSc overrides (2022-10-25): <https://community.geodynamics.org/t/petsc-ts-options-ignored-when-computing-greens-functions-gmres-recursion-formula-far-from-computed-residual-norm/2645>
- CRS mismatch (2024-09): <https://community.geodynamics.org/t/using-pylith-to-calculate-the-subduction-zone-greens-functions/3629>
- PyLith GFs converted for TDEFNODE; C. Williams points to a GJI paper at academic.oup.com/gji/article/240/2/886 whose title and authors I could **not** verify (2026-04-22): <https://community.geodynamics.org/t/taking-greens-functions-from-step-7-a-and-b-subduction-3d-to-use-in-tdefnode/4520>

**Papers (metadata verified via Crossref; I did not read the full texts):**
- Aagaard, Knepley & Williams (2013), "A domain decomposition approach to implementing fault slip in finite-element models of quasi-static and dynamic crustal deformation", JGR Solid Earth, doi:10.1002/jgrb.50217. This is the PyLith method paper.
- Williams & Wallace (2015), "Effects of material property variations on slip estimates for subduction interface slow-slip events", GRL, doi:10.1002/2014GL062505. It uses finite-element GFs with the NZ-wide velocity model for SSE inversions; the abstract says "a finite element code". PyLith is the likely code but **[UNVERIFIED]**. This is the closest template for an SSE inversion with heterogeneous GFs.
- Hsu, Simons, Williams & Casarotti (2011), "Three-dimensional FEM derived elastic Green's functions for the coseismic deformation of the 2005 Mw 8.7 Nias-Simeulue, Sumatra earthquake", G-cubed, doi:10.1029/2011GC003553. Which FE code it used is **[UNVERIFIED]**.
- Hearn & Bürgmann (2005), "The effect of elastic layering on inversions of GPS data for coseismic slip and resulting stress changes: strike-slip earthquakes", BSSA, doi:10.1785/0120040158. Strike-slip layering effects; not PyLith.
- Tung & Masterlark (2016), "Coseismic slip distribution of the 2015 Mw7.8 Gorkha, Nepal, earthquake from joint inversion of GPS and InSAR data for slip within a 3-D heterogeneous domain", JGR Solid Earth, doi:10.1002/2015JB012497. InSAR with 3D FEM GFs; the FE code is not stated in the abstract.
- Langer, Ragon, Sladen & Tromp (2020), "Impact of topography on earthquake static slip estimates", Tectonophysics, doi:10.1016/j.tecto.2020.228566.
- **Not found:** a published PyLith GF study using a 3D velocity model for shallow strike-slip creep or the Hayward/SAF. I found none, so there is no direct template for this application.

**Local context:** `drafts/07/sfcvm_extract_dense.py` (SFCVM HDF5 reader and CRS handling); `07_slip_model.ipynb` (CSI mesh: 1510 nodes, 2610 triangles, 24.6 km, 3 km deep, vertical); `/Users/dlindsay/tools/external/csi/csi/Fault.py` (`setGFs`, line 2137).
