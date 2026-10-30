# pylith/: the PyLith side of the 3D slip model

This folder holds everything needed to remake the PyLith 5.0.2 runs behind the slip models. No notebook runs PyLith:
`07a_uniform_model`, `08a_3d_model` and `scripts/sensitivity/*.py` read the archived G matrices (`<medium>/gf/G_<run>.npz`)
built from the run products in the results directory (`TulePond_filepaths.data_dir`, the Zenodo deposit). 08a's
"PyLith files" cell lists every file of the preferred run with its path. The preferred run and every setting of the model
are declared in `model_config.py` (`gf_run`, `gf_files`).

**The preferred model: `gf_shallow_dip70w_sfcvm_order2`** = `pylithapp.cfg` + `mat_sfcvm_nocap.cfg` + `gf.cfg` +
`order2.cfg` (order-2 displacement and Lagrange multiplier, quadrature order 2, materials order 1, slip impulses order 1),
mesh `mesh_dip70w_tet_shallow.msh` (fault dipping 70 deg west; 60 m cells within 1 km of the fault in the top 600 m),
USGS SFCVM v26.0 aligned on the trace, no Vs floor, no Poisson cap, 5 MPI ranks:

```bash
cd $DATA   # the results directory
bash $REPO/pylith/run_gf.sh gf_shallow_dip70w_sfcvm_order2 mat_sfcvm_nocap.cfg modeling_inputs/mesh/mesh_dip70w_tet_shallow.msh \
     modeling_inputs/spatialdb/impulses_km54-65_z3000.spatialdb 5 modeling_inputs/points/insar_points.txt order2.cfg
```
(log `modeling_SFCVM/runs/gf_shallow_dip70w_sfcvm_order2.log`.)

The same fault with a vertical dip, 70 deg east, 60 deg west and 50 deg west on the same mesh type, and the
forward-model variants on the default mesh, are sensitivity tests (`scripts/sensitivity/sensitivity_rows.py`). The
homogeneous comparison model is not a PyLith run (see below).

```
pylith/
  README.md          this file
  cfg/               the parameter files of the runs (PyLith .cfg, Pyre format)
    pylithapp.cfg      common: mesh reader (gmsh .msh via MeshIOPetsc, tmerc CRS), solution (SolnDispLagrange),
                       scales, one elastic material, Dirichlet zero on the sides and bottom (free top)
    mat_homog.cfg      homogeneous medium (rho 2700, vs 3333, vp 5774: mu 30 GPa, nu 0.25); outputs -> modeling_uniform/
    mat_sfcvm*.cfg     USGS SFCVM v26.0 as a SimpleGridDB (aligned on the trace; _e50/_w50 boundary 50 m E/W;
                       _nocap no Poisson cap; _asis not aligned); outputs -> modeling_SFCVM/
    gf.cfg             GreensFns + FaultCohesiveImpulses (left-lateral impulses, slip basis order 1) + OutputSolnPoints
    fwd_kin.cfg        FaultCohesiveKin, one static step, a SimpleGridDB slip patch (the validation runs)
    fwd_kin_simpledb.cfg  FaultCohesiveKin with an inverted vertex slip as a SimpleDB (nearest)
    order2.cfg         quadratic solution, linear materials, slip impulses order 1 (append after gf.cfg)
    order2_fwd.cfg     the same for forward runs (append after fwd_kin.cfg)
  pylith_env.sh      runs a command inside the PyLith 5.0.2 binary bundle with a clean environment
  _common.sh         shared by the wrappers: the work directory (data_dir) and the log folder of a medium
  run_gf.sh          one GreensFns run
  run_fwd.sh         one forward (FaultCohesiveKin) run
  build/             the inputs of the runs
    generate_gmsh_vertical.py   gmsh mesh of the vertical fault (PyLith gmsh_utils.GenerateMesh; presets default, fine, shallow, coarse)
    generate_gmsh_dip.py        the same for a dipping fault (top edge on the trace; --dip e.g. 70w, 70e, 60w, 50w)
    sfcvm_grid.py               SFCVM v26.0 resampled on a non-uniform grid around the fault, aligned on the trace (opera env)
    write_sfcvm_griddb.py       that grid as a spatialdata SimpleGridDB (spatialdata's own writer; PyLith env)
    make_impulse_db.py          the impulse region (km 54-65 of the trace, full 3 km depth) as a SimpleGridDB
    make_kin_slip_db.py         the validation slip patch (10 mm, km 57-61, top 1 km) as a SimpleGridDB
    make_vertex_slip_db.py      an inverted vertex slip as a SimpleDB
```

## Where things are

Every run is made from the results directory `data_dir` (`TulePond_filepaths.paths_pylith["workdir"]`; override
with `PYLITH_WORKDIR`), and every path in the cfg files and on the command line is relative to it
(`TulePond_filepaths.paths_inputs / paths_uniform / paths_sfcvm`):

| Folder | Holds | Written by |
|---|---|---|
| `modeling_inputs/mesh/` | `mesh_vertical_tet_<preset>.msh`, `mesh_dip<d>_tet_<preset>.msh`, their `_fault.npz` and `_report.txt`; `trace_tmerc.txt` | `build/generate_gmsh_*.py` |
| `modeling_inputs/points/` | the PyLith output points = the quadtree samples: `insar_points.txt` (the paper's sample set), `insar_points_coh07.txt` (coherence-0.7 sensitivity test), each with an `.npz` carrying the data and the noise model; `insar_points_swath7.txt` and `fig2_points_<label>.txt` (order-1 runs only) | notebook 07a |
| `modeling_inputs/spatialdb/` | `impulses_km54-65_z3000.spatialdb`, `impulses_test_km58.8-59.3_z1100.spatialdb` (strip runs), `kin_slip_km57-61_z1000.spatialdb` | `build/make_impulse_db.py`, `build/make_kin_slip_db.py` |
| `modeling_uniform/pylith_output/` | every run in the homogeneous medium (`<run>-points.h5`, `-fault.h5`, `-elastic_info.h5`, `-parameters.json`, boundary outputs) | `run_gf.sh`, `run_fwd.sh` with `mat_homog.cfg` |
| `modeling_uniform/gf/` | `G_<run>.npz` of the homogeneous runs; `G_analytic_<run>_nsub<n>.npz`, the analytic Green's functions of the homogeneous comparison model | `model_utils.build_G`, `model_utils.analytic_tent_G_cached` |
| `modeling_uniform/spatialdb/` | slip databases of the homogeneous forward runs (`fig2_slip_<label>.spatialdb` + `.npz`) | `build/make_vertex_slip_db.py` |
| `modeling_uniform/validation/` | PyLith vs CSI and the 2D screw dislocation (`csi_vs_pylith_homog.npz`) | notebook 08a |
| `modeling_SFCVM/spatialdb/` | the SFCVM grids `sfcvm_v26-0_<tag>_grid.npz` and databases `mat_sfcvm_v26-0_<tag>.spatialdb`; the SFCVM forward runs' slip databases | `build/sfcvm_grid.py`, `build/write_sfcvm_griddb.py`, `build/make_vertex_slip_db.py` |
| `modeling_SFCVM/pylith_output/`, `gf/` | every run in an SFCVM medium and its G matrix | `run_gf.sh`, `run_fwd.sh` with `mat_sfcvm*.cfg`; `model_utils.build_G` |
| `<medium>/runs/` | the PyLith log of each run and `run_gf.log` (start / done times) | the wrappers |

The output folder of a run is set by its material cfg (`defaults.output_directory`), so homogeneous and SFCVM runs
are written to separate folders.

## How to run

The PyLith 5.0.2 binary bundle for macOS arm64 (`pylith-5.0.2-macOS-12.0-arm64`, at `$PYLITH_BUNDLE`, default
`~/tools/external/pylith/`) provides PyLith, spatialdata and gmsh; the opera_disp_feb13 conda environment provides the
notebooks. `pylith_env.sh` keeps the two apart (`env -i`, the bundle's `setup.sh` sourced in its own folder).

```bash
REPO=~/tools/internal/TulePond_DISP
DATA=$(cd $REPO && python3 -c 'import TulePond_filepaths as f; print(f.data_dir)')
# a Green's-function run: name, material cfg, mesh, impulse db, MPI ranks, points [extra cfg]
bash $REPO/pylith/run_gf.sh gf_default_sfcvm_order2 mat_sfcvm_nocap.cfg modeling_inputs/mesh/mesh_vertical_tet_default.msh \
     modeling_inputs/spatialdb/impulses_km54-65_z3000.spatialdb 5 modeling_inputs/points/insar_points.txt order2.cfg
# a forward run: name, material cfg, mesh, MPI ranks, forward cfg [extra cfg] [--pylith options]
bash $REPO/pylith/run_fwd.sh fwd_default_homog mat_homog.cfg modeling_inputs/mesh/mesh_vertical_tet_default.msh 2 fwd_kin.cfg
# a build step inside the PyLith env (gmsh, spatialdata), from the results directory
bash $REPO/pylith/pylith_env.sh $DATA python3 $REPO/pylith/build/generate_gmsh_dip.py --write --preset shallow --dip 70w
```

The cfg stack of a run is always `pylithapp.cfg` + one `mat_*.cfg` + `gf.cfg` or a forward cfg [+ `order2*.cfg`],
followed by the command-line overrides the wrappers add: `--problem.defaults.name=<run>`, the mesh
(`--problem.mesh_initializer.phases.read_mesh.reader.filename`), for GreensFns the impulse database
(`--problem.interfaces.fault.db_auxiliary_field.filename`) and the output points
(`--problem.solution_observers.points.reader.filename`), and `--nodes=<ranks>`. Run times below are wall times on an
Apple M-series laptop (from `runs/run_gf.log`).

## The Green's-function runs (GreensFns)

### Order-2 runs: the preferred model and its sensitivity tests

All with `gf.cfg` + `order2.cfg`, impulses `impulses_km54-65_z3000.spatialdb`, 5 ranks, SFCVM without the Poisson cap
and with no Vs floor. Start and done times in `<medium>/runs/run_gf.log`.

| Run | Material cfg | Mesh | Points | Role | Time |
|---|---|---|---|---|---|
| `gf_shallow_dip70w_sfcvm_order2` | `mat_sfcvm_nocap.cfg` | `mesh_dip70w_tet_shallow.msh` | `insar_points.txt` | **Preferred model** (07a, 08a; baseline of the inversion-choice sensitivity tests; depth scan) | 35 h |
| `gf_shallow_sfcvm_order2` | `mat_sfcvm_nocap.cfg` | `mesh_vertical_tet_shallow.msh` | `insar_points.txt` | Sensitivity test: vertical fault, fine mesh | 17 h |
| `gf_shallow_dip70e_sfcvm_order2` | `mat_sfcvm_nocap.cfg` | `mesh_dip70e_tet_shallow.msh` | `insar_points.txt` | Sensitivity test: 70 deg E fault, fine mesh | 32 h |
| `gf_shallow_dip60w_sfcvm_order2` / `_dip50w_` | `mat_sfcvm_nocap.cfg` | `mesh_dip60w_tet_shallow.msh` / `mesh_dip50w_tet_shallow.msh` | `insar_points.txt` | Sensitivity tests: 60 deg W and 50 deg W faults, fine mesh | see `runs/run_gf.log` |
| `gf_default_sfcvm_order2` | `mat_sfcvm_nocap.cfg` | `mesh_vertical_tet_default.msh` | `insar_points.txt` | Sensitivity tests: baseline of the forward-model tests (vertical fault, default mesh) and of the default-mesh dip sweep | 4.2 h |
| `gf_default_sfcvm_e50_order2` / `_w50_order2` | `mat_sfcvm_e50_nocap.cfg` / `_w50_nocap.cfg` | `mesh_vertical_tet_default.msh` | `insar_points.txt` | Sensitivity tests: SFCVM alignment +/- 50 m | 4.5 / 4.3 h |
| `gf_default_sfcvmasis_order2` | `mat_sfcvm_asis_nocap.cfg` | `mesh_vertical_tet_default.msh` | `insar_points.txt` | Sensitivity test: SFCVM as is (not aligned) | 4.8 h |
| `gf_default_sfcvm_coh07_order2` | `mat_sfcvm_nocap.cfg` | `mesh_vertical_tet_default.msh` | `insar_points_coh07.txt` | Sensitivity test: data at coherence 0.7 | 4.5 h |
| `gf_default_dip{80e,80w,70e,70w}_sfcvm_order2` | `mat_sfcvm_nocap.cfg` | `mesh_dip<d>_tet_default.msh` | `insar_points.txt` | Sensitivity tests: dip sweep on the default mesh (`scripts/sensitivity/dip_rows.py`) | 4.5-5 h each |
| `gf_default_homog_order2` | `mat_homog.cfg` | `mesh_vertical_tet_default.msh` | `insar_points.txt` | Green's-function check against the analytic solution (`scripts/sensitivity/gf_check.py`, A and B) | 143 min |
| `gf_shallow_homog_order2_strip` / `gf_fine_homog_order2_strip` | `mat_homog.cfg` | `mesh_vertical_tet_shallow.msh` / `_fine.msh` | `insar_points.txt` | Green's-function check B (mesh refinement); impulses only in the strip `impulses_test_km58.8-59.3_z1100.spatialdb` | 42 / 46 min |

The homogeneous comparison model is not a PyLith run: it uses the analytic half-space solution of CSI (triangular
dislocations, Meade 2007) on the tents of the preferred run (`model_utils.analytic_tent_G`, 07a; cache
`modeling_uniform/gf/G_analytic_<gf_run>_nsub<n>.npz`).

The order-1 SFCVM runs made with a 150 m/s Vs floor are archived in `backup_vsfloor150_2026-10-25/` in the results
directory and are not used. The current `mat_sfcvm.cfg` / `mat_sfcvm_asis.cfg` (order 1, Poisson cap 0.40) read the
grids without the floor; no run of the paper uses them.

**Meshes:** the report header of each mesh (`<mesh>_report.txt`) records the script that generated it. The vertical
meshes `mesh_vertical_tet_{default,shallow}.msh` were generated with an earlier copy of
`build/generate_gmsh_vertical.py` with the same presets (`PRESETS`); bit identity of the two scripts' output is not
recorded.

### Order-1 runs (not used by the paper)

All with `gf.cfg`, impulses `modeling_inputs/spatialdb/impulses_km54-65_z3000.spatialdb`, 8 ranks. Products in the
medium's `pylith_output/`; `G_<run>.npz` (fault-parallel G) in its `gf/`.

| Run | Material cfg | Mesh (`modeling_inputs/mesh/`) | Points (`modeling_inputs/points/`) | Extra cfg | Use | Time |
|---|---|---|---|---|---|---|
| `gf_default_homog` | `mat_homog.cfg` | `mesh_vertical_tet_default.msh` | `insar_points.txt` | | Green's-function check A (element order, `scripts/sensitivity/gf_check.py`) | 32 min |
| `gf_default_sfcvm` | `mat_sfcvm.cfg` | `mesh_vertical_tet_default.msh` | `insar_points.txt` | | none | 33 min |
| `gf_default_sfcvm_e50` / `_w50` | `mat_sfcvm_e50.cfg` / `_w50.cfg` | `mesh_vertical_tet_default.msh` | `insar_points.txt` | | none | 36 / 33 min |
| `gf_shallow_homog` / `_sfcvm` | `mat_homog.cfg` / `mat_sfcvm.cfg` | `mesh_vertical_tet_shallow.msh` | `insar_points.txt` | | none | 125 / 131 min |
| `gf_default_dip80e_homog` / `_sfcvm` | `mat_homog.cfg` / `mat_sfcvm.cfg` | `mesh_dip80e_tet_default.msh` | `insar_points.txt` | | none | 34 / 34 min |
| `gf_default_dip70e_homog` / `_sfcvm` | `mat_homog.cfg` / `mat_sfcvm.cfg` | `mesh_dip70e_tet_default.msh` | `insar_points.txt` | | none | 39 / 38 min |
| `gf_default_dip80w_homog` / `_sfcvm`, `gf_default_dip70w_homog` / `_sfcvm` | `mat_homog.cfg` / `mat_sfcvm.cfg` | `mesh_dip<d>_tet_default.msh` | `insar_points.txt` | | none | see `runs/run_gf.log` |
| `gf_default_dip<d>_homog`, d = 50e, 60e, 60w, 50w | `mat_homog.cfg` | `mesh_dip<d>_tet_default.msh` | `insar_points.txt` | | none | see `runs/run_gf.log` |
| `gf_default_homog_swath7` / `gf_default_sfcvm_swath7` | `mat_homog.cfg` / `mat_sfcvm.cfg` | `mesh_vertical_tet_default.msh` | `insar_points_swath7.txt` | | none | 34 / 33 min |
| `gf_default_homog_coh07` / `gf_default_sfcvm_coh07` | `mat_homog.cfg` / `mat_sfcvm.cfg` | `mesh_vertical_tet_default.msh` | `insar_points_coh07.txt` | | none | 32 / 33 min |
| `gf_default_sfcvmasis` | `mat_sfcvm_asis.cfg` | `mesh_vertical_tet_default.msh` | `insar_points.txt` | | none | 33 min |

The sample positions of every point set are asserted unchanged each time 07a rewrites them
(`model_utils.write_points_list`): a set whose positions change requires its GreensFns runs to be remade.

## The forward runs (FaultCohesiveKin; order 1, not used by the paper)

| Run | Material cfg | Mesh | Forward cfg (+ extra) | Overrides | Ranks |
|---|---|---|---|---|---|
| `fwd_default_homog` | `mat_homog.cfg` | `mesh_vertical_tet_default.msh` | `fwd_kin.cfg` | | 2 |
| `fwd_fine_homog` | `mat_homog.cfg` | `mesh_vertical_tet_fine.msh` | `fwd_kin.cfg` | | 2 |
| `fwd_default_o2_homog` / `fwd_default_o2_sfcvm` | `mat_homog.cfg` / `mat_sfcvm_nocap.cfg` | `mesh_vertical_tet_default.msh` | `fwd_kin.cfg` + `order2_fwd.cfg` | | 2 |
| `fwd_fig2_nuggetcd` / `fwd_fig2_full` | `mat_sfcvm.cfg` | `mesh_vertical_tet_default.msh` | `fwd_kin_simpledb.cfg` | points `modeling_inputs/points/fig2_points_<label>.txt`; slip db `modeling_SFCVM/spatialdb/fig2_slip_<label>.spatialdb` | 4 |
| `fwd_fig2_nuggetcd_homog` | `mat_homog.cfg` | `mesh_vertical_tet_default.msh` | `fwd_kin_simpledb.cfg` | as above, slip db in `modeling_uniform/spatialdb/` | 4 |
| `fwd_traction_nuggetcd` / `fwd_traction_nuggetcd_homog` | `mat_sfcvm.cfg` / `mat_homog.cfg` | `mesh_vertical_tet_default.msh` | `fwd_kin_simpledb.cfg` | as `fwd_fig2_<label>`, plus `'--problem.interfaces.fault.observers.observer.data_fields=[slip,traction_change]'` | 4 |

Each forward run takes about 2 minutes. The slip database of a forward run of an inverted model is written with
`build/make_vertex_slip_db.py` from the inverted vertex slip.

## The inputs (build/)

| Product | Command (from the results directory) | Env |
|---|---|---|
| `modeling_inputs/mesh/mesh_vertical_tet_<preset>.msh` | `python3 $REPO/pylith/build/generate_gmsh_vertical.py --write --preset <default/fine/shallow/coarse>` | PyLith |
| `modeling_inputs/mesh/mesh_dip<d>_tet_<preset>.msh` | `python3 $REPO/pylith/build/generate_gmsh_dip.py --write --preset <default/shallow> --dip <d>` (d = 80e, 70e, 80w, 70w, 60w, 50w, ...) | PyLith |
| `modeling_SFCVM/spatialdb/sfcvm_v26-0_aligned_grid.npz` (and `_asis`, `_aligned_e50`/`_w50`, `_aligned_nocap`, ...) | `python $REPO/pylith/build/sfcvm_grid.py [--no-align / --offset 50 / --offset -50 / --no-cap]` (the dipping faults use the grid aligned at the surface trace) | opera_disp_feb13 |
| `modeling_SFCVM/spatialdb/mat_sfcvm_v26-0_<tag>.spatialdb` | `python3 $REPO/pylith/build/write_sfcvm_griddb.py modeling_SFCVM/spatialdb/sfcvm_v26-0_<tag>_grid.npz modeling_SFCVM/spatialdb/mat_sfcvm_v26-0_<tag>.spatialdb` | PyLith |
| `modeling_inputs/spatialdb/impulses_km54-65_z3000.spatialdb` | `python3 $REPO/pylith/build/make_impulse_db.py -4713 4380 3000 modeling_inputs/spatialdb/impulses_km54-65_z3000.spatialdb` | PyLith |
| `modeling_inputs/spatialdb/kin_slip_km57-61_z1000.spatialdb` | `python3 $REPO/pylith/build/make_kin_slip_db.py -1479 1943 1000 0.01 modeling_inputs/spatialdb/kin_slip_km57-61_z1000.spatialdb` | PyLith |
| `modeling_inputs/points/insar_points*.txt` | notebook `07a_uniform_model` | opera_disp_feb13 |

(PyLith-environment commands run through `bash $REPO/pylith/pylith_env.sh $DATA <command>`.)

## From the PyLith output to the inversion

`model_utils.build_G` reads a GreensFns run (`-fault.h5`: the fault vertices and the unit impulse on each;
`-points.h5`: the displacement at the output points), matches each impulse to its fault vertex and each output point
to its sample, projects the 3D displacement on the fault-parallel look vector of 07a (the strike of the 06 field) and
derives the sign of right-lateral slip; `model_utils.load_G` rebuilds a cached `G_<run>.npz` whose strike differs
from the current field's. The inversion (`model_utils.invert_tents`) uses CSI on the order-1 tents of the impulse
vertices. This follows the reading pattern of `examples/subduction-3d/invert_slip.py` in the PyLith source tree.

## The PyLith documentation each piece follows

- Example `crustal-strikeslip-3d` (the pattern of `pylithapp.cfg` and of `fwd_kin*.cfg`, `step01_slip.cfg`):
  https://pylith.readthedocs.io/en/latest/user/examples/crustal-strikeslip-3d/index.html,
  https://pylith.readthedocs.io/en/latest/user/examples/crustal-strikeslip-3d/common-information.html,
  https://pylith.readthedocs.io/en/latest/user/examples/crustal-strikeslip-3d/step01-slip.html
- Example `subduction-3d` (`pylithapp.cfg`; `gf.cfg` follows `step07a_leftlateral.cfg`; `invert_slip.py`; the
  quadratic-order settings of `order2.cfg` follow its `step08a`):
  https://pylith.readthedocs.io/en/latest/user/examples/subduction-3d/index.html,
  https://pylith.readthedocs.io/en/latest/user/examples/subduction-3d/step07-greensfns.html,
  https://pylith.readthedocs.io/en/latest/user/examples/subduction-3d/step08-gravity.html
- Example `strikeslip-2d` (`gf.cfg` follows `step05_greensfns.cfg`; the inversion of step 6; the quadrature order of
  `order2.cfg` follows `step04_varslip.cfg`):
  https://pylith.readthedocs.io/en/latest/user/examples/strikeslip-2d/step04-varslip.html,
  https://pylith.readthedocs.io/en/latest/user/examples/strikeslip-2d/step05-greensfns.html,
  https://pylith.readthedocs.io/en/latest/user/examples/strikeslip-2d/step06-inversion-leastsquares.html
- Components: https://pylith.readthedocs.io/en/latest/user/components/problems/GreensFns.html,
  https://pylith.readthedocs.io/en/latest/user/components/faults/FaultCohesiveImpulses.html,
  https://pylith.readthedocs.io/en/latest/user/components/faults/FaultCohesiveKin.html,
  https://pylith.readthedocs.io/en/latest/user/components/faults/KinSrcStep.html,
  https://pylith.readthedocs.io/en/latest/user/components/meshio/OutputSolnPoints.html,
  https://pylith.readthedocs.io/en/latest/user/components/meshio/PointsList.html,
  https://pylith.readthedocs.io/en/latest/user/components/meshio/MeshIOPetsc.html,
  https://pylith.readthedocs.io/en/latest/user/components/materials/IsotropicLinearElasticity.html,
  https://pylith.readthedocs.io/en/latest/user/components/problems/SolnDispLagrange.html,
  https://pylith.readthedocs.io/en/latest/user/components/bc/ZeroDB.html

## Provenance of the earlier runs

The order-1 runs and the early order-2 runs were made before the cfg files were placed in this folder; their cfg
files differ from the ones here only in file paths (and in the default impulse database of `gf.cfg`, which every run
sets on the command line). A forward run remade with the current cfg files and wrappers
(`smoke_2026-10-06_fwd_default_homog` in `modeling_uniform/pylith_output/`) reproduces `fwd_default_homog` bit for bit.
