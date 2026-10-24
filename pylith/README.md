# pylith/ -- the PyLith side of the 3D slip model

Everything needed to remake the PyLith 5.0.2 runs behind notebooks `08c_3d_model_sensitivity`, `08a_3d_model` and
`08b_3d_model_tests`. The notebooks never run PyLith unless `TULEPOND_RUN_PYLITH=1`; they read the run products
from the results directory (`TulePond_filepaths.data_dir`, the Zenodo deposit) and stop with the command below
when one is missing.

```
pylith/
  README.md          this file
  cfg/               the parameter files the runs are made with (PyLith .cfg, Pyre format)
    pylithapp.cfg      common: mesh reader (gmsh .msh via MeshIOPetsc, tmerc CRS), solution (SolnDispLagrange),
                       scales, one elastic material, Dirichlet zero on the sides and bottom (free top)
    mat_homog.cfg      homogeneous medium (rho 2700, vs 3333, vp 5774: mu 30 GPa, nu 0.25); outputs -> modeling_uniform/
    mat_sfcvm*.cfg     USGS SFCVM v26.0 as a SimpleGridDB (aligned on the trace; _e50/_w50 boundary 50 m E/W;
                       _nocap no Poisson cap; _asis not aligned;
                       _dip90v per-depth control); outputs -> modeling_SFCVM/
    gf.cfg             GreensFns + FaultCohesiveImpulses (left-lateral impulses, slip basis order 1) + OutputSolnPoints
    fwd_kin.cfg        FaultCohesiveKin, one static step, a SimpleGridDB slip patch (the validation runs)
    fwd_kin_simpledb.cfg  FaultCohesiveKin with an inverted vertex slip as a SimpleDB (nearest): Fig-2 and traction runs
    order2.cfg         quadratic solution, linear materials, slip impulses order 1 (append after gf.cfg)
    order2_fwd.cfg     the same for forward runs (append after fwd_kin.cfg)
  pylith_env.sh      runs a command inside the PyLith 5.0.2 binary bundle with a clean environment
  _common.sh         shared by the wrappers: the work directory (data_dir) and the log folder of a medium
  run_gf.sh          one GreensFns run
  run_fwd.sh         one forward (FaultCohesiveKin) run
  build/             the inputs of the runs
    generate_gmsh_vertical.py   gmsh mesh of the vertical fault (PyLith gmsh_utils.GenerateMesh; presets default, fine, shallow, coarse)
    generate_gmsh_dip.py        the same for a dipping fault (top edge on the trace; --dip 80e / 70e / 80w)
    sfcvm_grid.py               SFCVM v26.0 resampled on a non-uniform grid around the fault, aligned on the trace (opera env)
    write_sfcvm_griddb.py       that grid as a spatialdata SimpleGridDB (spatialdata's own writer; PyLith env)
    make_impulse_db.py          the impulse region (km 54-65 of the trace, full 3 km depth) as a SimpleGridDB
    make_kin_slip_db.py         the validation slip patch (10 mm, km 57-61, top 1 km) as a SimpleGridDB
    make_vertex_slip_db.py      an inverted vertex slip (08a [14]) as a SimpleDB
```

## Where things are

Every run is made FROM the results directory `data_dir` (`TulePond_filepaths.paths_pylith["workdir"]`; override
with `PYLITH_WORKDIR`), and every path in the cfg files and on the command line is relative to it
(2026-10-06 layout, `TulePond_filepaths.paths_inputs / paths_uniform / paths_sfcvm`):

| Folder | Holds | Written by |
|---|---|---|
| `modeling_inputs/mesh/` | `mesh_vertical_tet_<preset>.msh`, `mesh_dip<d>_tet_default.msh`, their `_fault.npz` and `_report.txt`; `trace_tmerc.txt` | `build/generate_gmsh_*.py` |
| `modeling_inputs/points/` | the PyLith output points = the quadtree samples: `insar_points.txt` (2386, the paper's), `insar_points_swath7.txt` (2396), `insar_points_coh07.txt` (3148), each with an `.npz` carrying the data and noise model; `fig2_points_<label>.txt` (profile points + samples) | notebook 07a [6]; 08a [14] |
| `modeling_inputs/spatialdb/` | `impulses_km54-65_z3000.spatialdb`, `kin_slip_km57-61_z1000.spatialdb` | `build/make_impulse_db.py`, `build/make_kin_slip_db.py` |
| `modeling_uniform/pylith_output/` | every run in the homogeneous medium (`<run>-points.h5`, `-fault.h5`, `-elastic_info.h5`, `-parameters.json`, boundary outputs) | `run_gf.sh`, `run_fwd.sh` with `mat_homog.cfg` |
| `modeling_uniform/gf/` | `G_<run>.npz` of the homogeneous runs | `model_utils.build_G` (notebooks) |
| `modeling_uniform/spatialdb/` | slip databases of the homogeneous forward runs (`fig2_slip_<label>.spatialdb` + `.npz`) | 08a [14], `build/make_vertex_slip_db.py` |
| `modeling_uniform/validation/` | PyLith vs CSI and the 2D screw dislocation (`csi_vs_pylith_homog.npz`) | 08a [9] |
| `modeling_SFCVM/spatialdb/` | the SFCVM grids `sfcvm_v26-0_<tag>_grid.npz` and databases `mat_sfcvm_v26-0_<tag>.spatialdb`; the SFCVM forward runs' slip databases | `build/sfcvm_grid.py`, `build/write_sfcvm_griddb.py`; 08a [14] |
| `modeling_SFCVM/pylith_output/`, `gf/` | every run in an SFCVM medium and its G matrix | `run_gf.sh`, `run_fwd.sh` with `mat_sfcvm*.cfg`; `model_utils.build_G` |
| `<medium>/runs/` | the PyLith log of each run and `run_gf.log` (start / done times) | the wrappers |

The output folder of a run is set by its material cfg (`defaults.output_directory`), so a homogeneous run cannot
land among the SFCVM runs or the other way round.

## How to run

The PyLith 5.0.2 binary bundle for macOS arm64 (`pylith-5.0.2-macOS-12.0-arm64`, at `$PYLITH_BUNDLE`, default
`~/tools/external/pylith/`) provides PyLith, spatialdata and gmsh; the opera_disp_feb13 conda env provides the
notebooks. `pylith_env.sh` keeps the two apart (`env -i`, the bundle's `setup.sh` sourced in its own folder).

```bash
REPO=~/tools/internal/TulePond_DISP
DATA=$(cd $REPO && python3 -c 'import TulePond_filepaths as f; print(f.data_dir)')
# a Green's-function run: name, material cfg, mesh, impulse db, MPI ranks, points [extra cfg]
bash $REPO/pylith/run_gf.sh gf_default_sfcvm mat_sfcvm.cfg modeling_inputs/mesh/mesh_vertical_tet_default.msh \
     modeling_inputs/spatialdb/impulses_km54-65_z3000.spatialdb 8 modeling_inputs/points/insar_points.txt
# a forward run: name, material cfg, mesh, MPI ranks, forward cfg [extra cfg] [--pylith options]
bash $REPO/pylith/run_fwd.sh fwd_default_homog mat_homog.cfg modeling_inputs/mesh/mesh_vertical_tet_default.msh 2 fwd_kin.cfg
# a build step inside the PyLith env (gmsh, spatialdata), from the results directory
bash $REPO/pylith/pylith_env.sh $DATA python3 $REPO/pylith/build/generate_gmsh_vertical.py --write --preset default
```

The cfg stack of a run is always `pylithapp.cfg` + one `mat_*.cfg` + `gf.cfg` or a forward cfg [+ `order2*.cfg`],
then the command-line overrides the wrappers add: `--problem.defaults.name=<run>`, the mesh
(`--problem.mesh_initializer.phases.read_mesh.reader.filename`), for GreensFns the impulse database
(`--problem.interfaces.fault.db_auxiliary_field.filename`) and the output points
(`--problem.solution_observers.points.reader.filename`), and `--nodes=<ranks>`. Run times below are wall times on an
Apple M-series laptop (from `runs/run_gf.log`).

## The Green's-function runs (GreensFns)

All with `gf.cfg`, impulses `modeling_inputs/spatialdb/impulses_km54-65_z3000.spatialdb`, 8 ranks. Products in the
medium's `pylith_output/`; `G_<run>.npz` (fault-parallel G on 07a's strike) in its `gf/`.

| Run | Material cfg | Mesh (`modeling_inputs/mesh/`) | Points (`modeling_inputs/points/`) | Extra cfg | Read by | Time |
|---|---|---|---|---|---|---|
| `gf_default_homog` | `mat_homog.cfg` | `mesh_vertical_tet_default.msh` | `insar_points.txt` | | 08a, 08b, 08c (default, nuggetcd, diagcd, floor15, floor4, nosurf, nosurf_diagcd) | 32 min |
| `gf_default_sfcvm` | `mat_sfcvm.cfg` | `mesh_vertical_tet_default.msh` | `insar_points.txt` | | 08a, 08b, 08c (as above) | 33 min |
| `gf_default_sfcvm_e50` / `_w50` | `mat_sfcvm_e50.cfg` / `_w50.cfg` | `mesh_vertical_tet_default.msh` | `insar_points.txt` | | 08c (e50, w50 runs), 08a [15c] | 36 / 33 min |
| `gf_shallow_homog` / `_sfcvm` | `mat_homog.cfg` / `mat_sfcvm.cfg` | `mesh_vertical_tet_shallow.msh` | `insar_points.txt` | | 08c (shallow) | 125 / 131 min |
| `gf_default_dip80e_homog` / `_sfcvm` | `mat_homog.cfg` / `mat_sfcvm.cfg` (since 2026-10-25; was a per-dip warped grid) | `mesh_dip80e_tet_default.msh` | `insar_points.txt` | | 08c (dip80e runs) | 34 / 34 min |
| `gf_default_dip70e_homog` / `_sfcvm` | `mat_homog.cfg` / `mat_sfcvm.cfg` (since 2026-10-25) | `mesh_dip70e_tet_default.msh` | `insar_points.txt` | | 08c (dip70e runs) | 39 / 38 min |
| `gf_default_dip80w_homog` / `_sfcvm`, `gf_default_dip70w_homog` / `_sfcvm` (2026-10-25) | `mat_homog.cfg` / `mat_sfcvm.cfg` | `mesh_dip<d>_tet_default.msh` | `insar_points.txt` | | 08c (west dips) | see `runs/run_gf.log` |
| `gf_default_dip<d>_homog`, d = 50e, 60e, 60w, 50w (2026-10-24) | `mat_homog.cfg` | `mesh_dip<d>_tet_default.msh` | `insar_points.txt` | | 08c (the homogeneous dip scan, homogeneous only) | see `runs/run_gf.log` |
| `gf_default_homog_swath7` / `gf_default_sfcvm_swath7` | `mat_homog.cfg` / `mat_sfcvm.cfg` | `mesh_vertical_tet_default.msh` | `insar_points_swath7.txt` | | 08c (swath7 runs) | 34 / 33 min |
| `gf_default_homog_coh07` / `gf_default_sfcvm_coh07` | `mat_homog.cfg` / `mat_sfcvm.cfg` | `mesh_vertical_tet_default.msh` | `insar_points_coh07.txt` | | 08c (coh07 runs) | 32 / 33 min |
| `gf_default_homog_order2` / `gf_default_sfcvm_order2` | `mat_homog.cfg` / `mat_sfcvm_nocap.cfg` | `mesh_vertical_tet_default.msh` | `insar_points.txt` | `order2.cfg` | 08c (order2 runs) | 143 / 165 min |
| `gf_default_sfcvmasis` | `mat_sfcvm_asis.cfg` | `mesh_vertical_tet_default.msh` | `insar_points.txt` | | none since 2026-09-30 (the unaligned SFCVM was dropped) | 33 min |

The sample positions of every point set are asserted unchanged each time 07a rewrites them
(`model_utils.write_points_list`): a set whose positions moved needs its GreensFns runs remade.

## The forward runs (FaultCohesiveKin)

| Run | Material cfg | Mesh | Forward cfg (+ extra) | Overrides | Ranks | Read by |
|---|---|---|---|---|---|---|
| `fwd_default_homog` | `mat_homog.cfg` | `mesh_vertical_tet_default.msh` | `fwd_kin.cfg` | | 2 | 08a [9] (superposition (c), mesh convergence (d)), 08c (superposition) |
| `fwd_fine_homog` | `mat_homog.cfg` | `mesh_vertical_tet_fine.msh` | `fwd_kin.cfg` | | 2 | 08a [9] (mesh convergence (d)) |
| `fwd_default_o2_homog` / `fwd_default_o2_sfcvm` | `mat_homog.cfg` / `mat_sfcvm_nocap.cfg` | `mesh_vertical_tet_default.msh` | `fwd_kin.cfg` + `order2_fwd.cfg` | | 2 | 08c (order-2 superposition, within 5 %) |
| `fwd_fig2_nuggetcd` / `fwd_fig2_full` | `mat_sfcvm.cfg` | `mesh_vertical_tet_default.msh` | `fwd_kin_simpledb.cfg` | points `modeling_inputs/points/fig2_points_<label>.txt`; slip db `modeling_SFCVM/spatialdb/fig2_slip_<label>.spatialdb` | 4 | 08a [14] (Fig-2 profiles; imposed slip = inverted slip; G x slip) |
| `fwd_fig2_nuggetcd_homog` | `mat_homog.cfg` | `mesh_vertical_tet_default.msh` | `fwd_kin_simpledb.cfg` | as above, slip db in `modeling_uniform/spatialdb/` | 4 | 08a [14] |
| `fwd_traction_nuggetcd` / `fwd_traction_nuggetcd_homog` | `mat_sfcvm.cfg` / `mat_homog.cfg` | `mesh_vertical_tet_default.msh` | `fwd_kin_simpledb.cfg` | as `fwd_fig2_<label>`, plus `'--problem.interfaces.fault.observers.observer.data_fields=[slip,traction_change]'` | 4 | 08a [15b] (stress drop) |

Each forward run takes about 2 minutes. 08a writes the slip database of a Fig-2 / traction run from the slip it
has just inverted (`build/make_vertex_slip_db.py`) and asserts that the cached run carries exactly that slip.

## The inputs (build/)

| Product | Command (from the results directory) | Env |
|---|---|---|
| `modeling_inputs/mesh/mesh_vertical_tet_<preset>.msh` | `python3 $REPO/pylith/build/generate_gmsh_vertical.py --write --preset <default/fine/shallow/coarse>` | PyLith |
| `modeling_inputs/mesh/mesh_dip<d>_tet_default.msh` | `python3 $REPO/pylith/build/generate_gmsh_dip.py --write --preset default --dip <80e/70e/80w>` | PyLith |
| `modeling_SFCVM/spatialdb/sfcvm_v26-0_aligned_grid.npz` (and `_asis`, `_aligned_e50`/`_w50`, `_aligned_nocap`) | `python $REPO/pylith/build/sfcvm_grid.py [--no-align / --offset 50 / --offset -50 / --no-cap]` (`--dip` retired 2026-10-25: dipping faults use the paper grid) | opera_disp_feb13 |
| `modeling_SFCVM/spatialdb/mat_sfcvm_v26-0_<tag>.spatialdb` | `python3 $REPO/pylith/build/write_sfcvm_griddb.py modeling_SFCVM/spatialdb/sfcvm_v26-0_<tag>_grid.npz modeling_SFCVM/spatialdb/mat_sfcvm_v26-0_<tag>.spatialdb` | PyLith |
| `modeling_inputs/spatialdb/impulses_km54-65_z3000.spatialdb` | `python3 $REPO/pylith/build/make_impulse_db.py -4713 4380 3000 modeling_inputs/spatialdb/impulses_km54-65_z3000.spatialdb` | PyLith |
| `modeling_inputs/spatialdb/kin_slip_km57-61_z1000.spatialdb` | `python3 $REPO/pylith/build/make_kin_slip_db.py -1479 1943 1000 0.01 modeling_inputs/spatialdb/kin_slip_km57-61_z1000.spatialdb` | PyLith |
| `modeling_inputs/points/insar_points*.txt` | notebook `07a_uniform_model` [6] | opera_disp_feb13 |

(PyLith-env commands run through `bash $REPO/pylith/pylith_env.sh $DATA <command>`.)

## From the PyLith output to the inversion

`model_utils.build_G` reads a GreensFns run (`-fault.h5`: the fault vertices and the unit impulse on each;
`-points.h5`: the displacement at the output points), matches each impulse to its fault vertex and each output point
to its sample, projects the 3D displacement on 07a's fault-parallel look vector (the strike of the 06 field) and
derives the sign of right-lateral slip; `model_utils.load_G` rebuilds a cached `G_<run>.npz` whose strike differs
from the current field's. The inversion (`model_utils.invert_tents`) is CSI's, on the order-1 tents of the impulse
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

## History

Until 2026-10-06 the runs were made in `drafts/pylith/` (outputs in `drafts/pylith/output/`, cfg in `cfg/`, paths
relative to that folder), then copied to `modeling_SFCVM/`. Every product was copied (APFS clones, nothing moved or
deleted) into the layout above (manifest: `model_tidy_baseline_2026-10-06/copy_manifest.csv` in the results
directory); the cfg files differ from the ones the runs were made with only in their file paths (and the default
impulse database of `gf.cfg`, which every run overrode). A forward run remade with the new cfg and wrappers
(`smoke_2026-10-06_fwd_default_homog` in `modeling_uniform/pylith_output/`) reproduces `fwd_default_homog` bit for bit.
The retired driver, tests driver and queue scripts are in `drafts/retired_2026-10-06/pylith/`.
