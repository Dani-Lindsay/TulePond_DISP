# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
# ---

# %% [markdown]
# # 08a The 3D slip model: PyLith Green's functions in the USGS SFCVM
#
# ```
# Created on Wed Sep 30 2026
# PREPUB: promoted from drafts/retired_2026-10-06/pylith/pylith_sfcvm_gf.py, the 2026-09-29/30 draft; 2026-10-06: modelling functions in
# PREPUB: model_utils.py, 07a's data read from its product (no exec of 07a), PyLith files in the 2026-10-06 layout.
# @author: daniellelindsay
# Run in opera_disp_feb13, after 07a and 08c. About 15 min (about 60 tent inversions of ~10 s each);
# no PyLith run: the Green's functions, mesh, materials and forward runs are read from the results directory.
# ```
#
# **What this notebook does.** The slip model of the paper: the five-event mean fault-parallel
# field of notebook 06 inverted on **3D finite-element Green's functions (PyLith 5.0.2)** for the
# vertical fault under the mapped creeping trace, in the **USGS SFCVM v26.0** elastic structure
# aligned on the trace (the sediment-over-bedrock contrast that the homogeneous half-space of 07a
# cannot represent). The data, the quadtree samples and the noise model are 07a's (read from its
# `model_data_07a.npz`), and the inversion is 07a's CSI machinery with the PyLith G (one definition of every
# function, in `model_utils.py`), so the two models are compared like for like. The surface constraint is NOT
# 07a's: every measured fault crossing inside the mesh is one datum, tied to the slip at its own position on the
# fault's top edge (`model_utils.crossing_constraint`, Dani 2026-10-02; cell [10]). The 3D model supersedes
# the uniform one (07a) in the paper; 07a stays as the baseline it is measured against. After the
# models: the PyLith stress drop of the selected model [15b], its Mw under each rigidity choice
# [15c], where the 3D medium improves the fit [15d], and the numbers the text quotes about the X-X' deflections
# of the homogeneous model (and their range over the sensitivity runs), the SFSVM boundary offset from the trace and
# the selected model's potency above 300 and 500 m [15e].
#
# The **SELECTED model** (Dani, 2026-09-30): run tag `default_nuggetcd`, aligned SFCVM medium,
# vertical fault, `default` mesh (100 m fault cells in the Tule Pond patch), order-1 elements,
# nugget-only diagonal data covariance, NO surface constraint (`TulePond_filepaths.model_use_surface`, Dani 2026-10-16;
# ON until then), sigma at the corner of its own L-curve on the refined 3D-model grid (locked,
# `TulePond_filepaths.model_sigma_locked`, asserted in [11]). Its **companion** is the same inversion with 07a's full correlated covariance (tag
# `default`). Both media (homogeneous 30 GPa and aligned SFCVM) are inverted in each, so every table carries four
# models plus 07a's CSI baseline. The sensitivity runs (mesh, Cd floors, WITH the surface constraint, wider swath, dip,
# order 2, alignment +/- 50 m, coherence 0.7) and the bootstrap are made by `08c_3d_model_sensitivity` and drawn by
# `08b_3d_model_tests`.
#
# | Step | Tool (validated library) | Glue written here (`model_utils`) |
# |---|---|---|
# | mesh | gmsh via PyLith `gmsh_utils.GenerateMesh` (`pylith/build/generate_gmsh_vertical.py`) | size fields |
# | materials | SFCVM read with h5py (`pylith/build/sfcvm_grid.py`); spatialdata `SimpleGridAscii` writer | alignment shift, vs floor, Poisson cap |
# | Green's functions | PyLith `GreensFns` + `FaultCohesiveImpulses` + `OutputSolnPoints` (`pylith/cfg/gf.cfg`) | impulse-region db |
# | G assembly | reading pattern of PyLith `examples/subduction-3d/invert_slip.py` (`build_G`, `load_G`) | vertex/point matching, projection, sign |
# | inversion | CSI `TriangularTents`, `setGFs`, `buildCmXY`, `transformation`, `multisourcesolve` NNLS, `surfaceslip` (as 07a) | surface constraint on vertices (`crossing_constraint`) |
# | validation | PyLith `FaultCohesiveKin` forward runs; CSI Meade dislocations; 2D screw dislocation | comparisons |
#
# **The PyLith side is not re-run here.** The Green's-function runs (33 min each, 8 MPI ranks,
# PyLith 5.0.2 bundle) were made once with the parameter files in `pylith/cfg/` and the wrappers and build scripts
# of `pylith/` (every run, its cfg stack, inputs and outputs: `pylith/README.md`). Their products are read from the
# results directory: `modeling_inputs/` (mesh, sample points, impulse database), `modeling_uniform/` (the
# homogeneous runs) and `modeling_SFCVM/` (the SFCVM grids and runs). A missing product stops the notebook with the
# command that makes it; `TULEPOND_RUN_PYLITH=1` lets the notebook shell out to PyLith instead.
#
# **Inputs** (made by): 07a `model_data_07a.npz` (samples, data, noise model, crossings, strike, prior, the trace and
# uniform mesh) and `fig2_model_row.npz` (07a's baseline on the same samples); 06 `fp_maps.npz` (fingerprint) and
# `fig2_inputs.npz` (the profile lines); 04 `section_lengths.csv`; `modeling_inputs/mesh/`, `points/insar_points.*`
# (07a), `spatialdb/` (pylith/build); `modeling_uniform/` and `modeling_SFCVM/` `pylith_output/` (pylith/run_gf.sh,
# run_fwd.sh), `gf/` (G matrices, model_utils.build_G), `modeling_SFCVM/spatialdb/` grids (pylith/build/sfcvm_grid.py);
# from `outputs/08c_3d_model_sensitivity/`: `lcurve_<medium><tag>.csv` and `results_comparison<tag>.csv` (compared
# in [11] and [12]), `results_comparison_{e50,w50}<tag>.csv`, `lcurve_sfcvm_{e50,w50}_*` and
# `inversion_products_{e50,w50}<tag>.npz` (the +/- 50 m alignment runs of the Mw range, [14], [15c]) and
# `uncertainty_<run>_<medium>.csv` (the bootstrap, [14]).
# **Outputs** (`outputs/08a_3d_model/`): `lcurve_<medium><tag>.csv`, `results_comparison<tag>.csv`,
# `inversion_products<tag>.npz`, the Fig-2 products `fig2_model_row_pylith_{nuggetcd,full,nuggetcd_homog}.npz`
# (read by `07c_fig2`), `mw_rigidity_options.csv` and `mw_range_models.csv` ([15c]), `xx_deflections_homog_runs.csv` ([15e]), the stats sidecar
# `08a_3d_model_stats.txt`; `modeling_uniform/validation/csi_vs_pylith_homog.npz` ([8], [9]).
# **Figures**: `figS_pylith_materials_sfcvm_v26` and `qc08a_materials_boundary_offset` ([3]),
# `qc08a_mesh_fault` ([2]), `figS_pylith_validation` ([8]), `qc08a_validation_mesh` ([9]),
# `figS_pylith_lcurves` ([11]), `figS_pylith_fit_maps_<tag>`, `figS_pylith_slip`, `figS_pylith_profile_XX` ([13]).
# **Registry keys** `08a_*` (selected), `08a_hom_*`, `08a_full_*`, `08a_full_hom_*`.
#
# **What to check.** Every CHECK prints PASS: PyLith's cell rigidity within 5 % of the aligned grid
# (median) [6]; no sample more than 50 m from the mesh fault on the other side of 07a's trace [7];
# PyLith against CSI beyond 300 m of the trace under 2 % of the signal and on agreeing sides under
# 10 % rms [8]; superposition to 1e-4 mm and fine minus default mesh under 0.2 mm [9]; every model
# at the locked sigma 12 and equal to 08c's run of the same settings [11], [12]; each forward run's imposed slip
# equal to the inverted slip and its displacement equal to G x slip to 0.05 mm [14].

# %%
## [1]
# ------------------------
# Paths & constants
# ------------------------
import copy
import json
import os
import time
import warnings

import h5py
import numpy as np
import pandas as pd
import pyproj
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.tri as mtri
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", category=DeprecationWarning)   # library noise only; numerical warnings stay visible
warnings.filterwarnings("ignore", category=FutureWarning)
from TulePond_filepaths import (common_paths, repo_dir, paths_out, paths_inputs, paths_uniform, paths_sfcvm, paths_pylith,
                                medium_paths, model_crs, model_sigma_locked, model_sigma_grid, model_use_surface, model_near_trace_cut_km, dist_min, dist_max,
                                lake_elizabeth_trace_lonlat)   # dist_min/max: 06's near-fault band, [15e]
import insar_utils as utils
from insar_utils import mpl_paper_style
import model_utils                                  # every modelling function, once (with its source)

# CSI: Classic Slip Inversion (Jolivet et al.; external/csi at 1277bb4)
from csi import insar

notebook = "08a_3d_model"
out_dir = paths_out[notebook]
fig_dir = common_paths["fig_dir"]
dir_07a = paths_out["07a_uniform_model"]
dir_08c = paths_out["08c_3d_model_sensitivity"]
field_dir = paths_out["06_event_field_geometry"]
events_dir = paths_out["04_event_detection"]
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)
WORK = paths_pylith["workdir"]                      # PyLith runs are made from here (pylith/README.md)
for folder in (paths_inputs["mesh"], paths_inputs["points"], paths_inputs["spatialdb"], paths_uniform["pylith_output"],
               paths_sfcvm["pylith_output"], paths_sfcvm["spatialdb"]):
    assert os.path.isdir(folder), f"missing {folder}: the PyLith products (pylith/README.md)"
run_pylith = os.environ.get("TULEPOND_RUN_PYLITH", "0") == "1"   # 1: make a missing PyLith product here (33 min per GF run)
CRS = model_crs                                     # mesh CRS = the CSI frame of 07a
pylith_dir = paths_pylith["root"]
build_dir = paths_pylith["build"]

# the FIXED configuration of the selected model and its companion (Dani 2026-09-30); the
# alternatives (mesh, dip, order, swath, coherence, Cd floors, no surface constraint) are 08c's runs
mesh_preset = "default"                             # 100 m fault cells in km 56-63 / top 1.5 km, 250 m elsewhere on the fault
elem_order = "1"                                    # linear elements; slip impulses are tents on the fault vertices
points_set = "default"                              # 07a's 2386 quadtree samples (swath 4 km)
use_surface = model_use_surface                     # measured-crossing surface constraint (TulePond_filepaths; OFF since 2026-10-16, Dani)
sigma_locked = model_sigma_locked                   # the model sigma, locked for every model (Dani 2026-10-03; asserted in [11])
media = {"homog": "mat_homog.cfg", "sfcvm": "mat_sfcvm.cfg"}     # 30 GPa / SFCVM v26.0 aligned on the trace (pylith/cfg)
run_names = {m: f"gf_{mesh_preset}_{m}" for m in media}         # PyLith GreensFns runs (<medium>/pylith_output/<run>-*.h5)
CD_TAG = {"nugget": "_nuggetcd", "full": ""}        # product suffix of each data-covariance treatment
RUNS = [("sfcvm", "nugget"), ("sfcvm", "full"), ("homog", "nugget"), ("homog", "full")]   # (medium, Cd); first = SELECTED
SELECTED = ("sfcvm", "nugget")
COMPANION = ("sfcvm", "full")
KEY = {("sfcvm", "nugget"): "08a", ("homog", "nugget"): "08a_hom", ("sfcvm", "full"): "08a_full", ("homog", "full"): "08a_full_hom"}
FIG2_PRODUCTS = {"nuggetcd": ("sfcvm", "nugget"), "full": ("sfcvm", "full"), "nuggetcd_homog": ("homog", "nugget")}  # label -> model
n_procs = 8                                         # MPI ranks of the Green's-function runs
impulse_y_m = (-4713.0, 4380.0)                     # impulses on km 54-65 from Point Pinole (y of km 65, km 54 on the trace)
impulse_depth_m = 3000.0                            # whole fault depth (bottom edge buried -> no impulse there)
mu_ref = 3.0e10                                     # 30 GPa reference rigidity (07a)
test_km = (57.0, 61.0)                              # validation (a) slip patch: along strike (km from Point Pinole)
test_depth_m = 1000.0                               # ... from the surface to this depth
test_slip_mm = 10.0                                 # ... with this slip
kin_y_m = (-1479.0, 1943.0)                         # y window of km 57-61 (validation (a)/(c)/(d) forward runs)
sfcvm_label = "v26-0 aligned"                       # sfcvm_version of the Fig-2 products (07c reads "aligned" -> 3D colours)
timings = {}
OK = dict(blue="#0072B2", orange="#E69F00", vermillion="#D55E00", purple="#CC79A7", black="#000000", grey="#999999")


def run_tag(cd):
    """The product tag of one data-covariance treatment: default (full Cd) or default_nuggetcd."""
    return "default" + CD_TAG[cd]


def cached(path, recipe):
    """A product of the PyLith side; with TULEPOND_RUN_PYLITH=1 a missing one is made by `recipe`, run in the results
    directory (model_utils.ensure_product)."""
    return model_utils.ensure_product(path, recipe, WORK, run_pylith, timings)


in_pylith = f"bash {pylith_dir}/pylith_env.sh {WORK}"   # "<in_pylith> <command>" runs the command in the PyLith env, from the results directory
to_mesh = pyproj.Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
REG = utils.Registry(notebook, common_paths, repo_dir)
print(f"selected model: {run_tag(SELECTED[1])} / {SELECTED[0]}; companion {run_tag(COMPANION[1])} / {COMPANION[0]}; "
      f"PyLith products from {WORK} (TULEPOND_RUN_PYLITH={int(run_pylith)})")

# %% [markdown]
# ## [2] Mesh
#
# `pylith/build/generate_gmsh_vertical.py` (PyLith `gmsh_utils.GenerateMesh`): the mapped trace
# (every 3rd 100 m point, km 52-67) as a spline, extruded 3 km straight down, fragmented into a
# 100 x 100 x 40 km box. Cell size from three linear-growth fields combined by a gmsh Min field.
# Preset `default`: 100 m on the fault in km 56-63 / top 1.5 km, 250 m elsewhere on the fault,
# 250 m on the surface over the data, bias 1.15. Read from `modeling_inputs/mesh/`.

# %%
## [2]
mesh_file = cached(os.path.join(paths_inputs["mesh"], f"mesh_vertical_tet_{mesh_preset}.msh"),
                   f"{in_pylith} python3 {build_dir}/generate_gmsh_vertical.py --write --preset {mesh_preset}")
mesh_rel = os.path.relpath(mesh_file, WORK)         # the mesh as the PyLith runs name it (relative to the results directory)
mesh_report = mesh_file.replace(".msh", "_report.txt")
mesh_fault_npz = mesh_file.replace(".msh", "_fault.npz")
assert os.path.exists(mesh_report) and os.path.exists(mesh_fault_npz), f"mesh report / fault npz missing next to {mesh_file}"
print(open(mesh_report).read())
mesh_npz = np.load(mesh_fault_npz)
fault_tri = mesh_npz["xyz"]                  # (n_tri, 3 vertices, xyz) in m
edge_length_m = mesh_npz["edge_len"]         # mean edge length per triangle (m)

plt.rcParams.update({"font.size": 12, "savefig.facecolor": "white", "figure.facecolor": "white"})
fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
ax = axes[0]
ax.add_collection(PolyCollection(fault_tri[:, :, [1, 2]] / 1e3, facecolors="none", edgecolors="0.3", linewidths=0.3))
ax.axvspan(-3.1, 2.8, ymin=0.5, color=OK["orange"], alpha=0.15, label="fine patch (km 56-63, top 1.5 km)")
ax.axvline(impulse_y_m[0] / 1e3, color=OK["vermillion"], ls="--")
ax.axvline(impulse_y_m[1] / 1e3, color=OK["vermillion"], ls="--", label="impulse region (km 54-65)")
ax.set_xlim(-6.6, 6.2)
ax.set_ylim(-3.05, 0.05)
ax.set_xlabel("y north (km, tmerc) -- south <- km 67 ... km 52 -> north")
ax.set_ylabel("z (km)")
ax.set_title(f"(a) PyLith fault mesh ({mesh_preset}): {len(fault_tri)} triangles, projected on y-z", loc="left")
ax.legend(loc="lower left", fontsize=10)
ax = axes[1]
mesh_centroids = fault_tri.mean(1)
cell_size_dots = ax.scatter(mesh_centroids[:, 1] / 1e3, mesh_centroids[:, 2] / 1e3, c=edge_length_m, s=4, cmap="cividis", vmin=50, vmax=300)
fig.colorbar(cell_size_dots, ax=ax, label="mean edge length (m)")
ax.set_xlim(-6.6, 6.2)
ax.set_ylim(-3.05, 0.05)
ax.set_xlabel("y north (km)")
ax.set_ylabel("z (km)")
ax.set_title("(b) fault cell size", loc="left")
fig.savefig(os.path.join(fig_dir, "qc08a_mesh_fault.png"), dpi=300)
plt.close(fig)
print("wrote qc08a_mesh_fault.png")

# %% [markdown]
# ## [3] Output points and materials
#
# The 2386 quadtree samples of 07a in the mesh CRS (`modeling_inputs/points/insar_points.{txt,npz}`, written by 07a
# [6]). SFCVM v26.0 on a non-uniform grid (50 m horizontally around the Tule Pond section, 25 m
# vertically to 500 m), aligned so that its Hayward block boundary ({22|23}|7) at the surface lies
# on the mapped trace (a local cross-fault shift b(km), full within 3 km of the trace, tapered to
# zero at 6 km; dip kept), vs floored at 150 m/s, Poisson's ratio capped at 0.40
# (`pylith/build/sfcvm_grid.py`), written as a spatialdata SimpleGridDB by spatialdata's own writer
# (`pylith/build/write_sfcvm_griddb.py`). The unshifted ("as is") grid is read too, for the figure.

# %%
## [3]
model_data = np.load(os.path.join(dir_07a, "model_data_07a.npz"))
points_file = str(model_data["points_default"])                      # .txt (PyLith PointsList) / .npz, written by 07a [6]
assert os.path.exists(points_file + ".npz") and os.path.exists(points_file + ".txt"), f"missing {points_file}: run 07a_uniform_model"
points = np.load(points_file + ".npz")
for grid_tag, flag in (("aligned", ""), ("asis", "--no-align")):
    cached(os.path.join(paths_sfcvm["spatialdb"], f"sfcvm_v26-0_{grid_tag}_grid.npz"),
           f"python {build_dir}/sfcvm_grid.py {flag}")
cached(os.path.join(paths_sfcvm["spatialdb"], "mat_sfcvm_v26-0_aligned.spatialdb"),
       f"{in_pylith} python3 {build_dir}/write_sfcvm_griddb.py modeling_SFCVM/spatialdb/sfcvm_v26-0_aligned_grid.npz "
       f"modeling_SFCVM/spatialdb/mat_sfcvm_v26-0_aligned.spatialdb")
grid_file = os.path.join(paths_sfcvm["spatialdb"], "sfcvm_v26-0_aligned_grid.npz")
grid = np.load(grid_file)
print(str(grid["provenance"]))
print(f"{points['lon'].size} output points; grid {grid['x'].size} x {grid['y'].size} x {grid['z'].size}, "
      f"vs floor {float(grid['vs_floor']):g} m/s, nu cap {float(grid['nu_cap']):g}")
section_lengths = pd.read_csv(os.path.join(events_dir, "section_lengths.csv")).set_index("name")
sse_km = (float(section_lengths.loc["tulepond_creep_length", "km_lo"]), float(section_lengths.loc["tulepond_creep_length", "km_hi"]))
import sys
sys.path.insert(0, os.path.join(repo_dir, "scripts", "model_plots"))
import plot_materials
plot_materials.main(paths_sfcvm["spatialdb"], fig_dir, sse_km)

# %% [markdown]
# ## [4] PyLith Green's functions, both media
#
# `GreensFns` with `FaultCohesiveImpulses` (`pylith/cfg/gf.cfg`): unit left-lateral
# impulses (right-lateral = -left-lateral, sign derived in [5]) on every fault vertex of km 54-65
# over the full 3 km depth (buried edges excluded automatically), the rest of the fault locked;
# Dirichlet zero on the sides and bottom, free top; displacement output at the samples. Slip basis
# order 1 (tent per vertex), as the PyLith examples require for inversion. The runs
# (`gf_default_homog` in `modeling_uniform/pylith_output/`, `gf_default_sfcvm` in `modeling_SFCVM/pylith_output/`;
# 33 min each on 8 ranks; `pylith/run_gf.sh`).

# %%
## [4]
impulse_db = os.path.relpath(os.path.join(paths_inputs["spatialdb"], "impulses_km54-65_z3000.spatialdb"), WORK)   # y window = impulse_y_m, depth impulse_depth_m
cached(os.path.join(WORK, impulse_db), f"{in_pylith} python3 {build_dir}/make_impulse_db.py {impulse_y_m[0]} {impulse_y_m[1]} "
                                       f"{impulse_depth_m} {impulse_db}")
points_rel = os.path.relpath(points_file + ".txt", WORK)
for medium, mat_cfg in media.items():
    name = run_names[medium]
    output_dir = medium_paths(medium)["pylith_output"]
    cached(os.path.join(output_dir, f"{name}-points.h5"),
           f"bash {pylith_dir}/run_gf.sh {name} {mat_cfg} {mesh_rel} {impulse_db} {n_procs} {points_rel}")
    for suffix in ("-fault.h5", "-elastic_info.h5", "-parameters.json"):
        assert os.path.exists(os.path.join(output_dir, name + suffix)), f"missing {name}{suffix}"
    run_parameters = json.load(open(os.path.join(output_dir, f"{name}-parameters.json")))
    print(f"{name}: PyLith {run_parameters['version']['pylith']['version']}, {run_parameters['timestamp']}, "
          f"{run_parameters['platform']['machine']}")

# %% [markdown]
# ## [5] G matrices, the tent fault and the surface constraint
#
# `model_utils.tent_model_setup` (the same call as every run of 08c): for each medium `model_utils.load_G` reads
# `G_<run>.npz` from the medium's `gf/` folder, built by `model_utils.build_G` (pattern of PyLith `invert_slip.py`):
# impulse -> vertex (unit tent asserted), output point -> sample (KD-tree, < 1 mm asserted), 3D displacement
# projected on 07a's fault-parallel look vector, sign derived (right-lateral must move the east side along +look, as
# the data); a G projected on another strike than the current field's is rebuilt from the cached PyLith output (no
# PyLith run). Both media must share the fault mesh and the impulse order. The same call builds the tent fault
# (`model_utils.TentFault`: triangles, centroids, normals, areas, the impulse vertices and their km from Point Pinole)
# and the surface constraint of [10].

# %%
## [5]
frame_lonlat = tuple(model_data["frame_lonlat"])
strike_bearing = float(model_data["strike_deg"])
look = model_data["look"]
profile_lats = pd.read_csv(common_paths["fault_profiles"]).ft_lat.values
lake_km = np.sort(model_utils.km_from_pinole(np.array([lat for _, lat in lake_elizabeth_trace_lonlat]), profile_lats))   # Lake Elizabeth on the trace (Fig 2K bars)
trace = model_utils.model_trace(model_data["trace_lonlat"], -90.0, 90.0, frame_lonlat, profile_lats,
                                every_km=float(model_data["trace_every_km"]))
trace_south_north = trace["lonlat"][np.argsort(trace["lonlat"][:, 1])]
creep_lo_km, creep_hi_km = float(model_data["creep_lo_km"]), float(model_data["creep_hi_km"])
km_window = tuple(model_data["km_window"])
surf_floor_mm = float(model_data["surf_floor_mm"])
crossings = dict(km=model_data["crossing_km"], lat=model_data["crossing_lat"], mean_mm=model_data["crossing_mean_mm"])
medium_dirs = {medium: (medium_paths(medium)["pylith_output"], medium_paths(medium)["gf"]) for medium in media}
t0 = time.time()
setup = model_utils.tent_model_setup({"sfcvm": run_names["sfcvm"], "homog": run_names["homog"]}, points_file + ".npz",
                                     strike_bearing, medium_dirs, crossings, km_window, creep_lo_km, creep_hi_km, surf_floor_mm,
                                     trace_south_north, (trace["x_km"] * 1e3, trace["y_km"] * 1e3), frame_lonlat, CRS, profile_lats,
                                         near_trace_cut_km=model_near_trace_cut_km, trace_km_pp=trace["km_pp"])
G = setup["G"]                              # medium -> G matrix (samples x impulses), right-lateral positive, m per m
gnpz = setup["gnpz"]                        # medium -> the G_<run>.npz product
tents = setup["tents"]
fault_xyz, fault_cells, vertex_index = tents.fault_xyz, tents.fault_cells, tents.vertex_index
n_imp = tents.n_imp
tri_xyz, tri_cen, tri_n, tri_area = tents.tri_xyz, tents.tri_cen, tents.tri_n, tents.tri_area
imp_xyz, imp_km, imp_depth_km, sub_faces = tents.imp_xyz, tents.imp_km, tents.imp_depth_km, tents.sub_faces
top = tents.top                                                               # the mesh fault's top-edge vertices, south to north
print(f"G: {G['sfcvm'].shape[0]} samples x {n_imp} impulses (of {len(fault_xyz)} fault vertices); "
      + ", ".join(f"max |G| {m} {np.abs(G[m]).max():.3f}" for m in media) + f" m/m; G strike {strike_bearing:.4f} "
      f"(07a's field); CSI TriangularTents: {len(setup['template'].tent)} tents, {len(setup['template'].Faces)} faces "
      f"({time.time() - t0:.0f} s)")

# %% [markdown]
# ## [6] The rigidity beside the fault, and CHECK: PyLith read the SFCVM material as intended
#
# The per-cell shear modulus PyLith actually used (`-elastic_info.h5`, basis order 0) against the
# aligned grid interpolated at the cell centroids, for cells in the fine grid box; and mu of the
# cells next to the fault (nearest cell centroid on each side of every fault triangle), used for
# the SFCVM moment (`Mw_sfcvm`) and the rigidity options of [15c] (`model_utils.mu_beside_fault`).

# %%
## [6]
elastic_info = os.path.join(paths_sfcvm["pylith_output"], f"{run_names['sfcvm']}-elastic_info.h5")
mu_mean, mu_west, mu_east, mu_rel_median = model_utils.mu_beside_fault(elastic_info, grid_file, tents, "sfcvm")
assert mu_rel_median < 0.05, "PyLith's cell rigidity differs from the aligned grid by more than 5 % (median)"
mu_fault = {"sfcvm": mu_mean}
shallow_tri = tri_cen[:, 2] > -100
mu_top100 = dict(west_gpa=float(np.median(mu_west[shallow_tri]) / 1e9), east_gpa=float(np.median(mu_east[shallow_tri]) / 1e9))
print("PASS: PyLith used the aligned SFCVM rigidity")

# %% [markdown]
# ## [7] 07a's data, and CHECK: the samples sit on the same side of both traces
#
# The quadtree samples, their data and noise model from 07a's `model_data_07a.npz` (a `csi.insar` data set rebuilt
# with `model_utils.samples_from_arrays`; its full Cd with `model_utils.build_cd`), the measured crossings of the
# surface constraint, and 07a's `fig2_model_row.npz` (its baseline on the same samples) for the comparison rows; the
# samples must be the PyLith output points. Then the side check: the mesh fault (a polyline through its top-edge
# vertices, on the spline through every 3rd trace point) vs 07a's trace (20 m discretisation): are any samples on
# different sides?

# %%
## [7]
cov_p = model_data["cov_p"]
nug_power = float(model_data["nug_power"])
lam_xz = tuple(model_data["lam_xz"])
sigma_grid = sorted(set(float(s) for s in model_sigma_grid) | {float(sigma_locked)})   # the 3D-model grid (2026-10-16), not 07a's
samples = model_utils.samples_from_arrays("Downsampled fault-parallel", model_data["sample_lon"], model_data["sample_lat"],
                                          model_data["sample_x_km"], model_data["sample_y_km"], model_data["data_mm"],
                                          model_data["wgt"], look, frame_lonlat, cov_p, nug_power)
field_fingerprint = model_utils.field_fingerprint(os.path.join(field_dir, "fp_maps.npz"))   # the 06 field every inversion here fits
assert str(model_data["field_fingerprint"]) == field_fingerprint, "07a's model data were made from another 06 field: re-run 07a"
row07_file = os.path.join(dir_07a, "fig2_model_row.npz")
assert os.path.exists(row07_file), f"missing {row07_file}: run 07a_uniform_model first"
row07 = np.load(row07_file)
assert samples.lon.size == points["lon"].size and np.array_equal(samples.lon, points["lon"]) \
    and np.array_equal(samples.lat, points["lat"]), "the quadtree samples are not the PyLith output points"
assert np.array_equal(row07["sample_lon"], samples.lon) and np.ptp(samples.vel - row07["data_mm"]) < 1e-9, \
    "07a's fig2_model_row.npz is not on the same samples / data as this run: re-run 07a"
block_lon = model_data["block_lon"]
block_lat = model_data["block_lat"]
mesh_nodes, mesh_faces, dip_deg = model_data["mesh_nodes_sz"], model_data["mesh_faces"], float(model_data["mesh_dip_deg"])
fault07_trace = model_utils.wrap_fault(mesh_nodes, mesh_faces, dip_deg, "mean event", trace, frame_lonlat)   # 07a's fault (trace), for maps
cell_area_km2 = model_utils.triangle_areas_km2(mesh_nodes, mesh_faces)
signed_d_km = setup["signed_d_km"]                                           # from 07a's trace (model_utils.signed_distance_m)
print(f"07a baseline: sigma {float(row07['sigma']):g}, rms {float(row07['rms_mm']):.3f} mm, Mw {float(row07['mw_30gpa']):.3f}; "
      f"{samples.vel.size} samples, sigma grid {sigma_grid}, lambda {lam_xz} km, surface floor {surf_floor_mm} mm")


def lat_to_km_pp(lats):
    """km from Point Pinole of each latitude (model_utils.km_from_pinole on the fault-profile trace)."""
    return model_utils.km_from_pinole(lats, profile_lats)


dense_s = np.linspace(0, 1, 20000)
top_arc = np.r_[0, np.cumsum(np.hypot(np.diff(top[:, 0]), np.diff(top[:, 1])))]
top_arc /= top_arc[-1]
top_dense_x = np.interp(dense_s, top_arc, top[:, 0])
top_dense_y = np.interp(dense_s, top_arc, top[:, 1])
signed_mesh_m = model_utils.signed_distance_m(top_dense_x, top_dense_y, points["x"], points["y"])
in_mesh = (points["y"] > top[0, 1]) & (points["y"] < top[-1, 1])
side_mismatch = in_mesh & (np.sign(signed_mesh_m) != np.sign(signed_d_km))
side_max_m = float(np.abs(signed_mesh_m[side_mismatch]).max()) if side_mismatch.any() else 0.0
print(f"samples on opposite sides of the mesh fault and of 07a's trace: {side_mismatch.sum()} "
      f"(largest distance from the mesh fault {side_max_m:.0f} m); "
      f"|mesh - 07a| signed distance: median {np.median(np.abs(signed_mesh_m[in_mesh] - 1e3 * signed_d_km[in_mesh])):.1f} m, "
      f"max {np.abs(signed_mesh_m[in_mesh] - 1e3 * signed_d_km[in_mesh]).max():.1f} m")
assert side_max_m < 50.0, "a sample more than 50 m from the mesh fault sits on the other side of 07a's trace: the meshes disagree"

# %% [markdown]
# ## [8] Validation of the Green's functions
#
# (a) Homogeneous PyLith against CSI's homogeneous triangular dislocations (Meade) for the same
# slip: 10 mm uniform on km 57-61, top 1 km. PyLith: 10 mm on every impulse vertex in the patch
# (so slip tapers to zero over one cell outside it). CSI: the PyLith piecewise-linear slip field
# interpolated (in y, z) to the centroid of every triangle of 07a's mesh, so both carry the same
# slip. The all-sample rms is dominated by the few samples that lie on opposite sides of the mesh
# fault and of 07a's trace (a full step each, [7]); the numbers that matter are the far-field
# (> 300 m) and the agreeing-side rms. (b) Homogeneous PyLith against the 2D screw dislocation
# u = (s/pi) arctan(D/x) (`model_utils.screw_dislocation_mm`) for uniform slip on km 54-65, top 1 km, at samples
# within 2 km of the trace and at least 3 km from the slip ends (D = 1 km + half a fault cell for the tent taper);
# PyLith is lower at 1-2 km, the 3D end effect of an 11 km patch, so this is a sanity check.

# %%
## [8]
slip_test = np.where((imp_km >= test_km[0]) & (imp_km <= test_km[1]) & (imp_xyz[:, 2] >= -test_depth_m - 1e-6), test_slip_mm, 0.0)
pred_pylith = G["homog"].dot(slip_test)
vertex_slip = tents.vertex_slip(slip_test)
tri_yz = mtri.Triangulation(fault_xyz[:, 1], fault_xyz[:, 2], fault_cells)
interp_slip = mtri.LinearTriInterpolator(tri_yz, vertex_slip)
fault_val = model_utils.wrap_fault(mesh_nodes, mesh_faces, dip_deg, "validation", trace, frame_lonlat)
csi_cen = np.array([p.mean(0) for p in fault_val.patch])              # km, depth +
csi_slip = np.asarray(interp_slip(csi_cen[:, 1] * 1e3, -csi_cen[:, 2] * 1e3).filled(0.0))
line = insar("validation", lon0=frame_lonlat[0], lat0=frame_lonlat[1], verbose=False)
line.read_from_binary(np.zeros(samples.lon.size), samples.lon, samples.lat, err=np.ones(samples.lon.size),
                      los=np.tile(look, (samples.lon.size, 1)), remove_nan=False, remove_zeros=False)
fault_val.buildGFs(line, vertical=True, slipdir="s", verbose=False)
pred_csi = -fault_val.G[line.name]["strikeslip"].dot(csi_slip)         # right-lateral positive (07a convention)
diff = pred_pylith - pred_csi
near_300 = np.abs(signed_d_km) <= 0.3
val_a = dict(max_abs_diff_mm=float(np.abs(diff).max()), rms_diff_mm=float(np.sqrt(np.mean(diff ** 2))),
             rms_signal_mm=float(np.sqrt(np.mean(pred_csi ** 2))),
             rms_diff_near300_mm=float(np.sqrt(np.mean(diff[near_300] ** 2))),
             rms_diff_far_mm=float(np.sqrt(np.mean(diff[~near_300] ** 2))),
             potency_pylith_m3=float(np.sum(tri_area * vertex_slip[fault_cells].mean(1) * 1e-3)),
             potency_csi_m3=float(np.sum(cell_area_km2 * 1e6 * csi_slip * 1e-3)),
             rms_diff_okside_mm=float(np.sqrt(np.mean(diff[~side_mismatch] ** 2))))
val_a["rel_rms"] = val_a["rms_diff_mm"] / val_a["rms_signal_mm"]
val_a["rel_rms_okside"] = val_a["rms_diff_okside_mm"] / val_a["rms_signal_mm"]
print("validation (a) PyLith homog vs CSI:", {k: round(v, 4) for k, v in val_a.items()})
# the all-sample rms is dominated by the few samples that sit on opposite sides of the two traces (a full step each);
# away from the trace and on agreeing sides the two dislocation codes must agree closely
assert val_a["rms_diff_far_mm"] < 0.02 * val_a["rms_signal_mm"], "PyLith homogeneous and CSI Meade disagree beyond 300 m of the trace"
assert val_a["rel_rms_okside"] < 0.10, "PyLith homogeneous and CSI Meade disagree by more than 10 % rms on the samples on agreeing sides"

# (b) 2D screw dislocation
slip_b = np.where(imp_xyz[:, 2] >= -test_depth_m - 1e-6, test_slip_mm, 0.0)
pred_b = G["homog"].dot(slip_b)
patch_cells = (np.abs(tri_cen[:, 1]) < 3000) & (tri_cen[:, 2] > -1500)
dx_fault_m = model_utils.median_edge_length_m(tri_xyz[patch_cells])
D_eff = test_depth_m + 0.5 * dx_fault_m
x_m = signed_mesh_m                                                  # distance from the MESH fault (the source)
km_samples = lat_to_km_pp(samples.lat)
sel_b = (np.abs(x_m) <= 2000) & (np.abs(x_m) >= 20) & (km_samples >= 57.0) & (km_samples <= 62.0)
screw = model_utils.screw_dislocation_mm(test_slip_mm, D_eff, x_m)
diff_b = pred_b[sel_b] - screw[sel_b]
val_b = dict(n=int(sel_b.sum()), D_eff_m=D_eff, rms_diff_mm=float(np.sqrt(np.mean(diff_b ** 2))),
             max_abs_diff_mm=float(np.abs(diff_b).max()), rms_signal_mm=float(np.sqrt(np.mean(screw[sel_b] ** 2))))
val_b["rel_rms"] = val_b["rms_diff_mm"] / val_b["rms_signal_mm"]
print("validation (b) PyLith homog vs 2D screw dislocation:", {k: round(v, 4) for k, v in val_b.items()})
assert val_b["rel_rms"] < 0.10, "PyLith homogeneous and the 2D screw dislocation disagree by more than 10 % rms"

mpl_paper_style(base=7)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), constrained_layout=True)
ax = axes[0]
ax.scatter(pred_csi, pred_pylith, s=3, c=np.where(near_300, OK["vermillion"], OK["blue"]), linewidths=0)
lim = np.abs(np.r_[pred_csi, pred_pylith]).max() * 1.05
ax.plot([-lim, lim], [-lim, lim], "k-", lw=0.6)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
ax.set_xlabel("CSI homogeneous (mm)")
ax.set_ylabel("PyLith homogeneous (mm)")
ax.set_title(r"$\bf{A}$  " + f"10 mm on km 57-61, top 1 km: rms diff {val_a['rms_diff_mm']:.3f} mm ({100 * val_a['rel_rms']:.1f}%)", loc="left", fontsize=7)
ax.plot([], [], "o", color=OK["vermillion"], label="within 300 m of trace")
ax.plot([], [], "o", color=OK["blue"], label="farther")
ax.legend(fontsize=5)
ax = axes[1]
ax.scatter(signed_d_km, diff, s=3, color=OK["black"], linewidths=0)
ax.axhline(0, color="0.5", lw=0.6)
ax.set_xlabel("Distance from 07a trace, east + (km)")
ax.set_ylabel("PyLith - CSI (mm)")
ax.set_title(r"$\bf{B}$  Difference vs distance", loc="left", fontsize=7)
ax = axes[2]
order_b = np.argsort(x_m[sel_b])
ax.plot(x_m[sel_b][order_b] / 1e3, screw[sel_b][order_b], "-", color=OK["orange"], lw=1.5, label=f"2D screw, D = {D_eff:.0f} m")
ax.scatter(x_m[sel_b] / 1e3, pred_b[sel_b], s=4, color=OK["blue"], label="PyLith homogeneous", zorder=3)
ax.set_xlabel("Distance from mesh fault, east + (km)")
ax.set_ylabel("Fault-parallel (mm)")
ax.set_title(r"$\bf{C}$  " + f"km 54-65 top 1 km: rms diff {val_b['rms_diff_mm']:.3f} mm ({100 * val_b['rel_rms']:.1f}%)", loc="left", fontsize=7)
ax.legend(fontsize=5)
fig.savefig(os.path.join(fig_dir, "figS_pylith_validation.png"), dpi=300)
plt.close(fig)
print("wrote figS_pylith_validation.png")

# %% [markdown]
# ## [9] CHECK: superposition and mesh convergence with forward runs
#
# PyLith `FaultCohesiveKin` forward runs (`pylith/cfg/fwd_kin.cfg`, one static step; `pylith/run_fwd.sh`) with
# 10 mm right-lateral on km 57-61 (y window), top 1 km, homogeneous, on the `default` and the
# `fine` mesh (60 m fault cells in the patch), read from `modeling_uniform/pylith_output/`.
# (c) superposition: default-mesh forward = G x (the forward run's own vertex slip) -- tests G
# assembly, point matching and sign, exact to 1e-4 mm (`model_utils.superposition_check`); (d) mesh convergence: the
# fine-mesh forward against CSI with the same (fine) slip field, and against the default-mesh forward. The same
# superposition check for the SFCVM medium is made in [14] with the Fig-2 forward run of the
# inverted slip. The CSI and PyLith predictions of (a)-(d) are saved in `modeling_uniform/validation/`.

# %%
## [9]
kin_db = os.path.relpath(os.path.join(paths_inputs["spatialdb"], "kin_slip_km57-61_z1000.spatialdb"), WORK)
cached(os.path.join(WORK, kin_db), f"{in_pylith} python3 {build_dir}/make_kin_slip_db.py {kin_y_m[0]} {kin_y_m[1]} {test_depth_m} "
                                   f"{test_slip_mm / 1e3} {kin_db}")
sample_xy = np.c_[points["x"], points["y"]]


def read_forward(name):
    """Fault-parallel displacement at the samples (mm) and the right-lateral vertex slip (mm) of a homogeneous forward run."""
    run_dir = paths_uniform["pylith_output"]
    u_fp = model_utils.read_forward_points(os.path.join(run_dir, f"{name}-points.h5"), sample_xy).dot(look)
    f_xyz, f_cells, slip_rl = model_utils.read_forward_slip(os.path.join(run_dir, f"{name}-fault.h5"), gnpz["homog"]["sign_rl"])
    return dict(u=u_fp, xyz=f_xyz, cells=f_cells, slip=slip_rl)


forward_runs = {}
for pre in ("default", "fine"):
    name = f"fwd_{pre}_homog"
    if pre == "fine":
        cached(os.path.join(paths_inputs["mesh"], "mesh_vertical_tet_fine.msh"),
               f"{in_pylith} python3 {build_dir}/generate_gmsh_vertical.py --write --preset fine")
    cached(os.path.join(paths_uniform["pylith_output"], f"{name}-points.h5"),
           f"bash {pylith_dir}/run_fwd.sh {name} mat_homog.cfg modeling_inputs/mesh/mesh_vertical_tet_{pre}.msh 2 fwd_kin.cfg")
    forward_runs[pre] = read_forward(name)
    print(f"{name}: slip {forward_runs[pre]['slip'].min():.2f}..{forward_runs[pre]['slip'].max():.2f} mm on {np.sum(forward_runs[pre]['slip'] > 1e-6)} vertices; "
          f"max |u| {np.abs(forward_runs[pre]['u']).max():.2f} mm")
    assert np.all(forward_runs[pre]["slip"] > -1e-6), f"{name}: forward slip is not right-lateral"

# (c) superposition on the default mesh, homogeneous medium
val_c = model_utils.superposition_check(G["homog"], imp_xyz, os.path.join(paths_uniform["pylith_output"], "fwd_default_homog-points.h5"),
                                        os.path.join(paths_uniform["pylith_output"], "fwd_default_homog-fault.h5"), sample_xy, look,
                                        gnpz["homog"]["sign_rl"])
print("validation (c) superposition, G[homog] x slip vs forward run fwd_default_homog:", {k: round(v, 6) for k, v in val_c.items()})
assert val_c["rms_diff_mm"] < 1e-4, "superposition failed: G[homog] is not consistent with fwd_default_homog"
print("PASS: G x slip reproduces the forward run (homogeneous)")


def csi_prediction(f_xyz, f_cells, slip_mm):
    """CSI Meade TDE prediction at the samples for a PyLith piecewise-linear slip field (interpolated in y-z to
    07a's triangle centroids)."""
    triangulation_yz = mtri.Triangulation(f_xyz[:, 1], f_xyz[:, 2], f_cells)
    slip_at_csi_centroids = mtri.LinearTriInterpolator(triangulation_yz, slip_mm)(csi_cen[:, 1] * 1e3, -csi_cen[:, 2] * 1e3)
    slip_at_csi_centroids = np.asarray(slip_at_csi_centroids.filled(0.0))
    return -fault_val.G[line.name]["strikeslip"].dot(slip_at_csi_centroids)


ok_side = ~side_mismatch
val_d = {}
csi_forward = {}
for pre in forward_runs:
    csi_forward[pre] = csi_prediction(forward_runs[pre]["xyz"], forward_runs[pre]["cells"], forward_runs[pre]["slip"])
    diff_pre = forward_runs[pre]["u"] - csi_forward[pre]
    val_d[pre] = dict(rms_diff_mm=float(np.sqrt(np.mean(diff_pre[ok_side] ** 2))),
                      rms_diff_near300_mm=float(np.sqrt(np.mean(diff_pre[ok_side & near_300] ** 2))),
                      rms_diff_far_mm=float(np.sqrt(np.mean(diff_pre[~near_300] ** 2))),
                      max_abs_diff_mm=float(np.abs(diff_pre[ok_side]).max()))
    print(f"validation (d) forward {pre} mesh vs CSI (same slip, {ok_side.sum()} samples on agreeing sides):",
          {k: round(v, 4) for k, v in val_d[pre].items()})
d_mesh = forward_runs["fine"]["u"] - forward_runs["default"]["u"]
val_d["fine_minus_default_rms_mm"] = float(np.sqrt(np.mean(d_mesh ** 2)))
val_d["fine_minus_default_near300_rms_mm"] = float(np.sqrt(np.mean(d_mesh[near_300] ** 2)))
print(f"validation (d) fine - default forward: rms {val_d['fine_minus_default_rms_mm']:.4f} mm, within 300 m "
      f"{val_d['fine_minus_default_near300_rms_mm']:.4f} mm, farther {np.sqrt(np.mean(d_mesh[~near_300] ** 2)):.4f} mm")
assert val_d["fine_minus_default_rms_mm"] < 0.2, "the default mesh is not converged: fine - default forward differ by > 0.2 mm rms (data rms ~1.4 mm)"
os.makedirs(paths_uniform["validation"], exist_ok=True)
np.savez(os.path.join(paths_uniform["validation"], "csi_vs_pylith_homog.npz"),
         sample_lon=samples.lon, sample_lat=samples.lat, signed_d_km=signed_d_km, signed_mesh_m=signed_mesh_m,
         side_mismatch=side_mismatch, a_pylith_mm=pred_pylith, a_csi_mm=pred_csi, b_pylith_mm=pred_b, b_screw_mm=screw,
         b_selected=sel_b, d_default_pylith_mm=forward_runs["default"]["u"], d_default_csi_mm=csi_forward["default"],
         d_fine_pylith_mm=forward_runs["fine"]["u"], d_fine_csi_mm=csi_forward["fine"],
         numbers=json.dumps(dict(a=val_a, b=val_b, c=val_c, d=val_d)),
         provenance=f"08a_3d_model {time.strftime('%Y-%m-%d %H:%M')}: validation (a) 10 mm on km 57-61 top 1 km, PyLith "
                    f"{run_names['homog']} vs CSI Meade on 07a's mesh; (b) vs the 2D screw dislocation (D {D_eff:.0f} m); "
                    f"(c), (d) forward runs fwd_default_homog, fwd_fine_homog")

plt.rcParams.update({"font.size": 12})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
for ax, pre in zip(axes, forward_runs):
    ax.scatter(signed_d_km[ok_side], (forward_runs[pre]["u"] - csi_forward[pre])[ok_side], s=6, color=OK["black"], linewidths=0)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-2, 2)
    ax.set_xlabel("Distance from trace, east + (km)")
    ax.set_ylabel("PyLith forward - CSI (mm)")
    ax.set_title(f"(d) {pre} mesh: rms {val_d[pre]['rms_diff_mm']:.3f} mm, within 300 m {val_d[pre]['rms_diff_near300_mm']:.3f} mm",
                 loc="left", fontsize=11)
fig.savefig(os.path.join(fig_dir, "qc08a_validation_mesh.png"), dpi=300)
plt.close(fig)
print("wrote qc08a_validation_mesh.png")

# %% [markdown]
# ## [10] The CSI fault on the PyLith vertices, and the inversion
#
# PyLith impulses are tents on fault vertices, not CSI triangles, so both media are inverted on
# the vertex parameterisation with CSI's `TriangularTents` (its `tent` list and `Vertices` = the
# impulse vertices, in G's column order; faces = the PyLith fault triangles among them;
# `model_utils.tent_fault_template`, built in [5]). The G matrix enters through `Fault.setGFs`. Everything else is
# 07a's inversion (`model_utils.invert_tents`): exponential prior
# `buildCmXY` with lambda (0.1, 0.05) km on the vertex positions, per-dataset constant
# (`csi.transformation`), NNLS (`multisourcesolve.ConstrainedLeastSquareSoln`). The surface
# constraint (Dani 2026-10-02, `model_utils.crossing_constraint`, built in [5]): every measured fault crossing
# inside the mesh (04's five-event mean step, one every 0.1 km) is one datum, tied to the modelled
# slip at its own position on the top edge, i.e. linear between the two neighbouring top-edge
# vertices (the order-1 tents); error = the local scatter of the crossings (1.4826 MAD within
# +/- 0.2 km), floored at 0.5 mm. No running median, nothing invented across data gaps.
# PREPUB: (Until 2026-10-02: 07a's 0.4 km running median interpolated to every vertex; its window test lost the
# PREPUB: edge crossings to float roundoff and kept 29 of 57 points. On the selected model the change leaves
# PREPUB: Mw30 at 3.756 and the rms within 0.02 mm.)
# The surface % of [12] compares the same measured crossings with the model: the median modelled
# slip on the top-edge vertices of the SSE section over the median of the measured crossings in the
# SSE section (`model_utils.surface_pct_denominator_mm`; Dani 2026-10-03). It is the ONE denominator of every
# surface % (07a, 08b and 08c use the same function; asserted equal to 07a's), so every surface % read from 08c in
# [12] and [14] is compared as it is.
# Also here: the numbers of one solution (`model_utils.tent_numbers`), and Mw from the moment.
# NB: `buildCmXY` normalises by lambda0 from the parameter spacing, so sigma is not numerically
# comparable with 07a's triangle sigma; each model gets its own L-curve corner, by 07a's rule.
#
# **Data covariance** (`model_utils.data_cd`). The one thing that differs between the selected model and its
# companion; a standard modelling choice, not new physics:
# - `full` (07a [4], the companion): Cd from the semivariogram of the mean-event field away from
#   the fault, nugget + exponential terms; the nugget is divided down by each quadtree block's
#   pixel count and the exponential terms give the OFF-diagonal (correlated) part. Estimating Cd
#   from the far-field semivariogram and carrying the correlated part into the inversion follows
#   Lohman and Simons (2005) and Sudhaus and Jonsson (2009).
# - `nugget` (the SELECTED model): the same per-block nugget variances on the diagonal, correlated
#   part dropped: samples treated as independent, weights scaled by block size. This is the
#   quadtree convention of Jonsson et al. (2002) and most InSAR slip inversions since. Under it
#   the west lobe must be fitted (the aligned SFCVM does; the homogeneous medium cannot); the
#   reduced chi2 (~5.5, printed in [12]) shows these variances are too small in absolute terms, so the bootstrap
#   errors of 08c are formal, and the structural range comes from the sensitivity runs.
#
# References (author, year, journal; check the DOI before citing):
# Lohman, R. B., and M. Simons (2005), Some thoughts on the use of InSAR data to constrain models
#   of surface deformation: Noise structure and data downsampling, Geochem. Geophys. Geosyst., 6, Q01007.
# Sudhaus, H., and S. Jonsson (2009), Improved source modelling through combined use of InSAR and
#   GPS under consideration of correlated data errors: application to the June 2000 Kleifarvatn
#   earthquake, Iceland, Geophys. J. Int., 176, 389-404.
# Jonsson, S., H. Zebker, P. Segall, and F. Amelung (2002), Fault slip distribution of the 1999
#   Mw 7.1 Hector Mine, California, earthquake, estimated from satellite radar and GPS
#   measurements, Bull. Seismol. Soc. Am., 92, 1377-1389.

# %%
## [10]
surface_rows = setup["constraint"]
section_crossing_median_mm = setup["denominator_mm"]
n_section_crossings = setup["n_section_crossings"]
assert abs(section_crossing_median_mm - float(model_data["crossing_median_mm"])) < 1e-12, \
    f"surface % denominators differ: 08a {section_crossing_median_mm} vs 07a {float(model_data['crossing_median_mm'])} (07a out of date?)"
print(f"surface % denominator: median of {n_section_crossings} measured crossings in the SSE section "
      f"{section_crossing_median_mm:.2f} mm (= 07a's crossing_median)")
print(f"surface constraint: {surface_rows['vel'].size} crossings inside the mesh top edge "
      f"(km {surface_rows['km'].min():.1f}-{surface_rows['km'].max():.1f}; {surface_rows['n_outside']} beyond its ends), "
      f"distance to the edge median {np.median(surface_rows['dist_m']):.1f} m, max {surface_rows['dist_m'].max():.1f} m; "
      f"error median {np.median(surface_rows['err']):.2f} mm")
west300, east300 = setup["west300"], setup["east300"]
in_sec_surface = setup["in_section_surface"]
surface_imp = tents.surface_imp
data_covariance = {cd: model_utils.data_cd(cd, samples, cov_p, nug_power, keep=setup["keep"]) for cd in CD_TAG}   # the two Cd treatments


def invert_tents(medium, sigma, cd, vel_insar=None, vel_surface=None):
    """One solve on the tents (model_utils.invert_tents) with this notebook's data, G and Cd; returns
    (fault, InSAR data, surface data). vel_insar / vel_surface: replacement data vectors (08b's recovery test)."""
    return model_utils.invert_tents(setup["template"], samples, G[medium], data_covariance[cd], setup["surface"],
                                    setup["G_surface"], sigma, lam_xz, frame_lonlat, use_surface=use_surface,
                                    vel_insar=vel_insar, vel_surface=vel_surface, name=f"{medium} {cd} sigma {sigma:g}")


def numbers(fault_model, insar_data, surface):
    """The numbers of one solution (model_utils.tent_numbers; Mw_sfcvm with the SFCVM rigidity beside the fault)."""
    return model_utils.tent_numbers(fault_model.slip[:, 0], insar_data, surface, tents, west300, east300, in_sec_surface,
                                    section_crossing_median_mm, mu_ref, mu_fault["sfcvm"], keep=setup["keep"])


moment_magnitude = model_utils.moment_magnitude
print(f"surface constraint {'ON' if use_surface else 'OFF'} (model_use_surface); the other setting is 08c's `_surf` / `_nosurf` run")

# %% [markdown]
# ## [11] Sigma: the L-curve corner for each model (07a [9] rule)
#
# The 3D-model sigma grid (`TulePond_filepaths.model_sigma_grid`: 07a's grid with 1-wide steps from 10 to 16), the weighted
# misfit (InSAR chi2, plus the surface chi2 only when the surface data are fitted) against the prior-weighted roughness
# (`model_utils.tent_lcurve`), corner by maximum curvature of cubic splines in log sigma (end intervals excluded;
# `model_utils.lcurve_corner`). The locked sigma IS the corner of the SELECTED model's L-curve, to 0.1 (asserted); every
# model, and every 08c run, is solved at it, and each records its own corner. The locked sigma is the value on the
# 1-wide part of the grid nearest the corner (within 0.5, asserted).
# CHECK: each L-curve equals 08c's run of the same settings (`lcurve_<medium><tag>.csv`), made by the same
# functions on the same products.
#
# History: until 2026-10-16 the rule was the smallest grid value at or above the corner on 07a's coarse grid (sigma 12;
# Dani 2026-10-03, after a one-off check that the corner itself changed Mw30 by < 0.01). On 2026-10-16, with the
# surface constraint off, the corner moved to ~12.4 and Dani chose the refined grid and the corner itself; with the
# 60 m near-trace cut (same day) it moved to ~15.9 (sigma 16); the cut was dropped again that night: corner 12.4, sigma 12.

# %%
## [11]
def lcurve(medium, cd):
    """One L-curve over sigma_grid: misfit, roughness, rms and Mw30 per sigma; the corner and the selected sigma."""
    def solve(sigma):
        return invert_tents(medium, sigma, cd)
    lcurve_table = model_utils.tent_lcurve(solve, sigma_grid, tents, mu_ref, f"{medium} {cd}", use_surface=use_surface)
    corner, _, _ = model_utils.lcurve_corner(lcurve_table.sigma.values, np.sqrt(lcurve_table.chi2_total.values),
                                             lcurve_table.prior_norm.values)
    selected = sigma_locked                                   # every model is solved at the locked sigma
    if (medium, cd) == SELECTED:                              # the locked sigma IS the selected model's corner, to 0.1
        assert abs(corner - sigma_locked) <= 0.5 + 1e-9, \
            f"the selected model's L-curve corner is now {corner:.2f}: set model_sigma_locked to the nearest 1-wide grid value"
    lcurve_table["corner"] = corner
    lcurve_table["selected"] = selected
    lcurve_table.to_csv(os.path.join(out_dir, f"lcurve_{medium}{CD_TAG[cd]}.csv"), index=False)
    print(f"{medium} {cd}: L-curve corner sigma {corner:.1f} -> selected {selected:g}")
    return lcurve_table, corner, selected


families = {}
for medium, cd in RUNS:
    families[(medium, cd)] = lcurve(medium, cd)
    lcurve_08c = pd.read_csv(os.path.join(dir_08c, f"lcurve_{medium}{CD_TAG[cd]}.csv"))     # CHECK: 08c's run of the same settings
    lcurve_table = families[(medium, cd)][0]
    assert np.array_equal(lcurve_08c.sigma.values, lcurve_table.sigma.values), "08c's L-curve is on another sigma grid"
    lc_diff = float(np.max(np.abs(lcurve_08c.chi2_total.values / lcurve_table.chi2_total.values - 1)))
    assert lc_diff < 1e-9 and abs(float(lcurve_08c.corner.iloc[0]) - families[(medium, cd)][1]) < 1e-9, \
        f"{medium} {cd}: L-curve differs from 08c's (max rel chi2 diff {lc_diff:.1e}): re-run 08c"
    print(f"PASS: {medium} {cd} L-curve equals 08c's lcurve_{medium}{CD_TAG[cd]}.csv (max rel chi2 diff {lc_diff:.1e})")

mpl_paper_style(base=7)
fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), constrained_layout=True)
for ax, (medium, cd), letter in zip(axes.ravel(), RUNS, "ABCD"):
    lcurve_table, corner, selected = families[(medium, cd)]
    colour = OK["vermillion"] if medium == "sfcvm" else OK["blue"]
    ax.plot(np.sqrt(lcurve_table.chi2_total), lcurve_table.prior_norm, "o-", color=colour, ms=3, lw=0.8)
    for sigma_value, misfit_value, rough_value in zip(lcurve_table.sigma, np.sqrt(lcurve_table.chi2_total), lcurve_table.prior_norm):
        ax.annotate(f"{sigma_value:g}", (misfit_value, rough_value), xytext=(3, 2), textcoords="offset points", fontsize=5)
    pick = lcurve_table.sigma == selected
    ax.plot(np.sqrt(lcurve_table.chi2_total[pick]), lcurve_table.prior_norm[pick], "s", ms=7, mfc="none", mec="k",
            label=f"selected sigma {selected:g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=6)
    ax.set_xlabel(r"weighted misfit $\sqrt{\chi^2}$")
    ax.set_ylabel(r"roughness $\sqrt{m^T C^{-1} m}$")
    ax.set_title(r"$\bf{" + letter + r"}$  " + f"PyLith {medium}, Cd {cd} (corner {corner:.1f})", loc="left", fontsize=7)
fig.savefig(os.path.join(fig_dir, "figS_pylith_lcurves.png"), dpi=300)
plt.close(fig)
print("wrote figS_pylith_lcurves.png")

# %% [markdown]
# ## [12] The four models and their numbers
#
# rms, rms within 300 m west / east of 07a's trace, chi2 (InSAR, the run's own Cd), surface %
# (median modelled slip over the top-edge vertices in the SSE section / median of the measured
# crossings in the SSE section, [10]), potency (exact tent integral), Mw at 30 GPa and with the SFCVM
# rigidity of the PyLith cells beside the fault (`mu_effective` = that moment / potency). 07a's CSI
# baseline is re-scored from `fig2_model_row.npz` on the same samples under each Cd (its surface % is on
# the same measured-crossing denominator; 07a quotes Mw at 30 GPa only, so its Mw_sfcvm is empty).
# CHECK: the two PyLith rows equal 08c's `results_comparison<tag>.csv` (same settings) on sigma, rms, rms west,
# potency, Mw30, Mw_sfcvm and surface %, and fit the same 06 field.

# %%
## [12]
res07 = row07["residual_mm"]
models = {}                     # (medium, cd) -> the solution at the selected sigma and its numbers
results = {}                    # cd -> the comparison table
for cd in CD_TAG:
    Cd07 = data_covariance[cd]
    table = [dict(model="CSI homogeneous (07)", sigma=float(row07["sigma"]),
                  rms_mm=float(np.sqrt(np.mean(res07 ** 2))), rms_west300_mm=float(np.sqrt(np.mean(res07[west300] ** 2))),
                  rms_east300_mm=float(np.sqrt(np.mean(res07[east300] ** 2))), chi2=float(np.dot(res07, np.linalg.solve(Cd07, res07))),
                  surface_pct=float(row07["surf_ratio_pct"]), potency_m3=np.nan,
                  Mw_30gpa=float(row07["mw_30gpa"]), Mw_sfcvm=np.nan)]
    for medium in media:
        lcurve_table, corner, selected = families[(medium, cd)]
        fault_model, insar_data, surface = invert_tents(medium, selected, cd)
        num = numbers(fault_model, insar_data, surface)
        models[(medium, cd)] = dict(slip=fault_model.slip[:, 0].copy(), data=insar_data.vel.copy(), model=insar_data.synth.copy(),
                                    residual=insar_data.vel - insar_data.synth, sigma=selected, corner=corner, **num)
        table.append(dict(model=f"PyLith {medium}", sigma=selected, **num))
    results[cd] = pd.DataFrame(table).set_index("model")
    results[cd].to_csv(os.path.join(out_dir, f"results_comparison{CD_TAG[cd]}.csv"))
    results_08c = pd.read_csv(os.path.join(dir_08c, f"results_comparison{CD_TAG[cd]}.csv")).set_index("model")   # CHECK: 08c's run
    for medium in media:
        assert str(results_08c.loc[f"PyLith {medium}", "field_fingerprint"]) == field_fingerprint, \
            f"08c's results_comparison{CD_TAG[cd]}.csv ({medium}) fits another 06 field: re-run 08c"
        for col in ("sigma", "rms_mm", "rms_west300_mm", "potency_m3", "Mw_30gpa", "Mw_sfcvm", "surface_pct"):
            mine = float(results[cd].loc[f"PyLith {medium}", col])
            theirs = float(results_08c.loc[f"PyLith {medium}", col])
            assert abs(mine - theirs) <= 1e-9 * max(abs(theirs), 1e-9), f"{medium} {cd} {col}: {mine} vs 08c's {theirs}"
    print(f"PASS: results ({run_tag(cd)}) equal 08c's results_comparison{CD_TAG[cd]}.csv")
    print(f"--- Cd {cd} ({run_tag(cd)})")
    print(results[cd].round(3).T.to_string())
selected_model = models[SELECTED]
print(f"\nSELECTED ({run_tag(SELECTED[1])} / {SELECTED[0]}): sigma {selected_model['sigma']:g}, rms {selected_model['rms_mm']:.2f} mm "
      f"(W {selected_model['rms_west300_mm']:.2f} / E {selected_model['rms_east300_mm']:.2f}), surface {selected_model['surface_pct']:.0f} %, "
      f"potency {selected_model['potency_m3']:.0f} m3, Mw30 {selected_model['Mw_30gpa']:.2f}, Mw_sfcvm {selected_model['Mw_sfcvm']:.2f} "
      f"(mu_eff {selected_model['mu_effective_gpa']:.2f} GPa); reduced chi2 {selected_model['chi2'] / samples.vel.size:.2f} "
      f"(chi2 {selected_model['chi2']:.0f} on {samples.vel.size} samples)")

# %% [markdown]
# ## [13] Figures: fit maps, slip, the X-X' profile
#
# For each covariance treatment: data / model / residual at the quadtree blocks (07a's CSI
# baseline, PyLith homogeneous, PyLith SFCVM). Then the four slip distributions on the unrolled
# fault, and the X-X' profile of notebook 06 with every model at the samples within 300 m of the line.

# %%
## [13]
block_poly = np.stack([block_lon, block_lat], axis=-1)
trace_plot = np.c_[fault07_trace.lon, fault07_trace.lat]
mpl_paper_style(base=7)
for cd in CD_TAG:
    rows_maps = [("CSI homogeneous (07a)", row07["data_mm"], row07["model_mm"], row07["residual_mm"])]
    for medium in media:
        rows_maps.append((f"PyLith {medium}", models[(medium, cd)]["data"], models[(medium, cd)]["model"], models[(medium, cd)]["residual"]))
    fig, axes = plt.subplots(len(rows_maps), 3, figsize=(7.2, 2.5 * len(rows_maps)), squeeze=False, constrained_layout=True)
    for row, (label, data_mm, model_mm, residual_mm) in enumerate(rows_maps):
        for col, (values, title) in enumerate(((data_mm, "data"), (model_mm, "model"), (residual_mm, "residual"))):
            ax = axes[row, col]
            blocks = PolyCollection(block_poly, array=values, cmap="RdBu_r", edgecolors="none")
            blocks.set_clim(-4, 4)
            ax.add_collection(blocks)
            ax.plot(trace_plot[:, 0], trace_plot[:, 1], "k-", lw=0.5)
            ax.set_xlim(-122.08, -121.86)
            ax.set_ylim(37.47, 37.63)
            ax.set_aspect(1 / np.cos(np.deg2rad(frame_lonlat[1])))
            rms_txt = f" (rms {np.sqrt(np.mean(residual_mm ** 2)):.2f} mm)" if col == 2 else ""
            ax.set_title(r"$\bf{" + "ABCDEFGHI"[3 * row + col] + r"}$  " + f"{label}: {title}{rms_txt}", loc="left", fontsize=7)
            ax.set_xticks([-122.05, -121.95], ["-122.05", "-121.95"])
            ax.set_yticks([37.50, 37.55, 37.60], ["37.50", "37.55", "37.60"])
    fig.colorbar(blocks, ax=axes, shrink=0.5, orientation="horizontal", label="fault-parallel displacement (mm)")
    fig.savefig(os.path.join(fig_dir, f"figS_pylith_fit_maps_{run_tag(cd)}.png"), dpi=300)
    plt.close(fig)
    print(f"wrote figS_pylith_fit_maps_{run_tag(cd)}.png")

fig, axes = plt.subplots(len(RUNS), 1, figsize=(7.2, 1.9 * len(RUNS)), squeeze=False, constrained_layout=True)
axes = axes[:, 0]
vmax = 0.0
for run in RUNS:
    vmax = max(vmax, models[run]["slip"].max())
tri_imp = mtri.Triangulation(imp_km, imp_xyz[:, 2] / 1e3, sub_faces)
for ax, (medium, cd), letter in zip(axes, RUNS, "ABCD"):
    mdl = models[(medium, cd)]
    slip_image = ax.tripcolor(tri_imp, mdl["slip"], shading="gouraud", cmap="magma", vmin=0, vmax=vmax)
    ax.set_xlim(54, 65)
    ax.set_ylim(-1.5, 0)
    ax.invert_xaxis()
    ax.set_xlabel("km from Point Pinole (orange bar: SSE section)")
    ax.set_ylabel("z (km)")
    if (medium, cd) == SELECTED:
        tag_txt = "SELECTED, "
    elif (medium, cd) == COMPANION:
        tag_txt = "companion, "
    else:
        tag_txt = ""
    ax.set_title(r"$\bf{" + letter + r"}$  " + f"{tag_txt}PyLith {medium}, Cd {cd}: sigma {mdl['sigma']:g}, rms {mdl['rms_mm']:.2f} mm, "
                 f"Mw {mdl['Mw_30gpa']:.2f} (30 GPa), {mdl['Mw_sfcvm']:.2f} (SFSVM mu)", loc="left", fontsize=7)
    ax.hlines(0.04, creep_lo_km, creep_hi_km, color=OK["orange"], lw=4, clip_on=False)   # SSE section (04)
fig.colorbar(slip_image, ax=axes, label="right-lateral slip (mm)", shrink=0.6)
fig.savefig(os.path.join(fig_dir, "figS_pylith_slip.png"), dpi=300)
plt.close(fig)
print("wrote figS_pylith_slip.png")

fig2_in = np.load(os.path.join(field_dir, "fig2_inputs.npz"))
to_field = pyproj.Transformer.from_crs(4326, int(fig2_in["epsg"]), always_xy=True)
xw, yw, xe, ye = fig2_in["prof_fp_X_ends"]                       # the X-X' line ends, west and east (m, field CRS)
line_dir = np.array([xe - xw, ye - yw])
length = np.linalg.norm(line_dir)
line_dir /= length
sample_x, sample_y = to_field.transform(samples.lon, samples.lat)
along = (sample_x - xw) * line_dir[0] + (sample_y - yw) * line_dir[1]
perp = -(sample_x - xw) * line_dir[1] + (sample_y - yw) * line_dir[0]
sel = (np.abs(perp) <= 300) & (along >= 0) & (along <= length)
order_p = np.argsort(signed_d_km[sel])
fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
ax.plot(fig2_in["prof_fp_X_d"] / 1e3, fig2_in["prof_fp_X_v"], "-", color="0.7", lw=0.8, label="06 X-X' pixel profile (far field zeroed)")
ax.plot(signed_d_km[sel], samples.vel[sel] - np.median(samples.vel[sel]), "o", color=OK["black"], ms=2.5,
        label="data at quadtree samples within 300 m (median removed)")
ax.plot(signed_d_km[sel][order_p], (row07["model_mm"] - np.median(row07["model_mm"][sel]))[sel][order_p], "-", color=OK["grey"], lw=1.2,
        label="CSI homogeneous (07a)")
styles = {("sfcvm", "nugget"): (OK["vermillion"], "-"), ("sfcvm", "full"): (OK["vermillion"], "--"),
          ("homog", "nugget"): (OK["blue"], "-"), ("homog", "full"): (OK["blue"], "--")}
for (medium, cd) in RUNS:
    model_mm = models[(medium, cd)]["model"]
    label = f"PyLith {medium}, Cd {cd}"
    if (medium, cd) == SELECTED:
        label += " (selected)"
    ax.plot(signed_d_km[sel][order_p], (model_mm - np.median(model_mm[sel]))[sel][order_p], styles[(medium, cd)][1],
            color=styles[(medium, cd)][0], lw=1.2, label=label)
ax.axvline(0, color="k", lw=0.5)
ax.set_xlim(-2.5, 2.5)
ax.set_xlabel("distance from the trace, east + (km)")
ax.set_ylabel("fault-parallel (mm)")
ax.set_title("X-X' (lat ~37.5612 N): data vs models at the samples within 300 m of the line", loc="left", fontsize=7)
ax.legend(fontsize=5)
fig.savefig(os.path.join(fig_dir, "figS_pylith_profile_XX.png"), dpi=300)
plt.close(fig)
print("wrote figS_pylith_profile_XX.png")

# %% [markdown]
# ## [14] Fig-2 products, with the forward-run superposition CHECK
#
# `07c_fig2` draws the model row of Fig 2 from `fig2_model_row_pylith_nuggetcd.npz` (the selected
# model), with the homogeneous PyLith model of the same inversion (`_nuggetcd_homog`) as the
# comparison line on the profiles, and `_full` for the companion. Same keys as 07a's
# `fig2_model_row.npz` [14], with the fault on the impulse vertices (`nodes_kmz`, `faces`,
# `slip_vertex_mm`). The profile lines X-X' and Y-Y' (fault-parallel and vertical, every 10 m,
# zeroed on the far field as 06 zeroes the observed profiles) come from ONE PyLith
# `FaultCohesiveKin` forward run of the inverted vertex slip (`pylith/cfg/fwd_kin_simpledb.cfg`: a SimpleDB
# with the vertex slip, "nearest", the same mesh and material as the GF run, output at the profile
# points and the samples; `pylith/run_fwd.sh`), read from the medium's `pylith_output/fwd_fig2_<label>`.
#
# CHECKS: the forward run's imposed slip equals the slip just inverted (< 1e-4 mm: the cached
# forward run IS this model), and the forward displacement at the samples equals G x slip
# (< 0.05 mm rms: superposition for the SFCVM medium, the counterpart of [9]'s homogeneous check).
# The Mw30 range on the product is the L-curve grid neighbours of this run plus the +/- 50 m
# alignment runs of 08c (`results_comparison_{e50,w50}<tag>.csv`), both at the locked sigma (asserted: every
# 08c run is solved at `sigma_locked`, Dani 2026-10-03; the alignment row must also equal its own L-curve at that
# sigma and fit the current 06 field); the bootstrap 2-sigma comes from 08c's `uncertainty_<run>_<medium>.csv`
# ([6] there); the bootstrap's reference row must match this model, surface % included.

# %%
## [14]
to_mesh_from_field = pyproj.Transformer.from_crs(int(fig2_in["epsg"]), CRS, always_xy=True)
cross_m = float(fig2_in["cross_m"])                         # half length of the profile lines (m)
far_m = 1500.0                                              # the far field the profiles are zeroed on (07a [14])
d_line = np.arange(-cross_m, cross_m + 1.0, 10.0)
d_line = d_line[np.abs(d_line) > 5.0]                       # 07a [14]: the trace itself is a discontinuity
frac = (d_line + cross_m) / (2 * cross_m)
prof_names = []
for profile_name in fig2_in["names"]:
    prof_names.append(str(profile_name))
pts = []
for name in prof_names:
    pxw, pyw, pxe, pye = fig2_in[f"prof_fp_{name}_ends"]
    px, py = to_mesh_from_field.transform(pxw + frac * (pxe - pxw), pyw + frac * (pye - pyw))
    pts.append(np.c_[px, py])
pts.append(np.c_[points["x"], points["y"]])
pts = np.vstack(pts)
crossing_keys = ("crossing_km", "crossing_mm")              # 07a's individual crossings, drawn by 07c
fig2_products = {}
fwd_validation = {}
range_models = {}                                           # label -> the models of its Mw ranges ([15c] recomputes each one's Mw)
for label, (medium, cd) in FIG2_PRODUCTS.items():
    mdl = models[(medium, cd)]
    slip_all = tents.vertex_slip(mdl["slip"] * 1e-3)        # m, on every fault vertex
    fwd_name = f"fwd_fig2_{label}"
    folders = medium_paths(medium)
    slip_npz = os.path.join(folders["spatialdb"], f"fig2_slip_{label}.npz")
    slip_db = os.path.relpath(os.path.join(folders["spatialdb"], f"fig2_slip_{label}.spatialdb"), WORK)
    pts_file = os.path.relpath(os.path.join(paths_inputs["points"], f"fig2_points_{label}.txt"), WORK)
    if not os.path.exists(os.path.join(folders["pylith_output"], f"{fwd_name}-points.h5")) and run_pylith:
        np.savez(slip_npz, xyz=fault_xyz, slip_rl_m=slip_all, sign_rl=float(gnpz[medium]["sign_rl"]))
        model_utils.run_logged(f"{in_pylith} python3 {build_dir}/make_vertex_slip_db.py {os.path.relpath(slip_npz, WORK)} {slip_db}", WORK)
        with open(os.path.join(WORK, pts_file), "w") as fh:
            fh.write(f"# 07a [14] profile points ({', '.join(prof_names)}: {d_line.size} each, +/- {cross_m:.0f} m every 10 m) "
                     f"+ the {points['lon'].size} quadtree samples; 08a_3d_model {time.strftime('%Y-%m-%d %H:%M')}\n# name x y z (m) in {CRS}\n")
            for k, (x, y) in enumerate(pts):
                fh.write(f"p{k:05d} {x:.3f} {y:.3f} 0.0\n")
    cached(os.path.join(folders["pylith_output"], f"{fwd_name}-points.h5"),
           f"bash {pylith_dir}/run_fwd.sh {fwd_name} {media[medium]} {mesh_rel} 4 fwd_kin_simpledb.cfg "
           f"--problem.solution_observers.points.reader.filename={pts_file} "
           f"--problem.interfaces.fault.eq_ruptures.rupture.db_auxiliary_field.iohandler.filename={slip_db}")
    displacement_3d_mm = model_utils.read_forward_points(os.path.join(folders["pylith_output"], f"{fwd_name}-points.h5"), pts)
    f_xyz, _, slip_fwd = model_utils.read_forward_slip(os.path.join(folders["pylith_output"], f"{fwd_name}-fault.h5"),
                                                       gnpz[medium]["sign_rl"])
    vertex_dist, vertex_match = cKDTree(f_xyz).query(fault_xyz)
    assert vertex_dist.max() < 1e-6, f"{fwd_name}: fault vertices differ from the GF fault"
    slip_check = float(np.abs(slip_fwd[vertex_match] - slip_all * 1e3).max())
    assert slip_check < 1e-4, f"{fwd_name} was made from another slip model (max |imposed - inverted| {slip_check:.2e} mm): re-run it"
    u_samples = displacement_3d_mm[-points["lon"].size:]
    d_val = u_samples.dot(look) - mdl["model"]
    val = dict(rms_diff_mm=float(np.sqrt(np.mean(d_val ** 2))), max_abs_diff_mm=float(np.abs(d_val).max()),
               rms_model_mm=float(np.sqrt(np.mean(mdl["model"] ** 2))), slip_check_mm=slip_check)
    assert val["rms_diff_mm"] < 0.05, f"{fwd_name}: forward run and G x slip disagree at the samples ({val['rms_diff_mm']:.3f} mm rms)"
    fwd_validation[label] = val
    print(f"PASS {fwd_name}: imposed slip = inverted slip (max |diff| {slip_check:.1e} mm); forward at the samples vs G x slip "
          f"rms {val['rms_diff_mm']:.4f} mm (max {val['max_abs_diff_mm']:.3f}; model rms {val['rms_model_mm']:.3f} mm)")
    profiles = {}
    first_point = 0
    for name in prof_names:
        u_line = displacement_3d_mm[first_point:first_point + d_line.size]
        first_point += d_line.size
        for component, vec in (("fp", look), ("up", np.array([0.0, 0.0, 1.0]))):
            vals = u_line.dot(vec)
            profiles[f"prof_{name}_{component}_d"] = d_line
            profiles[f"prof_{name}_{component}_mm"] = vals - np.median(vals[np.abs(d_line) >= far_m])   # 07a [14]: far field zeroed
        step = profiles[f"prof_{name}_fp_mm"]
        print(f"  {name}-{name}' model fault-parallel: west {step[d_line < 0].min():+.2f} mm, east {step[d_line > 0].max():+.2f} mm")

    # the alignment runs and the bootstrap of 08c
    alignment_rows = {}                                       # e50 / w50 -> the SFCVM row of 08c's alignment runs (this Cd)
    if medium == "sfcvm":
        for off in ("e50", "w50"):
            f_off = os.path.join(dir_08c, f"results_comparison_{off}{CD_TAG[cd]}.csv")
            assert os.path.exists(f_off), f"missing alignment-sensitivity product {f_off} (08c)"
            row_off = pd.read_csv(f_off).set_index("model").loc[f"PyLith sfcvm_{off}"]
            assert float(row_off.sigma) == sigma_locked, f"{f_off}: solved at sigma {row_off.sigma:g}, not the locked {sigma_locked:g}"
            assert str(row_off.field_fingerprint) == field_fingerprint, f"{f_off} fits another 06 field: re-run 08c"
            lc_off = pd.read_csv(os.path.join(dir_08c, f"lcurve_sfcvm_{off}_{off}{CD_TAG[cd]}.csv"))
            mw_lc_off = float(lc_off.loc[np.isclose(lc_off.sigma, sigma_locked), "Mw_30"].iloc[0])
            assert abs(mw_lc_off - float(row_off.Mw_30gpa)) < 1e-9, f"{f_off}: Mw30 differs from its own L-curve at sigma {sigma_locked:g}"
            alignment_rows[off] = row_off
    unc_file = os.path.join(dir_08c, f"uncertainty_{run_tag(cd)}_{medium}.csv")
    assert os.path.exists(unc_file), f"missing bootstrap product {unc_file} (08c)"
    unc = pd.read_csv(unc_file)
    boot_reference = unc[unc.kind == "reference"].iloc[0]
    boot_2sd = unc[unc.kind == "boot_2sd"].iloc[0]
    assert abs(float(boot_reference.Mw_30gpa) - mdl["Mw_30gpa"]) < 1e-6, f"{unc_file} is not the bootstrap of this model"
    assert abs(float(boot_reference.surface_pct) - mdl["surface_pct"]) <= 1e-6 * mdl["surface_pct"], \
        f"{unc_file}: its surface % is not this model's (made on another denominator or field?)"

    # Mw30 range: the selected sigma and its L-curve grid neighbours (this run), plus the +/- 50 m SFCVM alignment
    # runs where they exist for this Cd (08c's table; SFCVM medium only)
    lcurve_sorted = families[(medium, cd)][0].sort_values("sigma").reset_index(drop=True)
    i_sel = int(np.nonzero(np.isclose(lcurve_sorted.sigma.values, mdl["sigma"]))[0][0])
    sigma_neighbours = lcurve_sorted.iloc[max(i_sel - 1, 0):i_sel + 2]
    range_vals = list(sigma_neighbours.Mw_30.values)
    neighbour_sigmas = []
    for sigma_value in sigma_neighbours.sigma:
        neighbour_sigmas.append(f"{sigma_value:g}")
    basis = [f"sigma {', '.join(neighbour_sigmas)} (L-curve grid neighbours)"]
    for off, row_off in alignment_rows.items():
        range_vals.append(float(row_off.Mw_30gpa))
        basis.append(f"SFSVM boundary 50 m {off[0].upper()} at sigma {row_off.sigma:g} ({row_off.Mw_30gpa:.2f})")
    mw_range = [float(min(range_vals)), float(max(range_vals))]
    # the same range at the SFCVM rigidity (Dani 2026-10-01): sigma neighbours from 08c's bootstrap table (its
    # sigma_neighbour rows), alignment runs from the same tables; no new inversion
    sfcvm_vals = [mdl["Mw_sfcvm"]] + list(unc[unc.kind == "sigma_neighbour"].Mw_sfcvm.values)
    for off, row_off in alignment_rows.items():
        sfcvm_vals.append(float(row_off.Mw_sfcvm))
    mw_sfcvm_range = [float(min(sfcvm_vals)), float(max(sfcvm_vals))]
    range_models[label] = dict(lcurve_rows=sigma_neighbours[["sigma", "Mw_30"]].reset_index(drop=True),
                               bootstrap_neighbour_rows=unc[unc.kind == "sigma_neighbour"][["sigma", "Mw_30gpa", "Mw_sfcvm"]].reset_index(drop=True),
                               alignment_rows=dict(alignment_rows))
    assert len(sfcvm_vals) >= 3 and mw_sfcvm_range[0] <= mdl["Mw_sfcvm"] + 1e-9 <= mw_sfcvm_range[1] + 2e-9
    assert mw_range[0] <= mdl["Mw_30gpa"] + 1e-9 <= mw_range[1] + 2e-9
    top_v = np.abs(imp_xyz[:, 2]) < 1e-6
    order_top = np.argsort(imp_km[top_v])
    out = dict(sample_lon=samples.lon, sample_lat=samples.lat, block_lon=block_lon, block_lat=block_lat,
               data_mm=mdl["data"], model_mm=mdl["model"], residual_mm=mdl["residual"],
               nodes_kmz=np.c_[imp_km, imp_depth_km], faces=sub_faces, slip_vertex_mm=mdl["slip"], s0_km_pp=0.0,
               surface_km=imp_km[top_v][order_top], surface_model_mm=mdl["slip"][top_v][order_top],
               mw_30gpa=mdl["Mw_30gpa"], mw_sfcvm=mdl["Mw_sfcvm"], mu_effective_gpa=mdl["mu_effective_gpa"],
               rms_mm=mdl["rms_mm"], surf_ratio_pct=mdl["surface_pct"], sigma=mdl["sigma"], strike_deg=strike_bearing,
               potency_m3=mdl["potency_m3"], rms_west300_mm=mdl["rms_west300_mm"], rms_east300_mm=mdl["rms_east300_mm"],
               chi2=mdl["chi2"], forward_validation=json.dumps(val), model_tag=f"pylith_{label}",
               mw_30gpa_2sig=float(boot_2sd.Mw_30gpa), mw_sfcvm_2sig=float(boot_2sd.Mw_sfcvm),
               surf_ratio_pct_2sig=float(boot_2sd.surface_pct),
               potency_2sig_m3=float(boot_2sd.potency_m3), mw_30gpa_range=np.array(mw_range), mw_30gpa_range_basis="; ".join(basis),
               mw_sfcvm_range=np.array(mw_sfcvm_range),
               provenance=f"08a_3d_model {time.strftime('%Y-%m-%d %H:%M')}: run {run_tag(cd)}, medium {medium}, PyLith GF {run_names[medium]}, "
                          f"Cd {cd}, surface constraint {'ON' if use_surface else 'OFF'}, forward run {fwd_name}; bootstrap 2-sigma from 08c {os.path.basename(unc_file)}; "
                          f"06 field {field_fingerprint}")
    for key in crossing_keys:
        out[key] = row07[key]
    if medium == "sfcvm":
        out["sfcvm_version"] = sfcvm_label
    else:
        out["sfcvm_version"] = "v26-0 aligned (rescale of the homogeneous model)"
    out.update(profiles)
    # like for like with the crossings (Dani 2026-10-16): the model's cross-fault step at the samples, 30-150 m either side
    # within +/- 200 m along strike, as 03/04 measure the crossings (model_utils.crossing_steps_like), compared crossing by
    # crossing over the SSE section (no further binning: Dani 2026-10-16 late). The model's top-edge slip is not what the
    # crossings measure (no pixel lies within 30 m of the trace).
    like_km = np.asarray(out["crossing_km"], float)
    out["like_km"] = like_km
    out["like_model_mm"] = model_utils.crossing_steps_like(mdl["model"], signed_d_km, setup["sample_km_pp"], like_km)
    out["keep"] = setup["keep"]
    out["near_trace_cut_km"] = float(model_near_trace_cut_km)
    misfit = out["like_model_mm"] - np.asarray(out["crossing_mm"], float)
    in_section = (like_km >= creep_lo_km) & (like_km <= creep_hi_km) & np.isfinite(misfit)
    at_lake = (like_km >= lake_km[0]) & (like_km <= lake_km[1])
    assert np.any(in_section & at_lake), "no crossing along Lake Elizabeth with a model value"
    out["like_rms_mm"] = float(np.sqrt(np.mean(misfit[in_section & ~at_lake] ** 2)))
    out["like_rms_all_mm"] = float(np.sqrt(np.mean(misfit[in_section] ** 2)))
    out["like_lake_mean_mm"] = float(np.mean(misfit[in_section & at_lake]))
    out["like_n_crossings"] = int(in_section.sum())
    print(f"{label}: model step minus crossings, {out['like_n_crossings']} crossings in km {creep_lo_km:.1f}-{creep_hi_km:.1f}: "
          f"rms {out['like_rms_mm']:.2f} mm outside Lake Elizabeth ({out['like_rms_all_mm']:.2f} mm with it), "
          f"mean {out['like_lake_mean_mm']:+.2f} mm along the lake (km {lake_km[0]:.1f}-{lake_km[1]:.1f})")
    np.savez(os.path.join(out_dir, f"fig2_model_row_pylith_{label}.npz"), **out)
    fig2_products[label] = out
    print(f"wrote fig2_model_row_pylith_{label}.npz: Mw30 {out['mw_30gpa']:.2f} +/- {out['mw_30gpa_2sig']:.2f} (2 sigma), "
          f"range {mw_range[0]:.2f}-{mw_range[1]:.2f} ({out['mw_30gpa_range_basis']})")

# %% [markdown]
# ## [15] Products and provenance
#
# `inversion_products<tag>.npz` per covariance treatment (slip, model, residual at the samples,
# both media; the settings, the validation numbers and the results table as json).

# %%
## [15]
settings_common = dict(mesh=mesh_preset, dip="90 (vertical)", elem_order=int(elem_order), use_surface=use_surface,
                       points=f"{points_set} ({samples.vel.size} samples, 07a quadtree swath 4 km)", run_names=run_names,
                       n_impulses=int(n_imp), mesh_file=os.path.basename(mesh_file), media_cfg=media)
for cd in CD_TAG:
    products = dict(impulse_vertex_xyz=imp_xyz, impulse_km=imp_km, sub_faces=sub_faces)
    for medium in media:
        products[f"slip_{medium}_mm"] = models[(medium, cd)]["slip"]
        products[f"model_{medium}_mm"] = models[(medium, cd)]["model"]
        products[f"residual_{medium}_mm"] = models[(medium, cd)]["residual"]
    products.update(sample_lon=samples.lon, sample_lat=samples.lat,
                    data_mm=models[("sfcvm", cd)]["data"], signed_d_km=signed_d_km, block_lon=block_lon, block_lat=block_lat,
                    trace_lon=fault07_trace.lon, trace_lat=fault07_trace.lat,
                    settings=json.dumps(dict(settings_common, cd_mode=cd, cd_floor=None)),
                    validation_a=pd.Series(val_a).to_json(), validation_b=pd.Series(val_b).to_json(), validation_c=json.dumps(val_c),
                    validation_d=json.dumps(val_d), validation_fwd=json.dumps(fwd_validation),
                    results=results[cd].to_json(), provenance=f"08a_3d_model {time.strftime('%Y-%m-%d %H:%M')}; runs {run_names}; "
                                                              f"mesh {os.path.basename(mesh_file)}; {grid['provenance']}")
    np.savez(os.path.join(out_dir, f"inversion_products{CD_TAG[cd]}.npz"), **products)
    print(f"wrote inversion_products{CD_TAG[cd]}.npz")
for product_name, seconds in timings.items():
    print(f"timing: {product_name} {seconds:.0f} s")

# %% [markdown]
# ## [15b] Stress drop from PyLith: the shear-traction change on the fault
#
# The inverted slip of the selected model (and of its homogeneous twin) is run forward once more in PyLith
# (`FaultCohesiveKin`, the same mesh, medium and slip database as [14]) with the fault observer writing
# `traction_change` as well as `slip`. The stress drop is the slip-weighted mean of the shear-traction drop
# over the fault, sum(dtau s A) / sum(s A) over the fault vertices (A = one third of the adjacent triangle
# areas; `model_utils.vertex_areas_m2`, `slip_weighted_stress_drop_pa`): the standard definition for a distributed
# slip model (Noda et al., 2013). PyLith's fault vectors
# are (opening, left-lateral, reverse); right-lateral slip is minus the left-lateral component, and a drop in
# right-lateral shear stress is a positive left-lateral traction change (CHECKED below). No single rigidity
# or depth scale is assumed: the traction change is carried by the SFCVM cells beside the fault. Also
# reported: the mean over the patch where slip exceeds half its peak, its 95th percentile, and the mean over
# that patch in the top 100 m (the soft near-surface rock next to the free surface carries almost no stress change).

# %%
## [15b]
stress_drop = {}
for label, medium in (("nuggetcd", "sfcvm"), ("nuggetcd_homog", "homog")):
    fwd_t = f"fwd_traction_{label}"
    folders = medium_paths(medium)
    slip_db = os.path.relpath(os.path.join(folders["spatialdb"], f"fig2_slip_{label}.spatialdb"), WORK)
    pts_file = os.path.relpath(os.path.join(paths_inputs["points"], f"fig2_points_{label}.txt"), WORK)
    cached(os.path.join(folders["pylith_output"], f"{fwd_t}-fault.h5"),
           f"bash {pylith_dir}/run_fwd.sh {fwd_t} {media[medium]} {mesh_rel} 4 fwd_kin_simpledb.cfg "
           f"--problem.solution_observers.points.reader.filename={pts_file} "
           f"--problem.interfaces.fault.eq_ruptures.rupture.db_auxiliary_field.iohandler.filename={slip_db} "
           f"'--problem.interfaces.fault.observers.observer.data_fields=[slip,traction_change]'")
    with h5py.File(os.path.join(folders["pylith_output"], f"{fwd_t}-fault.h5"), "r") as fault_h5:
        f_xyz = fault_h5["geometry/vertices"][:]
        f_cells = np.asarray(fault_h5["viz/topology/cells"][:], int)
        slip_vectors = fault_h5["vertex_fields/slip"][-1]                  # (opening, left-lateral, reverse), m
        dtau = fault_h5["vertex_fields/traction_change"][-1][:, 1]
    assert np.abs(slip_vectors[:, [0, 2]]).max() < 1e-9, "opening or reverse slip in a strike-slip forward run"
    slip_rl = -slip_vectors[:, 1]
    assert abs(slip_rl.max() * 1e3 - fig2_products[label]["slip_vertex_mm"].max()) < 1e-3, f"{fwd_t}: slip is not the inverted slip"
    v_area = model_utils.vertex_areas_m2(f_xyz, f_cells)
    patch = slip_rl > 0.5 * slip_rl.max()
    patch_top100 = patch & (-f_xyz[:, 2] < 100.0)
    stress_drop[medium] = dict(weighted_kpa=model_utils.slip_weighted_stress_drop_pa(dtau, slip_rl, v_area) / 1e3,
                               patch_kpa=float(dtau[patch].mean() / 1e3), top100_kpa=float(dtau[patch_top100].mean() / 1e3),
                               p95_kpa=float(np.percentile(dtau[patch], 95) / 1e3))
    drop = stress_drop[medium]
    print(f"{medium}: slip-weighted stress drop {drop['weighted_kpa']:.0f} kPa; patch (slip > 50 % of peak) mean {drop['patch_kpa']:.0f} kPa, "
          f"95th pct {drop['p95_kpa']:.0f} kPa; patch in the top 100 m {drop['top100_kpa']:.0f} kPa")

# %% [markdown]
# ## [15c] Mw and the rigidity: the options side by side (Dani 2026-10-02)
#
# Potency P = sum over fault triangles of area x mean tent slip is what the inversion constrains; the
# moment M0 = sum(mu_t x P_t) needs a rigidity, and Mw = 2/3 (log10 M0 - 9.1) (Hanks & Kanamori 1979, M0 in
# N m). One table, the selected model's slip throughout, every row a different rigidity choice:
# - 30 GPa: the crustal reference value (the convention for comparing with other slow-slip events).
# - Per triangle, from the PyLith cells either side of the fault (`model_utils.mu_beside_fault`, [6]):
#   arithmetic mean (mu_W + mu_E) / 2 (the rigidity of `Mw_sfcvm`, registered as `08a_mw_sfcvm`); harmonic mean
#   2 mu_W mu_E / (mu_W + mu_E), the effective rigidity of a fault on a material interface (Vavrycuk 2013,
#   GJI 194, 395-400, doi:10.1093/gji/ggt084; `model_utils.harmonic_mean_rigidity`); west side only; east side only.
# - mu_eff = M0 / P for each per-triangle row: the single rigidity that gives the same Mw.
# - A sweep of single uniform rigidities, Mw(mu) = Mw30 + 2/3 log10(mu / 30 GPa), for quoting a range.
# Potency is the same in every row, so only the rigidity differs. Each row also carries the model range of
# [14]: the selected model, its L-curve sigma neighbours (on the `sfcvm` medium, re-inverted here with
# `invert_tents`) and the +/- 50 m alignment models (08c's runs on their own media `sfcvm_e50` /
# `sfcvm_w50`: slip from 08c's `inversion_products_{e50,w50}<tag>.npz`, rigidity from their own PyLith
# `-elastic_info.h5` and grid `sfcvm_v26-0_aligned_{e50,w50}_grid.npz`). The 30 GPa, arithmetic and harmonic
# rows take their range PER MODEL (Dani 2026-10-06): each range model's per-triangle potency x the rigidity
# of ITS medium, min/max over the models (saved per model as `mw_range_models.csv`). CHECK: each model's
# Mw30 and arithmetic Mw equal the values [14] built its ranges from (L-curve, 08c's bootstrap table, the
# alignment tables) to < 1e-6, so the arithmetic range is [14]'s `mw_sfcvm_range` (same models, same
# rigidity definition). The west, east and uniform rows are SHIFTED: the 30 GPa range moved by the row's
# rigidity, Mw_lo/hi = Mw + (range - Mw30). Column `range_kind` says which. Printed and saved
# (`mw_rigidity_options.csv`, a short `key` per row: 30gpa, arithmetic, harmonic, west, east, uniform_<mu>).
# The paper quotes 30 GPa and the HARMONIC row (Dani 2026-10-05): registered as `08a_mw_sfcvm_harmonic`
# (selected model), `_lo`/`_hi` (per model), its effective rigidity `08a_mu_sfcvm_harmonic_gpa`, and drawn
# on Fig 2 panel K by 07c from this csv.

# %%
## [15c]
slip_sel = models[SELECTED]["slip"]
pot_tri = tents.potency_per_triangle_m3(slip_sel)                          # m3 per triangle (tent slip exact for linear tents)
P_sel = float(pot_tri.sum())
assert abs(P_sel - tents.potency_m3(slip_sel)) < 1e-6 * P_sel, "per-triangle potency does not sum to the potency"
mu_options = {"arithmetic mean of the two sides": 0.5 * (mu_west + mu_east),
              "harmonic mean of the two sides (Vavrycuk 2013)": model_utils.harmonic_mean_rigidity(mu_west, mu_east),
              "west side only": mu_west, "east side only": mu_east}
assert np.allclose(mu_options["arithmetic mean of the two sides"], mu_fault["sfcvm"]), "arithmetic row is not Mw_sfcvm's rigidity"
mu_option_key = {"arithmetic mean of the two sides": "arithmetic", "harmonic mean of the two sides (Vavrycuk 2013)": "harmonic",
                 "west side only": "west", "east side only": "east"}     # the short row keys 07c and the registry use
rows_mu = [dict(key="30gpa", option="reference 30 GPa (uniform)", mu_eff_gpa=mu_ref / 1e9, M0_Nm=mu_ref * P_sel,
                Mw=float(moment_magnitude(mu_ref * P_sel)))]
for name, mu_t in mu_options.items():
    m0 = float(np.sum(mu_t * pot_tri))
    rows_mu.append(dict(key=mu_option_key[name], option=f"SFSVM per triangle, {name}", mu_eff_gpa=m0 / P_sel / 1e9, M0_Nm=m0,
                        Mw=float(moment_magnitude(m0))))
for mu_gpa in (0.5, 1.0, 2.0, 3.0, 5.0, 10.0):
    m0 = mu_gpa * 1e9 * P_sel
    rows_mu.append(dict(key=f"uniform_{mu_gpa:g}", option=f"uniform {mu_gpa:g} GPa", mu_eff_gpa=mu_gpa, M0_Nm=m0,
                        Mw=float(moment_magnitude(m0))))
mw_rigidity = pd.DataFrame(rows_mu)
assert mw_rigidity.key.is_unique, "mw_rigidity_options keys repeat"
mw_rigidity["potency_m3"] = P_sel
selected_label = None
for label, medium_cd in FIG2_PRODUCTS.items():
    if medium_cd == SELECTED:
        selected_label = label
assert selected_label is not None, "the selected model has no Fig 2 product"
range_30 = fig2_products[selected_label]["mw_30gpa_range"]             # [14]: from the models themselves, 30 GPa
range_sfcvm = fig2_products[selected_label]["mw_sfcvm_range"]          # [14]: from the models themselves, arithmetic rigidity
mw30_selected = float(mw_rigidity.Mw.iloc[0])
assert abs(mw30_selected - models[SELECTED]["Mw_30gpa"]) < 1e-9, "first row is not the selected model's Mw30"


def range_model_mw(model_name, medium_name, slip_mm, rigidity_west, rigidity_east):
    """Potency, Mw30 and the SFSVM Mw of one range model: the per-triangle potency of its slip x the arithmetic or
    the harmonic mean of the PyLith cells either side of the fault in its own medium."""
    potency_tri = tents.potency_per_triangle_m3(slip_mm)
    return dict(model=model_name, medium=medium_name, potency_m3=float(potency_tri.sum()),
                Mw_30gpa=float(moment_magnitude(mu_ref * potency_tri.sum())),
                Mw_arithmetic=float(moment_magnitude(np.sum(0.5 * (rigidity_west + rigidity_east) * potency_tri))),
                Mw_harmonic=float(moment_magnitude(np.sum(model_utils.harmonic_mean_rigidity(rigidity_west, rigidity_east) * potency_tri))))


# the models of [14]'s range, each with its own slip and the rigidity of its own medium
medium_sel, cd_sel = SELECTED
members = range_models[selected_label]
lcurve_rows = members["lcurve_rows"]
neighbour_rows = members["bootstrap_neighbour_rows"]
lcurve_neighbour_sigmas = sorted(float(v) for v in lcurve_rows.sigma if not np.isclose(v, models[SELECTED]["sigma"]))
assert len(lcurve_neighbour_sigmas) == len(neighbour_rows) and np.allclose(lcurve_neighbour_sigmas, sorted(neighbour_rows.sigma)), \
    "[14]'s 30 GPa range (L-curve) and SFSVM range (08c's bootstrap table) are not on the same sigma neighbours"
range_rows = [dict(**range_model_mw(f"selected (sigma {models[SELECTED]['sigma']:g})", medium_sel, slip_sel, mu_west, mu_east),
                   Mw_30gpa_14=models[SELECTED]["Mw_30gpa"], Mw_arithmetic_14=models[SELECTED]["Mw_sfcvm"])]
for _, neighbour in neighbour_rows.iterrows():
    fault_neighbour, _, _ = invert_tents(medium_sel, float(neighbour.sigma), cd_sel)
    mw30_lcurve = float(lcurve_rows.loc[np.isclose(lcurve_rows.sigma, neighbour.sigma), "Mw_30"].iloc[0])
    range_rows.append(dict(**range_model_mw(f"sigma {neighbour.sigma:g} (L-curve grid neighbour)", medium_sel,
                                            fault_neighbour.slip[:, 0], mu_west, mu_east),
                           Mw_30gpa_14=mw30_lcurve, Mw_arithmetic_14=float(neighbour.Mw_sfcvm)))
for off, row_off in members["alignment_rows"].items():
    medium_off = f"sfcvm_{off}"                                          # 08c's medium name
    run_off = f"gf_{mesh_preset}_{medium_off}"                          # 08c's GF run of that medium (same mesh)
    grid_off = os.path.join(paths_sfcvm["spatialdb"], f"sfcvm_v26-0_aligned_{off}_grid.npz")
    assert os.path.exists(grid_off), f"missing {grid_off} (the grid of the {medium_off} medium)"
    g_off = np.load(os.path.join(paths_sfcvm["gf"], f"G_{run_off}.npz"))
    assert (np.array_equal(g_off["fault_xyz"], fault_xyz) and np.array_equal(g_off["fault_cells"], fault_cells)
            and np.array_equal(g_off["vertex_index"], vertex_index)), f"G_{run_off}: another fault mesh or impulse order"
    _, mu_west_off, mu_east_off, rel_off = model_utils.mu_beside_fault(
        os.path.join(paths_sfcvm["pylith_output"], f"{run_off}-elastic_info.h5"), grid_off, tents, medium_off)
    assert rel_off < 0.05, f"{run_off}: PyLith's cell rigidity differs from its grid by more than 5 % (median)"
    products_off = np.load(os.path.join(dir_08c, f"inversion_products_{off}{CD_TAG[cd_sel]}.npz"))
    assert np.allclose(products_off["impulse_vertex_xyz"], imp_xyz), f"inversion_products_{off}: another impulse order"
    assert json.loads(str(products_off["settings"]))["field_fingerprint"] == field_fingerprint, \
        f"inversion_products_{off}{CD_TAG[cd_sel]}.npz fits another 06 field: re-run 08c"
    range_rows.append(dict(**range_model_mw(f"SFSVM boundary 50 m {off[0].upper()} (sigma {row_off.sigma:g})", medium_off,
                                            products_off[f"slip_{medium_off}_mm"], mu_west_off, mu_east_off),
                           Mw_30gpa_14=float(row_off.Mw_30gpa), Mw_arithmetic_14=float(row_off.Mw_sfcvm)))
range_table = pd.DataFrame(range_rows)
range_table["provenance"] = (f"08a_3d_model {time.strftime('%Y-%m-%d %H:%M')}: the Mw range models of fig2_model_row_pylith_{selected_label} "
                             f"([14]); sigma neighbours re-inverted here, alignment slip from 08c inversion_products_{{e50,w50}}{CD_TAG[cd_sel]}.npz; "
                             f"06 field {field_fingerprint}")
# CHECK: the same models as [14] (Mw30 and arithmetic Mw model by model), so the arithmetic range IS [14]'s
worst_30 = float(np.max(np.abs(range_table.Mw_30gpa - range_table.Mw_30gpa_14)))
worst_arithmetic = float(np.max(np.abs(range_table.Mw_arithmetic - range_table.Mw_arithmetic_14)))
assert worst_30 < 1e-6 and worst_arithmetic < 1e-6, \
    f"a range model's Mw differs from the value [14] used (Mw30 {worst_30:.1e}, arithmetic {worst_arithmetic:.1e}): not the same models"
assert abs(range_table.Mw_30gpa.min() - range_30[0]) < 1e-6 and abs(range_table.Mw_30gpa.max() - range_30[1]) < 1e-6, \
    "the per-model 30 GPa range is not [14]'s"
assert abs(range_table.Mw_arithmetic.min() - range_sfcvm[0]) < 1e-6 and abs(range_table.Mw_arithmetic.max() - range_sfcvm[1]) < 1e-6, \
    "the per-model arithmetic range is not [14]'s mw_sfcvm_range"
print(f"PASS: {len(range_table)} range models reproduce [14]'s Mw30 and SFSVM (arithmetic) values "
      f"(max |diff| {worst_30:.1e} / {worst_arithmetic:.1e})")
print(range_table[["model", "medium", "potency_m3", "Mw_30gpa", "Mw_arithmetic", "Mw_harmonic"]].to_string(
    index=False, float_format="{:.3f}".format))
range_table.to_csv(os.path.join(out_dir, "mw_range_models.csv"), index=False)
# the range on every row: per model for 30 GPa, arithmetic and harmonic; the 30 GPa range shifted for the rest
mw_rigidity["Mw_lo"] = mw_rigidity.Mw + (range_30[0] - mw30_selected)
mw_rigidity["Mw_hi"] = mw_rigidity.Mw + (range_30[1] - mw30_selected)
mw_rigidity["range_kind"] = "shifted: the 30 GPa range of [14] moved by this rigidity"
for key, column in (("30gpa", "Mw_30gpa"), ("arithmetic", "Mw_arithmetic"), ("harmonic", "Mw_harmonic")):
    is_key = mw_rigidity.key == key
    assert is_key.sum() == 1, f"no single {key} row"
    assert abs(float(mw_rigidity.loc[is_key, "Mw"].iloc[0]) - range_table[column].iloc[0]) < 1e-9, \
        f"the {key} row is not the selected range model's {column}"
    mw_rigidity.loc[is_key, "Mw_lo"] = float(range_table[column].min())
    mw_rigidity.loc[is_key, "Mw_hi"] = float(range_table[column].max())
    mw_rigidity.loc[is_key, "range_kind"] = "per model: min/max over the range models of [14] (mw_range_models.csv)"
arithmetic_row = mw_rigidity[mw_rigidity.key == "arithmetic"].iloc[0]
harmonic_row = mw_rigidity[mw_rigidity.key == "harmonic"].iloc[0]          # the paper's SFSVM Mw (Dani 2026-10-05)
assert abs(arithmetic_row.Mw - models[SELECTED]["Mw_sfcvm"]) < 1e-9, "arithmetic row differs from the selected model's Mw_sfcvm"
# where the moment sits: potency-weighted rigidity (harmonic) by depth band
depth_t = -tri_cen[:, 2]
for lo_m, hi_m in ((0, 100), (100, 300), (300, 1000), (1000, 1e6)):
    in_band = (depth_t >= lo_m) & (depth_t < hi_m)
    if pot_tri[in_band].sum() > 0:
        mu_harmonic_band = mu_options["harmonic mean of the two sides (Vavrycuk 2013)"][in_band]
        print(f"  depth {lo_m:>4.0f}-{min(hi_m, 99999):>5.0f} m: {100 * pot_tri[in_band].sum() / P_sel:5.1f} % of the potency, "
              f"potency-weighted mu: harmonic {np.sum(mu_harmonic_band * pot_tri[in_band]) / pot_tri[in_band].sum() / 1e9:.2f}, "
              f"arithmetic {np.sum(mu_fault['sfcvm'][in_band] * pot_tri[in_band]) / pot_tri[in_band].sum() / 1e9:.2f} GPa")
mw_rigidity.to_csv(os.path.join(out_dir, "mw_rigidity_options.csv"), index=False)
print(f"\nselected model ({run_tag(SELECTED[1])} / {SELECTED[0]}), potency {P_sel:.0f} m3:")
print(mw_rigidity[["option", "mu_eff_gpa", "Mw", "Mw_lo", "Mw_hi"]].to_string(index=False, float_format="{:.2f}".format))
per_triangle = mw_rigidity[mw_rigidity.option.str.startswith("SFSVM per triangle")]
print(f"SFSVM rigidity, all four ways of taking it: Mw {per_triangle.Mw.min():.2f} to {per_triangle.Mw.max():.2f} "
      f"(with the model range {per_triangle.Mw_lo.min():.2f} to {per_triangle.Mw_hi.max():.2f}); "
      f"30 GPa: {mw30_selected:.2f} ({range_30[0]:.2f} to {range_30[1]:.2f})")
print(f"the paper's SFSVM Mw (harmonic, Fig 2 K): {harmonic_row.Mw:.2f} ({harmonic_row.Mw_lo:.2f} to {harmonic_row.Mw_hi:.2f}), "
      f"mu_eff {harmonic_row.mu_eff_gpa:.2f} GPa")

# %% [markdown]
# ## [15d] Where the 3D medium improves the fit (Dani 2026-10-02)
#
# The selected model against its homogeneous twin (same inversion, nugget Cd, both at sigma 12):
# rms of the residual within 200 m of the trace (near field) and beyond it (and beyond 1 km), the drop in
# percent, and the share of the total misfit reduction (sum of squared residuals, uniform minus 3D) that
# comes from the near-field samples.

# %%
## [15d]
residual_3d = models[SELECTED]["residual"]
residual_uniform = models[("homog", SELECTED[1])]["residual"]
abs_d = np.where(setup["keep"], np.abs(signed_d_km), np.nan)      # samples inside the near-trace cut take no part (2026-10-16)
near_far = {}
for name, mask in (("near200", abs_d <= 0.2), ("beyond200", abs_d > 0.2), ("beyond1km", abs_d > 1.0)):
    rms_uniform = float(np.sqrt(np.mean(residual_uniform[mask] ** 2)))
    rms_3d = float(np.sqrt(np.mean(residual_3d[mask] ** 2)))
    near_far[name] = dict(n=int(mask.sum()), rms_uniform_mm=rms_uniform, rms_3d_mm=rms_3d, drop_pct=100 * (rms_uniform - rms_3d) / rms_uniform)
    print(f"{name:10s} n {mask.sum():4d}: uniform {rms_uniform:.2f} mm, 3D {rms_3d:.2f} mm, {near_far[name]['drop_pct']:.0f} % lower")
misfit_reduction = residual_uniform ** 2 - residual_3d ** 2
assert misfit_reduction.sum() > 0, "the 3D model does not reduce the total misfit"
near_far["near200_share_of_reduction_pct"] = float(100 * misfit_reduction[abs_d <= 0.2].sum() / misfit_reduction.sum())
print(f"share of the total misfit reduction from samples within 200 m: {near_far['near200_share_of_reduction_pct']:.0f} %")

# %% [markdown]
# ## [15e] Numbers the text quotes: the homogeneous model's deflections at X-X', the SFSVM boundary offset, potency with depth
#
# 1. **Near-fault deflections at X-X' predicted by the homogeneous (uniform, 30 GPa) PyLith model**, defined as 06
#    defines the observed ones: the median of the model's fault-parallel profile 30 to 150 m from the trace
#    (`dist_min`, `dist_max`) minus the median of the same block 1.5 to 2 km away (`far_band_m`), on the X-X' line of
#    [14] (the forward run of the inverted slip, every 10 m). For the selected model (nugget Cd) and the SFSVM model
#    beside it. The range over the sensitivity runs cannot use forward runs (only the Fig-2 models have one): there the
#    same median difference is taken on the model AT THE SAMPLES within 300 m of the X-X' line (as `figS_pylith_profile_XX`;
#    the far band holds only 1-2 quadtree blocks, so this estimator is coarser), for the homogeneous model of: the
#    selected and companion inversions, their L-curve grid neighbours of sigma 12 (re-inverted here), and 08c's runs
#    with dip 80 E, 70 E, 80 W and 70 W (west added 2026-10-25), element order 2 (each with both Cd) and the `shallow` mesh (both Cd since 2026-10-25). The same
#    estimator is applied to the selected run, so the range is like for like. Written to `xx_deflections_homog_runs.csv`.
# 2. **The SFSVM Hayward block boundary relative to the mapped trace**: the offset b(km) that the alignment removes
#    (`boundary_smooth_m` of the aligned SFCVM grid, the 5-point running median that `pylith/build/sfcvm_grid.py`
#    shifts the model by; east +), its range over the SSE section (04's creep length) and over the 90 % band.
# 3. **Potency with depth** of the selected model: the share of the potency above 300 m and above 500 m. The tent
#    slip is linear on each PyLith fault triangle, so each triangle is split into `n_sub` x `n_sub` sub-triangles
#    (barycentric), the slip and depth taken at each sub-triangle's centroid; the share is the sum over those above the
#    depth over the total (the total equals the exact tent potency, asserted).

# %%
## [15e]
# 1) the homogeneous model's near-fault deflections at X-X'
near_band_m = (dist_min * 1e3, dist_max * 1e3)              # 06's xfault_step_m: 30 to 150 m from the trace
far_band_m = (far_m, cross_m)                               # 06's far_m: 1.5 to 2 km (the far field the profiles are zeroed on)


def block_deflections(distance_m, values_mm):
    """(west, east) near-fault deflection: median at near_band_m minus median at far_band_m on the same side (06's
    definition of the observed block amplitudes); each band must hold at least one value (asserted)."""
    out = []
    for side in (-1, 1):
        outward = side * distance_m
        near = (outward >= near_band_m[0]) & (outward < near_band_m[1])
        far = (outward >= far_band_m[0]) & (outward < far_band_m[1])
        assert near.any() and far.any(), f"side {side:+d}: {near.sum()} near and {far.sum()} far values"
        out.append(float(np.median(values_mm[near]) - np.median(values_mm[far])))
    return tuple(out)


forward_deflections = {}                                    # Fig-2 label -> (W, E) on the forward-run profile
for label in FIG2_PRODUCTS:
    forward_deflections[label] = block_deflections(fig2_products[label]["prof_X_fp_d"], fig2_products[label]["prof_X_fp_mm"])
xx_samples = sel & np.isfinite(signed_d_km)                 # the samples within 300 m of the X-X' line ([13])


def sample_deflections(model_mm):
    """(west, east) near-fault deflection of a model at the samples within 300 m of the X-X' line."""
    return block_deflections(signed_d_km[xx_samples] * 1e3, model_mm[xx_samples])


deflection_rows = []
for cd in CD_TAG:                                           # the selected / companion inversions and their sigma neighbours
    lcurve_sorted = families[("homog", cd)][0].sort_values("sigma").reset_index(drop=True)
    i_sel = int(np.nonzero(np.isclose(lcurve_sorted.sigma.values, sigma_locked))[0][0])
    for sigma_value in lcurve_sorted.sigma.values[max(i_sel - 1, 0):i_sel + 2]:
        if np.isclose(sigma_value, sigma_locked):
            model_mm = models[("homog", cd)]["model"]
            setting = "selected" if cd == SELECTED[1] else "companion"
        else:
            _, insar_neighbour, _ = invert_tents("homog", float(sigma_value), cd)
            model_mm = insar_neighbour.synth.copy()
            setting = f"sigma {sigma_value:g}"
        west, east = sample_deflections(model_mm)
        deflection_rows.append(dict(run=run_tag(cd), setting=setting, cd=cd, sigma=float(sigma_value), W_mm=west, E_mm=east))
for tag, setting in (("_dip80e", "dip 80 E"), ("_dip70e", "dip 70 E"), ("_dip80w", "dip 80 W"), ("_dip70w", "dip 70 W"),
                     ("_order2", "order 2"), ("_shallow", "mesh shallow")):
    for cd in CD_TAG:
        products_file = os.path.join(dir_08c, f"inversion_products{tag}{CD_TAG[cd]}.npz")
        if not os.path.exists(products_file):
            assert tag == "_shallow" and cd == "nugget", f"missing 08c product {products_file}"
            continue                                        # the shallow mesh was run with the full Cd only (08c's table)
        products_08c = np.load(products_file)
        assert np.allclose(products_08c["sample_lon"], samples.lon) and np.allclose(products_08c["sample_lat"], samples.lat), \
            f"{products_file}: another sample set"
        settings_08c = json.loads(str(products_08c["settings"]))
        assert settings_08c["field_fingerprint"] == field_fingerprint, f"{products_file} fits another 06 field: re-run 08c"
        west, east = block_deflections(products_08c["signed_d_km"][xx_samples] * 1e3, products_08c["model_homog_mm"][xx_samples])
        deflection_rows.append(dict(run=f"default{tag}{CD_TAG[cd]}", setting=setting, cd=cd, sigma=sigma_locked, W_mm=west, E_mm=east))
deflections = pd.DataFrame(deflection_rows)
deflections.to_csv(os.path.join(out_dir, "xx_deflections_homog_runs.csv"), index=False, float_format="%.3f")
# the homogeneous dip scan of 08c (2026-10-24, Dani): the same estimator at every dip 50 W-50 E, both Cd; its own table and
# keys (08a_hom_xx_dipscan_*), NOT in the sensitivity ranges above (vertical, 80 E and 70 E are those rows again)
dipscan_rows = []
for dip in ("50w", "60w", "70w", "80w", "90", "80e", "70e", "60e", "50e"):
    for cd in CD_TAG:
        tag = "" if dip == "90" else f"_dip{dip}"
        products_file = os.path.join(dir_08c, f"inversion_products{tag}{CD_TAG[cd]}.npz")
        assert os.path.exists(products_file), f"missing 08c dip-scan product {products_file}: run 08c"
        products_08c = np.load(products_file)
        assert np.allclose(products_08c["sample_lon"], samples.lon) and np.allclose(products_08c["sample_lat"], samples.lat), \
            f"{products_file}: another sample set"
        assert json.loads(str(products_08c["settings"]))["field_fingerprint"] == field_fingerprint, f"{products_file} fits another 06 field: re-run 08c"
        west, east = block_deflections(products_08c["signed_d_km"][xx_samples] * 1e3, products_08c["model_homog_mm"][xx_samples])
        dipscan_rows.append(dict(dip=dip, run=f"default{tag}{CD_TAG[cd]}", cd=cd, sigma=sigma_locked, W_mm=west, E_mm=east))
deflections_dipscan = pd.DataFrame(dipscan_rows)
deflections_dipscan.to_csv(os.path.join(out_dir, "xx_deflections_homog_dipscan.csv"), index=False, float_format="%.3f")
dipscan_ranges = {}
for family_name, family in (("nugget", deflections_dipscan[deflections_dipscan.cd == "nugget"]), ("all", deflections_dipscan)):
    dipscan_ranges[family_name] = dict(W_min=float(family.W_mm.min()), W_max=float(family.W_mm.max()),
                                       E_min=float(family.E_mm.min()), E_max=float(family.E_mm.max()))
print("homogeneous dip scan, X-X' deflections at the samples:")
print(deflections_dipscan.round(2).to_string(index=False))
selected_hom_samples = deflections[deflections.setting == "selected"].iloc[0]
deflection_ranges = {}
for family_name, family in (("nugget", deflections[deflections.cd == "nugget"]), ("all", deflections)):
    deflection_ranges[family_name] = dict(W_min=float(family.W_mm.min()), W_max=float(family.W_mm.max()),
                                          E_min=float(family.E_mm.min()), E_max=float(family.E_mm.max()), n=len(family))
print(deflections.round(2).to_string(index=False))
deflection_lines = [f"X-X' near-fault deflection (median {near_band_m[0]:.0f}-{near_band_m[1]:.0f} m minus the same block "
                    f"{far_band_m[0]:.0f}-{far_band_m[1]:.0f} m):"]
for label, (west, east) in forward_deflections.items():
    deflection_lines.append(f"   forward-run profile, {label}: W {west:+.2f} mm, E {east:+.2f} mm")
deflection_lines.append(f"   homogeneous model at the samples within 300 m of the line ({xx_samples.sum()} samples): selected W "
                        f"{selected_hom_samples.W_mm:+.2f}, E {selected_hom_samples.E_mm:+.2f} mm; over the runs (nugget Cd, "
                        f"{deflection_ranges['nugget']['n']}): W {deflection_ranges['nugget']['W_min']:+.2f} to {deflection_ranges['nugget']['W_max']:+.2f}, "
                        f"E {deflection_ranges['nugget']['E_min']:+.2f} to {deflection_ranges['nugget']['E_max']:+.2f} mm; all "
                        f"{deflection_ranges['all']['n']}: W {deflection_ranges['all']['W_min']:+.2f} to {deflection_ranges['all']['W_max']:+.2f}, "
                        f"E {deflection_ranges['all']['E_min']:+.2f} to {deflection_ranges['all']['E_max']:+.2f} mm")

# 2) the SFSVM block boundary relative to the mapped trace, over the SSE section and the 90 % band
band90 = section_lengths.loc["tulepond_band90"]
boundary_km = grid["trace_km"]
boundary_offset = {}
for span_name, (span_lo, span_hi) in (("section", sse_km), ("band90", (float(band90.km_lo), float(band90.km_hi)))):
    in_span = (boundary_km >= span_lo) & (boundary_km <= span_hi)
    assert in_span.sum() >= 5, f"only {in_span.sum()} boundary points in the {span_name}"
    smooth = grid["boundary_smooth_m"][in_span]
    raw = grid["boundary_raw_m"][in_span]
    boundary_offset[span_name] = dict(lo_km=span_lo, hi_km=span_hi, min_m=float(smooth.min()), max_m=float(smooth.max()),
                                      median_m=float(np.median(smooth)), raw_min_m=float(raw.min()), raw_max_m=float(raw.max()),
                                      east_fraction=float(np.mean(smooth > 0)))
    deflection_lines.append(f"SFSVM boundary offset from the trace (east +), {span_name} km {span_lo:.1f}-{span_hi:.1f}: smoothed (applied) "
                            f"{smooth.min():+.0f} to {smooth.max():+.0f} m (median {np.median(smooth):+.0f}; raw {raw.min():+.0f} to {raw.max():+.0f}); "
                            f"east of the trace at {100 * np.mean(smooth > 0):.0f} % of the {in_span.sum()} trace points")

# 3) potency with depth of the selected model
n_sub = 20                                                  # sub-triangles per edge
depth_thresholds_m = (300.0, 500.0)
bary = []
for i in range(n_sub):
    for j in range(n_sub - i):
        bary.append(((i + 1 / 3) / n_sub, (j + 1 / 3) / n_sub))           # centroid of an upward sub-triangle
        if i + j < n_sub - 1:
            bary.append(((i + 2 / 3) / n_sub, (j + 2 / 3) / n_sub))       # ... and of the downward one beside it
bary = np.array(bary)
bary = np.c_[bary, 1 - bary.sum(axis=1)]                     # (n_sub^2, 3) weights on the three vertices
assert len(bary) == n_sub ** 2 and (bary > 0).all()
slip_vertices_mm = tents.vertex_slip(models[SELECTED]["slip"])[fault_cells]      # (n_tri, 3)
depth_vertices_m = -tri_xyz[:, :, 2]                                             # (n_tri, 3), depth + down
sub_slip = slip_vertices_mm @ bary.T                                             # (n_tri, n_sub^2)
sub_depth = depth_vertices_m @ bary.T
sub_potency = sub_slip * (tri_area[:, None] / bary.shape[0]) * 1e-3              # m^3
assert abs(sub_potency.sum() - models[SELECTED]["potency_m3"]) <= 1e-6 * models[SELECTED]["potency_m3"], "sub-triangle potency != tent potency"
potency_above = {}
for depth_m in depth_thresholds_m:
    potency_above[depth_m] = float(100 * sub_potency[sub_depth < depth_m].sum() / sub_potency.sum())
deflection_lines.append("selected model potency above " + ", ".join(f"{d:.0f} m: {v:.0f} %" for d, v in potency_above.items()))
print("\n".join(deflection_lines))

# %% [markdown]
# ## [16] Registry -- the 3D model (the paper's slip model)
#
# `08a_*` = the selected model (aligned SFCVM, nugget Cd); `08a_hom_*` its homogeneous twin;
# `08a_full_*` / `08a_full_hom_*` the companion pair under 07a's full covariance. The 2-sigma and
# range keys come from the Fig-2 product (bootstrap of 08c, L-curve neighbours, alignment +/- 50 m).
# The sidecar also carries the surface % denominator, the stress drop [15b] and the near/far fit [15d].

# %%
## [16]
input_files = []
for medium in media:
    input_files.append(os.path.join(medium_paths(medium)["gf"], f"G_{run_names[medium]}.npz"))
input_files += [mesh_fault_npz, grid_file, points_file + ".npz", os.path.join(field_dir, "fp_maps.npz"),
                os.path.join(field_dir, "fig2_inputs.npz"), os.path.join(dir_07a, "model_data_07a.npz"), row07_file,
                os.path.join(events_dir, "section_lengths.csv")]
for medium, cd in RUNS:                                     # 08c's tables this notebook reads or checks against
    input_files.append(os.path.join(dir_08c, f"lcurve_{medium}{CD_TAG[cd]}.csv"))
for cd in CD_TAG:
    input_files.append(os.path.join(dir_08c, f"results_comparison{CD_TAG[cd]}.csv"))
    for off in ("e50", "w50"):
        input_files.append(os.path.join(dir_08c, f"results_comparison_{off}{CD_TAG[cd]}.csv"))
for label, (medium, cd) in FIG2_PRODUCTS.items():
    input_files.append(os.path.join(dir_08c, f"uncertainty_{run_tag(cd)}_{medium}.csv"))
for off in members["alignment_rows"]:                       # [15c]: the alignment range models, slip and medium
    input_files += [os.path.join(dir_08c, f"inversion_products_{off}{CD_TAG[cd_sel]}.npz"),
                    os.path.join(paths_sfcvm["gf"], f"G_gf_{mesh_preset}_sfcvm_{off}.npz"),
                    os.path.join(paths_sfcvm["spatialdb"], f"sfcvm_v26-0_aligned_{off}_grid.npz")]
REG.inputs(*input_files)
forward_vs_g = []
for label, val in fwd_validation.items():
    forward_vs_g.append(f"{label} {val['rms_diff_mm']:.4f} mm")
lcurve_lines = []
for medium, cd in RUNS:
    lcurve_lines.append(f"{medium} {cd}: corner {families[(medium, cd)][1]:.1f} -> sigma {families[(medium, cd)][2]:g}")
lines = [f"selected model: {run_tag(SELECTED[1])} / {SELECTED[0]} (vertical fault, {mesh_preset} mesh, order {elem_order}, aligned SFSVM v26-0, "
         f"nugget Cd, surface constraint {'ON' if use_surface else 'OFF'}); companion {run_tag(COMPANION[1])} / {COMPANION[0]} (07a full Cd)",
         f"PyLith runs: {run_names}; {n_imp} impulses on {len(fault_xyz)} fault vertices; {samples.vel.size} samples",
         f"validation (a) PyLith homog vs CSI Meade: rel rms {100 * val_a['rel_rms']:.2f} % (agreeing sides {100 * val_a['rel_rms_okside']:.2f} %, "
         f"far field {val_a['rms_diff_far_mm']:.3f} mm); (b) vs 2D screw: {100 * val_b['rel_rms']:.2f} %; "
         f"(c) superposition homog: {val_c['rms_diff_mm']:.2e} mm rms; (d) fine - default mesh: {val_d['fine_minus_default_rms_mm']:.4f} mm rms; "
         f"forward-vs-G at the samples: " + ", ".join(forward_vs_g),
         f"fault-adjacent mu, top 100 m: west {mu_top100['west_gpa']:.3f}, east {mu_top100['east_gpa']:.3f} GPa; side mismatches {side_mismatch.sum()}",
         f"surface % denominator: median of {n_section_crossings} measured crossings in the SSE section {section_crossing_median_mm:.3f} mm",
         f"06 field fp_maps.npz fingerprint {field_fingerprint} (asserted on 07a's model data and on every 08c table read here)",
         "L-curves: " + "; ".join(lcurve_lines)]
for (medium, cd) in RUNS:
    mdl = models[(medium, cd)]
    model_numbers = []
    for k in ("sigma", "rms_mm", "rms_west300_mm", "rms_east300_mm", "chi2", "surface_pct", "potency_m3",
              "Mw_30gpa", "Mw_sfcvm", "mu_effective_gpa", "peak_slip_mm", "peak_depth_m", "surf_rms_mm"):
        model_numbers.append(f"{k} {mdl[k]:.4g}")
    lines.append(f"{KEY[(medium, cd)]} ({run_tag(cd)} / {medium}): " + ", ".join(model_numbers))
for label, out in fig2_products.items():
    lines.append(f"fig2_model_row_pylith_{label}: Mw30 2-sigma {out['mw_30gpa_2sig']:.3f}, Mw_sfcvm 2-sigma {out['mw_sfcvm_2sig']:.3f}, "
                 f"surface 2-sigma {out['surf_ratio_pct_2sig']:.1f} %, Mw30 range {out['mw_30gpa_range'][0]:.3f}-{out['mw_30gpa_range'][1]:.3f} "
                 f"({out['mw_30gpa_range_basis']})")
for medium, drop in stress_drop.items():
    lines.append(f"stress drop ({medium}): slip-weighted {drop['weighted_kpa']:.1f} kPa, patch mean {drop['patch_kpa']:.1f} kPa, "
                 f"95th pct {drop['p95_kpa']:.1f} kPa, patch in the top 100 m {drop['top100_kpa']:.1f} kPa")
lines.append(f"Mw with the SFSVM rigidity, harmonic mean of the two sides per triangle (the paper's, Fig 2 K): {harmonic_row.Mw:.3f} "
             f"(range {harmonic_row.Mw_lo:.3f}-{harmonic_row.Mw_hi:.3f}: per model, the range models of [14] each with its own medium), "
             f"mu_eff {harmonic_row.mu_eff_gpa:.3f} GPa; arithmetic {arithmetic_row.Mw:.3f} ({arithmetic_row.mu_eff_gpa:.3f} GPa)")
for _, range_row in range_table.iterrows():
    lines.append(f"Mw range model {range_row.model} ({range_row.medium}): potency {range_row.potency_m3:.0f} m3, Mw30 {range_row.Mw_30gpa:.3f}, "
                 f"SFSVM arithmetic {range_row.Mw_arithmetic:.3f}, harmonic {range_row.Mw_harmonic:.3f}")
for name in ("near200", "beyond200", "beyond1km"):
    lines.append(f"fit {name}: n {near_far[name]['n']}, uniform {near_far[name]['rms_uniform_mm']:.3f} mm, 3D {near_far[name]['rms_3d_mm']:.3f} mm, "
                 f"{near_far[name]['drop_pct']:.1f} % lower")
lines.append(f"share of the total misfit reduction from samples within 200 m: {near_far['near200_share_of_reduction_pct']:.1f} %")
lines += deflection_lines
sidecar = REG.sidecar("08a_3d_model", lines)
for (medium, cd) in RUNS:
    key = KEY[(medium, cd)]
    mdl = models[(medium, cd)]
    cell = "selected" if (medium, cd) == SELECTED else "comparison"
    REG.register(f"{key}_sigma", mdl["sigma"], "mm", 0, where="M8", cell=cell, sidecar=sidecar)
    REG.register(f"{key}_sigma_corner", mdl["corner"], "mm", 1, where="M8", cell=cell, sidecar=sidecar)
    REG.register(f"{key}_rms_mm", mdl["rms_mm"], "mm", 2, where="M8", cell=cell, sidecar=sidecar)
    REG.register(f"{key}_rms_west300_mm", mdl["rms_west300_mm"], "mm", 2, where="M8", cell=cell, sidecar=sidecar)
    REG.register(f"{key}_rms_east300_mm", mdl["rms_east300_mm"], "mm", 2, where="M8", cell=cell, sidecar=sidecar)
    REG.register(f"{key}_chi2", mdl["chi2"], "", 0, where="M8", cell=cell, sidecar=sidecar)
    REG.register(f"{key}_surface_pct", mdl["surface_pct"], "%", 0, where="M8", cell=cell, sidecar=sidecar)
    REG.register(f"{key}_potency_m3", mdl["potency_m3"], "m3", 0, where="M8", cell=cell, sidecar=sidecar)
    REG.register(f"{key}_mw_30gpa", mdl["Mw_30gpa"], "Mw", 2, where="M8", cell=cell, sidecar=sidecar)
    REG.register(f"{key}_mw_sfcvm", mdl["Mw_sfcvm"], "Mw", 2, where="M8", cell=cell, sidecar=sidecar)
    REG.register(f"{key}_mu_effective_gpa", mdl["mu_effective_gpa"], "GPa", 2, where="M8", cell=cell, sidecar=sidecar)
    REG.register(f"{key}_peak_slip_mm", mdl["peak_slip_mm"], "mm", 1, where="M8", cell=cell, sidecar=sidecar)
    REG.register(f"{key}_peak_depth_m", mdl["peak_depth_m"], "m", 0, where="M8", cell=cell, sidecar=sidecar)
for label, out in fig2_products.items():
    key = KEY[FIG2_PRODUCTS[label]]
    REG.register(f"{key}_mw_30gpa_2sig", out["mw_30gpa_2sig"], "Mw", 2, where="M8", cell="fig2 product", sidecar=sidecar)
    REG.register(f"{key}_mw_sfcvm_2sig", out["mw_sfcvm_2sig"], "Mw", 2, where="M8", cell="fig2 product", sidecar=sidecar)
    REG.register(f"{key}_surface_pct_2sig", out["surf_ratio_pct_2sig"], "%", 0, where="M8", cell="fig2 product", sidecar=sidecar)
    REG.register(f"{key}_mw_30gpa_range_lo", float(out["mw_30gpa_range"][0]), "Mw", 2, where="M8", cell="fig2 product", sidecar=sidecar)
    REG.register(f"{key}_mw_30gpa_range_hi", float(out["mw_30gpa_range"][1]), "Mw", 2, where="M8", cell="fig2 product", sidecar=sidecar)
    REG.register(f"{key}_mw_sfcvm_range_lo", float(out["mw_sfcvm_range"][0]), "Mw", 2, where="M8", cell="fig2 product", sidecar=sidecar)
    REG.register(f"{key}_mw_sfcvm_range_hi", float(out["mw_sfcvm_range"][1]), "Mw", 2, where="M8", cell="fig2 product", sidecar=sidecar)
REG.register("08a_mw_sfcvm_harmonic", float(harmonic_row.Mw), "Mw", 2, where="M8, Fig 2K", cell="rigidity", sidecar=sidecar)
REG.register("08a_mw_sfcvm_harmonic_lo", float(harmonic_row.Mw_lo), "Mw", 2, where="M8", cell="rigidity", sidecar=sidecar)
REG.register("08a_mw_sfcvm_harmonic_hi", float(harmonic_row.Mw_hi), "Mw", 2, where="M8", cell="rigidity", sidecar=sidecar)
REG.register("08a_mu_sfcvm_harmonic_gpa", float(harmonic_row.mu_eff_gpa), "GPa", 2, where="M8, Fig 2K", cell="rigidity", sidecar=sidecar)
REG.register("08a_n_impulses", int(n_imp), "count", 0, where="M8", cell="G", sidecar=sidecar)
REG.register("08a_n_samples", int(samples.vel.size), "count", 0, where="M8", cell="G", sidecar=sidecar)
REG.register("08a_mu_west_top100_gpa", mu_top100["west_gpa"], "GPa", 2, where="M8", cell="materials", sidecar=sidecar)
REG.register("08a_mu_east_top100_gpa", mu_top100["east_gpa"], "GPa", 2, where="M8", cell="materials", sidecar=sidecar)
for name in ("near200", "beyond200", "beyond1km"):                                            # [15d] 3D vs uniform fit
    REG.register(f"08a_rms_{name}_uniform_mm", near_far[name]["rms_uniform_mm"], "mm", 2, where="M8", cell="near/far fit", sidecar=sidecar)
    REG.register(f"08a_rms_{name}_3d_mm", near_far[name]["rms_3d_mm"], "mm", 2, where="M8", cell="near/far fit", sidecar=sidecar)
    REG.register(f"08a_rms_{name}_drop_pct", near_far[name]["drop_pct"], "%", 0, where="M8", cell="near/far fit", sidecar=sidecar)
REG.register("08a_rms_near200_share_of_reduction_pct", near_far["near200_share_of_reduction_pct"], "%", 0, where="M8", cell="near/far fit", sidecar=sidecar)
REG.register("08a_val_csi_rel_rms_pct", 100 * val_a["rel_rms_okside"], "%", 1, where="M8", cell="validation", sidecar=sidecar)
REG.register("08a_val_csi_far_rms_mm", val_a["rms_diff_far_mm"], "mm", 3, where="M8", cell="validation", sidecar=sidecar)
REG.register("08a_val_screw_rel_rms_pct", 100 * val_b["rel_rms"], "%", 1, where="M8", cell="validation", sidecar=sidecar)
REG.register("08a_val_mesh_fine_minus_default_mm", val_d["fine_minus_default_rms_mm"], "mm", 3, where="M8", cell="validation", sidecar=sidecar)
REG.register("08a_val_fwd_vs_G_mm", fwd_validation["nuggetcd"]["rms_diff_mm"], "mm", 4, where="M8", cell="validation", sidecar=sidecar)
REG.register("08a_stress_drop_pylith_kpa", stress_drop["sfcvm"]["weighted_kpa"], "kPa", 0, where="P7, M8", cell="stress drop", sidecar=sidecar)
REG.register("08a_stress_drop_pylith_patch_kpa", stress_drop["sfcvm"]["patch_kpa"], "kPa", 0, where="M8", cell="stress drop", sidecar=sidecar)
REG.register("08a_stress_drop_pylith_top100m_kpa", stress_drop["sfcvm"]["top100_kpa"], "kPa", 0, where="P7, M8", cell="stress drop", sidecar=sidecar)
REG.register("08a_stress_drop_pylith_p95_kpa", stress_drop["sfcvm"]["p95_kpa"], "kPa", 0, where="M8", cell="stress drop", sidecar=sidecar)
REG.register("08a_hom_stress_drop_pylith_kpa", stress_drop["homog"]["weighted_kpa"], "kPa", 0, where="M8", cell="stress drop", sidecar=sidecar)
# [15e]: the X-X' deflections, the SFSVM boundary offset and the potency with depth (2026-10-06)
for label, key in (("nuggetcd_homog", "08a_hom"), ("nuggetcd", "08a")):
    REG.register(f"{key}_xx_fp_W_mm", forward_deflections[label][0], "mm", 2, where="P7", cell="deflections", sidecar=sidecar)
    REG.register(f"{key}_xx_fp_E_mm", forward_deflections[label][1], "mm", 2, where="P7", cell="deflections", sidecar=sidecar)
REG.register("08a_hom_xx_samples_W_mm", selected_hom_samples.W_mm, "mm", 2, where="P7", cell="deflections", sidecar=sidecar)
REG.register("08a_hom_xx_samples_E_mm", selected_hom_samples.E_mm, "mm", 2, where="P7", cell="deflections", sidecar=sidecar)
for family_name, family_range in deflection_ranges.items():
    for quantity in ("W_min", "W_max", "E_min", "E_max"):
        REG.register(f"08a_hom_xx_sens_{family_name}_{quantity}_mm", family_range[quantity], "mm", 2, where="P7", cell="deflections", sidecar=sidecar)
for family_name, family_range in dipscan_ranges.items():          # the homogeneous dip scan 50 W-50 E (2026-10-24; not in the text yet)
    for quantity in ("W_min", "W_max", "E_min", "E_max"):
        REG.register(f"08a_hom_xx_dipscan_{family_name}_{quantity}_mm", family_range[quantity], "mm", 2, where="dip scan", cell="deflections", sidecar=sidecar)
for span_name, offset in boundary_offset.items():
    REG.register(f"08a_sfcvm_boundary_{span_name}_min_m", offset["min_m"], "m", 0, where="P7", cell="materials", sidecar=sidecar)
    REG.register(f"08a_sfcvm_boundary_{span_name}_max_m", offset["max_m"], "m", 0, where="P7", cell="materials", sidecar=sidecar)
for depth_m, share in potency_above.items():
    REG.register(f"08a_potency_above_{int(depth_m)}m_pct", share, "%", 0, where="P7", cell="potency with depth", sidecar=sidecar)
# like for like with the crossings (2026-10-16), from the Fig 2 products ([14]): selected model and its homogeneous twin
for label, prefix in (("nuggetcd", "08a"), ("nuggetcd_homog", "08a_hom")):
    product = fig2_products[label]
    for k, unit, places in (("like_rms_mm", "mm", 2), ("like_rms_all_mm", "mm", 2), ("like_lake_mean_mm", "mm", 2),
                            ("like_n_crossings", "count", 0)):
        REG.register(f"{prefix}_{k}", product[k], unit, places, where="P7, M8", cell="like for like", sidecar=sidecar)
REG.register("08a_near_trace_cut_m", 1e3 * model_near_trace_cut_km, "m", 0, where="M8", cell="settings", sidecar=sidecar)
REG.register("08a_n_samples_fitted", int(np.sum(setup["keep"])), "count", 0, where="M8", cell="settings", sidecar=sidecar)
REG.render()
print("08a_3d_model complete")
