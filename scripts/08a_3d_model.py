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
# PREPUB: promoted from drafts/pylith/pylith_sfcvm_gf.py, the 2026-09-29/30 draft.
# @author: daniellelindsay
# Run in opera_disp_feb13, after 06 and 07a. About 15 min (about 60 tent inversions of ~10 s each);
# no PyLith run: the Green's functions, mesh, materials and forward runs are read from modeling_SFCVM/.
# ```
#
# **What this notebook does.** The slip model of the paper: the five-event mean fault-parallel
# field of notebook 06 inverted on **3D finite-element Green's functions (PyLith 5.0.2)** for the
# vertical fault under the mapped creeping trace, in the **USGS SFCVM v26.0** elastic structure
# aligned on the trace (the sediment-over-bedrock contrast that the homogeneous half-space of 07a
# cannot represent). The data, the quadtree samples, the noise model and the CSI inversion
# machinery are 07a's, exec'd unchanged from `scripts/07a_uniform_model.py` cells [1]-[6], so the
# two models are compared like for like. The surface constraint is NOT 07a's: every measured fault
# crossing inside the mesh is one datum, tied to the slip at its own position on the fault's top
# edge (`pylith_helpers/surface_crossings.py`, Dani 2026-10-02; cell [10]). The 3D model supersedes
# the uniform one (07a) in the paper; 07a stays as the baseline it is measured against. After the
# models: the PyLith stress drop of the selected model [15b], its Mw under each rigidity choice
# [15c], and where the 3D medium improves the fit [15d].
#
# The **SELECTED model** (Dani, 2026-09-30): run tag `default_nuggetcd`, aligned SFCVM medium,
# vertical fault, `default` mesh (100 m fault cells in the Tule Pond patch), order-1 elements,
# nugget-only diagonal data covariance, measured-crossing surface constraint ON, sigma 12 (the
# grid value at or above the L-curve corner, 07a [8] rule; locked, asserted in [11]). Its
# **companion** is the same inversion with 07a's full correlated covariance (tag `default`). Both
# media (homogeneous 30 GPa and aligned SFCVM) are inverted in each, so every table carries four
# models plus 07a's CSI baseline. The sensitivity runs (mesh, Cd floors, no surface constraint,
# wider swath, dip, order 2, alignment +/- 50 m, coherence 0.7) are collected in `08b_3d_model_tests`.
#
# | Step | Tool (validated library) | Glue written here |
# |---|---|---|
# | mesh | gmsh via PyLith `gmsh_utils.GenerateMesh` (`pylith_helpers/generate_gmsh_vertical.py`) | size fields |
# | materials | SFCVM read with h5py (`pylith_helpers/sfcvm_grid.py`); spatialdata `SimpleGridAscii` writer | alignment shift, vs floor, Poisson cap |
# | Green's functions | PyLith `GreensFns` + `FaultCohesiveImpulses` + `OutputSolnPoints` (`scripts/pylith_cfg/`) | impulse-region db |
# | G assembly | reading pattern of PyLith `examples/subduction-3d/invert_slip.py` (`pylith_helpers/build_G.py`) | vertex/point matching, projection, sign |
# | inversion | CSI `TriangularTents`, `setGFs`, `buildCmXY`, `transformation`, `multisourcesolve` NNLS, `surfaceslip` (as 07a) | surface constraint on vertices (`surface_crossings.py`) |
# | validation | PyLith `FaultCohesiveKin` forward runs; CSI Meade dislocations; 2D screw dislocation | comparisons |
#
# **The PyLith side is not re-run here.** The Green's-function runs (33 min each, 8 MPI ranks,
# PyLith 5.0.2 bundle) were made once with the parameter files in `scripts/pylith_cfg/` and the
# helpers in `scripts/pylith_helpers/`; their products (mesh, spatialdb, points, G matrices, the
# forward runs) are read from `TulePond_filepaths.model_sfcvm_dir` (`modeling_SFCVM/`, part of
# the Zenodo deposit). A missing product stops the notebook with the command that makes it;
# `TULEPOND_RUN_PYLITH=1` lets the notebook shell out to PyLith instead.
#
# **Inputs**: `modeling_SFCVM/{mesh,spatialdb,points,gf,pylith_output}`; 07a's data and machinery
# (06 `fp_maps.npz` and `fig2_inputs.npz`, 04 `event_fraction_by_crossing.csv` = the measured
# crossings of the surface constraint, 04 `section_lengths.csv`, the trace);
# `07b_uniform_model_tests/fig2_model_row.npz` (07a's baseline on the same samples); and, from
# `modeling_SFCVM/gf/`, the products of the earlier PyLith driver run: `lcurve_<medium><tag>.csv`
# and `results_comparison<tag>.csv` (compared in [11] and [12]), `results_comparison_{e50,w50}<tag>.csv`
# (the +/- 50 m alignment runs of the Mw range, [14]), their `inversion_products_{e50,w50}<tag>.npz`,
# `G_gf_default_sfcvm_{e50,w50}.npz` and PyLith `-elastic_info.h5` (slip and rigidity of those range
# models, [15c]) and `uncertainty_<run>_<medium>.csv` (08b's bootstrap, [14]).
# **Outputs** (`outputs/08a_3d_model/`): `lcurve_<medium><tag>.csv`, `results_comparison<tag>.csv`,
# `inversion_products<tag>.npz`, the Fig-2 products `fig2_model_row_pylith_{nuggetcd,full,nuggetcd_homog}.npz`
# (read by `07c_fig2`), `mw_rigidity_options.csv` and `mw_range_models.csv` ([15c]), the stats sidecar
# `08a_3d_model_stats.txt`.
# **Figures**: `figS_pylith_materials_sfcvm_v26` and `qc08a_materials_boundary_offset` ([3]),
# `qc08a_mesh_fault` ([2]), `figS_pylith_validation` ([8]), `qc08a_validation_mesh` ([9]),
# `figS_pylith_lcurves` ([11]), `figS_pylith_fit_maps_<tag>`, `figS_pylith_slip`, `figS_pylith_profile_XX` ([13]).
# **Registry keys** `08a_*` (selected), `08a_hom_*`, `08a_full_*`, `08a_full_hom_*`.
#
# **What to check.** Every CHECK prints PASS: PyLith's cell rigidity within 5 % of the aligned grid
# (median) [6]; no sample more than 50 m from the mesh fault on the other side of 07a's trace [7];
# PyLith against CSI beyond 300 m of the trace under 2 % of the signal and on agreeing sides under
# 10 % rms [8]; superposition to 1e-4 mm and fine minus default mesh under 0.2 mm [9]; every model
# at the locked sigma 12 [11]; each forward run's imposed slip equal to the inverted slip and its
# displacement equal to G x slip to 0.05 mm [14].

# %%
## [1]
# ------------------------
# Paths & constants
# ------------------------
import copy
import json
import os
import subprocess
import sys
import time
import warnings

import h5py
import numpy as np
import pandas as pd
import pyproj
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.tri as mtri
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", category=DeprecationWarning)   # library noise only; numerical warnings stay visible
warnings.filterwarnings("ignore", category=FutureWarning)
from TulePond_filepaths import common_paths, repo_dir, paths_out, paths_sfcvm, model_sfcvm_dir
import insar_utils as utils
from insar_utils import mpl_paper_style

# CSI: Classic Slip Inversion (Jolivet et al.; external/csi at 1277bb4)
from csi import TriangularTents, insar, surfaceslip
from csi import transformation as transform
from csi import multisourcesolve as multisrc

notebook = "08a_3d_model"
out_dir = paths_out[notebook]
fig_dir = common_paths["fig_dir"]
helpers_dir = os.path.join(repo_dir, "scripts", "pylith_helpers")
cfg_dir = paths_sfcvm["cfg"]                        # scripts/pylith_cfg: the PyLith parameter files the runs were made with
sys.path.insert(0, helpers_dir)
from surface_crossings import crossing_constraint, surface_pct_denominator_mm   # surface constraint (2026-10-02), surface % denominator (2026-10-05)
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)
WORK = model_sfcvm_dir                              # the PyLith side (mesh/, spatialdb/, points/, gf/, pylith_output/)
for key in ("mesh", "spatialdb", "points", "gf", "pylith_output"):
    assert os.path.isdir(paths_sfcvm[key]), f"missing {paths_sfcvm[key]}: modeling_SFCVM/ holds the PyLith products"
run_pylith = os.environ.get("TULEPOND_RUN_PYLITH", "0") == "1"   # 1: make a missing PyLith product here (33 min per GF run)
opera_py = sys.executable
CRS = "+proj=tmerc +datum=WGS84 +lon_0=-121.968 +lat_0=37.557 +k=1 +units=m +type=crs"   # mesh CRS = the CSI frame of 07a

# the FIXED configuration of the selected model and its companion (Dani 2026-09-30); the
# alternatives (mesh, dip, order, swath, coherence, Cd floors, no surface constraint) live in 08b
mesh_preset = "default"                             # 100 m fault cells in km 56-63 / top 1.5 km, 250 m elsewhere on the fault
elem_order = "1"                                    # linear elements; slip impulses are tents on the fault vertices
points_set = "default"                              # 07a's 2386 quadtree samples (swath 4 km)
use_surface = True                                  # measured-crossing surface constraint ON (surface_crossings.py, [10])
sigma_locked = 12.0                                 # the model sigma, locked for every model (Dani 2026-10-03; asserted in [11])
media = {"homog": "mat_homog.cfg", "sfcvm": "mat_sfcvm.cfg"}     # 30 GPa / SFCVM v26.0 aligned on the trace (scripts/pylith_cfg)
run_names = {m: f"gf_{mesh_preset}_{m}" for m in media}         # PyLith GreensFns runs (pylith_output/<run>-*.h5)
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


def run_shell(cmd):
    """Run a shell command in modeling_SFCVM (PYLITH_WORKDIR / PYLITH_CFG for the run wrappers), echo it, fail loudly."""
    print("$", cmd)
    t0 = time.time()
    env = dict(os.environ, PYLITH_WORKDIR=WORK, PYLITH_CFG=cfg_dir)
    out = subprocess.run(cmd, shell=True, cwd=WORK, env=env, capture_output=True, text=True)
    assert out.returncode == 0, f"command failed ({out.returncode}):\n{out.stdout[-3000:]}\n{out.stderr[-3000:]}"
    print(out.stdout[-1500:])
    return time.time() - t0


def cached(path, recipe):
    """A product of the PyLith side. It must exist in modeling_SFCVM (made once with the cfg in
    scripts/pylith_cfg); with TULEPOND_RUN_PYLITH=1 a missing one is made by `recipe` (run in modeling_SFCVM)."""
    if not os.path.exists(path):
        assert run_pylith, (f"missing PyLith product {path}\n  made once with: {recipe}\n  set TULEPOND_RUN_PYLITH=1 to run it "
                            f"here (needs the PyLith 5.0.2 bundle, see scripts/pylith_helpers/pylith_env.sh)")
        timings[os.path.basename(path)] = run_shell(recipe)
        assert os.path.exists(path), f"{recipe} did not write {path}"
    return path


pyenv = f"bash {helpers_dir}/pylith_env.sh {WORK}"   # "<pyenv> <command>" runs the command in the PyLith env, in modeling_SFCVM
to_mesh = pyproj.Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
REG = utils.Registry(notebook, common_paths, repo_dir)
print(f"selected model: {run_tag(SELECTED[1])} / {SELECTED[0]}; companion {run_tag(COMPANION[1])} / {COMPANION[0]}; "
      f"PyLith products from {WORK} (TULEPOND_RUN_PYLITH={int(run_pylith)})")

# %% [markdown]
# ## [2] Mesh
#
# `pylith_helpers/generate_gmsh_vertical.py` (PyLith `gmsh_utils.GenerateMesh`): the mapped trace
# (every 3rd 100 m point, km 52-67) as a spline, extruded 3 km straight down, fragmented into a
# 100 x 100 x 40 km box. Cell size from three linear-growth fields combined by a gmsh Min field.
# Preset `default`: 100 m on the fault in km 56-63 / top 1.5 km, 250 m elsewhere on the fault,
# 250 m on the surface over the data, bias 1.15. Read from `modeling_SFCVM/mesh/`.

# %%
## [2]
mesh_file = cached(os.path.join(paths_sfcvm["mesh"], f"mesh_vertical_tet_{mesh_preset}.msh"),
                   f"{pyenv} python3 {helpers_dir}/generate_gmsh_vertical.py --write --preset {mesh_preset}")
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
# The 2386 quadtree samples of 07a (`fig2_model_row.npz`) in the mesh CRS (`make_points.py`).
# SFCVM v26.0 on a non-uniform grid (50 m horizontally around the Tule Pond section, 25 m
# vertically to 500 m), aligned so that its Hayward block boundary ({22|23}|7) at the surface lies
# on the mapped trace (a local cross-fault shift b(km), full within 3 km of the trace, tapered to
# zero at 6 km; dip kept), vs floored at 150 m/s, Poisson's ratio capped at 0.40
# (`sfcvm_grid.py`), written as a spatialdata SimpleGridDB by spatialdata's own writer
# (`write_sfcvm_griddb.py`). The unshifted ("as is") grid is read too, for the figure.

# %%
## [3]
points_file = os.path.join(paths_sfcvm["points"], "insar_points")          # .txt (PyLith PointsList) / .npz
cached(points_file + ".npz", f"{opera_py} {helpers_dir}/make_points.py")
assert os.path.exists(points_file + ".txt"), f"missing {points_file}.txt"
points = np.load(points_file + ".npz")
for grid_tag, flag in (("aligned", ""), ("asis", "--no-align")):
    cached(os.path.join(paths_sfcvm["spatialdb"], f"sfcvm_v26-0_{grid_tag}_grid.npz"), f"{opera_py} {helpers_dir}/sfcvm_grid.py {flag}")
cached(os.path.join(paths_sfcvm["spatialdb"], "mat_sfcvm_v26-0_aligned.spatialdb"),
       f"{pyenv} python3 {helpers_dir}/write_sfcvm_griddb.py spatialdb/sfcvm_v26-0_aligned_grid.npz "
       f"spatialdb/mat_sfcvm_v26-0_aligned.spatialdb")
grid_file = os.path.join(paths_sfcvm["spatialdb"], "sfcvm_v26-0_aligned_grid.npz")
grid = np.load(grid_file)
print(str(grid["provenance"]))
print(f"{points['lon'].size} output points; grid {grid['x'].size} x {grid['y'].size} x {grid['z'].size}, "
      f"vs floor {float(grid['vs_floor']):g} m/s, nu cap {float(grid['nu_cap']):g}")
section_lengths = pd.read_csv(os.path.join(paths_out["04_event_detection"], "section_lengths.csv")).set_index("name")
sse_km = (float(section_lengths.loc["tulepond_creep_length", "km_lo"]), float(section_lengths.loc["tulepond_creep_length", "km_hi"]))
import plot_materials
plot_materials.main(paths_sfcvm["spatialdb"], fig_dir, sse_km)

# %% [markdown]
# ## [4] PyLith Green's functions, both media
#
# `GreensFns` with `FaultCohesiveImpulses` (`scripts/pylith_cfg/gf.cfg`): unit left-lateral
# impulses (right-lateral = -left-lateral, sign derived in [5]) on every fault vertex of km 54-65
# over the full 3 km depth (buried edges excluded automatically), the rest of the fault locked;
# Dirichlet zero on the sides and bottom, free top; displacement output at the samples. Slip basis
# order 1 (tent per vertex), as the PyLith examples require for inversion. The runs
# (`gf_default_homog`, `gf_default_sfcvm`; 33 min each on 8 ranks) are read from
# `modeling_SFCVM/pylith_output/`.

# %%
## [4]
impulse_db = "spatialdb/impulses_km54-65_z3000.spatialdb"          # y window = impulse_y_m, depth impulse_depth_m
cached(os.path.join(WORK, impulse_db), f"{pyenv} python3 {helpers_dir}/make_impulse_db.py {impulse_y_m[0]} {impulse_y_m[1]} "
                                       f"{impulse_depth_m} {impulse_db}")
for medium, mat_cfg in media.items():
    name = run_names[medium]
    cached(os.path.join(paths_sfcvm["pylith_output"], f"{name}-points.h5"),
           f"bash {helpers_dir}/run_gf.sh {name} {mat_cfg} mesh/{os.path.basename(mesh_file)} {impulse_db} {n_procs} points/insar_points.txt")
    for suffix in ("-fault.h5", "-elastic_info.h5", "-parameters.json"):
        assert os.path.exists(os.path.join(paths_sfcvm["pylith_output"], name + suffix)), f"missing {name}{suffix}"
    run_parameters = json.load(open(os.path.join(paths_sfcvm["pylith_output"], f"{name}-parameters.json")))
    print(f"{name}: PyLith {run_parameters['version']['pylith']['version']}, {run_parameters['timestamp']}, "
          f"{run_parameters['platform']['machine']}")

# %% [markdown]
# ## [5] G matrices
#
# `pylith_helpers/build_G.py` (pattern of PyLith `invert_slip.py`): impulse -> vertex (unit tent
# asserted), output point -> sample (KD-tree, < 1 mm asserted), 3D displacement projected on 07a's
# fault-parallel look vector, sign derived (right-lateral must move the east side along +look, as
# the data). Both media must share the fault mesh and the impulse order. `G_<run>.npz` is read from
# `modeling_SFCVM/gf/` (numpy only: rebuilt there from the PyLith output when absent).

# %%
## [5]
import build_G
G = {}                                   # medium -> G matrix (samples x impulses), right-lateral positive, m per m
gnpz = {}                                # medium -> the G_<run>.npz product
for medium in media:
    path = os.path.join(paths_sfcvm["gf"], f"G_{run_names[medium]}.npz")
    if not os.path.exists(path):
        build_G.main([run_names[medium]], points_file=points_file + ".npz")
    gnpz[medium] = np.load(path)
    # the cached G is projected on the strike it was built with: rebuild it (from the cached PyLith outputs, no PyLith
    # run) when the current field's strike differs (2026-10-05: the along-trace section moved the strike 146.40 -> 146.51)
    strike_now = float(np.load(os.path.join(paths_out["07b_uniform_model_tests"], "fig2_model_row.npz"))["strike_deg"])   # 07a's strike of the current field (what build_G projects on)
    if abs(float(gnpz[medium]["strike_deg"]) - strike_now) > 1e-6:
        print(f"G_{run_names[medium]}: built at strike {float(gnpz[medium]['strike_deg']):.4f}, field at {strike_now:.4f} -> rebuilding")
        build_G.main([run_names[medium]], points_file=points_file + ".npz")
        gnpz[medium] = np.load(path)
    assert abs(float(gnpz[medium]["strike_deg"]) - strike_now) < 1e-6, f"G_{run_names[medium]} strike differs from the field's"
    G[medium] = gnpz[medium]["G_rl_m_per_m"]
    assert np.array_equal(gnpz[medium]["sample_lon"], points["lon"]), f"G_{run_names[medium]} is not on the output points"
ref_medium = "sfcvm"
for medium in media:
    assert np.array_equal(gnpz[ref_medium]["vertex_index"], gnpz[medium]["vertex_index"]), "impulse order differs between media"
    assert np.array_equal(gnpz[ref_medium]["fault_xyz"], gnpz[medium]["fault_xyz"])
fault_xyz = gnpz[ref_medium]["fault_xyz"]
fault_cells = gnpz[ref_medium]["fault_cells"]
vertex_index = gnpz[ref_medium]["vertex_index"]
n_imp = vertex_index.size
# PyLith fault triangles: centroid, unit normal (east +), area (m^2)
tri_xyz = fault_xyz[fault_cells]
tri_cen = tri_xyz.mean(1)
tri_n = np.cross(tri_xyz[:, 1] - tri_xyz[:, 0], tri_xyz[:, 2] - tri_xyz[:, 0])
tri_area = 0.5 * np.linalg.norm(tri_n, axis=1)
tri_n /= np.linalg.norm(tri_n, axis=1)[:, None]
tri_n[tri_n[:, 0] < 0] *= -1.0                                             # east +
print(f"G: {G[ref_medium].shape[0]} samples x {n_imp} impulses (of {len(fault_xyz)} fault vertices); "
      + ", ".join(f"max |G| {m} {np.abs(G[m]).max():.3f}" for m in media) + " m/m")

# %% [markdown]
# ## [6] The rigidity beside the fault, and CHECK: PyLith read the SFCVM material as intended
#
# The per-cell shear modulus PyLith actually used (`-elastic_info.h5`, basis order 0) against the
# aligned grid interpolated at the cell centroids, for cells in the fine grid box; and mu of the
# cells next to the fault (nearest cell centroid on each side of every fault triangle), used for
# the SFCVM moment (`Mw_sfcvm`) and the rigidity options of [15c].

# %%
## [6]
def shear_modulus_pa(density_kg_m3, vs_m_s):
    """Shear modulus mu = rho vs^2 (Pa)."""
    return density_kg_m3 * vs_m_s ** 2


def mu_beside_fault(medium, grid_npz, run_name=None):
    """PyLith cell mu (elastic_info) vs the grid it was read from; mu of the nearest cell on each side of every
    fault triangle (mean of the two sides returned, then west, then east). run_name: the PyLith run of another
    medium on the same mesh (default run_names[medium]; [15c] reads the +/- 50 m alignment media this way)."""
    if run_name is None:
        run_name = run_names[medium]
    with h5py.File(os.path.join(paths_sfcvm["pylith_output"], f"{run_name}-elastic_info.h5"), "r") as elastic_h5:
        cells = np.asarray(elastic_h5["viz/topology/cells"][:], int)
        cell_xyz = elastic_h5["geometry/vertices"][:][cells].mean(1)
        if "cell_fields" in elastic_h5 and "shear_modulus" in elastic_h5["cell_fields"]:
            cell_mu = elastic_h5["cell_fields/shear_modulus"][0, :, 0]                      # materials basis order 0
        else:
            cell_mu = elastic_h5["vertex_fields/shear_modulus"][0, :, 0][cells].mean(1)     # basis order 1: cell mean of the vertex values
    grid_data = np.load(grid_npz)
    grid_mu_values = shear_modulus_pa(grid_data["density"].astype(float), grid_data["vs"].astype(float))
    grid_mu = RegularGridInterpolator((grid_data["x"], grid_data["y"], grid_data["z"]), grid_mu_values)
    fine = (cell_xyz[:, 0] > -3500) & (cell_xyz[:, 0] < 2500) & (cell_xyz[:, 1] > -4000) & (cell_xyz[:, 1] < 3600) & (cell_xyz[:, 2] > -1500)
    rel = cell_mu[fine] / grid_mu(cell_xyz[fine]) - 1.0
    print(f"{medium}: PyLith cell mu vs grid at centroids ({fine.sum()} cells in the fine box, top 1.5 km): median |rel diff| "
          f"{np.median(np.abs(rel)):.3f}, 90th pct {np.percentile(np.abs(rel), 90):.3f}; PyLith mu range "
          f"{cell_mu.min() / 1e9:.3f}-{cell_mu.max() / 1e9:.1f} GPa")
    assert cell_mu.min() >= 0.99 * float(np.min(grid_mu_values)), "mu below the floor"
    _, near = cKDTree(cell_xyz).query(tri_cen, k=30)
    rel_xyz = cell_xyz[near] - tri_cen[:, None, :]
    offset = np.einsum("tkc,tc->tk", rel_xyz, tri_n)
    assert np.all((offset > 0).any(1)) and np.all((offset < 0).any(1)), "a fault triangle lacks a cell on one side"
    dist = np.linalg.norm(rel_xyz, axis=2)
    east_i = near[np.arange(len(near)), np.argmin(np.where(offset > 0, dist, np.inf), axis=1)]
    west_i = near[np.arange(len(near)), np.argmin(np.where(offset < 0, dist, np.inf), axis=1)]
    shallow = tri_cen[:, 2] > -100
    print(f"{medium}: fault-adjacent mu, top 100 m median: west {np.median(cell_mu[west_i][shallow]) / 1e9:.3f}, "
          f"east {np.median(cell_mu[east_i][shallow]) / 1e9:.3f} GPa")
    return 0.5 * (cell_mu[east_i] + cell_mu[west_i]), cell_mu[west_i], cell_mu[east_i], float(np.median(np.abs(rel)))


mu_mean, mu_west, mu_east, mu_rel_median = mu_beside_fault("sfcvm", grid_file)
assert mu_rel_median < 0.05, "PyLith's cell rigidity differs from the aligned grid by more than 5 % (median)"
mu_fault = {"sfcvm": mu_mean}
shallow_tri = tri_cen[:, 2] > -100
mu_top100 = dict(west_gpa=float(np.median(mu_west[shallow_tri]) / 1e9), east_gpa=float(np.median(mu_east[shallow_tri]) / 1e9))
print("PASS: PyLith used the aligned SFCVM rigidity")

# %% [markdown]
# ## [7] 07a's data and machinery
#
# `scripts/07a_uniform_model.py` cells [1]-[6] exec'd unchanged: data, quadtree (2386 samples),
# noise covariance, the table of measured crossings (`crossings_all`, used by [10]'s surface
# constraint), the CSI triangle fault and `invert`. Only 07a's registry object is neutralised (this
# notebook has its own). The samples must be the PyLith output points, and 07a's
# `fig2_model_row.npz` (its baseline on the same samples) is loaded for the comparison rows. Then
# the side check: the mesh fault (a polyline through its top-edge vertices, on the spline through
# every 3rd trace point) vs 07a's trace (20 m discretisation): are any samples on different sides?
#
# Names used below that come from 07a: `samples`, `look`, `fault`, `mesh_nodes`, `mesh_faces`,
# `dip_deg`, `wrap_fault`, `cell_area_km2`, `crossings_all`, `km_window`, `trace_south_north`,
# `crossing_median` (the surface % denominator of 07a [5], re-derived and asserted equal in [10]),
# `creep_lo_km`, `creep_hi_km`, `cov_p`, `sigma_grid`, `lam_xz`, `surf_floor_mm`, `n_boot`,
# `frame_lon0`, `frame_lat0`, `lat_to_km_pp`, `strike_bearing`, `field_dir`, `events_dir`, `tests_dir`.
# NB: 07a's cell [1] silences every warning again (`warnings.simplefilter("ignore")`).

# %%
## [7]
source_07a = open(os.path.join(repo_dir, "scripts", "07a_uniform_model.py")).read()
head_07a = source_07a.split("# %% [markdown]\n# ### CHECK")[0]                 # cells [1]-[6]
for old, new in (("REG = utils.Registry(notebook, common_paths, repo_dir)", "REG = None"), ("REG.inputs(", "inputs_07a = (")):
    assert head_07a.count(old) == 1, f"07a_uniform_model.py layout changed: {old!r} not found once"
    head_07a = head_07a.replace(old, new)
assert "def invert(" in head_07a and "def make_surface(" in head_07a, "07a_uniform_model.py layout changed: cells [1]-[6] not found"
t0 = time.time()
ns07 = {"__name__": "nb07a"}
exec(head_07a, ns07)
protected = {"notebook", "out_dir", "fig_dir", "REG"}                        # this notebook's own
for name_07a, value_07a in ns07.items():
    if name_07a not in protected and not name_07a.startswith("__"):
        globals()[name_07a] = value_07a
print(f"07a cells [1]-[6] exec'd in {time.time() - t0:.0f} s: {samples.vel.size} samples, sigma grid {sigma_grid}, "
      f"lambda {lam_xz} km, surface floor {surf_floor_mm} mm, n_boot {n_boot}")
row07_file = os.path.join(tests_dir, "fig2_model_row.npz")
assert os.path.exists(row07_file), f"missing {row07_file}: run 07a_uniform_model first"
row07 = np.load(row07_file)
fault07_trace = copy.deepcopy(fault)                                          # 07a's fault (trace), for maps
assert samples.lon.size == points["lon"].size and np.array_equal(samples.lon, points["lon"]) \
    and np.array_equal(samples.lat, points["lat"]), "the quadtree samples are not the PyLith output points"
assert np.array_equal(row07["sample_lon"], samples.lon) and np.ptp(samples.vel - row07["data_mm"]) < 1e-9, \
    "07a's fig2_model_row.npz is not on the same samples / data as this run: re-run 07a"
block_lon = row07["block_lon"]
block_lat = row07["block_lat"]
from plot_run_summary import signed_distance_m                                # one definition, shared with 08b's one-page diagnostic

signed_d_km = signed_distance_m(fault07_trace.xi * 1e3, fault07_trace.yi * 1e3, points["x"], points["y"]) / 1e3
print(f"07a baseline: sigma {float(row07['sigma']):g}, rms {float(row07['rms_mm']):.3f} mm, Mw {float(row07['mw_30gpa']):.3f}")

top = fault_xyz[np.abs(fault_xyz[:, 2]) < 1e-6]                           # the mesh fault's top-edge vertices
top = top[np.argsort(top[:, 1])]                                            # ... sorted south to north (mesh y)
dense_s = np.linspace(0, 1, 20000)
top_arc = np.r_[0, np.cumsum(np.hypot(np.diff(top[:, 0]), np.diff(top[:, 1])))]
top_arc /= top_arc[-1]
top_dense_x = np.interp(dense_s, top_arc, top[:, 0])
top_dense_y = np.interp(dense_s, top_arc, top[:, 1])
signed_mesh_m = signed_distance_m(top_dense_x, top_dense_y, points["x"], points["y"])
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
# u = (s/pi) arctan(D/x) for uniform slip on km 54-65, top 1 km, at samples within 2 km of the trace
# and at least 3 km from the slip ends (D = 1 km + half a fault cell for the tent taper); PyLith is
# lower at 1-2 km, the 3D end effect of an 11 km patch, so this is a sanity check.

# %%
## [8]
def screw_dislocation_mm(slip_mm, locking_depth_m, x_m):
    """Surface displacement of a 2D screw dislocation, slip s from the surface to depth D:
    u(x) = (s / pi) arctan(D / x), x the signed distance from the fault (m)."""
    return (slip_mm / np.pi) * np.arctan(locking_depth_m / x_m)


def median_edge_length_m(triangles_xyz):
    """Median length of all the edges of a set of triangles (n, 3 vertices, xyz), m."""
    edge_01 = np.linalg.norm(triangles_xyz[:, 1] - triangles_xyz[:, 0], axis=1)
    edge_12 = np.linalg.norm(triangles_xyz[:, 2] - triangles_xyz[:, 1], axis=1)
    edge_20 = np.linalg.norm(triangles_xyz[:, 0] - triangles_xyz[:, 2], axis=1)
    return float(np.median(np.concatenate([edge_01, edge_12, edge_20])))


def vertex_slip_on_mesh(slip_mm):
    """Slip on every PyLith fault vertex: the impulse vertices carry slip_mm, the locked ones zero."""
    slip_all = np.zeros(len(fault_xyz))
    slip_all[vertex_index] = slip_mm
    return slip_all


imp_xyz = fault_xyz[vertex_index]
top_km = lat_to_km_pp(to_mesh.transform(top[:, 0], top[:, 1], direction="INVERSE")[1])    # km of the mesh trace (top sorted by y)
imp_km = np.interp(imp_xyz[:, 1], top[:, 1], top_km)                        # km from Point Pinole of every impulse vertex
slip_test = np.where((imp_km >= test_km[0]) & (imp_km <= test_km[1]) & (imp_xyz[:, 2] >= -test_depth_m - 1e-6), test_slip_mm, 0.0)
pred_pylith = G["homog"].dot(slip_test)
vertex_slip = vertex_slip_on_mesh(slip_test)
tri_yz = mtri.Triangulation(fault_xyz[:, 1], fault_xyz[:, 2], fault_cells)
interp_slip = mtri.LinearTriInterpolator(tri_yz, vertex_slip)
fault_val = wrap_fault(mesh_nodes, mesh_faces, dip_deg, "validation")
csi_cen = np.array([p.mean(0) for p in fault_val.patch])              # km, depth +
csi_slip = np.asarray(interp_slip(csi_cen[:, 1] * 1e3, -csi_cen[:, 2] * 1e3).filled(0.0))
line = insar("validation", lon0=frame_lon0, lat0=frame_lat0, verbose=False)
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
dx_fault_m = median_edge_length_m(tri_xyz[patch_cells])
D_eff = test_depth_m + 0.5 * dx_fault_m
x_m = signed_mesh_m                                                  # distance from the MESH fault (the source)
km_samples = lat_to_km_pp(samples.lat)
sel_b = (np.abs(x_m) <= 2000) & (np.abs(x_m) >= 20) & (km_samples >= 57.0) & (km_samples <= 62.0)
screw = screw_dislocation_mm(test_slip_mm, D_eff, x_m)
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
# PyLith `FaultCohesiveKin` forward runs (`scripts/pylith_cfg/fwd_kin.cfg`, one static step) with
# 10 mm right-lateral on km 57-61 (y window), top 1 km, homogeneous, on the `default` and the
# `fine` mesh (60 m fault cells in the patch), read from `modeling_SFCVM/pylith_output/`.
# (c) superposition: default-mesh forward = G x (the forward run's own vertex slip) -- tests G
# assembly, point matching and sign, exact to 1e-4 mm; (d) mesh convergence: the fine-mesh forward
# against CSI with the same (fine) slip field, and against the default-mesh forward. The same
# superposition check for the SFCVM medium is made in [14] with the Fig-2 forward run of the
# inverted slip.

# %%
## [9]
kin_db = "spatialdb/kin_slip_km57-61_z1000.spatialdb"
cached(os.path.join(WORK, kin_db), f"{pyenv} python3 {helpers_dir}/make_kin_slip_db.py {kin_y_m[0]} {kin_y_m[1]} {test_depth_m} "
                                   f"{test_slip_mm / 1e3} {kin_db}")


def read_forward(name):
    """Fault-parallel displacement at the samples (mm) and the right-lateral vertex slip (mm) of a forward run."""
    with h5py.File(os.path.join(paths_sfcvm["pylith_output"], f"{name}-points.h5"), "r") as points_h5:
        dist, order = cKDTree(points_h5["geometry/vertices"][:, :2]).query(np.c_[points["x"], points["y"]])
        assert dist.max() < 1e-3, f"{name}: output points are not the samples"
        u_fp = points_h5["vertex_fields/displacement"][-1][order].dot(look) * 1e3            # mm
    with h5py.File(os.path.join(paths_sfcvm["pylith_output"], f"{name}-fault.h5"), "r") as fault_h5:
        f_xyz = fault_h5["geometry/vertices"][:]
        f_cells = np.asarray(fault_h5["viz/topology/cells"][:], int)
        slip_rl = float(gnpz["homog"]["sign_rl"]) * fault_h5["vertex_fields/slip"][-1][:, 1] * 1e3    # RL = sign_rl x LL (build_G)
    return dict(u=u_fp, xyz=f_xyz, cells=f_cells, slip=slip_rl)


forward_runs = {}
for pre in ("default", "fine"):
    name = f"fwd_{pre}_homog"
    if pre == "fine":
        cached(os.path.join(paths_sfcvm["mesh"], "mesh_vertical_tet_fine.msh"),
               f"{pyenv} python3 {helpers_dir}/generate_gmsh_vertical.py --write --preset fine")
    cached(os.path.join(paths_sfcvm["pylith_output"], f"{name}-points.h5"),
           f"{pyenv} pylith {cfg_dir}/pylithapp.cfg {cfg_dir}/mat_homog.cfg {cfg_dir}/fwd_kin.cfg --problem.defaults.name={name} "
           f"--problem.mesh_initializer.phases.read_mesh.reader.filename=mesh/mesh_vertical_tet_{pre}.msh --nodes=2")
    forward_runs[pre] = read_forward(name)
    print(f"{name}: slip {forward_runs[pre]['slip'].min():.2f}..{forward_runs[pre]['slip'].max():.2f} mm on {np.sum(forward_runs[pre]['slip'] > 1e-6)} vertices; "
          f"max |u| {np.abs(forward_runs[pre]['u']).max():.2f} mm")
    assert np.all(forward_runs[pre]["slip"] > -1e-6), f"{name}: forward slip is not right-lateral"

# (c) superposition on the default mesh, homogeneous medium
vertex_dist, vertex_match = cKDTree(forward_runs["default"]["xyz"]).query(imp_xyz)
assert vertex_dist.max() < 1e-6, "forward-run fault vertices differ from the GF fault vertices"
assert np.setdiff1d(np.nonzero(forward_runs["default"]["slip"] > 1e-6)[0], vertex_match).size == 0, "forward slip on vertices without an impulse"
superposed_mm = G["homog"].dot(forward_runs["default"]["slip"][vertex_match])
superposition_diff_mm = superposed_mm - forward_runs["default"]["u"]
val_c = dict(max_abs_diff_mm=float(np.abs(superposition_diff_mm).max()), rms_diff_mm=float(np.sqrt(np.mean(superposition_diff_mm ** 2))),
             rms_signal_mm=float(np.sqrt(np.mean(superposed_mm ** 2))))
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
for pre in forward_runs:
    diff_pre = forward_runs[pre]["u"] - csi_prediction(forward_runs[pre]["xyz"], forward_runs[pre]["cells"], forward_runs[pre]["slip"])
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

plt.rcParams.update({"font.size": 12})
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
for ax, pre in zip(axes, forward_runs):
    cpred = csi_prediction(forward_runs[pre]["xyz"], forward_runs[pre]["cells"], forward_runs[pre]["slip"])
    ax.scatter(signed_d_km[ok_side], (forward_runs[pre]["u"] - cpred)[ok_side], s=6, color=OK["black"], linewidths=0)
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
# impulse vertices, in G's column order; faces = the PyLith fault triangles among them). The G
# matrix enters through `Fault.setGFs`. Everything else is 07a's `invert`: exponential prior
# `buildCmXY` with lambda (0.1, 0.05) km on the vertex positions, per-dataset constant
# (`csi.transformation`), NNLS (`multisourcesolve.ConstrainedLeastSquareSoln`). The surface
# constraint (Dani 2026-10-02, `pylith_helpers/surface_crossings.py`): every measured fault crossing
# inside the mesh (04's five-event mean step, one every 0.1 km) is one datum, tied to the modelled
# slip at its own position on the top edge, i.e. linear between the two neighbouring top-edge
# vertices (the order-1 tents); error = the local scatter of the crossings (1.4826 MAD within
# +/- 0.2 km), floored at 0.5 mm. No running median, nothing invented across data gaps.
# PREPUB: (Until 2026-10-02: 07a's 0.4 km running median interpolated to every vertex; its window test lost the
# PREPUB: edge crossings to float roundoff and kept 29 of 57 points. On the selected model the change leaves
# PREPUB: Mw30 at 3.756 and the rms within 0.02 mm.)
# The surface % of [12] compares the same measured crossings with the model: the median modelled
# slip on the top-edge vertices of the SSE section over the median of the measured crossings in the
# SSE section (`section_crossing_median_mm`, defined here by `surface_crossings.surface_pct_denominator_mm`;
# Dani 2026-10-03). It is the ONE denominator of every surface % (07a, 07b, 08b and the PyLith driver
# use the same helper; asserted equal to 07a's `crossing_median`), so every cached surface % read in [12]
# and [14] is compared as it is.
# PREPUB: (Until 2026-10-05 07a and the driver divided by the median of 07a's running-median profile and
# PREPUB: this notebook re-expressed their numbers by the ratio of the two medians.)
# Also here: the numbers of one solution (`numbers`, shared with 08b), and Mw from the moment.
# NB: `buildCmXY` normalises by lambda0 from the parameter spacing, so sigma is not numerically
# comparable with 07a's triangle sigma; each model gets its own L-curve corner, by 07a's rule.
#
# **Data covariance.** The one thing that differs between the selected model and its companion;
# a standard modelling choice, not new physics:
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
#   errors of 08b are formal, and the structural range comes from the sensitivity runs.
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
imp_lon, imp_lat = to_mesh.transform(imp_xyz[:, 0], imp_xyz[:, 1], direction="INVERSE")
column_of_vertex = {}                                   # PyLith vertex index -> its column in G (impulse order)
for column, vertex in enumerate(vertex_index):
    column_of_vertex[vertex] = column
sub_faces = []                                          # the fault triangles whose three vertices all carry an impulse
for triangle in fault_cells:
    if all(vertex in column_of_vertex for vertex in triangle):
        sub_faces.append([column_of_vertex[vertex] for vertex in triangle])
sub_faces = np.array(sub_faces)
surface_imp = np.nonzero(np.abs(imp_xyz[:, 2]) < 1e-6)[0]
imp_depth_km = -imp_xyz[:, 2] / 1e3


def tent_fault_template():
    """CSI TriangularTents on the impulse vertices (km, depth +), in G's column order (built once: its adjacency
    map is slow; each solve deep-copies it, as 07a builds a fresh fault per solve)."""
    template = TriangularTents("tents", lon0=frame_lon0, lat0=frame_lat0, verbose=False)
    x_km, y_km = template.ll2xy(imp_lon, imp_lat)
    assert np.allclose(x_km * 1e3, imp_xyz[:, 0], atol=1e-3) and np.allclose(y_km * 1e3, imp_xyz[:, 1], atol=1e-3), \
        "CSI frame differs from the mesh CRS"
    template.Vertices = np.c_[x_km, y_km, -imp_xyz[:, 2] / 1e3]
    template.Vertices_ll = np.c_[imp_lon, imp_lat, -imp_xyz[:, 2] / 1e3]
    template.Faces = sub_faces.copy()
    template.patch = [template.Vertices[face] for face in template.Faces]
    template.patchll = [template.Vertices_ll[face] for face in template.Faces]
    template.vertices2tents()
    template.initializeslip(n=n_imp)
    assert len(template.tent) == n_imp and template.N_slip == n_imp
    return template


t0 = time.time()
TENTS = tent_fault_template()
print(f"CSI TriangularTents: {len(TENTS.tent)} tents, {len(TENTS.Faces)} faces ({time.time() - t0:.0f} s)")


def tent_fault(name):
    """A fresh copy of the tent fault for one solve."""
    fault_model = copy.deepcopy(TENTS)
    fault_model.name = name
    return fault_model


def trace_lon_of_lat(lat):
    """Longitude of the mapped trace at a latitude (07a's trace, south to north)."""
    return np.interp(lat, trace_south_north[:, 1], trace_south_north[:, 0])


def lonlat_to_mesh_xy(lon, lat):
    """(x, y) in the mesh CRS (m) of lon/lat points."""
    return to_mesh.transform(lon, lat)


top_order = surface_imp[np.argsort(imp_xyz[surface_imp, 1])]          # top-edge impulse vertices, south to north (mesh y)
assert np.all(np.diff(imp_xyz[top_order, 1]) > 0), "top-edge vertices not monotone in mesh y"
crossings_in = crossings_all[(crossings_all.km >= km_window[0]) & (crossings_all.km <= km_window[1])]

# the denominator of surface % ([12]): the median of the measured crossings in the SSE section (Dani 2026-10-03),
# from the one shared helper; 07a's cells [1]-[6] computed the same number (`crossing_median`) with it
n_section_crossings = int(np.sum((crossings_in.km >= creep_lo_km) & (crossings_in.km <= creep_hi_km)
                                 & np.isfinite(crossings_in.mean_event_mm)))
section_crossing_median_mm = surface_pct_denominator_mm(crossings_in.km.values, crossings_in.mean_event_mm.values,
                                                        creep_lo_km, creep_hi_km)
assert abs(section_crossing_median_mm - crossing_median) < 1e-12, \
    f"surface % denominators differ: 08a {section_crossing_median_mm} vs 07a {crossing_median} (07a out of date?)"
print(f"surface % denominator: median of {n_section_crossings} measured crossings in the SSE section "
      f"{section_crossing_median_mm:.2f} mm (= 07a's crossing_median)")
# the 06 field every inversion here fits: the cached driver tables read in [12] and [14] must carry the same fingerprint
field_fingerprint = f"fp_maps.npz:{utils.file_fingerprint(os.path.join(field_dir, 'fp_maps.npz'))}"   # the driver's form


def surface_constraint():
    """The measured crossings, each tied to the tent slip at its own top-edge position (surface_crossings.py).
    Returns the CSI surfaceslip data, its G rows and the crossing_constraint dictionary."""
    constraint = crossing_constraint(crossings_in.km.values, crossings_in.lat.values, crossings_in.mean_event_mm.values,
                                     trace_lon_of_lat, lonlat_to_mesh_xy, imp_xyz[top_order, :2], top_order, n_imp,
                                     floor_mm=surf_floor_mm)
    surface_data = surfaceslip(f"crossings f{surf_floor_mm:g}", lon0=frame_lon0, lat0=frame_lat0, verbose=False)
    surface_data.read_from_binary(constraint["vel"], constraint["lon"], constraint["lat"], err=constraint["err"])
    surface_data.buildCd()
    return surface_data, constraint["G"], constraint


_, _, surface_rows = surface_constraint()
print(f"surface constraint: {surface_rows['vel'].size} crossings inside the mesh top edge "
      f"(km {surface_rows['km'].min():.1f}-{surface_rows['km'].max():.1f}; {surface_rows['n_outside']} beyond its ends), "
      f"distance to the edge median {np.median(surface_rows['dist_m']):.1f} m, max {surface_rows['dist_m'].max():.1f} m; "
      f"error median {np.median(surface_rows['err']):.2f} mm")


def data_cd(cd):
    """The data covariance of one treatment: 07a's full correlated Cd (samples.Cd, 07a [4]) or its diagonal
    nugget part alone (diag of the full Cd = nugget / n_px^0.5 + the two sills at zero lag)."""
    if cd == "full":
        return samples.Cd.copy()
    assert cd == "nugget", f"unknown Cd treatment {cd!r}"
    Cd = np.diag(np.diag(samples.Cd) - (cov_p[1] + cov_p[3]))
    assert np.all(np.diag(Cd) > 0)
    return Cd


def invert_tents(medium, sigma, cd, vel_insar=None, vel_surface=None):
    """07a's invert() with PyLith G on the tent parameters (setGFs instead of buildGFs). Returns fault, data, surface.
    vel_insar / vel_surface: replacement data vectors (bootstrap and recovery tests, 08b)."""
    fault_model = tent_fault(f"{medium} {cd} sigma {sigma}")
    insar_data = copy.deepcopy(samples)
    if vel_insar is not None:
        insar_data.vel[:] = vel_insar
    insar_data.Cd = data_cd(cd)
    surface, G_surf, _ = surface_constraint()
    if vel_surface is not None:
        surface.vel[:] = vel_surface
    fault_model.setGFs(insar_data, strikeslip=[G[medium].copy()])     # already right-lateral positive (build_G sign)
    fault_model.setGFs(surface, strikeslip=[G_surf])
    solve_datas = [insar_data, surface]                               # surface constraint ON (use_surface)
    trans = transform("ramps", lon0=frame_lon0, lat0=frame_lat0, verbose=False)
    trans.buildGFs(solve_datas, ["1", "1"], verbose=False)
    fault_model.assembleGFs(solve_datas, slipdir="s", verbose=False)
    fault_model.assembled(solve_datas)
    fault_model.assembleCd(solve_datas)
    trans.assembleGFs(solve_datas)
    trans.assembled(solve_datas)
    trans.assembleCd(solve_datas)
    fault_model.buildCmXY(sigma, lam_xz, verbose=False)
    trans.buildCm(1e4, verbose=False)
    solver = multisrc(f"{medium} mean event", [fault_model, trans], verbose=False)
    solver.assembleGFs()
    solver.assembleCm()
    mprior = np.zeros(solver.Np)
    ramp_idx = []                                                     # the parameters of the per-dataset constants
    for i, param_type in enumerate(solver.paramTypes):
        if param_type[0] == trans.name:
            ramp_idx.append(i)
    mprior[ramp_idx] = -1e3
    solver.ConstrainedLeastSquareSoln(method="nnls", mprior=mprior)
    solver.distributem()
    for dataset in solve_datas:
        dataset.buildsynth([fault_model], direction="s")
        dataset.removeTransformation(trans)
    return fault_model, insar_data, surface


def potency_m3(slip_mm):
    """Exact integral of the piecewise-linear (tent) slip over the PyLith fault triangles (locked vertices = 0):
    P = sum over triangles of area x mean of its three vertex slips."""
    slip_all = vertex_slip_on_mesh(slip_mm)
    return float(np.sum(tri_area * slip_all[fault_cells].mean(1)) * 1e-3)


def moment_sfcvm(slip_mm):
    """Sum over fault triangles of area x mean tent slip x mu (mean of the PyLith cells either side, aligned SFCVM).
    For the homogeneous model this is a rescale, as 07a's Mw_sfcvm."""
    slip_all = vertex_slip_on_mesh(slip_mm)
    return float(np.sum(tri_area * slip_all[fault_cells].mean(1) * 1e-3 * mu_fault["sfcvm"]))


def moment_magnitude(m0_nm):
    """Mw = 2/3 (log10 M0 - 9.1), M0 in N m (Hanks & Kanamori 1979)."""
    return (2.0 / 3.0) * (np.log10(m0_nm) - 9.1)


# the numbers of one solution (here, not in [12], so that 08b, which execs cells [1]-[10], uses the same function)
west300 = (signed_d_km < 0) & (signed_d_km >= -0.3)                   # samples within 300 m west of 07a's trace
east300 = (signed_d_km > 0) & (signed_d_km <= 0.3)                    # ... and east
in_sec_surface = np.isin(np.arange(n_imp), surface_imp) & (imp_km >= creep_lo_km) & (imp_km <= creep_hi_km)


def numbers(fault_model, insar_data, surface):
    """The numbers of one solution: rms (all, 300 m west / east), chi2 with the run's Cd, surface % (median top-edge
    slip in the SSE section / median measured crossing there), potency, Mw at 30 GPa and SFCVM mu, peak slip."""
    slip_vec = fault_model.slip[:, 0]
    res = insar_data.vel - insar_data.synth
    pot = potency_m3(slip_vec)
    m0_sfcvm = moment_sfcvm(slip_vec)
    return dict(rms_mm=float(np.sqrt(np.mean(res ** 2))), rms_west300_mm=float(np.sqrt(np.mean(res[west300] ** 2))),
                rms_east300_mm=float(np.sqrt(np.mean(res[east300] ** 2))), chi2=float(np.dot(res, np.linalg.solve(insar_data.Cd, res))),
                surface_pct=float(100 * np.median(slip_vec[in_sec_surface]) / section_crossing_median_mm), potency_m3=pot,
                Mw_30gpa=float(moment_magnitude(mu_ref * pot)), Mw_sfcvm=float(moment_magnitude(m0_sfcvm)),
                mu_effective_gpa=m0_sfcvm / pot / 1e9,
                peak_slip_mm=float(slip_vec.max()), peak_depth_m=float(imp_depth_km[np.argmax(slip_vec)] * 1e3),
                surf_rms_mm=float(np.sqrt(np.mean((surface.vel - surface.synth) ** 2))))


assert use_surface, "the selected model has the surface constraint ON; the InSAR-only run is a sensitivity case (08b)"

# %% [markdown]
# ## [11] Sigma: the L-curve corner for each model (07a [8] rule)
#
# Same sigma grid, same weighted misfit sqrt(chi2 InSAR + chi2 surface) against the prior-weighted
# roughness, corner by maximum curvature of cubic splines in log sigma (end intervals excluded),
# selected = smallest grid sigma at or above the corner. Four L-curves: the two media under each
# covariance treatment. Every model must land on the locked sigma 12 (asserted).
# PREPUB: CHECK against the earlier run: each L-curve equals the cached `lcurve_<medium><tag>.csv` in
# PREPUB: `modeling_SFCVM/gf/`, made by the PyLith driver run of 2026-10-03 (`drafts/pylith/pylith_sfcvm_gf.py`
# PREPUB: via `drafts/pylith/overnight_2026-10-02.sh`), the run 08b's sensitivity table comes from.
#
# Why the grid value and not the corner itself (Dani 2026-10-03, locked in): the corner is broad. For
# the selected model it sits at sigma 10.3 on this grid and at 11.1 when steps at 7, 9, 10, 11, 13 and
# 14 are added; solved at the corner the model has Mw30 3.747 against 3.756 at sigma 12, rms 1.36
# against 1.35 mm, and the same surface slip (a one-off check, 2026-10-03, not part of this notebook).
# Sigma 12, the nearest grid value above the corner, is kept for every model.

# %%
## [11]
def lcurve_corner(sigma, misfit, roughness):
    """The L-curve corner (07a [8] rule): cubic splines of log10 misfit and log10 roughness against log10 sigma, the
    maximum of the curvature |m' r'' - r' m''| / (m'^2 + r'^2)^1.5 on a fine grid, the two end intervals excluded."""
    log_sigma = np.log10(sigma)
    misfit_spline = CubicSpline(log_sigma, np.log10(misfit))
    roughness_spline = CubicSpline(log_sigma, np.log10(roughness))
    fine_ls = np.linspace(log_sigma[1], log_sigma[-2], 400)
    d_misfit = misfit_spline(fine_ls, 1)
    dd_misfit = misfit_spline(fine_ls, 2)
    d_rough = roughness_spline(fine_ls, 1)
    dd_rough = roughness_spline(fine_ls, 2)
    curvature = np.abs(d_misfit * dd_rough - d_rough * dd_misfit) / (d_misfit ** 2 + d_rough ** 2) ** 1.5
    return float(10 ** fine_ls[np.argmax(curvature)])


def lcurve(medium, cd):
    """One L-curve over sigma_grid: misfit, roughness, rms and Mw30 per sigma; the corner and the selected sigma."""
    rows = []
    for sigma in sigma_grid:
        t0 = time.time()
        fault_model, insar_data, surface = invert_tents(medium, sigma, cd)
        res_insar = insar_data.vel - insar_data.synth
        res_surface = surface.vel - surface.synth
        slip_vec = fault_model.slip[:, 0]
        rows.append(dict(sigma=sigma, chi2=float(np.dot(res_insar, np.linalg.solve(insar_data.Cd, res_insar))),
                         chi2_surf=float(np.dot(res_surface, np.linalg.solve(surface.Cd, res_surface))),
                         prior_norm=float(np.sqrt(np.dot(slip_vec, np.linalg.solve(fault_model.Cm / sigma ** 2, slip_vec)))),
                         rms_mm=float(np.sqrt(np.mean(res_insar ** 2))), peak_slip_mm=float(slip_vec.max()),
                         Mw_30=moment_magnitude(mu_ref * potency_m3(slip_vec)), seconds=time.time() - t0))
        rows[-1]["chi2_total"] = rows[-1]["chi2"] + rows[-1]["chi2_surf"]
        print(f"  {medium} {cd} sigma {sigma:>4g}: chi2 {rows[-1]['chi2']:.0f} + {rows[-1]['chi2_surf']:.0f}, "
              f"roughness {rows[-1]['prior_norm']:.1f}, rms {rows[-1]['rms_mm']:.3f}, Mw30 {rows[-1]['Mw_30']:.2f} "
              f"({rows[-1]['seconds']:.0f} s)")
    lcurve_table = pd.DataFrame(rows)
    corner = lcurve_corner(lcurve_table.sigma.values, np.sqrt(lcurve_table.chi2_total.values), lcurve_table.prior_norm.values)
    selected = float(lcurve_table.sigma[lcurve_table.sigma >= corner].min())
    assert selected == sigma_locked, f"{medium} {cd}: corner {corner:.1f} now selects sigma {selected:g}, not the locked {sigma_locked:g}"
    lcurve_table["corner"] = corner
    lcurve_table["selected"] = selected
    lcurve_table.to_csv(os.path.join(out_dir, f"lcurve_{medium}{CD_TAG[cd]}.csv"), index=False)
    print(f"{medium} {cd}: L-curve corner sigma {corner:.1f} -> selected {selected:g}")
    return lcurve_table, corner, selected


families = {}
for medium, cd in RUNS:
    families[(medium, cd)] = lcurve(medium, cd)
    # PREPUB-START: comparison with the cached products of the earlier PyLith driver run
    cached_lc = pd.read_csv(os.path.join(paths_sfcvm["gf"], f"lcurve_{medium}{CD_TAG[cd]}.csv"))
    lcurve_table = families[(medium, cd)][0]
    assert np.array_equal(cached_lc.sigma.values, lcurve_table.sigma.values), "cached L-curve is on another sigma grid"
    lc_diff = float(np.max(np.abs(cached_lc.chi2_total.values / lcurve_table.chi2_total.values - 1)))
    assert lc_diff < 1e-6 and float(cached_lc.selected.iloc[0]) == families[(medium, cd)][2], \
        f"{medium} {cd}: L-curve differs from the cached earlier run (max rel chi2 diff {lc_diff:.1e})"
    print(f"PASS: {medium} {cd} L-curve equals modeling_SFCVM/gf/lcurve_{medium}{CD_TAG[cd]}.csv (max rel chi2 diff {lc_diff:.1e})")
    # PREPUB-END

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
# PREPUB: CHECK against the earlier run: the two PyLith rows equal the cached `results_comparison<tag>.csv`
# PREPUB: in `modeling_SFCVM/gf/` on sigma, rms, rms west, potency, Mw30, Mw_sfcvm and surface %.

# %%
## [12]
res07 = row07["residual_mm"]
models = {}                     # (medium, cd) -> the solution at the selected sigma and its numbers
results = {}                    # cd -> the comparison table
for cd in CD_TAG:
    Cd07 = data_cd(cd)
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
    # PREPUB-START: comparison with the cached products of the earlier PyLith driver run
    cached_res = pd.read_csv(os.path.join(paths_sfcvm["gf"], f"results_comparison{CD_TAG[cd]}.csv")).set_index("model")
    for medium in media:
        assert str(cached_res.loc[f"PyLith {medium}", "field_fingerprint"]) == field_fingerprint, \
            f"cached results_comparison{CD_TAG[cd]}.csv ({medium}) fits another 06 field: re-run the driver"
    for medium in media:
        for col in ("sigma", "rms_mm", "rms_west300_mm", "potency_m3", "Mw_30gpa", "Mw_sfcvm", "surface_pct"):
            mine = float(results[cd].loc[f"PyLith {medium}", col])
            theirs = float(cached_res.loc[f"PyLith {medium}", col])
            assert abs(mine - theirs) <= 1e-6 * max(abs(theirs), 1e-9), f"{medium} {cd} {col}: {mine} vs cached {theirs}"
    print(f"PASS: results ({run_tag(cd)}) equal modeling_SFCVM/gf/results_comparison{CD_TAG[cd]}.csv")
    # PREPUB-END
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
            ax.set_aspect(1 / np.cos(np.deg2rad(frame_lat0)))
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
# `fig2_model_row.npz` [13], with the fault on the impulse vertices (`nodes_kmz`, `faces`,
# `slip_vertex_mm`). The profile lines X-X' and Y-Y' (fault-parallel and vertical, every 10 m,
# zeroed on the far field as 06 zeroes the observed profiles) come from ONE PyLith
# `FaultCohesiveKin` forward run of the inverted vertex slip (`fwd_kin_simpledb.cfg`: a SimpleDB
# with the vertex slip, "nearest", the same mesh and material as the GF run, output at the profile
# points and the samples), read from `modeling_SFCVM/pylith_output/fwd_fig2_<label>`.
#
# CHECKS: the forward run's imposed slip equals the slip just inverted (< 1e-4 mm: the cached
# forward run IS this model), and the forward displacement at the samples equals G x slip
# (< 0.05 mm rms: superposition for the SFCVM medium, the counterpart of [9]'s homogeneous check).
# The Mw30 range on the product is the L-curve grid neighbours of this run plus the +/- 50 m
# alignment runs of 08b's table (`results_comparison_{e50,w50}<tag>.csv` in `modeling_SFCVM/gf/`), both
# at the locked sigma (asserted: every driver run is solved at `sigma_locked`, Dani 2026-10-03; the
# alignment row must also equal its own L-curve at that sigma and fit the current 06 field); the
# bootstrap 2-sigma comes from 08b's `uncertainty_<run>_<medium>.csv` in `modeling_SFCVM/gf/` (08b re-makes
# and checks it); the bootstrap's reference row must match this model, surface % included.
# PREPUB: the alignment and bootstrap numbers are read from the cached products of the earlier PyLith driver run.

# %%
## [14]
to_mesh_from_field = pyproj.Transformer.from_crs(int(fig2_in["epsg"]), CRS, always_xy=True)
cross_m = float(fig2_in["cross_m"])                         # half length of the profile lines (m)
far_m = 1500.0                                              # the far field the profiles are zeroed on (07a [13])
d_line = np.arange(-cross_m, cross_m + 1.0, 10.0)
d_line = d_line[np.abs(d_line) > 5.0]                       # 07a [13]: the trace itself is a discontinuity
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
    slip_all = vertex_slip_on_mesh(mdl["slip"] * 1e-3)      # m, on every fault vertex
    fwd_name = f"fwd_fig2_{label}"
    slip_npz = f"gf/fig2_slip_{label}.npz"
    slip_db = f"spatialdb/fig2_slip_{label}.spatialdb"
    pts_file = f"points/fig2_points_{label}.txt"
    if not os.path.exists(os.path.join(paths_sfcvm["pylith_output"], f"{fwd_name}-points.h5")) and run_pylith:
        np.savez(os.path.join(WORK, slip_npz), xyz=fault_xyz, slip_rl_m=slip_all, sign_rl=float(gnpz[medium]["sign_rl"]))
        run_shell(f"{pyenv} python3 {helpers_dir}/make_vertex_slip_db.py {slip_npz} {slip_db}")
        with open(os.path.join(WORK, pts_file), "w") as fh:
            fh.write(f"# 07a [13] profile points ({', '.join(prof_names)}: {d_line.size} each, +/- {cross_m:.0f} m every 10 m) "
                     f"+ the {points['lon'].size} quadtree samples; 08a_3d_model {time.strftime('%Y-%m-%d %H:%M')}\n# name x y z (m) in {CRS}\n")
            for k, (x, y) in enumerate(pts):
                fh.write(f"p{k:05d} {x:.3f} {y:.3f} 0.0\n")
    cached(os.path.join(paths_sfcvm["pylith_output"], f"{fwd_name}-points.h5"),
           f"{pyenv} pylith {cfg_dir}/pylithapp.cfg {cfg_dir}/{media[medium]} {cfg_dir}/fwd_kin_simpledb.cfg --problem.defaults.name={fwd_name} "
           f"--problem.mesh_initializer.phases.read_mesh.reader.filename=mesh/{os.path.basename(mesh_file)} "
           f"--problem.solution_observers.points.reader.filename={pts_file} "
           f"--problem.interfaces.fault.eq_ruptures.rupture.db_auxiliary_field.iohandler.filename={slip_db} --nodes=4")
    with h5py.File(os.path.join(paths_sfcvm["pylith_output"], f"{fwd_name}-points.h5"), "r") as points_h5:
        dist, order = cKDTree(points_h5["geometry/vertices"][:, :2]).query(pts)
        assert dist.max() < 1e-3, f"{fwd_name}: output points are not the profile points + samples"
        displacement_3d_mm = points_h5["vertex_fields/displacement"][-1][order] * 1e3       # mm, 3 components
    with h5py.File(os.path.join(paths_sfcvm["pylith_output"], f"{fwd_name}-fault.h5"), "r") as fault_h5:
        f_xyz = fault_h5["geometry/vertices"][:]
        slip_fwd = float(gnpz[medium]["sign_rl"]) * fault_h5["vertex_fields/slip"][-1][:, 1] * 1e3
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
            profiles[f"prof_{name}_{component}_mm"] = vals - np.median(vals[np.abs(d_line) >= far_m])   # 07a [13]: far field zeroed
        step = profiles[f"prof_{name}_fp_mm"]
        print(f"  {name}-{name}' model fault-parallel: west {step[d_line < 0].min():+.2f} mm, east {step[d_line > 0].max():+.2f} mm")

    # PREPUB-START: the alignment-run and bootstrap numbers are read from the cached products of the earlier PyLith driver run
    alignment_rows = {}                                       # e50 / w50 -> the SFCVM row of 08b's alignment runs (this Cd)
    if medium == "sfcvm":
        for off in ("e50", "w50"):
            f_off = os.path.join(paths_sfcvm["gf"], f"results_comparison_{off}{CD_TAG[cd]}.csv")
            assert os.path.exists(f_off), f"missing alignment-sensitivity product {f_off} (08b's table)"
            row_off = pd.read_csv(f_off).set_index("model").loc[f"PyLith sfcvm_{off}"]
            assert float(row_off.sigma) == sigma_locked, (f"{f_off}: solved at sigma {row_off.sigma:g}, not the locked "
                                                          f"{sigma_locked:g} (re-run the driver with PYLITH_SIGMA=locked)")
            assert str(row_off.field_fingerprint) == field_fingerprint, f"{f_off} fits another 06 field: re-run the driver"
            lc_off = pd.read_csv(os.path.join(paths_sfcvm["gf"], f"lcurve_sfcvm_{off}_{off}{CD_TAG[cd]}.csv"))
            mw_lc_off = float(lc_off.loc[np.isclose(lc_off.sigma, sigma_locked), "Mw_30"].iloc[0])
            assert abs(mw_lc_off - float(row_off.Mw_30gpa)) < 1e-9, f"{f_off}: Mw30 differs from its own L-curve at sigma {sigma_locked:g}"
            alignment_rows[off] = row_off
    unc_file = os.path.join(paths_sfcvm["gf"], f"uncertainty_{run_tag(cd)}_{medium}.csv")
    assert os.path.exists(unc_file), f"missing bootstrap product {unc_file} (08b)"
    unc = pd.read_csv(unc_file)
    boot_reference = unc[unc.kind == "reference"].iloc[0]
    boot_2sd = unc[unc.kind == "boot_2sd"].iloc[0]
    assert abs(float(boot_reference.Mw_30gpa) - mdl["Mw_30gpa"]) < 1e-6, f"{unc_file} is not the bootstrap of this model"
    assert abs(float(boot_reference.surface_pct) - mdl["surface_pct"]) <= 1e-6 * mdl["surface_pct"], \
        f"{unc_file}: its surface % is not this model's (made on another denominator or field?)"
    # PREPUB-END

    # Mw30 range: the selected sigma and its L-curve grid neighbours (this run), plus the +/- 50 m SFCVM alignment
    # runs where they exist for this Cd (08b's cached table; SFCVM medium only)
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
    # the same range at the SFCVM rigidity (Dani 2026-10-01): sigma neighbours from 08b's bootstrap table (its
    # sigma_neighbour rows), alignment runs from the same cached tables; no new inversion
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
                          f"Cd {cd}, surface constraint ON, forward run {fwd_name}; bootstrap 2-sigma from {os.path.basename(unc_file)}; "
                          f"06 field {field_fingerprint}")
    for key in crossing_keys:
        out[key] = row07[key]
    if medium == "sfcvm":
        out["sfcvm_version"] = sfcvm_label
    else:
        out["sfcvm_version"] = "v26-0 aligned (rescale of the homogeneous model)"
    out.update(profiles)
    np.savez(os.path.join(out_dir, f"fig2_model_row_pylith_{label}.npz"), **out)
    fig2_products[label] = out
    print(f"wrote fig2_model_row_pylith_{label}.npz: Mw30 {out['mw_30gpa']:.2f} +/- {out['mw_30gpa_2sig']:.2f} (2 sigma), "
          f"range {mw_range[0]:.2f}-{mw_range[1]:.2f} ({out['mw_30gpa_range_basis']})")

# %% [markdown]
# ## [15] Products and provenance
#
# `inversion_products<tag>.npz` per covariance treatment (slip, model, residual at the samples,
# both media; the settings, the validation numbers and the results table as json), the inputs of
# 08b's one-page diagnostics.

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
                    data_mm=models[(ref_medium, cd)]["data"], signed_d_km=signed_d_km, block_lon=block_lon, block_lat=block_lat,
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
# areas): the standard definition for a distributed slip model (Noda et al., 2013). PyLith's fault vectors
# are (opening, left-lateral, reverse); right-lateral slip is minus the left-lateral component, and a drop in
# right-lateral shear stress is a positive left-lateral traction change (CHECKED below). No single rigidity
# or depth scale is assumed: the traction change is carried by the SFCVM cells beside the fault. Also
# reported: the mean over the patch where slip exceeds half its peak, its 95th percentile, and the mean over
# that patch in the top 100 m (the soft near-surface rock next to the free surface carries almost no stress change).

# %%
## [15b]
def vertex_areas_m2(vertices_xyz, cells):
    """Area attached to every fault vertex: one third of the area of each triangle it belongs to (m^2)."""
    triangles = vertices_xyz[cells]
    triangle_area = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    areas = np.zeros(len(vertices_xyz))
    np.add.at(areas, cells.ravel(), np.repeat(triangle_area / 3.0, 3))
    return areas


def slip_weighted_stress_drop_pa(dtau_pa, slip_m, vertex_area_m2):
    """Stress drop of a distributed slip model: sum(dtau s A) / sum(s A) over the fault vertices (Noda et al., 2013),
    negative slip counted as zero."""
    weight = np.clip(slip_m, 0.0, None) * vertex_area_m2
    assert np.sum(dtau_pa * weight) > 0, "right-lateral slip did not lower the right-lateral shear traction: sign convention broken"
    return float(np.sum(dtau_pa * weight) / np.sum(weight))


stress_drop = {}
for label, medium in (("nuggetcd", "sfcvm"), ("nuggetcd_homog", "homog")):
    fwd_t = f"fwd_traction_{label}"
    cached(os.path.join(paths_sfcvm["pylith_output"], f"{fwd_t}-fault.h5"),
           f"{pyenv} pylith {cfg_dir}/pylithapp.cfg {cfg_dir}/{media[medium]} {cfg_dir}/fwd_kin_simpledb.cfg --problem.defaults.name={fwd_t} "
           f"--problem.mesh_initializer.phases.read_mesh.reader.filename=mesh/{os.path.basename(mesh_file)} "
           f"--problem.solution_observers.points.reader.filename=points/fig2_points_{label}.txt "
           f"--problem.interfaces.fault.eq_ruptures.rupture.db_auxiliary_field.iohandler.filename=spatialdb/fig2_slip_{label}.spatialdb "
           f"'--problem.interfaces.fault.observers.observer.data_fields=[slip,traction_change]' --nodes=4")
    with h5py.File(os.path.join(paths_sfcvm["pylith_output"], f"{fwd_t}-fault.h5"), "r") as fault_h5:
        f_xyz = fault_h5["geometry/vertices"][:]
        f_cells = np.asarray(fault_h5["viz/topology/cells"][:], int)
        slip_vectors = fault_h5["vertex_fields/slip"][-1]                  # (opening, left-lateral, reverse), m
        dtau = fault_h5["vertex_fields/traction_change"][-1][:, 1]
    assert np.abs(slip_vectors[:, [0, 2]]).max() < 1e-9, "opening or reverse slip in a strike-slip forward run"
    slip_rl = -slip_vectors[:, 1]
    assert abs(slip_rl.max() * 1e3 - fig2_products[label]["slip_vertex_mm"].max()) < 1e-3, f"{fwd_t}: slip is not the inverted slip"
    v_area = vertex_areas_m2(f_xyz, f_cells)
    patch = slip_rl > 0.5 * slip_rl.max()
    patch_top100 = patch & (-f_xyz[:, 2] < 100.0)
    stress_drop[medium] = dict(weighted_kpa=slip_weighted_stress_drop_pa(dtau, slip_rl, v_area) / 1e3, patch_kpa=float(dtau[patch].mean() / 1e3),
                               top100_kpa=float(dtau[patch_top100].mean() / 1e3), p95_kpa=float(np.percentile(dtau[patch], 95) / 1e3))
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
# - Per triangle, from the PyLith cells either side of the fault (`mu_beside_fault`, [6]):
#   arithmetic mean (mu_W + mu_E) / 2 (the rigidity of `Mw_sfcvm`, registered as `08a_mw_sfcvm`); harmonic mean
#   2 mu_W mu_E / (mu_W + mu_E), the effective rigidity of a fault on a material interface (Vavrycuk 2013,
#   GJI 194, 395-400, doi:10.1093/gji/ggt084); west side only; east side only.
# - mu_eff = M0 / P for each per-triangle row: the single rigidity that gives the same Mw.
# - A sweep of single uniform rigidities, Mw(mu) = Mw30 + 2/3 log10(mu / 30 GPa), for quoting a range.
# Potency is the same in every row, so only the rigidity differs. Each row also carries the model range of
# [14]: the selected model, its L-curve sigma neighbours (on the `sfcvm` medium, re-inverted here with
# `invert_tents`) and the +/- 50 m alignment models (08b's driver runs on their own media `sfcvm_e50` /
# `sfcvm_w50`: slip from `inversion_products_{e50,w50}<tag>.npz`, rigidity from their own PyLith
# `-elastic_info.h5` and grid `sfcvm_v26-0_aligned_{e50,w50}_grid.npz`). The 30 GPa, arithmetic and harmonic
# rows take their range PER MODEL (Dani 2026-10-06): each range model's per-triangle potency x the rigidity
# of ITS medium, min/max over the models (saved per model as `mw_range_models.csv`). CHECK: each model's
# Mw30 and arithmetic Mw equal the values [14] built its ranges from (L-curve, 08b's bootstrap table, the
# alignment tables) to < 1e-6, so the arithmetic range is [14]'s `mw_sfcvm_range` (same models, same
# rigidity definition). The west, east and uniform rows are SHIFTED: the 30 GPa range moved by the row's
# rigidity, Mw_lo/hi = Mw + (range - Mw30). Column `range_kind` says which. Printed and saved
# (`mw_rigidity_options.csv`, a short `key` per row: 30gpa, arithmetic, harmonic, west, east, uniform_<mu>).
# The paper quotes 30 GPa and the HARMONIC row (Dani 2026-10-05): registered as `08a_mw_sfcvm_harmonic`
# (selected model), `_lo`/`_hi` (per model), its effective rigidity `08a_mu_sfcvm_harmonic_gpa`, and drawn
# on Fig 2 panel K by 07c from this csv.

# %%
## [15c]
def harmonic_mean_rigidity(mu_a, mu_b):
    """Effective rigidity of a fault on a material interface: 2 mu_a mu_b / (mu_a + mu_b) (Vavrycuk 2013)."""
    return 2.0 * mu_a * mu_b / (mu_a + mu_b)


slip_sel = models[SELECTED]["slip"]
v_sel = vertex_slip_on_mesh(slip_sel)
pot_tri = tri_area * v_sel[fault_cells].mean(1) * 1e-3                     # m3 per triangle (tent slip exact for linear tents)
P_sel = float(pot_tri.sum())
assert abs(P_sel - potency_m3(slip_sel)) < 1e-6 * P_sel, "per-triangle potency does not sum to potency_m3"
mu_options = {"arithmetic mean of the two sides": 0.5 * (mu_west + mu_east),
              "harmonic mean of the two sides (Vavrycuk 2013)": harmonic_mean_rigidity(mu_west, mu_east),
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
    potency_tri = tri_area * vertex_slip_on_mesh(slip_mm)[fault_cells].mean(1) * 1e-3
    return dict(model=model_name, medium=medium_name, potency_m3=float(potency_tri.sum()),
                Mw_30gpa=float(moment_magnitude(mu_ref * potency_tri.sum())),
                Mw_arithmetic=float(moment_magnitude(np.sum(0.5 * (rigidity_west + rigidity_east) * potency_tri))),
                Mw_harmonic=float(moment_magnitude(np.sum(harmonic_mean_rigidity(rigidity_west, rigidity_east) * potency_tri))))


# the models of [14]'s range, each with its own slip and the rigidity of its own medium
medium_sel, cd_sel = SELECTED
members = range_models[selected_label]
lcurve_rows = members["lcurve_rows"]
neighbour_rows = members["bootstrap_neighbour_rows"]
lcurve_neighbour_sigmas = sorted(float(v) for v in lcurve_rows.sigma if not np.isclose(v, models[SELECTED]["sigma"]))
assert len(lcurve_neighbour_sigmas) == len(neighbour_rows) and np.allclose(lcurve_neighbour_sigmas, sorted(neighbour_rows.sigma)), \
    "[14]'s 30 GPa range (L-curve) and SFSVM range (08b's bootstrap table) are not on the same sigma neighbours"
range_rows = [dict(**range_model_mw(f"selected (sigma {models[SELECTED]['sigma']:g})", medium_sel, slip_sel, mu_west, mu_east),
                   Mw_30gpa_14=models[SELECTED]["Mw_30gpa"], Mw_arithmetic_14=models[SELECTED]["Mw_sfcvm"])]
for _, neighbour in neighbour_rows.iterrows():
    fault_neighbour, _, _ = invert_tents(medium_sel, float(neighbour.sigma), cd_sel)
    mw30_lcurve = float(lcurve_rows.loc[np.isclose(lcurve_rows.sigma, neighbour.sigma), "Mw_30"].iloc[0])
    range_rows.append(dict(**range_model_mw(f"sigma {neighbour.sigma:g} (L-curve grid neighbour)", medium_sel,
                                            fault_neighbour.slip[:, 0], mu_west, mu_east),
                           Mw_30gpa_14=mw30_lcurve, Mw_arithmetic_14=float(neighbour.Mw_sfcvm)))
for off, row_off in members["alignment_rows"].items():
    medium_off = f"sfcvm_{off}"                                          # the driver's medium name
    run_off = f"gf_{mesh_preset}_{medium_off}"                          # the driver's GF run of that medium (same mesh)
    grid_off = os.path.join(paths_sfcvm["spatialdb"], f"sfcvm_v26-0_aligned_{off}_grid.npz")
    assert os.path.exists(grid_off), f"missing {grid_off} (the grid of the {medium_off} medium)"
    g_off = np.load(os.path.join(paths_sfcvm["gf"], f"G_{run_off}.npz"))
    assert (np.array_equal(g_off["fault_xyz"], fault_xyz) and np.array_equal(g_off["fault_cells"], fault_cells)
            and np.array_equal(g_off["vertex_index"], vertex_index)), f"G_{run_off}: another fault mesh or impulse order"
    _, mu_west_off, mu_east_off, rel_off = mu_beside_fault(medium_off, grid_off, run_name=run_off)
    assert rel_off < 0.05, f"{run_off}: PyLith's cell rigidity differs from its grid by more than 5 % (median)"
    products_off = np.load(os.path.join(paths_sfcvm["gf"], f"inversion_products_{off}{CD_TAG[cd_sel]}.npz"))
    assert np.allclose(products_off["impulse_vertex_xyz"], imp_xyz), f"inversion_products_{off}: another impulse order"
    assert json.loads(str(products_off["settings"]))["field_fingerprint"] == field_fingerprint, \
        f"inversion_products_{off}{CD_TAG[cd_sel]}.npz fits another 06 field: re-run the driver"
    range_rows.append(dict(**range_model_mw(f"SFSVM boundary 50 m {off[0].upper()} (sigma {row_off.sigma:g})", medium_off,
                                            products_off[f"slip_{medium_off}_mm"], mu_west_off, mu_east_off),
                           Mw_30gpa_14=float(row_off.Mw_30gpa), Mw_arithmetic_14=float(row_off.Mw_sfcvm)))
range_table = pd.DataFrame(range_rows)
range_table["provenance"] = (f"08a_3d_model {time.strftime('%Y-%m-%d %H:%M')}: the Mw range models of fig2_model_row_pylith_{selected_label} "
                             f"([14]); sigma neighbours re-inverted here, alignment slip from inversion_products_{{e50,w50}}{CD_TAG[cd_sel]}.npz; "
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
abs_d = np.abs(signed_d_km)
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
# ## [16] Registry -- the 3D model (the paper's slip model)
#
# `08a_*` = the selected model (aligned SFCVM, nugget Cd); `08a_hom_*` its homogeneous twin;
# `08a_full_*` / `08a_full_hom_*` the companion pair under 07a's full covariance. The 2-sigma and
# range keys come from the Fig-2 product (bootstrap of 08b, L-curve neighbours, alignment +/- 50 m).
# The sidecar also carries the surface % denominator, the stress drop [15b] and the near/far fit [15d].

# %%
## [16]
input_files = []
for medium in media:
    input_files.append(os.path.join(paths_sfcvm["gf"], f"G_{run_names[medium]}.npz"))
input_files += [mesh_fault_npz, grid_file, points_file + ".npz", os.path.join(field_dir, "fp_maps.npz"),
                os.path.join(field_dir, "fig2_inputs.npz"), row07_file, os.path.join(events_dir, "section_lengths.csv"),
                os.path.join(events_dir, "event_fraction_by_crossing.csv")]
# PREPUB-START: the cached products of the earlier PyLith driver run that this notebook reads
for medium, cd in RUNS:
    input_files.append(os.path.join(paths_sfcvm["gf"], f"lcurve_{medium}{CD_TAG[cd]}.csv"))
for cd in CD_TAG:
    input_files.append(os.path.join(paths_sfcvm["gf"], f"results_comparison{CD_TAG[cd]}.csv"))
    for off in ("e50", "w50"):
        input_files.append(os.path.join(paths_sfcvm["gf"], f"results_comparison_{off}{CD_TAG[cd]}.csv"))
for label, (medium, cd) in FIG2_PRODUCTS.items():
    input_files.append(os.path.join(paths_sfcvm["gf"], f"uncertainty_{run_tag(cd)}_{medium}.csv"))
for off in members["alignment_rows"]:                       # [15c]: the alignment range models, slip and medium
    input_files += [os.path.join(paths_sfcvm["gf"], f"inversion_products_{off}{CD_TAG[cd_sel]}.npz"),
                    os.path.join(paths_sfcvm["gf"], f"G_gf_{mesh_preset}_sfcvm_{off}.npz"),
                    os.path.join(paths_sfcvm["spatialdb"], f"sfcvm_v26-0_aligned_{off}_grid.npz")]
# PREPUB-END
REG.inputs(*input_files)
forward_vs_g = []
for label, val in fwd_validation.items():
    forward_vs_g.append(f"{label} {val['rms_diff_mm']:.4f} mm")
lcurve_lines = []
for medium, cd in RUNS:
    lcurve_lines.append(f"{medium} {cd}: corner {families[(medium, cd)][1]:.1f} -> sigma {families[(medium, cd)][2]:g}")
lines = [f"selected model: {run_tag(SELECTED[1])} / {SELECTED[0]} (vertical fault, {mesh_preset} mesh, order {elem_order}, aligned SFSVM v26-0, "
         f"nugget Cd, surface constraint ON); companion {run_tag(COMPANION[1])} / {COMPANION[0]} (07a full Cd)",
         f"PyLith runs: {run_names}; {n_imp} impulses on {len(fault_xyz)} fault vertices; {samples.vel.size} samples",
         f"validation (a) PyLith homog vs CSI Meade: rel rms {100 * val_a['rel_rms']:.2f} % (agreeing sides {100 * val_a['rel_rms_okside']:.2f} %, "
         f"far field {val_a['rms_diff_far_mm']:.3f} mm); (b) vs 2D screw: {100 * val_b['rel_rms']:.2f} %; "
         f"(c) superposition homog: {val_c['rms_diff_mm']:.2e} mm rms; (d) fine - default mesh: {val_d['fine_minus_default_rms_mm']:.4f} mm rms; "
         f"forward-vs-G at the samples: " + ", ".join(forward_vs_g),
         f"fault-adjacent mu, top 100 m: west {mu_top100['west_gpa']:.3f}, east {mu_top100['east_gpa']:.3f} GPa; side mismatches {side_mismatch.sum()}",
         f"surface % denominator: median of {n_section_crossings} measured crossings in the SSE section {section_crossing_median_mm:.3f} mm",
         f"06 field fp_maps.npz fingerprint {field_fingerprint} (asserted on every cached driver results table read here)",
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
REG.render()
print("08a_3d_model complete")
