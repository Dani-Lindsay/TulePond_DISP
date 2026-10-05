# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
# ---

# %% [markdown]
# # PyLith 3D Green's functions with USGS SFCVM v26 for the Tule Pond mean event
#
# ```
# Created on Tue Sep 29 2026 (overnight agent run, DRAFT)
# @author: daniellelindsay
# ```
#
# A standalone draft (07 is not changed). It builds 3D finite-element Green's functions with **PyLith 5.0.2**
# for the vertical fault under the mapped creeping trace, in two media (plus a sensitivity medium):
#
# 1. **homogeneous**: mu = 30 GPa, Poisson 0.25 (the CSI baseline's rock);
# 2. **SFCVM v26 aligned**: USGS SFCVM v26.0 with its Hayward block boundary shifted onto the mapped trace.
#
# 3. (sensitivity) **SFCVM v26 as is**: not shifted.
#
# and inverts the same fault-parallel mean-event data as `scripts/07_slip_model.py` with CSI, like for like.
#
# | Step | Tool (validated library) | Glue written here |
# |---|---|---|
# | mesh | gmsh via PyLith `gmsh_utils.GenerateMesh` (`generate_gmsh_vertical.py`) | size fields |
# | materials | SFCVM read with h5py (logic of `drafts/07/sfcvm_extract_dense.py`); spatialdata `SimpleGridAscii` writer | alignment shift, caps |
# | Green's functions | PyLith `GreensFns` + `FaultCohesiveImpulses` + `OutputSolnPoints` (`cfg/`) | impulse-region db |
# | G assembly | reading pattern of PyLith `examples/subduction-3d/invert_slip.py` (`build_G.py`) | vertex/point matching, projection, sign |
# | inversion | CSI `TriangularTents`, `setGFs`, `buildCmXY`, `transformation`, `multisourcesolve` NNLS, `surfaceslip` (as 07) | surface-constraint mapping to vertices |
#
# Two environments: this notebook runs in `opera_disp_feb13` (CSI, pyproj, scipy); the PyLith steps are shelled out
# to the PyLith 5.0.2 bundle (`pylith_env.sh`, clean env, `setup.sh` sourced in bash). PyLith outputs are
# re-used when present (cache keyed by run name: delete `output/<name>-*.h5` to recompute).

# %%
## [1]
# ------------------------
# Paths & constants
# ------------------------
import copy
import json
import os
import re
import subprocess
import sys
import time

import h5py
import numpy as np
import pandas as pd
import pyproj
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib.tri as mtri
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from scipy.spatial import cKDTree

REPO = "/Users/dlindsay/tools/internal/TulePond_DISP"
HERE = os.path.join(REPO, "drafts", "pylith")
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
OPERA_PY = sys.executable
CRS = "+proj=tmerc +datum=WGS84 +lon_0=-121.968 +lat_0=37.557 +k=1 +units=m +type=crs"   # mesh CRS = CSI frame

mesh_preset = os.environ.get("PYLITH_MESH_PRESET", "default")   # generate_gmsh_vertical.py preset: default = 100 m fault
                                        # cells in the Tule Pond patch; "shallow" adds a 60 m near-surface band (sensitivity)
n_procs = 8                             # MPI ranks per PyLith run
impulse_y_m = (-4713.0, 4380.0)         # impulses on km 54-65 from Point Pinole (y of km 65, km 54 on the trace)
impulse_depth_m = 3000.0                # whole fault depth (bottom edge buried -> no impulse there)
media = {"homog": "cfg/mat_homog.cfg", "sfcvm": "cfg/mat_sfcvm.cfg",      # the comparison: homogeneous vs SFCVM aligned
         "sfcvmasis": "cfg/mat_sfcvm_asis.cfg",                           # sensitivity: SFCVM as is (boundary 35-215 m east of the trace)
         "sfcvmv": "cfg/mat_sfcvm_dip90v.cfg",                            # control: vertical fault, contrast held ON the fault at all depths
         "sfcvm_e50": "cfg/mat_sfcvm_e50.cfg", "sfcvm_w50": "cfg/mat_sfcvm_w50.cfg"}   # alignment sensitivity: boundary 50 m E / W
if "PYLITH_MEDIA" in os.environ:        # e.g. PYLITH_MEDIA=homog,sfcvm for the shallow-mesh sensitivity run
    media = {m: media[m] for m in os.environ["PYLITH_MEDIA"].split(",")}
dip = os.environ.get("PYLITH_DIP", "90")                          # 90 = vertical (generate_gmsh_vertical.py); "80e", "70e", "80w" =
                                                                  # generate_gmsh_dip.py (top edge on the trace, dip along the mean normal)
assert dip == "90" or (dip[:-1].isdigit() and dip[-1] in "ew"), f"PYLITH_DIP {dip!r}"
dip_tag = "" if dip == "90" else f"_dip{dip}"
if dip != "90":                                                   # SFCVM aligned to the dipping fault (sfcvm_grid.py --dip)
    media = {m: (f"cfg/mat_sfcvm{dip_tag}.cfg" if m == "sfcvm" else c) for m, c in media.items()}
    assert "sfcvmasis" not in media, "the as-is SFCVM is a vertical-fault sensitivity only"
elem_order = os.environ.get("PYLITH_ORDER", "1")                        # 2: quadratic solution + linear materials + SFCVM without nu cap
assert elem_order in ("1", "2")                                        # (cfg/order2.cfg, run_gf_o2.sh; Charles Williams 2026-09-30)
order_tag = "" if elem_order == "1" else "_order2"
if elem_order == "2" and "sfcvm" in media:
    assert dip == "90", "order-2 runs are set up for the vertical fault only"
    media["sfcvm"] = "cfg/mat_sfcvm_nocap.cfg"
points_set = os.environ.get("PYLITH_POINTS", "default")           # default: 07's 2386 samples (swath 4 km); swath7: make_points_swath.py 7
assert points_set in ("default", "swath7", "coh07")              # coh07: 07's quadtree (swath 4 km) on 06's field at map coherence 0.7
                                                                  # (coh07/make_fp_coh.py + make_points_swath.py 4 ... coh07; 3148 samples)
points_tag = "" if points_set == "default" else f"_{points_set}"
points_file = os.path.join(HERE, "points", "insar_points" + points_tag)   # .txt (PyLith PointsList) / .npz
run_names = {m: f"gf_{mesh_preset}{dip_tag}_{m}{points_tag}{order_tag}" for m in media}
use_surface = os.environ.get("PYLITH_SURFACE", "1") == "1"       # 0: drop the crossing-stack surface constraint (InSAR only)
cd_mode = os.environ.get("PYLITH_CD", "full")                     # full: 07's correlated-noise Cd; diag: equal independent weights;
                                                                  # floor: 07's Cd with n_px floored at PYLITH_CD_FLOOR (phase-linking window)
assert cd_mode in ("full", "diag", "floor", "nugget")   # nugget: 07's per-block nugget variances only (diagonal, no correlation)
cd_floor = int(os.environ.get("PYLITH_CD_FLOOR", "15")) if cd_mode == "floor" else None   # (115 m / 30 m)^2 ~ 15 pixels
cd_tag = {"full": "", "diag": "_diagcd", "floor": f"_floor{cd_floor}", "nugget": "_nuggetcd"}[cd_mode]
align_tag = "".join(f"_{m[6:]}" for m in media if m.startswith("sfcvm_"))   # alignment-sensitivity media (e.g. _e50), so
                                                                  # their products never overwrite the default run's
out_tag = (("" if mesh_preset == "default" else f"_{mesh_preset}") + dip_tag + points_tag + order_tag + align_tag
           + ("" if use_surface else "_nosurf") + cd_tag)         # suffix of every product of this run


def sigma_locked_08a():
    """The model sigma locked in scripts/08a_3d_model.py (Dani 2026-10-03): the literal on its `sigma_locked = ...`
    line, read here so it is typed in one place only."""
    found = re.findall(r"^sigma_locked\s*=\s*([0-9.]+)", open(os.path.join(REPO, "scripts", "08a_3d_model.py")).read(), re.M)
    assert len(found) == 1, f"08a_3d_model.py: {len(found)} `sigma_locked = <number>` lines, expected 1"
    return float(found[0])


SIGMA_LOCKED_08A = sigma_locked_08a()
sigma_env = os.environ.get("PYLITH_SIGMA", "")                   # "" = each run's own L-curve pick (the rule of 07a [8]);
sigma_fixed = {"": None, "locked": SIGMA_LOCKED_08A}.get(sigma_env)   # "locked" = 08a's sigma_locked for every run (the
if sigma_env not in ("", "locked"):                               # sensitivity runs, Dani 2026-10-05); or a number
    sigma_fixed = float(sigma_env)
print(f"run: mesh {mesh_preset}, dip {dip}, points {points_set}, order {elem_order}, surface {use_surface}, Cd {cd_mode} (floor {cd_floor}); "
      f"sigma {'L-curve pick' if sigma_fixed is None else f'fixed {sigma_fixed:g}'}; tag {out_tag!r}")
mu_ref = 3.0e10                         # 30 GPa reference rigidity (07)
test_km, test_depth_m, test_slip_mm = (57.0, 61.0), 1000.0, 10.0      # validation (a) slip patch
plt.rcParams.update({"font.size": 12, "savefig.facecolor": "white", "figure.facecolor": "white"})
okabe = dict(blue="#0072B2", orange="#E69F00", vermillion="#D55E00", black="#000000")
timings = {}


def sh(cmd, cwd=HERE):
    """Run a shell command, echo it, fail loudly."""
    print("$", cmd)
    t0 = time.time()
    out = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    assert out.returncode == 0, f"command failed ({out.returncode}):\n{out.stdout[-3000:]}\n{out.stderr[-3000:]}"
    print(out.stdout[-1500:])
    return time.time() - t0


to_mesh = pyproj.Transformer.from_crs("EPSG:4326", CRS, always_xy=True)

# %% [markdown]
# ## [2] Mesh
#
# `generate_gmsh_vertical.py` (PyLith `gmsh_utils.GenerateMesh`): the mapped trace (every 3rd 100 m point, km
# 52-67) as a spline, extruded 3 km straight down, fragmented into a 100 x 100 x 40 km box. Cell size from
# three linear-growth fields combined by a gmsh Min field (a gmsh Box field, and any MathEval that references
# another MathEval, hang gmsh 4.15.2's 1D stage). Preset `default`: 100 m on the fault in km 56-63 / top 1.5 km,
# 250 m elsewhere on the fault, 250 m on the surface over the data, bias 1.15.

# %%
## [2]
mesh_file = os.path.join(HERE, "mesh", f"mesh_vertical_tet_{mesh_preset}.msh" if dip == "90" else f"mesh_dip{dip}_tet_{mesh_preset}.msh")
if not os.path.exists(mesh_file):
    if dip == "90":
        timings["mesh"] = sh(f"bash pylith_env.sh {HERE} python3 generate_gmsh_vertical.py --write --preset {mesh_preset}")
    else:
        timings["mesh"] = sh(f"bash pylith_env.sh {HERE} python3 generate_gmsh_dip.py --write --preset {mesh_preset} --dip {dip}")
print(open(mesh_file.replace(".msh", "_report.txt")).read())
fault_tri = np.load(mesh_file.replace(".msh", "_fault.npz"))["xyz"]      # (n_tri, 3 vertices, xyz) in m

fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
ax = axes[0]
yz = fault_tri[:, :, [1, 2]] / 1e3
ax.add_collection(PolyCollection(yz, facecolors="none", edgecolors="0.3", linewidths=0.3))
ax.axvspan(-3.1, 2.8, ymin=0.5, color=okabe["orange"], alpha=0.15, label="fine patch (km 56-63, top 1.5 km)")
ax.axvline(impulse_y_m[0] / 1e3, color=okabe["vermillion"], ls="--"); ax.axvline(impulse_y_m[1] / 1e3, color=okabe["vermillion"], ls="--",
                                                                                  label="impulse region (km 54-65)")
ax.set_xlim(-6.6, 6.2); ax.set_ylim(-3.05, 0.05)
ax.set_xlabel("y north (km, tmerc) -- south <- km 67 ... km 52 -> north"); ax.set_ylabel("z (km)")
ax.set_title(f"(a) PyLith fault mesh ({mesh_preset}): {len(fault_tri)} triangles, projected on y-z", loc="left")
ax.legend(loc="lower left", fontsize=10)
ax = axes[1]
edge = np.load(mesh_file.replace(".msh", "_fault.npz"))["edge_len"]
cen = fault_tri.mean(1)
sc = ax.scatter(cen[:, 1] / 1e3, cen[:, 2] / 1e3, c=edge, s=4, cmap="cividis", vmin=50, vmax=300)
fig.colorbar(sc, ax=ax, label="mean edge length (m)")
ax.set_xlim(-6.6, 6.2); ax.set_ylim(-3.05, 0.05); ax.set_xlabel("y north (km)"); ax.set_ylabel("z (km)")
ax.set_title("(b) fault cell size", loc="left")
fig.savefig(os.path.join(FIG, f"fig_i_mesh_fault{out_tag}.png"), dpi=300)

# %% [markdown]
# ## [3] Output points and materials
#
# The 2386 quadtree samples of 07 (`fig2_model_row.npz`) in the mesh CRS (`make_points.py`). SFCVM v26.0 on a
# non-uniform grid (50 m horizontally around the Tule Pond section, 25 m vertically to 500 m), aligned so that
# its Hayward block boundary ({22|23}|7) at the surface lies on the mapped trace (a local cross-fault shift b(km),
# full within 3 km of the trace, tapered to zero at 6 km; dip kept), vs floored at 150 m/s, Poisson's ratio
# capped at 0.40 (`sfcvm_grid.py`), written as a spatialdata SimpleGridDB by spatialdata's own writer
# (`write_sfcvm_griddb.py`). The unshifted ("as is") grid is written too, for the figure.

# %%
## [3]
if not os.path.exists(os.path.join(HERE, "points", "insar_points.txt")):
    sh(f"{OPERA_PY} make_points.py")
if points_set == "coh07":
    assert os.path.exists(points_file + ".npz"), "run coh07/make_fp_coh.py 0.9, 0.7 and make_points_swath.py 4 coh07/coh07/fp_maps.npz coh07 first"
if points_set == "swath7" and not os.path.exists(points_file + ".npz"):
    sh(f"{OPERA_PY} drafts/pylith/make_points_swath.py 7", cwd=REPO)
for grid_tag, flag in (("aligned", ""), ("asis", "--no-align")):
    if not os.path.exists(os.path.join(HERE, "spatialdb", f"sfcvm_v26-0_{grid_tag}_grid.npz")):
        timings[f"sfcvm_grid_{grid_tag}"] = sh(f"{OPERA_PY} drafts/pylith/sfcvm_grid.py {flag}", cwd=REPO)
griddb = os.path.join(HERE, "spatialdb", "mat_sfcvm_v26-0_aligned.spatialdb")
if not os.path.exists(griddb):
    timings["griddb"] = sh(f"bash pylith_env.sh {HERE} python3 write_sfcvm_griddb.py "
                           f"spatialdb/sfcvm_v26-0_aligned_grid.npz spatialdb/mat_sfcvm_v26-0_aligned.spatialdb")
if dip != "90" and "sfcvm" in media:                              # SFCVM block boundary aligned to the DIPPING fault at every depth
    grid_dip = os.path.join(HERE, "spatialdb", f"sfcvm_v26-0_aligned{dip_tag}_grid.npz")
    if not os.path.exists(grid_dip):
        timings[f"sfcvm_grid{dip_tag}"] = sh(f"{OPERA_PY} drafts/pylith/sfcvm_grid.py --dip {dip}", cwd=REPO)
    if not os.path.exists(griddb.replace("aligned", f"aligned{dip_tag}")):
        timings[f"griddb{dip_tag}"] = sh(f"bash pylith_env.sh {HERE} python3 write_sfcvm_griddb.py "
                                         f"spatialdb/sfcvm_v26-0_aligned{dip_tag}_grid.npz spatialdb/mat_sfcvm_v26-0_aligned{dip_tag}.spatialdb")
    assert os.path.exists(os.path.join(HERE, media["sfcvm"])), f"missing {media['sfcvm']}"
sh(f"{OPERA_PY} plot_materials.py")
points = np.load(points_file + ".npz")
grid_file = os.path.join(HERE, "spatialdb", f"sfcvm_v26-0_aligned{dip_tag}{'_nocap' if elem_order == '2' else ''}_grid.npz")
if elem_order == "2" and "sfcvm" in media and not os.path.exists(grid_file):
    timings["sfcvm_grid_nocap"] = sh(f"{OPERA_PY} drafts/pylith/sfcvm_grid.py --no-cap", cwd=REPO)
    timings["griddb_nocap"] = sh(f"bash pylith_env.sh {HERE} python3 write_sfcvm_griddb.py "
                                 f"spatialdb/sfcvm_v26-0_aligned_nocap_grid.npz spatialdb/mat_sfcvm_v26-0_aligned_nocap.spatialdb")
grid = np.load(grid_file)
print(grid["provenance"])

# %% [markdown]
# ## [4] PyLith Green's functions, all media
#
# `GreensFns` with `FaultCohesiveImpulses`: unit left-lateral impulses (right-lateral = -left-lateral, sign derived
# in [5]) on every fault vertex of km 54-65 over the full 3 km depth (buried edges excluded automatically), the
# rest of the fault locked; Dirichlet zero on the sides and bottom, free top; displacement output at the samples.
# Slip basis order 1 (tent per vertex), as the PyLith examples require for inversion.

# %%
## [4]
impulse_db = "spatialdb/impulses_km54-65_z3000.spatialdb"          # y window = impulse_y_m, depth impulse_depth_m
if not os.path.exists(os.path.join(HERE, impulse_db)):
    sh(f"bash pylith_env.sh {HERE} python3 make_impulse_db.py {impulse_y_m[0]} {impulse_y_m[1]} {impulse_depth_m} {impulse_db}")
for medium, mat_cfg in media.items():
    name = run_names[medium]
    if not os.path.exists(os.path.join(HERE, "output", f"{name}-points.h5")):
        timings[f"pylith_{medium}"] = sh(f"bash {'run_gf.sh' if elem_order == '1' else 'run_gf_o2.sh'} {name} {mat_cfg} {os.path.relpath(mesh_file, HERE)} "
                                         f"{impulse_db} {n_procs} {os.path.relpath(points_file + '.txt', HERE)}")
    prog = open(os.path.join(HERE, "output", f"{name}-progress.txt")).read().strip().splitlines()
    print(f"{name}: progress first/last:\n  {prog[1]}\n  {prog[-1]}")

# %% [markdown]
# ## [5] G matrices
#
# `build_G.py` (pattern of PyLith `invert_slip.py`): impulse -> vertex (unit tent asserted), output point ->
# sample (KD-tree, < 1 mm asserted), 3D displacement projected on 07's fault-parallel look vector, sign derived
# (right-lateral must move the east side along +look, as the data). Both media must share the fault mesh and the
# impulse order.

# %%
## [5]
import build_G
G, gnpz = {}, {}
for medium in media:
    path = os.path.join(HERE, "gf", f"G_{run_names[medium]}.npz")
    if not os.path.exists(path):
        build_G.main([run_names[medium]], points_file=points_file + ".npz")
    gnpz[medium] = np.load(path)
    # the cached G is projected on the strike it was built with: rebuild it (from the cached PyLith outputs, no PyLith
    # run) when the current field's strike differs (2026-10-05: the along-trace section moved the strike 146.40 -> 146.51)
    strike_now = float(np.load("/Users/dlindsay/work/proj_TulePond_DISP/outputs/07b_uniform_model_tests/fig2_model_row.npz")["strike_deg"])   # 07a's strike of the current field (what build_G projects on)
    if abs(float(gnpz[medium]["strike_deg"]) - strike_now) > 1e-6:
        print(f"G_{run_names[medium]}: built at strike {float(gnpz[medium]['strike_deg']):.4f}, field at {strike_now:.4f} -> rebuilding")
        build_G.main([run_names[medium]], points_file=points_file + ".npz")
        gnpz[medium] = np.load(path)
    assert abs(float(gnpz[medium]["strike_deg"]) - strike_now) < 1e-6, f"G_{run_names[medium]} strike differs from the field's"
    G[medium] = gnpz[medium]["G_rl_m_per_m"]
ref_medium = list(media)[0]                                  # order-2 SFCVM can be inverted before order-2 homog exists
for medium in media:
    assert np.array_equal(gnpz[ref_medium]["vertex_index"], gnpz[medium]["vertex_index"]), "impulse order differs between media"
    assert np.array_equal(gnpz[ref_medium]["fault_xyz"], gnpz[medium]["fault_xyz"])
fault_xyz, fault_cells = gnpz[ref_medium]["fault_xyz"], gnpz[ref_medium]["fault_cells"]
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
# ### CHECK -- PyLith read the SFCVM material as intended
#
# The per-cell shear modulus PyLith actually used (`-elastic_info.h5`, basis order 0) against the aligned grid
# interpolated at the cell centroids, for cells in the fine grid box; and mu of the cells next to the fault
# (nearest cell centroid on each side of every fault triangle), used for the SFCVM moment.

# %%
## [5b]
def mu_beside_fault(medium, grid_npz):
    """PyLith cell mu (elastic_info) vs the grid it was read from; mu of the nearest cell on each side of every
    fault triangle (mean of the two sides returned)."""
    with h5py.File(os.path.join(HERE, "output", f"{run_names[medium]}-elastic_info.h5"), "r") as h:
        cells = np.asarray(h["viz/topology/cells"][:], int)
        cell_xyz = h["geometry/vertices"][:][cells].mean(1)
        if "cell_fields" in h and "shear_modulus" in h["cell_fields"]:
            cell_mu = h["cell_fields/shear_modulus"][0, :, 0]                      # materials basis order 0
        else:
            cell_mu = h["vertex_fields/shear_modulus"][0, :, 0][cells].mean(1)     # basis order 1: cell mean of the vertex values
    g = np.load(grid_npz)
    grid_mu = RegularGridInterpolator((g["x"], g["y"], g["z"]), g["density"].astype(float) * g["vs"].astype(float) ** 2)
    fine = (cell_xyz[:, 0] > -3500) & (cell_xyz[:, 0] < 2500) & (cell_xyz[:, 1] > -4000) & (cell_xyz[:, 1] < 3600) & (cell_xyz[:, 2] > -1500)
    rel = cell_mu[fine] / grid_mu(cell_xyz[fine]) - 1.0
    print(f"{medium}: PyLith cell mu vs grid at centroids ({fine.sum()} cells in the fine box, top 1.5 km): median |rel diff| "
          f"{np.median(np.abs(rel)):.3f}, 90th pct {np.percentile(np.abs(rel), 90):.3f}; PyLith mu range "
          f"{cell_mu.min() / 1e9:.3f}-{cell_mu.max() / 1e9:.1f} GPa")
    assert cell_mu.min() >= 0.99 * float(np.min(g["density"].astype(float) * g["vs"].astype(float) ** 2)), "mu below the floor"
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
    return 0.5 * (cell_mu[east_i] + cell_mu[west_i]), cell_mu[west_i], cell_mu[east_i]


mu_fault = {"sfcvm": mu_beside_fault("sfcvm", grid_file)[0]} if "sfcvm" in media else {}
for medium in media:                                        # alignment-sensitivity media sfcvm_<tag> (e.g. sfcvm_e50, boundary 50 m east)
    if medium.startswith("sfcvm_"):
        mu_fault[medium] = mu_beside_fault(medium, os.path.join(HERE, "spatialdb", f"sfcvm_v26-0_aligned_{medium[6:]}_grid.npz"))[0]
if "sfcvmv" in media:
    mu_fault["sfcvmv"] = mu_beside_fault("sfcvmv", os.path.join(HERE, "spatialdb", "sfcvm_v26-0_aligned_dip90v_grid.npz"))[0]
if "sfcvmasis" in media:
    mu_fault["sfcvmasis"] = mu_beside_fault("sfcvmasis", os.path.join(HERE, "spatialdb", "sfcvm_v26-0_asis_grid.npz"))[0]

# %% [markdown]
# ## [6] 07's data and machinery
#
# `scripts/07a_uniform_model.py` cells [1]-[6] exec'd unchanged (as 08a does): data, quadtree (2386 samples), noise
# covariance, the measured crossings, the surface % denominator `crossing_median` (re-derived here from the shared
# helper and asserted equal), the CSI triangle fault and `invert`. The samples must be the PyLith output points.
# Every product carries the fingerprint of the 06 field it fits (`field_fingerprint`); for the swath-7 and
# coherence-0.7 sample sets, whose data live in their points npz, that npz must be newer than 06's field.

# %%
## [6]
source_07 = open(os.path.join(REPO, "scripts", "07a_uniform_model.py")).read()      # renamed from 07_slip_model.py 2026-09-30
head_07 = source_07.split("# %% [markdown]\n# ### CHECK")[0]
for old, new in (("REG = utils.Registry(notebook, common_paths, repo_dir)", "REG = None"), ("REG.inputs(", "inputs_07a = (")):
    assert head_07.count(old) == 1, f"07a_uniform_model.py layout changed: {old!r} not found once"   # as 08a: no registry writes from here
    head_07 = head_07.replace(old, new)
exec(head_07)
field_fp_file = os.path.join(field_dir, "fp_maps.npz")              # the 06 field every run fits (directly or via its points npz)
field_fingerprint = f"fp_maps.npz:{utils.file_fingerprint(field_fp_file)}"   # registry form; never parsed as a number
if points_set in ("swath7", "coh07"):                    # their data live in the points npz (make_points_swath.py)
    assert os.path.getmtime(points_file + ".npz") > os.path.getmtime(field_fp_file) \
        and str(points["field_fingerprint"] if "field_fingerprint" in points.files else "") == field_fingerprint, \
        (f"{points_file}.npz was made from another (older) 06 field; re-make it (models_overnight_2026-10-05.sh step 2)")
row07 = np.load(os.path.join(tests_dir, "fig2_model_row.npz"))
fault07_trace = copy.deepcopy(fault)                    # 07's fault (trace), for maps
if points_set in ("swath7", "coh07"):                   # make_points_swath.py: 07's cells [1]-[4] with swath_km 7 (2396 samples) or on
                                                        # the coherence-0.7 field (3148 samples; its own variogram, cov_p in the npz)
    samples = insar(f"Downsampled fault-parallel {points_set}", lon0=frame_lon0, lat0=frame_lat0, verbose=False)
    samples.read_from_binary(points["vel_mm"], points["lon"], points["lat"], err=np.ones(points["lon"].size),
                             los=np.tile(look, (points["lon"].size, 1)), remove_nan=False, remove_zeros=False)
    samples.lon, samples.lat = points["lon"].copy(), points["lat"].copy()      # CSI stores lon in 0-360; keep the points' -122 form
    assert np.allclose(samples.x * 1e3, points["x"], atol=1e-3) and np.allclose(samples.y * 1e3, points["y"], atol=1e-3), \
        f"CSI frame differs from the mesh CRS for the {points_set} samples"
    samples.wgt = points["wgt"].copy()
    build_cd(samples, points["cov_p"])                  # 07's build_cd with the swath-7 variogram parameters
    print(f"{points_set} samples: {samples.vel.size}; noise nugget {np.sqrt(points['cov_p'][0]):.3f} mm")
assert samples.lon.size == points["lon"].size and np.array_equal(samples.lon, points["lon"]) \
    and np.array_equal(samples.lat, points["lat"]), "the quadtree samples are not the PyLith output points"
block_lon, block_lat = (row07["block_lon"], row07["block_lat"]) if points_set == "default" else (points["block_lon"], points["block_lat"])
from plot_run_summary import signed_distance_m           # one definition, shared with the one-page diagnostic


signed_d_km = signed_distance_m(fault07_trace.xi * 1e3, fault07_trace.yi * 1e3, points["x"], points["y"]) / 1e3
print(f"07 baseline: sigma {float(row07['sigma']):g}, rms {float(row07['rms_mm']):.3f} mm, Mw {float(row07['mw_30gpa']):.3f}")

# the PyLith mesh fault (a polyline through its top-edge vertices, which lie on the spline through every 3rd trace
# point) vs 07's trace (20 m discretisation): are any samples on different sides?
top = fault_xyz[np.abs(fault_xyz[:, 2]) < 1e-6]; top = top[np.argsort(top[:, 1])]
dense_s = np.linspace(0, 1, 20000)
top_arc = np.r_[0, np.cumsum(np.hypot(np.diff(top[:, 0]), np.diff(top[:, 1])))]; top_arc /= top_arc[-1]
top_dense_x, top_dense_y = np.interp(dense_s, top_arc, top[:, 0]), np.interp(dense_s, top_arc, top[:, 1])
signed_mesh_m = signed_distance_m(top_dense_x, top_dense_y, points["x"], points["y"])
in_mesh = (points["y"] > top[0, 1]) & (points["y"] < top[-1, 1])
side_mismatch = in_mesh & (np.sign(signed_mesh_m) != np.sign(signed_d_km))
print(f"samples on opposite sides of the mesh fault and of 07's trace: {side_mismatch.sum()} "
      f"(largest distance from the mesh fault {np.abs(signed_mesh_m[side_mismatch]).max() if side_mismatch.any() else 0:.0f} m); "
      f"|mesh - 07| signed distance: median {np.median(np.abs(signed_mesh_m[in_mesh] - 1e3 * signed_d_km[in_mesh])):.1f} m, "
      f"max {np.abs(signed_mesh_m[in_mesh] - 1e3 * signed_d_km[in_mesh]).max():.1f} m")

# %% [markdown]
# ## [7] Validation
#
# (a) Homogeneous PyLith against CSI's homogeneous triangular dislocations (Meade) for the same slip: 10 mm
# uniform on km 57-61, top 1 km. PyLith: 10 mm on every impulse vertex in the patch (so slip tapers to zero over
# one cell outside it). CSI: the PyLith piecewise-linear slip field interpolated (in y, z) to the centroid of every
# triangle of 07's mesh, so both carry the same slip. (b) Homogeneous PyLith against the 2D screw dislocation
# u = (s/pi) arctan(D/x) for uniform slip on km 54-65, top 1 km, at samples within 2 km of the trace and at
# least 3 km from the slip ends (D = 1 km + half a fault cell for the tent taper).

# %%
## [7]
imp_xyz = fault_xyz[vertex_index]
top_km = lat_to_km_pp(to_mesh.transform(top[:, 0], top[:, 1], direction="INVERSE")[1])    # km of the mesh trace (top sorted by y)
km_of_y = lambda y: np.interp(np.asarray(y), top[:, 1], top_km)
imp_km = km_of_y(imp_xyz[:, 1])
slip_test = np.where((imp_km >= test_km[0]) & (imp_km <= test_km[1]) & (imp_xyz[:, 2] >= -test_depth_m - 1e-6), test_slip_mm, 0.0)
has_homog = "homog" in G                                    # validations (a), (b), (d) are homogeneous-medium checks
pred_pylith = G["homog" if has_homog else ref_medium].dot(slip_test)   # (a)/(b) below are only meaningful with homog
vertex_slip = np.zeros(len(fault_xyz)); vertex_slip[vertex_index] = slip_test
tri_yz = mtri.Triangulation(fault_xyz[:, 1], fault_xyz[:, 2], fault_cells)
interp_slip = mtri.LinearTriInterpolator(tri_yz, vertex_slip)
fault_val = wrap_fault(mesh_nodes, mesh_faces, dip_deg, "validation")
csi_cen = np.array([p.mean(0) for p in fault_val.patch])              # km, depth +
csi_slip = np.asarray(interp_slip(csi_cen[:, 1] * 1e3, -csi_cen[:, 2] * 1e3).filled(0.0))
line = insar("validation", lon0=frame_lon0, lat0=frame_lat0, verbose=False)
line.read_from_binary(np.zeros(samples.lon.size), samples.lon, samples.lat, err=np.ones(samples.lon.size),
                      los=np.tile(look, (samples.lon.size, 1)), remove_nan=False, remove_zeros=False)
fault_val.buildGFs(line, vertical=True, slipdir="s", verbose=False)
pred_csi = -fault_val.G[line.name]["strikeslip"].dot(csi_slip)         # right-lateral positive (07 convention)
diff = pred_pylith - pred_csi
near_300 = np.abs(signed_d_km) <= 0.3
val_a = dict(max_abs_diff_mm=float(np.abs(diff).max()), rms_diff_mm=float(np.sqrt(np.mean(diff ** 2))),
             rms_signal_mm=float(np.sqrt(np.mean(pred_csi ** 2))),
             rms_diff_near300_mm=float(np.sqrt(np.mean(diff[near_300] ** 2))),
             rms_diff_far_mm=float(np.sqrt(np.mean(diff[~near_300] ** 2))),
             potency_pylith_m3=float(np.sum(tri_area * vertex_slip[fault_cells].mean(1) * 1e-3)),
             potency_csi_m3=float(np.sum(cell_area_km2 * 1e6 * csi_slip * 1e-3)))
val_a["rel_rms"] = val_a["rms_diff_mm"] / val_a["rms_signal_mm"]
print(("validation (a) PyLith homog vs CSI:" if has_homog else f"validation (a) SKIPPED (no homog medium in this run; numbers below are {ref_medium} vs CSI homog, not a check):"),
      {k: round(v, 4) for k, v in val_a.items()})

# (b) 2D screw dislocation
slip_b = np.where(imp_xyz[:, 2] >= -test_depth_m - 1e-6, test_slip_mm, 0.0)
pred_b = G["homog" if has_homog else ref_medium].dot(slip_b)
patch_cells = (np.abs(tri_cen[:, 1] - 0) < 3000) & (tri_cen[:, 2] > -1500)
dx_fault_m = float(np.median(np.linalg.norm(tri_xyz[patch_cells][:, [1, 2, 0]] - tri_xyz[patch_cells], axis=2)))
D_eff = test_depth_m + 0.5 * dx_fault_m
x_m = signed_mesh_m                                                  # distance from the MESH fault (the source)
km_samples = lat_to_km_pp(samples.lat)
sel_b = (np.abs(x_m) <= 2000) & (np.abs(x_m) >= 20) & (km_samples >= 57.0) & (km_samples <= 62.0)
screw = (test_slip_mm / np.pi) * np.arctan(D_eff / x_m)
diff_b = pred_b[sel_b] - screw[sel_b]
val_b = dict(n=int(sel_b.sum()), D_eff_m=D_eff, rms_diff_mm=float(np.sqrt(np.mean(diff_b ** 2))),
             max_abs_diff_mm=float(np.abs(diff_b).max()), rms_signal_mm=float(np.sqrt(np.mean(screw[sel_b] ** 2))))
val_b["rel_rms"] = val_b["rms_diff_mm"] / val_b["rms_signal_mm"]
print("validation (b) PyLith homog vs 2D screw dislocation:", {k: round(v, 4) for k, v in val_b.items()})

fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), constrained_layout=True)
ax = axes[0]
ax.scatter(pred_csi, pred_pylith, s=6, c=np.where(near_300, okabe["vermillion"], okabe["blue"]), linewidths=0)
lim = np.abs(np.r_[pred_csi, pred_pylith]).max() * 1.05
ax.plot([-lim, lim], [-lim, lim], "k-", lw=0.8); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
ax.set_xlabel("CSI homogeneous (mm)"); ax.set_ylabel("PyLith homogeneous (mm)")
ax.set_title(f"(a) 10 mm on km 57-61, top 1 km: rms diff {val_a['rms_diff_mm']:.3f} mm ({100 * val_a['rel_rms']:.1f}%)", loc="left", fontsize=11)
ax.plot([], [], "o", color=okabe["vermillion"], label="within 300 m of trace"); ax.plot([], [], "o", color=okabe["blue"], label="farther")
ax.legend(fontsize=10)
ax = axes[1]
ax.scatter(signed_d_km, diff, s=6, color=okabe["black"], linewidths=0)
ax.axhline(0, color="0.5", lw=0.8); ax.set_xlabel("Distance from 07 trace, east + (km)"); ax.set_ylabel("PyLith - CSI (mm)")
ax.set_title("(a) difference vs distance", loc="left", fontsize=11)
ax = axes[2]
order_b = np.argsort(x_m[sel_b])
ax.plot(x_m[sel_b][order_b] / 1e3, screw[sel_b][order_b], "-", color=okabe["orange"], lw=2, label=f"2D screw, D = {D_eff:.0f} m")
ax.scatter(x_m[sel_b] / 1e3, pred_b[sel_b], s=8, color=okabe["blue"], label="PyLith homogeneous", zorder=3)
ax.set_xlabel("Distance from mesh fault, east + (km)"); ax.set_ylabel("Fault-parallel (mm)")
ax.set_title(f"(b) km 54-65 top 1 km, samples km 57-62: rms diff {val_b['rms_diff_mm']:.3f} mm ({100 * val_b['rel_rms']:.1f}%)", loc="left", fontsize=11)
ax.legend(fontsize=10)
fig.savefig(os.path.join(FIG, f"fig_iii_validation{out_tag}.png"), dpi=300)

# %% [markdown]
# ### [7b] Validation with forward runs (superposition and mesh convergence)
#
# PyLith `FaultCohesiveKin` forward runs (`cfg/fwd_kin.cfg`, one static step) with 10 mm right-lateral on km
# 57-61 (y window), top 1 km, homogeneous, on the `default` and the `fine` mesh (60 m fault cells in the patch).
# (c) superposition: default-mesh forward = G x (the forward run's own vertex slip) -- tests G assembly, point
# matching and sign; (d) mesh convergence: the fine-mesh forward against CSI with the same (fine) slip field, and
# against the default-mesh forward.

# %%
## [7b]
fwd_ok = points_set == "default" and dip == "90" and has_homog   # the forward runs are on the vertical meshes at the 2386 samples
kin_db = "spatialdb/kin_slip_km57-61_z1000.spatialdb"
if not os.path.exists(os.path.join(HERE, kin_db)):
    sh(f"bash pylith_env.sh {HERE} python3 make_kin_slip_db.py -1479 1943 {test_depth_m} {test_slip_mm / 1e3} {kin_db}")
fwd = {}
for pre in (("default", "fine") if fwd_ok else ()):
    name = f"fwd_{pre}_homog"
    if not os.path.exists(os.path.join(HERE, "output", f"{name}-points.h5")):
        if pre == "fine" and not os.path.exists(os.path.join(HERE, "mesh", "mesh_vertical_tet_fine.msh")):
            sh(f"bash pylith_env.sh {HERE} python3 generate_gmsh_vertical.py --write --preset fine")
        timings[name] = sh(f"bash pylith_env.sh {HERE} pylith cfg/pylithapp.cfg cfg/mat_homog.cfg cfg/fwd_kin.cfg "
                           f"--problem.defaults.name={name} "
                           f"--problem.mesh_initializer.phases.read_mesh.reader.filename=mesh/mesh_vertical_tet_{pre}.msh --nodes=2")
    with h5py.File(os.path.join(HERE, "output", f"{name}-points.h5"), "r") as h:
        dist, order = cKDTree(h["geometry/vertices"][:, :2]).query(np.c_[points["x"], points["y"]])
        assert dist.max() < 1e-3
        u_fp = h["vertex_fields/displacement"][-1][order].dot(look) * 1e3            # mm
    with h5py.File(os.path.join(HERE, "output", f"{name}-fault.h5"), "r") as h:
        f_xyz, f_cells = h["geometry/vertices"][:], np.asarray(h["viz/topology/cells"][:], int)
        slip_rl = float(gnpz["homog"]["sign_rl"]) * h["vertex_fields/slip"][-1][:, 1] * 1e3    # RL = sign_rl x LL (build_G)
    fwd[pre] = dict(u=u_fp, xyz=f_xyz, cells=f_cells, slip=slip_rl)
    print(f"{name}: slip {slip_rl.min():.2f}..{slip_rl.max():.2f} mm on {np.sum(slip_rl > 1e-6)} vertices; "
          f"max |u| {np.abs(u_fp).max():.2f} mm")
assert all(np.all(fwd[pre]["slip"] > -1e-6) for pre in fwd), "forward slip is not right-lateral"
if not fwd_ok:
    print(f"validation (c)/(d) skipped: forward runs exist only for the vertical default/fine meshes at the 2386 samples")

# (c) superposition on the default mesh (the forward runs are on the default and fine meshes, so this
# check only applies when the GFs are also from the default mesh; skipped for other presets)
def superposition(medium, fwd_run):
    """G[medium] x (the forward run's own vertex slip) against that forward run's displacement at the samples: exact
    (1e-6 mm) when G and the forward run share mesh, medium and element order."""
    with h5py.File(os.path.join(HERE, "output", f"{fwd_run}-points.h5"), "r") as h:
        dist, order = cKDTree(h["geometry/vertices"][:, :2]).query(np.c_[points["x"], points["y"]])
        assert dist.max() < 1e-3
        u_fp = h["vertex_fields/displacement"][-1][order].dot(look) * 1e3
    with h5py.File(os.path.join(HERE, "output", f"{fwd_run}-fault.h5"), "r") as h:
        f_xyz = h["geometry/vertices"][:]
        slip_rl = float(gnpz[medium]["sign_rl"]) * h["vertex_fields/slip"][-1][:, 1] * 1e3
    d_v, i_v = cKDTree(f_xyz).query(imp_xyz)
    assert d_v.max() < 1e-6, "forward-run fault vertices differ from the GF fault vertices"
    assert np.setdiff1d(np.nonzero(slip_rl > 1e-6)[0], i_v).size == 0, "forward slip on vertices without an impulse"
    sup = G[medium].dot(slip_rl[i_v])
    d_sup = sup - u_fp
    out = dict(max_abs_diff_mm=float(np.abs(d_sup).max()), rms_diff_mm=float(np.sqrt(np.mean(d_sup ** 2))),
               rms_signal_mm=float(np.sqrt(np.mean(sup ** 2))))
    print(f"validation (c) superposition, G[{medium}] x slip vs forward run {fwd_run}:", {k: round(v, 6) for k, v in out.items()})
    return out


val_c = None
if mesh_preset == "default" and points_set == "default" and dip == "90":
    fwd_tag = "" if elem_order == "1" else "_o2"                     # order-2 GFs are checked against the P2 forward runs
    val_c = {}
    for medium in media:
        fwd_run = f"fwd_default{fwd_tag}_{medium}"
        if os.path.exists(os.path.join(HERE, "output", f"{fwd_run}-points.h5")):
            val_c[medium] = superposition(medium, fwd_run)
            rel_c = val_c[medium]["rms_diff_mm"] / val_c[medium]["rms_signal_mm"]
            if elem_order == "1":
                assert val_c[medium]["rms_diff_mm"] < 1e-4, f"superposition failed for {medium}: G is not consistent with {fwd_run}"
            else:                                            # order 2 (2026-09-30): NOT exact -- the P2 kinematic forward run
                assert rel_c < 0.05, f"order-2 superposition off by {100 * rel_c:.1f} % for {medium}"   # evaluates its "nearest"
                print(f"WARNING order-2 superposition for {medium}: rms diff {val_c[medium]['rms_diff_mm']:.3f} mm = "   # slip db at the P2
                      f"{100 * rel_c:.1f} % of the signal (order 1: exact); the difference sits at the patch edges, "     # edge nodes, so its
                      "where the P2 forward run's patch is sharper than the tent sum; accepted below 5 %, recorded")     # patch edge is sharper
        else:
            print(f"validation (c) superposition: no forward run {fwd_run} for {medium}; skipped")
else:
    print(f"validation (c) superposition: skipped for mesh preset {mesh_preset!r} (forward run is on the default mesh)")


def csi_prediction(f_xyz, f_cells, slip_mm):
    """CSI Meade TDE prediction at the samples for a PyLith piecewise-linear slip field (interpolated in y-z to
    07's triangle centroids)."""
    tri = mtri.Triangulation(f_xyz[:, 1], f_xyz[:, 2], f_cells)
    s_cen = np.asarray(mtri.LinearTriInterpolator(tri, slip_mm)(csi_cen[:, 1] * 1e3, -csi_cen[:, 2] * 1e3).filled(0.0))
    return -fault_val.G[line.name]["strikeslip"].dot(s_cen)


ok_side = ~side_mismatch
val_d = {}
for pre in fwd:
    diff_pre = fwd[pre]["u"] - csi_prediction(fwd[pre]["xyz"], fwd[pre]["cells"], fwd[pre]["slip"])
    val_d[pre] = dict(rms_diff_mm=float(np.sqrt(np.mean(diff_pre[ok_side] ** 2))),
                      rms_diff_near300_mm=float(np.sqrt(np.mean(diff_pre[ok_side & near_300] ** 2))),
                      rms_diff_far_mm=float(np.sqrt(np.mean(diff_pre[~near_300] ** 2))),
                      max_abs_diff_mm=float(np.abs(diff_pre[ok_side]).max()))
    print(f"validation (d) forward {pre} mesh vs CSI (same slip, {ok_side.sum()} samples on agreeing sides):",
          {k: round(v, 4) for k, v in val_d[pre].items()})
if fwd_ok:
    d_mesh = fwd["fine"]["u"] - fwd["default"]["u"]
    print(f"validation (d) fine - default forward: rms {np.sqrt(np.mean(d_mesh ** 2)):.4f} mm, within 300 m "
          f"{np.sqrt(np.mean(d_mesh[near_300] ** 2)):.4f} mm, farther {np.sqrt(np.mean(d_mesh[~near_300] ** 2)):.4f} mm")
    val_d["fine_minus_default_rms_mm"] = float(np.sqrt(np.mean(d_mesh ** 2)))

fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)
for ax, pre in zip(axes, fwd):
    cpred = csi_prediction(fwd[pre]["xyz"], fwd[pre]["cells"], fwd[pre]["slip"])
    ax.scatter(signed_d_km[ok_side], (fwd[pre]["u"] - cpred)[ok_side], s=6, color=okabe["black"], linewidths=0)
    ax.axhline(0, color="0.5", lw=0.8); ax.set_xlim(-1, 1); ax.set_ylim(-2, 2)
    ax.set_xlabel("Distance from trace, east + (km)"); ax.set_ylabel("PyLith forward - CSI (mm)")
    ax.set_title(f"(d) {pre} mesh: rms {val_d[pre]['rms_diff_mm']:.3f} mm, within 300 m {val_d[pre]['rms_diff_near300_mm']:.3f} mm",
                 loc="left", fontsize=11)
fig.savefig(os.path.join(FIG, f"fig_iii_validation_mesh{out_tag}.png"), dpi=300)

# %% [markdown]
# ## [8] The CSI fault on the PyLith vertices, and the inversion
#
# PyLith impulses are tents on fault vertices, not CSI triangles, so both PyLith media are inverted on the vertex
# parameterisation with CSI's `TriangularTents` (its `tent` list and `Vertices` = the impulse vertices, in G's
# column order; faces = the PyLith fault triangles among them). The G matrix enters through `Fault.setGFs`.
# Everything else is 07's `invert`: exponential prior `buildCmXY` with lambda (0.1, 0.05) km on the vertex
# positions, per-dataset constant (`csi.transformation`), NNLS (`multisourcesolve.ConstrainedLeastSquareSoln`),
# the same data covariance. The surface constraint: 07's crossing-stack running median (`make_surface`)
# interpolated to every top-edge impulse vertex, errors scaled by sqrt(vertices per 0.2 km) as 07 does for its
# triangles, with G = 1 on that vertex's own slip (the tent equals the slip at its vertex).
# NB: `buildCmXY` normalises by lambda0 from the parameter spacing, so sigma is not numerically comparable with
# 07's triangle sigma; each medium gets its own L-curve corner, with the same rule as 07 [8]. With PYLITH_SIGMA
# set (`locked` = 08a's sigma_locked, the sensitivity runs from 2026-10-05) the model is solved at that sigma
# instead; the L-curve is still computed and saved, its pick recorded as `lcurve_pick` / `sigma_lcurve_pick`.
#
# **Data covariance (`PYLITH_CD`).** The choice of Cd is the one thing that changes between the runs on this
# fault; it is a standard modelling choice, not new physics:
# - `full` (the paper's, 07 [4]): Cd from the semivariogram of the mean-event field away from the fault, nugget +
#   exponential terms; the nugget is divided down by each quadtree block's pixel count and the exponential terms
#   give the OFF-diagonal (correlated) part. Estimating Cd from the far-field semivariogram and carrying the
#   correlated part into the inversion follows Lohman and Simons (2005) and Sudhaus and Jonsson (2009).
# - `nugget`: the same per-block nugget variances on the diagonal, correlated part dropped: samples treated as
#   independent, weights scaled by block size. This is the quadtree convention of Jonsson et al. (2002) and most
#   InSAR slip inversions since. Under it the west lobe must be fitted (aligned SFCVM does; homogeneous cannot),
#   but the potency roughly doubles and the reduced chi2 (~5.6) shows the variances are too small: the correlated
#   Cd is the statistically consistent one. Report `full` as primary and `nugget` as the sensitivity.
# - `diag` (equal weights) and `floor` (nugget floor at the phase-linking window) are diagnostics only.
#
# References (author, year, journal; check the DOI before citing):
# Lohman, R. B., and M. Simons (2005), Some thoughts on the use of InSAR data to constrain models of surface
#   deformation: Noise structure and data downsampling, Geochem. Geophys. Geosyst., 6, Q01007.
# Sudhaus, H., and S. Jonsson (2009), Improved source modelling through combined use of InSAR and GPS under
#   consideration of correlated data errors: application to the June 2000 Kleifarvatn earthquake, Iceland,
#   Geophys. J. Int., 176, 389-404.
# Jonsson, S., H. Zebker, P. Segall, and F. Amelung (2002), Fault slip distribution of the 1999 Mw 7.1 Hector Mine,
#   California, earthquake, estimated from satellite radar and GPS measurements, Bull. Seismol. Soc. Am., 92,
#   1377-1389.

# %%
## [8]
from csi import TriangularTents

imp_lon, imp_lat = to_mesh.transform(imp_xyz[:, 0], imp_xyz[:, 1], direction="INVERSE")
imp_set = {v: i for i, v in enumerate(vertex_index)}
sub_faces = np.array([[imp_set[v] for v in tri] for tri in fault_cells if all(v in imp_set for v in tri)])
surface_imp = np.nonzero(np.abs(imp_xyz[:, 2]) < 1e-6)[0]


def tent_fault_template():
    """CSI TriangularTents on the impulse vertices (km, depth +), in G's column order (built once: its adjacency
    map is slow; each solve deep-copies it, as 07 builds a fresh fault per solve)."""
    flt = TriangularTents("tents", lon0=frame_lon0, lat0=frame_lat0, verbose=False)
    x_km, y_km = flt.ll2xy(imp_lon, imp_lat)
    assert np.allclose(x_km * 1e3, imp_xyz[:, 0], atol=1e-3) and np.allclose(y_km * 1e3, imp_xyz[:, 1], atol=1e-3), \
        "CSI frame differs from the mesh CRS"
    flt.Vertices = np.c_[x_km, y_km, -imp_xyz[:, 2] / 1e3]
    flt.Vertices_ll = np.c_[imp_lon, imp_lat, -imp_xyz[:, 2] / 1e3]
    flt.Faces = sub_faces.copy()
    flt.patch = [flt.Vertices[f] for f in flt.Faces]
    flt.patchll = [flt.Vertices_ll[f] for f in flt.Faces]
    flt.vertices2tents()
    flt.initializeslip(n=n_imp)
    assert len(flt.tent) == n_imp and flt.N_slip == n_imp
    return flt


t0 = time.time()
TENTS = tent_fault_template()
print(f"CSI TriangularTents: {len(TENTS.tent)} tents, {len(TENTS.Faces)} faces ({time.time() - t0:.0f} s)")


def tent_fault(name):
    flt = copy.deepcopy(TENTS)
    flt.name = name
    return flt


sys.path.insert(0, os.path.join(REPO, "scripts", "pylith_helpers"))
from surface_crossings import crossing_constraint, surface_pct_denominator_mm   # surface constraint (2026-10-02) and the one
                                                                                 # surface % denominator (2026-10-05), shared with 07a/08a/08b
top_order = surface_imp[np.argsort(imp_xyz[surface_imp, 1])]          # top-edge impulse vertices, south to north (mesh y)
assert np.all(np.diff(imp_xyz[top_order, 1]) > 0), "top-edge vertices not monotone in mesh y"
crossings_in = crossings_all[(crossings_all.km >= km_window[0]) & (crossings_all.km <= km_window[1])]
assert abs(surface_pct_denominator_mm(crossings_in.km.values, crossings_in.mean_event_mm.values, creep_lo_km, creep_hi_km)
           - crossing_median) < 1e-12, "07a's crossing_median is not the median of the measured crossings in the SSE section"


def surface_constraint(floor_mm=surf_floor_mm):
    """The measured crossings, each tied to the tent slip at its own top-edge position (surface_crossings.py;
    until 2026-10-02: 07's running median interpolated to every vertex, which lost half its windows to float roundoff)."""
    sc = crossing_constraint(crossings_in.km.values, crossings_in.lat.values, crossings_in.mean_event_mm.values,
                             lambda la: np.interp(la, trace_south_north[:, 1], trace_south_north[:, 0]),
                             lambda lo, la: to_mesh.transform(lo, la), imp_xyz[top_order, :2], top_order, n_imp, floor_mm=floor_mm)
    ss = surfaceslip(f"crossings f{floor_mm:g}", lon0=frame_lon0, lat0=frame_lat0, verbose=False)
    ss.read_from_binary(sc["vel"], sc["lon"], sc["lat"], err=sc["err"])
    ss.buildCd()
    return ss, sc["G"]


def invert_tents(medium, sigma, lam=lam_xz, vel_insar=None, vel_surface=None):
    """07's invert() with PyLith G on the tent parameters (setGFs instead of buildGFs). Returns fault, data, surface.
    vel_insar / vel_surface: replacement data vectors (bootstrap and recovery tests, pylith_sfcvm_tests.py)."""
    flt = tent_fault(f"{medium} sigma {sigma}")
    ds = copy.deepcopy(samples)
    if vel_insar is not None:
        ds.vel[:] = vel_insar
    if cd_mode == "diag":                                    # PYLITH_CD=diag: independent samples, equal variance (the 2D test's
        ds.Cd = np.eye(ds.vel.size) * float(np.median(np.diag(samples.Cd)))   # weighting; diagnostic only, Dani 2026-09-30)
    elif cd_mode == "nugget":                                # PYLITH_CD=nugget: 07's diagonal (nugget / n_px^0.5) without the correlated terms
        cp = cov_p if points_set == "default" else points["cov_p"]
        ds.Cd = np.diag(np.diag(samples.Cd) - (cp[1] + cp[3]))                # (diag of full Cd = nugget term + s1 + s2 at zero lag)
        assert np.all(np.diag(ds.Cd) > 0)
    elif cd_mode == "floor":                                 # PYLITH_CD=floor: 07's build_cd, but a block's pixel count is floored at
        ds.wgt = np.maximum(np.asarray(samples.wgt, float), cd_floor)         # cd_floor (DISP-S1 pixels inside one ~115 m phase-linking
        build_cd(ds, cov_p if points_set == "default" else points["cov_p"])   # window are not independent) -> nugget / max(n_px, floor)^0.5
    surface, G_surf = surface_constraint()
    if vel_surface is not None:
        surface.vel[:] = vel_surface
    flt.setGFs(ds, strikeslip=[G[medium].copy()])            # already right-lateral positive (build_G sign)
    flt.setGFs(surface, strikeslip=[G_surf])
    solve_datas = [ds, surface] if use_surface else [ds]     # PYLITH_SURFACE=0: InSAR only (Dani 2026-09-30)
    trans = transform("ramps", lon0=frame_lon0, lat0=frame_lat0, verbose=False)
    trans.buildGFs(solve_datas, ["1", "1"], verbose=False)
    flt.assembleGFs(solve_datas, slipdir="s", verbose=False); flt.assembled(solve_datas)
    flt.assembleCd(solve_datas)
    trans.assembleGFs(solve_datas); trans.assembled(solve_datas)
    trans.assembleCd(solve_datas)
    flt.buildCmXY(sigma, lam, verbose=False)
    trans.buildCm(1e4, verbose=False)
    solver = multisrc(f"{medium} mean event", [flt, trans], verbose=False)
    solver.assembleGFs(); solver.assembleCm()
    mprior = np.zeros(solver.Np)
    ramp_idx = [i for i, t in enumerate(solver.paramTypes) if t[0] == trans.name]
    mprior[ramp_idx] = -1e3
    solver.ConstrainedLeastSquareSoln(method="nnls", mprior=mprior)
    solver.distributem()
    for d in solve_datas:
        d.buildsynth([flt], direction="s")
        d.removeTransformation(trans)
    if not use_surface:                                      # not inverted, but still predicted (reported as a check)
        surface.buildsynth([flt], direction="s")
    return flt, ds, surface


def potency_m3(slip_mm):
    """Exact integral of the piecewise-linear (tent) slip over the PyLith fault triangles (locked vertices = 0)."""
    v = np.zeros(len(fault_xyz)); v[vertex_index] = slip_mm
    return float(np.sum(tri_area * v[fault_cells].mean(1)) * 1e-3)


def moment_sfcvm(slip_mm, medium="sfcvm"):
    """sum over fault triangles of area x mean tent slip x mu (mean of the PyLith cells either side) of `medium`
    (the aligned SFCVM for the homogeneous model: a rescale, as 07's Mw_sfcvm)."""
    if not mu_fault:                                         # no SFCVM medium in this run (e.g. a homog-only smoke test)
        return np.nan
    v = np.zeros(len(fault_xyz)); v[vertex_index] = slip_mm
    return float(np.sum(tri_area * v[fault_cells].mean(1) * 1e-3 * mu_fault[medium if medium in mu_fault else 'sfcvm']))


mw = lambda m0: (2.0 / 3.0) * (np.log10(m0) - 9.1)

# %% [markdown]
# ## [9] L-curve for each medium (07 [8] rule)
#
# Same sigma grid, same weighted misfit sqrt(chi2 InSAR + chi2 surface) against the prior-weighted roughness,
# corner by maximum curvature of cubic splines in log sigma (end intervals excluded), L-curve pick = smallest grid
# sigma at or above the corner; selected = the pick, or the fixed sigma (PYLITH_SIGMA) when one is given.

# %%
## [9]
def lcurve(medium):
    rows = []
    for sigma in sigma_grid:
        t0 = time.time()
        flt, ds, surface = invert_tents(medium, sigma)
        r_i, r_s = ds.vel - ds.synth, surface.vel - surface.synth
        slip_vec = flt.slip[:, 0]
        rows.append(dict(sigma=sigma, chi2=float(np.dot(r_i, np.linalg.solve(ds.Cd, r_i))),
                         chi2_surf=float(np.dot(r_s, np.linalg.solve(surface.Cd, r_s))),
                         prior_norm=float(np.sqrt(np.dot(slip_vec, np.linalg.solve(flt.Cm / sigma ** 2, slip_vec)))),
                         rms_mm=float(np.sqrt(np.mean(r_i ** 2))), peak_slip_mm=float(slip_vec.max()),
                         Mw_30=mw(mu_ref * potency_m3(slip_vec)), seconds=time.time() - t0))
        rows[-1]["chi2_total"] = rows[-1]["chi2"] + rows[-1]["chi2_surf"]
        print(f"  {medium} sigma {sigma:>4g}: chi2 {rows[-1]['chi2']:.0f} + {rows[-1]['chi2_surf']:.0f}, "
              f"roughness {rows[-1]['prior_norm']:.1f}, rms {rows[-1]['rms_mm']:.3f}, Mw30 {rows[-1]['Mw_30']:.2f} "
              f"({rows[-1]['seconds']:.0f} s)")
    fam = pd.DataFrame(rows)
    log_sigma = np.log10(fam.sigma.values)
    s_mis = CubicSpline(log_sigma, np.log10(np.sqrt(fam.chi2_total.values)))
    s_rough = CubicSpline(log_sigma, np.log10(fam.prior_norm.values))
    fine_ls = np.linspace(log_sigma[1], log_sigma[-2], 400)
    d1, dd1, d2, dd2 = s_mis(fine_ls, 1), s_mis(fine_ls, 2), s_rough(fine_ls, 1), s_rough(fine_ls, 2)
    curvature = np.abs(d1 * dd2 - d2 * dd1) / (d1 ** 2 + d2 ** 2) ** 1.5
    corner = float(10 ** fine_ls[np.argmax(curvature)])
    pick = float(fam.sigma[fam.sigma >= corner].min())
    selected = pick if sigma_fixed is None else sigma_fixed
    fam["corner"], fam["lcurve_pick"], fam["selected"] = corner, pick, selected
    fam["sigma_rule"] = "L-curve pick" if sigma_fixed is None else f"fixed (PYLITH_SIGMA={sigma_env})"
    fam.to_csv(os.path.join(HERE, "gf", f"lcurve_{medium}{out_tag}.csv"), index=False)
    print(f"{medium}: L-curve corner sigma {corner:.1f} -> pick {pick:g}; model solved at sigma {selected:g}"
          + ("" if selected == pick else f" (fixed: PYLITH_SIGMA={sigma_env}; NOT the pick)"))
    return fam, corner, selected


assert sigma_fixed is None or sigma_fixed in sigma_grid, f"fixed sigma {sigma_fixed:g} is not on the grid {sigma_grid} (08b needs its neighbours)"
families = {}
for medium in media:
    families[medium] = lcurve(medium)

fig, axes = plt.subplots(1, len(media), figsize=(6.5 * len(media), 5), constrained_layout=True)
for ax, medium, colour in zip(np.atleast_1d(axes), media, (okabe["blue"], okabe["vermillion"], "#CC79A7")):
    fam, corner, selected = families[medium]
    ax.plot(np.sqrt(fam.chi2_total), fam.prior_norm, "o-", color=colour)
    for s, xv, yv in zip(fam.sigma, np.sqrt(fam.chi2_total), fam.prior_norm):
        ax.annotate(f"{s:g}", (xv, yv), xytext=(3, 2), textcoords="offset points", fontsize=9)
    pick = fam.sigma == selected
    ax.plot(np.sqrt(fam.chi2_total[pick]), fam.prior_norm[pick], "s", ms=10, mfc="none", mec="k", label=f"selected sigma {selected:g}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.legend()
    ax.set_xlabel(r"weighted misfit $\sqrt{\chi^2}$"); ax.set_ylabel(r"roughness $\sqrt{m^T C^{-1} m}$")
    ax.set_title(f"L-curve, PyLith {medium} (corner {corner:.1f})", loc="left")
fig.savefig(os.path.join(FIG, f"fig_lcurves{out_tag}.png"), dpi=300)

# %% [markdown]
# ## [10] The selected models and their numbers
#
# rms, rms within 300 m west / east of 07's trace, chi2 (InSAR, full covariance), surface slip against the
# crossings (median over top-edge vertices in the SSE section / median of the measured crossings there, the one
# denominator `crossing_median`), potency (exact tent integral), Mw at 30 GPa and with the SFCVM rigidity of the
# PyLith cells beside the fault; and the provenance columns: the sigma used, the L-curve pick and corner, the
# sigma rule, the 06 field fingerprint, the time. The 07 CSI baseline is re-scored from `fig2_model_row.npz` on
# the same samples (07a quotes Mw at 30 GPa only: Mw_sfcvm empty).

# %%
## [10]
models, table = {}, []
west = (signed_d_km < 0) & (signed_d_km >= -0.3)
east = (signed_d_km > 0) & (signed_d_km <= 0.3)
res07 = row07["residual_mm"]
made = time.strftime("%Y-%m-%d %H:%M")
provenance_cols = dict(field_fingerprint=field_fingerprint, made=made)
if points_set == "default":
  table.append(dict(model="CSI homogeneous (07)", sigma=float(row07["sigma"]),
                  rms_mm=float(np.sqrt(np.mean(res07 ** 2))), rms_west300_mm=float(np.sqrt(np.mean(res07[west] ** 2))),
                  rms_east300_mm=float(np.sqrt(np.mean(res07[east] ** 2))),
                  chi2=float(np.dot(res07, np.linalg.solve(samples.Cd, res07))),
                  surface_pct=float(row07["surf_ratio_pct"]), potency_m3=np.nan,
                  Mw_30gpa=float(row07["mw_30gpa"]), Mw_sfcvm=np.nan, sigma_lcurve_pick=np.nan, sigma_corner=np.nan,
                  sigma_rule="07a L-curve pick", **provenance_cols))
for medium in media:
    fam, corner, selected = families[medium]
    flt, ds, surface = invert_tents(medium, selected)
    slip_vec = flt.slip[:, 0].copy()
    res = ds.vel - ds.synth
    in_sec = np.isin(np.arange(n_imp), surface_imp) & (imp_km >= creep_lo_km) & (imp_km <= creep_hi_km)
    pot = potency_m3(slip_vec)
    models[medium] = dict(slip=slip_vec, data=ds.vel.copy(), model=ds.synth.copy(), residual=res,
                          surface_km=lat_to_km_pp(surface.lat), surface_model=surface.synth.copy(), surface_obs=surface.vel.copy())
    table.append(dict(model=f"PyLith {medium}", sigma=selected, rms_mm=float(np.sqrt(np.mean(res ** 2))),
                      rms_west300_mm=float(np.sqrt(np.mean(res[west] ** 2))), rms_east300_mm=float(np.sqrt(np.mean(res[east] ** 2))),
                      chi2=float(np.dot(res, np.linalg.solve(ds.Cd, res))),
                      surface_pct=float(100 * np.median(slip_vec[in_sec]) / crossing_median),
                      potency_m3=pot, Mw_30gpa=mw(mu_ref * pot), Mw_sfcvm=mw(moment_sfcvm(slip_vec, medium)),
                      peak_slip_mm=float(slip_vec.max()), peak_depth_m=float(-imp_xyz[np.argmax(slip_vec), 2]),
                      sigma_lcurve_pick=float(fam.lcurve_pick.iloc[0]), sigma_corner=corner, sigma_rule=str(fam.sigma_rule.iloc[0]),
                      **provenance_cols))
results = pd.DataFrame(table).set_index("model")
results.to_csv(os.path.join(HERE, "gf", f"results_comparison{out_tag}.csv"))
print(results.round(3).T.to_string())

# %% [markdown]
# ## [11] Figures: Fig-2-style maps, slip, X-X' profile

# %%
## [11]
block_poly = np.stack([block_lon, block_lat], axis=-1)
trace_plot = np.c_[fault07_trace.lon, fault07_trace.lat]
rows_maps = ([("CSI homogeneous (07)", row07["data_mm"], row07["model_mm"], row07["residual_mm"])] if points_set == "default" else []) + \
            [(f"PyLith {m}", models[m]["data"], models[m]["model"], models[m]["residual"]) for m in media]
fig, axes = plt.subplots(len(rows_maps), 3, figsize=(15, 5.3 * len(rows_maps)), squeeze=False, constrained_layout=True)
for r, (label, d, m, res) in enumerate(rows_maps):
    for c, (vals, title) in enumerate(((d, "data"), (m, "model"), (res, "residual"))):
        ax = axes[r, c]
        pc = PolyCollection(block_poly, array=vals, cmap="RdBu_r", edgecolors="none")
        pc.set_clim(-4, 4); ax.add_collection(pc)
        ax.plot(trace_plot[:, 0], trace_plot[:, 1], "k-", lw=0.6)
        ax.set_xlim(-122.08, -121.86); ax.set_ylim(37.47, 37.63); ax.set_aspect(1 / np.cos(np.deg2rad(frame_lat0)))
        rms_txt = f" (rms {np.sqrt(np.mean(res ** 2)):.2f} mm)" if c == 2 else ""
        ax.set_title(f"{label}: {title}{rms_txt}", loc="left", fontsize=11)
        ax.set_xticks([-122.05, -121.95], ["-122.05", "-121.95"]); ax.set_yticks([37.50, 37.55, 37.60], ["37.50", "37.55", "37.60"])
        ax.tick_params(labelsize=10)
fig.colorbar(pc, ax=axes, shrink=0.5, orientation="horizontal", label="fault-parallel displacement (mm)")
fig.savefig(os.path.join(FIG, f"fig_iv_fit_maps{out_tag}.png"), dpi=300)

imp_s_km = imp_km
fig, axes = plt.subplots(len(media), 1, figsize=(13, 4 * len(media)), squeeze=False, constrained_layout=True)
axes = axes[:, 0]
vmax = max(models[m]["slip"].max() for m in media)
tri_imp = mtri.Triangulation(imp_s_km, imp_xyz[:, 2] / 1e3, sub_faces)
for ax, medium in zip(axes, media):
    tp = ax.tripcolor(tri_imp, models[medium]["slip"], shading="gouraud", cmap="magma", vmin=0, vmax=vmax)
    r = results.loc[f"PyLith {medium}"]
    ax.set_xlim(54, 65); ax.set_ylim(-1.5, 0); ax.invert_xaxis()
    ax.set_xlabel("km from Point Pinole (orange bar: SSE section)"); ax.set_ylabel("z (km)")
    ax.set_title(f"PyLith {medium}: sigma {r.sigma:g}, rms {r.rms_mm:.2f} mm, Mw {r.Mw_30gpa:.2f} (30 GPa), "
                 f"{r.Mw_sfcvm:.2f} (SFCVM mu)", loc="left", fontsize=11)
    ax.hlines(0.04, creep_lo_km, creep_hi_km, color=okabe["orange"], lw=5, clip_on=False)   # SSE section (04)
fig.colorbar(tp, ax=axes, label="right-lateral slip (mm)", shrink=0.8)
fig.savefig(os.path.join(FIG, f"fig_v_slip{out_tag}.png"), dpi=300)

fig2_in = np.load(os.path.join(field_dir, "fig2_inputs.npz"))
to_field = pyproj.Transformer.from_crs(4326, int(fig2_in["epsg"]), always_xy=True)
xw, yw, xe, ye = fig2_in["prof_fp_X_ends"]
u = np.array([xe - xw, ye - yw]); length = np.linalg.norm(u); u /= length
sx, sy = to_field.transform(samples.lon, samples.lat)
along = (sx - xw) * u[0] + (sy - yw) * u[1]
perp = -(sx - xw) * u[1] + (sy - yw) * u[0]
sel = (np.abs(perp) <= 300) & (along >= 0) & (along <= length)
order_p = np.argsort(signed_d_km[sel])
fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
ax.plot(fig2_in["prof_fp_X_d"] / 1e3, fig2_in["prof_fp_X_v"], "-", color="0.7", lw=1, label="06 X-X' pixel profile (far field zeroed)")
ax.plot(signed_d_km[sel], models[list(media)[0]]["data"][sel], "o", color=okabe["black"], ms=4, label="data at quadtree samples within 300 m")
for (label, _, m, _), colour in zip(rows_maps, (okabe["blue"], okabe["orange"], okabe["vermillion"], "#CC79A7")):
    ax.plot(signed_d_km[sel][order_p], m[sel][order_p], "-", color=colour, lw=2, label=label)
ax.axvline(0, color="k", lw=0.6); ax.set_xlim(-2.5, 2.5)
ax.set_xlabel("distance from the trace, east + (km)"); ax.set_ylabel("fault-parallel (mm)")
ax.set_title("X-X' (lat ~37.5612 N): data vs models at the samples within 300 m of the line", loc="left")
ax.legend(fontsize=10)
fig.savefig(os.path.join(FIG, f"fig_vi_profile_XX{out_tag}.png"), dpi=300)

# %% [markdown]
# ## [12] Products and provenance

# %%
## [12]
np.savez(os.path.join(HERE, "gf", f"inversion_products{out_tag}.npz"),
         impulse_vertex_xyz=imp_xyz, impulse_km=imp_km, sub_faces=sub_faces,
         **{f"slip_{m}_mm": models[m]["slip"] for m in media}, **{f"model_{m}_mm": models[m]["model"] for m in media},
         **{f"residual_{m}_mm": models[m]["residual"] for m in media}, sample_lon=samples.lon, sample_lat=samples.lat,
         data_mm=models[list(media)[0]]["data"], signed_d_km=signed_d_km, block_lon=block_lon, block_lat=block_lat,
         trace_lon=fault07_trace.lon, trace_lat=fault07_trace.lat,
         settings=json.dumps(dict(mesh=mesh_preset + ("" if elem_order == "1" else " order2 (P2 solution, P1 materials, no nu cap)"),
                                  dip=("90 (vertical)" if dip == "90" else f"{dip[:-1]} {dip[-1].upper()} (top edge on the trace)"),
                                  cd_mode=cd_mode, cd_floor=cd_floor, use_surface=use_surface,
                                  points=f"{points_set} ({samples.vel.size} samples, " + {"default": "07 quadtree swath 4 km", "swath7": "swath 7 km",
                                                                                          "coh07": "coh 0.7 field, swath 4 km"}[points_set] + ")",
                                  run_names=run_names, n_impulses=int(n_imp), mesh_file=os.path.basename(mesh_file),
                                  sigma_rule="L-curve pick" if sigma_fixed is None else f"fixed {sigma_fixed:g} (PYLITH_SIGMA={sigma_env})",
                                  field_fingerprint=field_fingerprint)),
         validation_a=pd.Series(val_a).to_json(), validation_b=pd.Series(val_b).to_json(), validation_c=json.dumps(val_c),
         results=results.to_json(), provenance=f"pylith_sfcvm_gf.py {made}; runs {run_names}; 06 field {field_fingerprint}; "
                                               f"mesh {mesh_file}; {grid['provenance']}")
import plot_run_summary                                   # the one-page diagnostic per medium (from the products just written)
plot_run_summary.main(out_tag, list(media))
if os.path.exists(os.path.join(HERE, "plot_test_summary.py")):   # where the selected model sits among all runs (coordinator, 2026-09-30)
    try:                                                   # a summary-plot failure must not discard this run's products (written above)
        sh(f"{OPERA_PY} plot_test_summary.py")
    except AssertionError as err:
        print(f"WARNING: plot_test_summary.py failed; products of {out_tag!r} are written, the summary figure is stale:\n{err}")
print("timings (s):", {k: round(v) for k, v in timings.items()})
print("done")
