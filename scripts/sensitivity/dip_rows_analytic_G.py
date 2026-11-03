"""dip_rows_analytic_G.py: the homogeneous (analytic) Green's-function matrices of the fault-dip sweep (dip_rows.py).

Author: Danielle Lindsay

For each of the seven geometries of dip_rows.py (vertical, 80 E / 80 W, 70 E / 70 W on the default mesh with a PyLith
run; 60 W and 50 W on the default mesh with no PyLith run) this computes the Green's functions of the tents in a
homogeneous elastic half-space (Poisson solid) with CSI triangular dislocations (Meade 2007; model_utils.analytic_tent_G),
each fault triangle split into n_sub^2 constant-slip sub-triangles, n_sub = dip_rows.N_SUB (8, the value notebook 08
uses for its homogeneous model).
  - PyLith-run geometries: the tents of the run's G (common.make_setup, as dip_rows.py reads them), so the columns are
    in the impulse order of the run.
  - 60 W and 50 W: the tents are rebuilt from the default mesh alone (modeling_inputs/mesh/mesh_dip<dip>_tet_default_fault.npz):
    fault vertices = the corners of the mesh fault triangles (rounded to 1 mm, sorted by np.unique), impulse vertices =
    PyLith's impulse rule (pylith/build/make_impulse_db.py: y_lo <= y <= y_hi and z >= -depth, the limits read from the
    impulse database model_config.gf_files["impulse_db"], the database of the default-mesh runs too). The rule is
    checked on the five PyLith geometries: the rebuilt impulse vertices must equal the PyLith run's.
The samples, the look vector and the CSI frame are those of dip_rows.py (notebook 07's model_data.npz and its default
point set).

Checks, per geometry: right-lateral uniform slip on every tent moves the east side along +look (as in 08); for a PyLith
geometry, the columns agree in sign with the PyLith G (> 99 %, as dip_rows.py asserts). Against the earlier n_sub 4
matrix of the same geometry (G_analytic_<run>_nsub4.npy / G_analytic_dip<dip>_nsub4.npz, made 26 Oct 2026 by
dip_checks_2026-10-26/{dip_inversions,analytic_west_scan}.py), when that file exists: the same shape (and for 60 W /
50 W the same fault and impulse vertices), the columns agree in sign (> 99 %), and the relative difference per column
||g8 - g4|| / ||g8|| over the samples (median and max) is printed and written to the table below. A stored matrix is
read, not recomputed (an interrupted run keeps the finished geometries).

Inputs: notebook 07's model_data.npz and default point set; the PyLith G npz of the five runs
(modeling_SFCVM/gf); the default-mesh fault files of 60 W and 50 W; the impulse database.
Outputs (modeling_uniform/gf_dip_rows/): G_analytic_<run>_nsub8.npy (PyLith-run geometries; n_samples x n_impulses,
m per m of right-lateral slip), G_analytic_dip{60w,50w}_nsub8.npz (the npz layout of model_utils.build_G, so
common.make_setup reads it with run_dir=(gf_dip_rows, gf_dip_rows)), analytic_G_nsub8_check.csv (one row per geometry).
Run (repo root), one process per geometry, at most --processes at a time (each about 2 GB of memory):
    OMP_NUM_THREADS=2 PYTHONPATH=. nice -n 10 python scripts/sensitivity/dip_rows_analytic_G.py [geometry ...] [--processes 4]
Run time: 2 h 49 min for the seven geometries with 4 processes (1 Nov 2026): 73-82 min per PyLith-run
geometry, 95 min (60 W) and 106 min (50 W); peak memory 2.1 GB per process.
"""
import argparse
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common                                         # noqa: E402
import dip_rows                                       # noqa: E402  (the geometries, the n_sub and the output folder)
from common import cfg, model_utils                   # noqa: E402
from TulePond_filepaths import paths_inputs          # noqa: E402

N_SUB = dip_rows.N_SUB
N_SUB_OLD = 4                                         # the subdivision of the earlier matrices (compared, not used)
GEOMETRIES = list(dip_rows.PYLITH_RUNS) + list(dip_rows.ANALYTIC_ONLY)


def impulse_rule(spatialdb):
    """The impulse region of a make_impulse_db.py database: (y_lo, y_hi, depth_m), read from its grid lines (y: -60 km,
    y_lo - 0.5, y_lo, y_hi, y_hi + 0.5, 60 km; z: -50 km, -depth - 0.5, -depth, 1 km)."""
    lines = open(spatialdb).read().splitlines()

    def grid(name):
        start = lines.index(f"// {name}-coordinates") + 1
        return np.array(lines[start].split(), float)
    y, z = grid("y"), grid("z")
    assert y.size == 6 and z.size == 4, f"{spatialdb}: not a make_impulse_db.py grid"
    y_lo, y_hi, depth_m = y[2], y[3], -z[2]
    assert abs(y[1] - (y_lo - 0.5)) < 1e-6 and abs(y[4] - (y_hi + 0.5)) < 1e-6 and abs(z[1] - (-depth_m - 0.5)) < 1e-6, \
        f"{spatialdb}: the grid lines are not those of make_impulse_db.py"
    return y_lo, y_hi, depth_m


def mesh_tents(mesh_tag):
    """The fault and impulse vertices of a default mesh with no PyLith run: (fault_xyz (m, z up), fault_cells,
    vertex_index), as PyLith would set them up from the mesh and the impulse database."""
    triangles = np.load(os.path.join(paths_inputs["mesh"], f"mesh_{mesh_tag}_tet_default_fault.npz"))["xyz"]   # (n_tri, 3, 3) m
    fault_xyz, inverse = np.unique(np.round(triangles.reshape(-1, 3), 3), axis=0, return_inverse=True)
    fault_cells = inverse.reshape(-1, 3)
    y_lo, y_hi, depth_m = impulse_rule(cfg.gf_files["impulse_db"])
    vertex_index = np.nonzero((fault_xyz[:, 1] >= y_lo) & (fault_xyz[:, 1] <= y_hi) & (fault_xyz[:, 2] >= -depth_m - 1e-3))[0]
    return fault_xyz, fault_cells, vertex_index


def check_impulse_rule(geometry, tents):
    """The impulse rule rebuilt from the default mesh of a PyLith geometry gives the run's impulse vertices (as sets of
    coordinates, to 1 mm)."""
    mesh_tag = "vertical" if geometry == "vertical" else geometry
    fault_xyz, _, vertex_index = mesh_tents(mesh_tag)
    rebuilt = {tuple(v) for v in np.round(fault_xyz[vertex_index], 3)}
    pylith = {tuple(v) for v in np.round(tents.imp_xyz, 3)}
    assert rebuilt == pylith, (f"{geometry}: the impulse rule gives {len(rebuilt)} vertices, PyLith {len(pylith)} "
                               f"({len(rebuilt ^ pylith)} differ)")
    return len(pylith)


def east_minus_west_step(G, setup):
    """East-minus-west fault-parallel step (m/m) of uniform unit slip on every tent, samples within 300 m of the trace
    (the check of notebook 08)."""
    uniform = G.sum(axis=1)
    near = np.abs(setup["signed_d_km"]) < 0.3
    return uniform[near & (setup["signed_d_km"] > 0)].mean() - uniform[near & (setup["signed_d_km"] < 0)].mean()


def save_atomic(path, write):
    # write to a temporary name, then rename, so that an interrupted run leaves no partial matrix
    tmp = path + ".part"
    with open(tmp, "wb") as handle:
        write(handle)
    os.replace(tmp, path)


def compare_old(G_new, old_file, label):
    """Shape, column signs and relative difference per column of the new matrix against the n_sub 4 one."""
    G_old = np.load(old_file)
    G_old = G_old["G_rl_m_per_m"] if old_file.endswith(".npz") else G_old
    assert G_old.shape == G_new.shape, f"{label}: n_sub {N_SUB} G {G_new.shape} vs n_sub {N_SUB_OLD} {G_old.shape}"
    same_sign = np.einsum("ij,ij->j", G_new, G_old) > 0
    assert same_sign.mean() > 0.99, f"{label}: n_sub {N_SUB} and {N_SUB_OLD} disagree in sign for {(~same_sign).sum()} columns"
    rel = np.linalg.norm(G_new - G_old, axis=0) / np.linalg.norm(G_new, axis=0)
    return same_sign, rel


def one_geometry(geometry):
    """Compute (or read) the n_sub analytic G of one geometry, check it, and return its row of the check table."""
    t0 = time.time()
    inputs = common.load_inputs()
    if geometry in dip_rows.PYLITH_RUNS:
        run = dip_rows.PYLITH_RUNS[geometry]
        setup = common.make_setup(inputs, run)
        tents = setup["tents"]
        n_impulse_rule = check_impulse_rule(geometry, tents)
        out_file = os.path.join(dip_rows.DIP_ANALYTIC, f"G_analytic_{run}_nsub{N_SUB}.npy")
        old_file = os.path.join(dip_rows.DIP_ANALYTIC, f"G_analytic_{run}_nsub{N_SUB_OLD}.npy")
        computed = not os.path.exists(out_file)
        if computed:
            G = model_utils.analytic_tent_G(tents, setup["points"]["lon"], setup["points"]["lat"], inputs["look"],
                                            inputs["frame_lonlat"], setup["to_mesh"], n_sub=N_SUB)
            save_atomic(out_file, lambda handle: np.save(handle, G))
        G = np.load(out_file)
        G_pylith = setup["G"]["m"]
        assert G.shape == G_pylith.shape, f"{geometry}: analytic G {G.shape} vs PyLith {G_pylith.shape}"
        sign_pylith = np.einsum("ij,ij->j", G, G_pylith) > 0
        assert sign_pylith.mean() > 0.99, f"{geometry}: analytic and PyLith columns disagree in sign for {(~sign_pylith).sum()} tents"
        sign_pylith_pct = 100 * sign_pylith.mean()
    else:
        name = dip_rows.ANALYTIC_ONLY[geometry]                 # analytic_dip<dip>_nsub<N_SUB>
        run = name
        out_file = os.path.join(dip_rows.DIP_ANALYTIC, f"G_{name}.npz")
        old_file = os.path.join(dip_rows.DIP_ANALYTIC, f"G_analytic_{geometry}_nsub{N_SUB_OLD}.npz")
        n_impulse_rule = np.nan
        computed = not os.path.exists(out_file)
        if computed:
            # the sample fields, look, strike and sign of a PyLith G on the same samples (the vertical run), for the npz
            # layout of build_G (sign_rl is kept for that layout only; the analytic G is right-lateral already), and the
            # lon/lat -> mesh CRS transformer of the set-ups
            ref_setup = common.make_setup(inputs, dip_rows.PYLITH_RUNS["vertical"])
            ref, to_mesh = ref_setup["gnpz"]["m"], ref_setup["to_mesh"]
            assert np.allclose(ref["look"], inputs["look"], atol=1e-12), "the PyLith G's look differs from 07's"
            fault_xyz, fault_cells, vertex_index = mesh_tents(geometry)
            tents = model_utils.TentFault(dict(fault_xyz=fault_xyz, fault_cells=fault_cells, vertex_index=vertex_index), to_mesh,
                                          inputs["profile_lats"])
            points = np.load(inputs["points"]["default"])
            assert np.array_equal(points["lon"], ref["sample_lon"]), "the PyLith G is not on the default point set"
            G = model_utils.analytic_tent_G(tents, points["lon"], points["lat"], inputs["look"], inputs["frame_lonlat"], to_mesh,
                                            n_sub=N_SUB)
            minutes = (time.time() - t0) / 60
            save_atomic(out_file, lambda handle: np.savez(
                handle, G_rl_m_per_m=G, vertex_index=vertex_index, fault_xyz=fault_xyz, fault_cells=fault_cells,
                sample_x=ref["sample_x"], sample_y=ref["sample_y"], sample_lon=ref["sample_lon"], sample_lat=ref["sample_lat"],
                look=ref["look"], strike_deg=ref["strike_deg"], sign_rl=ref["sign_rl"], runs=np.array([name]),
                provenance=np.array(f"dip_rows_analytic_G.py {time.strftime('%Y-%m-%d %H:%M')}: CSI Meade half-space "
                                    f"(model_utils.analytic_tent_G) on mesh_{geometry}_tet_default tents, n_sub {N_SUB}, "
                                    f"{minutes:.0f} min; NOT PyLith")))
        setup = common.make_setup(inputs, name, run_dir=(dip_rows.DIP_ANALYTIC, dip_rows.DIP_ANALYTIC))
        G = setup["G"]["m"]
        tents = setup["tents"]
        sign_pylith_pct = np.nan
        if os.path.exists(old_file):
            old = np.load(old_file)
            for key in ("vertex_index", "fault_xyz", "fault_cells", "sample_lon", "sample_lat", "look", "strike_deg"):
                assert np.array_equal(old[key], setup["gnpz"]["m"][key]), f"{geometry}: {key} differs from the n_sub {N_SUB_OLD} npz"
    step = east_minus_west_step(G, setup)
    assert step > 0, f"{geometry}: right-lateral slip does not move the east side along +look ({step:+.3f} m/m)"
    row = dict(geometry=geometry, run=run, file=os.path.basename(out_file), n_sub=N_SUB, n_samples=G.shape[0], n_tents=G.shape[1],
               computed_now=computed, minutes=(time.time() - t0) / 60, peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
               impulse_rule_vertices=n_impulse_rule, sign_vs_pylith_pct=sign_pylith_pct, uniform_step_m_per_m=step)
    if os.path.exists(old_file):
        same_sign, rel = compare_old(G, old_file, geometry)
        surface = tents.imp_depth_km < 1e-6
        worst = int(np.argmax(rel))
        row.update(old_file=os.path.basename(old_file), sign_vs_nsub4_pct=100 * same_sign.mean(),
                   rel_diff_median=float(np.median(rel)), rel_diff_max=float(rel.max()),
                   rel_diff_max_depth_m=float(tents.imp_depth_km[worst] * 1e3) + 0.0, rel_diff_max_km=float(tents.imp_km[worst]),
                   rel_diff_median_surface=float(np.median(rel[surface])), rel_diff_median_below_100m=float(np.median(rel[tents.imp_depth_km > 0.1])))
    else:
        print(f"{geometry}: no n_sub {N_SUB_OLD} file ({os.path.basename(old_file)}): comparison skipped", flush=True)
    print(f"{geometry}: " + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("geometries", nargs="*", default=GEOMETRIES, help=f"any of {GEOMETRIES} (default: all)")
    parser.add_argument("--processes", type=int, default=4, help="geometries computed at the same time (at most 4)")
    args = parser.parse_args()
    assert set(args.geometries) <= set(GEOMETRIES), f"unknown geometry in {args.geometries}"
    assert 1 <= args.processes <= 4, "at most 4 processes (about 2 GB each)"
    os.makedirs(dip_rows.DIP_ANALYTIC, exist_ok=True)
    elapsed = common.timer()
    # largest first (60 W and 50 W have the most fault triangles), so the processes finish together
    order = sorted(args.geometries, key=lambda g: (g not in dip_rows.ANALYTIC_ONLY, GEOMETRIES.index(g)))
    with ProcessPoolExecutor(max_workers=min(args.processes, len(order)), mp_context=get_context("spawn")) as pool:
        rows = list(pool.map(one_geometry, order))
    table = pd.DataFrame(rows).set_index("geometry").loc[[g for g in GEOMETRIES if g in args.geometries]].reset_index()
    table["provenance"] = common.stamp(__file__)
    check_csv = os.path.join(dip_rows.DIP_ANALYTIC, f"analytic_G_nsub{N_SUB}_check.csv")
    if os.path.exists(check_csv) and set(args.geometries) != set(GEOMETRIES):    # a partial run adds to the table
        previous = pd.read_csv(check_csv)
        table = pd.concat([previous[~previous.geometry.isin(table.geometry)], table]).set_index("geometry")
        table = table.loc[[g for g in GEOMETRIES if g in table.index]].reset_index()
    table.to_csv(check_csv, index=False)
    print(table.drop(columns="provenance").to_string(index=False))
    print(f"wrote {check_csv} in {elapsed()}")
