#!/usr/bin/env python3
"""
Assemble the fault-parallel Green's-function matrix G (n_samples x n_impulses) from a PyLith GreensFns run,
following the reading pattern of PyLith examples/subduction-3d/invert_slip.py (readGreens: fault
geometry/vertices, viz/topology/cells, vertex_fields/slip[:, :, 1]; responses vertex_fields/displacement) and its
matchCoords idea (points matched to the data by coordinates, here with a KD-tree).

Glue (custom, assert-checked):
  - impulse -> fault vertex: the vertex carrying the unit left-lateral slip of each impulse (argmax), asserting that
    exactly one vertex has |slip| = 1 and every other vertex 0;
  - output point -> data sample: nearest neighbour in the mesh CRS, asserting < 1 mm;
  - projection: 3D displacement . look, look = horizontal unit vector along 06's strike bearing (07's `look`);
  - sign: right-lateral = -left-lateral with PyLith's fault orientation; the sign is DERIVED (east-side mean
    fault-parallel motion of each impulse must be positive for right-lateral slip, as in the data) and asserted for
    every impulse.
Run in opera_disp_feb13:  python build_G.py <run name> [<run name> ...]   (reads <model_sfcvm_dir>/pylith_output/<name>-points.h5, -fault.h5)
Writes <model_sfcvm_dir>/gf/G_<first run name>.npz (runs concatenated along the impulse axis, e.g. MPI-split jobs).
Optional --points <npz> names the output-point set of the run (default points/insar_points.npz).
"""
import os
# Ported from drafts/pylith (2026-09-30) for notebooks 08a/08b: every path comes from TulePond_filepaths
# (paths_sfcvm = the PyLith side under model_sfcvm_dir; paths_out = the repo's outputs/).
import sys
import time

import h5py
import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
from TulePond_filepaths import paths_sfcvm, paths_out, common_paths   # noqa: E402


def read_run(name):
    with h5py.File(os.path.join(paths_sfcvm["pylith_output"], f"{name}-fault.h5"), "r") as h:
        fault_xyz = h["geometry/vertices"][:]
        fault_cells = np.asarray(h["viz/topology/cells"][:], dtype=np.int64)
        slip = h["vertex_fields/slip"][:]
    with h5py.File(os.path.join(paths_sfcvm["pylith_output"], f"{name}-points.h5"), "r") as h:
        point_xyz = h["geometry/vertices"][:]
        disp = h["vertex_fields/displacement"][:]
    assert slip.shape[0] == disp.shape[0], "impulse count differs between the fault and points files"
    return fault_xyz, fault_cells, slip, point_xyz, disp


def main(names, points_file=None):
    """points_file: the .npz of the output points the run used (default points/insar_points.npz, the 2386 samples)."""
    points = np.load(points_file or os.path.join(paths_sfcvm["points"], "insar_points.npz"))
    row = np.load(os.path.join(paths_out["07b_uniform_model_tests"], "fig2_model_row.npz"))
    strike = float(row["strike_deg"])
    look = np.array([np.sin(np.deg2rad(strike)), np.cos(np.deg2rad(strike)), 0.0])     # 07 [3]
    columns, vertex_index, fault_xyz0, cells0 = [], [], None, None
    for name in names:
        fault_xyz, fault_cells, slip, point_xyz, disp = read_run(name)
        if fault_xyz0 is None:
            fault_xyz0, cells0 = fault_xyz, fault_cells
        assert np.array_equal(fault_xyz, fault_xyz0), f"{name}: different fault mesh from {names[0]}"
        # impulse -> vertex (unit left-lateral slip on exactly one vertex)
        ll = slip[:, :, 1]
        k_vertex = np.argmax(np.abs(ll), axis=1)
        peak = ll[np.arange(ll.shape[0]), k_vertex]
        assert np.allclose(np.abs(peak), 1.0, atol=1e-6), f"{name}: impulse amplitude not 1 m: {peak.min()}..{peak.max()}"
        others = ll.copy(); others[np.arange(ll.shape[0]), k_vertex] = 0.0
        assert np.max(np.abs(others)) < 1e-6 and np.max(np.abs(slip[:, :, [0, 2]])) < 1e-6, f"{name}: impulse not a single-vertex tent"
        # output point -> data sample
        dist, order = cKDTree(point_xyz[:, :2]).query(np.c_[points["x"], points["y"]])
        assert dist.max() < 1e-3, f"{name}: a data sample has no PyLith output point within 1 mm ({dist.max():.3g} m)"
        assert np.unique(order).size == order.size, "two samples matched the same output point"
        columns.append(np.einsum("kpc,c->pk", disp[:, order, :], look) * np.sign(peak)[None, :])
        vertex_index.append(k_vertex)
    G_ll = np.hstack(columns)                              # m of fault-parallel displacement per m of LEFT-lateral slip
    vertex_index = np.concatenate(vertex_index)
    assert np.unique(vertex_index).size == vertex_index.size, "a fault vertex received two impulses"
    # sign: right-lateral slip moves the east side along +look (07's data convention)
    # (a) uniform left-lateral slip over the whole impulse region (sum of all tents): east-minus-west step of the
    #     samples within 500 m of the trace along the region; (b) every SURFACE-vertex impulse individually (a buried
    #     tent's fault-parallel field has lobes of both signs, so it is not a clean test)
    cross = points["x"] - np.interp(points["y"], *trace_top(fault_xyz0))
    y_imp = fault_xyz0[vertex_index, 1]
    band = (points["y"] > y_imp.min()) & (points["y"] < y_imp.max()) & (np.abs(cross) < 500.0)
    assert (band & (cross > 0)).sum() >= 5 and (band & (cross < 0)).sum() >= 5, "too few samples beside the impulse region"
    uniform = G_ll.sum(axis=1)
    step_uniform = uniform[band & (cross > 0)].mean() - uniform[band & (cross < 0)].mean()
    sign_rl = -1.0 if step_uniform < 0 else 1.0
    surface_k = np.nonzero(np.abs(fault_xyz0[vertex_index, 2]) < 1e-6)[0]
    step_k = []
    for k in surface_k:
        near = (np.abs(points["y"] - y_imp[k]) < 500.0) & (np.abs(cross) < 300.0)
        step_k.append(G_ll[near & (cross > 0), k].mean() - G_ll[near & (cross < 0), k].mean())
    step_k = sign_rl * np.array(step_k)
    print(f"right-lateral = {sign_rl:+.0f} x left-lateral (uniform-slip east-minus-west step {sign_rl * step_uniform:+.3f} m/m); "
          f"{np.sum(step_k > 0)}/{step_k.size} surface impulses east-minus-west positive")
    assert np.all(step_k > 0), "a surface impulse has the wrong east-west sign"
    G_rl = sign_rl * G_ll
    out = os.path.join(paths_sfcvm["gf"], f"G_{names[0]}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, G_rl_m_per_m=G_rl, vertex_index=vertex_index, fault_xyz=fault_xyz0, fault_cells=cells0,
             sample_x=points["x"], sample_y=points["y"], sample_lon=points["lon"], sample_lat=points["lat"],
             look=look, strike_deg=strike, sign_rl=sign_rl, runs=np.array(names),
             provenance=f"build_G.py {time.strftime('%Y-%m-%d %H:%M')} from {names}")
    print(f"wrote {out}: G {G_rl.shape}, max |G| {np.abs(G_rl).max():.3g} m/m")
    return out


def trace_top(fault_xyz):
    """(y, x) of the fault's top edge (z = 0), sorted by y: the mesh's own surface trace."""
    top = fault_xyz[np.abs(fault_xyz[:, 2]) < 1e-6]
    order = np.argsort(top[:, 1])
    return top[order, 1], top[order, 0]


if __name__ == "__main__":
    argv = sys.argv[1:]
    pts = None
    if "--points" in argv:
        k = argv.index("--points"); pts = argv[k + 1]; argv = argv[:k] + argv[k + 2:]
    main(argv, points_file=pts)
