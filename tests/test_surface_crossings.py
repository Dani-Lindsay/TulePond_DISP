"""The 3D model's surface constraint (scripts/pylith_helpers/surface_crossings.py)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "pylith_helpers"))
import surface_crossings as sc                                                 # noqa: E402


def test_tent_rows_reproduce_linear_slip_and_sum_to_one():
    s_vert = np.array([0.0, 100.0, 200.0, 300.0])
    cols = np.array([5, 2, 7, 0])                                                # top-edge vertices in arc order -> columns
    slip = np.zeros(8); slip[cols] = [1.0, 3.0, 2.0, 6.0]
    s = np.array([0.0, 50.0, 100.0, 250.0, 300.0])
    G = sc.tent_rows(s, s_vert, cols, 8)
    assert np.allclose(G.sum(1), 1.0)
    assert np.allclose(G @ slip, np.interp(s, s_vert, slip[cols]))              # = the linear tent along the edge


def test_arc_length_on_a_bent_polyline():
    vx, vy = np.array([0.0, 0.0, 100.0]), np.array([0.0, 100.0, 100.0])        # up 100 m, then right 100 m
    s, d, s_vert = sc.arc_length_on_polyline(np.array([5.0, 50.0]), np.array([30.0, 110.0]), vx, vy)
    assert np.allclose(s, [30.0, 150.0]) and np.allclose(d, [5.0, 10.0]) and np.allclose(s_vert, [0, 100, 200])


def test_local_scatter_keeps_window_edges_and_floor():
    km = 57.0 + 0.1 * np.arange(21)                                              # float km on a 0.1 grid (roundoff at the edges)
    v = 0.3 * np.arange(21) ** 2                                                 # deviations differ with and without the edges
    err = sc.local_scatter_mm(km, v, floor_mm=0.01)
    for i in range(2, 19):                                                       # the 5-point window by index = both edges kept
        nb = v[i - 2:i + 3]
        assert np.isclose(err[i], max(1.4826 * np.median(np.abs(nb - np.median(nb))), 0.01))
    assert np.all(sc.local_scatter_mm(km, np.full(km.size, 3.0), floor_mm=0.5) == 0.5)
