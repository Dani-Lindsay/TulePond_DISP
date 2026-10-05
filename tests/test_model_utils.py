"""model_utils.py: the surface constraint of the 3D model (until 2026-10-06 scripts/pylith_helpers/surface_crossings.py)
and the small closed-form pieces of the slip models (covariance, L-curve, magnitudes, rigidity, stress drop)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_utils as sc                                                       # noqa: E402


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


def test_covariance_matrix_nugget_on_diagonal_only():
    x, y = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 2.0])
    cov_p = (4.0, 1.0, 0.5, 2.0, 5.0)                                          # nugget2, s1, l1, s2, l2
    cd = sc.covariance_matrix(x, y, np.array([1.0, 4.0, 16.0]), cov_p, 0.5)
    assert np.allclose(np.diag(cd), 3.0 + 4.0 / np.sqrt([1.0, 4.0, 16.0]))      # sills at zero lag + nugget / n_px^0.5
    assert np.isclose(cd[0, 1], np.exp(-1.0 / 0.5) + 2.0 * np.exp(-1.0 / 5.0))
    assert np.allclose(cd, cd.T)


def test_lcurve_corner_and_pick_on_a_synthetic_l():
    sigma = np.array([1.0, 2, 4, 8, 16, 32, 64])
    misfit = np.array([100.0, 60, 30, 12, 10, 9.5, 9.3])                       # flattens after sigma 8
    rough = np.array([1.0, 1.2, 1.5, 2.0, 6, 20, 70])                          # rises after sigma 8
    corner, _, curvature = sc.lcurve_corner(sigma, misfit, rough)
    assert 4 < corner < 16 and np.all(curvature >= 0)
    assert sc.lcurve_pick(sigma, corner) == sigma[sigma >= corner].min()


def test_moment_magnitude_and_rigidity():
    assert np.isclose(sc.moment_magnitude(10 ** 16.1), 2 / 3 * 7.0)            # Hanks & Kanamori
    assert np.isclose(sc.harmonic_mean_rigidity(1.0, 1.0), 1.0)
    assert np.isclose(sc.harmonic_mean_rigidity(1.0, 3.0), 1.5)                 # below the arithmetic mean 2
    assert np.isclose(sc.potency_from_slip(np.array([1.0, 2.0]), np.array([10.0, 5.0])), 1e6 * 0.01 + 2e6 * 0.005)


def test_stress_drop_weights_and_sign():
    xyz = np.array([[0.0, 0, 0], [1, 0, 0], [0, 0, -1], [1, 0, -1]])
    cells = np.array([[0, 1, 2], [1, 3, 2]])
    area = sc.vertex_areas_m2(xyz, cells)
    assert np.isclose(area.sum(), 1.0)
    assert np.isclose(sc.slip_weighted_stress_drop_pa(np.full(4, 5.0), np.ones(4), area), 5.0)
    with pytest.raises(AssertionError):
        sc.slip_weighted_stress_drop_pa(np.full(4, -5.0), np.ones(4), area)


def test_km_from_pinole_index_rule():
    lats = np.linspace(37.4, 38.0, 7)                                          # south -> north, 0.1 km per vertex
    km = sc.km_from_pinole(lats[[6, 0, 3]], lats)
    assert np.allclose(km, [0.0, 0.6, 0.3])


def test_screw_dislocation_far_and_near():
    assert np.isclose(sc.screw_dislocation_mm(10.0, 1000.0, 1e9), 0.0, atol=1e-5)
    assert np.isclose(sc.screw_dislocation_mm(10.0, 1000.0, 1e-9), 5.0, atol=1e-5)   # s/2 at the fault, east side

