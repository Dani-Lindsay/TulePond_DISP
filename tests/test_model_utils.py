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



def test_match_impulse_order_undoes_a_vertex_renumbering():
    """Two GreensFns runs of one mesh with different MPI rank counts number the fault vertices differently
    (2026-10-25): the match must recover the reference impulse order from coordinates alone."""
    rng = np.random.default_rng(0)
    fault = rng.uniform(-1000, 1000, (50, 3))
    vertex_index = np.array([3, 7, 11, 20, 42, 5])                               # the impulse vertices of the reference run
    perm = rng.permutation(50)                                                   # the other run's vertex numbering
    fault_b = fault[perm]
    inv = np.argsort(perm)
    vertex_index_b = rng.permutation(inv[vertex_index])                          # same impulses, numbered and ordered differently
    idx = sc.match_impulse_order(fault, vertex_index, fault_b, vertex_index_b)
    assert np.allclose(fault_b[vertex_index_b][idx], fault[vertex_index])
    G_b = np.arange(6)[None, :] * np.ones((4, 1))                                # column j = impulse vertex_index_b[j]
    assert np.allclose(fault_b[vertex_index_b[G_b[0, idx].astype(int)]], fault[vertex_index])


def test_match_impulse_order_rejects_a_different_mesh():
    fault = np.random.default_rng(1).uniform(-1000, 1000, (20, 3))
    other = fault.copy(); other[4] += 0.5                                        # one vertex moved 0.5 m
    with pytest.raises(AssertionError):
        sc.match_impulse_order(fault, np.arange(5), other, np.arange(5))


def test_match_impulse_order_on_the_5_and_8_rank_order2_runs():
    """The real case: gf_default_sfcvm_order2 (5 ranks, 2026-10-25) vs gf_default_homog_order2 (8 ranks, 2026-09-30)."""
    data = os.path.expanduser("~/work/proj_TulePond_DISP")
    a = os.path.join(data, "modeling_SFCVM", "gf", "G_gf_default_sfcvm_order2.npz")
    b = os.path.join(data, "modeling_uniform", "gf", "G_gf_default_homog_order2.npz")
    if not (os.path.exists(a) and os.path.exists(b)):
        pytest.skip("order-2 G matrices not available")
    ga, gb = np.load(a), np.load(b)
    idx = sc.match_impulse_order(ga["fault_xyz"], ga["vertex_index"], gb["fault_xyz"], gb["vertex_index"])
    assert np.allclose(gb["fault_xyz"][gb["vertex_index"]][idx], ga["fault_xyz"][ga["vertex_index"]], atol=1e-3)


# ---------------------------------------------------------------------------------------------------------------------
# 2026-10-30 (rewire, code_audit_2026-10-28): the locked weighting, the noise-model mask, the metrics, the Monte Carlo
# ---------------------------------------------------------------------------------------------------------------------
from types import SimpleNamespace                                               # noqa: E402


def _samples(n_px):
    n = len(n_px)
    return SimpleNamespace(x=np.arange(n, dtype=float), y=np.zeros(n), wgt=np.asarray(n_px, float), vel=np.zeros(n))


def test_data_cd_nugget_uses_its_exponent_argument():
    """N^0 (the locked weighting) = every sample the nugget variance, whatever the block sizes; N^0.5 = nugget / sqrt(n)."""
    cov_p = (0.81, 0.5, 1.7, 1e-6, 2.0)
    smp = _samples([1, 4, 16, 64])
    assert np.allclose(sc.data_cd("nugget", smp, cov_p, 0.0), 0.81 * np.eye(4))
    assert np.allclose(np.diag(sc.data_cd("nugget", smp, cov_p, 0.5)), 0.81 / np.sqrt([1, 4, 16, 64]))
    with pytest.raises(AssertionError):
        sc.data_cd("nugget", smp, cov_p, None)                                  # no silent default exponent


def test_data_cd_full_and_the_near_trace_cut():
    cov_p = (0.81, 0.5, 1.7, 0.2, 5.0)
    smp = _samples([1, 4, 16, 64])
    full = sc.data_cd("full", smp, cov_p, 0.5)
    assert np.allclose(full, sc.covariance_matrix(smp.x, smp.y, smp.wgt, cov_p, 0.5))
    keep = np.array([True, False, True, True])
    cut = sc.data_cd("full", smp, cov_p, 0.5, keep=keep)
    assert np.allclose(cut[1, [0, 2, 3]], 0) and np.isclose(cut[1, 1], 1e8 * full[1, 1])   # decoupled, no weight
    assert np.allclose(cut[np.ix_(keep, keep)], full[np.ix_(keep, keep)])


def test_covariance_refuses_a_mask_the_swath_cannot_hold():
    """Until 2026-10-30 a 5 km mask on a 4 km swath silently became 1.6 km."""
    sar = SimpleNamespace(x=np.zeros(5), y=np.zeros(5), vel=np.zeros(5))
    fault = SimpleNamespace(xf=np.zeros(2), yf=np.array([-1.0, 1.0]))
    with pytest.raises(AssertionError):
        sc.covariance(sar, fault, 4.0, 5.0, 8.0)


def test_covariance_recovers_white_noise_and_returns_its_bins():
    rng = np.random.default_rng(3)
    gx, gy = np.meshgrid(np.arange(-15, 15, 0.1), np.arange(-15, 15, 0.1))
    sar = SimpleNamespace(x=gx.ravel(), y=gy.ravel(), vel=rng.normal(0, 0.9, gx.size))
    fault = SimpleNamespace(xf=np.zeros(4001), yf=np.linspace(-20.0, 20.0, 4001))   # a densely sampled N-S trace at x = 0
    params, bins, n_pix = sc.covariance(sar, fault, 15.0, 5.0, 8.0, return_bins=True)
    total = params[0] + params[1] + params[3]
    assert np.isclose(np.sqrt(total), 0.9, rtol=0.03)                           # white noise: total sill = the variance
    assert n_pix == int((np.abs(sar.x) > 5.0).sum())
    assert {"distance_km", "n_pairs", "gamma_mm2", "model_mm2"} <= set(bins.columns)


def test_potency_extent_quantiles():
    pot = np.array([1.0, 1.0, 1.0, 1.0, 6.0])
    depth = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    km = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    w50, _ = sc.potency_extent(pot, depth, km, 0.5)
    w90, _ = sc.potency_extent(pot, depth, km, 0.9)
    assert w50 == 50.0 and w90 == 50.0                                          # 60 % of the potency is at 50 m
    w30, _ = sc.potency_extent(pot, depth, km, 0.3)
    assert w30 == 30.0


def test_correlated_noise_has_the_requested_covariance():
    Cd = np.array([[1.0, 0.6], [0.6, 2.0]])
    draws = sc.correlated_noise(Cd, 200000, seed=11)
    assert np.allclose(np.cov(draws.T), Cd, atol=0.02)
    assert np.allclose(sc.correlated_noise(Cd, 3, seed=11), sc.correlated_noise(Cd, 3, seed=11))   # the seed fixes it


def test_materna_stress_drop():
    assert np.isclose(sc.materna_stress_drop_pa(30e9, 0.0128, 175.0), 30e9 * 0.0128 / 175.0)
    with pytest.raises(AssertionError):
        sc.materna_stress_drop_pa(30e9, 0.0, 175.0)


def test_strike_slip_stress_drop():
    # (2 / pi) mu d / w: pi times smaller than mu d / (w / 2)
    assert np.isclose(sc.strike_slip_stress_drop_pa(30e9, 0.0128, 350.0), 2 / np.pi * 30e9 * 0.0128 / 350.0)
    assert np.isclose(sc.materna_stress_drop_pa(30e9, 0.0128, 175.0) / sc.strike_slip_stress_drop_pa(30e9, 0.0128, 350.0), np.pi)
    with pytest.raises(AssertionError):
        sc.strike_slip_stress_drop_pa(30e9, 0.0128, 0.0)


def test_crossing_steps_like_measures_a_step():
    d = np.r_[np.linspace(-0.2, -0.01, 20), np.linspace(0.01, 0.2, 20)]
    values = np.where(d > 0, 2.0, -1.0)
    steps = sc.crossing_steps_like(values, d, np.full(d.size, 59.0), [59.0, 70.0], inner_km=0.06, outer_km=0.15)
    assert np.isclose(steps[0], 3.0) and np.isnan(steps[1])                      # 70 km: no samples


def test_analytic_tent_G_right_lateral_sign():
    """A surface tent on a N-S vertical fault, strike 180 (look = south): right-lateral slip moves the east side along
    +look, as the data (build_G's convention), with analytic_tent_G's fixed -G sign."""
    import pyproj
    from TulePond_filepaths import model_crs, model_frame_lonlat
    to_mesh = pyproj.Transformer.from_crs("EPSG:4326", model_crs, always_xy=True)
    ys, zs = np.array([-600.0, 0.0, 600.0]), np.array([0.0, -600.0])
    xyz = np.array([[0.0, y, z] for z in zs for y in ys])                       # 0..2 at the surface, 3..5 at 600 m
    cells = np.array([[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]])
    tents = SimpleNamespace(fault_xyz=xyz, fault_cells=cells, vertex_index=np.array([1]), n_imp=1)
    px, py = np.array([-100.0, 100.0]), np.array([0.0, 0.0])
    lon, lat = to_mesh.transform(px, py, direction="INVERSE")
    G = sc.analytic_tent_G(tents, lon, lat, sc.look_vector(180.0), model_frame_lonlat, to_mesh, n_sub=4)
    assert G.shape == (2, 1) and G[1, 0] - G[0, 0] > 0                          # east minus west along +look


def test_side_deflection_near_minus_far_per_side():
    d = np.r_[-np.linspace(0.07, 0.24, 6), -np.linspace(1.6, 2.9, 6), np.linspace(0.07, 0.24, 6), np.linspace(1.6, 2.9, 6)]
    values = np.r_[np.full(6, -5.0), np.full(6, -1.0), np.full(6, 3.0), np.full(6, 1.0)]
    ok = np.ones(d.size, bool)
    assert sc.side_deflection_mm(values, d, ok, ok, -1, (0.06, 0.25), (1.5, 3.0)) == (-4.0, 6, 6)
    assert sc.side_deflection_mm(values, d, ok, ok, 1, (0.06, 0.25), (1.5, 3.0)) == (2.0, 6, 6)


# 2026-10-30: the LOS test (los_inversion_2026-10-30): several data sets in one CSI solve, LOS Green's functions
def _real_setup():
    data = os.path.expanduser("~/work/proj_TulePond_DISP")
    run = "gf_default_sfcvm_order2"
    if not os.path.exists(os.path.join(data, "modeling_SFCVM", "gf", f"G_{run}.npz")) or \
            not os.path.exists(os.path.join(data, "outputs", "07a_uniform_model", "model_data.npz")):
        pytest.skip("real products not available")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "sensitivity"))
    import common
    inputs = common.load_inputs()
    return common, inputs, common.make_setup(inputs, run), common.samples_for(inputs, "default")


def test_invert_tents_multi_with_one_dataset_equals_invert_tents():
    common, inputs, setup, samples = _real_setup()
    G = setup["G"]["m"]
    Cd = sc.data_cd("nugget", samples, inputs["cov_p"], 0.0, keep=setup["keep"])
    one, data_one, _ = sc.invert_tents(setup["template"], samples, G, Cd, None, None, 27.0, (0.1, 0.05), inputs["frame_lonlat"])
    multi, datas = sc.invert_tents_multi(setup["template"], [(samples, G, Cd)], 27.0, (0.1, 0.05), inputs["frame_lonlat"])
    assert np.allclose(one.slip[:, 0], multi.slip[:, 0], atol=1e-6)
    assert np.allclose(data_one.vel, datas[0].vel, atol=1e-6) and np.allclose(data_one.synth, datas[0].synth, atol=1e-6)


def test_build_G_los_reproduces_the_fault_parallel_G():
    import model_config as cfg
    common, inputs, setup, samples = _real_setup()
    gnpz = setup["gnpz"]["m"]
    look = np.tile(np.asarray(gnpz["look"], float), (samples.vel.size, 1))
    G_los, G_fp = sc.build_G_los(gnpz, inputs["points"]["default"], look, cfg.gf_pylith_output_dir)
    assert np.allclose(G_fp, gnpz["G_rl_m_per_m"], atol=1e-12) and np.allclose(G_los, G_fp, atol=1e-12)
