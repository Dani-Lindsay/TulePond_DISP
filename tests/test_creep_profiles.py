"""creep_profiles.py: the cross-fault extraction recovers a known step with the east-minus-west sign, the sign flips
with the profile direction, the interpolation keeps real dates and leaves long gaps empty, and the per-crossing
decomposition returns right-lateral motion as positive fault-parallel slip. (Until 3 Nov 2026 nothing tested this
module; the synthetic of the 2026-11-03 audit is the first test.)"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import creep_profiles as cp                                                     # noqa: E402
import insar_utils as utils                                                     # noqa: E402

STEP_MM = 10.0


def synthetic_track(profile_az, rng):
    """A 120 x 120 px UTM grid (30 m, X_FIRST / Y_FIRST at the corner, as MintPy) with a +STEP_MM LOS step of the east
    side relative to the west side from epoch 5, centred on one crossing, plus noise. Returns (insar dict, crossing)."""
    nx = ny = 120
    dx = 30.0
    x0, y0 = 500000.0, 4200000.0
    cols, rows = np.meshgrid(np.arange(nx), np.arange(ny))
    easting, northing = x0 + (cols + 0.5) * dx, y0 - (rows + 0.5) * dx
    e0, n0 = x0 + nx * dx / 2 + 7.0, y0 - ny * dx / 2 - 4.0
    a = np.deg2rad(profile_az)
    across_km = ((easting - e0) * np.sin(a) + (northing - n0) * np.cos(a)) / 1000
    n_epochs = 10
    ts = np.zeros((n_epochs, ny, nx))
    for k in range(n_epochs):
        ts[k] = np.where(across_km > 0, 0.5, -0.5) * STEP_MM * (k >= 5) / 1000.0     # m
    ts += rng.normal(0, 0.0005, ts.shape)
    velocity = np.where(across_km > 0, 1.0, -1.0) + rng.normal(0, 0.2, (ny, nx))     # mm/yr
    insar = dict(inc=np.full((ny, nx), 40.0), azi=np.full((ny, nx), 260.0), mask=np.ones((ny, nx), bool), vel=velocity, ts=ts,
                 ts_dates=np.arange(n_epochs) * 0.1 + 2020, x_first=x0, y_first=y0, x_step=dx, y_step=-dx, length=ny, width=nx,
                 win=int(np.ceil(np.hypot(0.15, 0.2) * 1000 / dx)) + 1)
    crossing = pd.DataFrame(dict(ft_lat=[37.5], easting=[e0], northing=[n0], az_smooth=[profile_az]))
    return insar, crossing


def test_extract_track_recovers_the_east_minus_west_step():
    insar, crossing = synthetic_track(54.0, np.random.default_rng(1))
    out = cp.extract_track(insar, crossing, 0.03, 0.15, 0.2, 5, 3.0, 1000.0, "synthetic")
    series = out["creep_ts"].values[0]
    assert np.all(np.abs(series[:5]) < 0.5)                                     # no step before epoch 5 (noise 0.5 mm per pixel)
    assert np.all(np.abs(series[5:] - STEP_MM) < 0.5)                 # east minus west = +STEP_MM after


def test_extract_track_sign_flips_with_the_profile_direction():
    insar, crossing = synthetic_track(54.0, np.random.default_rng(2))
    flipped = crossing.assign(az_smooth=54.0 + 180.0)
    out = cp.extract_track(insar, flipped, 0.03, 0.15, 0.2, 5, 3.0, 1000.0, "flipped")
    assert abs(out["creep_ts"].values[0][-1] + STEP_MM) < 0.5


def test_interp_to_dates_keeps_real_dates_and_leaves_long_gaps_empty():
    day = 1 / 365.25
    dates = np.array([2020.0, 2020.0 + 12 * day, 2020.0 + 24 * day, 2020.0 + 84 * day])     # a 60 d gap at the end
    values = np.array([0.0, 1.0, 2.0, 3.0])
    target = np.array([2020.0 + 6 * day, 2020.0 + 24 * day, 2020.0 + 50 * day, 2020.0 + 84 * day])
    out = cp.interp_to_dates(target, dates, values, 45)
    assert np.isclose(out[0], 0.5) and np.isclose(out[1], 2.0) and np.isclose(out[3], 3.0)
    assert np.isnan(out[2])                                                     # inside the 60 d gap: not interpolated


def test_decompose_fp_returns_right_lateral_slip_as_positive():
    strike = 146.5
    s = np.deg2rad(strike)
    inc = {"asc": np.array([40.9]), "desc": np.array([43.2])}
    azi = {"asc": np.array([99.9]), "desc": np.array([260.4])}
    dates = np.array([2020.0, 2020.5, 2021.0])
    fp_true = np.array([0.0, 4.0, 10.0])                                        # mm, the east side toward the strike (SE)
    up_true = np.array([0.0, -1.0, 0.5])
    los = {}
    for track in ("asc", "desc"):
        unit = utils.los_unit_vector(inc[track][0], azi[track][0])
        motion = np.outer(fp_true, [np.sin(s), np.cos(s), 0.0]) + np.outer(up_true, [0.0, 0.0, 1.0])
        los[track] = pd.DataFrame([motion @ unit], index=[37.5], columns=dates)
    fp = cp.decompose_fp(los, inc, azi, np.array([strike]), np.array([37.5]), dates)
    assert np.allclose(fp.values[0], fp_true, atol=1e-6)
