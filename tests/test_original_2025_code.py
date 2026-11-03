"""Checks against the first author's pre-repository code (2025).

`los2fault_projection` is copied verbatim from the February 2025 script in original_2025_code/ (the code behind the
SCEC 2025 poster, Lindsay, Taira and Burgmann 2025), and `invert_single_point_ad2eu` verbatim from the first author's
August 2025 code (not in the repository). They are an independent statement of the same geometry, and the tests pin
the repository's conventions to them. The projection error they record in the 2025 script (the projection vector
along the cross-fault bearing instead of the strike) is described in original_2025_code/README.md, and notebook 15
computes its effect.
"""
import numpy as np
import pandas as pd
import pytest

import insar_utils as utils

# the repository's test geometry (tests/test_insar_utils.py)
ASC = dict(inc=40.33, azi=99.94)
DESC = dict(inc=43.93, azi=260.52)
STRIKE = 144.09                     # compass bearing, cw from north
CROSS_FAULT_AZ = STRIKE - 90.0      # profile_az in hayward_fault_profiles.csv (toward the east side)


# ---- verbatim, HW1_creeprate.py (DL, 14 Feb 2025) --------------------------------------------
def los2fault_projection(observations, los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg):
    """
    Calculate the design matrix for converting LOS displacement to slope direction.
    Converts aspect from positive clockwise from north to positive anticlockwise from east.
    """
    # Convert aspect to match LOS azimuth reference
    aspect_deg = -aspect_deg + 90

    # LOS components
    los_E = np.sin(np.deg2rad(los_inc_angle_deg)) * np.sin(np.deg2rad(los_az_angle_deg)) * -1
    los_N = np.sin(np.deg2rad(los_inc_angle_deg)) * np.cos(np.deg2rad(los_az_angle_deg))
    los_U = np.cos(np.deg2rad(los_inc_angle_deg))

    # Downslope components
    fault_E = np.sin(np.deg2rad(slope_deg)) * np.sin(np.deg2rad(aspect_deg)) * -1
    fault_N = np.sin(np.deg2rad(slope_deg)) * np.cos(np.deg2rad(aspect_deg))
    fault_U = np.cos(np.deg2rad(slope_deg))

    # Normalize vectors
    L = np.array([los_E, los_N, los_U])
    L = L / np.linalg.norm(L)

    F = np.array([fault_E, fault_N, fault_U])
    F = F / np.linalg.norm(F)

    # Compute design matrix (dot product)
    G = np.dot(L, F)

    fault_parallel = observations / G if G != 0 else np.nan

    return fault_parallel
# ---- end verbatim ------------------------------------------------------------------------------


# ---- verbatim, NC_ALOS2_S1_Seismica/Scripts/insar_utils.py (DL, 15 Aug 2025) --------------------
def invert_single_point_ad2eu(row):
    """
    Perform least-squares inversion using only ascending and descending LOS vectors
    to solve for East and Up components (North is excluded).
    """
    # Check if 'asc_vel' or 'des_vel' is NaN
    if pd.isnull(row['asc_vel']) or pd.isnull(row['des_vel']):
        # If any of the specified columns are NaN, return NaN for east and up
        return pd.Series({'east': np.nan, 'up': np.nan})

    # Design matrix (only solving for East and Up, excluding North)
    G = np.array([
        [np.sin(np.radians(row['asc_azi'])) * np.sin(np.radians(row['asc_inc'])) * -1,  # East component (ascending)
         np.cos(np.radians(row['asc_inc']))],  # Up component (ascending)
        [np.sin(np.radians(row['des_azi'])) * np.sin(np.radians(row['des_inc'])) * -1,  # East component (descending)
         np.cos(np.radians(row['des_inc']))]   # Up component (descending)
    ])

    # Observation vector (LOS displacements for ascending and descending)
    dlos = np.array([row['asc_vel'], row['des_vel']])

    # Perform the least-squares inversion for this point
    EN_result, _, _, _ = np.linalg.lstsq(G, dlos, rcond=None)

    # Return the East and Up components
    return pd.Series({'east': EN_result[0], 'up': EN_result[1]})
# ---- end verbatim ------------------------------------------------------------------------------


def _los_vector_2025(inc, azi):
    """The LOS unit vector exactly as los2fault_projection builds it."""
    return np.array([np.sin(np.deg2rad(inc)) * np.sin(np.deg2rad(azi)) * -1,
                     np.sin(np.deg2rad(inc)) * np.cos(np.deg2rad(azi)),
                     np.cos(np.deg2rad(inc))])


@pytest.mark.parametrize("g", [ASC, DESC], ids=["asc", "desc"])
def test_2025_los_vector_equals_repository(g):
    assert np.allclose(_los_vector_2025(g["inc"], g["azi"]), utils.los_unit_vector(g["inc"], g["azi"]), atol=1e-12)


@pytest.mark.parametrize("g", [ASC, DESC], ids=["asc", "desc"])
def test_2025_projection_with_cross_fault_azimuth_equals_repository_fp_column(g):
    """Fed the cross-fault azimuth (54 deg), the 2025 function's vector F is the strike reversed
    (bearing 324), so its factor is minus the repository's FP column. Same equation, opposite
    sign convention for 'along strike'."""
    G_repo = utils.design_matrix_up_fp([g["inc"]], [g["azi"]], STRIKE)[0, 1]
    factor_2025 = 1.0 / los2fault_projection(1.0, g["inc"], g["azi"], 90.0, CROSS_FAULT_AZ)
    assert np.isclose(factor_2025, -G_repo, atol=1e-12)


def test_2025_projection_as_called_in_the_hayward_loop_points_across_the_fault():
    """As called in 2025 (aspect = profile_az + 90 = strike), F points along bearing 54 deg, the
    cross-fault direction. Descending factor within 1.4 % of the correct strike factor; ascending
    about 2x. Recorded so the 2025 poster/COZ1 results are understood, not to change them."""
    for g, expected_ratio_lo, expected_ratio_hi in ((DESC, 0.97, 1.03), (ASC, 1.8, 2.3)):
        G_repo = utils.design_matrix_up_fp([g["inc"]], [g["azi"]], STRIKE)[0, 1]
        factor_2025 = 1.0 / los2fault_projection(1.0, g["inc"], g["azi"], 90.0, STRIKE)
        # the 2025 vector is the unit normal to the strike (bearing 54): check it against the repo's cross-fault unit vector
        n_hat = utils.horizontal_unit_vector(CROSS_FAULT_AZ)
        assert np.isclose(factor_2025, utils.los_unit_vector(g["inc"], g["azi"]) @ n_hat, atol=1e-12)
        assert expected_ratio_lo < abs(factor_2025 / G_repo) < expected_ratio_hi


def test_2025_east_up_inversion_matrix_equals_repository_los_components():
    """The 2025 asc/desc -> (East, Up) solve uses the E and U components of the same LOS vector."""
    row = pd.Series(dict(asc_vel=1.0, des_vel=1.0, asc_inc=ASC["inc"], asc_azi=ASC["azi"],
                         des_inc=DESC["inc"], des_azi=DESC["azi"]))
    # rebuild its design matrix exactly as the function does and compare with the repo vectors
    for g, key in ((ASC, "asc"), (DESC, "des")):
        L = utils.los_unit_vector(g["inc"], g["azi"])
        assert np.isclose(np.sin(np.radians(row[f"{key}_azi"])) * np.sin(np.radians(row[f"{key}_inc"])) * -1, L[0], atol=1e-12)
        assert np.isclose(np.cos(np.radians(row[f"{key}_inc"])), L[2], atol=1e-12)
    # and the solve reproduces a known (East, Up) pair from its two LOS projections
    east, up = 3.0, -1.5
    row["asc_vel"] = east * utils.los_unit_vector(ASC["inc"], ASC["azi"])[0] + up * utils.los_unit_vector(ASC["inc"], ASC["azi"])[2]
    row["des_vel"] = east * utils.los_unit_vector(DESC["inc"], DESC["azi"])[0] + up * utils.los_unit_vector(DESC["inc"], DESC["azi"])[2]
    out = invert_single_point_ad2eu(row)
    assert np.isclose(out["east"], east, atol=1e-9) and np.isclose(out["up"], up, atol=1e-9)


def test_repository_decomposition_recovers_fp_and_up_from_2025_los_vectors():
    """Project a known (FP, Up) pair onto both LOS vectors with the 2025 vector formula, then
    recover it with the repository's decompose_asc_desc."""
    fp, up = 8.0, -2.0
    s_hat = utils.horizontal_unit_vector(STRIKE)
    d = {}
    for g, key in ((ASC, "asc"), (DESC, "desc")):
        L = _los_vector_2025(g["inc"], g["azi"])
        d[key] = fp * (L @ s_hat) + up * L[2]
    U, FP = utils.decompose_asc_desc([d["asc"]], [d["desc"]], ASC["inc"], ASC["azi"], DESC["inc"], DESC["azi"], STRIKE)
    assert np.isclose(FP[0], fp, atol=1e-9) and np.isclose(U[0], up, atol=1e-9)
