#!/usr/bin/env python3
"""
Tests pinning the LOS / fault projection conventions and the small shared
helpers in insar_utils.py.

The projection tests exist because a 90-degree convention error survived in
an earlier pipeline for years: the fault vector was built from a clockwise
compass bearing but fed to the (-sin, cos) pairing that expects MintPy's
ANTI-CLOCKWISE azimuth. Descending was only 1.4% wrong (so creepmeter
validation still passed) but ascending fault-parallel came out 2.05x too
small. The legacy construction is kept INSIDE this file so the regression
tests keep pinning it.

The ground truth for the azimuth convention is OPERA's own los_east/los_north
rasters, see test_los_unit_vector_matches_opera_los_east_north.

Run:  pytest tests/test_insar_utils.py -v      (env: opera_disp_feb13)
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import insar_utils as utils                                   # noqa: E402
from TulePond_filepaths import insar_dic_list                  # noqa: E402

# Our real per-track mean geometry (from geometryGeo.h5).
ASC = dict(inc=40.33, azi=99.94)
DESC = dict(inc=43.93, azi=260.52)
STRIKE = 144.09          # mean(profile_az) + 90, compass bearing cw from north
PROFILE_AZ = 54.09       # mean cross-fault azimuth

# Canonical near-polar geometries from MintPy's own definitions. Sentinel-1 is
# right-looking; NISAR is left-looking, so for the SAME orbit direction its
# LOS azimuth is 180 deg away.
SEN_A = dict(inc=40, azi=102)
SEN_D = dict(inc=40, azi=-102)
NI_A = dict(inc=42, azi=-78)
NI_D = dict(inc=42, azi=78)


def fault_unit_vector_LEGACY_BUGGY(slope_deg, aspect_deg):
    """The old construction, kept here ONLY so the regression tests can pin
    its behaviour. `aspect_adj = -aspect + 90` converts a compass bearing to a
    maths angle and then feeds it to the (-sin, cos) pairing that expects an
    anti-clockwise azimuth; the result points along `aspect - 90`."""
    aspect_adj = -aspect_deg + 90
    return np.array([
        np.sin(np.deg2rad(slope_deg)) * np.sin(np.deg2rad(aspect_adj)) * -1,
        np.sin(np.deg2rad(slope_deg)) * np.cos(np.deg2rad(aspect_adj)),
        np.cos(np.deg2rad(slope_deg)),
    ])


# ---------------------------------------------------------------------------
# LOS unit vector
# ---------------------------------------------------------------------------

def test_los_unit_vector_is_unit_length():
    for g in (ASC, DESC):
        assert np.isclose(np.linalg.norm(utils.los_unit_vector(g["inc"], g["azi"])), 1.0)


def test_los_unit_vector_up_component_is_cos_inc():
    for g in (ASC, DESC):
        L = utils.los_unit_vector(g["inc"], g["azi"])
        assert np.isclose(L[2], np.cos(np.deg2rad(g["inc"])))


def test_ascending_looks_west_descending_looks_east():
    """Sentinel-1 is right-looking. Ascending puts the satellite WEST of the
    target, so the ground->satellite vector has a negative East component."""
    assert utils.los_unit_vector(ASC["inc"], ASC["azi"])[0] < 0
    assert utils.los_unit_vector(DESC["inc"], DESC["azi"])[0] > 0


def test_los_unit_vector_handles_left_and_right_looking():
    assert utils.los_unit_vector(SEN_A["inc"], SEN_A["azi"])[0] < 0
    assert utils.los_unit_vector(SEN_D["inc"], SEN_D["azi"])[0] > 0
    assert utils.los_unit_vector(NI_A["inc"], NI_A["azi"])[0] > 0
    assert utils.los_unit_vector(NI_D["inc"], NI_D["azi"])[0] < 0


def test_left_and_right_looking_are_east_mirrored():
    for right in (SEN_A, SEN_D):
        r = utils.los_unit_vector(right["inc"], right["azi"])
        l = utils.los_unit_vector(right["inc"], right["azi"] - 180)
        assert np.isclose(r[0], -l[0])
        assert np.isclose(r[1], -l[1])
        assert np.isclose(r[2], l[2])


@pytest.mark.parametrize("dic", insar_dic_list, ids=lambda d: d["ORBIT_DIRECTION"])
def test_los_unit_vector_matches_opera_los_east_north(dic):
    """GROUND TRUTH: our inc/azi formula must reproduce OPERA's own los_east
    and los_north rasters on the same grid."""
    h5py = pytest.importorskip("h5py")
    rasterio = pytest.importorskip("rasterio")
    le_tif = os.path.join(dic["geometry_dir"], "los_east.tif")
    ln_tif = os.path.join(dic["geometry_dir"], "los_north.tif")
    if not (os.path.exists(dic["geo_file"]) and os.path.exists(le_tif)):
        pytest.skip("geometry products not available on this machine")
    with h5py.File(dic["geo_file"], "r") as h:
        inc = h["incidenceAngle"][:].astype(np.float64)
        azi = h["azimuthAngle"][:].astype(np.float64)
    with rasterio.open(le_tif) as d:
        le = d.read(1).astype(np.float64)
    with rasterio.open(ln_tif) as d:
        ln = d.read(1).astype(np.float64)
    assert le.shape == inc.shape, "geometry tif and geometryGeo.h5 grids differ"
    ok = np.isfinite(inc) & np.isfinite(azi) & np.isfinite(le) & np.isfinite(ln) & (inc != 0)
    assert ok.sum() > 1000
    idx = np.where(ok.ravel())[0][::37]
    inc_f, azi_f = inc.ravel()[idx], azi.ravel()[idx]
    E = -np.sin(np.deg2rad(inc_f)) * np.sin(np.deg2rad(azi_f))
    N = np.sin(np.deg2rad(inc_f)) * np.cos(np.deg2rad(azi_f))
    # tolerance set by float16 storage of the geometry, not by the maths
    assert np.allclose(E, le.ravel()[idx], atol=4e-3)
    assert np.allclose(N, ln.ravel()[idx], atol=4e-3)
    # a wrong convention is a BIAS; float16 rounding is zero-mean scatter
    assert abs(np.mean(E - le.ravel()[idx])) < 5e-4
    assert abs(np.mean(N - ln.ravel()[idx])) < 5e-4


# ---------------------------------------------------------------------------
# Horizontal (compass bearing) unit vector
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bearing,expect", [
    (0.0, (0.0, 1.0)), (90.0, (1.0, 0.0)), (180.0, (0.0, -1.0)), (270.0, (-1.0, 0.0))])
def test_horizontal_unit_vector_cardinals(bearing, expect):
    v = utils.horizontal_unit_vector(bearing)
    assert np.allclose(v[:2], expect, atol=1e-12)
    assert v[2] == 0.0


def test_hayward_strike_points_southeast():
    v = utils.horizontal_unit_vector(STRIKE)
    assert v[0] > 0 and v[1] < 0


def test_strike_is_perpendicular_to_profile_az():
    assert np.isclose(abs(utils.horizontal_unit_vector(STRIKE) @ utils.horizontal_unit_vector(PROFILE_AZ)),
                      0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Design matrix and decomposition
# ---------------------------------------------------------------------------

def test_design_matrix_up_column_is_cos_inc():
    G = utils.design_matrix_up_fp([ASC["inc"], DESC["inc"]], [ASC["azi"], DESC["azi"]], STRIKE)
    assert np.isclose(G[0, 0], np.cos(np.deg2rad(ASC["inc"])))
    assert np.isclose(G[1, 0], np.cos(np.deg2rad(DESC["inc"])))


def test_design_matrix_matches_mintpy_convention():
    """MintPy's get_design_matrix4horz_vert uses sin(inc)*cos(azi - horz_az)
    with horz_az an ANTI-CLOCKWISE azimuth; a clockwise strike converts as
    horz_az = -strike."""
    G = utils.design_matrix_up_fp([ASC["inc"], DESC["inc"]], [ASC["azi"], DESC["azi"]], STRIKE)
    horz_az = -STRIKE
    for row, g in enumerate((ASC, DESC)):
        mintpy = np.sin(np.deg2rad(g["inc"])) * np.cos(np.deg2rad(g["azi"] - horz_az))
        assert np.isclose(G[row, 1], mintpy, atol=1e-12)


def test_design_matrix_is_well_conditioned():
    G = utils.design_matrix_up_fp([ASC["inc"], DESC["inc"]], [ASC["azi"], DESC["azi"]], STRIKE)
    assert np.linalg.cond(G) < 5.0


def _forward(u_enu):
    return (u_enu @ utils.los_unit_vector(ASC["inc"], ASC["azi"]),
            u_enu @ utils.los_unit_vector(DESC["inc"], DESC["azi"]))


def test_roundtrip_recovers_known_up_and_fp():
    for u_true, fp_true in [(0.0, 10.0), (5.0, 0.0), (-3.0, 7.5), (12.0, -4.0)]:
        u_enu = fp_true * utils.horizontal_unit_vector(STRIKE) + np.array([0.0, 0.0, u_true])
        d_a, d_d = _forward(u_enu)
        U, FP = utils.decompose_asc_desc(np.array([d_a]), np.array([d_d]),
                                         ASC["inc"], ASC["azi"], DESC["inc"], DESC["azi"], STRIKE)
        assert np.isclose(U[0], u_true, atol=1e-8)
        assert np.isclose(FP[0], fp_true, atol=1e-8)


def test_right_lateral_creep_gives_positive_fault_parallel():
    """THE sign test. Hayward is right-lateral: the east block moves SE relative
    to the west block. The creep series is (east - west), so that motion is
    +10 mm along bearing 144 -> FP must come out POSITIVE."""
    u_rel = 10.0 * utils.horizontal_unit_vector(STRIKE)
    d_a, d_d = _forward(u_rel)
    U, FP = utils.decompose_asc_desc(np.array([d_a]), np.array([d_d]),
                                     ASC["inc"], ASC["azi"], DESC["inc"], DESC["azi"], STRIKE)
    assert FP[0] > 0
    assert np.isclose(FP[0], 10.0, atol=1e-8)


def test_decompose_handles_multiple_epochs():
    fp = np.array([0.0, 1.0, 2.0, 3.0])
    d_a, d_d = zip(*[_forward(f * utils.horizontal_unit_vector(STRIKE)) for f in fp])
    U, FP = utils.decompose_asc_desc(np.array(d_a), np.array(d_d),
                                     ASC["inc"], ASC["azi"], DESC["inc"], DESC["azi"], STRIKE)
    assert np.allclose(FP, fp, atol=1e-8)
    assert np.allclose(U, 0.0, atol=1e-8)


# ---------------------------------------------------------------------------
# Regression tests pinning the LEGACY bug (so it cannot creep back in)
# ---------------------------------------------------------------------------

def test_legacy_fault_vector_is_90_degrees_off():
    F_bad = fault_unit_vector_LEGACY_BUGGY(90.0, STRIKE)
    F_good = utils.horizontal_unit_vector(STRIKE)
    assert np.isclose(F_bad @ F_good, 0.0, atol=1e-9)
    assert np.allclose(F_bad, utils.horizontal_unit_vector(PROFILE_AZ), atol=1e-9)


def test_legacy_bug_impact_factors():
    """Pins WHY it went unnoticed: descending near-immune, ascending ~2x."""
    F_bad = fault_unit_vector_LEGACY_BUGGY(90.0, STRIKE)
    F_good = utils.horizontal_unit_vector(STRIKE)
    factors = {}
    for name, g in (("asc", ASC), ("desc", DESC)):
        L = utils.los_unit_vector(g["inc"], g["azi"])
        factors[name] = (L @ F_bad) / (L @ F_good)
    assert np.isclose(factors["desc"], 0.986, atol=5e-3)
    assert np.isclose(factors["asc"], 2.053, atol=5e-3)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def test_decimal_year_roundtrip():
    for s in ("2017-02-06", "2020-02-29", "2024-12-31"):
        dy = utils.date_to_decimal_year(s)
        assert utils.decimal_year_to_date(dy).strftime("%Y-%m-%d") == s
    assert utils.date_to_decimal_year(b"20160717") == utils.date_to_decimal_year("2016-07-17")


def test_robust_sigma_matches_std_for_gaussian_and_ignores_outliers():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 2.0, 200000)
    assert np.isclose(utils.robust_sigma(x), 2.0, atol=0.02)
    x[:50] = 1000.0                                  # gross outliers
    assert np.isclose(utils.robust_sigma(x), 2.0, atol=0.02)
    assert np.isnan(utils.robust_sigma([np.nan, np.nan]))


def test_linear_rate_recovers_slope_and_ignores_nan():
    t = np.linspace(2016, 2026, 400)
    y = 6.6 * (t - 2016) + 3.0
    y[::7] = np.nan
    slope, intercept, std_err, n = utils.linear_rate(t, y)
    assert np.isclose(slope, 6.6, atol=1e-9)
    assert n == np.isfinite(y).sum()
    assert np.allclose(utils.detrend(t, y)[np.isfinite(y)], 0.0, atol=1e-9)


def test_render_value():
    assert utils.render_value(10.836, 2) == "10.84"
    assert utils.render_value(25.0, 0) == "25"
    assert utils.render_value("2026-12-15", 0) == "2026-12-15"


def test_mpl_paper_style_sets_helvetica_and_type42():
    import insar_utils as utils
    rc = utils.mpl_paper_style(base=8)
    assert rc["font.sans-serif"][0] == "Arial" and rc["font.family"] == ["sans-serif"]
    assert rc["pdf.fonttype"] == 42 and rc["font.size"] == 8
    import matplotlib.font_manager as fm
    assert "Arial" in fm.findfont(fm.FontProperties(family=["Arial"]))
    assert "Bold" in fm.findfont(fm.FontProperties(family=["Arial"], weight="bold"), fallback_to_default=False)


def test_text_backed_draws_twice_box_then_crisp():
    import insar_utils as utils

    class FakeFig:
        def __init__(self):
            self.calls = []

        def text(self, **kw):
            self.calls.append(kw)

    fig = FakeFig()
    utils.text_backed(fig, x=1.0, y=2.0, text="CFW1", font="7p,Helvetica")
    assert len(fig.calls) == 2
    assert fig.calls[0]["fill"] == "white" and fig.calls[0]["transparency"] == 30
    assert "fill" not in fig.calls[1] and "transparency" not in fig.calls[1]
    assert fig.calls[1]["text"] == "CFW1" and fig.calls[1]["x"] == 1.0


def test_read_creepmeter_10min_raw_matches_daily_reader():
    import os
    import numpy as np
    import insar_utils as utils
    from TulePond_filepaths import common_paths
    f = utils.creepmeter_file(common_paths["creepmeter_dir"], "CFW1")
    if not os.path.exists(f):
        pytest.skip("creepmeter data not available")
    raw = utils.read_creepmeter_10min_raw(f, date_start=2018.0, date_end=2018.1)
    daily = utils.read_creepmeter_10min(f, date_start=2018.0, date_end=2018.1)
    assert list(raw.columns) == ["decimal_year", "slip_mm"]
    assert len(raw) > 100 * len(daily)                       # 10-min samples vs daily means
    assert raw.decimal_year.is_monotonic_increasing
    # the daily reader is the daily mean of the raw samples (before its zeroing to the first day)
    d0 = raw[raw.decimal_year < raw.decimal_year.iloc[0] + 1 / 365.25].slip_mm.mean()
    d1 = daily.slip_mm.iloc[1] + d0
    day1 = raw[(raw.decimal_year >= daily.decimal_year.iloc[1]) & (raw.decimal_year < daily.decimal_year.iloc[1] + 1 / 365.25)]
    assert np.isclose(day1.slip_mm.mean(), d1, atol=0.05)
