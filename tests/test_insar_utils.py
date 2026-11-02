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

from datetime import datetime

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


def test_design_matrix_equals_los_dot_products():
    """The explicit trig form of design_matrix_up_fp equals the geometric derivation:
    each row is [L . up, L . strike_hat] with L the ground -> satellite LOS unit vector."""
    G = utils.design_matrix_up_fp([ASC["inc"], DESC["inc"]], [ASC["azi"], DESC["azi"]], STRIKE)
    strike_hat = utils.horizontal_unit_vector(STRIKE)
    for row, g in enumerate((ASC, DESC)):
        L = utils.los_unit_vector(g["inc"], g["azi"])
        assert np.isclose(G[row, 0], L @ np.array([0.0, 0.0, 1.0]), atol=1e-12)
        assert np.isclose(G[row, 1], L @ strike_hat, atol=1e-12)


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


def test_decimal_year_differences_are_exact_days():
    """A difference of decimal years times 365.25 is the exact time between them, across leap years and new year."""
    pairs = [(datetime(2017, 2, 6), datetime(2019, 3, 10), 762.0),                    # the first two onset days
             (datetime(2024, 12, 26, 14, 8), datetime(2025, 1, 7, 2, 8), 11.5),     # a window across the new year
             (datetime(2020, 2, 28), datetime(2020, 3, 1), 2.0)]                       # across 29 Feb
    for t0, t1, days in pairs:
        assert np.isclose((utils.date_to_decimal_year(t1) - utils.date_to_decimal_year(t0)) * 365.25, days, rtol=0, atol=1e-9)
    assert utils.date_to_decimal_year("2000-01-01") == 2000.0
    assert np.allclose(utils.decimal_years(["2017-02-06 02:07:06", "2017-02-18 14:07:33"]),
                       [utils.date_to_decimal_year(datetime(2017, 2, 6, 2, 7, 6)), utils.date_to_decimal_year(datetime(2017, 2, 18, 14, 7, 33))],
                       rtol=0, atol=1e-12)


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


def test_annual_increment_rate_steady_and_stepped():
    # steady creep: every year gains the rate, so the sigma is zero
    t = np.arange(2016.0, 2025.01, 1 / 365.25)        # nine whole years
    rate, sigma, n = utils.annual_increment_rate(t, 7.0 * (t - 2016))
    assert np.isclose(rate, 7.0, atol=1e-6) and sigma < 1e-6 and n == 9
    # all slip in steps: 14 mm every other year -> yearly gains 14, 0, 14, ... (n = 9)
    y = 14.0 * np.floor((t - 2016 + 0.5) / 2)
    rate, sigma, n = utils.annual_increment_rate(t, y)
    gains = np.array([0, 14, 0, 14, 0, 14, 0, 14, 0], float)
    assert np.isclose(rate, gains.mean()) and np.isclose(sigma, gains.std(ddof=1) / 3.0)


def test_render_value():
    assert utils.render_value(10.836, 2) == "10.84"
    assert utils.render_value(25.0, 0) == "25"
    assert utils.render_value("2026-12-15", 0) == "2026-12-15"


def test_mpl_paper_style_sets_helvetica_and_type42():
    import insar_utils as utils
    rc = utils.mpl_paper_style(base=8)
    family = rc["font.sans-serif"][0]
    assert family in ("Helvetica", "Arial") and rc["font.family"] == ["sans-serif"]      # Helvetica when its bold face registers, else Arial
    assert rc["pdf.fonttype"] == 42 and rc["font.size"] == 8
    assert rc["mathtext.bf"] == f"{family}:bold"                                          # the bold panel letters use the same face
    import matplotlib.font_manager as fm
    assert family in fm.findfont(fm.FontProperties(family=[family]))
    assert "Bold" in fm.findfont(fm.FontProperties(family=[family], weight="bold"), fallback_to_default=False)


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


def test_gaussian_smooth_km_constant_gaps_and_width():
    km = 70.0 - 0.1 * np.arange(200)                              # crossings every 0.1 km along the trace, south -> north
    x = np.full(200, 5.0); x[50:53] = np.nan
    out = utils.gaussian_smooth_km(x, km, 0.15)
    assert np.allclose(out[np.isfinite(x)], 5.0) and np.isnan(out[50:53]).all()   # constant kept, gaps stay gaps
    spike = np.zeros(200); spike[100] = 1.0
    w = utils.gaussian_smooth_km(spike, km, 0.15)
    assert np.isclose(w.sum(), 1.0, atol=1e-6) and np.isclose(np.sqrt(np.sum(w * (np.arange(200) - 100) ** 2)) * 0.1, 0.15, atol=0.01)


def test_gaussian_smooth_km_is_along_trace_and_rejects_latitudes():
    """sigma is km ALONG THE TRACE: a missing crossing is a hole, not a closer neighbour (the spike's
    neighbours across a 1 km hole get the weight of 1 km, not of 0.1 km), and latitudes (the old
    signature, spacing ~0.001) fail loud."""
    km = np.r_[np.arange(0.0, 5.0, 0.1), np.arange(6.0, 11.0, 0.1)]       # a 1 km hole between 4.9 and 6.0
    spike = np.zeros(km.size); spike[49] = 1.0                             # the crossing at 4.9 km, at the hole's edge
    w = utils.gaussian_smooth_km(spike, km, 0.3)
    assert w[50] < 1e-2 * w[49]                                            # 6.0 km is 1.1 km away (3.7 sigma): next to no weight
    assert w[48] > 0.5 * w[49]                                             # 4.8 km is 0.1 km away: full neighbour
    lats = 37.5 + 0.1 / 111.19 * np.arange(200)
    with pytest.raises(AssertionError, match="not latitude"):
        utils.gaussian_smooth_km(np.ones(200), lats, 0.15)
    with pytest.raises(AssertionError, match="fault order"):
        utils.gaussian_smooth_km(np.ones(4), np.array([0.0, 0.1, 0.3, 0.2]), 0.15)


def test_daily_creepmeter_drops_listed_glitches_and_zeroes():
    """The shared daily reader of notebooks 04 and 05: starts at zero, drops exactly the listed glitch days
    (no automatic filter), keeps a creep step, and fails on a listed day that is not in the record."""
    import os
    import tempfile
    rows = []
    for day in range(1, 61):                                   # 60 days of 10-minute samples, 0.1 mm/day creep
        for sample in range(144):
            doy = day + sample / 144.0
            slip = 100.0 + 0.1 * doy + 4.0 * (doy >= 40.5)        # and a 4 mm creep step half way through day 40
            rows.append((2018, doy, slip))
    rows[144 * 30] = (2018, 31.0, -338.0)                      # a CPP1-style spike on record day 31
    day_31 = "2018-01-31"                                      # record day 31: the doy is 1-based, so 31 January
    with tempfile.TemporaryDirectory() as folder:
        with open(os.path.join(folder, "test.10min"), "w") as f:
            for year, doy, slip in rows:
                f.write(f"{year} {doy:.6f} {slip:.3f}\n")
        dates, slip = utils.daily_creepmeter(folder, "TEST", 2018.0, 2019.0, [("TEST", day_31, day_31, "spike")])
        assert slip[0] == 0.0
        assert len(dates) == 59                                # only the listed day is dropped
        assert not np.any(np.isclose((dates - 2018.0) * 365, 30.0, rtol=0, atol=0.5))   # 31 January = 2018 + 30/365
        assert np.all(np.diff(dates) > 0)
        assert np.isclose(slip[-1], 0.1 * 59 + 4.0, atol=0.01)    # the creep and the whole step are kept
        unlisted_dates, _ = utils.daily_creepmeter(folder, "TEST", 2018.0, 2019.0, [("OTHER", day_31, day_31, "spike")])
        assert len(unlisted_dates) == 60                       # another instrument's glitch does not touch this one
        with pytest.raises(AssertionError):                    # a listed day missing from the record fails loud
            utils.daily_creepmeter(folder, "TEST", 2018.0, 2019.0, [("TEST", "2018-06-01", "2018-06-01", "gap")])


def test_creepmeter_day_of_year_is_one_based():
    """USGS creepmeter doy 1.000 is 1 January 00:00 (cfw1.10min.gz: '2020 1.000000000' is the first sample of
    2020, '2019 365.993055556' the last of 2019), so it maps to year + 0, as date_to_decimal_year does."""
    assert utils.creepmeter_decimal_year(np.array([2020]), np.array([1.0]))[0] == 2020.0
    assert np.isclose(utils.creepmeter_decimal_year(np.array([2019]), np.array([365.993055556]))[0],
                      utils.date_to_decimal_year(datetime(2019, 12, 31, 23, 50)), rtol=0, atol=1e-7)
    assert np.isclose(utils.creepmeter_decimal_year(np.array([2020]), np.array([60.5]))[0],     # leap year: 29 Feb 12:00
                      utils.date_to_decimal_year(datetime(2020, 2, 29, 12)), rtol=0, atol=1e-9)
    with pytest.raises(AssertionError):
        utils.creepmeter_decimal_year(np.array([2019]), np.array([0.5]))                      # doy 0.5 does not exist


def test_load_aa_is_one_day_earlier_than_the_archive_year():
    """The archive's decimal year is year + doy/days with a 1-based doy; load_aa takes the day off, so a survey
    lands on its calendar date (HRKT '1979.726, 9/22/1979'), and fails loud on a file in the other convention."""
    import tempfile
    from TulePond_filepaths import common_paths
    if os.path.exists(os.path.join(common_paths["aa_dir"], "HRKT.csv")):
        hrkt = utils.load_aa(common_paths["aa_dir"], "HRKT")
        assert utils.decimal_year_to_day(hrkt.year.iloc[0]).date().isoformat() == "1979-09-22"
    header = "Site_code,Measurement_yr,Measurement_date,Cumulative_dextral_movement_mm,Dextral_movement_st_dev_mm\n"
    days = ["3/1/2018", "6/15/2018", "9/30/2018", "12/31/2018", "1/1/2019"]

    def archive_year(day, first_day_of_year):
        """The archive's calendar decimal year: year + (day of year - 1 + first_day_of_year) / days in the year."""
        t = datetime.strptime(day, "%m/%d/%Y")
        days_in_year = 366 if t.year % 4 == 0 else 365
        return t.year + (t.timetuple().tm_yday - 1 + first_day_of_year) / days_in_year

    with tempfile.TemporaryDirectory() as folder:
        with open(os.path.join(folder, "TST1.csv"), "w") as f:                 # the archive's 1-based convention
            f.write(header + "".join(f"TST1,{archive_year(day, 1):.6f},{day},{k},0.1\n" for k, day in enumerate(days)))
        aa = utils.load_aa(folder, "TST1")
        assert np.allclose(aa.year, [utils.date_to_decimal_year(day) for day in days], rtol=0, atol=1e-5)
        with open(os.path.join(folder, "TST2.csv"), "w") as f:                 # already 0-based: must fail loud
            f.write(header + "".join(f"TST2,{archive_year(day, 0):.6f},{day},{k},0.1\n" for k, day in enumerate(days)))
        with pytest.raises(AssertionError, match="1-based"):
            utils.load_aa(folder, "TST2")


def test_render_retires_keys_of_a_run_that_registered_nothing(tmp_path):
    """A notebook whose latest run wrote its sidecar but registered no number has all its older keys left out of
    paper_numbers.md; a run that wrote a sidecar and registered keeps them."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = tmp_path / "outputs"
    paths = {"out_dir": str(out), "registry_csv": str(out / "registry.csv"), "paper_numbers": str(out / "paper_numbers.md")}
    first = utils.Registry("05_validation", paths, repo)                   # a notebook that exists in the repo
    first.sidecar("05_validation", ["first run"])
    first.register("05_kept_if_rerun", 1, "count", 0, "P3", "a")
    assert "`05_kept_if_rerun`" in open(first.render()).read()              # sidecar + registration: listed
    silent = utils.Registry("05_validation", paths, repo)                  # the next run registers nothing
    silent.sidecar("05_validation", ["second run"])
    assert "05_kept_if_rerun" not in open(silent.render()).read()


def test_strike_frame_origin_axes_and_roundtrip():
    """u = 0 at the northern end of the trace (Point Pinole), u grows along the strike bearing, v toward
    the north-east; to_uv and uv_to_lonlat invert each other."""
    strike_deg = 144.0
    trace_lat = np.array([37.50, 37.60, 37.70, 37.80])          # a short trace running NW to SE, listed south to north
    trace_lon = np.array([-121.95, -122.02, -122.10, -122.17])
    frame = utils.strike_frame(32610, strike_deg, trace_lon, trace_lat)
    u0, v0 = utils.to_uv(frame, trace_lon[-1], trace_lat[-1])
    assert abs(float(u0)) < 1e-9                                # the northern end is the origin of u
    u_south, _ = utils.to_uv(frame, trace_lon[0], trace_lat[0])
    assert float(u_south) > 0                                   # u increases to the south-east
    lon_1, lat_1 = utils.uv_to_lonlat(frame, 10.0, 0.0)         # 10 km along strike
    lon_e, lat_e = utils.uv_to_lonlat(frame, 10.0, 2.0)         # then 2 km across: north-east, the east side
    assert lat_e > lat_1 and lon_e > lon_1
    u_list = np.array([0.0, 5.0, 33.3, -3.0])
    v_list = np.array([0.0, -2.0, 7.5, 10.9])
    lon_r, lat_r = utils.uv_to_lonlat(frame, u_list, v_list)
    u_back, v_back = utils.to_uv(frame, lon_r, lat_r)
    assert np.allclose(u_back, u_list, atol=1e-6) and np.allclose(v_back, v_list, atol=1e-6)


def test_strike_frame_matches_the_notebook_construction():
    """The shared frame gives the same numbers as the construction 04 and 05 used to write out inline."""
    import pyproj
    strike_deg = 144.0
    trace_lat = np.array([37.50, 37.60, 37.70, 37.80])
    trace_lon = np.array([-121.95, -122.02, -122.10, -122.17])
    utm = pyproj.Transformer.from_crs(4326, 32610, always_xy=True)
    strike_rad = np.deg2rad(strike_deg)
    along_e, along_n = np.sin(strike_rad), np.cos(strike_rad)
    cross_e, cross_n = -np.cos(strike_rad), np.sin(strike_rad)
    trace_e, trace_n = utm.transform(trace_lon, trace_lat)
    u_origin = float((trace_e * along_e + trace_n * along_n)[int(np.argmax(trace_lat))])
    v_origin = float(np.median(trace_e * cross_e + trace_n * cross_n))
    lon, lat = np.array([-122.05, -121.99]), np.array([37.65, 37.55])
    easting, northing = utm.transform(lon, lat)
    u_inline = ((easting * along_e + northing * along_n) - u_origin) / 1e3
    v_inline = ((easting * cross_e + northing * cross_n) - v_origin) / 1e3
    frame = utils.strike_frame(32610, strike_deg, trace_lon, trace_lat)
    u_shared, v_shared = utils.to_uv(frame, lon, lat)
    assert np.allclose(u_shared, u_inline, atol=1e-12) and np.allclose(v_shared, v_inline, atol=1e-12)
