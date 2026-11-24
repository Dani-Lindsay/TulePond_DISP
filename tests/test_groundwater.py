#!/usr/bin/env python3
"""
Tests of groundwater_utils (notebook 16): the ACWD report table parser on real rows, the well position from the
State Well Number, the unit identities, the side of the fault and the selection.

Run:  pytest tests/test_groundwater.py -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import groundwater_utils as gw                                         # noqa: E402
from TulePond_filepaths import common_paths                            # noqa: E402

# Real rows of the 2025 report (Appendix E, Spring 2025, pdftotext -layout) and the 2021 report (Appendix E)
ROWS_2025 = """\
4S/1W-27D008   AHF Indicator         AHF       ALAMEDA COUNTY WATER DISTRICT   3/25/2025       66.59       47.22                    --       -- PWC
4S/1W-29A006   BHF Indicator         N         ALAMEDA COUNTY WATER DISTRICT         3/25/2025        61.23      18.31                      --        -- PWC
4S/1W-18N004                         C         Eleanor Kabrich                  4/3/2025        41.6        3.54  4/3/2025         79       400 NMP UTM UTS NSP
4S/1W-28C001   Mowry #1              N         ALAMEDA COUNTY WATER DISTRICT         3/27/2025        64.81            3/24/2025           73        480 RUN
page header line that is not a well row
"""
ROW_2021_SAMPLE_ONLY = \
    "4S/1W-27P002                          AHF       CITY OF FREMONT                                       52.65              4/9/2021        129      950"


def test_parser_real_rows():
    t = gw.parse_table(ROWS_2025).set_index("well")
    assert list(t.index) == ["4S/1W-27D008", "4S/1W-29A006", "4S/1W-18N004", "4S/1W-28C001"]
    assert t.loc["4S/1W-27D008", "aquifer"] == "AHF"
    assert t.loc["4S/1W-27D008", "reference_elevation_ft"] == 66.59
    assert t.loc["4S/1W-27D008", "water_elevation_ft"] == 47.22
    assert t.loc["4S/1W-29A006", "aquifer"] == "N" and t.loc["4S/1W-29A006", "water_elevation_ft"] == 18.31
    assert not gw.is_level(t.loc["4S/1W-18N004", "water_elevation_ft"])          # UTM: the number is not a level
    assert "RUN" in t.loc["4S/1W-28C001", "remarks"]
    assert not gw.is_level(t.loc["4S/1W-28C001", "water_elevation_ft"])          # no level after the reference elevation


def test_parser_sample_date_is_not_a_measurement():
    t = gw.parse_table(ROW_2021_SAMPLE_ONLY)
    assert t.date.iloc[0] is None and not gw.is_level(t.water_elevation_ft.iloc[0])


def test_static_levels_drop_run_and_missing():
    levels = gw.static_levels(gw.parse_table(ROWS_2025))
    assert set(levels.index) == {"4S/1W-27D008", "4S/1W-29A006"}


def test_tract_centre_first_section():
    # T1S R1W section 1 is the north-east section of the township whose north-east corner is the initial point;
    # tract A is its north-east 40 acres, so the centre is 1/8 mile west and 1/8 mile south of the initial point.
    lat, lon = gw.tract_centre("1S/1W-01A001")
    x, y = gw.local_xy(lat, lon)
    assert x == pytest.approx(-gw.MILE_M / 8) and y == pytest.approx(-gw.MILE_M / 8)


def test_tract_centre_boustrophedon():
    # section 6 is the north-west section of the first row, section 7 the west section of the second row
    x6, y6 = gw.local_xy(*gw.tract_centre("1S/1W-06D001"))
    x7, y7 = gw.local_xy(*gw.tract_centre("1S/1W-07D001"))
    assert x6 == pytest.approx(-6 * gw.MILE_M + gw.MILE_M / 8)
    assert x7 == pytest.approx(x6) and y7 == pytest.approx(y6 - gw.MILE_M)
    # tract R is the south-east 40 acres: 7/8 mile east and 7/8 mile south of the section's north-west corner
    xr, yr = gw.local_xy(*gw.tract_centre("1S/1W-06R001"))
    assert xr - x6 == pytest.approx(0.75 * gw.MILE_M) and y6 - yr == pytest.approx(0.75 * gw.MILE_M)


def test_tract_centre_not_a_well_number():
    assert gw.tract_centre("Mowry #1") is None


def test_units():
    assert gw.pressure_kpa(1.0) == pytest.approx(9.80665)                 # rho_w g per metre
    assert gw.winter_rise_m(100.0, 110.0) == pytest.approx(3.048)         # 10 ft
    assert gw.strength_change_kpa(10.0, 0.4) == pytest.approx(4.0)


def test_side_of_fault():
    assert gw.side_of_fault("AHF") == "east"
    assert all(gw.side_of_fault(c) == "west" for c in ("N", "C", "F", "D", "CF", "NC"))


def test_select_wells_limits():
    r = pd.DataFrame({"dist_km": [0.5, 2.0, 2.01, 1.0], "km": [56.3, 62.2, 59.0, 62.3]})
    s = gw.select_wells(r, 2.0, (56.3, 62.2))
    assert list(s.index) == [0, 1]


def test_report_files():
    archive = common_paths["acwd_gwmr_archive"]
    if not os.path.isdir(archive):
        pytest.skip("ACWD GWMR archive not present")
    files = gw.report_files(archive)
    assert sorted(files) == [2015, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2025]
    assert os.path.basename(files[2017]) == "2017 GWMR.pdf"
