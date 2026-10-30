#!/usr/bin/env python3
"""
Anchor tests: every number quoted in the manuscript, at its quoted precision,
against the latest row of the registry. When a number legitimately moves,
edit ANCHORS and the manuscript in the same commit.

The values were seeded on 2026-09-21 from the first full run (coherence 0.7
stream). Run the notebooks first; the test is skipped if the registry is absent.

Run:  pytest tests/test_anchors.py -v
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TulePond_filepaths import common_paths                              # noqa: E402

# key: (value, decimals)  -- strings are compared exactly
ANCHORS = {
    # 02 velocity decomposition
    "02_strike_mean_deg": (144.10, 2),
    "02_xfault_step_CFW1_mm_yr": (6.62, 2),
    # 03 creep time series (coherence 0.7; March-2026 data, refit 2026-09-30)
    "03_n_crossings_asc": (654, 0), "03_n_crossings_des": (629, 0),
    "03_n_acquisitions_asc": (359, 0), "03_n_acquisitions_des": (374, 0),
    "03_n_common_dates": (439, 0), "03_n_both_real": (291, 0),
    "03_interp_pct_asc": (19, 0), "03_interp_pct_desc": (15, 0),
    "03_n_crossings_common": (610, 0),
    # 04 event detection
    "04_n_events": (5, 0),
    "04_onset_lo_2017": (2017.099, 3), "04_onset_lo_2019": (2019.186, 3), "04_onset_lo_2021": (2021.107, 3),
    "04_onset_lo_2023": (2023.080, 3), "04_onset_lo_2024": (2024.984, 3),
    "04_onset_window_days_2017": (6, 0), "04_onset_window_days_2023": (12, 0),
    # the paper quotes the 90% band (Dani, 2026-09-21); the 75% band stays registered for Methods
    "04_band90_lo_km": (58.2, 1), "04_band90_hi_km": (59.8, 1), "04_band90_n_crossings": (17, 0),
    "04_band90_step_2017_mm": (10.63, 2), "04_band90_step_2019_mm": (10.48, 2), "04_band90_step_2021_mm": (11.20, 2),
    "04_band90_step_2023_mm": (11.74, 2), "04_band90_step_2024_mm": (7.73, 2),
    "04_band90_steps_sum_mm": (51.8, 1), "04_band90_inter_event_mm_yr": (0.93, 2), "04_band90_steps_mean_mm": (10.4, 1),
    "04_band75_steps_mean_mm": (10.2, 1), "04_band75_width_km": (2.6, 1), "04_band90_width_km": (1.6, 1),
    "04_creep_length_km": (6.2, 1), "04_creep_length_lo_km": (56.2, 1), "04_creep_length_hi_km": (62.4, 1),
    "04_interval_mean_yr": (1.97, 2), "04_interval_sd_d": (30, 0), "04_interval_cov": (0.04, 2),
    # 06 event field geometry (coherence 0.7, all five events in the mean, 30-150 m steps)
    "06_n_events_mean": (5, 0), "06_asc_desc_geom_ratio": (-0.56, 2),
    "06_fp_rate_step_mm_yr": (1.41, 2), "06_fp_event_step_mm": (8.74, 2),
    "06_XX_rate_step_mm_yr": (1.17, 2), "06_XX_event_step_mm": (9.17, 2),   # X-X' at 58.5 km, local strike, +/- 2 km (2026-09-23)
    "06_YY_rate_step_mm_yr": (6.54, 2), "06_YY_event_step_mm": (0.88, 2),
    # 08a the selected model, rewired 2026-10-30 (model_config.py: order 2, shallow mesh, N^0, 60 m cut, plane, own corner)
    "08a_mw_30gpa": (3.86, 2), "08a_mw_harmonic": (3.26, 2), "08a_rms_mm": (1.15, 2), "08a_rms_west500_mm": (1.44, 2),
    "08a_rms_east500_mm": (1.10, 2), "08a_potency_m3": (25997, 0), "08a_sigma": (43.2, 1), "08a_depth90_m": (933, 0),
    "07a_hom_mw_30gpa": (3.80, 2),
    # 09 alignment arrays: detector on the bimonthly era only, steps under 2 sigma dropped (Dani 2026-10-02);
    # the repeat intervals live only in 11
    "09_hrkt_n_surveys": (159, 0), "09_hrkt_n_events": (9, 0), "09_hrkt_amp_median_mm": (8.2, 1),
    # 11 forecast
    "11_next_clock_date": ("2026-12-15", 0), "11_pi95_lo": ("2026-08-29", 0), "11_pi95_hi": ("2027-04-02", 0),
}


@pytest.fixture(scope="module")
def latest():
    p = common_paths["registry_csv"]
    if not os.path.exists(p):
        pytest.skip("registry not built yet: run the notebooks first")
    df = pd.read_csv(p, dtype=str)
    return df.groupby("key", as_index=False).last().set_index("key")


@pytest.mark.parametrize("key,expect", list(ANCHORS.items()), ids=list(ANCHORS))
def test_anchor(latest, key, expect):
    value, dp = expect
    assert key in latest.index, f"{key} not registered"
    got = latest.loc[key, "rendered"]
    if isinstance(value, str):
        assert got == value
    else:
        assert round(float(got), dp) == round(float(value), dp), f"{key}: registry {got}, manuscript {value}"
