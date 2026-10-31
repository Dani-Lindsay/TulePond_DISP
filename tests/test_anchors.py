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
    # 03 creep time series (coherence 0.7)
    "03_n_crossings_asc": (654, 0), "03_n_crossings_des": (629, 0),
    "03_n_acquisitions_asc": (359, 0), "03_n_acquisitions_des": (374, 0),
    "03_n_common_dates": (439, 0), "03_n_both_real": (291, 0),
    "03_interp_pct_asc": (19, 0), "03_interp_pct_desc": (15, 0),
    "03_n_crossings_common": (611, 0),
    # 04 event detection
    "04_n_events": (5, 0),
    "04_onset_lo_2017": (2017.099, 3), "04_onset_lo_2019": (2019.186, 3), "04_onset_lo_2021": (2021.107, 3),
    "04_onset_lo_2023": (2023.080, 3), "04_onset_lo_2024": (2024.984, 3),
    "04_onset_window_days_2017": (12, 0), "04_onset_window_days_2023": (12, 0),
    # the 90 % band of the paper
    "04_band90_lo_km": (58.2, 1), "04_band90_hi_km": (59.8, 1), "04_band90_n_crossings": (17, 0),
    "04_band90_step_2017_mm": (10.69, 2), "04_band90_step_2019_mm": (10.47, 2), "04_band90_step_2021_mm": (11.25, 2),
    "04_band90_step_2023_mm": (11.73, 2), "04_band90_step_2024_mm": (7.74, 2),
    "04_band90_steps_sum_mm": (51.9, 1), "04_band90_inter_event_mm_yr": (0.96, 2), "04_band90_steps_mean_mm": (10.4, 1),
    "04_band90_width_km": (1.6, 1),
    "04_creep_length_km": (5.9, 1), "04_creep_length_lo_km": (56.3, 1), "04_creep_length_hi_km": (62.2, 1),
    # 06 event field geometry (coherence 0.7, all five events in the mean, 30-150 m steps)
    "06_n_events_mean": (5, 0), "06_asc_desc_geom_ratio": (-0.56, 2),
    "06_fp_rate_step_mm_yr": (0.98, 2), "06_fp_event_step_mm": (9.63, 2),
    "06_XX_rate_step_mm_yr": (0.71, 2), "06_XX_event_step_mm": (10.15, 2),   # X-X' at 58.5 km, local strike, +/- 2 km
    "06_YY_rate_step_mm_yr": (6.56, 2), "06_YY_event_step_mm": (1.20, 2),
    # 09 the selected 3D model (70 W fault, model_config.py); 08 the homogeneous model
    "09_mw_30gpa": (3.91, 2), "09_mw_harmonic": (3.27, 2), "09_rms_mm": (0.89, 2), "09_rms_west500_mm": (0.85, 2),
    "09_rms_east500_mm": (0.98, 2), "09_potency_m3": (30671, 0), "09_sigma": (49.6, 1), "09_depth90_m": (695, 0),
    "08_hom_mw_30gpa": (3.80, 2),
    # 11 alignment arrays: detector on the bimonthly era only, steps under 2 sigma dropped
    "11_hrkt_n_surveys": (159, 0), "11_hrkt_n_events": (9, 0), "11_hrkt_amp_median_mm": (8.2, 1),
    # 13 forecast (interval statistics from the onset-window draws)
    "13_insar_interval_mean_yr": (1.97, 2), "13_insar_interval_sd_d": (29, 0), "13_insar_interval_cov": (0.04, 2),
    "13_next_clock_date": ("2026-12-21", 0), "13_pi95_lo": ("2026-09-08", 0), "13_pi95_hi": ("2027-04-04", 0),
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
