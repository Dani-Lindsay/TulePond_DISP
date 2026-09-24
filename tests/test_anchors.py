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
    "02_xfault_step_CFW1_mm_yr": (6.68, 2),
    # 03 creep time series (coherence 0.7)
    "03_n_crossings_asc": (656, 0), "03_n_crossings_des": (633, 0),
    "03_n_acquisitions_asc": (344, 0), "03_n_acquisitions_des": (359, 0),
    "03_n_common_dates": (423, 0), "03_n_both_real": (277, 0),
    "03_interp_pct_asc": (19, 0), "03_interp_pct_desc": (15, 0),
    "03_n_crossings_common": (609, 0),
    # 04 event detection
    "04_n_events": (5, 0),
    "04_onset_2017": (2017.099, 3), "04_onset_2019": (2019.186, 3), "04_onset_2021": (2021.107, 3),
    "04_onset_2023": (2023.080, 3), "04_onset_2024": (2024.984, 3),
    "04_onset_window_days_2017": (6, 0), "04_onset_window_days_2023": (12, 0),
    # the paper quotes the 90% band (Dani, 2026-09-21); the 75% band stays registered for Methods
    "04_band90_lo_km": (58.2, 1), "04_band90_hi_km": (59.8, 1), "04_band90_n_crossings": (17, 0),
    "04_band90_step_2017_mm": (10.62, 2), "04_band90_step_2019_mm": (10.45, 2), "04_band90_step_2021_mm": (11.17, 2),
    "04_band90_step_2023_mm": (11.82, 2), "04_band90_step_2024_mm": (7.62, 2),
    "04_band90_steps_sum_mm": (51.7, 1), "04_band90_inter_event_mm_yr": (0.94, 2), "04_band90_steps_mean_mm": (10.3, 1),
    "04_band75_steps_mean_mm": (10.1, 1), "04_band75_width_km": (2.7, 1), "04_band90_width_km": (1.6, 1),
    "04_creep_length_km": (6.0, 1), "04_creep_length_lo_km": (56.3, 1), "04_creep_length_hi_km": (62.3, 1),
    "04_interval_mean_yr": (1.97, 2), "04_interval_sd_d": (30, 0), "04_interval_cov": (0.04, 2),
    # 06 event field geometry (coherence 0.7, all five events in the mean, 30-150 m steps)
    "06_n_events_mean": (5, 0), "06_asc_desc_geom_ratio": (-0.60, 2),
    "06_fp_rate_step_mm_yr": (1.21, 2), "06_fp_event_step_mm": (7.97, 2),
    "06_XX_rate_step_mm_yr": (0.96, 2), "06_XX_event_step_mm": (8.63, 2),   # X-X' at 58.5 km, local strike, +/- 2 km (2026-09-23)
    "06_YY_rate_step_mm_yr": (6.16, 2), "06_YY_event_step_mm": (0.75, 2),
    # 08 alignment arrays
    "08_hrkt_n_surveys": (177, 0), "08_hrkt_n_events": (14, 0), "08_hrkt_amp_median_mm": (7.4, 1),
    "08_aa_interval_median_yr": (2.8, 1), "08_aa_interval_n": (13, 0),
    "08_insar_interval_median_yr": (1.9, 1),
    # 10 forecast (was 09_* until the split of 2026-09-24)
    "10_next_clock_date": ("2026-12-15", 0), "10_pi95_lo": ("2026-08-29", 0), "10_pi95_hi": ("2027-04-02", 0),
    "10_ruleC_Z_mm": (40, 0), "10_ruleC_X_d": (21, 0), "10_ruleC_p_chance": (0.02, 2),
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
