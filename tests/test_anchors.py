#!/usr/bin/env python3
"""
Anchor tests: the headline numbers of the paper, at their quoted precision,
against the latest row of the registry. When a number legitimately moves,
edit ANCHORS and the manuscript in the same commit. Run the notebooks first;
the test is skipped if the registry is absent.

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
    # 01 the GNSS tie
    "01_gnss_n_stations_area": (52, 0), "01_gnss_n_stations_screened": (49, 0),
    "01_gnss_zoa1_shift_asc_mm_yr": (0.92, 2), "01_gnss_zoa1_shift_des_mm_yr": (0.78, 2),     # M1, run05 corr_ref (25 Nov 2026)
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
    "04_onset_lo_2017": (2017.101, 3), "04_onset_lo_2019": (2019.187, 3), "04_onset_lo_2021": (2021.109, 3),
    "04_onset_lo_2023": (2023.080, 3), "04_onset_lo_2024": (2024.986, 3),
    "04_onset_window_days_2017": (12.5, 1), "04_onset_window_days_2019": (6.0, 1), "04_onset_window_days_2021": (5.5, 1),
    "04_onset_window_days_2023": (11.5, 1), "04_onset_window_days_2024": (11.5, 1),
    # the 90 % band of the paper
    "04_band90_lo_km": (58.2, 1), "04_band90_hi_km": (60.2, 1), "04_band90_n_crossings": (21, 0),
    "04_band90_step_2017_mm": (10.74, 2), "04_band90_step_2019_mm": (10.81, 2), "04_band90_step_2021_mm": (10.89, 2),
    "04_band90_step_2023_mm": (11.53, 2), "04_band90_step_2024_mm": (8.03, 2),
    "04_band90_steps_sum_mm": (52.0, 1), "04_band90_inter_event_mm_yr": (1.09, 2), "04_band90_steps_mean_mm": (10.4, 1),
    "04_band90_width_km": (2.0, 1),
    "04_creep_length_km": (5.9, 1), "04_creep_length_lo_km": (56.3, 1), "04_creep_length_hi_km": (62.2, 1),
    # M3: InSAR rates against Johnson et al. (2022) (25 Nov 2026)
    "04_johnson_aa_n_sites": (46, 0), "04_johnson_aa_median_diff_mm_yr": (0.0, 1), "04_johnson_aa_rms_mm_yr": (0.8, 1),
    "04_johnson_sar_n_sites": (6, 0), "04_johnson_sar_median_diff_mm_yr": (1.5, 1), "04_johnson_sar_rms_mm_yr": (1.8, 1),
    # 05 validation: M4, COZ1 against notebook 04's detection stack (25 Nov 2026)
    "05_coz1_detect_stack_n": (31, 0), "05_coz1_detect_stack_det_r": (0.84, 2), "05_coz1_detect_stack_det_rmse_mm": (1.1, 1),
    # 06 event field geometry (coherence 0.7, all five events in the mean, 30-150 m steps)
    "06_n_events_mean": (5, 0), "06_asc_desc_geom_ratio": (-0.56, 2),
    "06_fp_rate_step_mm_yr": (0.95, 2), "06_fp_event_step_mm": (9.75, 2),
    "06_XX_rate_step_mm_yr": (0.71, 2), "06_XX_event_step_mm": (10.15, 2),   # X-X' at 58.5 km, local strike, +/- 2 km
    "06_YY_rate_step_mm_yr": (6.56, 2), "06_YY_event_step_mm": (1.20, 2),
    # 09 the selected 3D model (70 W fault, model_config.py); 08 the homogeneous model
    "09_mw_30gpa": (3.91, 2), "09_mw_harmonic": (3.27, 2), "09_rms_mm": (0.89, 2), "09_rms_west500_mm": (0.85, 2),
    "09_rms_east500_mm": (0.98, 2), "09_potency_m3": (30671, 0), "09_sigma": (49.6, 1), "09_depth90_m": (695, 0),
    "08_hom_mw_30gpa": (3.80, 2),
    "09_stress_drop_30gpa_kpa": (353, 0), "09_stress_drop_harmonic_kpa": (27, 0),
    # 11 alignment arrays: detector on the bimonthly era only, steps under 2 sigma dropped
    "11_hrkt_n_surveys": (159, 0), "11_hrkt_n_events": (9, 0), "11_hrkt_amp_median_mm": (8.2, 1),
    # M8: survey intervals since 2015 with and without an InSAR onset (25 Nov 2026)
    "11_hgil_event_median_mm": (14.1, 1), "11_hgil_quiet_median_mm": (0.4, 1), "11_hrkt_event_median_mm": (12.0, 1),
    "11_hrkt_quiet_median_mm": (3.8, 1), "11_hhan_event_median_mm": (9.7, 1), "11_hhan_quiet_median_mm": (5.5, 1),
    "11_happ_event_median_mm": (7.4, 1), "11_happ_quiet_median_mm": (6.6, 1), "11_hpin_event_median_mm": (5.1, 1),
    "11_hpin_quiet_median_mm": (5.1, 1),
    # 12 rain: M9, rule A over its thresholds (25 Nov 2026)
    "12_ruleA_thr_lo_mm": (50, 0), "12_ruleA_thr_hi_mm": (150, 0), "12_ruleA_spread_min_d": (78, 0),
    "12_ruleA_spread_max_d": (110, 0), "12_ruleA_best_Y_mm": (125, 0), "12_ruleA_best_aa_consistent": (7, 0),
    "12_ruleA_best_aa_reached": (9, 0),
    # 13 forecast (interval statistics from the onset-window draws)
    "13_insar_interval_mean_yr": (1.97, 2), "13_insar_interval_sd_d": (29, 0), "13_insar_interval_cov": (0.04, 2),
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
