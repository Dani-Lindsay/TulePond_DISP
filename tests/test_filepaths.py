#!/usr/bin/env python3
"""
Tests for TulePond_filepaths.py: every input path the notebooks read exists on
this machine, the track dictionaries share one key vocabulary, and the
along-fault axis (km from Point Pinole along the curved trace) is pinned
against the real hayward_fault_profiles.csv:

  * zero at the northern end (Point Pinole, index 790, 37.9915 N)
  * increasing southward, total length 79.0 km
  * equivalence with (790 - index) * 0.1 (the 100 m resampling guarantee)

Run:  pytest tests/test_filepaths.py -v      (env: opera_disp_feb13)
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import insar_utils as utils                                          # noqa: E402
import TulePond_filepaths as fp                                      # noqa: E402


REQUIRED_INPUTS = ["fault_trace", "fault_profiles", "creepmeter_locations", "creepmeter_dir",
                   "aa_dir", "aa_summary", "creep_rates_Johnson22", "ghcn_dir", "CRE_Taira", "CRE_Wald", "growclust_cat", "sweep_dir"]


@pytest.mark.parametrize("key", REQUIRED_INPUTS)
def test_common_input_exists(key):
    assert os.path.exists(fp.common_paths[key]), f"{key} -> {fp.common_paths[key]}"


@pytest.mark.parametrize("dic", fp.insar_dic_list, ids=lambda d: d["Frame"])
def test_track_products_exist(dic):
    for key in ("geo_file", "ts_file", "vel_file", "vel_NA_file"):
        assert os.path.exists(dic[key]), f"{dic['Frame']} {key} -> {dic[key]}"
    for c, m in dic["mask_file"].items():
        assert os.path.exists(m), f"{dic['Frame']} mask {c} -> {m}"


def test_track_dicts_share_key_vocabulary():
    keys = [set(d.keys()) for d in fp.insar_dic_list]
    assert all(k == keys[0] for k in keys)
    assert {d["ORBIT_DIRECTION"] for d in fp.insar_dic_list} == {"ascending", "descending"}


def test_mosaic_products_exist():
    for orbit, d in fp.paths_mosaic.items():
        for key, path in d.items():
            assert os.path.exists(path), f"{orbit} {key} -> {path}"


def test_params_string_encodes_the_aperture():
    assert fp.params == f"ap30-150m_coh{fp.coh_timeseries}_mp{fp.min_points}_sw200m"


@pytest.fixture(scope="module")
def trace():
    df = pd.read_csv(fp.common_paths["fault_profiles"])
    return df["ft_lon"].values, df["ft_lat"].values


def test_trace_shape_and_ends(trace):
    ft_lon, ft_lat = trace
    assert len(ft_lat) == 791
    assert ft_lat[0] == pytest.approx(37.4141, abs=1e-3)    # south end
    assert ft_lat[790] == pytest.approx(37.9915, abs=1e-3)  # Point Pinole


def test_zero_at_pinole_increasing_southward(trace):
    km = utils.along_fault_km(*trace)
    assert km[790] == 0.0
    assert np.all(np.diff(km) < 0)
    assert km[0] == pytest.approx(79.0, abs=0.1)


def test_matches_100m_index_rule(trace):
    km = utils.along_fault_km(*trace)
    index_rule = (790 - np.arange(791)) * 0.1
    assert np.abs(km - index_rule).max() < 0.010


def test_lat_to_km_map_matches_geodesic(trace):
    ft_lon, ft_lat = trace
    km_geo = utils.along_fault_km(ft_lon, ft_lat)
    f = utils.lat_to_km_map(fp.common_paths["fault_profiles"])
    assert np.abs(f(ft_lat) - km_geo).max() < 0.010
    # CFW1 creepmeter sits ~62.6 km from Point Pinole
    cfw1_lat = [c[1] for c in fp.creepmeters if c[0] == "CFW1"][0]
    assert f([cfw1_lat])[0] == pytest.approx(62.6, abs=0.2)


def test_north_first_trace_rejected(trace):
    ft_lon, ft_lat = trace
    with pytest.raises(ValueError):
        utils.along_fault_km(ft_lon[::-1], ft_lat[::-1])
