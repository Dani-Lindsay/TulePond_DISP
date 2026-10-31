#!/usr/bin/env python3
"""
Tests for the number registry (insar_utils.Registry) on a temporary
directory: append semantics, the conflict rule, paper_numbers.md, and
the rule that notebooks never READ the registry (they read products).

Run:  pytest tests/test_registry.py -v
"""
import glob
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import insar_utils as utils                                          # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def paths(tmp_path):
    out = tmp_path / "outputs"
    return {"out_dir": str(out),
            "registry_csv": str(out / "registry.csv"),
            "paper_numbers": str(out / "paper_numbers.md")}


def test_register_appends_and_renders(paths, tmp_path):
    reg = utils.Registry("05_validation", paths, REPO)                    # a notebook that exists in the repo
    f = tmp_path / "in.csv"; f.write_text("a,b\n1,2\n")
    reg.inputs(str(f))
    reg.register("05_n_events", 5, "count", 0, where="P3", cell="detector")
    reg.register("05_amp_2017_mm", 10.836, "mm", 2, where="P3", cell="joint-fit")
    df = pd.read_csv(paths["registry_csv"], dtype=str)   # rendered is a STRING, read it as one
    assert list(df.columns) == utils.REGISTRY_COLUMNS
    assert len(df) == 2
    assert df.loc[1, "rendered"] == "10.84"
    assert df.loc[0, "inputs"].startswith("in.csv:")
    txt = open(reg.render()).read()
    assert "| Amp 2017 | 10.84 | mm | joint-fit | `05_amp_2017_mm` |" in txt
    assert "| N events | 5 |  | detector | `05_n_events` |" in txt
    assert txt.index("05_n_events") < txt.index("05_amp_2017_mm")      # in the order the notebook registers them


def test_render_leaves_out_keys_of_missing_notebooks(paths):
    utils.Registry("05_validation", paths, REPO).register("05_n_events", 5, "count", 0, "P3", "detector")
    old = utils.Registry("99_gone_notebook", paths, REPO)                  # no 99_gone_notebook.ipynb in the repo
    old.register("99_old_value", 2.78, "yr", 2, "P10", "forecast")
    txt = open(old.render()).read()
    assert "99_old_value" not in txt and "`05_n_events`" in txt


def test_render_retires_keys_the_latest_run_no_longer_makes(paths):
    first = utils.Registry("05_validation", paths, REPO)
    first.register("05_kept", 1, "count", 0, "P3", "a"); first.register("05_dropped", 2, "count", 0, "P3", "a")
    second = utils.Registry("05_validation", paths, REPO)                      # the next run no longer makes 05_dropped
    second.register("05_kept", 1, "count", 0, "P3", "a")
    txt = open(second.render()).read()
    assert "`05_kept`" in txt and "05_dropped" not in txt


def test_same_value_rerun_is_allowed_and_kept(paths):
    reg = utils.Registry("05_event_detection", paths, REPO)
    reg.register("05_n_events", 5, "count", 0, "P3", "detector")
    reg2 = utils.Registry("05_event_detection", paths, REPO)
    reg2.register("05_n_events", 5.0, "count", 0, "P3", "detector")     # same rendered value
    df = pd.read_csv(paths["registry_csv"])
    assert len(df) == 2 and df["run_id"].nunique() == 2


def test_changed_value_raises_unless_superseded(paths, monkeypatch):
    monkeypatch.delenv("TULEPOND_SUPERSEDE", raising=False)          # a run with supersede on must not leak into this test
    reg = utils.Registry("05_event_detection", paths, REPO)
    reg.register("05_band_secular_mm_yr", 1.084, "mm/yr", 2, "P4", "band-stack")
    with pytest.raises(utils.RegistryConflict):
        reg.register("05_band_secular_mm_yr", 1.21, "mm/yr", 2, "P4", "band-stack")
    # a change within the stated precision is not a conflict
    reg.register("05_band_secular_mm_yr", 1.0849, "mm/yr", 2, "P4", "band-stack")
    monkeypatch.setenv("TULEPOND_SUPERSEDE", "1")
    reg3 = utils.Registry("05_event_detection", paths, REPO)
    reg3.register("05_band_secular_mm_yr", 1.21, "mm/yr", 2, "P4", "band-stack")
    assert reg3.latest().set_index("key").loc["05_band_secular_mm_yr", "rendered"] == "1.21"


def test_sidecar_has_provenance_header(paths):
    reg = utils.Registry("03_creep_timeseries", paths, REPO)
    rel = reg.sidecar("creep_rate", ["n crossings: 480", "median rate: 4.70 mm/yr"])
    text = open(os.path.join(paths["out_dir"], rel)).read()
    assert text.startswith("# 03_creep_timeseries | git ")
    assert "median rate: 4.70 mm/yr" in text


def test_notebooks_never_read_the_registry():
    """Numbers flow products -> registry -> manuscript, never registry -> notebook. Two notebooks read the registry
    on purpose: 05_validation, only to check that 04's creep_events.csv matches what 04 registered, and 13_forecast,
    which reads back the band-stack steps of the latest run of 04 for the prediction statement (and checks that run
    against creep_events.csv)."""
    allowed = {"insar_utils.py", "TulePond_filepaths.py", "05_validation.py", "13_forecast.py"}
    sources = glob.glob(os.path.join(REPO, "*.py")) + glob.glob(os.path.join(REPO, "scripts", "**", "*.py"), recursive=True)
    assert any(os.path.basename(path) == "04_event_detection.py" for path in sources), "the notebook sources were not found"
    for nb in sources:
        if os.path.basename(nb) in allowed:
            continue
        src = open(nb).read()
        assert ".latest(" not in src and "registry_csv" not in src, f"{nb} reads the registry"


def test_file_fingerprint_partial_for_large_files(tmp_path):
    small = tmp_path / "s.bin"; small.write_bytes(b"x" * 100)
    assert len(utils.file_fingerprint(str(small))) == 12
    assert utils.file_fingerprint(str(small), full_limit_mb=0).startswith("~")
