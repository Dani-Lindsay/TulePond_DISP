#!/usr/bin/env python3
"""
Tests for the number registry (insar_utils.Registry) on a temporary
directory: append semantics, the conflict rule, the rendered text view, and
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
            "registry_txt": str(out / "registry.txt")}


def test_register_appends_and_renders(paths, tmp_path):
    reg = utils.Registry("05_event_detection", paths, REPO)
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
    assert "05_amp_2017_mm = 10.84 mm (2 dp) -> P3" in txt
    assert "P3: 05_amp_2017_mm=10.84, 05_n_events=5" in txt


def test_same_value_rerun_is_allowed_and_kept(paths):
    reg = utils.Registry("05_event_detection", paths, REPO)
    reg.register("05_n_events", 5, "count", 0, "P3", "detector")
    reg2 = utils.Registry("05_event_detection", paths, REPO)
    reg2.register("05_n_events", 5.0, "count", 0, "P3", "detector")     # same rendered value
    df = pd.read_csv(paths["registry_csv"])
    assert len(df) == 2 and df["run_id"].nunique() == 2


def test_changed_value_raises_unless_superseded(paths, monkeypatch):
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
    """Numbers flow products -> registry -> manuscript, never registry -> notebook."""
    for nb in glob.glob(os.path.join(REPO, "*.py")):
        if os.path.basename(nb) in ("insar_utils.py", "TulePond_filepaths.py"):
            continue
        src = open(nb).read()
        assert ".latest(" not in src and "registry_csv" not in src, f"{nb} reads the registry"


def test_file_fingerprint_partial_for_large_files(tmp_path):
    small = tmp_path / "s.bin"; small.write_bytes(b"x" * 100)
    assert len(utils.file_fingerprint(str(small))) == 12
    assert utils.file_fingerprint(str(small), full_limit_mb=0).startswith("~")
