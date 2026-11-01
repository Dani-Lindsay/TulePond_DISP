"""zenodo_package.py: the files of the Zenodo deposit, built from a clean data root.

Author: Danielle Lindsay

Packs the data root of a clean run (TULEPOND_DATA_DIR, e.g. proj_TulePond_DISP_release) into the files of the deposit:
the three rebased OPERA DISP-S1 stacks as they are (netCDF, already compressed inside), one zip per group of products
(below), a snapshot of this repository at its current commit, and MANIFEST.csv (every file in every zip, with its size
and group). The raw DISP granules, the MintPy products and the static layers are not deposited: they are public
(OPERA DISP-S1, OPERA TROPO-ZENITH, OPERA CSLC-S1-STATIC) and data_prep/run01-04 rebuild them. The USGS SFSVM grid is
cited, not deposited. Of the external inputs only the files the code reads (common_paths of TulePond_filepaths) are
packed. Only the PyLith runs that the code names (model_config.py, scripts/) are packed, so superseded
runs and tests stay out.

Zips use deflate, except files that are already compressed (.nc, .npz, .gz, .png, .pdf, .zip), which are stored.
Nothing in the data root is changed. Nothing is uploaded.

Run (repo root):
    TULEPOND_DATA_DIR=<data root> PYTHONPATH=. python scripts/zenodo_package.py <out dir> [--manifest-only] [--only NAME ...]
"""
import argparse
import fnmatch
import glob
import os
import re
import subprocess
import sys
import zipfile

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from TulePond_filepaths import common_paths, data_dir, fig_dir, frames, executed_dir, inputs_dir, out_dir as outputs_dir   # noqa: E402

STORED = (".nc", ".npz", ".gz", ".png", ".pdf", ".zip", ".jpg")   # already compressed: stored, not deflated
SKIP = ["*/.DS_Store", "*/__pycache__/*", "*/_to_delete*", "*/retired_*", "*.pyc", "*/.ipynb_checkpoints/*"]
CREEPMETERS_USED = ["cpp1", "ctm1", "coz1", "chp1", "cfw1"]        # the five creepmeters of the paper (.10min.gz)
# unpublished catalogues of other authors: deposited only with their permission (cited; available from the authors)
NOT_REDISTRIBUTED = ["growclust_cat", "CRE_Taira"]


def referenced_runs():
    """PyLith run names that the code refers to (model_config.py and scripts/), so only those runs are packed.

    Returns
    -------
    set of str
    """
    sources = [os.path.join(REPO, "model_config.py")] + glob.glob(os.path.join(REPO, "scripts", "**", "*.py"), recursive=True)
    names = set()
    for path in sources:
        names |= set(re.findall(r"\b(gf_[a-z0-9_]+)\b", open(path).read()))
    return {name for name in names if name.endswith(("order2", "homog", "strip"))}


def run_of(file_name):
    """The PyLith run a file belongs to (`<run>-points.h5`, `G_<run>.npz`, `<run>.log`; forward and test runs included,
    so that they are left out unless the code names them), or None."""
    base = os.path.basename(file_name)
    match = re.match(r"^(?:G_(?:analytic_)?)?((?:gf|fwd|test|smoke)_[a-z0-9_]+?)(?:_nsub\d+)?(?:-.*|\.npz|\.npy|\.log)$", base)
    return match.group(1) if match else None


def groups(runs):
    """The zips of the deposit: name -> (description, list of (folder, keep rule)).

    Parameters
    ----------
    runs : set of str
        The PyLith runs the code names.

    Returns
    -------
    dict
    """
    def keep_run_files(path):
        run = run_of(path)
        return run is None or run in runs

    # the external inputs the code reads: every file and folder of common_paths under inputs_dir (TulePond_filepaths)
    read_paths = [p for key, p in common_paths.items()
                  if isinstance(p, str) and p.startswith(inputs_dir + os.sep) and key not in NOT_REDISTRIBUTED]

    def keep_inputs(path):
        if path.startswith(common_paths["creepmeter_dir"] + os.sep):
            return os.path.basename(path) in {f"{code}.10min.gz" for code in CREEPMETERS_USED}
        if os.path.basename(path).endswith(".zip"):
            return False                                    # archives of folders that are included unpacked
        return any(path == p or path.startswith(p + os.sep) for p in read_paths)

    def keep_homogeneous(path):
        rel = os.path.relpath(path, data_dir)
        if rel.startswith(os.path.join("modeling_uniform", "gf_dip_rows")):
            return "_nsub8" in os.path.basename(path)          # the n_sub 8 matrices of the dip sweep
        return keep_run_files(path)

    return {
        "inputs_external.zip": ("External inputs the code reads: fault trace and crossings, alignment arrays, the five "
                                "creepmeters, GHCNd rain, earthquake and repeating-earthquake catalogues, the creep-event "
                                "catalogue, the Johnson et al. (2022) compilation (the USGS SFSVM grid is cited, not included)",
                                [("external_inputs", keep_inputs)]),
        "gnss_velocity.zip": ("GNSS series and tie table, the tied velocity mosaics, the GNSS-tie record (data_prep run00, run04, run05)",
                              [("GPS", None), ("mosaic_velfield", None), ("prep_validation", None)]),
        "aperture_sweep.zip": ("Cross-fault aperture sweep (data_prep/run07)", [("aperture_sweep", None)]),
        "models_inputs.zip": ("Slip models, shared inputs: meshes, sample point sets, impulse and test-slip databases",
                              [("modeling_inputs", lambda p: "coh_sensitivity_drafts" not in p)]),
        "models_3d_sfsvm.zip": ("Slip models, 3D: SFSVM material grids and databases, PyLith runs (outputs and logs) and G "
                                "matrices of the runs the paper and its sensitivity tests use",
                                [("modeling_SFCVM", lambda p: "driver_logs" not in p and keep_run_files(p))]),
        "models_homogeneous.zip": ("Slip models, homogeneous: analytic (CSI) and PyLith homogeneous G matrices, PyLith "
                                   "verification runs, the n_sub 8 matrices of the dip sweep, run markers",
                                   [("modeling_uniform", keep_homogeneous), ("pylith_runs_done", None)]),
        "sensitivity.zip": ("Sensitivity tables and L-curves (scripts/sensitivity)", [("sensitivity", None)]),
        "outputs.zip": ("Every notebook's products, the number registry (registry.csv) and paper_numbers.md",
                        [(os.path.relpath(outputs_dir, data_dir), None)]),
        "figures.zip": ("All figures of the run, and the paper figures under their manuscript names (paper/)",
                        [(os.path.relpath(fig_dir, data_dir), None)]),
        "executed_notebooks.zip": ("HTML copies of the executed notebooks 01-14 of the run",
                                   [(os.path.relpath(executed_dir, data_dir), None)]),
    }


def files_of(folder, rule):
    """Files under data_dir/folder that the rule keeps (all if None), minus SKIP."""
    out = []
    for root, _, names in os.walk(os.path.join(data_dir, folder)):
        for name in sorted(names):
            path = os.path.join(root, name)
            if any(fnmatch.fnmatch(path, pattern) for pattern in SKIP):
                continue
            if rule is None or rule(path):
                out.append(path)
    assert out, f"nothing to pack in {folder}"
    return out


def write_zip(zip_path, paths):
    """Zip the files with their paths relative to data_dir (deflate, or stored if already compressed)."""
    with zipfile.ZipFile(zip_path, "w", allowZip64=True) as archive:
        for path in paths:
            method = zipfile.ZIP_STORED if path.endswith(STORED) else zipfile.ZIP_DEFLATED
            archive.write(path, os.path.relpath(path, data_dir), compress_type=method, compresslevel=6)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", help="folder for the deposit files (outside the data root)")
    parser.add_argument("--manifest-only", action="store_true", help="write MANIFEST.csv without zipping")
    parser.add_argument("--only", nargs="*", help="build only these zips")
    args = parser.parse_args()
    out = os.path.abspath(args.out)
    assert not out.startswith(os.path.abspath(data_dir)), "write the deposit outside the data root"
    os.makedirs(out, exist_ok=True)
    runs = referenced_runs()
    print(f"data root {data_dir}; {len(runs)} PyLith runs named by the code")
    rows = []
    for zip_name, (description, parts) in groups(runs).items():
        paths = []
        for folder, rule in parts:
            paths += files_of(folder, rule)
        size = sum(os.path.getsize(p) for p in paths)
        print(f"{zip_name:26s} {len(paths):6d} files {size / 1e9:7.2f} GB")
        for path in paths:
            rows.append(dict(deposit_file=zip_name, path=os.path.relpath(path, data_dir), bytes=os.path.getsize(path),
                             group=description))
        if not args.manifest_only and (not args.only or zip_name in args.only):
            write_zip(os.path.join(out, zip_name), paths)
            print(f"  wrote {zip_name}: {os.path.getsize(os.path.join(out, zip_name)) / 1e9:.2f} GB")
    for frame in frames:
        stack = frames[frame]["stack"]
        rows.append(dict(deposit_file=os.path.basename(stack), path=os.path.relpath(stack, data_dir),
                         bytes=os.path.getsize(stack), group="Rebased OPERA DISP-S1 stack of frame " + frame + " (data_prep run01)"))
    # the repository snapshot at its current commit
    commit = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    code_zip = f"code_TulePond_DISP_{commit}.zip"
    if not args.manifest_only and (not args.only or code_zip in args.only or "code" in args.only):
        dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
        assert not dirty, "commit the repository before packing its snapshot"
        subprocess.run(["git", "-C", REPO, "archive", "--format=zip", f"--prefix=TulePond_DISP/", "-o", os.path.join(out, code_zip), "HEAD"], check=True)
        print(f"  wrote {code_zip}")
    rows.append(dict(deposit_file=code_zip, path="TulePond_DISP/", bytes=None, group=f"The code repository at commit {commit}"))
    manifest = pd.DataFrame(rows)
    manifest.to_csv(os.path.join(out, "MANIFEST.csv"), index=False)
    print(f"MANIFEST.csv: {len(manifest)} rows; deposit {manifest.bytes.sum() / 1e9:.1f} GB before compression")


if __name__ == "__main__":
    main()
