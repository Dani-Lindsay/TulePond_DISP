"""zenodo_package.py: the files of the Zenodo deposit, built from a clean data root.

Author: Danielle Lindsay

Packs the data root of a clean run (TULEPOND_DATA_DIR, e.g. proj_TulePond_DISP_release) into the files of the deposit:
the three rebased OPERA DISP-S1 stacks as they are (netCDF, already compressed inside), one zip per group of products
(below), and MANIFEST.csv (every file in every zip, with its size and group). The code is archived separately. The raw
DISP granules, the MintPy products and the static layers are not deposited: they are public (OPERA DISP-S1, OPERA
TROPO-ZENITH, OPERA CSLC-S1-STATIC) and data_prep/run01-04 rebuild them. The USGS SFSVM grid is cited, not deposited. The inputs of the notebooks (external inputs, GNSS tables, velocity mosaics, aperture sweep) are
exactly the files a run of notebooks 01-15 opened for reading (the read audit of scripts/read_audit, --reads), less the
unpublished catalogues of other authors (NOT_REDISTRIBUTED). The slip models follow the PyLith manual (publishing
models): for every PyLith run the code names (model_config.py, scripts/), its input files (meshes, spatial databases,
output points, as named in its parameters; the cfg files are in the repository) and the outputs behind the results
(points, fault and elastic_info with their xmf headers, the parameters), plus its G matrix and log.

Zips use deflate, except files that are already compressed (.nc, .npz, .gz, .png, .pdf, .zip), which are stored.
Nothing in the data root is changed. Nothing is uploaded.

Run (repo root):
    TULEPOND_DATA_DIR=<data root> PYTHONPATH=. python scripts/zenodo_package.py <out dir> --reads <read list>
        [--manifest-only] [--only NAME ...]
"""
import argparse
import fnmatch
import glob
import os
import re
import sys
import zipfile

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from TulePond_filepaths import common_paths, data_dir, fig_dir, frames, out_dir as outputs_dir   # noqa: E402
import insar_utils as utils                                                                                        # noqa: E402
import model_config as cfg                                                                                     # noqa: E402

STORED = (".nc", ".npz", ".gz", ".png", ".pdf", ".zip", ".jpg")   # already compressed: stored, not deflated
SKIP = ["*/.DS_Store", "*/__pycache__/*", "*/_to_delete*", "*/retired_*", "*.pyc", "*/.ipynb_checkpoints/*"]
# 15's MintPy-format crop of the descending stack (11 GB), rebuilt by every run of 15 from the deposited stack
SKIP += ["*/15_check_original_2025_code/inputs_HW3/*"]
# unpublished catalogues of other authors: deposited only with their permission (cited; available from the authors)
NOT_REDISTRIBUTED = ["growclust_cat", "CRE_Taira"]


def referenced_runs():
    """PyLith run names that the code refers to (model_config.py, the notebooks and scripts/), so only those runs are packed.

    Returns
    -------
    set of str
    """
    sources = ([os.path.join(REPO, "model_config.py")] + glob.glob(os.path.join(REPO, "[0-9][0-9]_*.ipynb"))
               + glob.glob(os.path.join(REPO, "scripts", "**", "*.py"), recursive=True))
    names = set()
    for path in sources:
        text = utils.notebook_source(path) if path.endswith(".ipynb") else open(path).read()
        names |= set(re.findall(r"\b(gf_[a-z0-9_]+)\b", text))
    return {name for name in names if name.endswith(("order2", "homog", "strip"))}


def run_of(file_name):
    """The PyLith run a file belongs to (`<run>-points.h5`, `G_<run>.npz`, `<run>.log`; forward and test runs included,
    so that they are left out unless the code names them), or None."""
    base = os.path.basename(file_name)
    match = re.match(r"^(?:G_(?:analytic_)?)?((?:gf|fwd|test|smoke)_[a-z0-9_]+?)(?:_nsub\d+)?(?:-.*|\.npz|\.npy|\.log)$", base)
    return match.group(1) if match else None


PUBLISHED_RUN_FILES = ("-points.h5", "-points.xmf", "-fault.h5", "-fault.xmf", "-elastic_info.h5", "-elastic_info.xmf",
                       "-parameters.json")   # per PyLith run: the outputs behind the results and its parameters (PyLith manual)


def pylith_inputs(runs):
    """Names of the input files (meshes, spatial databases, output-point lists) named in the parameters of the runs.

    Parameters
    ----------
    runs : set of str

    Returns
    -------
    set of str
    """
    names = set()
    for run in runs:
        found = glob.glob(os.path.join(data_dir, "modeling_*", "pylith_output", f"{run}-parameters.json"))
        assert found, f"{run}: no parameters.json"
        names |= {os.path.basename(m) for m in re.findall(r'"([^"]*\.(?:msh|spatialdb|txt))"', open(found[0]).read())}
    return names - {"stdout.txt", "stderr.txt"}


def groups(runs, reads):
    """The zips of the deposit: name -> (description, list of (folder, keep rule)).

    Parameters
    ----------
    runs : set of str
        The PyLith runs the code names.
    reads : set of str
        Files (relative to the data root) that a run of notebooks 01-15 opened for reading (read audit).

    Returns
    -------
    dict
    """
    used = pylith_inputs(runs)
    mesh_stems = {name[:-len(".msh")] for name in used if name.endswith(".msh")}
    point_stems = {name[:-len(".txt")] for name in used if name.endswith(".txt")}
    material_variants = {name[len("mat_"):-len(".spatialdb")] for name in used if name.startswith("mat_")}
    not_redistributed = {os.path.relpath(common_paths[key], data_dir) for key in NOT_REDISTRIBUTED}

    def read_by_code(path):
        rel = os.path.relpath(path, data_dir)
        return rel in reads and rel not in not_redistributed

    def keep_model(path):
        rel, base = os.path.relpath(path, data_dir), os.path.basename(path)
        parts = rel.split(os.sep)
        if "pylith_output" in parts:
            return any(base == run + suffix for run in runs for suffix in PUBLISHED_RUN_FILES)
        if "gf_dip_rows" in parts:
            return "_nsub8" in base                              # the n_sub 8 matrices of the dip sweep
        if base.startswith("G_analytic_"):
            return base.startswith(f"G_analytic_{cfg.gf_run}_")      # the analytic G of the preferred model (notebook 08)
        if "gf" in parts or "runs" in parts:
            return run_of(base) in runs                          # G matrices and run logs of the runs used
        if "mesh" in parts:
            return any(base in (f"{stem}.msh", f"{stem}_fault.npz", f"{stem}_report.txt") for stem in mesh_stems)
        if parts[-2] == "points":                                # the current sample sets (not older copies in subfolders)
            return any(base in (f"{stem}.txt", f"{stem}.npz") for stem in point_stems)
        if parts[-2] == "spatialdb" and parts[0] == "modeling_inputs":
            return base in used                                  # the impulse databases the runs read
        if parts[-2] == "spatialdb" and parts[0] == "modeling_SFCVM":
            return ((base.startswith("mat_") and base in used) or base == "sfcvm_v26-0_boundary_offset.csv"
                    or any(base == f"sfcvm_{variant[len('sfcvm_'):]}_grid.npz" for variant in material_variants))
        if parts[0] == "pylith_runs_done":
            return base in runs
        return False

    return {
        "inputs_external.zip": ("External inputs read by the notebooks: fault trace and crossings, alignment arrays, the "
                                "five creepmeters, GHCNd rain, the repeating-earthquake and creep-event catalogues, the "
                                "Johnson et al. (2022) compilation",
                                [("external_inputs", read_by_code)]),
        "gnss_velocity.zip": ("GNSS station tables and the GNSS-InSAR tie, the tied velocity mosaics (data_prep run00, "
                              "run04, run05), as read by the notebooks", [("GPS", read_by_code), ("mosaic_velfield", read_by_code)]),
        "aperture_sweep.zip": ("Cross-fault aperture sweep (data_prep/run07): the series and summary notebook 05 scores",
                               [("aperture_sweep", read_by_code)]),
        "models_inputs.zip": ("Slip models, PyLith inputs: the meshes (with their fault files and reports), output-point "
                              "sets and impulse databases named by the runs used", [("modeling_inputs", keep_model)]),
        "models_3d_sfsvm.zip": ("Slip models, 3D: SFSVM material databases and grids, and per PyLith run the points, fault "
                                "and elastic_info outputs with their xmf headers, the parameters, the log and the G matrix",
                                [("modeling_SFCVM", keep_model)]),
        "models_homogeneous.zip": ("Slip models, homogeneous: analytic (CSI) G matrices (n_sub 8, the dip sweep included), "
                                   "the PyLith homogeneous verification runs and the run markers",
                                   [("modeling_uniform", keep_model), ("pylith_runs_done", keep_model)]),
        "sensitivity.zip": ("Sensitivity tables and L-curves (scripts/sensitivity)", [("sensitivity", None)]),
        "outputs.zip": ("Every notebook's products, the number registry (registry.csv) and paper_numbers.md",
                        [(os.path.relpath(outputs_dir, data_dir), None)]),
        "figures.zip": ("All figures of the run, and the paper figures under their manuscript names (paper/)",
                        [(os.path.relpath(fig_dir, data_dir), None)]),
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
    parser.add_argument("--reads", required=True, help="read audit: files opened for reading by a run of notebooks 01-15 "
                                                       "(one path per line, relative to the data root; scripts/read_audit)")
    args = parser.parse_args()
    out = os.path.abspath(args.out)
    assert not out.startswith(os.path.abspath(data_dir)), "write the deposit outside the data root"
    os.makedirs(out, exist_ok=True)
    runs = referenced_runs()
    reads = {line.strip() for line in open(args.reads) if line.strip()}
    print(f"data root {data_dir}; {len(runs)} PyLith runs named by the code")
    rows = []
    for zip_name, (description, parts) in groups(runs, reads).items():
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
    manifest = pd.DataFrame(rows)
    manifest.to_csv(os.path.join(out, "MANIFEST.csv"), index=False)
    print(f"MANIFEST.csv: {len(manifest)} rows; deposit {manifest.bytes.sum() / 1e9:.1f} GB before compression")


if __name__ == "__main__":
    main()
