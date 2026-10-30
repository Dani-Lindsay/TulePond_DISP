"""zenodo_manifest.py: the manifest of the slip-model part of the Zenodo deposit (Dani 2026-10-28: archive EVERYTHING of the
slip models: every PyLith raw output, G matrix, mesh, cfg, spatial database, sweep csv and script). Written 2026-10-30.

Writes <data root>/zenodo_manifest.csv: one row per file with its path (relative to the data root, or `repo:<path>` for
the code), size, what it is, which notebook / script made it, and its role (paper = behind a number or figure of the
paper; sensitivity; verification; superseded = kept for the record). Nothing is uploaded or moved.
Descriptions come from the rules below (file-name patterns); a file no rule describes stops the script (fail loud), so
nothing enters the deposit undescribed.
Run (repo root): PYTHONPATH=. python scripts/zenodo_manifest.py
"""
import fnmatch
import os
import subprocess
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from TulePond_filepaths import data_dir          # noqa: E402
import model_config as cfg                       # noqa: E402

# the folders and root files of the deposit (slip models); everything else of the data root is outside this manifest
FOLDERS = ["modeling_inputs", "modeling_SFCVM", "modeling_uniform", "pylith_runs_done", "outputs/07_model_data",
           "outputs/09_model_3d", "outputs/10_model_sensitivity", "outputs/08_model_homogeneous", os.path.basename(cfg.sensitivity_dir),
           "sensitivity_2026-10-27", "depth_scan_2026-10-28", "order2_surface_tent_check_2026-10-25",
           "basement_depth_2026-10-28", "semivariogram_2026-10-28", "Figures_2026-10-28", "external_inputs/USGS_SFCVM_v26-0"]
ROOT_FILES = ["run_production_2026-10-25.sh", "run_production_2026-10-25.jobs", "run_production_2026-10-25.log",
              "run_rerun_2026-10-25.sh", "run_rerun_2026-10-25.jobs", "run_rerun_2026-10-25.log",
              "run_shallow_dips_2026-10-26.sh", "run_shallow_dips_2026-10-26.jobs", "run_shallow_dips_2026-10-26.log",
              "CHOICES_REGISTER_2026-10-25.md"]
REPO_FOLDERS = ["pylith", "scripts/sensitivity"]
REPO_FILES = ["model_config.py", "model_utils.py", "TulePond_filepaths.py", "insar_utils.py", "scripts/07_model_data.py",
              "scripts/09_model_3d.py", "scripts/10_model_sensitivity.py", "scripts/08_model_homogeneous.py", "scripts/zenodo_manifest.py",
              "07_model_data.ipynb", "09_model_3d.ipynb", "10_model_sensitivity.ipynb", "08_model_homogeneous.ipynb", "tests/test_model_utils.py"]
SKIP = ["*/__pycache__/*", "*/_to_delete_*", "*/.DS_Store", "*.pyc", "*/retired_2026-10-28/*"]
SELECTED_RUNS = {cfg.gf_run}
SENSITIVITY_RUNS = {"gf_shallow_dip70e_sfcvm_order2", "gf_shallow_dip70w_sfcvm_order2", "gf_default_sfcvm_order2",
                    "gf_default_sfcvm_e50_order2", "gf_default_sfcvm_w50_order2", "gf_default_sfcvmasis_order2",
                    "gf_default_sfcvm_coh07_order2", "gf_default_dip80e_sfcvm_order2", "gf_default_dip80w_sfcvm_order2",
                    "gf_default_dip70e_sfcvm_order2", "gf_default_dip70w_sfcvm_order2"}
VERIFICATION_RUNS = {"gf_default_homog", "gf_default_homog_order2", "gf_shallow_homog_order2_strip", "gf_fine_homog_order2_strip"}
OUTPUT_PART = {"points.h5": "displacement at the output points (the samples) for every impulse",
               "fault.h5": "fault vertices, cells and the unit slip of every impulse",
               "elastic_info.h5": "material properties PyLith used (density, Vs, Vp, shear modulus) per vertex",
               "elastic.h5": "solution in the elastic domain", "fault_info.h5": "fault auxiliary fields",
               "parameters.json": "every PyLith parameter of the run", "progress.txt": "PyLith progress record",
               "xmf": "XDMF header of the matching .h5 (ParaView)"}


def run_role(run):
    if run in SELECTED_RUNS:
        return "paper"
    if run in SENSITIVITY_RUNS:
        return "sensitivity"
    if run in VERIFICATION_RUNS:
        return "verification"
    return "superseded"


def describe(path):
    """(what it is, made by, role) of one file, from its path; None if no rule applies."""
    name = os.path.basename(path)
    parts = path.split("/")
    if path.startswith("repo:"):
        return ("code (GitHub repo TulePond_DISP, archived with the deposit)", "author", "code")
    if parts[0] in ("modeling_SFCVM", "modeling_uniform") and parts[1] == "pylith_output":
        run = name.split("-")[0]
        part = name.split("-", 1)[1] if "-" in name else name
        key = "xmf" if name.endswith(".xmf") else next((k for k in OUTPUT_PART if part.endswith(k)), None)
        if part.startswith("bc_"):
            what = f"boundary-condition output ({part.split('.')[0]})"
        elif key:
            what = OUTPUT_PART[key]
        else:
            what = "PyLith output"
        return (f"PyLith 5.0.2 raw output of run {run}: {what}", "pylith/run_gf.sh or run_fwd.sh (PyLith 5.0.2)", run_role(run))
    if parts[0] in ("modeling_SFCVM", "modeling_uniform") and parts[1] == "gf":
        if name.startswith("G_analytic"):
            return ("analytic homogeneous Green's functions on the PyLith tents (CSI Meade 2007, n_sub in the name)",
                    "model_utils.analytic_tent_G_cached (07 [7])", "paper")
        run = name[2:].replace(".npz", "")
        return (f"fault-parallel G matrix of run {run} (samples x impulses, m/m, right-lateral +)", "model_utils.build_G / load_G", run_role(run))
    if parts[0] in ("modeling_SFCVM", "modeling_uniform") and parts[1] == "runs":
        return ("PyLith run log (run_gf.log: start/done times of every run)", "pylith/run_gf.sh, run_fwd.sh", "record")
    if parts[0] in ("modeling_SFCVM", "modeling_uniform", "modeling_inputs") and parts[1] == "spatialdb":
        if name.endswith("_grid.npz"):
            return ("SFSVM v26.0 resampled on the model grid (density, Vs, Vp, zone; alignment / offset / cap in the name)",
                    "pylith/build/sfcvm_grid.py", "paper" if "aligned_nocap_grid" in name else "sensitivity")
        if name.startswith("mat_sfcvm"):
            return ("PyLith SimpleGridDB of the SFSVM material (from the matching _grid.npz)", "pylith/build/write_sfcvm_griddb.py",
                    "paper" if name == "mat_sfcvm_v26-0_aligned_nocap.spatialdb" else "sensitivity")
        if name.startswith("impulses"):
            return ("PyLith impulse-region SimpleGridDB (unit left-lateral slip where y and z are in the region)",
                    "pylith/build/make_impulse_db.py", "paper" if "km54-65_z3000" in name else "verification")
        if name.startswith("kin_slip") or name.startswith("fig2_slip") or name.startswith("traction"):
            return ("slip database of a forward run (order 1)", "pylith/build/make_kin_slip_db.py / make_vertex_slip_db.py", "superseded")
        if name.endswith("_vs_below_90.csv") or name.endswith("boundary_offset.csv"):
            return ("SFSVM grid diagnostics (Vs below 90 m/s; boundary offset)", "pylith/build/sfcvm_grid.py", "record")
        return ("spatial database / grid auxiliary file", "pylith/build/", "record")
    if parts[0] == "modeling_inputs" and parts[1] == "mesh":
        if name.endswith(".msh"):
            role = "paper" if name == "mesh_vertical_tet_shallow.msh" else ("sensitivity" if "default" in name or "dip70" in name else "superseded")
            return ("gmsh tetrahedral mesh (preset / dip in the name)", "pylith/build/generate_gmsh_vertical.py / generate_gmsh_dip.py "
                    "(default/shallow: the pre-tidy drafts/pylith copy, 29-30 Sep)", role)
        if name.endswith("_fault.npz") or name.endswith("_report.txt"):
            return ("fault triangles of a mesh / its quality report", "pylith/build/generate_gmsh_*.py", "record")
        return ("mesh-generation log or helper", "pylith/build/", "record")
    if parts[0] == "modeling_inputs" and parts[1] == "points":
        if name.startswith("insar_points"):
            return ("PyLith output points = 07's quadtree samples (txt: name x y z, m; npz: data, block sizes, noise model)",
                    "07_model_data [6]", "paper" if name.startswith("insar_points.") else "sensitivity")
        return ("sample-set auxiliary file (coherence-0.7 field, old Fig-2 profile points)", "07 / old 08", "superseded")
    if parts[0] == "modeling_inputs":
        return ("pre-2026-10-06 sample-set drafts", "drafts", "superseded")
    if parts[0] in ("modeling_SFCVM", "modeling_uniform"):
        return ("medium folder auxiliary file (validation products of the order-1 08)", "old 08", "superseded")
    if parts[0] == "pylith_runs_done":
        return (f"done marker of PyLith run {name}", "run_production_2026-10-25.sh / run_shallow_dips_2026-10-26.sh", "record")
    if parts[0] == "outputs":
        nb = parts[1]
        return (f"product of notebook {nb} ({name})", nb, "paper")
    if parts[0] == os.path.basename(cfg.sensitivity_dir):
        return (f"sensitivity result ({name}), the csvs 10_model_sensitivity draws", "scripts/sensitivity/ (provenance column of each csv)", "sensitivity")
    if parts[0] == "Figures_2026-10-28":
        return (f"figure ({name})", "09 / 08 / 10 (fig_date 2026-10-28)", "paper")
    if parts[0] == "external_inputs":
        return ("USGS SFCVM v26.0 (Hirakawa & Aagaard 2026), third-party input: check the licence before redistributing", "USGS", "input")
    if parts[0] in ("sensitivity_2026-10-27", "depth_scan_2026-10-28", "order2_surface_tent_check_2026-10-25",
                    "basement_depth_2026-10-28", "semivariogram_2026-10-28"):
        return (f"archived sweep of {parts[0][-10:]} ({name}); superseded by the re-run in {os.path.basename(cfg.sensitivity_dir)} "
                "except the n_sub 4 analytic G caches it holds", f"the script in {parts[0]}/", "superseded"
                if not name.startswith("G_analytic") and parts[1:2] != ["gf_analytic"] else "sensitivity")
    if path in ROOT_FILES:
        return ("PyLith run queue (script, job list, log) of the order-2 runs" if path.startswith("run_") else "document", "author", "record")
    return None


def files_under(root, folder):
    out = []
    for base, _, names in os.walk(os.path.join(root, folder)):
        for name in names:
            rel = os.path.relpath(os.path.join(base, name), root)
            if not any(fnmatch.fnmatch("/" + rel, pattern) for pattern in SKIP):
                out.append(rel)
    return sorted(out)


if __name__ == "__main__":
    rows = []
    for folder in FOLDERS:
        assert os.path.isdir(os.path.join(data_dir, folder)), f"missing folder {folder}"
        for rel in files_under(data_dir, folder):
            rows.append((rel, os.path.getsize(os.path.join(data_dir, rel))))
    for rel in ROOT_FILES:
        if os.path.exists(os.path.join(data_dir, rel)):
            rows.append((rel, os.path.getsize(os.path.join(data_dir, rel))))
    tracked = set(subprocess.run(["git", "-C", REPO, "ls-files"], capture_output=True, text=True, check=True).stdout.split())
    repo_files = [f for f in tracked if any(f.startswith(folder + "/") for folder in REPO_FOLDERS)] + REPO_FILES
    for rel in sorted(set(repo_files)):
        assert os.path.exists(os.path.join(REPO, rel)), f"missing repo file {rel}"
        rows.append(("repo:" + rel, os.path.getsize(os.path.join(REPO, rel))))
    table = []
    for path, size in rows:
        described = describe(path)
        assert described is not None, f"no description rule for {path}"
        what, made_by, role = described
        table.append(dict(path=path, size_bytes=size, size_mb=round(size / 1e6, 3), description=what, made_by=made_by, role=role))
    table = pd.DataFrame(table)
    out = os.path.join(data_dir, "zenodo_manifest.csv")
    table.to_csv(out, index=False)
    print(table.groupby("role").size_bytes.agg(["count", "sum"]).assign(GB=lambda t: (t["sum"] / 1e9).round(2)).to_string())
    print(f"total {table.size_bytes.sum() / 1e9:.2f} GB in {len(table)} files; wrote {out}")
