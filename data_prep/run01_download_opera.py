#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Danielle Lindsay

OPERA DISP-S1 download and rebase for the Hayward AOI, using the `opera-utils`
command-line tools:

1) Per frame: verify the subset ROI with a small test patch before committing
   to the full download
2) Full-range download, subset to the AOI in one call (~1 MB per product,
   corrections and metadata groups retained)
3) Consistency checks: one grid, corrections group present
4) `disp-s1-reformat`: rebase the moving reference dates into one continuous
   stack per frame, applying the correction/precision options below

Frames: ascending 09156 + 09157 (track 35) and descending 11115 (track 42).
09156 uses a bbox clipped to its own footprint because opera-utils errors
when asked for pixels beyond a frame's coverage.

Output per frame:  <data_dir>/<frame>/data/*.nc  and  <frame>/<frame>_stack.nc

Run in the conda environment (needs opera-utils, shapely, h5py, xarray).
Earthdata credentials in ~/.netrc.
Source: OPERA DISP-S1 v1.0 (NASA JPL OPERA project, distributed by ASF);
opera-utils (disp-s1-download, disp-s1-reformat).
"""
import glob
import os
import subprocess
from datetime import date, timedelta

import os as _os, sys as _sys        # path shim: import the repo root modules
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from TulePond_filepaths import common_paths, frames
import insar_utils as utils

# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------
OPERA = "opera-utils"     # on PATH when the conda environment is active

start = common_paths['date_start']
# `date_end` is the last acquisition used in the paper. CMR's --end-datetime is a
# datetime, so passing the bare date drops that day's acquisition (the Bay Area
# passes are ~02:07 UTC ascending, ~14:07 UTC descending, both after 00:00), so the
# search ends one day after it.
end   = (date.fromisoformat(common_paths['date_end']) + timedelta(days=1)).isoformat()

# Per-frame DOWNLOAD window: `aoi_bbox_disp` for 09157/11115 (the science AOI
# nudged one pixel so new granules land on the existing archive grid; see the
# comment on aoi_bbox_disp in TulePond_filepaths.py); 09156 clipped to its
# footprint, and its own bbox already reproduces its archive window.
frame_bbox = {
    "09156": common_paths['aoi_bbox_09156'],
    "09157": common_paths['aoi_bbox_disp'],
    "11115": common_paths['aoi_bbox_disp'],
}

VERIFY_WIN  = ("2020-06-01", "2020-08-01")  # dense period for the ROI test patch
ROI_TOL_DEG = 0.03                          # ~1 pixel-block tolerance on subset extent

# ---------------------------------------------------------------------------
# disp-s1-reformat options (not the tool defaults). Passed explicitly so a
# run is reproducible from the command alone.
#
#   solid earth tide ON   - tool default; cancels in cross-fault differencing
#                           but matters for the GNSS tie.
#   ionosphere ON         - tool default is OFF. Long-wavelength, so an iono
#                           residual is exactly what the GNSS deramp would
#                           otherwise absorb, making ramp and signal
#                           inseparable. Correct it before the tie.
#   --no-do-round         - default True rounds mantissa bits to compress.
#                           Lossy on a science product (~38% larger without).
#   HIGH_COHERENCE @ 0.90 - tool default threshold is 0.7. Per-frame reference;
#                           it largely cancels anyway (GNSS tie + differencing).
#
# Troposphere is NOT here: it is a separate L4 product applied in run03.
# Do not substitute an empirical phase-elevation fit.
# ---------------------------------------------------------------------------
REFORMAT_OPTS = [
    "--apply-solid-earth-corrections",
    "--apply-ionospheric-corrections",
    "--no-do-round",
    "--reference-method", "HIGH_COHERENCE",
    "--reference-coherence-threshold", "0.90",
]


# ---------------------------------------------------------------------------
# ROI verification: small test patch before the big download
# ---------------------------------------------------------------------------
def _subset_bounds(nc_path):
    """(lon_lo, lat_lo, lon_hi, lat_hi) of a subset .nc, in EPSG:4326."""
    import re
    import h5py
    from pyproj import Transformer
    with h5py.File(nc_path) as h:
        x = h["x"][:]; y = h["y"][:]
        wkt = h["spatial_ref"].attrs.get("crs_wkt", "")
    m = re.findall(r'ID\["EPSG",(\d+)\]', wkt)
    epsg = int(m[-1]) if m else 4326
    t = Transformer.from_crs(epsg, "EPSG:4326", always_xy=True)
    lo0, la0 = t.transform(x.min(), y.min())
    lo1, la1 = t.transform(x.max(), y.max())
    return (min(lo0, lo1), min(la0, la1), max(lo0, lo1), max(la0, la1))


def verify_roi(fid, bbox, out):
    """Download a test patch (VERIFY_WIN) and confirm the subset extent matches
    the requested AOI. Returns (ok, got_bounds), or None if the test window
    had no products."""
    vdir = os.path.join(out, "_verify_patch")
    os.makedirs(vdir, exist_ok=True)
    subprocess.run([OPERA, "disp-s1-download", "--frame-id", fid, "--bbox", *map(str, bbox),
                    "--start-datetime", VERIFY_WIN[0], "--end-datetime", VERIFY_WIN[1],
                    "--output-dir", vdir, "--num-workers", "4"], capture_output=True, text=True)
    ncs = sorted(glob.glob(os.path.join(vdir, "*.nc")))
    if not ncs:
        return None
    got = _subset_bounds(ncs[0])
    W, S, E, N = bbox
    ok = (abs(got[0] - W) < ROI_TOL_DEG and abs(got[1] - S) < ROI_TOL_DEG
          and abs(got[2] - E) < ROI_TOL_DEG and abs(got[3] - N) < ROI_TOL_DEG)
    return ok, got


# ---------------------------------------------------------------------------
# Consistency checks before reformat
# ---------------------------------------------------------------------------
def check_corrections_group(data_dir):
    """Abort if products lack the `corrections` group that REFORMAT_OPTS needs.

    `--apply-*-corrections` read `/corrections`, which is only present if the
    download kept it. Without this check, `disp-s1-reformat` fails late with
    `OSError: group not found: corrections` and no frame name.
    """
    import h5py
    ncs = sorted(glob.glob(os.path.join(data_dir, "*.nc")))
    missing = []
    for f in ncs:
        try:
            with h5py.File(f, "r") as h:
                if "corrections" not in h:
                    missing.append(f)
        except OSError:
            missing.append(f)
    if missing:
        raise SystemExit(
            f"ABORT: {len(missing)} of {len(ncs)} products in {data_dir} have no "
            f"'corrections' group (e.g. {os.path.basename(missing[0])[:60]}).\n"
            f"       Delete them and re-download.")


def check_grid_consistency(data_dir):
    """Abort if the .nc files in `data_dir` are not all on the same grid.

    `disp-s1-download` skips files that already exist by filename, and the
    filename does not encode the subset window: re-running with a slightly
    different bbox leaves two incompatible grids in one directory, and
    `disp-s1-reformat` then fails deep inside xarray ("Resulting object does
    not have monotonic global indexes") without naming the cause.
    """
    import xarray as xr
    ncs = sorted(glob.glob(os.path.join(data_dir, "*.nc")))
    if len(ncs) < 2:
        return
    grids = {}
    for f in ncs:
        with xr.open_dataset(f) as d:
            key = (len(d.x), len(d.y), round(float(d.x[0]), 1), round(float(d.y[0]), 1))
        grids.setdefault(key, []).append(f)
    if len(grids) > 1:
        for (nx, ny, x0, y0), fs in sorted(grids.items(), key=lambda kv: -len(kv[1])):
            print(f"  !! {len(fs):4d} files on grid {nx}x{ny} x0={x0:.1f} y0={y0:.1f}")
        raise SystemExit(
            "ABORT: mixed grids in one data dir. Delete the minority set, or\n"
            "       re-download everything with one bbox (old files removed first).")


def check_stack_coverage(stack_path, warn_below=5.0):
    """Warn if a rebased stack has almost no valid displacement over the AOI.

    A frame footprint can intersect the AOI while the bursts never image it
    (swath edge / water): the stack writes fine but is all-NaN there. Rebased
    NaN is monotonic (it only ever grows), so the first epoch is the best case:
    if that is empty the frame never covers the AOI at all.
    """
    import numpy as np
    import xarray as xr
    with xr.open_dataset(stack_path) as d:
        var = next((v for v in ("displacement", "short_wavelength_displacement")
                    if v in d.data_vars), None)
        if var is None:
            return
        da = d[var]
        first = da.isel({da.dims[0]: 0}).values
        pct = 100.0 * np.isfinite(first).mean()
    if pct < warn_below:
        print(f"  !! LOW COVERAGE: {os.path.basename(stack_path)} only {pct:.1f}% "
              f"valid over the AOI -- drop this frame from later stages.")
    else:
        print(f"        stack coverage over AOI: {pct:.1f}% valid")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':

    for frame, paths in frames.items():
        bbox = frame_bbox[frame]
        data = paths['data_dir']
        os.makedirs(data, exist_ok=True)
        print(f"\n=== frame {frame} -> {paths['dir']} ===")

        # 1) Small test patch first: verify the subset ROI before the big download
        print(f"  [1/3] ROI test patch ({VERIFY_WIN[0]}..{VERIFY_WIN[1]})...")
        res = verify_roi(frame, bbox, paths['dir'])
        if res is None:
            print("        no products in the test window -- proceeding with caution.")
        elif not res[0]:
            print(f"  !! ROI MISMATCH: subset {tuple(round(v, 4) for v in res[1])} "
                  f"vs requested {tuple(bbox)} -- skipping this frame.")
            continue
        else:
            print(f"        ROI verified: {tuple(round(v, 4) for v in res[1])}")

        # 2) Disk check (per-product size measured from the verify patch),
        #    then the full-range download: one call = complete
        vncs = glob.glob(os.path.join(paths['dir'], "_verify_patch", "*.nc"))
        if vncs:
            per_gb = sum(os.path.getsize(f) for f in vncs) / len(vncs) / 1e9
            n_est = max(1, (date.fromisoformat(end) - date.fromisoformat(start)).days // 12)
            utils.require_disk_space(paths['dir'], per_gb * n_est * 2,
                                     f"frame {frame} (~{n_est} dates)")
        print(f"  [2/3] full download {start}..{end} ...")
        subprocess.run([OPERA, "disp-s1-download", "--frame-id", frame,
                        "--bbox", *map(str, bbox),
                        "--start-datetime", start, "--end-datetime", end,
                        "--output-dir", data, "--num-workers", "8"])
        ncs = sorted(glob.glob(os.path.join(data, "*.nc")))
        print(f"        downloaded/subset {len(ncs)} products")

        # 3) Consistency checks, then rebase the moving reference dates into
        #    one continuous stack (corrections applied per REFORMAT_OPTS)
        check_grid_consistency(data)
        check_corrections_group(data)
        print("  [3/3] rebasing reference dates (disp-s1-reformat)...")
        r = subprocess.run([OPERA, "disp-s1-reformat", "--input-files", *ncs,
                            "--output-name", paths['stack'], *REFORMAT_OPTS])
        if r.returncode != 0:
            raise SystemExit(f"ABORT: reformat failed for frame {frame} (exit {r.returncode})")
        print(f"        rebased stack -> {paths['stack']}")
        check_stack_coverage(paths['stack'])
