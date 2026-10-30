#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Danielle Lindsay

MintPy conversion, referencing, masking, TROPO correction and velocity,
per frame, then a same-orbit mosaic:

1) `disp_nc_to_mintpy`: rebased stack -> mintpy/timeseries.h5 (+ geometry)
2) Reference every frame at the same ground point, the pixel co-located with
   GNSS ZOA1, so InSAR and GNSS share a datum before the tie (run05)
3) Masks: unbroken record; temporal coherence 0.7 (velocity path, and the
   creep time series of the analysis stage) and 0.9 (the per-pixel event maps
   of the analysis stage); both are written as maskTempCoh_07/09.h5
4) TROPO correction via diff.py; velocity via timeseries2velocity.py
5) Mosaic same-orbit frames (09156 + 09157) onto one union grid

The processing order is fixed by three constraints:

1. reference_point before zero-filling: MintPy's 3D path subtracts the
   reference value from every pixel including zeros, so filling first would
   be undone.
2. Zero-fill whole pixels only: MintPy's no-data convention is 0 and
   timeseries2velocity has no NaN switch. A pixel zeroed for part of its
   record would read as a return to the reference value, which is not a
   measured signal. Velocity pixels are therefore required to have an
   unbroken record.
3. TROPO after masking: diff.py preserves no-data zeros through the
   subtraction.

The creep analysis reads `timeseries_tropo.h5` (corrected, unmasked: the
temporal-coherence masks are applied per pixel in the analysis stage, and NaN
in the rebased stack is monotonic, so partial records are kept as NaN).

Output per frame:  <frame>/mintpy/  timeseries.h5, timeseries_tropo.h5,
                   velocity_mintpy.h5, velocity_masked.h5, maskRecord.h5,
                   maskTempCoh_07/09.h5, TROPO.h5
Mosaics:           <data_dir>/mosaic/<orbit>_velocity[_masked].h5,
                   <orbit>_geometry.h5

Run in the conda environment (MintPy CLI tools on PATH).
Source: MintPy (Yunjun et al. 2019): disp_nc_to_mintpy (opera-utils),
reference_point.py, diff.py, timeseries2velocity.py.
"""
import glob
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

import os as _os, sys as _sys        # path shim: import the repo root modules
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from TulePond_filepaths import data_dir, common_paths, frames
from tropo_to_mintpy import build as build_tropo_h5

# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------
COH_VELOCITY   = 0.7    # temporal-coherence mask for the velocity / GNSS-tie path
COH_TIMESERIES = 0.9    # stricter mask written for the analysis stage (notebook 06's per-pixel maps)
RELIABILITY_THRESHOLD = 0.9   # disp_nc_to_mintpy recommended-mask reliability

ref_lat = common_paths['ref_lat']    # ZOA1: shared reference pixel for all frames
ref_lon = common_paths['ref_lon']

ENV_BIN = os.path.dirname(sys.executable)


def coh_tag(thresh):
    """Two-digit tag for mask filenames: 0.7 -> '07', 0.9 -> '09'."""
    return f"{int(round(thresh * 10)):02d}"


def sh(*args):
    """Run a MintPy CLI tool from this env, raising on failure."""
    cmd = [os.path.join(ENV_BIN, args[0])] + [str(a) for a in args[1:]]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise SystemExit(f"ABORT: {args[0]} failed ({r.returncode})")
    return r.stdout


# ---------------------------------------------------------------------------
# Conversion, masks, small HDF5 helpers
# ---------------------------------------------------------------------------
def h5_dates(path):
    """Acquisition dates in a MintPy timeseries-format file, as YYYYMMDD."""
    with h5py.File(path, "r") as f:
        return [d.decode() for d in f["date"][:]]


def stack_dates(stack_path):
    """Acquisition dates in a rebased stack (run01), as YYYYMMDD."""
    import xarray as xr
    with xr.open_dataset(stack_path) as d:
        return [str(t)[:10].replace("-", "") for t in d["time"].values]


def ensure_mintpy(paths):
    """Convert the rebased stack unless mintpy/timeseries.h5 already covers the
    same dates.

    The check is on the dates, not on the file existing: a data refresh adds
    epochs to the stack and leaves the previous timeseries.h5 in place, and
    skipping on existence alone would carry the previous dates through the
    masks, TROPO.h5, the velocity fit and the mosaics. `_write_hdf5` opens with
    mode "w", so rebuilding overwrites; nothing has to be deleted first.
    """
    from opera_utils.disp.mintpy import disp_nc_to_mintpy
    ts = paths['timeseries']
    want = stack_dates(paths['stack'])
    if os.path.exists(ts):
        have = h5_dates(ts)
        if have == want:
            print(f"    timeseries.h5 already covers the stack "
                  f"({len(have)} epochs to {have[-1]})")
            return ts
        print(f"    timeseries.h5 has {len(have)} epochs to {have[-1]}, the stack has "
              f"{len(want)} to {want[-1]}: rebuilding")
    samples = sorted(glob.glob(os.path.join(paths['data_dir'], "*.nc")))
    print(f"    disp_nc_to_mintpy({os.path.basename(paths['stack'])}) -> mintpy/")
    disp_nc_to_mintpy(Path(paths['stack']), sample_disp_nc=Path(samples[0]),
                      geometry_dir=Path(os.path.join(paths['dir'], "geometry")),
                      outdir=Path(paths['mintpy_dir']),
                      reliability_threshold=RELIABILITY_THRESHOLD)
    return ts


def read_coherence(mdir):
    """average_temporal_coherence, via the copy disp_nc_to_mintpy writes.
    This is a harmonic mean over every 15th epoch, not a plain average."""
    with h5py.File(os.path.join(mdir, "avgSpatialCoh.h5"), "r") as f:
        return f[list(f.keys())[0]][:]


def record_mask(ts_path):
    """True where the pixel has an unbroken record (valid in every epoch).

    NaN in the rebased stack is monotonic (it only begins at a ministack
    reference crossover and never recovers), so this equals
    "valid at the last epoch", and intermediate densities carry no meaning.
    """
    with h5py.File(ts_path, "r") as f:
        ts = f["timeseries"]
        ok = np.ones(ts.shape[1:], dtype=bool)
        for i in range(ts.shape[0]):
            ok &= np.isfinite(ts[i])
    return ok


def write_mask(path, mask, meta):
    with h5py.File(path, "w") as f:
        f.create_dataset("mask", data=mask.astype(np.int8))
        for k, v in meta.items():
            f.attrs[k] = v
        f.attrs["FILE_TYPE"] = "mask"
        f.attrs["DATA_TYPE"] = "int8"
        f.attrs["UNIT"] = "1"


def grid_meta(ts_path):
    keep = ["EPSG", "LENGTH", "WIDTH", "X_FIRST", "Y_FIRST", "X_STEP", "Y_STEP",
            "X_UNIT", "Y_UNIT", "UTM_ZONE", "HEIGHT", "WAVELENGTH", "PLATFORM",
            "ORBIT_DIRECTION", "REF_Y", "REF_X", "REF_LAT", "REF_LON"]
    with h5py.File(ts_path, "r") as f:
        return {k: v for k, v in f.attrs.items() if k in keep}


def write_masked_timeseries(ts_path, out_path, keep):
    """Copy timeseries.h5, zeroing every pixel outside `keep` in ALL epochs."""
    with h5py.File(ts_path, "r") as src, h5py.File(out_path, "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        for name in src:
            if name != "timeseries":
                dst.create_dataset(name, data=src[name][:])
        ts = src["timeseries"]
        out = dst.create_dataset("timeseries", shape=ts.shape, dtype="float32")
        for i in range(ts.shape[0]):
            a = ts[i].astype("float32")
            a[~keep] = 0.0     # whole-pixel, every epoch; kept pixels have an
            #                    unbroken record so nothing here is a partial fill
            out[i] = np.nan_to_num(a, nan=0.0)
    return out_path


def apply_mask_to_velocity(vel_path, keep, out_path):
    with h5py.File(vel_path, "r") as src, h5py.File(out_path, "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        for name in src:
            d = src[name][:]
            if d.shape == keep.shape:
                d = d.astype("float32")
                d[~keep] = np.nan          # NaN, not 0: velocity 0 is meaningful
            dst.create_dataset(name, data=d)
    return out_path


# ---------------------------------------------------------------------------
# Same-orbit mosaic
# ---------------------------------------------------------------------------
# The two ascending frames (09156 + 09157, same track) share the ZOA1 reference
# pixel and the same look geometry, and their grids share the 30 m UTM lattice,
# so the merge is an integer index copy with no resampling (asserted below).
# The lone descending frame is copied through, so downstream always reads one
# file per orbit direction from mosaic/.

def frame_grid(mdir):
    """Grid description of one frame from its timeseries.h5 attributes.
    X_FIRST/Y_FIRST are the coordinates of the first pixel corner."""
    with h5py.File(os.path.join(mdir, "timeseries.h5"), "r") as f:
        a = f.attrs
        return dict(x0=float(a["X_FIRST"]), y0=float(a["Y_FIRST"]),
                    dx=float(a["X_STEP"]), dy=float(a["Y_STEP"]),
                    W=int(a["WIDTH"]), L=int(a["LENGTH"]), epsg=int(a["EPSG"]),
                    orbit=str(a["ORBIT_DIRECTION"]).lower())


def union_grid(grids):
    """Smallest grid on the shared pixel lattice containing every frame.
    Y_STEP is negative, so the top edge is the maximum y0."""
    dx, dy = grids[0]["dx"], grids[0]["dy"]
    x0 = min(g["x0"] for g in grids)
    y0 = max(g["y0"] for g in grids)
    x1 = max(g["x0"] + g["W"] * dx for g in grids)
    y1 = min(g["y0"] + g["L"] * dy for g in grids)
    return dict(x0=x0, y0=y0, dx=dx, dy=dy, epsg=grids[0]["epsg"],
                W=int(round((x1 - x0) / dx)), L=int(round((y1 - y0) / dy)))


def place(arr, grid, cg):
    """Drop `arr` (living on `grid`) into a cg-shaped array, NaN elsewhere.
    Placement is by integer pixel offset; nothing is resampled."""
    out = np.full((cg["L"], cg["W"]), np.nan, dtype="float32")
    c0 = int(round((grid["x0"] - cg["x0"]) / cg["dx"]))
    r0 = int(round((grid["y0"] - cg["y0"]) / cg["dy"]))
    out[r0:r0 + grid["L"], c0:c0 + grid["W"]] = arr
    return out


def merge_orbit(frame_dirs, in_name, out_path):
    """Mosaic one file (`in_name`) across same-orbit frames.

    The largest frame is primary; smaller frames backfill only where they
    actually observed (finite avgSpatialCoh) AND the primary did not, so in
    the overlap the primary's pixels stand. Every 2D dataset in the file is
    carried on the same footprint.
    """
    grids = [frame_grid(os.path.join(d, "mintpy")) for d in frame_dirs]
    order = sorted(range(len(grids)), key=lambda i: -grids[i]["W"] * grids[i]["L"])
    frame_dirs = [frame_dirs[i] for i in order]        # primary (largest) first
    grids = [grids[i] for i in order]
    cg = union_grid(grids)

    # refuse to merge grids that are not on a shared integer lattice: a
    # fractional pixel offset would mean the frames sample different ground
    # and the copy below would silently shift data
    for g in grids:
        cx = (g["x0"] - cg["x0"]) / cg["dx"]
        cy = (g["y0"] - cg["y0"]) / cg["dy"]
        if abs(cx - round(cx)) > 1e-6 or abs(cy - round(cy)) > 1e-6:
            raise SystemExit(f"ABORT: frames not on a shared pixel lattice "
                             f"({cx:.3f}, {cy:.3f} px); merge would resample")

    prim = os.path.join(frame_dirs[0], "mintpy", in_name)
    with h5py.File(prim, "r") as f:
        names = [k for k in f
                 if f[k].ndim == 2 and f[k].shape == (grids[0]["L"], grids[0]["W"])]
        attrs = dict(f.attrs.items())

    out = {k: np.full((cg["L"], cg["W"]), np.nan, dtype="float32") for k in names}
    covered = np.zeros((cg["L"], cg["W"]), bool)
    for d, g in zip(frame_dirs, grids):
        m = os.path.join(d, "mintpy")
        with h5py.File(os.path.join(m, "avgSpatialCoh.h5"), "r") as f:
            cov = np.isfinite(place(f["avgSpatialCoh"][:], g, cg))  # observed footprint
        take = cov & ~covered
        with h5py.File(os.path.join(m, in_name), "r") as f:
            for k in names:
                out[k][take] = place(f[k][:], g, cg)[take]
        covered |= take

    # reference pixel: REF_LON/REF_LAT are the UTM coordinate of the pixel
    # CENTRE, so subtract half a step before dividing
    attrs.update(X_FIRST=str(cg["x0"]), Y_FIRST=str(cg["y0"]),
                 WIDTH=str(cg["W"]), LENGTH=str(cg["L"]))
    if "REF_LON" in attrs and "REF_LAT" in attrs:
        rx = int(round((float(attrs["REF_LON"]) - cg["x0"]) / cg["dx"] - 0.5))
        ry = int(round((float(attrs["REF_LAT"]) - cg["y0"]) / cg["dy"] - 0.5))
        attrs.update(REF_X=str(rx), REF_Y=str(ry))
    with h5py.File(out_path, "w") as f:
        for k, v in attrs.items():
            f.attrs[k] = v
        for k in names:
            f.create_dataset(k, data=out[k])


def merge_frames():
    """Merge each orbit direction's frames into mosaic/<orbit>_*.h5.

    geometryGeo is merged alongside the velocities so the decomposition can
    read per-pixel incidence and azimuth on the same union grid, instead of a
    track-mean constant.
    """
    groups = defaultdict(list)
    for frame, paths in frames.items():
        g = frame_grid(paths['mintpy_dir'])
        groups[g["orbit"]].append(paths['dir'])

    mdir = os.path.join(data_dir, "mosaic")
    os.makedirs(mdir, exist_ok=True)
    print(f"\n=== merge -> mosaic/ ===")
    for orbit, dirs in groups.items():
        for in_name, tag in (("velocity_mintpy.h5", "velocity"),
                             ("velocity_masked.h5", "velocity_masked"),
                             ("geometryGeo.h5", "geometry")):
            out = os.path.join(mdir, f"{orbit}_{tag}.h5")
            if len(dirs) == 1:
                shutil.copy2(os.path.join(dirs[0], "mintpy", in_name), out)
            else:
                merge_orbit(dirs, in_name, out)
            with h5py.File(out, "r") as f:
                key = "velocity" if "velocity" in f else "incidenceAngle"
                fin = np.isfinite(f[key][:])
                print(f"    {orbit}_{tag}.h5  {len(dirs)} frame(s)  "
                      f"{f.attrs['WIDTH']}x{f.attrs['LENGTH']}  "
                      f"{100*fin.mean():.1f}% valid")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':

    for frame, paths in frames.items():
        mdir = paths['mintpy_dir']
        print(f"\n=== frame {frame} ===")

        # 1) Convert the rebased stack to MintPy layout
        ts = ensure_mintpy(paths)
        meta0 = grid_meta(ts)

        # 2) Reference at ZOA1 (constraint 1: before any zero-filling).
        #    Idempotent: referencing zeroes the ref pixel's series and MintPy
        #    treats 0 as no-data, so a second run would reject the pixel;
        #    skipped if already referenced (delete timeseries.h5 to move it).
        if "REF_Y" not in meta0:
            print(f"    reference_point.py at ZOA1 {ref_lat}, {ref_lon}")
            sh("reference_point.py", ts, "-l", ref_lat, "-L", ref_lon)
        else:
            print(f"    reference already set (y/x {meta0['REF_Y']}/{meta0['REF_X']})")
        meta = grid_meta(ts)

        # 3) Masks: unbroken record + temporal coherence
        coh = read_coherence(mdir)
        rec = record_mask(ts)
        keep07 = rec & np.isfinite(coh) & (coh >= COH_VELOCITY)
        keep09 = rec & np.isfinite(coh) & (coh >= COH_TIMESERIES)
        print(f"    unbroken record      {rec.sum():>7d}  ({100*rec.mean():4.1f}%)")
        print(f"    + coherence >= {COH_VELOCITY}   {keep07.sum():>7d}  ({100*keep07.mean():4.1f}%)")
        print(f"    + coherence >= {COH_TIMESERIES}   {keep09.sum():>7d}  ({100*keep09.mean():4.1f}%)")
        write_mask(os.path.join(mdir, "maskRecord.h5"), rec, meta)
        write_mask(os.path.join(mdir, f"maskTempCoh_{coh_tag(COH_VELOCITY)}.h5"), keep07, meta)
        write_mask(os.path.join(mdir, f"maskTempCoh_{coh_tag(COH_TIMESERIES)}.h5"), keep09, meta)

        # 4) Zero-fill outside the velocity mask (constraint 2), then TROPO
        #    (constraint 3), then velocity
        msk = os.path.join(mdir, "timeseries_msk.h5")
        write_masked_timeseries(ts, msk, keep07)
        build_tropo_h5(paths['dir'])
        vel_in = os.path.join(mdir, "timeseries_msk_tropo.h5")
        print("    diff.py <- TROPO.h5")
        sh("diff.py", msk, os.path.join(mdir, "TROPO.h5"), "-o", vel_in)

        # corrected, unmasked time series for the creep analysis stage.
        # Rebuilt whenever its dates differ from timeseries.h5's: on a refresh
        # the previous file is still here and is what notebooks 03 and 06
        # read, so skipping on existence alone would pass them stale data.
        tropo_ts = paths['timeseries_tropo']
        if os.path.exists(tropo_ts) and h5_dates(tropo_ts) == h5_dates(ts):
            print(f"    timeseries_tropo.h5 already covers {len(h5_dates(ts))} epochs")
        else:
            print("    diff.py -> timeseries_tropo.h5 (unmasked, for analysis)")
            sh("diff.py", ts, os.path.join(mdir, "TROPO.h5"), "-o", tropo_ts)

        vel = os.path.join(mdir, "velocity_mintpy.h5")
        print(f"    timeseries2velocity.py {os.path.basename(vel_in)}")
        sh("timeseries2velocity.py", vel_in, "-o", vel)
        apply_mask_to_velocity(vel, keep07, paths['velocity_masked'])
        with h5py.File(paths['velocity_masked'], "r") as f:
            v = f["velocity"][:]
            g = np.isfinite(v)
            print(f"    velocity_masked.h5: {100*g.mean():.1f}% valid, "
                  f"median {1000*np.median(v[g]):+.2f} mm/yr")

        # 5) Drop the velocity intermediates: each is a full time-series copy
        #    and regenerates in seconds from the kept inputs
        for name in ("timeseries_msk.h5", "timeseries_msk_tropo.h5"):
            p = os.path.join(mdir, name)
            if os.path.exists(p):
                os.remove(p)

    # 6) Same-orbit mosaic: one velocity + geometry field per direction
    merge_frames()
    print("\ndone.")
