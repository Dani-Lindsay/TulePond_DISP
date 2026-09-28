#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 2026

@author: daniellelindsay

Ported into TulePond_DISP/data_prep on 2026-09-28 from
Hayward_DISP-S1/analysis/tropo_to_mintpy.py, unchanged except for the imports
(TulePond_filepaths, the repo-root insar_utils) and this note.

Stack run03's per-date TROPO GeoTIFFs into one MintPy `timeseries`-format
HDF5, ready for `diff.py`. Imported by run04; nothing about the physics
changes here -- this is a container conversion, and the layout is copied from
the frame's own `timeseries.h5` rather than assumed.

Target format (read from `mintpy/tropo_pyaps3.py` and a real timeseries.h5):

    datasets    date        (n,)      |S8   'YYYYMMDD'
                timeseries  (n, y, x) float32
    attrs       FILE_TYPE = 'timeseries', UNIT = 'm', plus the grid keys

Three things deliberately NOT done here:

1. No sign flip. opera-utils already negates ("positive means apparent
   uplift") and so does MintPy, so `diff.py` subtracts directly.
2. No spatial referencing. `diff.py` re-references the correction itself;
   referencing here would apply it twice.
3. No REF_DATE / REF_Y / REF_X written. `tropo_pyaps3` strips exactly these,
   and leaving them out is what lets diff.py do (2).

The first acquisition has no differential file (subtract_first_date makes it
identically zero), so that date is written as an explicit zero layer, keeping
the date list identical to timeseries.h5's.
"""
import glob
import os
import re

import h5py
import numpy as np
import rasterio

# Copied onto the output so the two files describe the same grid. Deliberately
# a fixed list: blindly copying every attribute would drag REF_* across too.
GRID_KEYS = [
    "EPSG", "LENGTH", "WIDTH", "X_FIRST", "Y_FIRST", "X_STEP", "Y_STEP",
    "X_UNIT", "Y_UNIT", "UTM_ZONE", "HEIGHT", "WAVELENGTH", "PLATFORM",
    "ORBIT_DIRECTION",
]


def date_of(path):
    """'YYYYMMDD' from a correction filename."""
    return re.search(r"(\d{8})T\d{6}", os.path.basename(path)).group(1)


def build(frame_dir, out_name="inputs/TROPO.h5", ts_name="mintpy/timeseries.h5"):
    ts_path = os.path.join(frame_dir, ts_name)
    corr_dir = os.path.join(frame_dir, "tropo", "corrections")
    out_path = os.path.join(frame_dir, os.path.dirname(ts_name), os.path.basename(out_name))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with h5py.File(ts_path, "r") as f:
        ts_dates = [d.decode() for d in f["date"][:]]
        meta = {k: v for k, v in f.attrs.items() if k in GRID_KEYS}
        ny, nx = f["timeseries"].shape[1:]

    tifs = sorted(glob.glob(os.path.join(corr_dir, "*.tif")))
    by_date = {date_of(p): p for p in tifs}
    ref = [p for p in tifs if "reference" in os.path.basename(p)]
    ref_date = date_of(ref[0]) if ref else None
    print(f"  {len(tifs)} GeoTIFFs, reference date {ref_date}")

    missing = [d for d in ts_dates if d not in by_date]
    if missing:
        raise SystemExit(f"ABORT: {len(missing)} timeseries dates have no TROPO "
                         f"correction, first {missing[0]}. diff.py needs all of them.")

    data = np.zeros((len(ts_dates), ny, nx), dtype=np.float32)
    for i, d in enumerate(ts_dates):
        if d == ref_date:
            # the reference date's differential is zero by construction; the
            # file on disk holds the ABSOLUTE delay and must not be inserted
            continue
        with rasterio.open(by_date[d]) as src:
            if (src.height, src.width) != (ny, nx):
                raise SystemExit(f"ABORT: {by_date[d]} is {src.width}x{src.height}, "
                                 f"timeseries is {nx}x{ny}")
            a = src.read(1)
        data[i] = np.nan_to_num(a, nan=0.0)      # MintPy no-data convention

    meta = dict(meta) | {"FILE_TYPE": "timeseries", "UNIT": "m"}
    with h5py.File(out_path, "w") as f:
        f.create_dataset("date", data=np.array(ts_dates, dtype=np.bytes_))
        f.create_dataset("timeseries", data=data, chunks=True, compression="gzip")
        for k, v in meta.items():
            f.attrs[k] = v
    print(f"  wrote {out_path}  ({len(ts_dates)}, {ny}, {nx})")
    return out_path
