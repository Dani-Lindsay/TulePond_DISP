#!/usr/bin/env python3
"""
Write the SFCVM grid (sfcvm_grid.py .npz) as a spatialdata SimpleGridDB (density, vs, vp) with spatialdata's own
SimpleGridAscii writer, in the mesh CRS (CSGeo, same PROJ string as the mesh).  Run in the PyLith env:
    python3 write_sfcvm_griddb.py spatialdb/sfcvm_v26-0_aligned_grid.npz spatialdb/mat_sfcvm_v26-0_aligned.spatialdb
"""
import sys
# PyLith build script (pylith/README.md). Ported from drafts/pylith (2026-09-30); every output path comes from
# TulePond_filepaths (2026-10-06 layout: meshes in paths_inputs['mesh'], SFCVM grids in paths_sfcvm['spatialdb']);
# the databases are written where the command line says, relative to the results directory.
import time

import numpy as np
from spatialdata.geocoords.CSGeo import CSGeo
from spatialdata.spatialdb.SimpleGridAscii import createWriter

src, filename = sys.argv[1], sys.argv[2]
g = np.load(src)
x, y, z = g["x"], g["y"], g["z"]
assert np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0) and np.all(np.diff(z) > 0)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
density, vs, vp = (np.asarray(g[k], float).ravel() for k in ("density", "vs", "vp"))
assert np.all(vs > 0) and np.all(vp > vs) and np.all(density > 0)
cs = CSGeo()
cs.inventory.crsString = str(g["crs"])
cs.inventory.spaceDim = 3
cs._configure()
t0 = time.time()
createWriter(filename).write({"points": points, "x": x, "y": y, "z": z, "coordsys": cs, "data_dim": 3,
                              "values": [{"name": "density", "units": "kg/m**3", "data": density},
                                         {"name": "vs", "units": "m/s", "data": vs},
                                         {"name": "vp", "units": "m/s", "data": vp}]})
print(f"wrote {filename}: {x.size} x {y.size} x {z.size} = {points.shape[0]:,} points in {time.time() - t0:.0f} s "
      f"(from {src}: {g['provenance']})")
