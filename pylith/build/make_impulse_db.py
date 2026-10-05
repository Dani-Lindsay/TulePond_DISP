#!/usr/bin/env python3
"""
Impulse-region spatial database for pylith.faults.FaultCohesiveImpulses: amplitude 1 m of left-lateral slip where
y_lo <= y <= y_hi and z >= -depth (the rest of the fault gets 0, i.e. no impulse = locked), written with
spatialdata's own SimpleGridAscii writer (the pattern of PyLith examples/poroelastic-outerrise-2d/simplegriddb_writer.py).
The fault is near-vertical and runs roughly N-S in the mesh frame, so an along-strike window is a y window
(km 54 at y = 4380 m, km 65 at y = -4713 m; trace in mesh/trace_tmerc.txt).

Run in the PyLith env:  python3 make_impulse_db.py <y_lo> <y_hi> <depth_m> <out.spatialdb>
The database is queried with the default "nearest"; grid lines 0.5 m either side of each edge make the edge sharp.
"""
import sys
# PyLith build script (pylith/README.md). Ported from drafts/pylith (2026-09-30); every output path comes from
# TulePond_filepaths (2026-10-06 layout: meshes in paths_inputs['mesh'], SFCVM grids in paths_sfcvm['spatialdb']);
# the databases are written where the command line says, relative to the results directory.

import numpy as np
from spatialdata.geocoords.CSGeo import CSGeo
from spatialdata.spatialdb.SimpleGridAscii import createWriter

CRS = "+proj=tmerc +datum=WGS84 +lon_0=-121.968 +lat_0=37.557 +k=1 +units=m +type=crs"   # = generate_gmsh_vertical.py
y_lo, y_hi, depth_m, filename = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
assert y_lo < y_hi and depth_m > 0

x = np.array([-60e3, 60e3])
y = np.array([-60e3, y_lo - 0.5, y_lo, y_hi, y_hi + 0.5, 60e3])
z = np.array([-50e3, -depth_m - 0.5, -depth_m, 1.0e3])
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
inside = (points[:, 1] >= y_lo) & (points[:, 1] <= y_hi) & (points[:, 2] >= -depth_m)
cs = CSGeo()
cs.inventory.crsString = CRS
cs.inventory.spaceDim = 3
cs._configure()
writer = createWriter(filename)
writer.write({"points": points, "x": x, "y": y, "z": z, "coordsys": cs, "data_dim": 3,
              "values": [{"name": "slip_left_lateral", "units": "m", "data": inside.astype(float)},
                         {"name": "slip_reverse", "units": "m", "data": np.zeros(len(points))},
                         {"name": "slip_opening", "units": "m", "data": np.zeros(len(points))}]})
print(f"wrote {filename}: impulses for y in [{y_lo:.0f}, {y_hi:.0f}] m, z >= {-depth_m:.0f} m")
