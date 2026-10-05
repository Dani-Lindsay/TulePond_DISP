#!/usr/bin/env python3
"""
Kinematic (FaultCohesiveKin) slip database for the forward validation runs: uniform RIGHT-lateral slip
(final_slip_left_lateral = -slip, PyLith's left-lateral with this mesh's fault orientation; the sign was derived
in build_G.py: right-lateral = -1 x left-lateral) where y_lo <= y <= y_hi and z >= -depth, 0 elsewhere.
Written with spatialdata's SimpleGridAscii writer, queried "nearest" (grid lines 0.5 m either side of each edge).
Run in the PyLith env:  python3 make_kin_slip_db.py <y_lo> <y_hi> <depth_m> <slip_m> <out.spatialdb>
"""
import sys

import numpy as np
from spatialdata.geocoords.CSGeo import CSGeo
from spatialdata.spatialdb.SimpleGridAscii import createWriter

CRS = "+proj=tmerc +datum=WGS84 +lon_0=-121.968 +lat_0=37.557 +k=1 +units=m +type=crs"
y_lo, y_hi, depth_m, slip_m, filename = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
x = np.array([-60e3, 60e3])
y = np.array([-60e3, y_lo - 0.5, y_lo, y_hi, y_hi + 0.5, 60e3])
z = np.array([-50e3, -depth_m - 0.5, -depth_m, 1.0e3])
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
points = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
inside = (points[:, 1] >= y_lo) & (points[:, 1] <= y_hi) & (points[:, 2] >= -depth_m)
cs = CSGeo(); cs.inventory.crsString = CRS; cs.inventory.spaceDim = 3; cs._configure()
zeros = np.zeros(len(points))
createWriter(filename).write({"points": points, "x": x, "y": y, "z": z, "coordsys": cs, "data_dim": 3, "values": [
    {"name": "initiation_time", "units": "s", "data": zeros},
    {"name": "final_slip_left_lateral", "units": "m", "data": -slip_m * inside},
    {"name": "final_slip_reverse", "units": "m", "data": zeros},
    {"name": "final_slip_opening", "units": "m", "data": zeros}]})
print(f"wrote {filename}: right-lateral {slip_m} m for y in [{y_lo:.0f}, {y_hi:.0f}], z >= {-depth_m:.0f}")
