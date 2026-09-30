#!/usr/bin/env python3
"""
Kinematic slip database for a PyLith FaultCohesiveKin forward run of an INVERTED slip model: one point per fault
vertex (all vertices of the fault mesh, locked ones with 0) with the vertex's slip, written with spatialdata's own
SimpleIOAscii writer (SimpleDB, queried "nearest": the fault vertices coincide with the points, so the tent slip
field of the inversion is reproduced exactly). Left-lateral = sign_rl x right-lateral (build_G's derived sign).
Run in the PyLith env:  python3 make_vertex_slip_db.py <in.npz: xyz (m, mesh CRS), slip_rl_m, sign_rl> <out.spatialdb>
"""
import sys

import numpy as np
from spatialdata.geocoords.CSGeo import CSGeo
from spatialdata.spatialdb.SimpleIOAscii import createWriter

CRS = "+proj=tmerc +datum=WGS84 +lon_0=-121.968 +lat_0=37.557 +k=1 +units=m +type=crs"
src, filename = sys.argv[1], sys.argv[2]
z = np.load(src)
xyz, slip_rl, sign_rl = np.asarray(z["xyz"], float), np.asarray(z["slip_rl_m"], float), float(z["sign_rl"])
assert xyz.shape == (slip_rl.size, 3) and abs(abs(sign_rl) - 1.0) < 1e-12
cs = CSGeo(); cs.inventory.crsString = CRS; cs.inventory.spaceDim = 3; cs._configure()
zeros = np.zeros(slip_rl.size)
createWriter(filename).write({"points": xyz, "coordsys": cs, "data_dim": 3, "values": [
    {"name": "initiation_time", "units": "s", "data": zeros},
    {"name": "final_slip_left_lateral", "units": "m", "data": sign_rl * slip_rl},
    {"name": "final_slip_reverse", "units": "m", "data": zeros},
    {"name": "final_slip_opening", "units": "m", "data": zeros}]})
print(f"wrote {filename}: {slip_rl.size} fault vertices, {np.sum(np.abs(slip_rl) > 1e-9)} with slip, "
      f"max right-lateral {slip_rl.max() * 1e3:.2f} mm (left-lateral = {sign_rl:+.0f} x)")
