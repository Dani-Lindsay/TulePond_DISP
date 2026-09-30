#!/usr/bin/env python3
"""
Write the PyLith PointsList of the 2386 quadtree samples of 07_slip_model (fig2_model_row.npz) in the mesh CRS.

Run in opera_disp_feb13 from the repo root (needs pyproj). Output:
  drafts/pylith/points/insar_points.txt   "name x y z" (m, mesh CRS, z = 0 on the flat free surface)
  drafts/pylith/points/insar_points.npz   lon, lat, x, y, names, crs, source file (for matching PyLith output)
"""
import os
import time

import numpy as np
import pyproj

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CRS = "+proj=tmerc +datum=WGS84 +lon_0=-121.968 +lat_0=37.557 +k=1 +units=m +type=crs"   # = generate_gmsh_vertical.py
source = os.path.join(REPO, "outputs", "07_slip_model_tests", "fig2_model_row.npz")

row = np.load(source)
lon, lat = np.asarray(row["sample_lon"], float), np.asarray(row["sample_lat"], float)
to_mesh = pyproj.Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
x, y = to_mesh.transform(lon, lat)
lon_back, lat_back = to_mesh.transform(x, y, direction="INVERSE")
assert np.max(np.abs(lon_back - lon)) < 1e-9 and np.max(np.abs(lat_back - lat)) < 1e-9, "CRS round trip failed"
assert np.all(np.abs(x) < 45e3) and np.all(np.abs(y) < 45e3), "a sample lies outside the mesh domain"
names = np.array([f"s{k:04d}" for k in range(lon.size)])

os.makedirs(os.path.join(HERE, "points"), exist_ok=True)
with open(os.path.join(HERE, "points", "insar_points.txt"), "w") as fh:
    fh.write(f"# quadtree samples of 07_slip_model ({source}, {lon.size} points), written {time.strftime('%Y-%m-%d %H:%M')}\n")
    fh.write(f"# name x y z (m) in {CRS}\n")
    for n, xi, yi in zip(names, x, y):
        fh.write(f"{n} {xi:.3f} {yi:.3f} 0.0\n")
np.savez(os.path.join(HERE, "points", "insar_points.npz"), lon=lon, lat=lat, x=x, y=y, names=names, crs=CRS,
         source=source)
print(f"wrote {lon.size} points: x {x.min():.0f}..{x.max():.0f} m, y {y.min():.0f}..{y.max():.0f} m")
