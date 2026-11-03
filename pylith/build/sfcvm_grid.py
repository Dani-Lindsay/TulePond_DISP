#!/usr/bin/env python3
"""
USGS SFCVM v26.0 sampled onto an axis-aligned, non-uniform grid in the mesh CRS, aligned so that the model's
Hayward fault (the boundary between SFCVM fault blocks 22 "Central Bay Block" and 7 "San Leandro") at the
surface lies on the mapped creeping trace (the PyLith mesh fault). Written as an .npz; write_sfcvm_griddb.py
(PyLith env) turns it into a spatialdata SimpleGridDB with spatialdata's own writer.

Run in the environment.yml environment:  python pylith/build/sfcvm_grid.py [--no-align] [--no-cap] [--offset <m>] [--dip <d>]
  -> <model_sfcvm_dir>/spatialdb/sfcvm_v26-0_<tag>_grid.npz
  --no-align: the model as is (sensitivity test)
  --no-cap: no Poisson cap (the grids of the order-2 runs, including the preferred model)
  --offset 50: alignment sensitivity, the surface boundary is put 50 m east of the trace (-50: west)
  --dip 80e: align to the dipping mesh fault of generate_gmsh_dip.py: at every depth level the block boundary
  (found at that depth, nearest the fault's position depth/tan(dip) on the dipping side) is moved onto the fault,
  so the material contrast sits on the fault at all depths (the vertical alignment shifts columns whole). Not used
  by the order-2 runs, whose dipping faults use the vertically aligned grid.

Reader: the GeoModelGrids HDF5 layout of the SFCVM (get_model_values: model CRS from attrs["crs"], subtract
origin, rotate by y_azimuth, depth d -> logical z = -d (topography squash neglected), four resolution blocks,
nearest grid point), read with h5py and vectorised by reading each block's bounding slab once.

Alignment (a local cross-fault shift, dip kept):
  1. b(km): at every trace point (100 m, km 50-69 from Point Pinole) the {22|23}|7 block boundary at the surface
     (west block 22 Central Bay north of km ~65, 23 Evergreen south of it; east block 7 San Leandro)
     nearest the trace along the local normal (east +), from 10 m samples within +/- 1.5 km. Missing values are
     interpolated along strike; b is then smoothed by a 5-point (500 m) running median.
  2. Every grid point takes the SFCVM value at  p + b(km_p) * taper(|d_p|) * n_p , where km_p, d_p (signed
     distance, east +) and n_p (unit normal) belong to the nearest trace point: taper = 1 within 3 km of the
     trace, falling linearly to 0 at 6 km (so the far field is the unshifted model); beyond the trace ends the end
     values are used. Each column is shifted as a whole, so the model's dip is kept.
Material rules:
  - water/air/no-data (vs <= 0 or density <= 0) take the first valid value below in the same column;
  - no Vs floor: the model's own values; Vs < vs_flag = 90 m/s is listed in a csv;
  - without --no-cap, Poisson's ratio is capped at 0.40: vp = min(vp, sqrt(6) vs) (vs kept); vp/vs >= 1.5 asserted.
Source: USGS San Francisco Bay region 3D seismic velocity model (SFCVM) v26.0, detailed model (GeoModelGrids HDF5).
"""
import json
# PyLith build script (pylith/README.md). Output paths come from TulePond_filepaths (meshes in paths_inputs['mesh'],
# SFCVM grids in paths_sfcvm['spatialdb']); the databases are written where the command line says, relative to the
# results directory.
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd
import pyproj
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
from TulePond_filepaths import common_paths, inputs_dir, paths_sfcvm   # noqa: E402
SPATIALDB = paths_sfcvm["spatialdb"]
os.makedirs(SPATIALDB, exist_ok=True)
align = "--no-align" not in sys.argv
dip = sys.argv[sys.argv.index("--dip") + 1] if "--dip" in sys.argv else "90"     # e.g. 80e: align to the DIPPING mesh fault
assert dip in ("90", "90v") or (dip[:-1].isdigit() and dip[-1] in "ew"), f"--dip {dip!r}"   # 90v: vertical, per-depth alignment
assert align or dip == "90"
nocap = "--no-cap" in sys.argv                                    # no Poisson cap (the order-2 runs)
offset_m = float(sys.argv[sys.argv.index("--offset") + 1]) if "--offset" in sys.argv else 0.0   # alignment sensitivity: the
assert align or offset_m == 0.0                                    # boundary placed offset_m EAST (+) of the trace instead of on it
off_tag = "" if offset_m == 0.0 else (f"_e{abs(offset_m):g}" if offset_m > 0 else f"_w{abs(offset_m):g}")
tag = ("aligned" if align else "asis") + ("" if dip == "90" else f"_dip{dip}") + ("_nocap" if nocap else "") + off_tag
CRS = "+proj=tmerc +datum=WGS84 +lon_0=-121.968 +lat_0=37.557 +k=1 +units=m +type=crs"   # = generate_gmsh_vertical.py
version = "v26-0"
model_file = os.path.join(inputs_dir, f"USGS_SFCVM_{version}", f"USGS_SFCVM_{version}_detailed.h5")
assert os.path.exists(model_file), f"missing {model_file}"
vs_flag, nu_cap = 90.0, 0.40     # vs_flag: Vs below this is reported, not changed (no Vs floor)
vpvs_cap = np.sqrt((2 - 2 * nu_cap) / (1 - 2 * nu_cap))            # sqrt(6) for nu 0.40
taper_full_m, taper_zero_m = 3000.0, 6000.0
blocks_west, block_east = (22, 23), 7                               # Central Bay Block (Evergreen south of km ~65) | San Leandro


def axis(fine_lo, fine_hi, d_fine, half, growth=1.4, d_max=4000.0):
    """Fine spacing d_fine on [fine_lo, fine_hi], then growing by `growth` per step (max d_max) out to +/-half."""
    fine = np.arange(fine_lo, fine_hi + 0.5 * d_fine, d_fine)
    up, step = [fine[-1]], d_fine
    while up[-1] < half:
        step = min(step * growth, d_max); up.append(min(up[-1] + step, half))
    down, step = [fine[0]], d_fine
    while down[-1] > -half:
        step = min(step * growth, d_max); down.append(max(down[-1] - step, -half))
    return np.r_[down[:0:-1], fine, up[1:]]


grid_half = 51e3                                  # 1 km beyond the 50 km domain: with materials of basis order 1 the
                                                  # db is queried at the boundary vertices
x_axis = axis(-3500.0, 2500.0, 50.0, grid_half)   # fine box: the trace in km 55-64 is at x -2700..2200 m
y_axis = axis(-4000.0, 3600.0, 50.0, grid_half)   # km ~55.3-63.6
z_axis = -np.r_[np.arange(0, 500, 25), np.arange(500, 1500, 50), np.arange(1500, 3000, 100),
                np.arange(3000, 5000, 250), np.arange(5000, 10000, 1000), np.arange(10000, 40001, 3000), 40000.0, 41000.0][::-1]
z_axis = np.unique(np.r_[z_axis, 100.0])          # +100 m level above the flat free surface (z = 0)
print(f"grid: {x_axis.size} x {y_axis.size} x {z_axis.size} = {x_axis.size * y_axis.size * z_axis.size:,} points; "
      f"x {x_axis.min():.0f}..{x_axis.max():.0f}, y {y_axis.min():.0f}..{y_axis.max():.0f}, z {z_axis.min():.0f}..{z_axis.max():.0f}")

# ------------------------------------------------------------------ SFCVM reader (GeoModelGrids layout)
model = h5py.File(model_file, "r")
attrs = model.attrs
crs_wkt = attrs["crs"] if isinstance(attrs["crs"], str) else attrs["crs"].decode()
to_model = pyproj.Transformer.from_crs(CRS, pyproj.CRS.from_wkt(crs_wkt), always_xy=True)
origin_x, origin_y, y_azimuth = float(attrs["origin_x"]), float(attrs["origin_y"]), float(attrs["y_azimuth"])
aux = json.loads(attrs["auxiliary"])
zone_names = {int(v): k for k, v in aux["zone_ids"].items() if v > -1e10}
blocks = [("vres25m", 100, 25, 0.0), ("vres50m", 200, 50, -500.0), ("vres125m", 500, 125, -3500.0),
          ("vres250m", 1000, 250, -10000.0)]
value_cols = [0, 1, 2, 5, 6]                     # density, Vp, Vs, fault_block_id, zone_id


def model_xy(x_mesh, y_mesh):
    """mesh CRS -> SFCVM logical x, y (m) [get_model_values]."""
    easting, northing = to_model.transform(x_mesh, y_mesh)
    d_east, d_north = easting - origin_x, northing - origin_y
    my = d_east * np.sin(np.deg2rad(y_azimuth)) + d_north * np.cos(np.deg2rad(y_azimuth))
    mx = d_east * np.sin(np.deg2rad(y_azimuth + 90)) + d_north * np.cos(np.deg2rad(y_azimuth + 90))
    return mx, my


def sample(mx, my, depth):
    """Nearest-grid-point values (density, vp, vs, fault_block, zone) at model x, y and depth (m below surface).
    Vectorised get_model_values: one bounding-slab read per resolution block; x, y clamped to the model."""
    mx, my, depth = np.broadcast_arrays(np.asarray(mx, float), np.asarray(my, float), np.asarray(depth, float))
    out = np.full(mx.shape + (len(value_cols),), np.nan)
    z = -depth
    for name, h_res, dz, z_top in blocks:
        ds = model["blocks/" + name]
        nx, ny, nz = ds.shape[:3]
        inb = (z <= z_top + 1e-6) & (z >= z_top - (nz - 1) * dz - 1e-6)
        if name == "vres250m":
            inb |= z < z_top - (nz - 1) * dz        # below the model bottom: deepest layer
        if not inb.any():
            continue
        ix = np.clip(np.rint(mx[inb] / h_res).astype(int), 0, nx - 1)
        iy = np.clip(np.rint(my[inb] / h_res).astype(int), 0, ny - 1)
        iz = np.clip(np.rint((z_top - z[inb]) / dz).astype(int), 0, nz - 1)
        x0, x1, y0, y1, z0, z1 = ix.min(), ix.max() + 1, iy.min(), iy.max() + 1, iz.min(), iz.max() + 1
        slab = ds[x0:x1, y0:y1, z0:z1, :][..., value_cols]
        out[inb] = slab[ix - x0, iy - y0, iz - z0]
    assert np.isfinite(out).all(), "a query fell outside every SFCVM block"
    return out


# ------------------------------------------------------------------ the mapped trace (the mesh fault) in the mesh CRS
profiles = pd.read_csv(common_paths["fault_profiles"])
km_pp = (len(profiles) - 1 - np.arange(len(profiles))) * 0.1       # km from Point Pinole (as model_utils.km_from_pinole)
keep = (km_pp >= 50.0 - 1e-9) & (km_pp <= 69.0 + 1e-9)
to_mesh = pyproj.Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
tx, ty = to_mesh.transform(profiles.ft_lon.values[keep], profiles.ft_lat.values[keep])
tkm = km_pp[keep]
order = np.argsort(tkm); tx, ty, tkm = tx[order], ty[order], tkm[order]
tangent = np.c_[np.gradient(tx), np.gradient(ty)]                 # direction of increasing km (southward)
tangent /= np.linalg.norm(tangent, axis=1)[:, None]
normal = np.c_[-tangent[:, 1], tangent[:, 0]]
normal[normal[:, 0] < 0] *= -1.0                                  # east +
assert np.all(normal[:, 0] > 0.5), "trace normal not roughly east"

# 1. b(km): the Hayward block boundary at the surface ({22 or 23} | 7), east of the trace
offsets = np.arange(-1500.0, 1500.1, 10.0)
b_raw = np.full(tkm.size, np.nan)
n_multi = 0
for i in range(tkm.size):
    px, py = tx[i] + offsets * normal[i, 0], ty[i] + offsets * normal[i, 1]
    fb = np.rint(sample(*model_xy(px, py), 0.0)[:, 3]).astype(int)
    change = np.nonzero(fb[1:] != fb[:-1])[0]
    cand = [0.5 * (offsets[j] + offsets[j + 1]) for j in change if block_east in (fb[j], fb[j + 1]) and ({fb[j], fb[j + 1]} - {block_east}) <= set(blocks_west)]
    if cand:
        n_multi += len(cand) > 1
        b_raw[i] = cand[int(np.argmin(np.abs(cand)))]
n_missing = int(np.isnan(b_raw).sum())
good = np.isfinite(b_raw)
assert good.sum() > 0.8 * tkm.size, f"22|7 boundary found at only {good.sum()} of {tkm.size} trace points"
b_filled = np.interp(tkm, tkm[good], b_raw[good])
b_smooth = pd.Series(b_filled).rolling(5, center=True, min_periods=1).median().values
band = (tkm >= 57.5) & (tkm <= 61.5)
print(f"Hayward block boundary ({{22|23}}|7) east of the trace: found at {good.sum()}/{tkm.size} trace points ({n_missing} interpolated, "
      f"{n_multi} with >1 crossing, nearest used); km 57.5-61.5: {b_smooth[band].min():.0f}..{b_smooth[band].max():.0f} m "
      f"(median {np.median(b_smooth[band]):.0f}); km 50-69: {b_smooth.min():.0f}..{b_smooth.max():.0f} m")
b_used = (b_smooth - offset_m) if align else np.zeros_like(b_smooth)   # --no-align: the model as is; --offset: boundary
                                                                          # left offset_m east of the trace (shift reduced by offset_m)


def boundary_at_depth(depth_m, target_m):
    """b(km) of the {22|23}|7 boundary at `depth_m` (m below the surface), the crossing nearest `target_m` (the
    dipping fault's offset from the trace at that depth, east +) at every trace point; missing values interpolated
    along strike, 5-point running median (as at the surface)."""
    P = tx[:, None] + offsets[None, :] * normal[:, 0:1]
    Q = ty[:, None] + offsets[None, :] * normal[:, 1:2]
    fb = np.rint(sample(*model_xy(P.ravel(), Q.ravel()), depth_m)[:, 3]).astype(int).reshape(P.shape)
    b = np.full(tkm.size, np.nan)
    for i in range(tkm.size):
        row = fb[i]
        change = np.nonzero(row[1:] != row[:-1])[0]
        cand = [0.5 * (offsets[j] + offsets[j + 1]) for j in change
                if block_east in (row[j], row[j + 1]) and ({row[j], row[j + 1]} - {block_east}) <= set(blocks_west)]
        if cand:
            b[i] = cand[int(np.argmin(np.abs(np.asarray(cand) - target_m)))]
    good = np.isfinite(b)
    if good.sum() == 0:
        return None, 0
    b = np.interp(tkm, tkm[good], b[good])
    return pd.Series(b).rolling(5, center=True, min_periods=1).median().values, int(good.sum())


if dip != "90":
    # SFCVM aligned to the DIPPING mesh fault: at every grid depth level the block boundary is moved onto the fault's
    # position at that depth, f(z) = depth / tan(dip) on the dipping side (generate_gmsh_dip.py: the trace shifted along
    # the mean normal), i.e. shift(km, z) = b(km, z) - f(z). Below the fault (3 km) the 3 km shift is kept. The
    # horizontal taper (full within 3 km of the trace, zero at 6 km) is the same as for the vertical alignment.
    dip_deg, dip_side = (90.0, 0.0) if dip == "90v" else (float(dip[:-1]), (1.0 if dip[-1] == "e" else -1.0))
    fault_depth_m = 3000.0
    b_levels, n_found = {}, {}
    for d in np.unique(np.maximum(-z_axis, 0.0)):
        f_z = dip_side * min(d, fault_depth_m) / np.tan(np.deg2rad(dip_deg))
        b_z, n_ok = boundary_at_depth(min(d, fault_depth_m), f_z)
        if b_z is None:
            b_z, n_ok = b_levels[max(k for k in b_levels if k < d)], -1     # not found at this depth: previous level
        b_levels[d], n_found[d] = b_z, n_ok
    band = (tkm >= 57.5) & (tkm <= 61.5)
    for d in (0.0, 100.0, 300.0, 600.0, 1000.0, 2000.0, 3000.0):
        f_z = dip_side * min(d, fault_depth_m) / np.tan(np.deg2rad(dip_deg))
        print(f"  depth {d:5.0f} m: boundary found at {n_found[d]}/{tkm.size} trace points; b km 57.5-61.5 median "
              f"{np.median(b_levels[d][band]):+.0f} m; fault at {f_z:+.0f} m -> shift {np.median(b_levels[d][band]) - f_z:+.0f} m")
    assert all(n_found[d] > 0.5 * tkm.size for d in b_levels if d <= fault_depth_m), "boundary lost at some depth"
    assert abs(np.median(b_levels[0.0][band]) - np.median(b_smooth[band])) < 20, "surface boundary differs from the vertical search"

# 2. the grid, shifted and sampled
t0 = time.time()
X, Y = np.meshgrid(x_axis, y_axis, indexing="ij")
dense_tree = cKDTree(np.c_[tx, ty])
dist, nearest = dense_tree.query(np.c_[X.ravel(), Y.ravel()])
signed = np.einsum("ij,ij->i", np.c_[X.ravel() - tx[nearest], Y.ravel() - ty[nearest]], normal[nearest])
taper = np.clip((taper_zero_m - np.abs(signed)) / (taper_zero_m - taper_full_m), 0.0, 1.0)
shift = b_used[nearest] * taper
qx = X.ravel() + shift * normal[nearest, 0]
qy = Y.ravel() + shift * normal[nearest, 1]
mx, my = model_xy(qx, qy)
depth = np.maximum(-z_axis, 0.0)                                   # the +100 m level takes the surface values
values = np.empty((x_axis.size * y_axis.size, z_axis.size, len(value_cols)))
shift_levels = {}
for k, d in enumerate(depth):
    if dip != "90":                                                # per-level shift onto the dipping fault
        f_z = dip_side * min(d, fault_depth_m) / np.tan(np.deg2rad(dip_deg))
        shift_k = (b_levels[d][nearest] - f_z) * taper
        shift_levels[float(d)] = float(np.median(b_levels[d][band]) - f_z)
        mx, my = model_xy(X.ravel() + shift_k * normal[nearest, 0], Y.ravel() + shift_k * normal[nearest, 1])
    values[:, k, :] = sample(mx, my, d)
print(f"sampled {values.shape[0] * values.shape[1]:,} points in {time.time() - t0:.0f} s")
values = values.reshape(x_axis.size, y_axis.size, z_axis.size, len(value_cols))
density, vp, vs = values[..., 0].copy(), values[..., 1].copy(), values[..., 2].copy()
fault_block, zone = np.rint(values[..., 3]).astype(int), np.rint(values[..., 4]).astype(int)

# material rules
invalid = ~((vs > 0) & (density > 0) & (vp > 0))
n_invalid = int(invalid.sum())
order_z = np.argsort(-z_axis)                                      # top (z=+100) to bottom
zi = list(order_z)                                                  # indices from the top down
for pos in range(len(zi) - 2, -1, -1):                              # from the second-deepest level up
    here, below = zi[pos], zi[pos + 1]
    bad = invalid[:, :, here]
    density[:, :, here][bad] = density[:, :, below][bad]
    vp[:, :, here][bad] = vp[:, :, below][bad]
    vs[:, :, here][bad] = vs[:, :, below][bad]
    fault_block[:, :, here][bad] = fault_block[:, :, below][bad]
    zone[:, :, here][bad] = zone[:, :, below][bad]
    invalid[:, :, here] = invalid[:, :, here] & invalid[:, :, below]
assert not invalid.any(), "columns with no valid SFCVM value at any depth"
# No Vs floor: the SFCVM's own values are used (bay-margin sediment can be ~90 m/s). Values under vs_flag are only
# reported (and listed in a csv), not changed.
n_floor = int((vs < 150.0).sum())                                   # the points a 150 m/s floor would change (reported)
low = vs < vs_flag
print(f"Vs: min {vs.min():.1f} m/s; {n_floor:,} points below 150 m/s (the old floor), {int(low.sum()):,} below {vs_flag:g} m/s")
if low.any():
    il, jl, kl = np.nonzero(low)
    low_table = pd.DataFrame(dict(x_m=x_axis[il], y_m=y_axis[jl], depth_m=np.maximum(-z_axis[kl], 0.0), vs_m_s=vs[low],
                                  vp_m_s=vp[low], density=density[low], zone=zone[low]))
    low_file = os.path.join(SPATIALDB, f"sfcvm_{version}_{tag}_vs_below_{vs_flag:g}.csv")
    low_table.to_csv(low_file, index=False)
    print(f"FLAG: {len(low_table):,} grid points with Vs < {vs_flag:g} m/s (min {vs.min():.1f}; depths "
          f"{low_table.depth_m.min():.0f}-{low_table.depth_m.max():.0f} m) -> {low_file}")
n_cap = int((vp > vpvs_cap * vs).sum())
if nocap:
    n_cap = 0                                                       # vp as given (nu up to ~0.5 in the wet sediment)
else:
    vp = np.minimum(vp, vpvs_cap * vs)
assert np.all(vp / vs >= 1.5), f"vp/vs below 1.5: min {np.min(vp / vs):.2f}"
nu = (vp ** 2 - 2 * vs ** 2) / (2 * (vp ** 2 - vs ** 2))
mu = density * vs ** 2
assert np.all(np.isfinite(mu)) and mu.min() > 0 and (nocap or nu.max() <= nu_cap + 1e-9)
print(f"filled {n_invalid:,} water/air/no-data points from below; no Vs floor ({n_floor:,} points below 150 m/s kept); "
      f"vp capped (nu <= {nu_cap}) at {n_cap:,} points of {vs.size:,}")
print(f"mu: {mu.min() / 1e9:.3f}..{mu.max() / 1e9:.1f} GPa; nu {nu.min():.3f}..{nu.max():.3f}")

out = os.path.join(SPATIALDB, f"sfcvm_{version}_{tag}_grid.npz")
np.savez_compressed(out, x=x_axis, y=y_axis, z=z_axis, density=density.astype(np.float32), vp=vp.astype(np.float32),
                    vs=vs.astype(np.float32), fault_block=fault_block.astype(np.int16), zone=zone.astype(np.int16),
                    trace_km=tkm, trace_x=tx, trace_y=ty, trace_normal=normal, boundary_raw_m=b_raw,
                    boundary_smooth_m=b_smooth, shift_used_m=b_used, align=align, crs=CRS, model_file=model_file,
                    zone_names=json.dumps(zone_names), vs_floor=np.nan, vs_flag=vs_flag, nu_cap=(np.nan if nocap else nu_cap),
                    taper_m=np.array([taper_full_m, taper_zero_m]), dip=dip,
                    shift_by_depth=json.dumps(shift_levels),
                    provenance=f"sfcvm_grid.py {time.strftime('%Y-%m-%d %H:%M')} ({tag}); SFCVM {version} {model_file}")
if dip == "90":
    pd.DataFrame(dict(km=tkm, x=tx, y=ty, boundary_raw_m=b_raw, boundary_smooth_m=b_smooth)).to_csv(
        os.path.join(SPATIALDB, f"sfcvm_{version}_boundary_offset.csv"), index=False)
else:
    pd.DataFrame(dict(km=tkm, **{f"b_{int(d)}m": b_levels[d] for d in sorted(b_levels) if d <= 3000})).to_csv(
        os.path.join(SPATIALDB, f"sfcvm_{version}_boundary_offset_dip{dip}.csv"), index=False)
print(f"wrote {out}")
