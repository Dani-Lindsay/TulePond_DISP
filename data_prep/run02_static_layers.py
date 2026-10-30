#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Danielle Lindsay

Per-frame geometry layers + a shared DEM, the two inputs the tropospheric
correction (run03) and the MintPy conversion (run04) need:

1) Per frame: download the CSLC static layers for the bursts that intersect
   the AOI, stitch los_east/los_north/layover_shadow_mask, derive incidence
2) Fetch one Copernicus GLO-30 DEM for the whole AOI (ellipsoidal height)
3) Per frame: resample DEM + geometry onto that frame's stack grid, and build
   the 3-band los_enu.tif that run04's converter expects

Two choices:

* Bursts are selected by AOI intersection, not by frame. A DISP-S1 frame has
  ~27 CSLC bursts and a small AOI touches only a few; fetching by frame_id
  would download them all.
* The DEM is ellipsoidal. OPERA TROPO's height coordinate is "Height above
  ellipsoid WGS84" and tropo-apply interpolates the delay cube onto the DEM
  surface across that axis; a geoid-referenced DEM would bias the delay.

Output per frame:  <frame>/geometry/  dem.tif, los_east.tif, los_north.tif,
                   incidence_angle.tif, layover_shadow_mask.tif, los_enu.tif
                   (all on the frame's stack grid; stitched-grid intermediates
                   kept under geometry/stitched/)
Shared:            <data_dir>/dem_glo30.tif

Run in the conda environment. Earthdata credentials in ~/.netrc.
Source: OPERA CSLC-S1-STATIC layers (ASF) and the Copernicus GLO-30 DEM, read and
stitched with opera-utils.
"""
import os
import sys

import numpy as np
import rasterio

import os as _os, sys as _sys        # path shim: import the repo root modules
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from TulePond_filepaths import data_dir, common_paths, frames
import insar_utils as utils

# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------
EPSG = 32610          # UTM 10N: the zone of the AOI centre, and the grid every
#                       frame's DISP-S1 stack is delivered on. One shared zone
#                       means displacement is never resampled and the asc/desc
#                       decomposition is pixel-exact.
DEM_BUFFER_DEG = 0.05

# Per-frame AOI, same as run01 (09156 clipped to its own footprint)
frame_bbox = {
    "09156": common_paths['aoi_bbox_09156'],
    "09157": common_paths['aoi_bbox'],
    "11115": common_paths['aoi_bbox'],
}

# Only these three are STORED datasets in the CSLC static product; incidence is
# DERIVED from los_east/los_north below (asking the stitcher for it fails).
LAYERS = ["los_east", "los_north", "layover_shadow_mask"]
STITCHED_LAYERS = ["los_east", "los_north", "incidence_angle", "layover_shadow_mask"]

# opera_utils.stitching shells out to `gdal_merge.py`, which lives in the conda
# env's bin/ but is not on PATH for a subprocess unless it is added here.
_ENV_BIN = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "bin")
if _ENV_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _ENV_BIN + os.pathsep + os.environ.get("PATH", "")


# ---------------------------------------------------------------------------
# Geometry: burst selection, stitch, incidence
# ---------------------------------------------------------------------------
def bursts_for_aoi(frame_id, aoi):
    """Burst IDs of `frame_id` whose footprint intersects the AOI polygon."""
    from opera_utils.burst_frame_db import get_burst_ids_for_frame, get_burst_geodataframe
    bids = get_burst_ids_for_frame(int(frame_id))
    gdf = get_burst_geodataframe(burst_ids=bids)
    keep = gdf[gdf.intersects(aoi)]["burst_id_jpl"].tolist()
    return bids, keep


def derive_incidence(geo_dir):
    """incidence_angle.tif from los_east/los_north.

    OPERA los_east/los_north are the ground-to-satellite unit vector
    components, so up = sqrt(1 - e^2 - n^2) and incidence = arccos(up).
    This convention is the one tested in tests/test_insar_utils.py.
    """
    e_p = os.path.join(geo_dir, "los_east.tif")
    n_p = os.path.join(geo_dir, "los_north.tif")
    out = os.path.join(geo_dir, "incidence_angle.tif")
    with rasterio.open(e_p) as de, rasterio.open(n_p) as dn:
        e = de.read(1).astype("float64")
        n = dn.read(1).astype("float64")
        prof = de.profile
    bad = (e == 0) & (n == 0)          # nodata in the LOS rasters
    up = np.sqrt(np.clip(1.0 - e**2 - n**2, 0.0, 1.0))
    inc = np.degrees(np.arccos(np.clip(up, -1.0, 1.0)))
    inc[bad] = np.nan
    prof.update(dtype="float32", count=1, nodata=np.nan, compress="deflate")
    with rasterio.open(out, "w", **prof) as ds:
        ds.write(inc.astype("float32"), 1)
    good = inc[~np.isnan(inc)]
    print(f"    incidence: {good.min():.1f}..{good.max():.1f} deg (median {np.median(good):.1f})")
    return out


def build_geometry(frame_id, aoi, bbox, frame_dir):
    """Download AOI-intersecting bursts, stitch the layers cropped to the AOI."""
    from opera_utils.geometry import download_cslc_static_layers, stitch_geometry_layers
    from pyproj import Transformer

    all_b, keep = bursts_for_aoi(frame_id, aoi)
    print(f"    bursts: {len(keep)}/{len(all_b)} intersect the AOI")

    # The per-burst static HDF5s (~120 MB each) all land on disk before the
    # stitch, so the peak is what has to fit; they are deleted after stitching.
    os.makedirs(frame_dir, exist_ok=True)
    utils.require_disk_space(frame_dir, 0.120 * len(keep) * 1.3,
                             f"frame {frame_id} static layers ({len(keep)} bursts)")

    dl_dir = os.path.join(frame_dir, "static_lyrs")
    geo_dir = os.path.join(frame_dir, "geometry", "stitched")
    os.makedirs(dl_dir, exist_ok=True)
    os.makedirs(geo_dir, exist_ok=True)

    print(f"    downloading {len(keep)} CSLC static-layer files -> {dl_dir}")
    h5s = download_cslc_static_layers(burst_ids=keep, output_dir=dl_dir)

    # out_bounds must be expressed in the output EPSG
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True)
    x0, y0 = t.transform(bbox[0], bbox[1])
    x1, y1 = t.transform(bbox[2], bbox[3])
    out_bounds = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    print(f"    stitching {LAYERS} cropped to AOI (EPSG:{EPSG})")
    outs = stitch_geometry_layers(local_hdf5_files=h5s, layers=LAYERS,
                                  output_dir=geo_dir, out_bounds=out_bounds,
                                  out_bounds_epsg=EPSG)
    outs = list(outs) + [derive_incidence(geo_dir)]

    # the per-burst HDF5s are only an intermediate; the stitched tifs are the product
    for f in h5s:
        try:
            os.remove(f)
        except OSError:
            pass
    return outs


# ---------------------------------------------------------------------------
# DEM
# ---------------------------------------------------------------------------
def build_dem(bbox, out_path):
    """Copernicus GLO-30 for the AOI, ellipsoidal height, native EPSG:4326.

    Stitched with a small buffer beyond the AOI: the DEM is a rectangle in
    EPSG:4326 but the stack grid is a rectangle in UTM, and the two are
    rotated relative to each other, so an exactly-AOI-sized DEM leaves
    wedge-shaped nodata at the corners of the UTM grid (pixels that would
    then get no tropospheric delay).

    Not reprojected to UTM here: each frame's dem.tif is one reprojection from
    this DEM straight onto that frame's stack grid, so elevations are
    resampled once.
    """
    from dem_stitcher import stitch_dem
    w, s, e, n = bbox
    b = DEM_BUFFER_DEG
    print(f"  DEM: glo_30 over {bbox} +{b} deg buffer, ellipsoidal height")
    dem, prof = stitch_dem([w - b, s - b, e + b, n + b], dem_name="glo_30",
                           dst_ellipsoidal_height=True, dst_area_or_point="Point")
    prof.update(driver="GTiff", compress="deflate")
    with rasterio.open(out_path, "w", **prof) as ds:
        ds.write(dem, 1)
    return out_path


# ---------------------------------------------------------------------------
# Matching everything onto each frame's stack grid
# ---------------------------------------------------------------------------
def stack_grid(paths):
    """(crs, transform, width, height) of this frame's rebased stack.

    The stack is the authoritative grid: it is what the displacement lives on
    and what the correction is subtracted from. It does not match the
    stitched geometry grid, so everything that has to combine with
    displacement is resampled onto it here, once. xarray x/y are pixel
    centres; a rasterio transform takes the corner.
    """
    import xarray as xr
    from rasterio.transform import from_origin
    with xr.open_dataset(paths['stack']) as ds:
        x, y = ds.x.values, ds.y.values
        wkt = ds.spatial_ref.attrs.get("spatial_ref") or ds.spatial_ref.attrs.get("crs_wkt")
    res = abs(float(x[1] - x[0]))
    tr = from_origin(float(x.min()) - res / 2, float(y.max()) + res / 2, res, res)
    return wkt, tr, len(x), len(y)


def match_to_grid(src_path, grid, out_path, resampling="bilinear"):
    """Resample a raster onto `grid` = (crs_wkt, transform, width, height)."""
    from rasterio.warp import reproject, Resampling
    crs, tr, w, h = grid
    rs = getattr(Resampling, resampling)
    with rasterio.open(src_path) as src:
        prof = src.profile.copy()
        prof.update(crs=crs, transform=tr, width=w, height=h,
                    driver="GTiff", compress="deflate")
        with rasterio.open(out_path, "w", **prof) as dst:
            for b in range(1, src.count + 1):
                reproject(source=rasterio.band(src, b), destination=rasterio.band(dst, b),
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=tr, dst_crs=crs, resampling=rs)
    return out_path


def build_los_enu(geo_dir):
    """3-band los_enu.tif (east, north, up) on the stack grid.

    This is the file run04's converter wants: `disp_nc_to_mintpy` globs the
    geometry dir for `*los_enu.tif` and converts it to MintPy incidence and
    azimuth itself. OPERA stores only east and north; up comes from the
    unit-vector identity up = sqrt(1 - e^2 - n^2).
    """
    e_p = os.path.join(geo_dir, "los_east.tif")
    n_p = os.path.join(geo_dir, "los_north.tif")
    out = os.path.join(geo_dir, "los_enu.tif")
    with rasterio.open(e_p) as de, rasterio.open(n_p) as dn:
        e = de.read(1).astype("float32")
        n = dn.read(1).astype("float32")
        prof = de.profile
    bad = (e == 0) & (n == 0)
    up = np.sqrt(np.clip(1.0 - e.astype("f8")**2 - n.astype("f8")**2, 0.0, 1.0)).astype("float32")
    e[bad] = np.nan
    n[bad] = np.nan
    up[bad] = np.nan
    prof.update(count=3, dtype="float32", nodata=np.nan, compress="deflate")
    with rasterio.open(out, "w", **prof) as ds:
        for i, band in enumerate([e, n, up], start=1):
            ds.write(band, i)
        ds.descriptions = ("los_east", "los_north", "los_up")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    from shapely.geometry import box

    # 1) Per-frame geometry from the CSLC static layers
    for frame, paths in frames.items():
        print(f"\n=== frame {frame} ===")
        bbox = frame_bbox[frame]
        build_geometry(frame, box(*bbox), bbox, paths['dir'])

    # 2) One DEM for the whole AOI
    master_dem = os.path.join(data_dir, "dem_glo30.tif")
    print("\n=== DEM (master, fetched once for the whole AOI) ===")
    build_dem(common_paths['aoi_bbox'], master_dem)

    # 3) Per frame: put DEM + geometry on that frame's stack grid.
    #    The names matter: run04's converter globs geometry/ for
    #    `*dem.tif`, `*los_enu.tif`, `*layover_shadow_mask.tif`, so every file
    #    at that level must be on the stack grid; stitched-grid intermediates
    #    stay one level down in geometry/stitched/.
    print("\n=== matching DEM + geometry to each frame's stack grid ===")
    for frame, paths in frames.items():
        grid = stack_grid(paths)
        gdir = os.path.join(paths['dir'], "geometry")
        sdir = os.path.join(gdir, "stitched")
        # DEM: genuine reprojection (4326 -> stack UTM grid) -> bilinear.
        match_to_grid(master_dem, grid, os.path.join(gdir, "dem.tif"), "bilinear")
        # Geometry: stitched on the same UTM zone and offset by an exact integer
        # number of pixels, so this is a lossless crop/pad -> nearest keeps the
        # values untouched.
        for lyr in STITCHED_LAYERS:
            src = os.path.join(sdir, f"{lyr}.tif")
            if os.path.exists(src):
                match_to_grid(src, grid, os.path.join(gdir, f"{lyr}.tif"), "nearest")
        build_los_enu(gdir)
        print(f"  {frame}: geometry/ holds stack-grid layers; intermediates in stitched/")

    print("\ndone.")
