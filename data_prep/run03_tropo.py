#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Danielle Lindsay

OPERA TROPO zenith-delay correction, per frame:

1) Read the acquisition datetimes from the frame's rebased stack
2) Query CMR once for the whole range; keep the 6-hourly granules bracketing
   each acquisition (TROPO is at 00/06/12/18 UTC; S1 over the Bay Area is
   ~02:07 UTC ascending, ~14:07 UTC descending)
3) `crop_tropo`: subset each granule to the AOI over HTTPS; each ~2 GB granule
   is subset lazily before it is loaded
4) `apply_tropo`: interpolate the delay cube onto the frame's DEM surface and
   project into line of sight with the per-pixel incidence angle

`opera_utils.tropo` computes the correction but takes a file of URLs and does
not query CMR; this script builds the granule list.

The DEM must be ellipsoidal and on the frame's stack grid (run02): the delay
cube's height axis is "height above ellipsoid WGS84".

`subtract_first_date=True` (the tool default) is kept: the displacement stack
is rebased to its first reference date, so the correction is referenced the
same way; any residual constant is absorbed by the GNSS tie (run05).

Output per frame:  <frame>/tropo/cropped/      cropped+time-interpolated TROPO
                   <frame>/tropo/corrections/  per-date LOS correction GeoTIFFs

Run in the conda environment. Earthdata credentials in ~/.netrc.
Source: OPERA L4 TROPO-ZENITH v1 (ASF, found via NASA CMR); opera_utils.tropo
(crop_tropo, apply_tropo).
"""
import glob
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import os as _os, sys as _sys        # path shim: import the repo root modules
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from TulePond_filepaths import common_paths, frames
import insar_utils as utils

# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------
NUM_WORKERS = 8
HEIGHT_MAX  = 10000.0     # m; keep the full column, which the zenith integration needs

CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
TROPO_COLLECTION = "C3717139408-ASF"      # OPERA_L4_TROPO-ZENITH_V1
TROPO_STEP_HOURS = 6                      # granules at 00/06/12/18 UTC
TROPO_START = datetime(2016, 1, 1, tzinfo=timezone.utc)   # start of the record

# Per-frame AOI, same as run01/run02 (09156 clipped to its own footprint)
frame_bbox = {
    "09156": common_paths['aoi_bbox_09156'],
    "09157": common_paths['aoi_bbox'],
    "11115": common_paths['aoi_bbox'],
}


# ---------------------------------------------------------------------------
# Granule discovery (the part opera_utils does not do)
# ---------------------------------------------------------------------------
def stack_datetimes(paths):
    """Acquisition datetimes (UTC) from the frame's rebased stack."""
    import pandas as pd
    import xarray as xr
    with xr.open_dataset(paths['stack']) as ds:
        t = pd.to_datetime(ds.time.values)
    return [d.to_pydatetime().replace(tzinfo=timezone.utc) for d in t]


def bracketing_slots(dts):
    """The set of 6-hourly TROPO timestamps bracketing each acquisition, so
    `crop_tropo` can interpolate in time rather than snap to the nearest."""
    slots = set()
    step = timedelta(hours=TROPO_STEP_HOURS)
    for d in dts:
        floor = d.replace(minute=0, second=0, microsecond=0)
        floor -= timedelta(hours=floor.hour % TROPO_STEP_HOURS)
        slots.add(floor)
        slots.add(floor + step)
    return slots


def cmr_granules(t0, t1):
    """All TROPO granules in [t0, t1], as {start_datetime: data_url}.
    Pages with CMR-Search-After; one query for the whole range."""
    out = {}
    params = {
        "collection_concept_id": TROPO_COLLECTION,
        "temporal": f"{t0:%Y-%m-%dT%H:%M:%SZ},{t1:%Y-%m-%dT%H:%M:%SZ}",
        "page_size": "2000",
        "sort_key": "start_date",
    }
    after = None
    while True:
        req = urllib.request.Request(CMR + "?" + urllib.parse.urlencode(params))
        if after:
            req.add_header("CMR-Search-After", after)
        with urllib.request.urlopen(req, timeout=120) as r:
            after = r.headers.get("CMR-Search-After")
            items = json.load(r)["items"]
        for it in items:
            u = it["umm"]
            start = u["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
            urls = [x["URL"] for x in u["RelatedUrls"]
                    if x.get("Type") == "GET DATA" and x["URL"].endswith(".nc")]
            if urls:
                out[datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc)] = urls[0]
        if not items or not after:
            break
    return out


def write_url_list(dts, out_path):
    """Resolve the granules bracketing `dts`; write them one URL per line."""
    slots = bracketing_slots(dts)
    early = [s for s in slots if s < TROPO_START]
    if early:
        print(f"    !! {len(early)} acquisitions predate the TROPO record "
              f"({TROPO_START:%Y-%m-%d}); those dates will be uncorrected")
    lo, hi = min(slots), max(slots)
    found = cmr_granules(lo, hi)
    have = sorted(slots & set(found))
    missing = sorted(slots - set(found))
    if missing:
        print(f"    !! {len(missing)} bracketing granules not found in CMR, "
              f"first: {missing[0]:%Y-%m-%d %H:%M}")
    with open(out_path, "w") as f:
        for s in have:
            f.write(found[s] + "\n")
    print(f"    {len(have)} granules needed (of {len(found)} in range)")
    return out_path, len(have)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    from opera_utils.tropo import apply_tropo, crop_tropo

    for frame, paths in frames.items():
        print(f"\n=== frame {frame} ===")
        bbox = frame_bbox[frame]
        dts = stack_datetimes(paths)
        gdir = os.path.join(paths['dir'], "geometry")
        dem = os.path.join(gdir, "dem.tif")
        inc = os.path.join(gdir, "incidence_angle.tif")

        tdir = os.path.join(paths['dir'], "tropo")
        crop_dir = os.path.join(tdir, "cropped")
        corr_dir = os.path.join(tdir, "corrections")
        os.makedirs(tdir, exist_ok=True)
        print(f"  {len(dts)} acquisitions: {min(dts):%Y-%m-%d} .. {max(dts):%Y-%m-%d} "
              f"(~{min(dts):%H:%M} UTC)")

        # 1) Granule URLs bracketing every acquisition (one CMR query)
        urls_file, n = write_url_list(dts, os.path.join(tdir, "tropo_urls.txt"))
        if not n:
            print("    no granules -- skipping frame")
            continue

        # 2) Crop to the AOI (lazy HTTPS subset; ~40 MB per date is a generous
        #    estimate: the disk check catches a full volume, not the last MB)
        utils.require_disk_space(tdir, 0.040 * len(dts), f"{frame} tropo ({len(dts)} dates)")
        print(f"    cropping to {bbox} (height_max {HEIGHT_MAX} m, {NUM_WORKERS} workers)")
        crop_tropo(tropo_urls_file=Path(urls_file), datetimes=dts,
                   aoi_bounds=tuple(bbox), output_dir=Path(crop_dir),
                   height_max=HEIGHT_MAX, num_workers=NUM_WORKERS)

        # 3) Project onto the DEM surface and into line of sight
        cropped = sorted(Path(crop_dir).glob("*.nc"))
        print(f"    {len(cropped)} cropped TROPO files; applying with dem.tif + "
              f"incidence_angle.tif -> {corr_dir}")
        apply_tropo(cropped_tropo_list=list(cropped), dem_path=Path(dem),
                    incidence_angle_path=Path(inc), output_dir=Path(corr_dir),
                    num_workers=NUM_WORKERS)
        n_out = len(glob.glob(os.path.join(corr_dir, "*.tif")))
        print(f"    wrote {n_out} correction GeoTIFFs")

    print("\ndone.")
