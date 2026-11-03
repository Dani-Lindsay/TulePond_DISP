#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Danielle Lindsay

Daily precipitation at the Tule Pond section from NOAA GHCN-daily rain gauges, for the comparison of creep-event
timing with rainfall.

No single gauge covers the whole 1979-2026 event record next to the section
(37.56 N, -121.97 E):
  FREMONT (USC00043244, 4 km W)       1996-06 - present, closest, >=95% complete
  NEWARK (USC00046144, 7 km SW)       1906 - 2021-01-15, 97% complete
  HAYWARD AIR TERMINAL (USW00093228)  1998 - present, 17 km NW, 100% complete
  LIVERMORE (USC00044997)             1903 - present, 24 km NE (Livermore Valley)
  SAN JOSE (USW00023293)              1998 - present, 22 km S
  OAKLAND MUSEUM (USC00046336)        1970 - present, 37 km NW, gaps in 2017/18
The composite series takes, day by day, the first gauge that reports in the
order Fremont > Newark > Hayward Air Terminal: Fremont covers the InSAR era
and back to WY1997 with no splice, Newark covers the alignment-array era
(1979-1996), and Hayward fills the remaining gaps. The stats file lists the
gauge-to-gauge water-year ratios over the overlaps so the splice is
documented; daily values are not rescaled. The other stations are carried as
columns for sensitivity tests.

No gauge in the GHCN inventory on the Mission Peak side of the section has a
usable record: CALAVERAS RSVR (USC00041281) ran 1959-1977 only, and the
CoCoRaHS gauge FREMONT 3.7 NE (US1CAAL0024, 2014-) reports on few days per
year, so it is not used.

Observation-time offset: the COOP gauges (USC ids: Newark, Livermore,
Oakland Museum) are read by an observer in the morning and GHCN dates the
24-h total by the day the reading is made, whereas the airport ASOS gauges
(USW ids) are midnight-to-midnight. The lag was set from the correlation of
storm-day totals with Hayward Air Terminal at lags 0 and -1 d: Newark,
Livermore and Oakland Museum are shifted back one day to put every gauge on
the same calendar basis; the Fremont COOP gauge correlates best at zero lag
and is left unshifted. A residual half-day ambiguity remains for the COOP
gauges, so storms are identified over multi-day windows downstream, not
single days.

Water year: Oct 1 - Sep 30, named by the calendar year in which it ends
(WY2017 = Oct 2016 - Sep 2017); day-of-water-year 1 = Oct 1.

Downloads (once) from
  https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/{ID}.csv
into common_paths['ghcn_dir']; re-run with --refresh to update to the present.
GHCN PRCP is stored in tenths of a millimetre; converted to mm here.

Output (common_paths['precip_dir']):
         tulepond_daily_precip.csv   (date, decyear, wy, dowy, prcp_mm, src, per-station mm)
         tulepond_wateryear_totals.csv
         precip_station_stats.txt
Source: NOAA NCEI GHCN-daily (Menne et al. 2012); written for this project.
"""
import os
import sys
import urllib.request
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import os as _os, sys as _sys        # path shim: import the repo root modules
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from TulePond_filepaths import common_paths

NCEI = ("https://www.ncei.noaa.gov/data/"
        "global-historical-climatology-network-daily/access/{sid}.csv")
STATIONS = {                       # id: (short name, role, day shift)
    "USC00043244": ("FREMONT", "composite-1", 0),  # COOP but midnight basis (checked)
    "USC00046144": ("NEWARK", "composite-2", -1),  # COOP, morning observer
    "USW00093228": ("HAYWARD_AT", "composite-3", 0),  # ASOS, midnight basis
    "USC00044997": ("LIVERMORE", "alt", -1),       # COOP
    "USW00023293": ("SAN_JOSE", "alt", 0),         # ASOS
    "USC00046336": ("OAKLAND_MUS", "alt", -1),     # COOP
}
COMPOSITE = ["FREMONT", "NEWARK", "HAYWARD_AT"]    # priority order
TULE_POND = (37.557, -121.970)       # Tule Pond section centre (lat, lon)
T0 = "1978-10-01"                    # start of WY1979: first HRKT survey 1979.7


def decyear(dates):
    """Calendar decimal year (Jan 1 00:00 = year.000, fraction of the actual year length), for this script's
    own tables. Not the notebooks' convention (insar_utils.decimal_years: 365.25-day years from 2000)."""
    d = pd.DatetimeIndex(dates)
    y0 = pd.to_datetime(d.year.astype(str) + "-01-01")
    y1 = pd.to_datetime((d.year + 1).astype(str) + "-01-01")
    return d.year + (d - y0) / (y1 - y0)


def decyear_to_date(y):
    """Inverse of decyear() for scalars or arrays."""
    y = np.atleast_1d(np.asarray(y, float))
    out = []
    for v in y:
        yr = int(np.floor(v))
        y0, y1 = datetime(yr, 1, 1), datetime(yr + 1, 1, 1)
        out.append(y0 + (v - yr) * (y1 - y0))
    return pd.DatetimeIndex(out)


def water_year(dates):
    d = pd.DatetimeIndex(dates)
    return d.year + (d.month >= 10).astype(int)


def dowy(dates):
    """Day of water year: Oct 1 = 1."""
    d = pd.DatetimeIndex(dates)
    wy0 = pd.to_datetime((water_year(d) - 1).astype(str) + "-10-01")
    return (d - wy0).days + 1


def km(lat, lon):
    return np.hypot((lat - TULE_POND[0]) * 111.19,
                    (lon - TULE_POND[1]) * 111.19 * np.cos(np.radians(TULE_POND[0])))


def fetch(sid, refresh=False):
    os.makedirs(common_paths["ghcn_dir"], exist_ok=True)
    path = os.path.join(common_paths["ghcn_dir"], f"{sid}.csv")
    if refresh or not os.path.exists(path) or os.path.getsize(path) < 1000:
        print("downloading", sid)
        urllib.request.urlretrieve(NCEI.format(sid=sid), path)
    d = pd.read_csv(path, usecols=["DATE", "LATITUDE", "LONGITUDE", "NAME",
                                   "PRCP"], low_memory=False)
    d["DATE"] = pd.to_datetime(d.DATE)
    d = d[d.DATE >= T0]
    meta = dict(name=d.NAME.iloc[0], lat=float(d.LATITUDE.iloc[0]),
                lon=float(d.LONGITUDE.iloc[0]), first=d.DATE.min().date(),
                last=d.DATE.max().date())
    s = d.set_index("DATE").PRCP / 10.0          # tenths of mm -> mm
    s = s[~s.index.duplicated()]
    shift = STATIONS[sid][2]
    if shift:                                    # COOP morning-obs -> previous day
        s.index = s.index + pd.Timedelta(days=shift)
    return s, meta


def main(refresh=False):
    grid = pd.date_range(T0, datetime.today().date(), freq="D")
    tab = pd.DataFrame(index=grid)
    metas = {}
    for sid, (nm, role, shift) in STATIONS.items():
        s, meta = fetch(sid, refresh)
        tab[nm] = s.reindex(grid)
        metas[nm] = dict(sid=sid, role=role, shift=shift, **meta,
                         dist_km=km(meta["lat"], meta["lon"]),
                         n_valid=int(s.notna().sum()))
    # composite: first reporting gauge in priority order
    tab["prcp_mm"] = np.nan
    tab["src"] = "none"
    for nm in COMPOSITE:
        take = tab.prcp_mm.isna() & tab[nm].notna()
        tab.loc[take, "prcp_mm"] = tab.loc[take, nm]
        tab.loc[take, "src"] = nm
    tab["decyear"] = decyear(tab.index)
    tab["wy"] = water_year(tab.index)
    tab["dowy"] = dowy(tab.index)
    tab.index.name = "date"

    # water-year totals, composite and per station, with completeness
    g = tab.groupby("wy")
    wy = pd.DataFrame({"total_mm": g.prcp_mm.sum(min_count=1),
                       "n_days": g.prcp_mm.count(),
                       "n_missing": g.prcp_mm.apply(lambda x: x.isna().sum()),
                       "max_day_mm": g.prcp_mm.max(),
                       "src": g.src.agg(lambda x: "+".join(sorted(set(x) - {"none"})))})
    for nm in STATIONS.values():
        wy[f"{nm[0]}_mm"] = g[nm[0]].sum(min_count=1)
        wy[f"{nm[0]}_ndays"] = g[nm[0]].count()
    wy["complete"] = wy.n_missing <= 20

    # splice documentation: gauge pairs over complete overlapping WYs, plus
    # same-day agreement on storm days
    splice = []
    for a, b in [("FREMONT", "NEWARK"), ("FREMONT", "HAYWARD_AT"),
                 ("NEWARK", "HAYWARD_AT")]:
        both = wy[(wy[f"{a}_ndays"] >= 350) & (wy[f"{b}_ndays"] >= 350)]
        ratio = both[f"{a}_mm"] / both[f"{b}_mm"]
        ov = tab[[a, b]].dropna()
        ov = ov[ov.index.isin(tab.index[tab.wy.isin(both.index)])]
        big = ov[(ov[a] >= 10) | (ov[b] >= 10)]
        r_day = np.corrcoef(big[a], big[b])[0, 1] if len(big) > 10 else np.nan
        splice.append(f"  {a}/{b}: {len(both)} complete WYs "
                      f"({both.index.min()}-{both.index.max()}), WY-total ratio "
                      f"median {ratio.median():.2f} (range {ratio.min():.2f}-"
                      f"{ratio.max():.2f}); same-day totals on days >=10 mm: "
                      f"n={len(big)}, r={r_day:.2f}")

    os.makedirs(common_paths["precip_dir"], exist_ok=True)
    tab.to_csv(os.path.join(common_paths["precip_dir"],
                            "tulepond_daily_precip.csv"),
               float_format="%.4f")
    wy.to_csv(os.path.join(common_paths["precip_dir"],
                           "tulepond_wateryear_totals.csv"),
              float_format="%.1f")

    lines = ["=" * 78, "TULE POND DAILY PRECIPITATION, NOAA GHCN-daily  (run06)",
             "=" * 78,
             f"segment centre {TULE_POND[0]:.3f} N {TULE_POND[1]:.3f} E; series "
             f"{grid[0].date()} to {grid[-1].date()}; water year Oct 1-Sep 30, "
             "named by the year it ends", "",
             "stations (distance from the section centre, record used, valid days):"]
    for nm, m in metas.items():
        lines.append(f"  {nm:<12} {m['sid']}  {m['name']:<38} "
                     f"{m['dist_km']:5.1f} km  {m['first']}..{m['last']}  "
                     f"n={m['n_valid']}  [{m['role']}, day shift {m['shift']:+d}]")
    n_src = tab.src.value_counts()
    lines += ["",
              "composite = first reporting gauge in the order "
              + " > ".join(COMPOSITE) + ": "
              + ", ".join(f"{k} {v} d" for k, v in n_src.items()),
              f"  FREMONT reports from {metas['FREMONT']['first']}; NEWARK "
              f"carries 1979-1996 (last report {metas['NEWARK']['last']})",
              "  gauge overlaps (the splice is documented, NOT rescaled):"] + splice + [
              "  morning-observer COOP gauges (Newark, Livermore, Oakland Mus.) "
              "shifted -1 d to the midnight basis; Fremont already on it", "",
              "water-year totals, composite (mm; * = >20 missing days):"]
    for r in wy.itertuples():
        flag = "" if r.complete else "*"
        lines.append(f"  WY{r.Index}  {r.total_mm:6.1f}{flag:1}  "
                     f"max day {r.max_day_mm:5.1f}  [{r.src}]")
    comp = wy[wy.complete & (wy.index >= 1980) & (wy.index <= 2025)]
    lines += ["",
              f"WY1980-2025 complete years (n={len(comp)}): median "
              f"{comp.total_mm.median():.0f} mm, 10-90% "
              f"{comp.total_mm.quantile(.1):.0f}-{comp.total_mm.quantile(.9):.0f} mm"]
    txt = "\n".join(lines)
    with open(os.path.join(common_paths["precip_dir"],
                           "precip_station_stats.txt"), "w") as fh:
        fh.write(txt + "\n")
    print(txt)
    return tab, wy


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
