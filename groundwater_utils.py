#!/usr/bin/env python3
"""
groundwater_utils.py: the groundwater head beside the Tule Pond section (notebook 16), from the public ACWD
Groundwater Monitoring Reports only.

Author: Danielle Lindsay

Source: Alameda County Water District, annual Groundwater Monitoring Report (GWMR), Appendices "Spring <year>
Groundwater Monitoring Records" and "Fall <year> Groundwater Monitoring Records": one row per well and campaign
(State Well Number, aquifer code, owner, measurement date, reference-point elevation, water elevation in ft, remarks).
The PDFs are kept unchanged in external_inputs/ACWD_GWMR_archive (sources in its README, SHA256SUMS.txt checked here).
The ACWD-supplied indicator-well series (external_inputs/ACWD_water_well_data) are not public and are never read.

Two parts:
    Reading the reports (plumbing)   pdftotext of the appendix pages, the table parser, the archive reader
    Science kernel                   well position from its State Well Number, distance to our mapped trace,
                                     side of the fault, winter rise, pore pressure, strength change, the selection
"""
import glob
import hashlib
import os
import re
import subprocess

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------------------------------------------------
# Reading the reports (plumbing)
# ---------------------------------------------------------------------------------------------------------------------
# The table layouts of 2015-2025, read with pdftotext -layout:
# - Unicode hyphens/minus (U+2010, U+2011, U+2212, used in 2018) are normalised to '-';
# - '- -' (2015-2017) and '--' mark an empty cell;
# - some rows have no aquifer code (kept with aquifer = None) or no date (well not measured).
# A row's water level is the second number after its measurement date (the first is the reference-point elevation);
# 'A' after the reference elevation means artesian (no number). Levels on rows flagged UTM (unable to measure depth to
# water) are dropped; rows flagged RUN (pump running) keep their level and are flagged in `remarks`.

WELL = re.compile(r"^\dS/\dW-\d{2}[A-Z]\d{3}$")
DATE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")
NUM = re.compile(r"^-?\d+(\.\d+)?$")
BAD_LEVEL = {"UTM"}          # report remark code (Appendix abbreviations): unable to measure depth to water
AQUIFERS = {"AHF", "N", "C", "F", "D", "CF", "FD", "CFD", "CD", "NC", "ND", "NCF", "NF", "DF"}
TABLE_COLUMNS = ["well", "aquifer", "date", "reference_elevation_ft", "water_elevation_ft", "remarks"]

# The report used for each water year. The 2017 report was posted twice; the later posting (27 Apr 2018 capture) is
# used, the earlier one (23 Mar 2018) is kept in the archive but not read. 2016 and 2024 were not found.
REPORT_FILES = {2015: "2015 GWMR.pdf", 2017: "2017 GWMR.pdf", 2018: "2018 GWMR.pdf", 2019: "2019 GWMR.pdf",
                2020: "2020 GWMR.pdf", 2021: "2021 GWMR.pdf", 2022: "2022 GWMR.pdf", 2023: "2023 GWMR.pdf",
                2025: "2025 GWMR.pdf"}
NOT_READ = {"2017 GWMR (earlier posting 2018-03-23).pdf"}


def pdf_text(pdf, first, last):
    """Text of pages first..last of a PDF (pdftotext -layout), hyphens normalised, empty cells as '--'."""
    t = subprocess.run(["pdftotext", "-layout", "-f", str(first), "-l", str(last), pdf, "-"],
                       check=True, capture_output=True, text=True).stdout
    t = t.replace("‐", "-").replace("‑", "-").replace("−", "-")
    return re.sub(r"(?<=\s)-\s-(?=\s|$)", "--", t, flags=re.M)


def n_pages(pdf):
    out = subprocess.run(["pdfinfo", pdf], check=True, capture_output=True, text=True).stdout
    return int(re.search(r"Pages:\s+(\d+)", out).group(1))


def appendix_pages(pdf):
    """(spring first, spring last, fall first, fall last) table pages, found from the two appendix title pages."""
    n = n_pages(pdf)
    spring = fall = None
    for p in range(n // 2, n + 1):
        t = pdf_text(pdf, p, p)
        if re.search(r"SPRING \d{4} GROUNDWATER MONITORING RECORDS", t):
            spring = p
        if re.search(r"FALL \d{4} GROUNDWATER MONITORING RECORDS", t):
            fall = p
    assert spring and fall and spring < fall, f"{pdf}: appendix title pages not found (spring {spring}, fall {fall})"
    return spring + 1, fall - 1, fall + 1, n - 1


def parse_table(text):
    """The well rows of one appendix (text of its pages) as a frame with TABLE_COLUMNS."""
    rows = []
    for line in text.splitlines():
        toks = line.split()
        if not toks or not WELL.match(toks[0]):
            continue
        dates = [i for i, t in enumerate(toks) if DATE.match(t)]
        i = dates[0] if dates else len(toks)
        codes = [j for j, t in enumerate(toks[1:i], start=1) if t in AQUIFERS]
        aquifer = toks[codes[-1]] if codes else None
        water = None
        ref = None
        if dates and (NUM.match(toks[i - 1]) or toks[i - 1] == "--"):
            # the first date comes after a number: it is the water-sample date, no level was measured (the numbers after
            # it are chloride and TDS), e.g. "4S/1W-27P002 AHF CITY OF FREMONT 52.65 4/9/2021 129 950"
            dates = []
        remarks = {t.upper() for t in toks[i + 1:] if re.match(r"^[A-Za-z]{1,4}$", t)}
        if dates:
            after = toks[i + 1:i + 3]
            if len(after) == 2 and NUM.match(after[0]):
                ref = float(after[0])
                water = float(after[1]) if NUM.match(after[1]) else ("artesian" if after[1] == "A" else None)
        if sorted(remarks & BAD_LEVEL) and isinstance(water, float):
            water = None                                 # UTM: no level was measured, so the number is not a water level
        rows.append(dict(well=toks[0], aquifer=aquifer, date=toks[i] if dates else None, reference_elevation_ft=ref,
                         water_elevation_ft=water, remarks=" ".join(sorted(remarks))))
    df = pd.DataFrame(rows, columns=TABLE_COLUMNS, dtype=object)          # object: a missing level stays None, not NaN
    df["reference_elevation_ft"] = pd.to_numeric(df.reference_elevation_ft)
    num = df.water_elevation_ft.map(is_level)
    # a level more than 2 ft above the reference point is not a level (a flowing well is recorded as "A"): it is the
    # chloride/TDS pair of a row with no measurement date and no reference elevation (2021-2023, e.g. 4S/1W-27G002)
    above = num & (df.water_elevation_ft.where(num).astype(float) > df.reference_elevation_ft.astype(float) + 2.0)
    df.loc[above, "water_elevation_ft"] = None
    df.attrs["dropped_above_reference"] = df.loc[above, "well"].tolist()
    assert len(df), "no well rows"
    return df


def is_level(v):
    """True for a measured water level (a number); False for None, NaN, 'artesian'."""
    return isinstance(v, float) and not np.isnan(v)


def read_report(pdf):
    """{'spring': table, 'fall': table} of one report, and the appendix pages."""
    s0, s1, f0, f1 = appendix_pages(pdf)
    return {"spring": parse_table(pdf_text(pdf, s0, s1)), "fall": parse_table(pdf_text(pdf, f0, f1))}, (s0, s1, f0, f1)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def report_files(archive_dir):
    """{water year: path} of the reports read, each checked against SHA256SUMS.txt. Fails if a PDF in the archive is
    neither read nor listed in NOT_READ, if a listed file is missing, or if a checksum differs."""
    sums = {}
    with open(os.path.join(archive_dir, "SHA256SUMS.txt")) as f:
        for line in f:
            if line.strip():
                digest, name = line.rstrip("\n").split("  ", 1)
                sums[name] = digest
    on_disk = {os.path.basename(p) for p in glob.glob(os.path.join(archive_dir, "*.pdf"))}
    unknown = on_disk - set(REPORT_FILES.values()) - NOT_READ
    assert not unknown, f"PDFs in {archive_dir} not assigned to a water year: {sorted(unknown)}"
    assert len(set(REPORT_FILES.values())) == len(REPORT_FILES), "two water years point to the same report"
    files = {}
    for wy, name in sorted(REPORT_FILES.items()):
        path = os.path.join(archive_dir, name)
        assert os.path.exists(path), f"report missing: {path}"
        assert name in sums, f"{name} has no line in SHA256SUMS.txt"
        assert sha256(path) == sums[name], f"{name}: SHA256 differs from SHA256SUMS.txt"
        assert str(wy) in name, f"{name} is not the report of {wy}"
        files[wy] = path
    return files


# ---------------------------------------------------------------------------------------------------------------------
# Science kernel
# ---------------------------------------------------------------------------------------------------------------------
FT_M = 0.3048                    # ft -> m
KPA_PER_M = 9.80665              # 1 m of fresh water: rho_w g = 1000 kg/m3 x 9.80665 m/s2 = 9.80665 kPa
MAX_RISE_M = 15.0                # no aquifer here swings by more than 15 m in a winter: a larger value is a misread row

# Position of a well from its State Well Number, e.g. 4S/1W-27D008: township 4 South, range 1 West, section 27,
# 40-acre tract D, well 008 (Public Land Survey grid, Mount Diablo Meridian). The well is placed at the centre of its
# tract on an ideal grid of 1-mile sections; Fremont's sections are projected, so a position is good to about +/-200 m.
MILE_M = 1609.344
PLSS_ORIGIN = (37 + 52 / 60 + 54 / 3600, -(121 + 54 / 60 + 47 / 3600))   # Mount Diablo initial point (lat, lon)
M_PER_DEG_LAT = 111132.0
M_PER_DEG_LON = 111320.0 * np.cos(np.radians(37.55))                     # at the latitude of Tule Pond
TRACT_ROWS = ["DCBA", "EFGH", "MLKJ", "NPQR"]                            # 40-acre tracts in a section, from the north


def tract_centre(well):
    """(lat, lon) of the centre of the 40-acre tract of a State Well Number (south townships, west ranges), or None
    if the number is not of the form <t>S/<r>W-<section><tract><nnn>.
    Sections are numbered boustrophedon from the north-east corner of the township: 1-6 east to west, 7-12 west
    to east, and so on."""
    m = re.match(r"(\d)S/(\d)W-(\d{2})([A-R])", well)
    if not m:
        return None
    township, rng, section, tract = int(m[1]), int(m[2]), int(m[3]), m[4]
    row = (section - 1) // 6
    col = 5 - (section - 1) % 6 if row % 2 == 0 else (section - 1) % 6      # column from the west edge
    x = -rng * 6 * MILE_M + col * MILE_M                                       # west edge of the section (m east of origin)
    y = -(township - 1) * 6 * MILE_M - row * MILE_M                            # north edge of the section (m north)
    tract_row = [i for i, r in enumerate(TRACT_ROWS) if tract in r][0]
    tract_col = TRACT_ROWS[tract_row].index(tract)
    lat = PLSS_ORIGIN[0] + (y - (tract_row + 0.5) * MILE_M / 4) / M_PER_DEG_LAT
    lon = PLSS_ORIGIN[1] + (x + (tract_col + 0.5) * MILE_M / 4) / M_PER_DEG_LON
    return lat, lon


def local_xy(lat, lon):
    """Metres east and north of the PLSS origin (flat Earth, adequate over a few km)."""
    return (np.asarray(lon) - PLSS_ORIGIN[1]) * M_PER_DEG_LON, (np.asarray(lat) - PLSS_ORIGIN[0]) * M_PER_DEG_LAT


def nearest_on_trace(lat, lon, trace_lat, trace_lon, trace_km):
    """(distance from the point to the nearest trace vertex in km, along-fault km of that vertex)."""
    tx, ty = local_xy(trace_lat, trace_lon)
    x, y = local_xy(lat, lon)
    d = np.hypot(tx - x, ty - y)
    i = int(np.argmin(d))
    return float(d[i] / 1e3), float(np.asarray(trace_km)[i])


def side_of_fault(aquifer):
    """ACWD's aquifer codes: AHF (Above Hayward Fault) lies east of the fault; all other codes (Newark N, Centerville C,
    Fremont F, Deep D and their combinations) lie west of it."""
    return "east" if aquifer == "AHF" else "west"


def winter_rise_m(fall_level_ft, spring_level_ft):
    """Rise in head from the fall reading of one year to the spring reading of the next (m)."""
    return (spring_level_ft - fall_level_ft) * FT_M


def pressure_kpa(head_m):
    """Pore-pressure change of a head change: dp = rho_w g dh."""
    return head_m * KPA_PER_M


def strength_change_kpa(dp_kpa, friction):
    """Change in fault strength for a pore-pressure change dp at effective friction mu': d tau = mu' dp."""
    return friction * dp_kpa


def static_levels(table):
    """Well -> water level (ft) of one appendix: measured levels only (no UTM, no artesian, no blank), and not
    pump-running (RUN). One reading per well is asserted."""
    assert not table.well.duplicated().any(), "a well appears twice in one campaign"
    ok = table.water_elevation_ft.map(is_level) & ~table.remarks.fillna("").str.contains("RUN")
    return table[ok].set_index("well").water_elevation_ft.astype(float)


def aquifer_codes(tables):
    """Well -> aquifer code, from the first table in which the well has one (a few rows have no code)."""
    codes = {}
    for wy in sorted(tables):
        for campaign in ("spring", "fall"):
            for well, code in zip(tables[wy][campaign].well, tables[wy][campaign].aquifer):
                if isinstance(code, str) and code.strip():
                    codes.setdefault(well, code.strip())
    return codes


def all_winter_rises(tables, trace_lat, trace_lon, trace_km, sse_water_years):
    """Winter rise of every well with a tract position and an aquifer code, for every winter with consecutive reports
    (fall of Y-1 and spring of Y). Rises larger than MAX_RISE_M are dropped as misread rows."""
    codes = aquifer_codes(tables)
    rows = []
    for wy in sorted(tables):
        if wy - 1 not in tables:
            continue
        both = pd.concat([static_levels(tables[wy - 1]["fall"]), static_levels(tables[wy]["spring"])],
                         axis=1, keys=["fall", "spring"], join="inner")
        for well, r in both.iterrows():
            pos = tract_centre(well)
            if pos is None or well not in codes:
                continue
            rise = winter_rise_m(r.fall, r.spring)
            if abs(rise) > MAX_RISE_M:
                continue
            dist_km, km = nearest_on_trace(pos[0], pos[1], trace_lat, trace_lon, trace_km)
            rows.append(dict(winter=f"{wy - 1}/{str(wy)[2:]}", water_year=wy, well=well, aquifer=codes[well],
                             side=side_of_fault(codes[well]), lat=pos[0], lon=pos[1], dist_km=dist_km, km=km,
                             rise_m=rise, rise_kpa=pressure_kpa(rise), sse=wy in sse_water_years))
    return pd.DataFrame(rows)


def select_wells(rises, max_dist_km, along_km):
    """The wells within max_dist_km of the trace whose nearest trace point lies in along_km = (lo, hi)."""
    keep = (rises.dist_km <= max_dist_km) & (rises.km >= along_km[0]) & (rises.km <= along_km[1])
    return rises[keep].copy()


def winter_medians_kpa(rises):
    """Median rise (kPa) per water year and side."""
    return rises.groupby(["water_year", "side"]).rise_kpa.median().unstack()
