#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Danielle Lindsay

GNSS processing (UNR daily solutions -> screened ENU velocities):
1) Download the UNR holdings and steps metadata; filter the station list by ROI and date window
2) Download daily NA-frame time series per station
3) For each station:
   a) Data length check
   b) Manual exclusion -> plot to Manual_Exclude/, skip summary
   c) Auto-correct equipment steps (type 1) and manual steps -> mark Corrected
   d) Noise threshold on Z -> plot to Noisy/, skip summary
   e) Trend fit and standard error; summary only for Raw or Corrected stations
   f) Plot with step lines and trend lines
4) Write kept stations to <frame>_enu_NA.csv, one file per frame

Every station is plotted and sorted into Uncorrected/, Corrected/, Noisy/ and
Manual_Exclude/ under paths_gps['Fig_Dir'] for visual review.

Source: GNSS time series from the Nevada Geodetic Laboratory (UNR; Blewitt et
al. 2018), IGS20 NA plate-fixed frame. The screening and step correction follow
NC_ALOS-2/run00_prepUNR.py (written for an earlier project by the same author).
"""
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Tuple
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _os, sys as _sys        # path shim: import the repo root modules
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)
from TulePond_filepaths import common_paths, paths_gps, frames
import insar_utils as utils

# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------
tol             = 1.0     # years slack: station record may start/end within +/- tol of the InSAR window
data_threshold  = 0.70    # fraction of days a station must have to be kept
noise_threshold = 1.0     # std(Z) > threshold mm -> noisy station, dropped
STEP_THRESHOLD  = 3.0     # mm jump at a documented step date -> corrected

skip_download   = False   # True to reuse already-downloaded UNR files

# GNSS velocities are computed in the NA (North America) plate-fixed frame, so
# the tied InSAR velocity fields are NA-referenced with no separate plate
# motion correction (applying one would double-correct).
ref_frame    = 'NA'
NA_URL       = 'https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/NA/'
steps_URL    = 'https://geodesy.unr.edu/NGLStationPages/steps.txt'
holdings_URL = 'https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt'

# ROI: the AOI padded outward so stations just outside the frame edges still
# constrain the GNSS ramp (run05) at the edges.
ROI_PAD = 0.15            # degrees
w, s, e, n = common_paths['aoi_bbox']
Range = [w - ROI_PAD, e + ROI_PAD, s - ROI_PAD, n + ROI_PAD]   # min_lon, max_lon, min_lat, max_lat

# Observation window: first and last DISP-S1 acquisition used in the paper.
# Stations are screened against this window (with +/- tol slack), and trends
# are fitted over it, so GNSS and InSAR velocities cover the same period.
t0 = utils.date_to_decimal_year(common_paths['date_start'])
te = utils.date_to_decimal_year(common_paths['date_end'])
d0 = datetime.strptime(common_paths['date_start'], "%Y-%m-%d")
d1 = datetime.strptime(common_paths['date_end'],   "%Y-%m-%d")

# Manual exclusion list, from visual review of every station plot. Stations
# with unexplained steps, strong non-linearity or local motions that would
# bias a whole-field ramp.
manual_exclude = [
    "ASHL", "DUBP", "EBBS", "FLNT", "INV1", "LRA1", "LRA2", "LRA4",
    "MCCM", "MNRC", "MONB", "OREO", "OXMT", "P136", "P141", "P144", "P144",
    "P149", "P150", "P170", "P215", "P242", "P274", "P299",
    "P312", "P332", "P340", "P348", "P655", "P656", "P658", "P663", "P666",
    "P668", "P671", "P673", "P674", "P794", "RAPT", "RBRU", "RDFD", "RDGM",
    "RNO1", "SLID", "TRAN", "YBHB", "P175", "P661", "P670", "P669", "P305",
    "MNDS", "FARB", "P534", "P313", "P336", "P219",
    "P231", "CAMT", "P171" ]

# Don't fix the equipment step for these stations: the documented step has no
# visible offset, and correcting it would introduce one.
skip_equip = [
    "CAP1", "CCSF", "CCSF", "CROW", "CYTE", "FARB", "HCRO", "HOPB", "LKVW",
    "LUTZ", "MCCM", "MODB", "MUSB", "MUSB", "P059", "P096", "P136", "P140",
    "P143", "P155", "P155", "P159", "P164", "P172", "P195", "P195",
    "P207", "P239", "P320", "P322", "P329", "P335", "P186",
    "P336", "P336", "P341", "P362", "P657", "P668", "P668", "P671", "P671",
    "P672", "P730", "P731", "SACR", "SAOB", "SARG", "SBRB",
]

# Steps missing from UNR steps.txt, found in visual review (mostly the 2023
# firmware campaign). [station, decimal year]; corrected like equipment steps.
manual_steps = [
    ["CARK", 2022.1437], ["GRDV", 2016.0301], ["GRDV", 2018.956],
    ["P096", 2023.2498], ["P160", 2023.258],  ["P161", 2023.2498], ["P161", 2023.9640],
    ["P165", 2023.2444], ["P166", 2023.4387], ["P167", 2023.4689], ["P168", 2021.3279],
    ["P169", 2023.2498], ["P183", 2023.2498], ["P185", 2023.2498], ["P187", 2023.2444],
    ["P188", 2023.6304], ["P189", 2023.5975], ["P198", 2023.2444], ["P198", 2023.6769],
    ["P198", 2023.8138], ["P198", 2023.9233], ["P199", 2023.2498], ["P199", 2024.0600],
    ["P200", 2023.2444], ["P202", 2023.2444], ["P204", 2023.269],  ["P265", 2023.3511],
    ["P265", 2023.9425], ["P268", 2023.2772], ["P314", 2023.2772], ["P315", 2023.3511],
    ["P317", 2023.2444], ["P321", 2023.2498], ["P332", 2023.2444], ["P349", 2023.3155],
    ["P670", 2023.2608], ["KFRC", 2021.5688], ["P319", 2023.8439], ["P326", 2021.3260],
    ["P329", 2023.4497], ["P335", 2023.4798], ["P336", 2023.4497], ["P322", 2021.3417],
    ["CCSF", 2016.9144], ["DIAB", 2022.5462], ["P175", 2023.2498], ["P177", 2023.2498],
    ["P178", 2023.2444], ["P212", 2023.2444], ["P214", 2023.2444], ["P215", 2023.269],
    ["P216", 2023.2444], ["P217", 2023.2444], ["P220", 2023.2772], ["P212", 2023.2444],
    ["P222", 2023.3511], ["P225", 2023.2444], ["P230", 2023.2772], ["P234", 2023.3812],
    ["P234", 2023.2444], ["P235", 2023.2444], ["P237", 2023.269],  ["P239", 2023.2444],
    ["P242", 2023.2444], ["P242", 2023.3155], ["P242", 2023.9343], ["P247", 2023.6304],
    ["P250", 2023.4579], ["P254", 2023.3046], ["P304", 2023.4196], ["P332", 2023.4935],
    ["P670", 2023.4004], ["P670", 2023.403] ]


# ---------------------------------------------------------------------------
# Helpers: download, station list
# ---------------------------------------------------------------------------
def dload_site_list(url: str, out_file: str) -> None:
    print(f"Downloading {url} -> {out_file}")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    urlretrieve(url, out_file)


def get_station_list(holdings_file: str, output_file: str) -> int:
    """Stations inside the padded ROI whose record spans the InSAR window."""
    df = pd.read_csv(
        holdings_file,
        sep=r'\s+', comment='#',
        usecols=[0, 1, 2, 7, 8], header=0,
        names=['StaID', 'lat', 'lon', 'dtbeg', 'dtend']
    )
    # 1) UNR holdings longitudes are 0-360; the ROI is in signed longitude
    df['lon'] = ((df['lon'] + 180) % 360) - 180
    df['dtbeg'] = pd.to_datetime(df['dtbeg'], errors='coerce')
    df['dtend'] = pd.to_datetime(df['dtend'], errors='coerce')
    df.dropna(subset=['dtbeg', 'dtend'], inplace=True)
    df['avail_start'] = df['dtbeg'].apply(utils.date_to_decimal_year)
    df['avail_end']   = df['dtend'].apply(utils.date_to_decimal_year)
    # 2) Inside the ROI, and record spans the InSAR window within +/- tol years
    mn_lon, mx_lon, mn_lat, mx_lat = Range
    geo_mask  = df.lon.between(mn_lon, mx_lon) & df.lat.between(mn_lat, mx_lat)
    time_mask = (df['avail_start'] <= t0 + tol) & (df['avail_end'] >= te - tol)
    df2 = df.loc[geo_mask & time_mask, ['StaID', 'lat', 'lon', 'avail_start', 'avail_end']]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df2.to_csv(output_file, sep=' ', index=False, header=False, float_format='%.6f')
    print(f"Wrote {len(df2)} stations -> {output_file}")
    return len(df2)


def download_data_with_sta_list(data_dir: str, sta_list_file: str) -> int:
    os.makedirs(data_dir, exist_ok=True)
    cnt = 0
    with open(sta_list_file) as f:
        for L in f:
            sta = L.split()[0]
            outp = os.path.join(data_dir, f"{sta}.{ref_frame}.tenv3")
            if os.path.exists(outp):
                continue
            url = f"{NA_URL}{sta}.{ref_frame}.tenv3"
            print(f"Downloading {sta} -> {outp}")
            try:
                subprocess.run(['wget', '-q', '-O', outp, url], check=True)
                cnt += 1
            except subprocess.CalledProcessError:
                # station listed in holdings but absent from this reference frame
                print(f"  !! {sta} not available in {ref_frame}, skipping")
                if os.path.exists(outp):
                    os.remove(outp)
    print(f"Downloaded {cnt} new -> {data_dir}")
    return cnt


# ---------------------------------------------------------------------------
# Processing & correction helpers
# ---------------------------------------------------------------------------
def parse_steps_file(file_path: str, sta_id: str) -> Tuple[List[float], List[float]]:
    """Step dates for one station from UNR steps.txt.

    Type 1 = equipment change (corrected below); type 2 = earthquake (kept as
    ground motion, plotted for reference only).
    """
    month_map = {m: i for i, m in enumerate(
        ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'], 1
    )}
    equip, quake = [], []
    for ln in open(file_path):
        p = ln.split()
        if not p or p[0] != sta_id:
            continue
        yy = int(p[1][:2]) + 2000
        mm = month_map[p[1][2:5]]
        dd = int(p[1][5:])
        dt = datetime(yy, mm, dd)
        decy = dt.year + (dt.timetuple().tm_yday - 1) / 365.25
        if p[2] == '1': equip.append(decy)
        if p[2] == '2': quake.append(decy)
    return equip, quake


def find_step_indices(years: np.ndarray, step: float) -> Tuple[int, int]:
    i = np.argmin(np.abs(years - step))
    return i - 1, i


def estimate_and_correct_offset(
    years: np.ndarray,
    E: np.ndarray, N: np.ndarray, Z: np.ndarray,
    window: Tuple[int, int], threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[float], List[float]]:
    """Measure the jump across a documented step date and remove it if real.

    1) Offset = mean of 10 samples after minus mean of 10 samples before
    2) Only correct when any component jumps by more than `threshold` mm,
       so documented-but-invisible steps are left untouched
    3) Correction is applied to all three components from the step onward
    """
    i0, i1 = window
    offE = np.nanmean(E[i1:i1+10]) - np.nanmean(E[i0-10:i0])
    offN = np.nanmean(N[i1:i1+10]) - np.nanmean(N[i0-10:i0])
    offZ = np.nanmean(Z[i1:i1+10]) - np.nanmean(Z[i0-10:i0])
    offsets = [offE, offN, offZ]

    steps: List[float] = []
    if abs(offE) > threshold or abs(offN) > threshold or abs(offZ) > threshold:
        steps.append(years[i1])
        E[i1:] -= offE
        N[i1:] -= offN
        Z[i1:] -= offZ

    return E, N, Z, steps, offsets


def process_station_data(
    file_path: str, sta_id: str, min_len: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read one .tenv3 file: keep the InSAR window, mm, zeroed at first sample."""
    yrs, Es, Ns, Zs = [], [], [], []
    with open(file_path) as f:
        for ln in f:
            p = ln.split()
            if not p or p[0] != sta_id:
                continue
            y = float(p[2])
            if not (t0 <= y <= te):
                continue
            yrs.append(y)
            Es.append(float(p[8]))
            Ns.append(float(p[10]))
            Zs.append(float(p[12]))
    if len(Es) < min_len:
        return None, None, None, None
    years = np.array(yrs)
    E = (np.array(Es) - Es[0]) * 1000
    N = (np.array(Ns) - Ns[0]) * 1000
    Z = (np.array(Zs) - Zs[0]) * 1000
    return years, E, N, Z


def compute_trend(years: np.ndarray, vals: np.ndarray) -> Tuple[float, float, float, float]:
    """Linear trend over the InSAR window. Returns (slope, stderr, r, intercept).

    The intercept is kept so the plotted trend line is the actual fit: a
    station whose record starts after t0 is zeroed at its own first sample,
    so forcing the line through (t0, 0) would draw it visibly off the data.
    """
    s, b, r, p, st = stats.linregress(years - t0, vals)
    return s, st, r, b


# ---------------------------------------------------------------------------
# Plotting function
# ---------------------------------------------------------------------------
def plot_timeseries(
    years: np.ndarray,
    E: np.ndarray, N: np.ndarray, Z: np.ndarray,
    equip: List[float], quake: List[float], corr: List[float],
    sta_id: str, fig_dir: str, status_label: str,
    E_ori: np.ndarray, N_ori: np.ndarray, Z_ori: np.ndarray,
    frame: str,
):
    paths_map = {
        'Raw':       os.path.join(fig_dir, 'Uncorrected'),
        'Corrected': os.path.join(fig_dir, 'Corrected'),
        'Noisy':     os.path.join(fig_dir, 'Noisy'),
        'Manual':    os.path.join(fig_dir, 'Manual_Exclude')
    }
    for p in paths_map.values():
        os.makedirs(p, exist_ok=True)
    folder = status_label if status_label in paths_map else 'Raw'

    # Re-zero series so first point is zero
    E, N, Z = E - E[0], N - N[0], Z - Z[0]
    E_ori, N_ori, Z_ori = E_ori - E_ori[0], N_ori - N_ori[0], Z_ori - Z_ori[0]

    Ve, se, Re, be = compute_trend(years, E)
    Vn, sn, Rn, bn = compute_trend(years, N)
    Vz, sz, Rz, bz = compute_trend(years, Z)

    plt.figure(figsize=(8, 4))
    # Documented steps: yellow = equipment, lightblue = earthquake, red dashed = corrected here
    for idx, st_ in enumerate(equip):
        plt.axvline(st_, color='yellow', lw=2, label='1: Instrument' if idx == 0 else None)
    for idx, st_ in enumerate(quake):
        plt.axvline(st_, color='lightblue', lw=2, label='2: Earthquake' if idx == 0 else None)
    for idx, st_ in enumerate(corr):
        plt.axvline(st_, color='red', ls='--', lw=2, label='Fixed step' if idx == 0 else None)

    # Original series in grey behind the corrected series
    plt.plot(years, Z_ori, 'o', ms=1, alpha=0.2, color='gray')
    plt.plot(years, N_ori, 'o', ms=1, alpha=0.2, color='gray')
    plt.plot(years, E_ori, 'o', ms=1, alpha=0.2, color='gray')

    plt.plot(years, Z, 'o', ms=1, alpha=0.5, label='Z', color='tab:blue')
    plt.plot(years, N, 'o', ms=1, alpha=0.5, label='N', color='tab:orange')
    plt.plot(years, E, 'o', ms=1, alpha=0.5, label='E', color='tab:green')

    # Trend lines span the station's OWN record and include the intercept
    xs = np.array([years[0], years[-1]])
    plt.plot(xs, Ve*(xs - t0) + be, '--', label=f'E {Ve:.2f}±{se:.2f} mm/yr', color='tab:green')
    plt.plot(xs, Vn*(xs - t0) + bn, '--', label=f'N {Vn:.2f}±{sn:.2f} mm/yr', color='tab:orange')
    plt.plot(xs, Vz*(xs - t0) + bz, '--', label=f'Z {Vz:.2f}±{sz:.2f} mm/yr', color='tab:blue')

    plt.xlabel('Year')
    plt.ylabel('Displacement (mm)')
    plt.title(f'Station {sta_id}, Frame {frame} ({folder})')
    plt.xlim(t0, te)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    outp = os.path.join(paths_map[folder], f"{frame}_{sta_id}_{folder}_{ref_frame}.png")
    plt.savefig(outp, dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Screen every station of one frame
# ---------------------------------------------------------------------------
def run_path(frame: str, sta_list_file: str, data_dir: str, fig_dir: str,
             steps_file: str, min_len: int) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    df = pd.read_csv(sta_list_file, sep=' ', header=None,
                     usecols=[0, 1, 2], names=['StaID', 'lat', 'lon'])
    for _, row in df.iterrows():
        sid, lat, lon = row['StaID'], row['lat'], row['lon']
        status_label = 'Raw'

        # a) Data length check: enough days inside the InSAR window
        ts_path = os.path.join(data_dir, f"{sid}.{ref_frame}.tenv3")
        if not os.path.exists(ts_path):
            continue
        yrs, E_raw, N_raw, Z_raw = process_station_data(ts_path, sid, min_len)
        if yrs is None:
            continue

        E_ori, N_ori, Z_ori = E_raw.copy(), N_raw.copy(), Z_raw.copy()
        E, N, Z = E_raw.copy(), N_raw.copy(), Z_raw.copy()

        equip_all, quake = parse_steps_file(steps_file, sid)
        equip = [x for x in equip_all if t0 < x < te]
        corr_steps: List[float] = []

        # b) Manual steps first (dates found in review, missing from steps.txt)
        for ms_id, ms_year in manual_steps:
            if sid == ms_id and t0 < ms_year < te:
                w0, w1 = find_step_indices(yrs, ms_year)
                E, N, Z, c, offsets = estimate_and_correct_offset(yrs, E, N, Z, (w0, w1), STEP_THRESHOLD)
                if c:
                    corr_steps.extend(c)
                    print(f"{sid}: manual correction at {c}, offsets={offsets}")

        # c) Auto-correct documented equipment steps (unless reviewed as spurious)
        if sid not in skip_equip:
            for st_ in equip:
                w0, w1 = find_step_indices(yrs, st_)
                E, N, Z, c, offsets = estimate_and_correct_offset(yrs, E, N, Z, (w0, w1), STEP_THRESHOLD)
                if c:
                    corr_steps.extend(c)
                    print(f"{sid}: auto correction at {c}, offsets={offsets}")

        corr_steps = sorted(set(corr_steps))
        if corr_steps and status_label == 'Raw':
            status_label = 'Corrected'

        # d) Trend fit; noisy or manually-excluded stations are plotted but dropped
        Ve, se, Re, _ = compute_trend(yrs, E)
        Vn, sn, Rn, _ = compute_trend(yrs, N)
        Vz, sz, Rz, _ = compute_trend(yrs, Z)

        if sz > noise_threshold:
            print(f"{sid}: noisy (Z std={sz:.2f} mm) -> Noisy")
            status_label = 'Noisy'
        if sid in manual_exclude:
            print(f"{sid}: manual exclude -> Manual")
            status_label = 'Manual'

        # e) Summary rows only for stations that survive screening
        if status_label in ['Raw', 'Corrected']:
            summary.append({
                'StaID': sid, 'Lon': lon, 'Lat': lat,
                'Ve': Ve, 'Vn': Vn, 'Vz': Vz,
                'Std_e': se, 'Std_n': sn, 'Std_z': sz,
                'r_e': Re, 'r_n': Rn, 'r_z': Rz
            })

        # f) Every station is plotted, sorted by its screening outcome
        plot_timeseries(yrs, E, N, Z, equip, quake, corr_steps, sid,
                        fig_dir, status_label, E_ori, N_ori, Z_ori, frame)
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':

    # 1) Refresh UNR metadata (holdings list + documented steps)
    if not skip_download:
        dload_site_list(holdings_URL, paths_gps['DataHoldings'])
        dload_site_list(steps_URL, paths_gps['Steps'])

    min_len = round((d1 - d0).days * data_threshold)
    print(f"\nROI (padded {ROI_PAD} deg): lon {Range[0]:.4f}..{Range[1]:.4f}  "
          f"lat {Range[2]:.4f}..{Range[3]:.4f}")
    print(f"InSAR window {common_paths['date_start']} .. {common_paths['date_end']} "
          f"({t0:.3f}-{te:.3f}), min {min_len} days of GNSS")

    for frame in frames:
        # 2) Station list for this frame's screening, then daily time series
        sta_list = paths_gps['StaList'].format(frame=frame)
        get_station_list(paths_gps['DataHoldings'], sta_list)
        if not skip_download:
            download_data_with_sta_list(paths_gps['NA_dir'], sta_list)

        # 3) Screen, correct and plot every station
        df = pd.DataFrame(run_path(frame, sta_list, paths_gps['NA_dir'],
                                   paths_gps['Fig_Dir'], paths_gps['Steps'], min_len))

        # 4) Screened ENU velocities, one CSV per frame -> input to run05
        out = paths_gps['enu'].format(frame=frame, ref_frame=ref_frame)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        df.to_csv(out, index=False)
        print(f"frame {frame}: {len(df)} stations kept -> {out}")

    print("\nProcessing complete!")
    print(f"Review the time series plots under: {paths_gps['Fig_Dir']}")
    for sub in ("Uncorrected", "Corrected", "Noisy", "Manual_Exclude"):
        d = os.path.join(paths_gps['Fig_Dir'], sub)
        if os.path.isdir(d):
            print(f"  {sub:16s} {len(os.listdir(d)):4d} plots")
