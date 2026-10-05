# %% [markdown]
# # [1] 10 Rainfall at Tule Pond and the onset seasonality
#
# ```
# Created on Thu Sep 24 2026
# @author: daniellelindsay
# Run in opera_disp_feb13. Seconds.
# ```
#
# **What this notebook does.**
#
# 1. **Daily rain at Tule Pond** from the two nearest NOAA GHCN-daily gauges,
#    spliced day by day, Fremont first, then Newark (no rescaling; the overlap is
#    documented). Hayward Air Terminal (18 km) is read only as the midnight-basis
#    reference for the check below; it is not in the composite (Dani 2026-10-02:
#    it filled only 69 days, so two gauges is the cleaner record). The morning-observer COOP gauge
#    (Newark) is shifted one day earlier onto the midnight basis, and that shift
#    is checked against Hayward and on a storm with a date in the news. Over the InSAR events
#    the record is Fremont alone (asserted), so the storm analysis, which uses only the InSAR
#    events, sees one consistent gauge; the Newark fill serves the other years, for display. Water year Oct 1 to Sep 30,
#    named by the year it ends. The gauge files in `external_inputs/GHCN_daily`
#    (`common_paths["ghcn_dir"]`) are the frozen inputs: a missing one stops the notebook
#    (`TULEPOND_DOWNLOAD_GHCN=1` downloads it instead).
# 2. **The event table**: the InSAR and HRKT (bimonthly-era) events with their
#    window, step and the rain around the window (`event_table.csv`).
# 3. **Onset seasonality**: every season-resolved onset (the InSAR windows and the
#    bimonthly-era array windows, all 130 d or less) on the water year over the rain
#    (a working figure here; the paper's Fig 3B is drawn by 11_forecast from the
#    panel products this notebook saves). The wet season of this count is November to
#    March (`wet_season_months`); 11_forecast uses its own, named windows.
#
# The forecast of the next event and the rainfall conditioning of the onsets
# are notebook 11 (11_forecast), which reads the rain table this notebook writes.
#
# PREPUB: Ported from `Hayward_DISP-S1/analysis/run20_precip_ghcn.py` and
# PREPUB: `plotting/fig_hrkt_onset_seasonality.py`.
#
# **Inputs**: the three GHCN-daily gauge files, 04 `creep_events.csv` (InSAR onset windows, `onset_lo`
# to `onset_hi`), 09 `hrkt_aa_events.csv` (array onset windows, the same columns, plus `onset_mid`, the
# window midpoint, for plot positions only) and `aa_insar_stacks.csv` (the start of the InSAR
# record), the HRKT array file (the end of its bimonthly surveys).
#
# **Outputs** (`outputs/10_rain_seasonality/`): `tulepond_daily_precip.csv`,
# `wateryear_totals.csv`, `event_table.csv`, the Fig 3 panel products `fig3_rain_panel.npz` and
# `fig3_onset_windows.csv` (one row per onset window of both catalogues: `onset_lo`, `onset_hi`,
# `onset_mid` for the plot position, the day of water year of each end; read by 11), the stats
# sidecar `10_rain_stats.txt`, registry rows `10_*`.
#
# **Figures**: `qc10_onset_seasonality_wateryear_density7d_vmax80mm` (working figure, not Fig 3).
#
# **What to check.** Fremont reads ~12 % more than Newark in the median water year (ratio
# 0.73-1.29 over the complete years; no rescaling); Newark shifted by one day agrees better with
# Hayward on storm days and lines up with it on the 11 Dec 2014 storm (both asserted); every InSAR
# onset window and the 30 days before it on Fremont alone (asserted: the Newark fill does not
# enter the InSAR analysis); the onset windows counted against the wet season BY WINDOW, not
# midpoint (November to March: wholly inside, partly, wholly outside; `10_n_wet_season_*`).

# %%
# [2]
import os
import urllib.request

import insar_utils as utils
from TulePond_filepaths import common_paths, paths_out, repo_dir, forecast_issue_date, rain_total_days, rain_total_mm

import numpy as np                               # Matrix calculations
import pandas as pd                              # Tables
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# %% [markdown]
# ## [3] Paths and parameters
#
# The gauges, the wet season, the figure settings and the input files; the registry records the inputs.

# %%
# [4]
notebook = "10_rain_seasonality"
out_dir = os.path.join(common_paths["out_dir"], notebook)
fig_dir = common_paths["fig_dir"]
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)


# the gauges: id -> (short name, order in the composite, day shift for morning-observer COOP gauges).
# The composite is the two nearest Tule Pond (4 and 7 km); Hayward Air Terminal (18 km) is the midnight-basis
# reference for the Newark observation-time check only.
# PREPUB: Livermore, San Jose and Oakland Museum (22 to 37 km) were carried as alternates until 2026-09-24 and entered no number.
ncei = "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/{sid}.csv"
download_missing = os.environ.get("TULEPOND_DOWNLOAD_GHCN", "0") == "1"   # 1: download a missing gauge file (the files are frozen inputs)
stations = {"USC00043244": ("FREMONT", "composite-1", 0),      # COOP but on the midnight basis (checked against Hayward)
            "USC00046144": ("NEWARK", "composite-2", -1),      # COOP, morning observer (checked on the 11 Dec 2014 storm below)
            "USW00093228": ("HAYWARD_AT", "check only", 0)}    # ASOS, midnight basis: the reference for the Newark shift check
composite = ["FREMONT", "NEWARK"]
tulepond = (37.557, -121.970)      # section centre (lat, lon)
t_start = "1978-10-01"             # start of WY1979 (first HRKT survey 1979.7)
series_end = pd.Timestamp(forecast_issue_date)   # the rain series ends at the forecast issue date so the run reproduces
max_missing_days = 20              # a water year with at most this many missing days is "complete"
min_gauge_days = 350               # gauge comparison: water years in which both gauges reported on at least this many days
storm_day_mm = 10.0                # gauge comparison: days on which either gauge has at least this much rain
complete_wy = (1980, 2025)         # the complete water years the rain climatology is quoted over
insar_rain_check_days = 30         # the one-gauge check: each InSAR window and this many days before it must be Fremont alone

# the seasonality
resolvable_days = 130      # array windows longer than this do not constrain the season (= 09's dense_window_days)
wet_season_months = [11, 12, 1, 2, 3]   # the wet season of the onset count, November to March (its share of the year's rain is computed below)
density_days = 7           # the figure: running rain total (days)
density_max = 80           # ... mm at full colour
rain_colours = ["#FFFFFF", "#C6DBEF", "#4292C6", "#08306B"]      # sequential blues for the rain (no other blue on the panel)
insar_bar_colour = "#D55E00"      # Okabe-Ito vermillion
array_bar_colour = "0.15"         # near black
x0, x1 = 1979.0, 2026.5           # the time axis of Fig 3: the HRKT record starts 1979.7
loma_prieta = 1989.79             # 17 Oct 1989

# the InSAR stacks at the six arrays (09); only HRKT's first date is used here (the start of the InSAR record for Fig 3A)
stacks_file = os.path.join(paths_out["09_alignment_arrays"], "aa_insar_stacks.csv")
hrkt_file = os.path.join(common_paths["aa_dir"], "HRKT.csv")
events_file = os.path.join(paths_out["04_event_detection"], "creep_events.csv")
aa_events_file = os.path.join(paths_out["09_alignment_arrays"], "hrkt_aa_events.csv")

REG = utils.Registry(notebook, common_paths, repo_dir)
gauge_files = []
for sid in stations:
    gauge_files.append(os.path.join(common_paths["ghcn_dir"], f"{sid}.csv"))
REG.inputs(events_file, aa_events_file, stacks_file, hrkt_file, *gauge_files)

# %% [markdown]
# ## [5] Functions: dates and water years
#
# Water year N runs Oct 1 of N-1 to Sep 30 of N; day of water year counts
# from Oct 1 = 1. A decimal year is turned into its calendar day first, so
# every day of water year in this notebook comes from the same calendar.

# %%
# [6]
def decyear(dates):
    """Decimal year of a DatetimeIndex, fraction of the actual year length."""
    dates = pd.DatetimeIndex(dates)
    year_start = pd.to_datetime(dates.year.astype(str) + "-01-01")
    next_year_start = pd.to_datetime((dates.year + 1).astype(str) + "-01-01")
    return dates.year + (dates - year_start) / (next_year_start - year_start)


def to_date(decimal_years):
    """The calendar day nearest to each decimal year (an acquisition's own day, not the day before)."""
    out = []
    for value in np.atleast_1d(decimal_years):
        out.append(utils.decimal_year_to_day(value))
    return pd.DatetimeIndex(out)


def water_year(dates):
    """Water year of each date: Oct 1 of N-1 to Sep 30 of N is water year N."""
    dates = pd.DatetimeIndex(dates)
    return dates.year + (dates.month >= 10).astype(int)


def dowy(dates):
    """Day of water year: Oct 1 = 1."""
    dates = pd.DatetimeIndex(dates)
    wy_start = pd.to_datetime((water_year(dates) - 1).astype(str) + "-10-01")
    return (dates - wy_start).days + 1


def dowy_frac(decimal_years):
    """Day of water year (Oct 1 = 1) of decimal years, through their calendar day."""
    return np.asarray(dowy(to_date(decimal_years)))


def sigma_from_ci95(lo, hi):
    """1 sigma from a 95 % interval: for a normal distribution it spans mean +/- 1.96 sigma (full width 3.92 sigma)."""
    return (hi - lo) / 3.92

# %% [markdown]
# ## [7] The daily rain composite
#
# One value per day: Fremont where it reported, otherwise Newark. The water-year totals table records which
# gauges made each year and how many days are missing.

# %%
# [8]
def fetch(sid):
    """One GHCN-daily gauge as a daily series (mm) from t_start, with its metadata. The file is a frozen input:
    a missing one stops the notebook unless TULEPOND_DOWNLOAD_GHCN=1."""
    path = os.path.join(common_paths["ghcn_dir"], f"{sid}.csv")
    if not os.path.exists(path):
        assert download_missing, f"missing frozen gauge file {path}; set TULEPOND_DOWNLOAD_GHCN=1 to download it from {ncei.format(sid=sid)}"
        print("downloading", sid)
        urllib.request.urlretrieve(ncei.format(sid=sid), path)
    assert os.path.getsize(path) >= 1000, f"{path} is truncated ({os.path.getsize(path)} bytes)"
    table = pd.read_csv(path, usecols=["DATE", "LATITUDE", "LONGITUDE", "NAME", "PRCP"], low_memory=False)
    table["DATE"] = pd.to_datetime(table.DATE)
    table = table[table.DATE >= t_start]
    info = dict(name=table.NAME.iloc[0], lat=float(table.LATITUDE.iloc[0]), lon=float(table.LONGITUDE.iloc[0]),
                first=table.DATE.min().date(), last=table.DATE.max().date())
    series = table.set_index("DATE").PRCP / 10.0          # tenths of mm -> mm
    series = series[~series.index.duplicated()]
    day_shift = stations[sid][2]
    if day_shift:
        series.index = series.index + pd.Timedelta(days=day_shift)      # the COOP morning-observer shift
    return series, info


def distance_km(lat_a, lon_a, lat_b, lon_b):
    """Flat-earth distance between two nearby points (km)."""
    return float(np.hypot((lat_a - lat_b) * 111.19, (lon_a - lon_b) * 111.19 * np.cos(np.radians(lat_b))))


days = pd.date_range(t_start, series_end, freq="D")
rain = pd.DataFrame(index=days)
gauge_info = {}
for sid, (name, role, shift) in stations.items():
    series, info = fetch(sid)
    rain[name] = series.reindex(days)
    gauge_info[name] = dict(sid=sid, role=role, shift=shift, **info, n_valid=int(series[series.index <= series_end].notna().sum()),
                            dist_km=distance_km(info["lat"], info["lon"], tulepond[0], tulepond[1]))
# the composite: the first gauge in the order that reported that day
rain["prcp_mm"] = np.nan
rain["src"] = "none"
for name in composite:
    take = rain.prcp_mm.isna() & rain[name].notna()
    rain.loc[take, "prcp_mm"] = rain.loc[take, name]
    rain.loc[take, "src"] = name
rain["decyear"] = decyear(rain.index)
rain["wy"] = water_year(rain.index)
rain["dowy"] = dowy(rain.index)
rain.index.name = "date"


def gauges_used(sources):
    """The gauges that made a water year of the composite, joined by '+'."""
    names = sorted(set(sources) - {"none"})
    return "+".join(names)


# water-year totals, per gauge and for the composite
by_wy = rain.groupby("wy")
missing_days = rain.prcp_mm.isna().groupby(rain.wy).sum()
wy_totals = pd.DataFrame({"total_mm": by_wy.prcp_mm.sum(min_count=1), "n_days": by_wy.prcp_mm.count(),
                          "n_missing": missing_days, "max_day_mm": by_wy.prcp_mm.max(), "src": by_wy.src.agg(gauges_used)})
for name, role, shift in stations.values():
    wy_totals[f"{name}_mm"] = by_wy[name].sum(min_count=1)
    wy_totals[f"{name}_ndays"] = by_wy[name].count()
wy_totals["complete"] = wy_totals.n_missing <= max_missing_days
rain.to_csv(os.path.join(out_dir, "tulepond_daily_precip.csv"), float_format="%.4f")
wy_totals.to_csv(os.path.join(out_dir, "wateryear_totals.csv"), float_format="%.1f")

stats_lines = ["10 RAIN AND SEASONALITY",
               f"rain series {days[0].date()} to {days[-1].date()} (ends at the pinned forecast issue date {forecast_issue_date})", "stations:"]
for name, info in gauge_info.items():
    stats_lines.append(f"  {name:<12} {info['sid']}  {info['name']:<38} {info['dist_km']:5.1f} km  {info['first']}..{info['last']}  "
                       f"n={info['n_valid']}  [{info['role']}, day shift {info['shift']:+d}]")
n_by_source = rain.src.value_counts()
source_counts = []
for source, n_source in n_by_source.items():
    source_counts.append(f"{source} {n_source} d")
stats_lines.append("composite = first reporting gauge in the order " + " > ".join(composite) + ": " + ", ".join(source_counts))
# how the gauges agree where they overlap (Hayward rows are the reference for the check, not the composite)
for gauge_a, gauge_b in [("FREMONT", "NEWARK"), ("FREMONT", "HAYWARD_AT"), ("NEWARK", "HAYWARD_AT")]:
    both_complete = wy_totals[(wy_totals[f"{gauge_a}_ndays"] >= min_gauge_days) & (wy_totals[f"{gauge_b}_ndays"] >= min_gauge_days)]
    ratio = both_complete[f"{gauge_a}_mm"] / both_complete[f"{gauge_b}_mm"]
    overlap = rain[[gauge_a, gauge_b]].dropna()
    overlap = overlap[overlap.index.isin(rain.index[rain.wy.isin(both_complete.index)])]
    storm_days_both = overlap[(overlap[gauge_a] >= storm_day_mm) | (overlap[gauge_b] >= storm_day_mm)]
    assert len(storm_days_both) > 10, f"{gauge_a}/{gauge_b}: only {len(storm_days_both)} storm days in common"
    r_day = np.corrcoef(storm_days_both[gauge_a], storm_days_both[gauge_b])[0, 1]
    stats_lines.append(f"  {gauge_a}/{gauge_b}: {len(both_complete)} complete WYs, WY-total ratio median {ratio.median():.2f} "
                       f"(range {ratio.min():.2f}-{ratio.max():.2f}); same-day totals on days >= {storm_day_mm:g} mm: "
                       f"n={len(storm_days_both)}, r={r_day:.2f}")
complete_years = wy_totals[wy_totals.complete & (wy_totals.index >= complete_wy[0]) & (wy_totals.index <= complete_wy[1])]
stats_lines.append(f"WY{complete_wy[0]}-{complete_wy[1]} complete years (n={len(complete_years)}): median {complete_years.total_mm.median():.0f} mm, "
                   f"10-90% {complete_years.total_mm.quantile(.1):.0f}-{complete_years.total_mm.quantile(.9):.0f} mm")
print("\n".join(stats_lines))

# %% [markdown]
# ## [9] CHECK: the COOP observation-time shift
#
# Newark's observer reads the gauge in the morning, so a storm is booked on
# the day after it fell. Shifted one day earlier it must agree better with Hayward
# Air Terminal (a midnight-basis ASOS gauge) on storm days than as published, and
# put the "storm of the decade", Thursday 11 December 2014, on the same day as
# Hayward (both asserted). Hayward is only the reference for this check; it is not
# in the composite.
#
# The Newark fill, shifted or not, does not enter the InSAR analysis: every InSAR
# onset window and the 30 days before it are Fremont alone (asserted in [12]). The
# storm analysis uses only the InSAR events, so it needs one consistent gauge over
# the InSAR period; the composite's other years are for display.

# %%
# [10]
def storm_day_correlation(series_a, series_b):
    """Correlation of two daily rain series on the days either has at least storm_day_mm; and the number of days."""
    both = pd.concat([series_a.reindex(days), series_b.reindex(days)], axis=1).dropna()
    both = both[(both.iloc[:, 0] >= storm_day_mm) | (both.iloc[:, 1] >= storm_day_mm)]
    return float(np.corrcoef(both.iloc[:, 0], both.iloc[:, 1])[0, 1]), len(both)


newark_as_published, _ = fetch("USC00046144")
newark_as_published.index = newark_as_published.index - pd.Timedelta(days=stations["USC00046144"][2])   # undo the shift
hayward = rain["HAYWARD_AT"]
r_as_published, n_as_published = storm_day_correlation(newark_as_published, hayward)
r_shifted, n_shifted = storm_day_correlation(rain["NEWARK"], hayward)
print(f"Newark vs Hayward on storm days, as published: r = {r_as_published:.2f} (n = {n_as_published})")
print(f"Newark vs Hayward on storm days, shifted -1 d: r = {r_shifted:.2f} (n = {n_shifted})")
assert r_shifted > r_as_published, "the one-day shift does not improve Newark's agreement with Hayward"

# Hayward books the storm on the 11th; Newark as published on the 12th (the morning its observer read the gauge),
# on the 11th once shifted. Fremont, unshifted, also puts most of it on the 11th.
storm = pd.date_range("2014-12-10", "2014-12-12")
storm_table = pd.DataFrame({"Newark as published": newark_as_published.reindex(storm),
                            "Newark shifted -1 d": rain["NEWARK"].reindex(storm),
                            "Hayward AT (midnight)": hayward.reindex(storm),
                            "Fremont (no shift)": rain["FREMONT"].reindex(storm)}, index=storm.strftime("%a %d %b %Y"))
print("\nthe 11 Dec 2014 storm, mm per day:")
print(storm_table.round(1).to_string())
assert storm_table["Hayward AT (midnight)"].idxmax() == "Thu 11 Dec 2014", "Hayward does not book the 2014 storm on the 11th"
assert storm_table["Newark shifted -1 d"].idxmax() == "Thu 11 Dec 2014", "the shifted Newark does not book the 2014 storm on the 11th"

stats_lines.append(f"CHECK Newark shift: vs Hayward on storm days r {r_as_published:.2f} as published, {r_shifted:.2f} shifted")

# %% [markdown]
# ## [11] The event catalogues
#
# InSAR events from 04 (the onset window runs from `onset_lo`, the last acquisition before the step,
# to `onset_hi`, the first acquisition after it), and the HRKT array events from 09, all from the
# bimonthly era, whose survey windows (130 d or less) resolve the season (asserted). Each InSAR
# event gets an `onset_mid` (the window midpoint), as the array events have: a plot position only.
#
# CHECK, one gauge for the InSAR analysis: every InSAR onset window and the `insar_rain_check_days`
# days before it must be Fremont, with no Newark-filled day (asserted); the days inside those spans
# with no data from either gauge are printed.

# %%
# [12]
creep_events = pd.read_csv(events_file)
insar_events = creep_events[creep_events.segment == "TULEPOND"].sort_values("onset_lo").reset_index(drop=True)
insar_events["onset_mid"] = 0.5 * (insar_events.onset_lo + insar_events.onset_hi)     # plot position only
aa_dense = pd.read_csv(aa_events_file)
assert (aa_dense.window_days <= resolvable_days).all(), f"09 returned an array window longer than {resolvable_days} d"
print(f"InSAR {len(insar_events)} events; array bimonthly era {len(aa_dense)} events (windows <= {resolvable_days} d)")

# one gauge over the InSAR analysis: each InSAR window and the insar_rain_check_days before it, Fremont only
no_data_days = []
for event in insar_events.itertuples():
    first_day = to_date([event.onset_lo])[0] - pd.Timedelta(days=insar_rain_check_days)
    last_day = to_date([event.onset_hi])[0]
    span = rain.loc[first_day:last_day]
    sources_in_span = set(span.src.unique())
    assert sources_in_span <= {"FREMONT", "none"}, \
        f"InSAR window {first_day.date()}..{last_day.date()} uses {sorted(sources_in_span - {'FREMONT', 'none'})}, not Fremont alone"
    for day in span.index[span.src == "none"]:
        no_data_days.append(str(day.date()))
print(f"every InSAR onset window and the {insar_rain_check_days} d before it: Fremont only; days with no data from any gauge: "
      + (", ".join(no_data_days) if no_data_days else "none"))
stats_lines.append(f"CHECK one gauge for the InSAR analysis: every InSAR window and the {insar_rain_check_days} d before it on FREMONT "
                   f"alone (asserted); no-data days in those spans: " + (", ".join(no_data_days) if no_data_days else "none"))


# %% [markdown]
# ## [13] The event table: every season-resolved onset, with its rain
#
# One row per event: the InSAR events (04) and the HRKT events of the bimonthly era (09). The onset is
# equally likely anywhere in its window, so every rain quantity is given at BOTH ends of the window (never at a
# midpoint): the day of the water year, the season, the water-year rain to date, and the days since the last
# storm, a `rain_total_days`-day total of at least `rain_total_mm` (TulePond_filepaths; a description of each window only: 11 tests a grid of totals and lags).
# The step is 04's or 09's fitted amplitude with its 1-sigma error (the InSAR 95 % bootstrap bounds / 3.92).
# Missing rain days in the water year up to the window are counted, so a gap in the composite shows. The share
# of each window inside the wet season (November to March) is added; written to `event_table.csv`.

# %%
# [14]
season_of_month = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
                   6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
season_order = ["autumn", "winter", "spring", "summer"]
rain_filled = rain.prcp_mm.fillna(0.0)
rain_to_date = rain_filled.groupby(rain.wy).cumsum()                       # water-year rain to date, mm
missing_to_date = rain.prcp_mm.isna().astype(int).groupby(rain.wy).cumsum()
storm_total = rain_filled.rolling(rain_total_days, min_periods=1).sum()
# the last storm day so far: each day's own date where it is a storm day, carried forward
last_storm = pd.Series(rain.index, index=rain.index).where(storm_total >= rain_total_mm).ffill()


def days_since_storm(day):
    """Days from the last storm day up to this day (NaN before the first storm of the record)."""
    if pd.isna(last_storm[day]):
        return np.nan
    return (day - last_storm[day]).days


catalogue_rows = []
for event in insar_events.itertuples():
    catalogue_rows.append(("InSAR", event.onset_lo, event.onset_hi, event.amplitude_mm, sigma_from_ci95(event.amp_lo, event.amp_hi)))
for event in aa_dense.itertuples():
    catalogue_rows.append(("HRKT array", event.onset_lo, event.onset_hi, event.amplitude_mm, event.amp_err))
table_rows = []
wet_fraction_by_window = {}            # (source, first day) -> fraction of the window's days in the wet season (unrounded)
for source, window_lo, window_hi, step_mm, step_err in catalogue_rows:
    day_lo = to_date([window_lo])[0]
    day_hi = to_date([window_hi])[0]
    assert rain.index[0] <= day_lo <= day_hi <= rain.index[-1], f"{source} window {day_lo.date()}-{day_hi.date()} outside the rain record"
    seasons = []
    for season in season_order:
        if season in (season_of_month[day_lo.month], season_of_month[day_hi.month]):
            seasons.append(season)
    window_days_all = pd.date_range(day_lo, day_hi)
    wet_fraction_by_window[(source, day_lo.date())] = float(window_days_all.month.isin(wet_season_months).mean())
    table_rows.append(dict(source=source, window_lo=day_lo.date(), window_hi=day_hi.date(), window_days=round((window_hi - window_lo) * 365.25),
                           step_mm=round(step_mm, 1), step_1sigma_mm=round(step_err, 1), water_year=int(rain.wy[day_lo]),
                           dowy_lo=int(rain.dowy[day_lo]), dowy_hi=int(rain.dowy[day_hi]), season="/".join(seasons),
                           wy_rain_to_date_lo_mm=round(float(rain_to_date[day_lo]), 0), wy_rain_to_date_hi_mm=round(float(rain_to_date[day_hi]), 0),
                           days_since_storm_lo=days_since_storm(day_lo), days_since_storm_hi=days_since_storm(day_hi),
                           wy_missing_rain_days=int(missing_to_date[day_hi]),
                           wet_season_fraction=round(wet_fraction_by_window[(source, day_lo.date())], 2)))
event_table = pd.DataFrame(table_rows).sort_values("window_lo").reset_index(drop=True)
event_table.to_csv(os.path.join(out_dir, "event_table.csv"), index=False)
print(f"storm = {rain_total_days}-day total >= {rain_total_mm:g} mm; days since storm and rain to date at the window start (lo) and end (hi)")
print(event_table.to_string(index=False))
print("saved event_table.csv")

# the season of each onset, by its WINDOW (Dani 2026-10-02: intervals, not midpoints): the fraction of the window's
# days that fall in the wet season; an event is "inside" if the whole window does, "partly" if some of it does
wet_fraction = []                      # unrounded, in the order of event_table
for event in event_table.itertuples():
    wet_fraction.append(wet_fraction_by_window[(event.source, event.window_lo)])
wet_fraction = np.array(wet_fraction)
is_insar = (event_table.source == "InSAR").values
n_wet_inside = int((wet_fraction == 1).sum())
n_wet_partly = int(((wet_fraction > 0) & (wet_fraction < 1)).sum())
n_wet_outside = int((wet_fraction == 0).sum())
n_insar_wet_inside = int((wet_fraction[is_insar] == 1).sum())
assert n_wet_inside + n_wet_partly + n_wet_outside == len(event_table)
# the wet season's share of the year's rain: median over the complete water years
in_wet = rain.index.month.isin(wet_season_months)
wet_share = (rain.prcp_mm[in_wet].groupby(rain.wy[in_wet]).sum() / rain.prcp_mm.groupby(rain.wy).sum()).reindex(complete_years.index)
print(f"wet season (months {wet_season_months}) = {100 * wet_share.median():.0f} % of the water-year rain (median of {len(wet_share)} complete years)")
print(f"onset windows: {n_wet_inside} of {len(event_table)} wholly inside the wet season ({n_insar_wet_inside} of {is_insar.sum()} InSAR, "
      f"{int((wet_fraction[~is_insar] == 1).sum())} of {(~is_insar).sum()} array), {n_wet_partly} partly, {n_wet_outside} wholly outside")

# %% [markdown]
# ## [15] Working figure: onsets on the water year over the rain (the paper's Fig 3B is drawn by 11_forecast)
#
# Each water year is a column filled with the running 7-day rain total; the
# water year runs down the page from Oct 1. Onsets are bars bounding the
# observation window only, no midpoint marker: the event is equally likely
# anywhere inside its window. Only bimonthly-era array detections exist (09 runs
# the detector there only), so every array bar constrains the season.

# %%
# [16]
def onset_bar(ax, x, day_lo, day_hi, color, lw):
    """A vertical bar from day_lo to day_hi of the water year at x, wrapping across Oct 1 if needed."""
    if day_hi >= day_lo:
        segments = [(day_lo, day_hi)]
    else:
        segments = [(day_lo, 366), (1, day_hi)]
    for segment_lo, segment_hi in segments:
        ax.plot([x, x], [segment_lo, segment_hi], "-", color=color, lw=lw, solid_capstyle="butt", zorder=5)
    for y in (day_lo, day_hi):
        ax.plot([x - 0.28, x + 0.28], [y, y], "-", color=color, lw=lw * 0.8, zorder=5)


insar_window_days = (insar_events.onset_hi - insar_events.onset_lo) * 365.25
utils.mpl_paper_style(base=8)          # Helvetica, 8 pt, editable PDF text
fig, ax = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
rain_plot = rain[(rain.wy >= int(np.floor(x0)) + 1) & (rain.wy <= int(np.ceil(x1)))]
rain_7d = rain_plot.prcp_mm.fillna(0).rolling(density_days, min_periods=1).sum()
water_years = sorted(rain_plot.wy.unique())
rain_grid = np.full((367, len(water_years)), np.nan)          # day of water year x water year
x_edges = []
for j, wy in enumerate(water_years):
    this_year = rain_plot[rain_plot.wy == wy]
    rain_grid[this_year.dowy.values, j] = rain_7d[this_year.index].values
    x_edges.append(wy - 1 + 273 / 365.25)
x_edges.append(water_years[-1] + 273 / 365.25)
cmap = LinearSegmentedColormap.from_list("rain", rain_colours)
mesh = ax.pcolormesh(np.array(x_edges), np.arange(-0.5, 367.5), rain_grid, cmap=cmap, vmin=0, vmax=density_max, shading="flat", rasterized=True, zorder=0)
for event in aa_dense.itertuples():
    onset_bar(ax, event.onset_mid, dowy_frac(event.onset_lo)[0], dowy_frac(event.onset_hi)[0], array_bar_colour, 1.8)
for event in insar_events.itertuples():
    onset_bar(ax, event.onset_mid, dowy_frac(event.onset_lo)[0], dowy_frac(event.onset_hi)[0], insar_bar_colour, 2.6)
ax.set_xlim(x0, x1)
ax.set_ylim(366, 1)
ax.set_yticks([1, 62, 124, 182, 244, 305, 366])
ax.set_yticklabels(["Oct 1", "Dec 1", "Feb 1", "Apr 1", "Jun 1", "Aug 1", "Oct 1"])
ax.set_xlabel("Year")
ax.set_ylabel("Day of water year")
cb = fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.03, extend="max")
cb.set_label(f"Precipitation, {density_days}-day total (mm)", fontsize=7)
cb.ax.tick_params(labelsize=6)
ax.legend(handles=[Line2D([], [], color=array_bar_colour, lw=1.8, label="Array onset window (bimonthly surveys)"),
                   Line2D([], [], color=insar_bar_colour, lw=2.6,
                          label=f"InSAR onset window ({insar_window_days.min():.0f} to {insar_window_days.max():.0f} d)")],
          fontsize=6, loc="lower left")
f_season = os.path.join(fig_dir, f"qc10_onset_seasonality_wateryear_density{density_days}d_vmax{density_max}mm.png")
fig.savefig(f_season, dpi=300)
fig.savefig(f_season.replace(".png", ".pdf"))
print("Saved figure to:", f_season)
plt.show()

# %% [markdown]
# ## [17] The Fig 3 panels: the HRKT history and the onsets on the water year
#
# Fig 3 is drawn by 11_forecast, which also has the repeat intervals and the forecast. This cell saves
# what its left column needs: the rain grid on the water year, the onset windows of both catalogues (day
# of water year of each window end), and the end of the bimonthly HRKT surveys.

# %%
# [18]
hrkt = utils.load_aa(common_paths["aa_dir"], "HRKT")
hrkt_stack = pd.read_csv(stacks_file).query("code == 'HRKT'")
# the bimonthly surveys end at the last HRKT survey that closes an interval of resolvable_days or less (as 09's
# bimonthly_era); the arrays were then surveyed annually until InSAR
survey_gap_days = np.diff(hrkt.year.values) * 365.25
dense_era_end = float(hrkt.year.values[np.max(np.where(survey_gap_days <= resolvable_days)[0]) + 1])
insar_start = float(hrkt_stack.date.min())
print(f"bimonthly surveys at HRKT end {dense_era_end:.2f}; annual surveys from then until the InSAR record starts {insar_start:.2f}")
window_tables = []
for src, cat in (("aa_dense", aa_dense), ("insar", insar_events)):
    window_tables.append(pd.DataFrame(dict(src=src, onset_lo=cat.onset_lo.values, onset_hi=cat.onset_hi.values, onset_mid=cat.onset_mid.values,
                                           dowy_lo=dowy_frac(cat.onset_lo.values), dowy_hi=dowy_frac(cat.onset_hi.values))))
windows = pd.concat(window_tables, ignore_index=True)
assert len(windows) == len(aa_dense) + len(insar_events) and windows[["dowy_lo", "dowy_hi"]].notna().all().all()
windows.to_csv(os.path.join(out_dir, "fig3_onset_windows.csv"), index=False)
np.savez(os.path.join(out_dir, "fig3_rain_panel.npz"), rain_grid=rain_grid, x_edges=np.asarray(x_edges), density_days=density_days,
         density_max=density_max, x0=x0, x1=x1, loma_prieta=loma_prieta, dense_era_end=dense_era_end, insar_start=insar_start,
         rain_colours=np.array(rain_colours))
print("saved fig3_onset_windows.csv and fig3_rain_panel.npz for 11_forecast")

# %% [markdown]
# ## [19] Registry: the stats sidecar `10_rain_stats.txt` and the `10_*` numbers

# %%
# [20]
stats_lines += ["", f"onset windows against the wet season (months {wet_season_months}, {100 * wet_share.median():.0f} % of the year's rain): "
                f"{n_wet_inside} of {len(event_table)} wholly inside, {n_wet_partly} partly, {n_wet_outside} wholly outside; "
                f"InSAR {n_insar_wet_inside} of {is_insar.sum()} wholly inside"]
sidecar = REG.sidecar("10_rain", stats_lines)
REG.register("10_rain_series_start", str(days[0].date()), "date", 0, where="M9", cell="rain", sidecar=sidecar)
REG.register("10_rain_series_end", str(days[-1].date()), "date", 0, where="M9", cell="rain", sidecar=sidecar)
REG.register("10_wy_median_mm", complete_years.total_mm.median(), "mm", 0, where="M9", cell="rain", sidecar=sidecar)
REG.register("10_wy_p10_mm", complete_years.total_mm.quantile(.1), "mm", 0, where="M9", cell="rain", sidecar=sidecar)
REG.register("10_wy_p90_mm", complete_years.total_mm.quantile(.9), "mm", 0, where="M9", cell="rain", sidecar=sidecar)
REG.register("10_n_aa_season_resolved", len(aa_dense), "count", 0, where="Fig3 caption", cell="catalogues", sidecar=sidecar)
REG.register("10_n_events", len(event_table), "count", 0, where="P10", cell="seasonality", sidecar=sidecar)
REG.register("10_n_wet_season_inside", n_wet_inside, "count", 0, where="P10", cell="seasonality", sidecar=sidecar)
REG.register("10_n_wet_season_partly", n_wet_partly, "count", 0, where="P10", cell="seasonality", sidecar=sidecar)
REG.register("10_n_wet_season_outside", n_wet_outside, "count", 0, where="P10", cell="seasonality", sidecar=sidecar)
REG.register("10_n_insar_wet_season_inside", n_insar_wet_inside, "count", 0, where="P10", cell="seasonality", sidecar=sidecar)
REG.register("10_wet_season_rain_pct", 100 * wet_share.median(), "%", 0, where="P10", cell="seasonality", sidecar=sidecar)
REG.render()
