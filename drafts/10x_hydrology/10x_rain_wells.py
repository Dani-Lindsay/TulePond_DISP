# %% [markdown]
# # [1] 10x (EXPERIMENTAL) Rain, groundwater and the Tule Pond onsets
#
# ```
# Created on Fri Oct 02 2026
# @author: daniellelindsay
# Run in opera_disp_feb13. About a minute.
# ```
#
# **Status: experimental, not part of the paper chain.** Nothing here is
# registered. Figures are `qc10x_*`. It reads notebook 10's products, so run 10
# first.
#
# **The question.** Do the slow slip event onsets sit anywhere unusual in the
# hydrological year, compared with the same time of year in years without an
# event? Three candidate relations (Dani's grid), for the rain and for the two
# ACWD wells either side of the fault:
#
# 1. **Cumulative**: water-year rain to date / the well's rise since its autumn low.
# 2. **Event-driven**: days since the last storm (a 3-day total of at least
#    `storm_mm`) / days since the well's seasonal rise began.
# 3. **Rate**: rain in the previous 30 days / the well's 4-week change.
# 4. **Open column**: the storm test of 11's rule C, redone so its search over
#    storm size and lag is paid for (rain); the east-minus-west head difference
#    (wells).
#
# **How the test works.** The onset is equally likely anywhere in its window
# (Dani's rule), so each event is drawn `n_draws` times uniformly in its window.
# One realisation takes one draw per event and averages the quantity over the
# events. The null takes, for each event, the same day of the water year in a
# random water year WITHOUT an event. The p-value is how often the null average
# is as extreme as the observed one (two-sided).
#
# **Inputs**: notebook 10's `tulepond_daily_precip.csv` and `event_table.csv`
# (the InSAR events and the HRKT bimonthly-era events, with their windows); the
# ACWD well records `WL 4S1W27D008.csv` (east of the fault) and
# `WL 4S1W29A006.csv` (west), exported from ACWD's .xlsx with day-first dates.
#
# **The wells (ACWD, Eileen, email 2026-10-02).** Levels in feet (converted to
# m here) above NGVD29. Screens: 27D008 (east, "above Hayward Fault") 60 to 110
# ft below ground, 29A006 (west, "below Hayward Fault") 67 to 127 ft. The water
# table sits ~9 m (east) and ~14 m (west) below the casing, so both wells read
# the head in a screened interval 18 to 39 m deep, below the water table, not
# the water table itself. ACWD's other wells near Tule Pond have semiannual
# readings only (not requested).

# %%
# [2]
import os
import sys

REPO = "/Users/dlindsay/tools/internal/TulePond_DISP"
sys.path.insert(0, REPO)
import insar_utils as utils
from TulePond_filepaths import common_paths, paths_out

import numpy as np                               # Matrix calculations
import pandas as pd                              # Tables
import matplotlib.pyplot as plt

# %% [markdown]
# ## [3] Paths and parameters

# %%
# [4]
out_dir = os.path.join(common_paths["out_dir"], "drafts_10x_hydrology")       # experimental products, apart from the paper's
fig_dir = common_paths["fig_dir"]
os.makedirs(out_dir, exist_ok=True)
rain_file = os.path.join(paths_out["10_rain_seasonality"], "tulepond_daily_precip.csv")
events_file = os.path.join(paths_out["10_rain_seasonality"], "event_table.csv")
well_dir = os.path.join(common_paths["inputs_dir"], "ACWD_water_well_data")
wells = {"east": ("WL 4S1W27D008.csv", "1990-10-02"),     # file, first date in the context file (a check on the date parsing)
         "west": ("WL 4S1W29A006.csv", "1991-10-09")}
ft_to_m = 0.3048            # the ACWD levels are in feet above NGVD29 (ACWD, 2026-10-02)
max_gap_days = 42           # well levels are interpolated across gaps up to 6 weeks, left missing beyond
storm_days, storm_mm = 3, 40.0                  # a storm: 3-day rain total of at least 40 mm (11's rule C)
rule_c_z, rule_c_x = [20, 30, 40, 50], [7, 14, 21, 30]   # 11's rule C search: storm size (mm) x lag (d)
rain_window_days = 30       # "rate": rain in the previous 30 days
well_rate_days = 28         # "rate": the well's change over the previous 4 weeks
n_draws = 2000              # onset draws per event, uniform in its window
n_null = 4000               # null realisations
seed = 20261002
okabe = dict(insar="#D55E00", aa="#0B2D6B", null="0.75", east="#0072B2", west="#E69F00")   # no blue/green pairs

# %% [markdown]
# ## [5] Load: the rain, the event windows, the two wells
#
# The well CSVs carry day-first dates (02/10/1990 = 2 Oct 1990). Duplicate dates
# are averaged and empty levels dropped. Each record is checked against the first
# date in the context file, so a day/month swap fails loudly.

# %%
# [6]
rain = pd.read_csv(rain_file, index_col="date", parse_dates=True)
assert rain.index.is_monotonic_increasing and (np.diff(rain.index.values).astype("timedelta64[D]").astype(int) == 1).all(), "rain is not daily"
events = pd.read_csv(events_file, parse_dates=["window_lo", "window_hi"])
print(f"rain {rain.index[0].date()} to {rain.index[-1].date()}; {len(events)} events "
      f"({(events.source == 'InSAR').sum()} InSAR, {(events.source != 'InSAR').sum()} array)")

levels = {}
for side, (name, first_expected) in wells.items():
    raw = pd.read_csv(os.path.join(well_dir, name), encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw.MeasurementDate, format="%d/%m/%Y")
    raw = raw.dropna(subset=["H2OElevation"])
    series = raw.groupby("date").H2OElevation.mean() * ft_to_m          # m, duplicate dates averaged
    assert str(series.index[0].date()) == first_expected, f"{side}: first date {series.index[0].date()}, expected {first_expected}"
    gaps = np.diff(series.index.values).astype("timedelta64[D]").astype(int)
    print(f"{side:4s} {raw.WellName.iloc[0]}: {len(series)} readings {series.index[0].date()} to {series.index[-1].date()}, "
          f"median spacing {np.median(gaps):.0f} d, longest gap {gaps.max()} d; level {series.min():.1f} to {series.max():.1f} m")
    # daily, linear in time across gaps up to max_gap_days, missing beyond
    days = pd.date_range(series.index[0], series.index[-1], freq="D")
    daily = series.reindex(series.index.union(days)).interpolate("time").reindex(days)
    # blank every day whose two neighbouring readings are more than max_gap_days apart
    after = np.searchsorted(series.index.values, days.values, side="left")
    before = np.clip(after - 1, 0, len(series) - 1)
    after = np.clip(after, 0, len(series) - 1)
    span = (series.index.values[after] - series.index.values[before]).astype("timedelta64[D]").astype(int)
    levels[side] = daily.where(span <= max_gap_days)

# %% [markdown]
# ## [7] The quantities at every day
#
# Rain (daily composite of notebook 10): the water-year rain to date; the days
# since the last storm; the rain in the previous `rain_window_days`.
# Wells: the rise since the autumn low (the lowest level from 1 Aug before the
# water year up to that day), the days since that low, and the change over the
# previous `well_rate_days`. Rises are measured within each water year, so the
# east well's long-term climb (Dani's context, section 4.10) does not enter.

# %%
# [8]
def days_since(flag):
    """Days since the last True in a daily boolean Series (NaN before the first)."""
    day_number = pd.Series(np.arange(len(flag)), index=flag.index, dtype=float)
    return day_number - day_number.where(flag).ffill()


rain_mm = rain.prcp_mm.fillna(0.0)
storm_total = rain_mm.rolling(storm_days, min_periods=1).sum()
daily = pd.DataFrame(index=rain.index)
daily["rain_wy_to_date"] = rain_mm.groupby(rain.wy).cumsum()
daily["days_since_storm"] = days_since(storm_total >= storm_mm)
daily["rain_previous"] = rain_mm.rolling(rain_window_days, min_periods=1).sum()

for side, level in levels.items():
    level = level.reindex(daily.index)
    rise, since_low = pd.Series(np.nan, index=daily.index), pd.Series(np.nan, index=daily.index)
    for wy in np.unique(rain.wy):
        start, end = pd.Timestamp(f"{wy - 1}-08-01"), pd.Timestamp(f"{wy}-09-30")
        this = level[start:end]
        if this.notna().sum() < 100:
            continue
        running_low = this.cummin()
        low_day = this.expanding().apply(lambda x: np.nanargmin(x) if np.isfinite(x).any() else np.nan, raw=True)
        in_wy = (this.index >= pd.Timestamp(f"{wy - 1}-10-01"))
        rise[this.index[in_wy]] = (this - running_low)[in_wy].values
        since_low[this.index[in_wy]] = (np.arange(len(this)) - low_day.values)[in_wy]
    daily[f"{side}_rise"] = rise
    daily[f"{side}_days_since_low"] = since_low
    daily[f"{side}_rate"] = (level - level.shift(well_rate_days)) / (well_rate_days / 7.0)     # m per week
    daily[f"{side}_level"] = level
    daily[f"{side}_anomaly"] = level - level.groupby(rain.wy).transform("mean")               # level minus its water-year mean
daily["head_difference"] = daily.east_level - daily.west_level
print(daily.describe().T[["count", "mean", "min", "max"]].round(2).to_string())

# %% [markdown]
# ## [9] Onset draws, the null, and the test
#
# For each event: `n_draws` days drawn uniformly in its window. A realisation
# takes one draw per event; the event average of a quantity is the statistic.
# The null replaces each drawn day by the same day of the water year in a random
# water year with no event of either catalogue, inside the record of that
# quantity ("other years": is the event's time of year unusual in this respect?).
# A second null keeps each event's OWN water year and draws the day from the
# season the events occur in (all onset draws pooled): "own year" asks whether
# the timing within the year matters, or only the kind of year. Two-sided p =
# twice the smaller tail of the null against the median observed statistic.

# %%
# [10]
rng = np.random.default_rng(seed)
day_index = {d: i for i, d in enumerate(daily.index)}
wy_start = {wy: pd.Timestamp(f"{wy - 1}-10-01") for wy in np.unique(rain.wy)}
event_wys = set(rain.wy[events.window_lo].values) | set(rain.wy[events.window_hi].values)
draw_days = np.array([rng.integers(day_index[e.window_lo], day_index[e.window_hi] + 1, n_draws) for e in events.itertuples()])   # events x draws
draw_dowy = rain.dowy.values[draw_days]


def null_days(column):
    """events x n_null days: the drawn day of the water year, in a random event-free water year where `column` exists."""
    valid_wy = sorted(wy for wy in wy_start if wy not in event_wys and daily.loc[wy_start[wy]:wy_start[wy] + pd.Timedelta(days=364), column].notna().mean() > 0.9)
    assert len(valid_wy) >= 10, f"{column}: only {len(valid_wy)} event-free water years with data"
    picked_wy = rng.choice(valid_wy, size=(len(events), n_null))
    picked_dowy = draw_dowy[np.arange(len(events))[:, None], rng.integers(0, n_draws, (len(events), n_null))]
    start_index = np.vectorize(lambda w: day_index.get(wy_start[w], -10 ** 6))(picked_wy)
    days = start_index + picked_dowy - 1
    days = np.where((days >= 0) & (days < len(daily)), days, -1)
    return days, len(valid_wy)


all_draw_dowy = draw_dowy.ravel()                                        # the season the events occur in: every onset draw, pooled


def null_days_same_year():
    """events x n_null days: each event's OWN water year, on a day of the water year drawn from all the events'
    onset draws (the season they occur in). Keeps the year's wetness; scrambles the timing within it."""
    own_wy = rain.wy.values[draw_days[:, 0]]
    start_index = np.array([day_index[wy_start[w]] for w in own_wy])
    days = start_index[:, None] + rng.choice(all_draw_dowy, size=(len(events), n_null)) - 1
    return np.where((days >= 0) & (days < len(daily)), days, -1)


def test(column):
    """Per-event values over the draws, the observed and null event averages, and the two-sided p."""
    values = daily[column].values
    per_event = values[draw_days]                                         # events x draws
    covered = np.isfinite(per_event).mean(axis=1) > 0.9                   # events inside this record
    observed = np.nanmean(per_event[covered][:, rng.integers(0, n_draws, n_null)], axis=0)    # realisations
    days, n_wy = null_days(column)
    null_values = np.where(days >= 0, values[np.clip(days, 0, None)], np.nan)[covered]
    null_mean = np.nanmean(null_values, axis=0)
    obs = float(np.median(observed))
    p_two = float(min(1.0, 2 * min(np.mean(null_mean >= obs), np.mean(null_mean <= obs))))
    days_b = null_days_same_year()
    null_b = np.nanmean(np.where(days_b >= 0, values[np.clip(days_b, 0, None)], np.nan)[covered], axis=0)
    p_same_year = float(min(1.0, 2 * min(np.mean(null_b >= obs), np.mean(null_b <= obs))))
    return dict(column=column, per_event=per_event, covered=covered, observed=observed, null_values=null_values,
                null_mean=null_mean, obs=obs, p=p_two, n_events=int(covered.sum()), n_null_wy=n_wy,
                null_same_year=null_b, p_same_year=p_same_year)


tests = {}
for column in ("rain_wy_to_date", "days_since_storm", "rain_previous",
               "east_anomaly", "east_rise", "east_rate", "west_anomaly", "west_rise", "west_rate",
               "east_days_since_low", "west_days_since_low", "head_difference"):
    tests[column] = test(column)
    t = tests[column]
    print(f"{column:22s} {t['n_events']:2d} events: event mean {t['obs']:8.2f} | same season, {t['n_null_wy']} event-free WYs: "
          f"{np.median(t['null_mean']):8.2f} (5-95% {np.percentile(t['null_mean'], 5):.2f} to {np.percentile(t['null_mean'], 95):.2f}) | p = {t['p']:.3f} "
          f"| own year, random time in season: {np.median(t['null_same_year']):8.2f}, p = {t['p_same_year']:.3f}")

# %% [markdown]
# ## [11] The storm test (11's rule C), with its search paid for
#
# How often do storms happen? For each water year: the number of storms and the
# fraction of Dec-Mar days within 21 d after one. Then the test: an event "hits"
# if its (drawn) onset falls within X days after a storm of at least Z mm. The
# expected number of hits is compared with the null at the same time of year.
# Rule C picked the best (Z, X) of 16; the CORRECTED p asks how often the null,
# searching the same 16 combinations, does at least as well as the best one.

# %%
# [12]
storms_per_wy = (storm_total >= storm_mm).astype(int).groupby(rain.wy).apply(lambda s: int(((s.diff() == 1) | ((s == 1) & (s.shift(1).isna()))).sum()))
wet = rain.dowy.between(62, 182)                                          # Dec 1 to Mar 31
covered_21 = (days_since(storm_total >= storm_mm) <= 21)[wet].groupby(rain.wy[wet]).mean()
print(f"storms (3-day >= {storm_mm:g} mm) per water year: median {storms_per_wy.median():.0f}, range {storms_per_wy.min()}-{storms_per_wy.max()}; "
      f"Dec-Mar days within 21 d after one: median {100 * covered_21.median():.0f} % (range {100 * covered_21.min():.0f}-{100 * covered_21.max():.0f} %)")

grid = [(z, x) for z in rule_c_z for x in rule_c_x]
null_d, _ = null_days("days_since_storm")
hits_obs, hits_null = {}, {}
for z, x in grid:
    since = days_since(storm_total >= z).values
    hits_obs[(z, x)] = float(np.mean(np.sum(since[draw_days] <= x, axis=0)))           # expected hits over the draws
    hits_null[(z, x)] = np.sum(np.where(null_d >= 0, since[np.clip(null_d, 0, None)], np.inf) <= x, axis=0)
p_raw = {g: float(np.mean(hits_null[g] >= hits_obs[g])) for g in grid}
null_b_days = null_days_same_year()
hits_null_b = {}
for z, x in grid:
    since = days_since(storm_total >= z).values
    hits_null_b[(z, x)] = np.sum(np.where(null_b_days >= 0, since[np.clip(null_b_days, 0, None)], np.inf) <= x, axis=0)
p_raw_b = {g: float(np.mean(hits_null_b[g] >= hits_obs[g])) for g in grid}
best = min(p_raw, key=p_raw.get)
# the corrected p: for every null realisation, its own best p over the grid (ranked against the same null)
null_matrix = np.array([hits_null[g] for g in grid])                     # grid x n_null
p_null_each = np.empty((n_null, len(grid)))                              # each null realisation's p in each combination
for k in range(len(grid)):
    ordered = np.sort(null_matrix[k])
    p_null_each[:, k] = (n_null - np.searchsorted(ordered, null_matrix[k], side="left")) / n_null     # = mean(null >= value)
p_corrected = float(np.mean(p_null_each.min(axis=1) <= p_raw[best]))
for z, x in grid:
    print(f"  Z {z:2d} mm, X {x:2d} d: hits {hits_obs[(z, x)]:5.2f} of {len(events)} vs same season {np.mean(hits_null[(z, x)]):5.2f} "
          f"(95% {np.percentile(hits_null[(z, x)], 95):.0f}); p {p_raw[(z, x)]:.3f}")
print(f"best combination Z {best[0]} mm, X {best[1]} d: p {p_raw[best]:.3f} on its own, {p_corrected:.3f} once the search over "
      f"{len(grid)} combinations is paid for")
# the same, against the own-year null (is it the timing within the year, or just wetter years?)
null_matrix_b = np.array([hits_null_b[g] for g in grid])
p_null_each_b = np.empty((n_null, len(grid)))
for k in range(len(grid)):
    ordered = np.sort(null_matrix_b[k])
    p_null_each_b[:, k] = (n_null - np.searchsorted(ordered, null_matrix_b[k], side="left")) / n_null
best_b = min(p_raw_b, key=p_raw_b.get)
p_corrected_b = float(np.mean(p_null_each_b.min(axis=1) <= p_raw_b[best_b]))
for z, x in grid:
    print(f"  own year: Z {z:2d} mm, X {x:2d} d: hits {hits_obs[(z, x)]:5.2f} vs random time in season {np.mean(hits_null_b[(z, x)]):5.2f} "
          f"(95% {np.percentile(hits_null_b[(z, x)], 95):.0f}); p {p_raw_b[(z, x)]:.3f}")
print(f"own-year null: best Z {best_b[0]} mm, X {best_b[1]} d: p {p_raw_b[best_b]:.3f}, search-corrected {p_corrected_b:.3f}")
subset_rows = []
for label, subset in (("InSAR only", events.source.values == "InSAR"), ("array only", events.source.values != "InSAR")):
    since = days_since(storm_total >= best[0]).values
    obs_sub = float(np.mean(np.sum(since[draw_days[subset]] <= best[1], axis=0)))
    null_sub = np.sum(np.where(null_b_days[subset] >= 0, since[np.clip(null_b_days[subset], 0, None)], np.inf) <= best[1], axis=0)
    print(f"  {label} (Z {best[0]}, X {best[1]}): hits {obs_sub:.2f} of {subset.sum()} vs own-year random {null_sub.mean():.2f}, "
          f"p {np.mean(null_sub >= obs_sub):.3f}")
    subset_rows.append(dict(null=f"{label}, own year", n=int(subset.sum()), hits=obs_sub, expected=float(null_sub.mean()),
                            p_best=float(np.mean(null_sub >= obs_sub)), p_search_paid=np.nan))
storm_summary = pd.DataFrame([dict(null="same day of year, event-free years", n=len(events), hits=hits_obs[best], expected=float(np.mean(hits_null[best])),
                                   p_best=p_raw[best], p_search_paid=p_corrected),
                              dict(null="own year, random day in the season", n=len(events), hits=hits_obs[best], expected=float(np.mean(hits_null_b[best])),
                                   p_best=p_raw_b[best], p_search_paid=p_corrected_b)] + subset_rows)
storm_summary["Z_mm"], storm_summary["X_d"] = best[0], best[1]
storm_summary["storms_per_wy_median"], storm_summary["storms_per_wy_min"], storm_summary["storms_per_wy_max"] = (
    float(storms_per_wy.median()), int(storms_per_wy.min()), int(storms_per_wy.max()))
storm_summary["wet_days_within_21d_pct"] = float(100 * covered_21.median())
storm_summary.to_csv(os.path.join(out_dir, "storm_test.csv"), index=False)

# %% [markdown]
# ## [13] Figure (working): each quantity through the water year, event years against the rest
#
# Every panel has the same axes: x = day of the water year (Oct 1 = 1), y = the
# quantity. Grey = the water years WITHOUT an event (10 to 90 % band and
# median): the general population. Coloured = each event's own water year
# (vermillion InSAR, navy array), with the onset window drawn thick on that
# curve. So the question in each panel is: at the time the events happen, do
# the event years sit outside the band? The p in the corner is the test of [10]
# (event mean against the same time of year in event-free years). Rows: rain,
# east well, west well; columns: Dani's three questions.

# %%
# [14]
utils.mpl_paper_style(base=7)
panels = [["rain_wy_to_date", "days_since_storm", "rain_previous"],
          ["east_anomaly", "east_rise", "east_rate"],
          ["west_anomaly", "west_rise", "west_rate"]]
titles = [["Is it the total over the water year?", "Is it the time after a storm?", "Is it the rate of rainfall?"],
          ["Threshold level?", "Amount of recharge?", "Recharge rate?"],
          ["Threshold level?", "Amount of recharge?", "Recharge rate?"]]
ylabels = {"rain_wy_to_date": "Rain since Oct 1 (mm)", "days_since_storm": f"Days since a {storm_mm:g} mm storm",
           "rain_previous": f"Rain in the previous {rain_window_days} d (mm)",
           "east_anomaly": "East: level minus WY mean (m)", "east_rise": "East: rise since autumn low (m)",
           "east_rate": "East: 4-week change (m/week)", "west_anomaly": "West: level minus WY mean (m)",
           "west_rise": "West: rise since autumn low (m)", "west_rate": "West: 4-week change (m/week)"}
all_wy = np.unique(rain.wy)
event_free_wy = [wy for wy in all_wy if wy not in event_wys]
month_ticks = [1, 32, 62, 93, 124, 152, 183, 213, 244, 274]
month_names = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul"]
fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.4), constrained_layout=True)
for i, row in enumerate(panels):
    for j, column in enumerate(row):
        ax = axes[i, j]
        by_year = daily[column].to_frame("v").assign(wy=rain.wy.values, dowy=rain.dowy.values).pivot(index="dowy", columns="wy", values="v")
        population = by_year[[wy for wy in event_free_wy if wy in by_year.columns and by_year[wy].notna().mean() > 0.9]]
        ax.fill_between(population.index, population.quantile(0.1, axis=1), population.quantile(0.9, axis=1), color="0.82", lw=0, label="event-free years, 10 to 90 %")
        ax.plot(population.index, population.median(axis=1), color="0.45", lw=1.0, label="event-free years, median")
        for event in events.itertuples():
            wy = int(rain.wy[event.window_lo])
            if wy not in by_year.columns or by_year[wy].notna().mean() < 0.9:
                continue
            col = okabe["insar"] if event.source == "InSAR" else okabe["aa"]
            curve = by_year[wy]
            ax.plot(curve.index, curve.values, color=col, lw=0.6, alpha=0.7)
            lo, hi = int(rain.dowy[event.window_lo]), int(rain.dowy[event.window_hi])
            in_window = (curve.index >= lo) & (curve.index <= hi)
            ax.plot(curve.index[in_window], curve.values[in_window], color=col, lw=2.2, solid_capstyle="butt")
        ax.set_xlim(1, 290)
        ax.set_xticks(month_ticks)
        ax.set_xticklabels(month_names if i == 2 else [])
        ax.set_ylabel(ylabels[column])
        ax.set_title(titles[i][j], fontsize=7, loc="left")
        ax.text(0.98, 0.95, f"p {tests[column]['p']:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=6)
        if column == "days_since_storm":                     # a different graph: one row per event, its storms as ticks, its window as a bar
            ax.clear()
            storm_end = (storm_total >= storm_mm) & ~(storm_total >= storm_mm).shift(1, fill_value=False)   # the first day each storm reaches storm_mm
            for k, event in enumerate(events.sort_values("window_lo").itertuples()):
                wy = int(rain.wy[event.window_lo])
                col = okabe["insar"] if event.source == "InSAR" else okabe["aa"]
                year_storms = rain.dowy[storm_end & (rain.wy == wy)].values
                ax.vlines(year_storms, k - 0.4, k + 0.4, color="0.3", lw=0.8)
                lo, hi = int(rain.dowy[event.window_lo]), int(rain.dowy[event.window_hi])
                ax.fill_between([lo - best[1], lo], k - 0.4, k + 0.4, color=col, alpha=0.15, lw=0)      # the X days before the window
                ax.plot([lo, hi], [k, k], color=col, lw=4, solid_capstyle="butt")
                ax.text(292, k, f"WY{wy}", fontsize=5, va="center")
            ax.set_ylim(-0.7, len(events) + 2.2)                 # room for the text above the rows
            ax.set_yticks([])
            ax.set_ylabel(f"Events (ticks: {storm_days}-day storms of {storm_mm:g} mm or more)")
            ax.set_xlim(1, 290)
            ax.set_xticks(month_ticks)
            ax.set_xticklabels(month_names if i == 2 else [])
            ax.set_title(titles[i][j], fontsize=7, loc="left")
            ax.text(0.98, 0.97, f"onset within {best[1]} d of a storm (pale box): {hits_obs[best]:.1f} of {len(events)}\n"
                    f"random time in the same year and season: {np.mean(hits_null_b[best]):.1f}\n"
                    f"p {p_raw_b[best]:.3f} ({p_corrected_b:.2f} with the search paid for)",
                    transform=ax.transAxes, ha="right", va="top", fontsize=5.5)
handles = [plt.Line2D([], [], color="0.82", lw=6), plt.Line2D([], [], color="0.45", lw=1),
           plt.Line2D([], [], color=okabe["insar"], lw=2.2), plt.Line2D([], [], color=okabe["aa"], lw=2.2)]
fig.legend(handles, ["event-free years, 10 to 90 %", "event-free years, median", "InSAR event year (thick: onset window)",
                     "array event year (thick: onset window)"], fontsize=6, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
f_fig = os.path.join(fig_dir, "qc10x_hydrology_tests.png")
fig.savefig(f_fig, dpi=300, bbox_inches="tight")
print("Saved", f_fig)

# %% [markdown]
# ## [15] Run summary (for the 30-second read)

# %%
# [16]
summary = pd.DataFrame([dict(quantity=c, n_events=t["n_events"], event_mean=round(t["obs"], 2),
                             other_years_median=round(float(np.median(t["null_mean"])), 2), p_other_years=round(t["p"], 3),
                             own_year_median=round(float(np.median(t["null_same_year"])), 2), p_own_year=round(t["p_same_year"], 3))
                        for c, t in tests.items()])
summary.to_csv(os.path.join(out_dir, "hydrology_tests.csv"), index=False)
print(summary.to_string(index=False))
print(f"storm test: best Z {best[0]} mm X {best[1]} d, p {p_raw[best]:.3f}, search-corrected {p_corrected:.3f}")
