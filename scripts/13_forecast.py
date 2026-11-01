# %% [markdown]
# # 13 The next-event forecast
#
# ```
# Author: Danielle Lindsay
# Environment: opera_disp_feb13. Run time: about 5 minutes (4000 timing draws x 5 bootstrap refits per model, record and
# aperiodicity branch).
# ```
#
# **What this notebook does.** It computes the repeat intervals, the renewal forecasts and the prediction statement
# from the event catalogues of 04 and 11 and the rainfall products of 12_rain_seasonality.
#
# **Terms.** An *onset* is the start of an SSE. It is known only to lie within its *onset window*, from the last
# acquisition (or array survey) before the SSE step to the first one after it. The *repeat interval* is the time
# between the onsets of two consecutive SSEs of the same record.
#
# 1. **Repeat intervals** (section 6; computed only in this notebook). For each of `n_timing` draws of the onsets,
#    uniform within their windows (no midpoints), the sample mean and sample SD of the repeat intervals of a record are
#    computed; both are averaged over the draws, and COV = SD / mean. "+/-" is 1 SD of the intervals, not a standard
#    error. The records are the InSAR record (5 events, 4 intervals), the bimonthly-era HRKT array record without the
#    interval that spans Loma Prieta (`aa_dense_nolp`, 9 events, 7 intervals) and the combined record (`merged`, 14
#    events, 11 intervals; no interval across the unobserved 2006 to 2017 gap and none across Loma Prieta). The array
#    record with the Loma Prieta interval (`aa_dense_withlp`, 8 intervals) is kept for its interval statistics only (the
#    onset-window table of the supplement). Fig 3C draws the same statistics.
# 2. **Renewal forecast** of the next onset (sections 4 to 6). The repeat intervals of each draw are fitted by the
#    Brownian passage time (BPT) model, with its mean and aperiodicity fitted by maximum likelihood (Ellsworth et al.
#    1999); the aperiodicity is close to, but not the same as, the sample COV of step 1, and both are reported.
#    Lognormal and Weibull fits are the check that the window does not depend on the model. Each fit is refitted to synthetic intervals (a parametric bootstrap, as Ellsworth et al. 1999
#    do), and the forecast is conditioned on no event through the last observation, `forecast_last_obs_date`
#    (TulePond_filepaths, the last descending acquisition of the CSLC + Dolphin extension). The InSAR record of the
#    paper ends at the data end, the last acquisition on both tracks (read from `fp_ts.csv` of 03).
# 3. **The Working Group aperiodicities** (section 7). WGCEP 1999 (USGS OFR 99-517, printed p16) used $\alpha$ = 0.3,
#    0.5 and 0.7 as weighted branches rather than a fitted $\alpha$. The BPT forecast of each record is computed with
#    $\alpha$ fixed at each branch (only the mean refitted), and the branches are mixed with the weights 0.2, 0.6 and
#    0.2, the three-way branch convention of the report (App. C, printed p31). The falsification dates are the dates
#    at which the conditional probability of no event under the InSAR BPT falls to 5 % and to 1 %.
# 4. **The inputs of Fig 3** (section 8). The interval ranges, interval statistics and BPT forecasts of the records
#    drawn in Fig 3C and 3D are written to `fig3_inputs.npz`. Notebook 14 draws Fig 3 from this file and the products
#    of 11 and 12.
# 5. **The prediction statement** (section 9). It combines the recurrence windows with the calendar window of the
#    onsets from `rain_conditioning_summary.csv` of 12_rain_seasonality, where the rainfall-conditioning tests are
#    computed.
#
# **Loma Prieta.** The array and combined forecasts leave out the interval spanning the 1989 earthquake, since creep
# on the southern Hayward fault was suppressed after Loma Prieta (Lienkaemper et al. 1997).
#
# **Not done.** The exponential (Poisson) model, which Ellsworth et al. (1999) reject for recurrent events; the
# Gaussian, which gives probability to negative intervals (Matthews et al. 2002, p2234); and model ranking by
# goodness of fit, which Matthews et al. (2002, Appendix 1) show cannot tell the two-parameter models apart.
#
# **Wet season.** October to April, `TulePond_filepaths.wet_season_months` = (first month, last month), shared with the
# onset-season count of 12_rain_seasonality; this notebook checks it against the months saved by 12_rain_seasonality.
# `forecast_wet_season` is that season in the first water year whose season ends after the last observation (for
# example WY2027, as in every "P(WY2027 wet season)").
#
# **Quantiles beyond the forecast grid.** The forecast curves are evaluated on a daily grid (`grid`, derived from the
# data in section 3). A mixture curve that does not reach a quantile on the grid has no date for it: `quantile_date`
# returns NaN, and the quantile is written and registered as '> <grid end>', never as the grid end itself. Each
# forecast row also carries `p_by_grid_end`, the probability the curve reaches by the grid end.
#
# **Inputs**: 03 `fp_ts.csv` (the data end); 04 `section_lengths.csv` and the 90 % band-stack steps registered by 04
# (`04_band90_step_<yr>_mm` and their CIs, read back from `registry.csv` for the latest run of 04), for the location
# and magnitude of the prediction statement; 04 `creep_events.csv` (InSAR onset windows from `onset_lo_utc`, the UTC
# time of the last acquisition before the step, to `onset_hi_utc`, that of the first acquisition after it); 11
# `hrkt_aa_events.csv` (array onset windows `onset_lo` to `onset_hi`, and `onset_mid`, the window midpoint, used only
# for sorting and for the reach of the forecast grid); 12 `fig3_rain_panel.npz` (the wet-season months of 12 and the
# end of the bimonthly surveys at HRKT) and `rain_conditioning_summary.csv`.
#
# **Outputs** (`outputs/13_forecast/`): `recurrence_forecast_models.csv` (every record and model, with the interval
# statistics, quantiles and P(wet season)); `recurrence_intervals.csv` (the interval statistics of each record);
# `recurrence_forecast_wg99.csv` (every record and aperiodicity branch, and the mixtures); `fig3_inputs.npz` (the
# inputs of Fig 3 from this notebook, section 8); `prediction_statement.txt`; the stats sidecar
# `13_forecast_stats.txt`; registry rows `13_*` and, per record, `13_<rec>_*` (section 10).
#
# **Figures**: the three renewal models (`figS_renewal_single_fits`, supplement). The paper's Fig 3
# (`fig3_history_forecast_ntiming4000_nboot5_issue<date>`) is drawn by notebook 14 from `fig3_inputs.npz`.
#
# **What to check.** The InSAR BPT 5-95 % window (`13_insar_bpt_q05`, `13_insar_bpt_q95`) against the coming wet
# season, with the fitted aperiodicity and with the Working Group aperiodicities
# (`13_insar_bpt_wg99_p_wet_wy<N>`), and the same for the array and merged records (sections 6 and 7). Every number is
# registered in section 10.

# %% [markdown]
# ## 1. Imports and settings
#
# The forecast settings, the Working Group aperiodicity branches, the wet season and the input files.
#
# **Random numbers.** Every generator is a fresh `np.random.default_rng(seed)`. The forecast of section 6 uses one
# generator, which feeds, in order, the interval draws of each record and then its three model forecasts. Each
# aperiodicity branch of section 7 has a generator of its own, so the three branches use the same onset and bootstrap
# draws and differ only in $\alpha$.

# %%
import calendar
import os

import insar_utils as utils
from TulePond_filepaths import common_paths, paths_out, repo_dir, forecast_issue_date, forecast_last_obs_date
from TulePond_filepaths import wet_season_months as wet_season_first_last, aa_dense_window_days, loma_prieta_yr, paper_band_frac

import numpy as np                               # Matrix calculations
import pandas as pd                              # Tables
from scipy import stats                          # the renewal distributions
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D             # legend handles of the single-fits figure

# %%
notebook = "13_forecast"
out_dir = os.path.join(common_paths["out_dir"], notebook)
fig_dir = common_paths["fig_dir"]
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)


# the forecast
loma_prieta = loma_prieta_yr   # 17 Oct 1989 (TulePond_filepaths, as in 11 and 12)
n_timing = 4000                # timing draws: onsets drawn uniformly in their windows
n_boot = 5                     # bootstrap refits per timing draw (per catalogue and model): 4000 x 5 = 20000 curves per mean
#                                forecast curve
seed = 7                      # every generator starts from it (section 1)
# The forecast is conditioned on no event through the last observation, forecast_last_obs_date
# (TulePond_filepaths: the last descending acquisition of the CSLC + Dolphin extension). The InSAR record of
# the paper ends earlier, at the data end (the end of the fault-parallel series, read from the data below).
# forecast_issue_date is the date the statement was first written (fixed, so that the file names are stable);
# nothing is stated for the months after the last observation.
models = ["bpt", "lognormal", "weibull"]          # the BPT is the forecast; lognormal and Weibull are the check
resolvable_days = aa_dense_window_days   # array windows longer than this (d) do not constrain the season (TulePond_filepaths: the bimonthly era of 11)
min_interval_yr = 0.05         # bootstrap intervals shorter than this are clipped to it (counted and reported)
max_degenerate_fraction = 0.01   # at most this share of the forecast curves may come from a refit with zero survival
grid_span_means = 4.0          # the forecast grid runs this many of the longest mean interval past the last observation,
#                                beyond the 95 % quantile of every forecast (p_by_grid_end is written and checked); its
#                                start and end are derived in section 3

# the Working Group aperiodicities: WGCEP (1999), "Earthquake Probabilities in the San Francisco Bay Region: 2000 to
# 2030 - A Summary of Findings", USGS Open-File Report 99-517, printed p16: "For the aperiodicity, WG99 used three
# (weighted) values: 0.3, 0.5, and 0.7" (representative of the 37 sequences of Ellsworth et al.). The alpha weights
# are not printed in the report; the weights below are its stated convention for a three-way branch point (App. C,
# printed p31: "branch weights of 0.2, 0.6, and 0.2 ... corresponding roughly to a confidence interval of one
# standard deviation"). WG99 mixes branch probabilities by weighted mean (printed p18), so the branch cdfs are mixed
# with the weights.
wg99_alpha_branches = (0.3, 0.5, 0.7)
wg99_alpha_weights = (0.2, 0.6, 0.2)
assert abs(sum(wg99_alpha_weights) - 1) < 1e-12
# falsification: the dates by which the conditional probability of no event under the InSAR BPT (its survival,
# 1 - the conditional cdf) falls to these levels; no event by then rejects that model at 1 - level
falsification_levels = (0.05, 0.01)

# the wet season, (first month, last month), October to April: the same definition as the onset count of
# 12_rain_seasonality, TulePond_filepaths.wet_season_months. forecast_wet_season (its dates in the coming water year)
# is set once the last observation is read, in section 3.
wet_season_source = "TulePond_filepaths.wet_season_months"
assert len(wet_season_first_last) == 2 and all(int(m) == m and 1 <= m <= 12 for m in wet_season_first_last), \
    f"wet_season_months must be (first month, last month), got {wet_season_first_last}"
first_wet_month, last_wet_month = (int(m) for m in wet_season_first_last)
wet_season_months = []                     # the months of the season in order, wrapping across the new year
for k in range((last_wet_month - first_wet_month) % 12 + 1):
    wet_season_months.append((first_wet_month - 1 + k) % 12 + 1)
wet_months_text = f"{calendar.month_abbr[first_wet_month]}-{calendar.month_abbr[last_wet_month]}"     # e.g. 'Oct-Apr'

# location and magnitude of the prediction statement: the measured extents of 04 and the band stack of the paper
magnitude_band_pct = int(round(100 * paper_band_frac))   # band_main of 04 (TulePond_filepaths paper_band_frac 0.90): the peak slip zone on which the steps are quoted
magnitude_band = f"band{magnitude_band_pct}"      # its name in 04 (section_lengths.csv row tulepond_band90, keys 04_band90_*)

fp_ts_file = os.path.join(paths_out["03_creep_timeseries"], "fp_ts.csv")
section_lengths_file = os.path.join(paths_out["04_event_detection"], "section_lengths.csv")
events_file = os.path.join(paths_out["04_event_detection"], "creep_events.csv")
aa_events_file = os.path.join(paths_out["11_alignment_arrays"], "hrkt_aa_events.csv")
rain_panel_file = os.path.join(paths_out["12_rain_seasonality"], "fig3_rain_panel.npz")
rain_summary_file = os.path.join(paths_out["12_rain_seasonality"], "rain_conditioning_summary.csv")

REG = utils.Registry(notebook, common_paths, repo_dir)
REG.inputs(fp_ts_file, section_lengths_file, events_file, aa_events_file, rain_panel_file, rain_summary_file)

# %% [markdown]
# ## 2. Functions: dates and water years
#
# Decimal years to calendar dates and back (as in notebook 12, section 2), and the wet season of a water year. Water
# year $N$ runs from 1 October of $N - 1$ to 30 September of $N$, so a wet-season month from October on belongs to the
# calendar year before $N$.

# %%
def decyear(dates):
    """Decimal year of each date or time (`insar_utils.decimal_years`).

    Parameters
    ----------
    dates : array-like of dates

    Returns
    -------
    pandas.Index
        Decimal years.
    """
    return pd.Index(utils.decimal_years(dates))


def decyear_to_date(years):
    """UTC calendar day that contains each time given as a decimal year (forecast dates, exact window ends).

    Parameters
    ----------
    years : float or array-like
        Decimal years.

    Returns
    -------
    pandas.DatetimeIndex
    """
    out = []
    for value in np.atleast_1d(np.asarray(years, float)):
        out.append(utils.decimal_year_to_utc_date(value))
    return pd.DatetimeIndex(out)


def ymd(year):
    """One decimal year as 'YYYY-MM-DD'.

    Parameters
    ----------
    year : float

    Returns
    -------
    str
    """
    return decyear_to_date(year)[0].strftime("%Y-%m-%d")


def wet_season_of(wy):
    """First and last day of the wet season (wet_season_months) of a water year.

    A month from October on belongs to the calendar year before the water year.

    Parameters
    ----------
    wy : int
        Water year.

    Returns
    -------
    first, last : str
        'YYYY-MM-DD'.
    """
    first = pd.Timestamp(year=wy - 1 if first_wet_month >= 10 else wy, month=first_wet_month, day=1)
    last = pd.Timestamp(year=wy - 1 if last_wet_month >= 10 else wy, month=last_wet_month, day=1) + pd.offsets.MonthEnd(0)
    return first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d")

# %% [markdown]
# ## 3. Load the event catalogues, the forecast dates and grid, and the rainfall summary of 12
#
# **Catalogues.** The InSAR events come from 04, with their exact windows: from the UTC time of the last acquisition
# before the step (`onset_lo_utc`) to that of the first acquisition after it (`onset_hi_utc`). The HRKT array events come from 11, all from the bimonthly era, since 11 runs the detector only
# there; their windows of `aa_dense_window_days` (130 d) or less resolve the season (asserted). The merged record holds
# the bimonthly-era array events and the InSAR events. The years between the last array event (2006) and the first
# InSAR event (2017) are treated as unobserved, because the arrays were surveyed only annually from mid-2007: no
# interval is formed across them, and the record restarts with the first InSAR detection.
#
# **Forecast grid.** The daily grid, with spacing $1/365.25$ yr, runs from
#
# $$t_\mathrm{start} = \lceil t_{\mathrm{lo},5} \rceil \quad \text{to} \quad
# t_\mathrm{end} = \lceil t_\mathrm{obs} + g\, \bar x_\mathrm{max} \rceil,$$
#
# where $t_{\mathrm{lo},5}$ is the start of the last InSAR window (decimal year), $\lceil\cdot\rceil$ rounds up to
# the next whole decimal year (1 January to within a day), $t_\mathrm{obs}$ is the last observation, $g$ =
# `grid_span_means` (4) and $\bar x_\mathrm{max}$ is the longest mean interval of the four records computed from
# point onsets (the window midpoints). This reaches well beyond the 95 % quantile of every forecast. Every conditional
# curve is zero until the last observation, which must lie on the grid (asserted).
#
# **Wet season of the forecast.** The first water year whose wet season ends after the last observation.
#
# `rain_conditioning_summary.csv` of 12_rain_seasonality gives part (3) of the prediction statement and the last
# Fremont day of its data line; they are read, not recomputed. The end of the bimonthly surveys at HRKT, quoted in the
# statement, is read from `fig3_rain_panel.npz` of 12.

# %%
creep_events = pd.read_csv(events_file)
insar_events = creep_events[creep_events.segment == "TULEPOND"].sort_values("onset_lo").reset_index(drop=True)
# the exact windows of 04 (UTC times of the bounding acquisitions); the grid dates are kept for the check against 04
insar_events["onset_lo_grid"] = insar_events.onset_lo
insar_events["onset_hi_grid"] = insar_events.onset_hi
insar_events["onset_lo"] = utils.decimal_years(insar_events.onset_lo_utc)
insar_events["onset_hi"] = utils.decimal_years(insar_events.onset_hi_utc)
insar_events["onset_mid"] = 0.5 * (insar_events.onset_lo + insar_events.onset_hi)     # as in 11: sort key and plot position only
aa_dense = pd.read_csv(aa_events_file)
assert (aa_dense.window_days <= resolvable_days).all(), f"11 returned an array window longer than {resolvable_days} d"
# every catalogue carries a `src` column; an interval is formed only between two events of the same source, so the
# merged catalogue does not bridge 2006 to 2017 (section 3)
insar_events = insar_events.assign(src="InSAR")
aa_dense = aa_dense.assign(src="AA")
merged_events = pd.concat([aa_dense[["onset_lo", "onset_hi", "onset_mid", "src"]], insar_events[["onset_lo", "onset_hi", "onset_mid", "src"]]])
merged_events = merged_events.sort_values("onset_mid").reset_index(drop=True)
# name -> (catalogue, whether to drop the interval spanning Loma Prieta)
catalogues = {"A_insar": (insar_events, False), "B_aa_dense": (aa_dense, False), "B_aa_dense_noLP": (aa_dense, True),
              "C_merged": (merged_events, True)}
# the key of each catalogue in the registry and the files: B_aa_dense keeps the Loma Prieta interval (withlp) and
# B_aa_dense_noLP leaves it out (nolp)
record_keys = {"A_insar": "insar", "B_aa_dense": "aa_dense_withlp", "B_aa_dense_noLP": "aa_dense_nolp", "C_merged": "merged"}
# the records that are forecast; B_aa_dense (with the Loma Prieta interval) gives its interval statistics only
forecast_records = ["A_insar", "B_aa_dense_noLP", "C_merged"]
n_insar_events = len(insar_events)
n_aa_events = len(aa_dense)
print(f"InSAR {n_insar_events} events; array bimonthly era {n_aa_events} events (windows <= {resolvable_days} d); "
      f"merged {len(merged_events)} events with 2006 to 2017 unobserved")

# the end of the InSAR record: the end of the fault-parallel series of 03, the last acquisition seen by both tracks
data_end_yr = float(pd.read_csv(fp_ts_file, index_col=0, nrows=1).columns.astype(float).max())
data_end = utils.decimal_year_to_day(data_end_yr).strftime("%Y-%m-%d")     # 03's columns are epochs at midnight, to 4 places
# the reference date of the forecast: the last observation with no event (later than the record end)
assert pd.Timestamp(forecast_last_obs_date) >= pd.Timestamp(data_end), \
    f"forecast_last_obs_date {forecast_last_obs_date} is before the end of the InSAR record {data_end}"
last_obs = forecast_last_obs_date
last_obs_day_yr = float(decyear(pd.to_datetime([last_obs]))[0])        # the reference date of the forecast, on the day
print(f"InSAR record ends {data_end} (the last acquisition on both tracks); forecast conditioned on no event through "
      f"{last_obs} (forecast_last_obs_date, the CSLC + Dolphin extension); statement first written {forecast_issue_date}")

# the wet season of the forecast: the first water year whose wet season ends after the last observation
forecast_wy = int(pd.Timestamp(last_obs).year + (1 if pd.Timestamp(last_obs).month >= 10 else 0))     # the water year of the last observation
if pd.Timestamp(wet_season_of(forecast_wy)[1]) < pd.Timestamp(last_obs):
    forecast_wy += 1
forecast_wet_season = wet_season_of(forecast_wy)
wet_label = f"WY{forecast_wy} wet season"            # every "P(WY<N> wet season)" and the registry keys _p_wet_wy<N>
# the same season in 12 and 13: the months in which 12 counted the onsets must be these
with np.load(rain_panel_file) as rain_panel_saved:
    assert "wet_season_months" in rain_panel_saved.files, f"{rain_panel_file} has no wet_season_months: rerun 12_rain_seasonality"
    rain_panel_months = [int(m) for m in rain_panel_saved["wet_season_months"]]
    dense_era_end = float(rain_panel_saved["dense_era_end"])      # the end of the bimonthly surveys at HRKT (12), for the prediction statement
assert rain_panel_months == wet_season_months, \
    f"12_rain_seasonality counted the onsets in months {rain_panel_months}, this notebook uses {wet_season_months}: rerun 12"
print(f"{wet_label}: {forecast_wet_season[0]} to {forecast_wet_season[1]} ({wet_months_text}, {wet_season_source}; same months as 12)")

# the forecast grid, daily (section 3); the longest mean interval uses point onsets (window midpoints), which set only
# the reach of the grid
longest_mean_yr = 0.0
for cat_name, (catalogue, drop_lp) in catalogues.items():
    point_intervals = np.diff(np.sort(catalogue.onset_mid.values))
    order = np.argsort(catalogue.onset_mid.values)
    keep_pairs = (catalogue.src.values[order][:-1] == catalogue.src.values[order][1:])
    if drop_lp:
        onsets_sorted = catalogue.onset_mid.values[order]
        keep_pairs &= ~((onsets_sorted[:-1] < loma_prieta) & (onsets_sorted[1:] > loma_prieta))
    longest_mean_yr = max(longest_mean_yr, float(point_intervals[keep_pairs].mean()))
grid_start = float(np.ceil(insar_events.onset_lo.max()))     # the first whole decimal year after the last onset window opened
grid_end = float(np.ceil(last_obs_day_yr + grid_span_means * longest_mean_yr))
grid = np.arange(grid_start, grid_end, 1 / 365.25)   # the time axis of the forecast curves (daily); a quantile beyond its end is '> end'
assert grid[0] <= last_obs_day_yr <= grid[-1], f"the last observation {last_obs} is not on the forecast grid {grid[0]:.2f}-{grid[-1]:.2f}"
print(f"forecast grid {grid[0]:.0f} to {grid_end:.0f} ({grid.size} days): {grid_span_means:g} x the longest mean interval "
      f"({longest_mean_yr:.2f} yr) past the last observation")

# the rainfall-conditioning summary of 12 (part (3) of the prediction statement, the data line)
rain_summary = pd.read_csv(rain_summary_file, dtype=str).set_index("name")
rain_summary_needed = ["n_insar_onsets", "onset_dowy_min", "onset_dowy_max", "onset_earliest", "onset_latest", "calendar_spread_d",
                       "cum_at_onset_min_mm", "cum_at_onset_max_mm", "frac_at_onset_min", "frac_at_onset_max", "wy_percentile_min",
                       "wy_percentile_max", "wy_first", "wy_last", "rain_total_days", "rain_sweep_best_Z_mm", "rain_sweep_best_X_d",
                       "rain_sweep_best_hits", "rain_sweep_best_null_hits", "rain_sweep_best_p", "rain_sweep_p_search_paid",
                       "rain_sweep_n_settings", "ruleA_best_spread_d", "ruleB_best_spread_d", "fremont_last_valid_day"]
missing_summary = sorted(set(rain_summary_needed) - set(rain_summary.index))
assert not missing_summary, f"{rain_summary_file} lacks {missing_summary}: rerun 12_rain_seasonality"


def rain_value(name):
    """One number of the rainfall summary of 12.

    Parameters
    ----------
    name : str
        Row name in rain_conditioning_summary.csv.

    Returns
    -------
    float
    """
    return float(rain_summary.loc[name, "value"])


assert int(rain_value("n_insar_onsets")) == n_insar_events, "12's rain summary was made from another InSAR catalogue: rerun 12"

# %% [markdown]
# ## 4. Renewal-model forecast of the next onset: the functions
#
# **Onsets and intervals.** Each onset is drawn uniformly within its window, $t_i \sim \mathcal{U}(t_{\mathrm{lo},i},
# t_{\mathrm{hi},i})$. With the onsets sorted, the repeat intervals are
#
# $$x_j = t_{(j+1)} - t_{(j)},$$
#
# kept only when both onsets come from the same source (InSAR or array, so that no interval bridges the unobserved 2006
# to 2017 gap) and, for the records that leave out Loma Prieta, when $t_{(j)} < t_\mathrm{LP} < t_{(j+1)}$ does not hold,
# with $t_\mathrm{LP}$ = `loma_prieta` (decimal year). Below, $x_1, \dots, x_n$ are the $n$ intervals of one draw (yr)
# and $\bar x$ their mean.
#
# **The three models** (`fit_interval_model`), each with density $f(x)$ and the fitted parameters:
#
# - Brownian passage time, the inverse-Gaussian distribution (Matthews et al. 2002), the forecast model:
#
#   $$f(x) = \sqrt{\frac{\mu}{2\pi\alpha^2 x^3}} \exp\left[-\frac{(x - \mu)^2}{2\alpha^2\mu x}\right],$$
#
#   where $\mu$ is the mean interval (yr) and $\alpha$ the aperiodicity, the ratio of the standard deviation of the
#   distribution to its mean, "equivalent to the familiar coefficient of variation" (Ellsworth et al. 1999, p3). In the
#   usual inverse-Gaussian form with shape $\lambda$ (yr), $\alpha^2 = \mu/\lambda$. The code fits both by maximum
#   likelihood, as Ellsworth et al. (1999) do,
#
#   $$\hat\mu = \bar x, \qquad \hat\alpha^2 = \frac{\bar x}{n} \sum_{j=1}^{n} \left(\frac{1}{x_j} - \frac{1}{\bar x}\right),$$
#
#   with $\hat\alpha^2 \ge 10^{-4}$. This $\hat\alpha$ is the coefficient of variation of the fitted distribution; it
#   depends on the reciprocals of the intervals, so it is close to, but not the same as, the sample COV $s/\bar x$.
#   In scipy the model is `invgauss(mu=alpha**2, scale=mu/alpha**2)`. With `alpha_fixed` (the Working Group branches
#   of section 7) only $\hat\mu = \bar x$ is fitted.
# - Lognormal: $f(x) = \dfrac{1}{x\,\sigma\sqrt{2\pi}} \exp\left[-\dfrac{(\ln x - m)^2}{2\sigma^2}\right]$, with
#   $\hat m = \frac{1}{n}\sum_j \ln x_j$ and $\hat\sigma^2 = \frac{1}{n}\sum_j (\ln x_j - \hat m)^2$ (maximum
#   likelihood; $\hat\sigma \ge 10^{-3}$).
# - Weibull: $f(x) = \dfrac{k}{\lambda_W} \left(\dfrac{x}{\lambda_W}\right)^{k-1} e^{-(x/\lambda_W)^k}$, with the shape
#   $k$ and the scale $\lambda_W$ (yr) fitted by maximum likelihood with the location fixed at 0 (scipy
#   `weibull_min.fit`).
#
# The models are not ranked by goodness of fit: with records this short, fits cannot tell the two-parameter models
# apart (Matthews et al. 2002, Appendix 1). The lognormal and Weibull windows are the check that the forecast does not
# depend on the choice of model.
#
# **Conditional probability** (`conditional_cdf`). With the last onset at $t_L$ and no onset up to the reference date
# $t_\mathrm{obs}$, the elapsed quiet time is $e = t_\mathrm{obs} - t_L$, and the probability of the next onset by
# time $t$ is
#
# $$F_c(t) = \frac{F(t - t_L) - F(e)}{S(e)} \quad \text{for } t - t_L > e, \qquad F_c(t) = 0 \text{ otherwise},$$
#
# where $F$ is the cumulative distribution of the fitted interval model and $S = 1 - F$ its survival function. The
# conditional probability of no event by $t$ is $1 - F_c(t)$.
#
# **Forecast mixture** (`forecast`). For each of $K$ = `n_timing` onset draws $k$, the model is fitted to the intervals
# of the draw ($\hat F_k$), the last onset is drawn as $t_L \sim \mathcal{U}$(last InSAR onset window), and a parametric
# bootstrap draws $B$ = `n_boot` sets of $n$ intervals from $\hat F_k$, clips them to at least `min_interval_yr`
# (0.05 yr) and refits the model to each ($\hat F_{kb}$). The forecast curve is the average
#
# $$\bar F_c(t) = \frac{1}{KB} \sum_{k=1}^{K} \sum_{b=1}^{B} F_c^{(kb)}(t),$$
#
# a mixture over the onset timing and the bootstrap, not over the models. A refit that gives the elapsed quiet time
# zero survival adds a curve of ones; these curves are counted and must stay under `max_degenerate_fraction`
# (asserted). The model mean and CoV in the tables are the averages over the $K$ draws of the mean and
# standard deviation / mean of $\hat F_k$; for the BPT the CoV is $\hat\alpha$, so the registered aperiodicity
# `13_<rec>_bpt_alpha` is $\frac{1}{K}\sum_k \hat\alpha_k$.
#
# **Quantiles and wet-season probability.** The $q$ quantile is the first grid day with $\bar F_c(t) \ge q$ (NaN when
# the curve does not reach $q$ on the grid). The probability of an onset between two dates $t_a$ and $t_b$ is
# $\bar F_c(t_b) - \bar F_c(t_a)$, with $\bar F_c$ interpolated linearly on the grid.
#
# The functions also hold the interval statistics (`interval_statistics`, section 6) and the BPT with a fixed
# aperiodicity (`alpha_fixed`, section 7).

# %%
def bpt_distribution(mean_yr, alpha):
    """Brownian passage time (inverse-Gaussian) distribution with a given mean and aperiodicity.

    In scipy's parametrisation, invgauss(mu=alpha^2, scale=mean/alpha^2): its mean is mean_yr and its standard
    deviation / mean is alpha (Ellsworth et al. 1999, p3 density).

    Parameters
    ----------
    mean_yr : float
        Mean interval (yr).
    alpha : float
        Aperiodicity.

    Returns
    -------
    scipy.stats frozen distribution
    """
    return stats.invgauss(mu=alpha ** 2, scale=mean_yr / alpha ** 2)


def fit_interval_model(model, intervals, alpha_fixed=None):
    """Fit one renewal model to a set of intervals (section 4).

    bpt (Brownian passage time, the inverse Gaussian; Matthews et al. 2002): mean mu = the mean interval, shape lambda
    from 1/lambda = mean(1/x - 1/mu) (maximum likelihood), aperiodicity alpha^2 = mu / lambda; scipy's
    invgauss(mu=alpha^2, scale=mu/alpha^2). With alpha_fixed (the Working Group branches): mu = the mean interval and
    alpha = alpha_fixed. lognormal: mean and standard deviation of the log intervals (maximum likelihood). weibull:
    scipy maximum likelihood, location 0.

    Parameters
    ----------
    model : str
        'bpt', 'lognormal' or 'weibull'.
    intervals : array-like
        Repeat intervals (yr).
    alpha_fixed : float, optional
        Fixed BPT aperiodicity.

    Returns
    -------
    scipy.stats frozen distribution
    """
    intervals = np.asarray(intervals, float)
    assert alpha_fixed is None or model == "bpt", "alpha_fixed is a BPT setting"
    if model == "bpt" and alpha_fixed is not None:
        return bpt_distribution(intervals.mean(), alpha_fixed)
    if model == "bpt":                            # Brownian passage time = inverse Gaussian, maximum likelihood
        mean = intervals.mean()
        alpha2 = max(mean * np.mean(1 / intervals - 1 / mean), 1e-4)
        return stats.invgauss(mu=alpha2, scale=mean / alpha2)
    if model == "lognormal":
        log_intervals = np.log(intervals)
        return stats.lognorm(s=max(log_intervals.std(), 1e-3), scale=np.exp(log_intervals.mean()))
    if model == "weibull":
        shape, _, scale = stats.weibull_min.fit(intervals, floc=0)
        return stats.weibull_min(c=shape, scale=scale)
    raise ValueError(model)


def interval_mask(onsets_sorted, src, drop_lp):
    """Which consecutive pairs of sorted onsets form a repeat interval (section 4).

    A pair forms an interval when both onsets come from the same source (so no interval bridges the unobserved 2006
    to 2017 gap) and, if drop_lp, when it does not span Loma Prieta.

    Parameters
    ----------
    onsets_sorted : numpy.ndarray
        Sorted onsets (decimal years).
    src : array-like of str or None
        Source of each onset, in the same order.
    drop_lp : bool
        Leave out the interval spanning Loma Prieta.

    Returns
    -------
    numpy.ndarray of bool
        One value per consecutive pair.
    """
    keep = np.ones(len(onsets_sorted) - 1, bool)
    if src is not None:
        keep &= np.asarray(src)[:-1] == np.asarray(src)[1:]
    if drop_lp:
        keep &= ~((onsets_sorted[:-1] < loma_prieta) & (onsets_sorted[1:] > loma_prieta))
    return keep


def intervals_from(onsets, src=None, drop_lp=False):
    """Repeat intervals between sorted onsets (see interval_mask).

    Parameters
    ----------
    onsets : array-like
        Onsets (decimal years), in any order.
    src : array-like of str, optional
        Source of each onset.
    drop_lp : bool
        Leave out the interval spanning Loma Prieta.

    Returns
    -------
    numpy.ndarray
        Intervals (yr).
    """
    order = np.argsort(onsets)
    onsets_sorted = np.asarray(onsets)[order]
    src_sorted = None if src is None else np.asarray(src)[order]
    return np.diff(onsets_sorted)[interval_mask(onsets_sorted, src_sorted, drop_lp)]


def conditional_cdf(dist, t_last, time_since_last):
    """Conditional probability of the next onset by each grid time, given no onset for time_since_last after t_last.

    (F(t - t_last) - F(elapsed)) / S(elapsed), zero up to the reference date (section 4).

    Parameters
    ----------
    dist : scipy.stats frozen distribution
        Interval model.
    t_last : float
        Last onset (decimal year).
    time_since_last : float
        Elapsed quiet time at the reference date (yr).

    Returns
    -------
    numpy.ndarray
        Conditional cdf on the grid.
    """
    survival = dist.sf(time_since_last)
    assert np.isfinite(survival) and survival > 0, f"zero survival at {time_since_last:.2f} yr elapsed"
    cdf = (dist.cdf(grid - t_last) - dist.cdf(time_since_last)) / survival
    return np.clip(np.where(grid - t_last > time_since_last, cdf, 0.0), 0, 1)


def interval_statistics(interval_draws, n_events):
    """Interval statistics of a record, as quoted in the paper (section 6).

    For each onset draw the sample mean and sample SD (ddof = 1) of its intervals; mean_yr and sd_yr are their
    averages over the draws and cov = sd_yr / mean_yr. '+/-' is 1 SD of the intervals.

    Parameters
    ----------
    interval_draws : list of numpy.ndarray
        The intervals (yr) of each onset draw.
    n_events : int
        Number of events in the record.

    Returns
    -------
    dict
        mean_yr, sd_yr, cov, n_intervals, n_events; per_interval_mean (each interval averaged over the draws, in onset
        order); draw_means and draw_sds (per draw).
    """
    lengths = {len(draw) for draw in interval_draws}
    assert len(lengths) == 1, f"the onset draws give different numbers of intervals {sorted(lengths)}"
    draws = np.array(interval_draws)
    draw_means = draws.mean(axis=1)
    draw_sds = draws.std(axis=1, ddof=1)
    mean_yr = float(draw_means.mean())
    sd_yr = float(draw_sds.mean())
    return dict(mean_yr=mean_yr, sd_yr=sd_yr, cov=sd_yr / mean_yr, n_intervals=draws.shape[1], n_events=int(n_events),
                per_interval_mean=draws.mean(axis=0), draw_means=draw_means, draw_sds=draw_sds)


def forecast(model, catalogue, drop_lp, t_last_window, ref_yr, rng, alpha_fixed=None):
    """Mixture forecast for one model and catalogue (section 4).

    n_timing draws of the onsets in their windows, each fitted and bootstrapped n_boot times.

    Parameters
    ----------
    model : str
        Model name (see fit_interval_model).
    catalogue : pandas.DataFrame
        Events with onset_lo, onset_hi and src.
    drop_lp : bool
        Leave out the interval spanning Loma Prieta.
    t_last_window : tuple of float
        Window of the last onset (decimal years).
    ref_yr : float
        Reference date, the last observation (decimal year).
    rng : numpy.random.Generator
    alpha_fixed : float, optional
        Fixed BPT aperiodicity; only the mean is fitted and refitted.

    Returns
    -------
    cdf : numpy.ndarray
        Mean conditional cdf on the grid.
    mean_yr : float
        Mean of the fitted means (yr).
    cov : float
        Mean of the fitted models' standard deviation / mean (for the BPT, its aperiodicity alpha).
    n_clipped : int
        Number of clipped bootstrap intervals.
    n_degenerate : int
        Number of curves from a refit with zero survival.
    """
    cdf_sum = np.zeros_like(grid)
    n_curves = 0
    n_clipped = 0
    n_degenerate = 0
    interval_means = []
    interval_covs = []
    for _ in range(n_timing):
        # 1) one realisation of the onsets, uniform in their windows, and its intervals
        onsets = rng.uniform(catalogue.onset_lo.values, catalogue.onset_hi.values)
        intervals = intervals_from(onsets, catalogue.src.values, drop_lp)
        # 2) the model fitted to those intervals
        fitted_dist = fit_interval_model(model, intervals, alpha_fixed)
        interval_means.append(fitted_dist.mean())
        interval_covs.append(np.sqrt(fitted_dist.var()) / fitted_dist.mean())
        # 3) the last onset (the same draw when the record holds the last InSAR event) and how long ago it was
        t_last = last_onset(catalogue, onsets, t_last_window, rng)
        time_since_last = ref_yr - t_last
        # 4) parametric bootstrap: refit to synthetic intervals from the fitted model, forecast from each refit
        for _ in range(n_boot):
            bootstrap_intervals = fitted_dist.rvs(size=len(intervals), random_state=rng)
            n_clipped += int(np.sum(bootstrap_intervals < min_interval_yr))
            bootstrap_intervals = np.clip(bootstrap_intervals, min_interval_yr, None)
            refit = fit_interval_model(model, bootstrap_intervals, alpha_fixed)
            survival = refit.sf(time_since_last)
            if np.isfinite(survival) and survival > 0:
                cdf_sum += conditional_cdf(refit, t_last, time_since_last)
            else:
                cdf_sum += np.ones_like(grid)              # the refit says the quiet time is impossible: counted below
                n_degenerate += 1
            n_curves += 1
    assert n_degenerate <= max_degenerate_fraction * n_curves, \
        f"{model}: {n_degenerate} of {n_curves} bootstrap refits give the elapsed quiet time zero survival"
    return cdf_sum / n_curves, float(np.mean(interval_means)), float(np.mean(interval_covs)), n_clipped, n_degenerate


def last_onset(catalogue, onsets, t_last_window, rng):
    """The last onset of one draw: the drawn onset of the last InSAR event if the record holds it, else a uniform
    draw in its window (the array record, which ends in 2006).

    Parameters
    ----------
    catalogue : pandas.DataFrame
        Events with onset_lo and onset_hi.
    onsets : numpy.ndarray
        This draw's onsets, in the order of the catalogue.
    t_last_window : tuple of float
        Window of the last InSAR onset (decimal years).
    rng : numpy.random.Generator

    Returns
    -------
    float
        The last onset (decimal year).
    """
    holds_last = (np.isclose(catalogue.onset_lo.values, t_last_window[0], rtol=0, atol=1e-9)
                  & np.isclose(catalogue.onset_hi.values, t_last_window[1], rtol=0, atol=1e-9))
    if holds_last.any():
        return float(onsets[np.flatnonzero(holds_last)[0]])
    return float(rng.uniform(*t_last_window))


def quantile_date(cdf, q):
    """Grid date at which a cdf first reaches q.

    Parameters
    ----------
    cdf : numpy.ndarray
        cdf on the grid.
    q : float
        Quantile level.

    Returns
    -------
    float
        Decimal year, or NaN if the cdf does not reach q on the grid (never the grid end itself).
    """
    index = int(np.searchsorted(cdf, q))
    if index >= len(grid):
        return np.nan
    return grid[index]


grid_end_label = "> " + decyear_to_date(grid[-1])[0].strftime("%Y-%m-%d")      # how a quantile beyond the grid is written


def quantile_label(cdf, q):
    """A quantile as a date string.

    Parameters
    ----------
    cdf : numpy.ndarray
        cdf on the grid.
    q : float
        Quantile level.

    Returns
    -------
    str
        'YYYY-MM-DD', or '> <grid end>' when the cdf does not reach q on the grid.
    """
    value = quantile_date(cdf, q)
    if np.isnan(value):
        return grid_end_label
    return ymd(value)


def prob_between(cdf, date_a, date_b):
    """Probability of an onset between two calendar dates, from a cdf on the grid (linear interpolation).

    Parameters
    ----------
    cdf : numpy.ndarray
        cdf on the grid.
    date_a, date_b : str or pandas.Timestamp
        First and last date.

    Returns
    -------
    float
    """
    year_a, year_b = decyear(pd.to_datetime([date_a, date_b]))
    return float(np.interp(year_b, grid, cdf) - np.interp(year_a, grid, cdf))


# %% [markdown]
# ## 5. The three renewal models, one at a time
#
# Each model on its own, fitted once (section 4) to the intervals between point onsets: for the InSAR record the step
# epochs `onset_lo` of 04, the last acquisition before each step; for the array record the window midpoints
# `onset_mid`, without the interval that spans Loma Prieta. The forecast of section 6 does not use these single fits,
# since it draws the onsets within their windows and bootstraps; they show the form of each model. For each fit
# `describe_fit` prints the fitted parameters and P(onset in the wet season of the forecast (`wet_label`) | no onset by
# the last observation). The hazard rate, the probability of an onset per unit time given none yet, is
#
# $$h(x) = \frac{f(x)}{S(x)},$$
#
# where $x$ is the time since the last onset (yr).

# %%
insar_intervals = intervals_from(insar_events.onset_lo.values)                        # point onsets, for illustration only
array_intervals = intervals_from(aa_dense.onset_mid.values, drop_lp=True)             # window midpoints, for illustration only
last_onset_lo = float(insar_events.onset_lo.iloc[-1])
time_since_last_yr = last_obs_day_yr - last_onset_lo
print(f"InSAR intervals (yr): {np.round(insar_intervals, 2)}; array intervals (yr, Loma Prieta interval left out): {np.round(array_intervals, 2)}")
print(f"last observation {last_obs} = {last_obs_day_yr:.3f}; last onset window start {last_onset_lo:.3f}; {time_since_last_yr:.2f} yr elapsed with no event")


def describe_fit(model, intervals, label):
    """Fit one model once and print its mean, CoV, own parameters and conditional wet-season probability.

    Parameters
    ----------
    model : str
        Model name (see fit_interval_model).
    intervals : numpy.ndarray
        Repeat intervals (yr).
    label : str
        Record label for the printout.

    Returns
    -------
    scipy.stats frozen distribution
        The fitted model.
    """
    dist = fit_interval_model(model, intervals)
    cdf = conditional_cdf(dist, last_onset_lo, time_since_last_yr)
    p_wet = prob_between(cdf, *forecast_wet_season)
    if model == "lognormal":
        own = f"median {dist.median():.2f} yr, sigma_log {dist.kwds['s']:.2f}"
    elif model == "weibull":
        own = f"shape k {dist.kwds['c']:.2f}, scale {dist.kwds['scale']:.2f} yr"
    else:
        own = f"aperiodicity alpha {np.sqrt(dist.kwds['mu']):.2f}"
    print(f"  {label:<8} {model:<10} mean {dist.mean():.2f} yr, CoV {np.sqrt(dist.var()) / dist.mean():.2f}, {own}; "
          f"P({wet_label} | quiet to last obs) {p_wet:.2f}; "
          f"5-50-95% {quantile_label(cdf, 0.05)} {quantile_label(cdf, 0.5)} {quantile_label(cdf, 0.95)}")
    return dist


single_fits = {}          # (catalogue, model) -> fitted distribution, for the comparison figure

# %% [markdown]
# ### 5.1 Brownian passage time: the forecast model
#
# The inverse-Gaussian distribution of the first time a steadily loaded state variable, perturbed by Brownian noise,
# reaches a fixed threshold (Matthews et al. 2002). Two parameters: the mean interval $\mu$ and the aperiodicity
# $\alpha$, the coefficient of variation of the distribution (section 4). The hazard rises after the last event to a
# peak and then declines to a finite asymptotic rate, $1/(2\mu\alpha^2)$, at times long compared with the mean
# (Ellsworth et al. 1999, p4). It is the model of WGCEP 1999 (with $\alpha$ = 0.3, 0.5 and 0.7 as branches, section
# 7), WGCEP 2003 and UCERF3 (Field et al. 2015), and the model quoted in the prediction statement.

# %%
print("bpt")
single_fits[("A", "bpt")] = describe_fit("bpt", insar_intervals, "A InSAR")
single_fits[("B", "bpt")] = describe_fit("bpt", array_intervals, "B array")

# %% [markdown]
# ### 5.2 Lognormal
#
# The logarithm of the interval is Gaussian. Two parameters: the median interval $e^{m}$ and the spread $\sigma$ of the
# logarithm. The hazard rises after the last event, peaks near the median and then decreases for long elapsed times.
# Nishenko and Buland (1987) introduced it for earthquake recurrence, and the Working Group on California Earthquake
# Probabilities used it (WGCEP 1988, 1990).

# %%
print("lognormal")
single_fits[("A", "lognormal")] = describe_fit("lognormal", insar_intervals, "A InSAR")
single_fits[("B", "lognormal")] = describe_fit("lognormal", array_intervals, "B array")

# %% [markdown]
# ### 5.3 Weibull
#
# Two parameters, a scale $\lambda_W$ and a shape $k$. The hazard is a power of the elapsed time,
#
# $$h(x) = \frac{k}{\lambda_W} \left(\frac{x}{\lambda_W}\right)^{k-1}.$$
#
# For $k > 1$ the hazard increases without limit with elapsed time (ageing, as in reliability engineering, where the
# family originates). A large $k$ corresponds to a regular recurrence. Hagiwara (1974) applied it to earthquakes.

# %%
print("weibull")
single_fits[("A", "weibull")] = describe_fit("weibull", insar_intervals, "A InSAR")
single_fits[("B", "weibull")] = describe_fit("weibull", array_intervals, "B array")

# %% [markdown]
# ### 5.4 The three models side by side (supplementary figure)
#
# (A) The interval density of each model fitted to the InSAR intervals (thick) and the array intervals (thin). (B) The hazard rate $h(x)$ against the time since the last onset; the dotted line marks the
# elapsed time at the last observation. (C) The conditional forecast $F_c(t)$ from each single fit, with the wet season
# shaded and the last observation dash-dotted.

# %%
utils.mpl_paper_style(base=8)          # Helvetica, 8 pt
model_colours = {"lognormal": "#0072B2", "weibull": "#CC79A7", "bpt": "#E69F00"}   # Okabe-Ito, no green
model_names = {"lognormal": "Lognormal", "weibull": "Weibull", "bpt": "BPT"}
wet_shade_colour = "#56B4E9"           # Okabe-Ito sky blue: the wet season
insar_lw, array_lw = 1.4, 0.7          # line weights: fits to the InSAR intervals, fits to the array intervals
interval_axis = np.linspace(0.05, 6.0, 600)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
for model in ("lognormal", "weibull", "bpt"):          # the BPT last, on top: its fit is almost the lognormal's
    for cat, lw in (("A", insar_lw), ("B", array_lw)):
        dist = single_fits[(cat, model)]
        axes[0].plot(interval_axis, dist.pdf(interval_axis), color=model_colours[model], lw=lw)
        hazard = dist.pdf(interval_axis) / np.clip(dist.sf(interval_axis), 1e-12, None)
        axes[1].plot(interval_axis, hazard, color=model_colours[model], lw=lw, label=model_names[model] if cat == "A" else None)
        axes[2].plot(grid, conditional_cdf(dist, last_onset_lo, time_since_last_yr), color=model_colours[model], lw=lw)
axes[0].set_xlim(0, 6)
axes[0].set_ylim(0, None)
axes[0].set_xlabel("Interval (yr)")
axes[0].set_ylabel("Density (1/yr)")
utils.panel_title(axes[0], "A", "Interval density")
# panel A explains the line weights, panel B the model colours (both in empty corners)
weight_handles = [Line2D([], [], color="0.4", lw=insar_lw, label="Fit to InSAR intervals"),
                  Line2D([], [], color="0.4", lw=array_lw, label="Fit to array intervals")]
axes[0].legend(handles=weight_handles, fontsize=6, loc="upper right")
axes[1].axvline(time_since_last_yr, color="0.3", lw=0.7, ls=":")
axes[1].text(time_since_last_yr - 0.05, 0.97, "Last obs.", transform=axes[1].get_xaxis_transform(), fontsize=6, color="0.3", va="top", ha="right")
axes[1].set_xlim(0, 6)
axes[1].set_yscale("log")
axes[1].set_ylim(0.01, 100)
axes[1].set_xlabel("Time since the last onset (yr)")
axes[1].set_ylabel("Hazard rate (1/yr), log")
utils.panel_title(axes[1], "B", "Hazard rate")
handles, labels = axes[1].get_legend_handles_labels()
legend_order = [labels.index(model_names[model]) for model in models]     # BPT first, as in the text
axes[1].legend([handles[k] for k in legend_order], [labels[k] for k in legend_order], fontsize=6, loc="upper right")
axes[2].axvspan(*decyear(pd.to_datetime(list(forecast_wet_season))), color=wet_shade_colour, alpha=0.15, lw=0)
axes[2].axvline(last_obs_day_yr, color="0.3", lw=0.7, ls="-.")
axes[2].set_xlim(2026.0, 2030.0)
axes[2].set_xlabel("Year")
axes[2].set_ylabel("P(onset by t)")
utils.panel_title(axes[2], "C", "Forecast from each single fit")
fig.tight_layout()
f_models = os.path.join(fig_dir, "figS_renewal_single_fits.png")
fig.savefig(f_models, dpi=300)
fig.savefig(f_models.replace(".png", ".pdf"))
print("Saved figure to:", f_models)
plt.show()

# %% [markdown]
# ## 6. The repeat intervals and the forecast (Fig 3C, 3D and the registered window)
#
# For each record, `n_timing` (4000) draws of the onsets within their windows. Their intervals give the interval
# statistics of the record (`interval_statistics`):
#
# $$\bar x_k = \frac{1}{m} \sum_{j=1}^{m} x_{kj}, \qquad
# s_k = \sqrt{\frac{1}{m - 1} \sum_{j=1}^{m} \left(x_{kj} - \bar x_k\right)^2},$$
#
# $$\mathrm{mean} = \frac{1}{K} \sum_{k=1}^{K} \bar x_k, \qquad \mathrm{SD} = \frac{1}{K} \sum_{k=1}^{K} s_k, \qquad
# \mathrm{COV} = \frac{\mathrm{SD}}{\mathrm{mean}},$$
#
# where $x_{kj}$ is interval $j$ of draw $k$ (yr), $m$ the number of intervals of the record and $K$ the number of
# draws. These are drawn in Fig 3C and written to `recurrence_intervals.csv`. Then, for each forecast record
# (`forecast_records`) and each model, the forecast mixture of section 4 is computed with `n_timing` further onset
# draws, each fitted and bootstrapped, conditioned on no event through the last observation. For every record the last
# onset $t_L$ is the last InSAR event: in the InSAR and merged records it is that event's onset in the same draw as the
# intervals, and the array record (which ends in 2006) draws it from the window of the last InSAR event.

# %%
rng = np.random.default_rng(seed)
t_last_window = (float(insar_events.onset_lo.iloc[-1]), float(insar_events.onset_hi.iloc[-1]))
forecast_rows = []
forecast_curves = {}
interval_stats = {}               # catalogue -> interval_statistics(...): the definition of mean, SD, COV
stats_lines = ["13 FORECAST", "", "REPEAT INTERVALS AND RENEWAL FORECAST",
               f"last event onset window {ymd(t_last_window[0])} to {ymd(t_last_window[1])}; reference date (last observation) {last_obs}; InSAR record end {data_end}; "
               f"{n_timing} timing draws x {n_boot} bootstrap refits; {wet_label} {forecast_wet_season[0]}..{forecast_wet_season[1]} ({wet_months_text}); "
               f"forecast grid {grid[0]:.0f} to {grid_end_label[2:]} (a quantile not reached on it is '{grid_end_label}')",
               "interval statistics: per onset draw the sample mean and SD (ddof = 1) of the intervals, averaged over the draws; "
               "COV = SD / mean; +/- = 1 SD of the intervals"]
interval_rows = []
for cat_name, (catalogue, drop_lp) in catalogues.items():
    # the onset draws, their intervals and the record's interval statistics
    interval_draws = []
    for _ in range(n_timing):
        onsets = rng.uniform(catalogue.onset_lo.values, catalogue.onset_hi.values)
        interval_draws.append(intervals_from(onsets, catalogue.src.values, drop_lp))
    interval_stats[cat_name] = interval_statistics(interval_draws, len(catalogue))
    record = interval_stats[cat_name]
    lp_text = " (Loma Prieta-spanning interval dropped)" if drop_lp else ""
    stats_lines += [f"{cat_name} ({record_keys[cat_name]}): {record['n_events']} events, {record['n_intervals']} intervals" + lp_text,
                    f"  interval mean {record['mean_yr']:.2f} +/- {record['sd_yr']:.2f} yr (1 SD), COV {record['cov']:.2f}; "
                    f"each interval over the draws: {', '.join(f'{v:.2f}' for v in record['per_interval_mean'])} yr"]
    interval_rows.append(dict(catalog=cat_name, record=record_keys[cat_name], n_events=record["n_events"], n_intervals=record["n_intervals"],
                              mean_yr=record["mean_yr"], sd_yr=record["sd_yr"], cov=record["cov"], drop_loma_prieta=drop_lp,
                              intervals_yr=" ".join(f"{v:.4f}" for v in record["per_interval_mean"])))
    if cat_name not in forecast_records:
        stats_lines.append("    interval statistics only (not forecast)")
        continue
    for model in models:
        cdf, model_mean, model_cov, n_clipped, n_degenerate = forecast(model, catalogue, drop_lp, t_last_window, last_obs_day_yr, rng)
        forecast_curves[(cat_name, model)] = cdf
        row = dict(catalog=cat_name, record=record_keys[cat_name], model=model, ref="last_obs", ref_date=last_obs,
                   n_intervals=record["n_intervals"], model_mean_yr=model_mean, model_cov=model_cov,
                   q05=quantile_label(cdf, 0.05), q50=quantile_label(cdf, 0.5), q95=quantile_label(cdf, 0.95),
                   p_wet_season=prob_between(cdf, *forecast_wet_season),
                   p_by_grid_end=float(cdf[-1]),
                   n_bootstrap_intervals_clipped=n_clipped, n_curves_zero_survival=n_degenerate)
        forecast_rows.append(row)
        model_cov_name = "alpha" if model == "bpt" else "sd/mean"
        stats_lines.append(f"    {model:<12} fitted mean {model_mean:.2f} yr, {model_cov_name} {model_cov:.3f} | next onset 5-50-95%: "
                           f"{row['q05']} {row['q50']} {row['q95']} | P({wet_label}) {row['p_wet_season']:.2f} "
                           f"| clipped intervals {n_clipped}, zero-survival curves {n_degenerate}")
forecasts = pd.DataFrame(forecast_rows)
interval_table = pd.DataFrame(interval_rows)
interval_table.to_csv(os.path.join(out_dir, "recurrence_intervals.csv"), index=False, float_format="%.4f")
# the records quoted in the text have these event and interval counts: a changed catalogue stops the notebook
expected_counts = {"A_insar": (5, 4), "B_aa_dense": (9, 8), "B_aa_dense_noLP": (9, 7), "C_merged": (14, 11)}
for cat_name, (n_events_expected, n_intervals_expected) in expected_counts.items():
    got = (interval_stats[cat_name]["n_events"], interval_stats[cat_name]["n_intervals"])
    assert got == (n_events_expected, n_intervals_expected), f"{cat_name}: {got} events/intervals, expected {(n_events_expected, n_intervals_expected)}"
# no quantile may fall before the conditioning date (the conditional cdf is zero up to it); the quantiles beyond the
# grid are listed (written '> <grid end>', with the probability the curve reaches by then)
beyond_grid = []
for quantile_column in ("q05", "q50", "q95"):
    on_grid = forecasts[quantile_column] != grid_end_label
    assert (pd.to_datetime(forecasts.loc[on_grid, quantile_column]) > pd.Timestamp(last_obs)).all(), \
        f"a forecast {quantile_column} falls on or before the last observation {last_obs}"
    for row in forecasts[~on_grid].itertuples():
        beyond_grid.append(f"{row.catalog} {row.model} {quantile_column} (P by grid end {row.p_by_grid_end:.3f})")
stats_lines.append(f"quantiles beyond the forecast grid ({grid_end_label}): " + (", ".join(beyond_grid) if beyond_grid else "none"))
forecasts.to_csv(os.path.join(out_dir, "recurrence_forecast_models.csv"), index=False, float_format="%.3f")
print("\n".join(stats_lines))

# %% [markdown]
# ## 7. The Working Group aperiodicities and the falsification dates
#
# **WGCEP 1999 branches.** For each forecast record, the BPT forecast of section 6 is recomputed with $\alpha$ fixed at
# each of `wg99_alpha_branches` (0.3, 0.5, 0.7; OFR 99-517, printed p16), with only the mean set from each onset draw
# and from each bootstrap refit. Each branch has its own generator started from `seed`, so the three branches use the
# same onset and bootstrap draws. The conditional curves of the branches are mixed as
#
# $$F_\mathrm{mix}(t) = \sum_{b=1}^{3} w_b\, \bar F_c^{(\alpha_b)}(t),$$
#
# where $\alpha_b$ is branch $b$ and $w_b$ its weight in `wg99_alpha_weights` (0.2, 0.6, 0.2, the three-way branch
# convention of the report, App. C, printed p31; the $\alpha$ weights themselves are not printed). WG99 mixes branch
# probabilities by a weighted mean (printed p18).
#
# **Falsification.** For the InSAR BPT forecast of section 6, the falsification date for level $\ell$
# (`falsification_levels`, 5 % and 1 %) is the first grid day on which the conditional survival falls to $\ell$,
#
# $$t_\ell = \min\{t : 1 - \bar F_c(t) \le \ell\},$$
#
# the $1 - \ell$ quantile of the forecast. No event by $t_\ell$ rejects the model at $1 - \ell$ (95 % and 99 %). The 5 %
# date is the 95 % quantile of the forecast, `13_insar_bpt_q95`.

# %%
wg99_rows = []
wg99_curves = {}                  # (catalogue, alpha or "mixture") -> mean conditional cdf
wg99_lines = ["", f"WORKING GROUP APERIODICITIES (WGCEP 1999, OFR 99-517 p16): BPT with alpha fixed at {wg99_alpha_branches}, "
              f"mixed with weights {wg99_alpha_weights} (App. C p31 three-way convention; the alpha weights are not printed in the report)"]
for cat_name in forecast_records:
    catalogue, drop_lp = catalogues[cat_name]
    for alpha in wg99_alpha_branches:
        cdf, model_mean, model_cov, n_clipped, n_degenerate = forecast("bpt", catalogue, drop_lp, t_last_window, last_obs_day_yr,
                                                                       np.random.default_rng(seed), alpha_fixed=alpha)
        assert abs(model_cov - alpha) < 1e-6, f"{cat_name}: the fixed-alpha fits have sd/mean {model_cov:.4f}, not {alpha}"
        wg99_curves[(cat_name, alpha)] = cdf
        wg99_rows.append(dict(catalog=cat_name, record=record_keys[cat_name], branch=f"alpha {alpha:g}", alpha=alpha,
                              weight=wg99_alpha_weights[wg99_alpha_branches.index(alpha)],
                              q05=quantile_label(cdf, 0.05), q50=quantile_label(cdf, 0.5), q95=quantile_label(cdf, 0.95),
                              p_wet_season=prob_between(cdf, *forecast_wet_season), p_by_grid_end=float(cdf[-1]),
                              n_bootstrap_intervals_clipped=n_clipped, n_curves_zero_survival=n_degenerate))
    mixture = np.zeros_like(grid)
    for alpha, weight in zip(wg99_alpha_branches, wg99_alpha_weights):
        mixture += weight * wg99_curves[(cat_name, alpha)]
    wg99_curves[(cat_name, "mixture")] = mixture
    mixture_p = prob_between(mixture, *forecast_wet_season)
    wg99_rows.append(dict(catalog=cat_name, record=record_keys[cat_name], branch="WG99 mixture", alpha=np.nan, weight=1.0,
                          q05=quantile_label(mixture, 0.05), q50=quantile_label(mixture, 0.5), q95=quantile_label(mixture, 0.95),
                          p_wet_season=mixture_p, p_by_grid_end=float(mixture[-1])))
    branch_text = []
    for alpha in wg99_alpha_branches:
        branch_text.append(f"{alpha:g}: {prob_between(wg99_curves[(cat_name, alpha)], *forecast_wet_season):.2f}")
    wg99_lines.append(f"  {cat_name}: mixture 5-50-95% {wg99_rows[-1]['q05']} {wg99_rows[-1]['q50']} {wg99_rows[-1]['q95']}, "
                      f"P({wet_label}) {mixture_p:.3f}; single branches P({wet_label}) " + ", ".join(branch_text))

# falsification: when the InSAR BPT's conditional survival falls to each level
falsification_dates = {}
for level in falsification_levels:
    falsification_dates[level] = quantile_label(forecast_curves[("A_insar", "bpt")], 1 - level)
    assert falsification_dates[level] != grid_end_label, f"the InSAR BPT never reaches survival {level} on the grid"
wg99_lines.append("  falsification (InSAR BPT): no event by " +
                  ", ".join(f"{falsification_dates[level]} (survival {level:g})" for level in falsification_levels) +
                  " rejects it at " + ", ".join(f"{1 - level:.0%}" for level in falsification_levels))
wg99_table = pd.DataFrame(wg99_rows)
wg99_table.to_csv(os.path.join(out_dir, "recurrence_forecast_wg99.csv"), index=False, float_format="%.4f")
stats_lines += wg99_lines
print("\n".join(wg99_lines))

# %% [markdown]
# ## 8. The inputs of Fig 3
#
# Fig 3 is drawn in notebook 14 from saved products. This section writes what the figure needs from this notebook to
# `fig3_inputs.npz`; panels A and B read the products of 11 and 12 directly. The records drawn in Fig 3C and 3D
# (`shown`) are the three quoted in the text: the InSAR record, the array record without the interval spanning Loma
# Prieta and the combined record. For each of them the file holds the possible range of each repeat interval, the
# interval mean and SD of section 6 (`13_<rec>_interval_mean_yr`, `13_<rec>_interval_sd_yr`) and the BPT forecast
# with the fitted $\alpha$ on the forecast grid. Interval $j$ ranges from $t_{\mathrm{lo},j+1} - t_{\mathrm{hi},j}$ to
# $t_{\mathrm{hi},j+1} - t_{\mathrm{lo},j}$, where $t_{\mathrm{lo},j}$ and $t_{\mathrm{hi},j}$ are the ends of window
# $j$, and the number of ranges must equal the number of intervals of section 6 (asserted). The file also holds the
# last observation, the wet season of the forecast, the data end (`data_end`), the settings that name the figure file
# and the onsets of the InSAR and array catalogues, against which notebook 14 checks the onset windows of 12.
#
# | Key | Content |
# |---|---|
# | `provenance` | notebook, git hash and time of the run |
# | `records` | the catalogue names of `shown`, in drawing order |
# | `interval_mean_yr`, `interval_sd_yr`, `n_intervals` | the interval statistics of section 6, one value per record |
# | `interval_shortest_<record>`, `interval_longest_<record>` | the two ends of each interval range (yr) |
# | `grid`, `bpt_cdf` | the forecast grid (decimal years) and the fitted-$\alpha$ BPT forecast of each record on it |
# | `last_obs`, `last_obs_day_yr` | the last observation, as a date and as a decimal year |
# | `forecast_wet_season`, `wet_season_yr` | the first and last day of the wet season of the forecast, as dates and decimal years |
# | `data_end`, `data_end_yr` | the end of the InSAR record, as a date and as a decimal year |
# | `insar_onset_lo`, `aa_dense_onset_lo` | the window starts of the InSAR and array catalogues |
# | `n_timing`, `n_boot`, `forecast_issue_date` | the settings in the file name of Fig 3 |

# %%
shown = ["A_insar", "B_aa_dense_noLP", "C_merged"]       # the array without the Loma Prieta interval; B (with it) stays in the table
fig3_ranges = {}
fig3c_numbers = {}                 # catalogue -> (mean, sd, number of intervals): interval_stats of section 6, printed
for cat_name in shown:
    # each interval as its possible range, from the end of the earlier window to the start of the later one, up to the
    # start of the earlier to the end of the later
    catalogue, drop_lp = catalogues[cat_name]
    catalogue = catalogue.sort_values("onset_mid").reset_index(drop=True)
    keep = interval_mask(catalogue.onset_mid.values, catalogue.src.values, drop_lp)     # which pairs form an interval (order and Loma Prieta side)
    shortest = (catalogue.onset_lo.values[1:] - catalogue.onset_hi.values[:-1])[keep]
    longest = (catalogue.onset_hi.values[1:] - catalogue.onset_lo.values[:-1])[keep]
    mean_yr = interval_stats[cat_name]["mean_yr"]
    sd_yr = interval_stats[cat_name]["sd_yr"]
    assert interval_stats[cat_name]["n_intervals"] == len(shortest), f"{cat_name}: Fig 3C draws {len(shortest)} intervals, section 6 has a different count"
    fig3c_numbers[cat_name] = (mean_yr, sd_yr, len(shortest))
    fig3_ranges[f"interval_shortest_{cat_name}"] = shortest
    fig3_ranges[f"interval_longest_{cat_name}"] = longest
for cat_name, (mean_yr, sd_yr, n_shown) in fig3c_numbers.items():
    stats_lines.append(f"Fig 3C {cat_name}: {n_shown} intervals, mean {mean_yr:.3f} yr, SD {sd_yr:.3f} yr (the interval statistics of section 6)")
    print(stats_lines[-1])

# one value or one curve per record, in the order of shown
shown_means = []
shown_sds = []
shown_counts = []
shown_curves = []
for cat_name in shown:
    shown_means.append(fig3c_numbers[cat_name][0])
    shown_sds.append(fig3c_numbers[cat_name][1])
    shown_counts.append(fig3c_numbers[cat_name][2])
    shown_curves.append(forecast_curves[(cat_name, "bpt")])
fig3_inputs_file = os.path.join(out_dir, "fig3_inputs.npz")
np.savez(fig3_inputs_file, provenance=utils.provenance_stamp(notebook, repo_dir), records=np.array(shown),
         interval_mean_yr=np.array(shown_means), interval_sd_yr=np.array(shown_sds), n_intervals=np.array(shown_counts),
         grid=grid, bpt_cdf=np.array(shown_curves), last_obs=last_obs, last_obs_day_yr=last_obs_day_yr,
         forecast_wet_season=np.array(forecast_wet_season),
         wet_season_yr=np.asarray(decyear(pd.to_datetime(list(forecast_wet_season))), float),
         data_end=data_end, data_end_yr=data_end_yr,
         insar_onset_lo=catalogues["A_insar"][0].onset_lo.values, aa_dense_onset_lo=aa_dense.onset_lo.values,
         n_timing=n_timing, n_boot=n_boot, forecast_issue_date=forecast_issue_date, **fig3_ranges)
print("Saved the Fig 3 inputs to:", fig3_inputs_file)

# %% [markdown]
# ## 9. The prediction statement
#
# The statement is built from the numbers above in a fixed template, so that it is generated rather than typed. Part
# (3) of the time window (the calendar window of the onsets) and the last day of the Fremont gauge come from
# 12_rain_seasonality (`rain_conditioning_summary.csv`). The location comes from `section_lengths.csv` of 04: the fault
# length hosting the events (`tulepond_creep_length`) and the peak slip zone (`tulepond_band90`). The magnitude is the
# step sizes of the paper, from the 90 % band stack of 04 (`04_band90_step_<yr>_mm` with their 95 % CIs), read back from
# the registry rows of the latest run of 04. That run must be newer than `section_lengths.csv` and must have registered
# the same onset windows as `creep_events.csv` (both asserted), so that the statement does not mix two runs of 04.
# These numbers belong to 04 and are not registered again here.

# %%
# location: the measured extents of 04
section_lengths = pd.read_csv(section_lengths_file).set_index("name")
creep_length = section_lengths.loc["tulepond_creep_length"]
peak_zone = section_lengths.loc[f"tulepond_{magnitude_band}"]
# magnitude: the band-stack steps of the latest run of 04, from the registry
registry_rows = pd.read_csv(common_paths["registry_csv"], dtype=str)
rows_04 = registry_rows[registry_rows.notebook == "04_event_detection"]
assert len(rows_04), "the registry has no rows of 04_event_detection"
last_run_04 = rows_04.sort_values("timestamp").run_id.iloc[-1]
rows_04 = rows_04[rows_04.run_id == last_run_04].set_index("key")
run_04_time = pd.Timestamp(rows_04.timestamp.min())          # the registry stamps local time to the second
section_lengths_time = pd.Timestamp.fromtimestamp(np.floor(os.path.getmtime(section_lengths_file)))     # local time, as the registry
assert run_04_time >= section_lengths_time - pd.Timedelta(seconds=1), \
    f"the latest registry run of 04 ({run_04_time}) is older than {section_lengths_file}: 04 stopped before registering; rerun it"
event_labels = []
for onset_lo, onset_hi in zip(insar_events.onset_lo_grid, insar_events.onset_hi_grid):
    label = f"{int(onset_lo)}"                     # key label in 04: the year of the step epoch
    event_labels.append(label)
    assert np.isclose(float(rows_04.loc[f"04_onset_lo_{label}", "value"]), onset_lo, rtol=0, atol=1e-4) and \
        np.isclose(float(rows_04.loc[f"04_onset_hi_{label}", "value"]), onset_hi, rtol=0, atol=1e-4), \
        f"the {label} window registered by 04 is not the one in {events_file}"
band_steps = []
band_ci_lo = []
band_ci_hi = []
for label in event_labels:
    band_steps.append(float(rows_04.loc[f"04_{magnitude_band}_step_{label}_mm", "value"]))
    band_ci_lo.append(float(rows_04.loc[f"04_{magnitude_band}_step_{label}_ci_lo", "value"]))
    band_ci_hi.append(float(rows_04.loc[f"04_{magnitude_band}_step_{label}_ci_hi", "value"]))
band_steps = np.array(band_steps)
assert np.isclose(band_steps.mean(), float(rows_04.loc[f"04_{magnitude_band}_steps_mean_mm", "value"]), rtol=0, atol=1e-6)
assert int(float(rows_04.loc[f"04_{magnitude_band}_n_crossings", "value"])) == int(peak_zone.n)
print(f"04 run {last_run_04} ({run_04_time}): {magnitude_band} steps {np.round(band_steps, 2).tolist()} mm, "
      f"95% CIs {min(band_ci_lo):.2f}-{max(band_ci_hi):.2f} mm")

insar_forecast = forecasts[forecasts.catalog == "A_insar"].set_index("model")
array_forecast = forecasts[forecasts.catalog == "B_aa_dense_noLP"].set_index("model")
merged_forecast = forecasts[forecasts.catalog == "C_merged"].set_index("model")
wg99_mixture = wg99_table[wg99_table.branch == "WG99 mixture"].set_index("catalog")
insar_window_days = (insar_events.onset_hi - insar_events.onset_lo) * 365.25
array_first_year = int(np.floor(aa_dense.onset_lo.min()))


def wg99_text(cat_name):
    """The WG99-branch mixture of one record as one phrase of the statement.

    Parameters
    ----------
    cat_name : str
        Catalogue name.

    Returns
    -------
    str
        Its P(wet season), the P of each branch and the 5-95 % window.
    """
    mixture_row = wg99_mixture.loc[cat_name]
    branch_p = []
    for alpha in wg99_alpha_branches:
        branch_p.append(f"{prob_between(wg99_curves[(cat_name, alpha)], *forecast_wet_season):.2f}")
    return (f"with the Working Group aperiodicities (alpha {', '.join(f'{a:g}' for a in wg99_alpha_branches)}, weights "
            f"{', '.join(f'{w:g}' for w in wg99_alpha_weights)}; WGCEP 1999 p16) P({wet_label}) = {mixture_row.p_wet_season:.2f} "
            f"(single branches {' / '.join(branch_p)}), 5-95% {mixture_row.q05} .. {mixture_row.q95}")


# the lag spreads of the tightest rules A and B (12) beside the calendar spread of the onsets
rules_text = (f"      tightest rainfall rules (12_rain_seasonality): lag spread A {rain_value('ruleA_best_spread_d'):.0f} d, "
              f"B {rain_value('ruleB_best_spread_d'):.0f} d, against the calendar spread of the onsets {rain_value('calendar_spread_d'):.0f} d")
statement_lines = [
    "TULE POND CREEP-EVENT PREDICTION STATEMENT (generated by notebook 13_forecast)",
    f"first written {forecast_issue_date}; conditioned on no event through the last observation, {last_obs} "
    f"(descending CSLC + Dolphin extension of the InSAR record); "
    f"data: OPERA DISP-S1 to {data_end} (the end of the paper's InSAR record), HRKT array events to {aa_dense.onset_hi.max():.2f} (bimonthly surveys to {dense_era_end:.2f}), "
    f"Fremont GHCN gauge to {rain_summary.loc['fremont_last_valid_day', 'value']} (12_rain_seasonality)",
    f"the renewal-model windows below run from the last observation; nothing is claimed for the months after {last_obs}", "",
    f"LOCATION  Hayward fault, Tule Pond section: the events slip {creep_length.km_lo:.1f}-{creep_length.km_hi:.1f} km from Point Pinole "
    f"({creep_length.km_hi - creep_length.km_lo:.1f} km of fault, {creep_length.lat_lo:.3f}-{creep_length.lat_hi:.3f} N); peak slip zone "
    f"({magnitude_band_pct} % of the peak event fraction) {peak_zone.km_lo:.1f}-{peak_zone.km_hi:.1f} km "
    f"({peak_zone.km_hi - peak_zone.km_lo:.1f} km, {peak_zone.lat_lo:.3f}-{peak_zone.lat_hi:.3f} N)",
    f"MAGNITUDE surface slip {band_steps.mean():.1f} mm on average, {band_steps.min():.1f}-{band_steps.max():.1f} mm per event, on the "
    f"{magnitude_band_pct} % band stack ({int(peak_zone.n)} crossings; {n_insar_events} InSAR events; 95% CIs {min(band_ci_lo):.1f}-{max(band_ci_hi):.1f} mm), "
    f"within one {insar_window_days.min():.0f}-{insar_window_days.max():.0f} d acquisition interval", "",
    "TIME WINDOW",
    f"  (1) InSAR record ({interval_stats['A_insar']['n_intervals']} intervals, {interval_stats['A_insar']['n_events']} events): repeat interval "
    f"{interval_stats['A_insar']['mean_yr']:.2f} +/- {interval_stats['A_insar']['sd_yr']:.2f} yr (1 SD, {interval_stats['A_insar']['sd_yr'] * 365.25:.0f} d), "
    f"COV {interval_stats['A_insar']['cov']:.2f}, over {n_timing} onset draws",
    f"      BPT with the aperiodicity fitted to these intervals ({insar_forecast.loc['bpt', 'model_cov']:.3f}, maximum likelihood; "
    f"a statement conditional on the 2017-2025 regularity persisting): 5-95% {insar_forecast.loc['bpt', 'q05']} .. {insar_forecast.loc['bpt', 'q95']}, "
    f"median {insar_forecast.loc['bpt', 'q50']}; P({wet_label}, {wet_months_text}) = {insar_forecast.loc['bpt', 'p_wet_season']:.2f}",
    f"      check, other models: lognormal 5-95% {insar_forecast.loc['lognormal', 'q05']} .. {insar_forecast.loc['lognormal', 'q95']}, "
    f"P = {insar_forecast.loc['lognormal', 'p_wet_season']:.2f}; Weibull 5-95% {insar_forecast.loc['weibull', 'q05']} .. "
    f"{insar_forecast.loc['weibull', 'q95']}, P = {insar_forecast.loc['weibull', 'p_wet_season']:.2f}",
    f"      falsification: under that model the chance of no event falls to " +
    " and ".join(f"{level:.0%} by {falsification_dates[level]}" for level in falsification_levels) +
    f": no event by then rejects it at " + " / ".join(f"{1 - level:.0%}" for level in falsification_levels),
    f"      {wg99_text('A_insar')}",
    f"  (2) HRKT bimonthly-era record ({array_first_year}-{dense_era_end:.0f}), Loma Prieta interval dropped "
    f"({interval_stats['B_aa_dense_noLP']['n_intervals']} intervals, {interval_stats['B_aa_dense_noLP']['mean_yr']:.2f} +/- "
    f"{interval_stats['B_aa_dense_noLP']['sd_yr']:.2f} yr, COV {interval_stats['B_aa_dense_noLP']['cov']:.2f}): BPT (fitted aperiodicity "
    f"{array_forecast.loc['bpt', 'model_cov']:.2f}) 5-95% {array_forecast.loc['bpt', 'q05']} .. {array_forecast.loc['bpt', 'q95']}, "
    f"median {array_forecast.loc['bpt', 'q50']}; P({wet_label}) = {array_forecast.loc['bpt', 'p_wet_season']:.2f}; {wg99_text('B_aa_dense_noLP')}",
    f"      merged ({interval_stats['C_merged']['n_intervals']} intervals, {interval_stats['C_merged']['mean_yr']:.2f} +/- "
    f"{interval_stats['C_merged']['sd_yr']:.2f} yr, COV {interval_stats['C_merged']['cov']:.2f}): BPT 5-95% {merged_forecast.loc['bpt', 'q05']} .. "
    f"{merged_forecast.loc['bpt', 'q95']}, median {merged_forecast.loc['bpt', 'q50']}; P({wet_label}) = {merged_forecast.loc['bpt', 'p_wet_season']:.2f}; "
    f"{wg99_text('C_merged')}",
    f"      (a quantile written '{grid_end_label}' is beyond the forecast grid: the curve has not reached it there)",
    f"  (3) all {int(rain_value('n_insar_onsets'))} InSAR onset windows fell {int(rain_value('onset_dowy_min'))}-{int(rain_value('onset_dowy_max'))} d into the water year "
    f"({rain_summary.loc['onset_earliest', 'value']}-{rain_summary.loc['onset_latest', 'value']}), after {rain_value('cum_at_onset_min_mm'):.0f}-"
    f"{rain_value('cum_at_onset_max_mm'):.0f} mm of WY rain ({rain_value('frac_at_onset_min'):.0%}-{rain_value('frac_at_onset_max'):.0%} of the WY total), "
    f"in years spanning P{rain_value('wy_percentile_min'):.0f}-P{rain_value('wy_percentile_max'):.0f} of the "
    f"{int(rain_value('wy_first'))}-{int(rain_value('wy_last'))} WY-total distribution (12_rain_seasonality)",
    f"      a {int(rain_value('rain_total_days'))}-day rain total of at least Z mm within X days before the onset (Fremont gauge, the "
    f"{int(rain_value('n_insar_onsets'))} InSAR onsets, {int(rain_value('rain_sweep_n_settings'))} settings of Z and X): the best, "
    f"{rain_value('rain_sweep_best_Z_mm'):.0f} mm within {rain_value('rain_sweep_best_X_d'):.0f} d, gives {rain_value('rain_sweep_best_hits'):.2f} of "
    f"{int(rain_value('n_insar_onsets'))} onsets against {rain_value('rain_sweep_best_null_hits'):.2f} for a random day of the same season in the same "
    f"water year, p = {rain_value('rain_sweep_best_p'):.3f}; p = {rain_value('rain_sweep_p_search_paid'):.3f} once all "
    f"{int(rain_value('rain_sweep_n_settings'))} settings are paid for",
    rules_text]
with open(os.path.join(out_dir, "prediction_statement.txt"), "w") as fh:
    fh.write("\n".join(statement_lines) + "\n")
print("\n".join(statement_lines))

# %% [markdown]
# ## 10. Registry: the stats sidecar `13_forecast_stats.txt` (with the prediction statement) and the registered numbers
#
# Per record `<rec>` (`insar`, `aa_dense_nolp`, `aa_dense_withlp`, `merged`), under `13_<rec>_*`: the interval
# statistics of section 6 (`_interval_mean_yr`, `_interval_sd_yr`, `_interval_cov`, `_n_intervals`, `_n_events`).
# Per forecast record (all but `aa_dense_withlp`): for each model the 5, 50 and 95 % dates and P(wet season); the BPT
# aperiodicity (`_bpt_alpha`, maximum likelihood, averaged over the forecast draws); the Working Group mixture
# (`_bpt_wg99_q05`, `_q50`, `_q95`, `_p_wet_wy<N>`) and the P(wet season) of each branch (`_bpt_alpha03_p_wet_wy<N>` and
# so on). Under `13_*`: the issue date, the data end and the last observation; each InSAR interval
# (`13_insar_interval_<y1>_<y2>_yr`) and each array interval with the Loma Prieta interval
# (`13_aa_dense_withlp_interval_<y1>_<y2>_yr`), averaged over the onset draws and labelled by the years in which the
# onset windows open; the InSAR interval SD in days; and the falsification dates.

# %%
sidecar = REG.sidecar("13_forecast", stats_lines + ["", *statement_lines])
REG.register("13_issue_date", forecast_issue_date, "date", 0, where="P10, M9", cell="constants", sidecar=sidecar)
REG.register("13_data_end", data_end, "date", 0, where="P10, M9", cell="load", sidecar=sidecar)
REG.register("13_last_obs_date", last_obs, "date", 0, where="P10, M9", cell="load", sidecar=sidecar)
# the interval statistics (section 6), every record
for cat_name, key in record_keys.items():
    record = interval_stats[cat_name]
    REG.register(f"13_{key}_interval_mean_yr", record["mean_yr"], "yr", 2, where="P10, Fig3", cell="intervals", sidecar=sidecar)
    REG.register(f"13_{key}_interval_sd_yr", record["sd_yr"], "yr", 2, where="P10, Fig3", cell="intervals", sidecar=sidecar)
    REG.register(f"13_{key}_interval_cov", record["cov"], "1", 2, where="P10", cell="intervals", sidecar=sidecar)
    REG.register(f"13_{key}_n_intervals", record["n_intervals"], "count", 0, where="P10, Fig3", cell="intervals", sidecar=sidecar)
    REG.register(f"13_{key}_n_events", record["n_events"], "count", 0, where="P10, Fig3", cell="intervals", sidecar=sidecar)
# the intervals one by one (averaged over the onset draws), labelled by the years in which the onset windows open: the
# InSAR record and the array record with the Loma Prieta interval (the onset-window table of the supplement)
for cat_name, catalogue in (("A_insar", insar_events), ("B_aa_dense", aa_dense)):
    window_years = []
    for onset_lo in catalogue.sort_values("onset_mid").onset_lo:
        window_years.append(f"{int(onset_lo)}")
    for k, value in enumerate(interval_stats[cat_name]["per_interval_mean"]):
        REG.register(f"13_{record_keys[cat_name]}_interval_{window_years[k]}_{window_years[k + 1]}_yr", value, "yr", 2,
                     where="P9, M10", cell="intervals", sidecar=sidecar)
REG.register("13_insar_interval_sd_d", interval_stats["A_insar"]["sd_yr"] * 365.25, "d", 0, where="P10", cell="intervals", sidecar=sidecar)
# the renewal forecasts and the BPT aperiodicity
for cat_name in forecast_records:
    key = record_keys[cat_name]
    issue = forecasts[forecasts.catalog == cat_name].set_index("model")
    for model in models:
        REG.register(f"13_{key}_{model}_q05", issue.loc[model, "q05"], "date", 0, where="P10, Fig3", cell="forecast", sidecar=sidecar)
        REG.register(f"13_{key}_{model}_q50", issue.loc[model, "q50"], "date", 0, where="P10, Fig3", cell="forecast", sidecar=sidecar)
        REG.register(f"13_{key}_{model}_q95", issue.loc[model, "q95"], "date", 0, where="P10, Fig3", cell="forecast", sidecar=sidecar)
        REG.register(f"13_{key}_{model}_p_wet_wy{forecast_wy}", issue.loc[model, "p_wet_season"], "1", 2, where="P10, Fig3", cell="forecast", sidecar=sidecar)
    REG.register(f"13_{key}_bpt_alpha", float(issue.loc["bpt", "model_cov"]), "1", 3, where="P10, M10", cell="forecast", sidecar=sidecar)
# the Working Group aperiodicities: the mixture and each branch
for cat_name in forecast_records:
    key = record_keys[cat_name]
    mixture_row = wg99_mixture.loc[cat_name]
    REG.register(f"13_{key}_bpt_wg99_q05", mixture_row.q05, "date", 0, where="P10, M10", cell="wg99", sidecar=sidecar)
    REG.register(f"13_{key}_bpt_wg99_q50", mixture_row.q50, "date", 0, where="P10, M10", cell="wg99", sidecar=sidecar)
    REG.register(f"13_{key}_bpt_wg99_q95", mixture_row.q95, "date", 0, where="P10, M10", cell="wg99", sidecar=sidecar)
    REG.register(f"13_{key}_bpt_wg99_p_wet_wy{forecast_wy}", mixture_row.p_wet_season, "1", 2, where="P10, M10", cell="wg99", sidecar=sidecar)
    for alpha in wg99_alpha_branches:
        branch_tag = f"alpha{int(round(10 * alpha)):02d}"                 # 0.3 -> alpha03
        REG.register(f"13_{key}_bpt_{branch_tag}_p_wet_wy{forecast_wy}", prob_between(wg99_curves[(cat_name, alpha)], *forecast_wet_season),
                     "1", 2, where="P10, M10", cell="wg99", sidecar=sidecar)
# the falsification dates of the InSAR BPT
for level in falsification_levels:
    level_tag = f"s{int(round(100 * level)):02d}"                         # 0.05 -> s05, 0.01 -> s01
    REG.register(f"13_insar_bpt_falsify_{level_tag}_date", falsification_dates[level], "date", 0, where="P10", cell="wg99", sidecar=sidecar)
REG.render()
