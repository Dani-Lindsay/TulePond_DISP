# %% [markdown]
# # [1] 11 The next-event forecast and the rainfall conditioning
#
# ```
# Created on Thu Sep 24 2026
# @author: daniellelindsay
# Run in opera_disp_feb13. The forecast (400 timing draws per model) takes ~1 min.
# ```
#
# **What this notebook does.** Everything the paper predicts is here, in one
# place, from the event catalogues (04, 09) and the rain table (10):
#
# 1. **Renewal forecast** of the next onset: onsets drawn uniformly within
#    their windows (no midpoints), intervals fitted by exponential, lognormal,
#    Weibull, BPT and Gaussian models with a parametric bootstrap, conditional
#    on no event through the last observation, `forecast_last_obs_date` (TulePond_filepaths:
#    the last descending acquisition of the CSLC + Dolphin extension check, 2026-10-05).
#    The paper's InSAR record still ends at the data end, the last acquisition on both
#    tracks (read from 03's fp_ts), registered as `11_data_end`. Four catalogues: InSAR only,
#    the bimonthly-era array record (with and without the Loma Prieta interval),
#    and the merged record. The InSAR clock is also given in closed form (mean
#    interval, t-based prediction interval), evaluated on the same kind of uniform
#    onset draws: the numbers the text quotes. It draws the paper's Fig 3.
# 2. **Rainfall conditioning**: for each onset window, the water-year rain accumulated
#    by then, the lag after fixed cumulative thresholds (rule A) and after the
#    first storm of the year (rule B). Then the test of a 3-day rain total before
#    the onset: does an InSAR onset fall within X days after a 3-day rain total of
#    at least Z mm more often than a random day of the same season in the same
#    water year? Only the five InSAR onsets and only the Fremont gauge; Z and X
#    are swept (16 settings) and the search is paid for.
# 3. **The prediction statement**, combining the recurrence window with the
#    calendar window.
#
# Two seasonal windows, each named where it is used: `forecast_wet_season` (Oct 1 to Apr 30
# of WY2027: every "P(WY2027 wet season)") and 10_rain_seasonality's `wet_season_months`
# (November to March: the onset-season count). The season of the rain-total test is not a fixed
# window: it is the days of the water year the InSAR onset draws fall on.
#
# PREPUB: Ported from `Hayward_DISP-S1/analysis/run21_recurrence_forecast.py` and
# PREPUB: `run22_precip_conditioning.py`; until 2026-09-24 this was the second half of
# PREPUB: the old notebook 09_seasonality_forecast. Its old registry rows (`10_issue_date`, `10_insar_*`,
# PREPUB: `10_aa_dense_*`, from `10_seasonality_forecast_v2`) are RETIRED in registry.txt: they are not
# PREPUB: 10_rain_seasonality's `10_*` rows, and must not be quoted.
#
# **Inputs**: 03 `fp_ts.csv` (the data end), 04 `creep_events.csv` (InSAR onset windows:
# `onset_lo`, the last acquisition before the step, which is 04's step epoch, to `onset_hi`, the
# first acquisition after it), 09 `hrkt_aa_events.csv` (array onset windows `onset_lo` to
# `onset_hi`; `onset_mid` = the window midpoint, for labels, plot positions and sort keys only) and
# `aa_insar_stacks.csv`, 10 `tulepond_daily_precip.csv`, `fig3_rain_panel.npz`
# and `fig3_onset_windows.csv`, the HRKT array file. The length of the running rain total
# (`rain_total_days`, 3 days) is TulePond_filepaths'.
#
# **Outputs** (`outputs/11_forecast/`): `recurrence_forecast_models.csv`,
# `precip_conditioning_{events,thresholds}.csv`, `rain_total_sweep.csv` (the 16 settings of the
# 3-day rain-total test: hits, null expectation, p), `prediction_statement.txt`,
# the stats sidecar `11_forecast_stats.txt`, registry rows `11_*`.
#
# **Figures**: the paper's Fig 3 (`fig3_history_forecast_ntiming400_nboot5_issue<date>`: A HRKT
# history, B onsets over the rain, C repeat intervals, D the forecast); the five renewal models
# (`qc11_renewal_models_single_fits`, working); the cumulative rain of every water year
# (`figS_wy_cumulative_rain_onsets`) and the rainfall conditioning with the p of every setting
# of the rain-total test (`figS_precip_conditioning_bestY<Y>mm_rain3d_sweep16_nnull<n>`), supplement.
#
# **What to check.** The InSAR onset windows fall in December to March (`11_onset_earliest`,
# `11_onset_latest`); the InSAR clock gives the next event about mid December 2026
# (`11_next_clock_date`, 95 % prediction interval `11_pi95_*`); the InSAR BPT 5-95 % window
# (`11_insar_bpt_q05`, `_q95`) sits inside the coming wet season; the array record without the
# Loma Prieta interval and the merged record give far lower wet-season probabilities
# (`11_aa_dense_nolp_bpt_p_wet_wy2027`, `11_merged_bpt_p_wet_wy2027`); the rain-total test: each
# InSAR window and the 30 days before it hold Fremont rain only (asserted; the days with no gauge
# are printed), and the search-paid p (`11_rain_sweep_p_search_paid`) is the number to quote,
# not the best setting's own p (`11_rain_sweep_best_p`).
# Every number is registered in the last cell (`11_*`).

# %%
# [2]
import os

import insar_utils as utils
from TulePond_filepaths import common_paths, paths_out, repo_dir, forecast_issue_date, forecast_last_obs_date, rain_total_days

import numpy as np                               # Matrix calculations
import pandas as pd                              # Tables
from scipy import stats                          # the renewal distributions
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator

# %% [markdown]
# ## [3] Paths and parameters
#
# The forecast settings, the WY2027 wet season, the rainfall-conditioning grids and the input files.

# %%
# [4]
notebook = "11_forecast"
out_dir = os.path.join(common_paths["out_dir"], notebook)
fig_dir = common_paths["fig_dir"]
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)


# the forecast
loma_prieta = 1989.79          # 17 Oct 1989 (as 09 and 10)
n_timing = 400                 # timing draws: onsets drawn uniformly in their windows
n_boot = 5                     # bootstrap refits per timing draw (per catalogue and model)
seed = 7                       # one random-number generator feeds every draw in order
# The forecast is conditioned on no event through the last observation, forecast_last_obs_date
# (TulePond_filepaths: the last descending acquisition of the CSLC + Dolphin extension check,
# extension_2026-09/RESULT.md, 2026-10-05; Dani 2026-10-05). The paper's InSAR record ends earlier, at
# the data end (the end of the fault-parallel series, read from the data below), and stays the record end.
# forecast_issue_date is only the date the statement was first written (fixed, so the file names hold);
# nothing is claimed for the months after the last observation.
forecast_wet_season = ("2026-10-01", "2027-04-30")      # Oct 1 to Apr 30: the "WY2027 wet season" of every P(wet season)
horizon = "2027-12-31"
models = ["exponential", "lognormal", "weibull", "bpt", "gaussian"]
grid = np.arange(2025.0, 2031.0, 1 / 365.25)   # the time axis of the forecast curves (daily)
resolvable_days = 130      # array windows longer than this do not constrain the season (09's dense_window_days)
min_interval_yr = 0.05     # bootstrap intervals shorter than this are clipped to it (counted and reported)
max_degenerate_fraction = 0.01   # at most this share of the forecast curves may come from a refit with zero survival

# the conditioning
y_grid = [50, 75, 100, 125, 150, 175, 200, 250, 300, 400]   # WY cumulative thresholds, mm (rule A)
z_grid = [15, 20, 25, 30, 40, 50, 75]                       # first-storm sizes (3-day totals), mm (rule B)
rain_total_grid_mm = [20, 30, 40, 50]                       # the rain-total test: 3-day rain total of at least Z mm ...
rain_lag_grid_d = [7, 14, 21, 30]                           # ... within X days before the InSAR onset
antecedent_storm_mm = 20                                    # the "days since a storm" column of the event table uses this smaller storm
wy_years = (1980, 2026)                                     # the water years drawn and scored
complete_wy_last = 2025                                     # the last complete water year (WY2026 ends after the issue date)

fp_ts_file = os.path.join(paths_out["03_creep_timeseries"], "fp_ts.csv")
events_file = os.path.join(paths_out["04_event_detection"], "creep_events.csv")
aa_events_file = os.path.join(paths_out["09_alignment_arrays"], "hrkt_aa_events.csv")
aa_stacks_file = os.path.join(paths_out["09_alignment_arrays"], "aa_insar_stacks.csv")
rain_file = os.path.join(paths_out["10_rain_seasonality"], "tulepond_daily_precip.csv")
rain_panel_file = os.path.join(paths_out["10_rain_seasonality"], "fig3_rain_panel.npz")
windows_file = os.path.join(paths_out["10_rain_seasonality"], "fig3_onset_windows.csv")
hrkt_file = os.path.join(common_paths["aa_dir"], "HRKT.csv")

REG = utils.Registry(notebook, common_paths, repo_dir)
REG.inputs(fp_ts_file, events_file, aa_events_file, aa_stacks_file, rain_file, rain_panel_file, windows_file, hrkt_file)

# %% [markdown]
# ## [5] Functions: dates and water years
#
# Decimal years to calendar dates and back, the day of the water year (Oct 1 = 1), and its label.

# %%
# [6]
def decyear(dates):
    """Decimal year of a DatetimeIndex, fraction of the actual year length."""
    dates = pd.DatetimeIndex(dates)
    year_start = pd.to_datetime(dates.year.astype(str) + "-01-01")
    next_year_start = pd.to_datetime((dates.year + 1).astype(str) + "-01-01")
    return dates.year + (dates - year_start) / (next_year_start - year_start)


def decyear_to_date(years):
    """The calendar day nearest to each decimal year, as a DatetimeIndex (an acquisition's own day)."""
    out = []
    for value in np.atleast_1d(np.asarray(years, float)):
        out.append(utils.decimal_year_to_day(value))
    return pd.DatetimeIndex(out)


def ymd(year):
    """One decimal year as 'YYYY-MM-DD'."""
    return decyear_to_date(year)[0].strftime("%Y-%m-%d")


def dowy_of(date):
    """(day of water year, water year) of one Timestamp; Oct 1 = 1."""
    wy = date.year + (1 if date.month >= 10 else 0)
    return (date - pd.Timestamp(f"{wy - 1}-10-01")).days + 1, wy


def dowy_label(day):
    """A day of water year as 'Mon DD' ('--' if undefined)."""
    if not np.isfinite(day):
        return "--"
    return (pd.Timestamp("2000-10-01") + pd.Timedelta(days=int(day) - 1)).strftime("%b %d")

# %% [markdown]
# ## [7] Load: the rain table and the event catalogues
#
# The rain table is notebook 10's composite (one value per day, Fremont, then Newark; Hayward is
# not in it). The catalogues: InSAR events from 04 (window `onset_lo` to `onset_hi`; `onset_lo` is
# the last acquisition before the step); the HRKT array events from 09, all from the bimonthly era
# (09 runs the detector only there), whose windows of 130 d or less resolve the season
# (asserted); and the merged record, the bimonthly-era array events and the InSAR events, with the
# years between the last array event (2006) and the first InSAR event (2017) treated as unobserved
# (annual surveys from mid-2007): no interval is formed across them, the clock simply restarts with
# the first InSAR detection. There are no array detections after 2006.

# %%
# [8]
rain = pd.read_csv(rain_file, index_col="date", parse_dates=True)
print(f"rain table: {len(rain)} days {rain.index[0].date()} to {rain.index[-1].date()}, columns {list(rain.columns)}")

creep_events = pd.read_csv(events_file)
insar_events = creep_events[creep_events.segment == "TULEPOND"].sort_values("onset_lo").reset_index(drop=True)
insar_events["onset_mid"] = 0.5 * (insar_events.onset_lo + insar_events.onset_hi)     # as 09's: sort key and plot position only
aa_dense = pd.read_csv(aa_events_file)
assert (aa_dense.window_days <= resolvable_days).all(), f"09 returned an array window longer than {resolvable_days} d"
# every catalogue carries a `src` column; an interval is only formed between two events of the same source, so the
# merged catalogue never bridges 2006 to 2017, when the arrays were surveyed annually and the section was unobserved
# by the clock's standard (Dani, 2026-09-24). The clock restarts with the first InSAR detection.
insar_events = insar_events.assign(src="InSAR")
aa_dense = aa_dense.assign(src="AA")
merged_events = pd.concat([aa_dense[["onset_lo", "onset_hi", "onset_mid", "src"]], insar_events[["onset_lo", "onset_hi", "onset_mid", "src"]]])
merged_events = merged_events.sort_values("onset_mid").reset_index(drop=True)
# name -> (catalogue, drop the Loma Prieta-spanning interval?); the combined record also leaves out the Loma
# Prieta-spanning interval (Dani, 2026-09-29)
catalogues = {"A_insar": (insar_events, False), "B_aa_dense": (aa_dense, False), "B_aa_dense_noLP": (aa_dense, True),
              "C_merged": (merged_events, True)}
n_insar_events = len(insar_events)
n_aa_events = len(aa_dense)
print(f"InSAR {n_insar_events} events; array bimonthly era {n_aa_events} events (windows <= {resolvable_days} d); "
      f"merged {len(merged_events)} events with 2006 to 2017 unobserved")

# the last date the data vouch for: the end of the fault-parallel series (03), i.e. the last acquisition seen by both tracks
data_end_yr = float(pd.read_csv(fp_ts_file, index_col=0, nrows=1).columns.astype(float).max())
data_end = (decyear_to_date(data_end_yr)[0] + pd.Timedelta(hours=12)).strftime("%Y-%m-%d")     # the columns are decimal years to 4 places: round to the day
# the reference date of the forecast: the last observation with no event (later than the record end)
assert pd.Timestamp(forecast_last_obs_date) >= pd.Timestamp(data_end), \
    f"forecast_last_obs_date {forecast_last_obs_date} is before the end of the InSAR record {data_end}"
last_obs = forecast_last_obs_date
last_obs_day_yr = float(decyear(pd.to_datetime([last_obs]))[0])        # the reference date of the forecast, on the day
print(f"InSAR record ends {data_end} (the last acquisition on both tracks); forecast conditioned on no event through "
      f"{last_obs} (forecast_last_obs_date, the extension check); statement first written {forecast_issue_date}")

# %% [markdown]
# ## [9] Renewal-model forecast of the next onset
#
# For each catalogue: onsets drawn uniformly in their windows (400 draws), the
# intervals fitted by each model, and for each fit a parametric bootstrap (5
# refits) of the conditional next-onset distribution given no event by the
# last observation. The quantiles and wet-season probabilities are pooled
# over all the draws and refits (a mixture over the onset timing, not a mix of models). Model families are
# compared by AICc on the median draw. A bootstrap interval shorter than `min_interval_yr` is clipped to it
# and a refit that gives the elapsed quiet time zero survival adds a curve of ones: both are counted and
# reported, and the second must stay under `max_degenerate_fraction` (asserted).
#
# One random-number generator (`seed`) feeds every draw in order, so the
# forecast reproduces exactly.

# %%
# [10]
def fit_interval_model(model, intervals):
    """The fitted scipy distribution of one renewal model to a set of intervals (yr).
    exponential: mean = the mean interval. lognormal: log-interval mean and sd (MLE). weibull: scipy MLE, location 0.
    bpt (Brownian passage time = inverse Gaussian, Matthews et al. 2002): mean mu = the mean interval; MLE of the shape
    lambda from 1/lambda = mean(1/x - 1/mu); aperiodicity alpha^2 = mu / lambda; scipy's invgauss(mu=alpha^2, scale=mu/alpha^2).
    gaussian: mean and sample sd."""
    intervals = np.asarray(intervals, float)
    if model == "exponential":                    # Poisson process: memoryless
        return stats.expon(scale=intervals.mean())
    if model == "lognormal":
        log_intervals = np.log(intervals)
        return stats.lognorm(s=max(log_intervals.std(), 1e-3), scale=np.exp(log_intervals.mean()))
    if model == "weibull":
        shape, _, scale = stats.weibull_min.fit(intervals, floc=0)
        return stats.weibull_min(c=shape, scale=scale)
    if model == "bpt":                            # Brownian passage time = inverse Gaussian
        mean = intervals.mean()
        alpha2 = max(mean * np.mean(1 / intervals - 1 / mean), 1e-4)
        return stats.invgauss(mu=alpha2, scale=mean / alpha2)
    if model == "gaussian":
        return stats.norm(loc=intervals.mean(), scale=max(intervals.std(ddof=1), 1e-3))
    raise ValueError(model)


def aicc(model, intervals):
    """Corrected Akaike information criterion of one model on the intervals, and its log-likelihood:
    AICc = -2 lnL + 2k + 2k(k + 1) / (n - k - 1), infinite when n - k - 1 <= 0."""
    dist = fit_interval_model(model, intervals)
    loglike = dist.logpdf(intervals).sum()
    n_param = 1 if model == "exponential" else 2
    n = len(intervals)
    if n - n_param - 1 > 0:
        correction = 2 * n_param * (n_param + 1) / (n - n_param - 1)
    else:
        correction = np.inf
    return -2 * loglike + 2 * n_param + correction, loglike


def interval_mask(onsets_sorted, src, drop_lp):
    """Which consecutive pairs of sorted onsets form a repeat interval: both from the same source (no interval
    bridges the unobserved 2006 to 2017 gap) and, if drop_lp, not spanning Loma Prieta."""
    keep = np.ones(len(onsets_sorted) - 1, bool)
    if src is not None:
        keep &= np.asarray(src)[:-1] == np.asarray(src)[1:]
    if drop_lp:
        keep &= ~((onsets_sorted[:-1] < loma_prieta) & (onsets_sorted[1:] > loma_prieta))
    return keep


def intervals_from(onsets, src=None, drop_lp=False):
    """Repeat intervals between sorted onsets (see interval_mask)."""
    order = np.argsort(onsets)
    onsets_sorted = np.asarray(onsets)[order]
    src_sorted = None if src is None else np.asarray(src)[order]
    return np.diff(onsets_sorted)[interval_mask(onsets_sorted, src_sorted, drop_lp)]


def conditional_cdf(dist, t_last, time_since_last):
    """P(next onset by t | no onset in the time_since_last after t_last), on the grid:
    (F(t - t_last) - F(elapsed)) / S(elapsed), zero before the reference date."""
    survival = dist.sf(time_since_last)
    assert np.isfinite(survival) and survival > 0, f"zero survival at {time_since_last:.2f} yr elapsed"
    cdf = (dist.cdf(grid - t_last) - dist.cdf(time_since_last)) / survival
    return np.clip(np.where(grid - t_last > time_since_last, cdf, 0.0), 0, 1)


def forecast(model, catalogue, drop_lp, t_last_window, ref_yr, rng):
    """Mixture forecast for one model and catalogue: n_timing draws of the onsets in their windows,
    each fitted and bootstrapped n_boot times. Returns (mean conditional cdf on the grid, mean interval, mean CoV,
    number of clipped bootstrap intervals, number of curves from a refit with zero survival)."""
    cdf_sum = np.zeros_like(grid)
    n_curves = 0
    n_clipped = 0
    n_degenerate = 0
    interval_means = []
    interval_covs = []
    for _ in range(n_timing):
        # 1) one realisation of the onsets, uniform in their windows, and its intervals
        intervals = intervals_from(rng.uniform(catalogue.onset_lo.values, catalogue.onset_hi.values), catalogue.src.values, drop_lp)
        # 2) the model fitted to those intervals
        fitted_dist = fit_interval_model(model, intervals)
        interval_means.append(fitted_dist.mean())
        interval_covs.append(np.sqrt(fitted_dist.var()) / fitted_dist.mean())
        # 3) the last onset, uniform in its window, and how long ago it was at the reference date
        t_last = rng.uniform(*t_last_window)
        time_since_last = ref_yr - t_last
        # 4) parametric bootstrap: refit to synthetic intervals from the fitted model, forecast from each refit
        for _ in range(n_boot):
            bootstrap_intervals = fitted_dist.rvs(size=len(intervals), random_state=rng)
            n_clipped += int(np.sum(bootstrap_intervals < min_interval_yr))
            bootstrap_intervals = np.clip(bootstrap_intervals, min_interval_yr, None)
            refit = fit_interval_model(model, bootstrap_intervals)
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


def quantile_date(cdf, q):
    """The grid date at which the cdf first reaches q."""
    return grid[min(np.searchsorted(cdf, q), len(grid) - 1)]


def prob_between(cdf, date_a, date_b):
    """P(onset between two calendar dates) from a cdf on the grid."""
    year_a, year_b = decyear(pd.to_datetime([date_a, date_b]))
    return float(np.interp(year_b, grid, cdf) - np.interp(year_a, grid, cdf))


# %% [markdown]
# ## [11] The five renewal models, one at a time
#
# Before the forecast, each model on its own, fitted once by maximum likelihood
# to the intervals between point onsets: for catalogue A (the InSAR events) 04's step epochs
# `onset_lo`, the last acquisition before each step; for catalogue B (the bimonthly-era array
# events) the window midpoints `onset_mid`. The forecast below never uses these single fits (it draws the onsets within
# their windows and bootstraps), but they show what each family assumes and what it would say. The
# number to compare is P(onset in the WY2027 wet season | no onset by the last observation). `describe_fit`
# prints the same lines for every model.

# %%
# [12]
insar_intervals = intervals_from(insar_events.onset_lo.values)          # point onsets, for illustration only
array_intervals = intervals_from(aa_dense.onset_mid.values)             # window midpoints, for illustration only
last_onset_lo = float(insar_events.onset_lo.iloc[-1])
time_since_last_yr = last_obs_day_yr - last_onset_lo
print(f"InSAR intervals (yr): {np.round(insar_intervals, 2)}; array intervals (yr): {np.round(array_intervals, 2)}")
print(f"last observation {last_obs} = {last_obs_day_yr:.3f}; last onset window start {last_onset_lo:.3f}; {time_since_last_yr:.2f} yr elapsed with no event")


def describe_fit(model, intervals, label):
    """One maximum-likelihood fit: mean, CoV, the family's own parameters, AICc, and the conditional
    wet-season probability. Returns the fitted distribution."""
    dist = fit_interval_model(model, intervals)
    aicc_value, loglike = aicc(model, intervals)
    cdf = conditional_cdf(dist, last_onset_lo, time_since_last_yr)
    p_wet = prob_between(cdf, *forecast_wet_season)
    if model == "exponential":
        own = f"rate {1 / dist.mean():.2f} /yr"
    elif model == "lognormal":
        own = f"median {dist.median():.2f} yr, sigma_log {dist.kwds['s']:.2f}"
    elif model == "weibull":
        own = f"shape k {dist.kwds['c']:.2f}, scale {dist.kwds['scale']:.2f} yr"
    elif model == "bpt":
        own = f"aperiodicity alpha {np.sqrt(dist.kwds['mu']):.2f}"
    else:
        own = f"sd {np.sqrt(dist.var()) * 365.25:.0f} d"
    print(f"  {label:<8} {model:<12} mean {dist.mean():.2f} yr, CoV {np.sqrt(dist.var()) / dist.mean():.2f}, {own}; "
          f"AICc {aicc_value:6.1f}; P(WY2027 wet season | quiet to last obs) {p_wet:.2f}; "
          f"5-50-95% {ymd(quantile_date(cdf, 0.05))} {ymd(quantile_date(cdf, 0.5))} {ymd(quantile_date(cdf, 0.95))}")
    return dist


single_fits = {}          # (catalogue, model) -> fitted distribution, for the comparison figure

# %% [markdown]
# ### [13] Exponential: the Poisson null
#
# One parameter, the mean interval. The hazard rate (the chance of an onset in
# the next instant, given none yet) is constant: the fault has no memory of
# when it last slipped, so the ~1.7 yr between the last (2024-25) event and the
# last observation changes nothing. This is the "no clock" null every time-dependent forecast is
# measured against (WGCEP 1988; Ellsworth et al. 1999). With a mean of 2 yr it
# gives roughly a 25 % chance of an onset in any given half-year, whenever
# that half-year is.

# %%
# [14]
print("exponential")
single_fits[("A", "exponential")] = describe_fit("exponential", insar_intervals, "A InSAR")
single_fits[("B", "exponential")] = describe_fit("exponential", array_intervals, "B array")

# %% [markdown]
# ### [15] Lognormal: the original time-dependent model
#
# The logarithm of the interval is Gaussian: two parameters, the median
# interval and the spread of the logarithm. The hazard rises after the last
# event, peaks near the median, then FALLS for long elapsed times, so an event
# that is very overdue is treated as less likely, not more. Nishenko and
# Buland (1987) introduced it for earthquake recurrence and the first Working
# Group on California Earthquake Probabilities (1988, 1990) used it.

# %%
# [16]
print("lognormal")
single_fits[("A", "lognormal")] = describe_fit("lognormal", insar_intervals, "A InSAR")
single_fits[("B", "lognormal")] = describe_fit("lognormal", array_intervals, "B array")

# %% [markdown]
# ### [17] Weibull: the ageing model
#
# Two parameters, a scale and a shape k. The hazard is a power of the elapsed
# time, t to the (k - 1): k = 1 is the exponential, k > 1 means the chance
# keeps rising the longer the fault waits (ageing, as in reliability
# engineering, where the family comes from), without limit. Hagiwara (1974)
# applied it to earthquakes; Utsu (1984) compared it with the others. A large
# k is a very regular clock.

# %%
# [18]
print("weibull")
single_fits[("A", "weibull")] = describe_fit("weibull", insar_intervals, "A InSAR")
single_fits[("B", "weibull")] = describe_fit("weibull", array_intervals, "B array")

# %% [markdown]
# ### [19] Brownian passage time: the physical model
#
# The inverse-Gaussian distribution of the first time a steadily loaded state
# variable, perturbed by Brownian noise, reaches a fixed threshold (Matthews,
# Ellsworth and Reasenberg 2002). Two parameters: the mean interval and the
# aperiodicity alpha, which is the CoV of the intervals. The hazard rises
# after the last event and then LEVELS OFF at a finite plateau, so an
# overdue event stays about equally likely rather than certain or fading.
# It is the standard of WGCEP 2003 and UCERF3 (Field et al. 2015) and the
# model the prediction statement quotes.

# %%
# [20]
print("bpt")
single_fits[("A", "bpt")] = describe_fit("bpt", insar_intervals, "A InSAR")
single_fits[("B", "bpt")] = describe_fit("bpt", array_intervals, "B array")

# %% [markdown]
# ### [21] Gaussian: the plain clock
#
# Mean and standard deviation of the intervals, the arithmetic the text
# quotes (next = last + mean, plus or minus the sd). Not a recurrence model in
# the forecasting literature: its hazard rises without bound and it allows
# negative intervals in principle. It is here because with four intervals the
# other families cannot be told apart, and the reader should see that the
# plain clock says the same thing they do.

# %%
# [22]
print("gaussian")
single_fits[("A", "gaussian")] = describe_fit("gaussian", insar_intervals, "A InSAR")
single_fits[("B", "gaussian")] = describe_fit("gaussian", array_intervals, "B array")

# %% [markdown]
# ### [23] The five side by side
#
# Left: the interval density of each model fitted to the InSAR intervals, the
# four intervals as ticks. Middle: the hazard rate, the chance of an onset per
# year given none yet, against time since the last event; this is where the
# families differ, and the dotted line marks where we are at the last observation.
# Right: the conditional forecast from each single fit, the wet season shaded.
# The array fits are the thin lines.

# %%
# [24]
utils.mpl_paper_style(base=8)          # Helvetica, 8 pt
model_colours = {"exponential": "0.4", "lognormal": "#0072B2", "weibull": "#CC79A7", "bpt": "#E69F00", "gaussian": "#56B4E9"}   # Okabe-Ito, no green
model_names = {"exponential": "Exponential", "lognormal": "Lognormal", "weibull": "Weibull", "bpt": "Brownian passage time", "gaussian": "Gaussian"}
insar_colour = "#D55E00"               # Okabe-Ito vermillion: InSAR in every figure of this notebook
array_colour = "#0072B2"               # Okabe-Ito blue: the alignment array
combined_colour = "#CC79A7"            # Okabe-Ito reddish purple: the combined record
onset_line_colour = "#E69F00"          # Okabe-Ito orange: InSAR onset lines
wet_shade_colour = "#56B4E9"           # Okabe-Ito sky blue: the wet season
interval_axis = np.linspace(0.05, 6.0, 600)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
for model in models:
    for cat, lw in (("A", 1.4), ("B", 0.7)):
        dist = single_fits[(cat, model)]
        axes[0].plot(interval_axis, dist.pdf(interval_axis), color=model_colours[model], lw=lw, label=model_names[model] if cat == "A" else None)
        hazard = dist.pdf(interval_axis) / np.clip(dist.sf(interval_axis), 1e-12, None)
        axes[1].plot(interval_axis, hazard, color=model_colours[model], lw=lw)
        axes[2].plot(grid, conditional_cdf(dist, last_onset_lo, time_since_last_yr), color=model_colours[model], lw=lw)
axes[0].plot(insar_intervals, np.full_like(insar_intervals, -0.1), "|", color=insar_colour, ms=10, mew=1.5, label="InSAR intervals")
axes[0].plot(array_intervals, np.full_like(array_intervals, -0.25), "|", color="0.2", ms=8, mew=1.0, label="Array intervals")
axes[0].set_xlim(0, 6)
axes[0].set_ylim(-0.4, None)
axes[0].set_xlabel("Interval (yr)")
axes[0].set_ylabel("Density (1/yr)")
utils.panel_title(axes[0], "A", "Interval density")
axes[0].legend(fontsize=6)
axes[1].axvline(time_since_last_yr, color="0.3", lw=0.7, ls=":")
axes[1].text(time_since_last_yr - 0.05, 0.97, "last obs.", transform=axes[1].get_xaxis_transform(), fontsize=6, color="0.3", va="top", ha="right")
axes[1].set_xlim(0, 6)
axes[1].set_yscale("log")
axes[1].set_ylim(0.01, 100)
axes[1].set_xlabel("Time since the last onset (yr)")
axes[1].set_ylabel("Hazard rate (1/yr), log")
utils.panel_title(axes[1], "B", "Hazard rate")
axes[2].axvspan(*decyear(pd.to_datetime(list(forecast_wet_season))), color=wet_shade_colour, alpha=0.15, lw=0)
axes[2].axvline(last_obs_day_yr, color="0.3", lw=0.7, ls="-.")
axes[2].set_xlim(2026.0, 2030.0)
axes[2].set_xlabel("Year")
axes[2].set_ylabel("P(onset by t)")
utils.panel_title(axes[2], "C", "Forecast from each single fit")
fig.tight_layout()
f_models = os.path.join(fig_dir, "qc11_renewal_models_single_fits.png")
fig.savefig(f_models, dpi=300)
fig.savefig(f_models.replace(".png", ".pdf"))
print("Saved figure to:", f_models)
plt.show()

# %% [markdown]
# ## [25] The forecast: onsets drawn within their windows (Fig 3D and the registered window)
#
# Now the real thing: for each catalogue and model, 400 draws of the onsets
# within their windows, each fitted and bootstrapped, conditioned on no event
# through the last observation. The spread of the intervals over the timing draws (mean
# and sd of each draw, averaged) is kept for Fig 3C.

# %%
# [26]
rng = np.random.default_rng(seed)
t_last_window = (float(insar_events.onset_lo.iloc[-1]), float(insar_events.onset_hi.iloc[-1]))
forecast_rows = []
forecast_curves = {}
interval_draw_stats = {}          # catalogue -> (mean interval, mean sd) over the timing draws (Fig 3C)
stats_lines = ["11 FORECAST AND RAINFALL CONDITIONING", "", "RENEWAL FORECAST",
               f"last event onset window {ymd(t_last_window[0])} to {ymd(t_last_window[1])}; reference date (last observation) {last_obs}; InSAR record end {data_end}; "
               f"{n_timing} timing draws x {n_boot} bootstrap refits; WY2027 wet season {forecast_wet_season[0]}..{forecast_wet_season[1]}"]
for cat_name, (catalogue, drop_lp) in catalogues.items():
    # the spread of the intervals over the timing draws, and the AICc of each model on the median draw
    interval_draws = []
    for _ in range(n_timing):
        interval_draws.append(intervals_from(rng.uniform(catalogue.onset_lo.values, catalogue.onset_hi.values), catalogue.src.values, drop_lp))
    draw_means = np.array([draw.mean() for draw in interval_draws])
    draw_sds = np.array([draw.std(ddof=1) for draw in interval_draws])
    interval_draw_stats[cat_name] = (float(draw_means.mean()), float(draw_sds.mean()))
    median_draw = interval_draws[int(np.argsort(draw_means)[len(draw_means) // 2])]
    aicc_by_model = {}
    for model in models:
        aicc_by_model[model] = aicc(model, median_draw)[0]
    best_model = min(aicc_by_model, key=aicc_by_model.get)
    aicc_text = []
    for model in models:
        aicc_text.append(f"{model} {aicc_by_model[model]:.1f}")
    lp_text = " (Loma Prieta-spanning intervals dropped)" if drop_lp else ""
    stats_lines += [f"{cat_name}: {len(catalogue)} events, {len(interval_draws[0])} intervals" + lp_text,
                    f"  interval mean {draw_means.mean():.2f} yr, sd {draw_sds.mean():.2f} yr, CoV {np.mean(draw_sds / draw_means):.2f}; AICc: " +
                    ", ".join(aicc_text)]
    for model in models:
        cdf, interval_mean, interval_cov, n_clipped, n_degenerate = forecast(model, catalogue, drop_lp, t_last_window, last_obs_day_yr, rng)
        forecast_curves[(cat_name, model)] = cdf
        row = dict(catalog=cat_name, model=model, ref="last_obs", ref_date=last_obs, n_intervals=len(interval_draws[0]),
                   mean_yr=interval_mean, cov=interval_cov,
                   aicc=aicc_by_model[model], daicc=aicc_by_model[model] - aicc_by_model[best_model],
                   q05=ymd(quantile_date(cdf, 0.05)), q50=ymd(quantile_date(cdf, 0.5)), q95=ymd(quantile_date(cdf, 0.95)),
                   p_wet_season=prob_between(cdf, *forecast_wet_season),
                   p_by_2027=float(np.interp(decyear(pd.to_datetime([horizon]))[0], grid, cdf)),
                   n_bootstrap_intervals_clipped=n_clipped, n_curves_zero_survival=n_degenerate)
        forecast_rows.append(row)
        stats_lines.append(f"    {model:<12} mean {interval_mean:.2f} yr CoV {interval_cov:.2f} | next onset 5-50-95%: "
                           f"{row['q05']} {row['q50']} {row['q95']} | P(wet season WY2027) {row['p_wet_season']:.2f}  "
                           f"P(by end 2027) {row['p_by_2027']:.2f} | clipped intervals {n_clipped}, zero-survival curves {n_degenerate}")
forecasts = pd.DataFrame(forecast_rows)
# no quantile may fall before the conditioning date (the conditional cdf is zero up to it)
for quantile_column in ("q05", "q50", "q95"):
    assert (pd.to_datetime(forecasts[quantile_column]) > pd.Timestamp(last_obs)).all(), \
        f"a forecast {quantile_column} falls on or before the last observation {last_obs}"
forecasts.to_csv(os.path.join(out_dir, "recurrence_forecast_models.csv"), index=False, float_format="%.3f")
print("\n".join(stats_lines))

# %% [markdown]
# ## [27] The InSAR clock in closed form
#
# The plain arithmetic the text quotes, on each of `n_timing` draws of the InSAR onsets uniformly
# within their windows (a generator of its own, `seed`, so the forecast above is untouched): the mean
# and sd of the intervals, next = last + mean, and a t-based 95 % prediction interval for a single
# new interval, mean +/- t(n-1) s sqrt(1 + 1/n). The registered numbers are the averages over the
# draws; the spread of the next date over the draws is printed alongside.

# %%
# [28]
def clock_prediction(intervals_yr, last_onset):
    """The plain clock from one set of intervals: (mean, sd, next = last + mean, 95 % prediction half-width
    t(n-1, 0.975) x sd x sqrt(1 + 1/n)), all in years."""
    n = len(intervals_yr)
    mean_yr = float(np.mean(intervals_yr))
    sd_yr = float(np.std(intervals_yr, ddof=1))
    half_width = float(stats.t.ppf(0.975, n - 1) * sd_yr * np.sqrt(1 + 1 / n))
    return mean_yr, sd_yr, last_onset + mean_yr, half_width


clock_rng = np.random.default_rng(seed)
clock_draws = []
interval_sum = None
for _ in range(n_timing):
    onsets_drawn = np.sort(clock_rng.uniform(insar_events.onset_lo.values, insar_events.onset_hi.values))
    intervals_drawn = np.diff(onsets_drawn)
    clock_draws.append(clock_prediction(intervals_drawn, onsets_drawn[-1]))
    interval_sum = intervals_drawn if interval_sum is None else interval_sum + intervals_drawn
clock_draws = np.array(clock_draws)
n_intervals = n_insar_events - 1
clock_intervals = interval_sum / n_timing                 # each interval, averaged over the draws (for the printout)
interval_mean_yr = float(clock_draws[:, 0].mean())
interval_sd_yr = float(clock_draws[:, 1].mean())
next_onset_yr = float(clock_draws[:, 2].mean())
half_width_yr = float(clock_draws[:, 3].mean())
next_spread_days = float(clock_draws[:, 2].std() * 365.25)
sd_ci_days = (interval_sd_yr * np.sqrt((n_intervals - 1) / stats.chi2.ppf(.975, n_intervals - 1)) * 365.25,
              interval_sd_yr * np.sqrt((n_intervals - 1) / stats.chi2.ppf(.025, n_intervals - 1)) * 365.25)
clock_interval_text = []
for value in clock_intervals:
    clock_interval_text.append(f"{value:.2f}")
clock_lines = ["InSAR clock, closed form, averaged over " + f"{n_timing} uniform onset draws: intervals " + ", ".join(clock_interval_text) +
               f" yr; mean {interval_mean_yr:.3f} yr, sd {interval_sd_yr * 365.25:.0f} d (95% CI {sd_ci_days[0]:.0f}-{sd_ci_days[1]:.0f} d), "
               f"CoV {interval_sd_yr / interval_mean_yr:.3f}",
               f"  next = last + mean = {next_onset_yr:.3f} ({ymd(next_onset_yr)}; spread over the draws {next_spread_days:.0f} d sd); "
               f"+/-1 sd {ymd(next_onset_yr - interval_sd_yr)}..{ymd(next_onset_yr + interval_sd_yr)}; "
               f"95% prediction interval (t, n={n_intervals}) {ymd(next_onset_yr - half_width_yr)}..{ymd(next_onset_yr + half_width_yr)}"]
# the clock is not conditioned on the quiet time: say which of its dates already lie before the last observation
clock_dates = {"next": next_onset_yr, "-1 sd": next_onset_yr - interval_sd_yr, "+1 sd": next_onset_yr + interval_sd_yr,
               "PI95 lo": next_onset_yr - half_width_yr, "PI95 hi": next_onset_yr + half_width_yr}
clock_before = []
for name, value in clock_dates.items():
    if value <= last_obs_day_yr:
        clock_before.append(f"{name} {ymd(value)}")
clock_lines.append(f"  clock dates on or before the last observation {last_obs} (the clock is unconditional): " +
                   (", ".join(clock_before) if clock_before else "none"))
stats_lines += clock_lines
print("\n".join(clock_lines))

# %% [markdown]
# ## [29] Figure 3: the HRKT history, the onsets on the water year, the repeat intervals and the forecast
#
# Left column on one time axis: A) the HRKT record with the bimonthly-era detections shaded, the
# annual-survey years greyed, the InSAR stack tied to it and the InSAR onsets dashed, HRKT since 2015
# as an inset (HGIL dropped, Dani 2026-10-01); B) the onset windows on the water year over the 7-day
# rain (from 10_rain_seasonality). Right column: C) the repeat intervals of the three catalogues the
# text quotes (the array and combined records leave out the interval spanning Loma Prieta): each
# interval as a bar over its possible range (window to window, no point), the mean and sd over the
# onset draws of [26] marked and written (registered: `11_*_fig3c_interval_*`); D) the conditional
# probability of the next onset, BPT, one line per catalogue, the 2026-27 wet season shaded.

# %%
# [30]
utils.mpl_paper_style(base=8)          # Helvetica, 8 pt, editable PDF text
shown = ["A_insar", "B_aa_dense_noLP", "C_merged"]       # the array without the Loma Prieta interval; B (with it) stays in the table
colours = {"A_insar": insar_colour, "B_aa_dense_noLP": array_colour, "C_merged": combined_colour}
labels = {"A_insar": "InSAR", "B_aa_dense_noLP": "A. Array", "C_merged": "Combined"}
label_pt = 7

# the left column's inputs, saved by 09 and 10
rain_panel = np.load(rain_panel_file)
windows = pd.read_csv(windows_file)
aa_stacks = pd.read_csv(aa_stacks_file)
hrkt = utils.load_aa(common_paths["aa_dir"], "HRKT")
hrkt_stack = aa_stacks[aa_stacks.code == "HRKT"]
x0 = float(rain_panel["x0"])
x1 = float(rain_panel["x1"])
dense_era_end = float(rain_panel["dense_era_end"])
# the catalogues drawn on the left are the ones the forecast fits: same onsets, in the same order
assert np.allclose(np.sort(windows.query("src == 'insar'").onset_lo), np.sort(catalogues["A_insar"][0].onset_lo))
assert np.allclose(np.sort(windows.query("src == 'aa_dense'").onset_lo), np.sort(aa_dense.onset_lo))


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


# full Science width (7.2 in = 18.3 cm); panels placed by hand (inches) so the columns align whatever the labels
fig_w = 7.2
fig_h = 5.05


def place(x, y, w, h):
    """An axes at (x, y) with width w and height h, all in inches on the figure."""
    return fig.add_axes([x / fig_w, y / fig_h, w / fig_w, h / fig_h])


fig = plt.figure(figsize=(fig_w, fig_h))
# all four panels the same height (Dani 2026-10-01); the row gap leaves room for C's x label above D's title
left_x = 0.50
left_w = 4.05
right_x = 5.20
right_w = 1.88
panel_h = 1.85
row_y = (2.90, 0.42)
ax_a = place(left_x, row_y[0], left_w, panel_h)
ax_b = place(left_x, row_y[1], left_w, panel_h)
ax_c = place(right_x, row_y[0], right_w, panel_h)
ax_d = place(right_x, row_y[1], right_w, panel_h)
for ax in (ax_a, ax_b, ax_c, ax_d):
    ax.tick_params(labelsize=label_pt)

# A) the HRKT record
ax = ax_a
ax.plot(hrkt.year, hrkt.cum_mm, "o", ms=1.6, color=array_colour, lw=0.6, ls="-", zorder=3,
        label=f"Alignment array ({hrkt.year.min():.0f} to {hrkt.year.max():.0f})")
ax.plot(hrkt_stack.date, hrkt_stack.slip_mm, "-", lw=0.8, color=insar_colour, zorder=5,
        label=f"InSAR FP stack (n = {int(hrkt_stack.n_crossings.iloc[0])})")       # shifted onto the array datum (caption)
for i, window in enumerate(windows.query("src == 'aa_dense'").itertuples()):
    ax.axvspan(window.onset_lo, window.onset_hi, color=array_colour, alpha=0.20, lw=0, zorder=1,
               label="Alignment array detection" if i == 0 else None)
ax.axvspan(dense_era_end, float(rain_panel["insar_start"]), color="0.5", alpha=0.10, lw=0, zorder=0, label="Annual surveys")
for i, step_epoch in enumerate(windows.query("src == 'insar'").onset_lo):       # at each step epoch (the window is too narrow to see)
    ax.axvline(step_epoch, color=onset_line_colour, lw=0.9, ls="--", zorder=2, label="InSAR detection" if i == 0 else None)
ax.axvline(float(rain_panel["loma_prieta"]), color="0.25", lw=0.7, ls=":", zorder=2)
ax.text(float(rain_panel["loma_prieta"]) + 0.3, 0.04, "Loma Prieta", transform=ax.get_xaxis_transform(), fontsize=label_pt, color="0.25",
        ha="left", va="bottom", bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))     # between the array curve and the x axis
ax.set_ylabel("Cumulative slip (mm)", fontsize=label_pt)
handles, leg_labels = ax.get_legend_handles_labels()
legend_order = []                                       # the grey "Annual surveys" last
for i, leg_label in enumerate(leg_labels):
    if leg_label != "Annual surveys":
        legend_order.append(i)
legend_order.append(leg_labels.index("Annual surveys"))
ordered_handles = []
ordered_labels = []
for i in legend_order:
    ordered_handles.append(handles[i])
    ordered_labels.append(leg_labels[i])
ax.legend(ordered_handles, ordered_labels, fontsize=label_pt, loc="lower right", frameon=True,
          facecolor="white", edgecolor="none", framealpha=0.9)
ax.set_title(r"$\bf{A}$  HRKT alignment array, detected events shaded", loc="left", fontsize=8)
ax.set_xlim(x0, x1)                                  # year labels but no "Year" title: it fills the row gap (Dani 2026-10-01)
# HRKT since 2015 as one larger inset (Dani 2026-10-01: HGIL dropped), zeroed at its first survey
zoom_t0 = 2015.0
recent = hrkt[hrkt.year >= zoom_t0]
zero = float(recent.cum_mm.iloc[0])
inset = ax.inset_axes([0.05, 0.50, 0.40, 0.46])      # top left, above the array curve, which reaches this height only after 2010
inset.plot(recent.year, recent.cum_mm - zero, "o", ms=1.6, color=array_colour, lw=0.6, ls="-")
inset.plot(hrkt_stack.date, hrkt_stack.slip_mm - zero, "-", lw=0.7, color=insar_colour)
for step_epoch in windows.query("src == 'insar'").onset_lo:
    inset.axvline(step_epoch, color="0.7", lw=0.5, ls="--", zorder=0)
inset.set_xlim(zoom_t0, 2026.2)
inset.set_xticks([2016, 2018, 2020, 2022, 2024])
inset.tick_params(labelsize=6, length=1.5, pad=1.5)
inset.text(0.04, 0.95, "HRKT since 2015", transform=inset.transAxes, fontsize=label_pt, va="top")

# B) the onset windows on the water year over the 7-day rain
ax = ax_b
rain_cmap = LinearSegmentedColormap.from_list("rain", list(rain_panel["rain_colours"]))
density_max = float(rain_panel["density_max"])
density_days = int(rain_panel["density_days"])
mesh = ax.pcolormesh(rain_panel["x_edges"], np.arange(-0.5, 367.5), rain_panel["rain_grid"], cmap=rain_cmap, vmin=0, vmax=density_max,
                     shading="flat", rasterized=True, zorder=0)
for window in windows.itertuples():
    bar_colour = "0.15" if window.src == "aa_dense" else insar_colour
    onset_bar(ax, window.onset_mid, window.dowy_lo, window.dowy_hi, bar_colour, 1.4)
ax.set_ylim(366, 1)
ax.set_yticks([1, 62, 124, 182, 244, 305, 366])
ax.set_yticklabels(["Oct 1", "Dec 1", "Feb 1", "Apr 1", "Jun 1", "Aug 1", "Oct 1"])
ax.set_xlabel("Year", fontsize=label_pt)
ax.set_ylabel("Day of water year", fontsize=label_pt)
ax.add_patch(plt.Rectangle((0.575, 0.04), 0.415, 0.24, transform=ax.transAxes, facecolor="white", edgecolor="none", alpha=0.85, zorder=4))
cax = ax.inset_axes([0.61, 0.11, 0.35, 0.035])
cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="max")
cb.set_label(f"Precipitation, {density_days}-day total (mm)", fontsize=label_pt, labelpad=2)
cb.ax.xaxis.set_label_position("top")
cb.ax.tick_params(labelsize=label_pt, length=2, pad=1.5)
ax.legend(handles=[Line2D([], [], color="0.15", lw=1.4, label="Alignment array onset window"),
                   Line2D([], [], color=insar_colour, lw=1.4, label="InSAR onset window")], fontsize=label_pt, loc="lower left",
          frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
ax.set_title(r"$\bf{B}$  Onset windows on the water year, over 7-day rainfall", loc="left", fontsize=8)
ax.set_xlim(x0, x1)
for ax in (ax_a, ax_b):                              # Roland: annual ticks, years labelled every 5 (B; A shares the axis)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

# C) the repeat intervals of each catalogue: each interval as a bar over its possible range (from the end of the earlier
# window to the start of the later one, up to the start of the earlier to the end of the later), the mean of the onset
# draws marked, mean and sd written under it
ax = ax_c
tick_labels = []
fig3c_numbers = {}                 # catalogue -> (mean, sd, number of intervals), registered below
for j, cat_name in enumerate(shown):
    catalogue, drop_lp = catalogues[cat_name]
    catalogue = catalogue.sort_values("onset_mid").reset_index(drop=True)
    keep = interval_mask(catalogue.onset_mid.values, catalogue.src.values, drop_lp)     # which pairs form an interval (order and Loma Prieta side)
    shortest = (catalogue.onset_lo.values[1:] - catalogue.onset_hi.values[:-1])[keep]
    longest = (catalogue.onset_hi.values[1:] - catalogue.onset_lo.values[:-1])[keep]
    rows_y = j + np.linspace(-0.15, 0.15, len(shortest))
    ax.hlines(rows_y, shortest, longest, color=colours[cat_name], lw=1.6)
    mean_yr, sd_yr = interval_draw_stats[cat_name]
    fig3c_numbers[cat_name] = (mean_yr, sd_yr, len(shortest))
    ax.plot([mean_yr, mean_yr], [j - 0.3, j + 0.3], "-", color=colours[cat_name], lw=1.5)
    ax.text(mean_yr, j + 0.36, f"{mean_yr:.2f} $\\pm$ {sd_yr:.2f} yr", ha="center", va="top", fontsize=7, color=colours[cat_name])
    tick_labels.append(f"{labels[cat_name]}\n(n = {len(shortest)})")      # the spans are read from A (Dani 2026-10-01)
ax.set_yticks(range(len(shown)))
ax.set_yticklabels(tick_labels, fontsize=label_pt, linespacing=1.1)
ax.set_ylim(len(shown) - 0.35, -0.6)          # inverted: first catalogue on top; room under the last bar for its number
ax.set_xlim(0, 6)
ax.set_xlabel("Repeat interval (yr)", fontsize=label_pt)
ax.set_title(r"$\bf{C}$  SSE repeat intervals", loc="left", fontsize=8)

# D) the conditional forecast, BPT, one line per catalogue; the wet season shaded, the last observation marked
ax = ax_d
wet_years = decyear(pd.to_datetime(list(forecast_wet_season)))
ax.axvspan(wet_years[0], wet_years[1], color=wet_shade_colour, alpha=0.15, lw=0)
for cat_name in shown:
    ax.plot(grid, forecast_curves[(cat_name, "bpt")], "-", color=colours[cat_name], lw=1.2, label=labels[cat_name])
ax.axvline(last_obs_day_yr, color="0.3", lw=0.7, ls="-.")
ax.text(last_obs_day_yr - 0.06, 0.98, "Last observation", fontsize=label_pt, color="0.3", va="top", ha="right", rotation=90)   # left of the line: the InSAR curve rises just right of it
ax.set_xlim(2025.5, 2030.0)
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.set_ylim(0, 1.02)
ax.set_xlabel("Year", fontsize=label_pt)
ax.set_ylabel("Cumulative probability", fontsize=label_pt)
ax.set_title(r"$\bf{D}$  Probability of the next SSE", loc="left", fontsize=8)
ax.legend(fontsize=label_pt, loc="lower right", handlelength=1.2, handletextpad=0.5, borderpad=0.3, labelspacing=0.3,
          frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
f_fig3 = os.path.join(fig_dir, f"fig3_history_forecast_ntiming{n_timing}_nboot{n_boot}_issue{forecast_issue_date}.png")
fig.savefig(f_fig3, dpi=600)
fig.savefig(f_fig3.replace(".png", ".pdf"))
print("Saved Fig 3 to:", f_fig3)
for cat_name, (mean_yr, sd_yr, n_shown) in fig3c_numbers.items():
    print(f"Fig 3C {labels[cat_name]}: {n_shown} intervals, mean {mean_yr:.3f} yr, sd {sd_yr:.3f} yr (over the onset draws of [26])")
    stats_lines.append(f"Fig 3C {cat_name}: {n_shown} intervals, mean {mean_yr:.3f} yr, sd {sd_yr:.3f} yr (over the onset draws)")
plt.show()

# %% [markdown]
# ## [31] Rainfall conditioning of the onset timing
#
# Every quantity is a range over the onset window (the event is equally likely
# anywhere in it). Rule A: lag after the water-year cumulative first exceeds
# Y mm. Rule B: lag after the first 3-day storm of at least Z mm. Both rules use
# 10's composite rain (Fremont, then Newark) and are scored on the InSAR onsets,
# with the array onsets as a consistency count. The calendar spread of the InSAR
# onsets is the baseline any rule has to beat. The test of a 3-day rain total
# before the onset, on the InSAR onsets and the Fremont gauge only, is the next
# cell pair ([33], [34]).

# %%
# [32]
def wy_series(wy):
    """One water year of the daily rain with its cumulative total and running rain_total_days-day storm total."""
    this_year = rain[rain.wy == wy].copy()
    this_year["cum"] = this_year.prcp_mm.fillna(0).cumsum()
    this_year["storm"] = this_year.prcp_mm.fillna(0).rolling(rain_total_days, min_periods=1).sum()
    return this_year


def first_date(this_year, column, threshold):
    """The first day of the water year on which `column` reaches `threshold` (NaT if never)."""
    hits = this_year.index[this_year[column] >= threshold]
    return hits[0] if len(hits) else pd.NaT


def lag_days(day, crossing):
    """Days from a threshold crossing to a day (NaN if the threshold was never reached)."""
    if pd.isna(crossing):
        return np.nan
    return (day - crossing).days


# antecedent rain at every day: 30-day total, largest storm in 30 days, days since the last >= antecedent_storm_mm storm
rain_daily = rain.prcp_mm.fillna(0)
storm_3d = rain_daily.rolling(rain_total_days, min_periods=1).sum()
antecedent = pd.DataFrame(index=rain.index)
antecedent["rain_30d"] = rain_daily.rolling(30, min_periods=1).sum()
antecedent["max_storm_30d"] = storm_3d.rolling(30, min_periods=1).max()
last_storm_day = pd.Series(np.where(storm_3d >= antecedent_storm_mm, np.arange(len(rain)), np.nan), index=rain.index).ffill()
antecedent["days_since_storm20"] = np.arange(len(rain)) - last_storm_day

# per event (array bimonthly era + InSAR): the rain state over its onset window, and the lags after each rule A / rule B threshold
events = pd.concat([aa_dense[["onset_lo", "onset_hi"]].assign(source="AA"), insar_events[["onset_lo", "onset_hi"]].assign(source="InSAR")])
events = events.sort_values("onset_lo").reset_index(drop=True)
event_rows = []
for event in events.itertuples():
    date_lo = decyear_to_date(event.onset_lo)[0]
    date_hi = decyear_to_date(event.onset_hi)[0]
    _, wy = dowy_of(date_hi)
    this_year = wy_series(wy)
    cum_at_lo = float(this_year.cum.asof(date_lo)) if date_lo >= this_year.index[0] else 0.0
    row = dict(source=event.source, onset_lo=date_lo.date(), onset_hi=date_hi.date(), window_days=(date_hi - date_lo).days, wy=wy,
               dowy_lo=dowy_of(date_lo)[0], dowy_hi=dowy_of(date_hi)[0], wy_total_mm=float(this_year.cum.iloc[-1]),
               cum_at_lo=cum_at_lo, cum_at_hi=float(this_year.cum.asof(date_hi)), gauge=this_year.src.mode().iloc[0])
    row["frac_at_hi"] = row["cum_at_hi"] / row["wy_total_mm"]
    window_antecedent = antecedent.loc[date_lo:date_hi]
    for column in ("rain_30d", "max_storm_30d", "days_since_storm20"):
        row[f"{column}_min"] = float(window_antecedent[column].min())
        row[f"{column}_max"] = float(window_antecedent[column].max())
    for Y in y_grid:
        crossing = first_date(this_year, "cum", Y)
        row[f"lagY{Y}_lo"] = lag_days(date_lo, crossing)
        row[f"lagY{Y}_hi"] = lag_days(date_hi, crossing)
    for Z in z_grid:
        crossing = first_date(this_year, "storm", Z)
        row[f"lagZ{Z}_lo"] = lag_days(date_lo, crossing)
        row[f"lagZ{Z}_hi"] = lag_days(date_hi, crossing)
    event_rows.append(row)
event_rain = pd.DataFrame(event_rows)
event_rain.to_csv(os.path.join(out_dir, "precip_conditioning_events.csv"), index=False, float_format="%.1f")
insar_rain = event_rain[event_rain.source == "InSAR"]
aa_rain = event_rain[event_rain.source == "AA"]
n_insar = len(insar_rain)
n_aa = len(aa_rain)
assert n_insar == n_insar_events and n_aa == n_aa_events


def rule_scores(kind, thresholds):
    """Rules A (kind 'Y', WY cumulative) and B (kind 'Z', first storm): for each threshold, the InSAR lag range,
    how many array events are consistent with it, and when the threshold is typically crossed."""
    scores = []
    for threshold in thresholds:
        lag_lo = insar_rain[f"lag{kind}{threshold}_lo"]
        lag_hi = insar_rain[f"lag{kind}{threshold}_hi"]
        if lag_lo.isna().any():
            scores.append(dict(kind=kind, thr=threshold, n_insar_reached=int(lag_lo.notna().sum())))
            continue
        insar_lag_min = int(lag_lo.min())
        insar_lag_max = int(lag_hi.max())
        aa_lag_lo = aa_rain[f"lag{kind}{threshold}_lo"]
        aa_lag_hi = aa_rain[f"lag{kind}{threshold}_hi"]
        reached = aa_lag_lo.notna()
        consistent = reached & (aa_lag_hi >= insar_lag_min) & (aa_lag_lo <= insar_lag_max)
        crossing_dowy = {}
        for wy in range(wy_years[0], wy_years[1] + 1):
            crossing = first_date(wy_series(wy), "cum" if kind == "Y" else "storm", threshold)
            crossing_dowy[wy] = dowy_of(crossing)[0] if pd.notna(crossing) else np.nan
        crossing_dowy = pd.Series(crossing_dowy)
        scores.append(dict(kind=kind, thr=threshold, n_insar_reached=n_insar, insar_lag_min=insar_lag_min, insar_lag_max=insar_lag_max,
                           insar_lag_spread=insar_lag_max - insar_lag_min, aa_reached=int(reached.sum()), aa_consistent=int(consistent.sum()),
                           wy_reached_frac=float(crossing_dowy.notna().mean()), cross_dowy_median=float(crossing_dowy.median()),
                           cross_dowy_10=float(crossing_dowy.quantile(.1)), cross_dowy_90=float(crossing_dowy.quantile(.9))))
    return pd.DataFrame(scores)


threshold_scores = pd.concat([rule_scores("Y", y_grid), rule_scores("Z", z_grid)])
threshold_scores.to_csv(os.path.join(out_dir, "precip_conditioning_thresholds.csv"), index=False, float_format="%.2f")

# the summary lines
calendar_spread = int(insar_rain.dowy_hi.max() - insar_rain.dowy_lo.min())
wy_total_series = rain.groupby("wy").prcp_mm.sum(min_count=1)
wy_total_series = wy_total_series[(wy_total_series.index >= wy_years[0]) & (wy_total_series.index <= complete_wy_last)]
conditioning_lines = ["", "RAINFALL CONDITIONING (per event: WY rain accumulated by the window start..end; antecedent quantities are min-max over the window)"]
for event in event_rain.itertuples():
    conditioning_lines.append(
        f"  {event.source:<5} {event.onset_lo}..{event.onset_hi} ({event.window_days:3d} d) WY{event.wy} total {event.wy_total_mm:5.0f} mm "
        f"(P{(wy_total_series < event.wy_total_mm).mean() * 100:2.0f}) | cum {event.cum_at_lo:4.0f}-{event.cum_at_hi:4.0f} mm, frac {event.frac_at_hi:.2f} | "
        f"rain 30 d {event.rain_30d_min:3.0f}-{event.rain_30d_max:3.0f} mm, max {rain_total_days}-d storm {event.max_storm_30d_min:3.0f}-{event.max_storm_30d_max:3.0f} mm, "
        f"{event.days_since_storm20_min:3.0f}-{event.days_since_storm20_max:3.0f} d since >= {antecedent_storm_mm} mm storm [{event.gauge}]")
wy_percentiles = []
for value in insar_rain.wy_total_mm:
    wy_percentiles.append(f"P{(wy_total_series < value).mean() * 100:.0f}")
conditioning_lines += [
    f"InSAR events: WY totals {insar_rain.wy_total_mm.min():.0f}-{insar_rain.wy_total_mm.max():.0f} mm (" + ", ".join(wy_percentiles) + ")",
    f"InSAR onsets: day-of-WY {insar_rain.dowy_lo.min():.0f}-{insar_rain.dowy_hi.max():.0f} ({dowy_label(insar_rain.dowy_lo.min())}-{dowy_label(insar_rain.dowy_hi.max())}), "
    f"calendar spread {calendar_spread} d",
    f"InSAR onsets: WY rain accumulated by the onset {insar_rain.cum_at_lo.min():.0f}-{insar_rain.cum_at_hi.max():.0f} mm, "
    f"fraction of the WY total {insar_rain.frac_at_hi.min():.2f}-{insar_rain.frac_at_hi.max():.2f}",
    "rule A (lag after WY cumulative exceeds Y):"]
for score in threshold_scores[threshold_scores.kind == "Y"].itertuples():
    if score.n_insar_reached < n_insar:
        conditioning_lines.append(f"  Y={score.thr:4.0f} mm: reached in only {score.n_insar_reached}/{n_insar} InSAR event years -> no rule")
    else:
        conditioning_lines.append(f"  Y={score.thr:4.0f} mm: lag {score.insar_lag_min:4.0f}..{score.insar_lag_max:4.0f} d (spread {score.insar_lag_spread:3.0f} d) | "
                                  f"AA consistent {score.aa_consistent:.0f}/{score.aa_reached:.0f} | "
                                  f"reached in {score.wy_reached_frac * 100:3.0f}% of WYs, crossing {dowy_label(score.cross_dowy_median)}")
conditioning_lines.append(f"rule B (lag after the first {rain_total_days}-day storm >= Z):")
for score in threshold_scores[threshold_scores.kind == "Z"].itertuples():
    if score.n_insar_reached < n_insar:
        conditioning_lines.append(f"  Z={score.thr:4.0f} mm: reached in only {score.n_insar_reached}/{n_insar} InSAR event years -> no rule")
    else:
        conditioning_lines.append(f"  Z={score.thr:4.0f} mm: lag {score.insar_lag_min:4.0f}..{score.insar_lag_max:4.0f} d (spread {score.insar_lag_spread:3.0f} d) | "
                                  f"AA consistent {score.aa_consistent:.0f}/{score.aa_reached:.0f} | first storm {dowy_label(score.cross_dowy_median)}")
# the tightest rule of each kind
reached_all = threshold_scores[threshold_scores.n_insar_reached == n_insar]
best_rule_a = reached_all[reached_all.kind == "Y"].sort_values("insar_lag_spread").iloc[0]
best_rule_b = reached_all[reached_all.kind == "Z"].sort_values("insar_lag_spread").iloc[0]
conditioning_lines.append(
    f"tightest rules: A Y={best_rule_a.thr:.0f} mm lag {best_rule_a.insar_lag_min:.0f}..{best_rule_a.insar_lag_max:.0f} d "
    f"(spread {best_rule_a.insar_lag_spread:.0f} vs calendar {calendar_spread} d); "
    f"B Z={best_rule_b.thr:.0f} mm lag {best_rule_b.insar_lag_min:.0f}..{best_rule_b.insar_lag_max:.0f} d (spread {best_rule_b.insar_lag_spread:.0f} d)")
stats_lines += conditioning_lines
print("\n".join(conditioning_lines))

# %% [markdown]
# ## [33] A 3-day rain total before the InSAR onsets: the sweep, with its search paid for
#
# Does an InSAR onset follow heavy rain more often than a random day of the same season would?
# A day counts as wet when its rain and that of the two days before it (`rain_total_days`) add up to
# at least Z mm; an onset is a hit when it falls within X days after a wet day. Z and X were not
# fixed in advance, so the test is the whole sweep: Z in `rain_total_grid_mm`, X in
# `rain_lag_grid_d`, 16 settings, and the search over them is paid for.
#
# Only the five InSAR onsets (their windows are one acquisition interval long) and only the
# Fremont gauge: the other gauges of 10's composite are there to draw the rain back in time. A day
# without a Fremont reading counts as 0 mm; each InSAR window and the `fremont_check_days` before
# it are checked to hold no other gauge's rain (the days without Fremont are printed).
#
# Each onset is drawn `n_draws` times, uniformly over the days of its window, both ends included
# (never the midpoint). The observed statistic is the number of hits out of five, averaged over the
# draws. The null keeps each event in its OWN water year and moves its onset to a day of the water
# year drawn from all the onset draws pooled (the season the onsets occur in), `n_null` times. For
# each setting, p = the fraction of null realisations with at least the observed hits. The best
# setting is the one with the smallest p; the search-paid p is the fraction of null realisations
# whose own best p over the 16 settings is at least as small. A random-number generator of its own
# (`seed`) leaves the forecast's draws untouched.

# %%
# [34]
n_draws = 2000                  # onset draws per InSAR event, uniform over the days of its window
n_null = 20000                  # null realisations: each event in its own water year, on a day of the onset season
fremont_check_days = 30         # each InSAR window and this many days before it must hold no rain but Fremont's


def days_since(flag):
    """Days since the last True in a daily boolean Series (NaN before the first)."""
    day_number = pd.Series(np.arange(len(flag)), index=flag.index, dtype=float)
    return day_number - day_number.where(flag).ffill()


def count_hits(days_since_wet, days, lag_d):
    """Hits summed over the events, one count per column of `days` (events x realisations, row numbers of the
    rain table): a hit is a day within lag_d days after a wet day."""
    return np.sum(days_since_wet[days] <= lag_d, axis=0)


def fraction_at_least(null_hits, value):
    """The fraction of the null realisations with at least `value` hits."""
    return float(np.mean(null_hits >= value))


# the Fremont gauge alone; a day without a Fremont reading counts as 0 mm in the 3-day totals
rain_step_days = np.diff(rain.index.values).astype("timedelta64[D]").astype(int)
assert (rain_step_days == 1).all(), "the rain table is not one row per day"
fremont_mm = rain.prcp_mm.where(rain.src == "FREMONT")
rain_3d_total = fremont_mm.fillna(0.0).rolling(rain_total_days, min_periods=1).sum()
row_of_day = pd.Series(np.arange(len(rain)), index=rain.index)

# each InSAR window: its first and last day as rows of the rain table, its water year, and the Fremont-only check
window_first_row = []
window_last_row = []
window_wy_start_row = []
window_wy = []
fremont_check_lines = []
for event in insar_events.itertuples():
    date_lo = decyear_to_date(event.onset_lo)[0]
    date_hi = decyear_to_date(event.onset_hi)[0]
    _, wy_lo = dowy_of(date_lo)
    _, wy_hi = dowy_of(date_hi)
    assert wy_lo == wy_hi, f"InSAR window {date_lo.date()} to {date_hi.date()} spans two water years"
    span = rain.loc[date_lo - pd.Timedelta(days=fremont_check_days):date_hi]
    not_fremont = span[span.src != "FREMONT"]
    assert set(not_fremont.src) <= {"none"}, \
        f"InSAR window {date_lo.date()} to {date_hi.date()}: days from {sorted(set(not_fremont.src) - {'none'})}, not Fremont"
    missing_text = []
    for day in not_fremont.index:
        missing_text.append(str(day.date()))
    fremont_check_lines.append(f"  InSAR window {date_lo.date()} to {date_hi.date()} and the {fremont_check_days} d before it: "
                               f"days without a Fremont reading (no gauge at all, counted as 0 mm): " + (", ".join(missing_text) if missing_text else "none"))
    window_first_row.append(int(row_of_day[date_lo]))
    window_last_row.append(int(row_of_day[date_hi]))
    window_wy_start_row.append(int(row_of_day[pd.Timestamp(f"{wy_lo - 1}-10-01")]))
    window_wy.append(wy_lo)

# the onset draws: each event n_draws times, uniformly over the days of its window, both ends included
sweep_rng = np.random.default_rng(seed)
draw_rows = np.empty((n_insar_events, n_draws), dtype=int)          # events x draws, rows of the rain table
for i in range(n_insar_events):
    draw_rows[i] = sweep_rng.integers(window_first_row[i], window_last_row[i] + 1, n_draws)
season_dowy = rain.dowy.values[draw_rows].ravel()                   # the season the onsets occur in: every onset draw, pooled

# the null: each event in its own water year, on a day of the water year drawn from the pooled onset draws
null_dowy = sweep_rng.choice(season_dowy, size=(n_insar_events, n_null))
null_rows = np.empty((n_insar_events, n_null), dtype=int)           # events x null realisations
for i in range(n_insar_events):
    null_rows[i] = window_wy_start_row[i] + null_dowy[i] - 1
    assert (rain.wy.values[null_rows[i]] == window_wy[i]).all(), f"a null day left water year {window_wy[i]}"

# the sweep: for each setting, the expected hits over the onset draws and the null hits
sweep_rows = []
null_hits_by_setting = []        # one array of n_null hit counts per setting, in the order of sweep_rows
for Z in rain_total_grid_mm:
    days_since_wet = days_since(rain_3d_total >= Z).values
    assert np.isfinite(days_since_wet[null_rows]).all(), f"no 3-day total of {Z} mm at Fremont before an InSAR season day"
    for X in rain_lag_grid_d:
        observed_hits = float(np.mean(count_hits(days_since_wet, draw_rows, X)))      # expected hits over the onset draws
        null_hits = count_hits(days_since_wet, null_rows, X)
        null_hits_by_setting.append(null_hits)
        sweep_rows.append(dict(Z_mm=Z, X_d=X, hits=observed_hits, hits_pct=100 * observed_hits / n_insar_events,
                               null_mean=float(null_hits.mean()), null_pct=100 * float(null_hits.mean()) / n_insar_events,
                               p=fraction_at_least(null_hits, observed_hits)))
rain_sweep = pd.DataFrame(sweep_rows)
n_settings = len(rain_sweep)
rain_sweep.to_csv(os.path.join(out_dir, "rain_total_sweep.csv"), index=False, float_format="%.4f")

# the best setting: the smallest p (the first in the sweep's order on a tie)
best_index = 0
for k in range(n_settings):
    if rain_sweep.p.iloc[k] < rain_sweep.p.iloc[best_index]:
        best_index = k
best_sweep = rain_sweep.iloc[best_index]

# the search paid for: each null realisation's own p in every setting (against the same null), its smallest over the
# settings, and how often that is at least as small as the observed best p
null_best_p = np.ones(n_null)
for k in range(n_settings):
    ordered = np.sort(null_hits_by_setting[k])
    p_each = (n_null - np.searchsorted(ordered, null_hits_by_setting[k], side="left")) / n_null      # = fraction of the null >= its own hits
    null_best_p = np.minimum(null_best_p, p_each)
p_search_paid = float(np.mean(null_best_p <= best_sweep.p))

# the summary lines and the 4 x 4 table of p (rows Z, columns X)
p_table = rain_sweep.pivot(index="Z_mm", columns="X_d", values="p")
sweep_lines = ["", f"A {rain_total_days}-DAY RAIN TOTAL BEFORE THE INSAR ONSETS (Fremont gauge only; {n_insar_events} InSAR onsets, each drawn "
               f"{n_draws} times uniformly in its window; null: own water year, a random day of the onset season, {n_null} realisations)"]
sweep_lines += fremont_check_lines
for row in rain_sweep.itertuples():
    sweep_lines.append(f"  {rain_total_days}-day total >= {row.Z_mm:2d} mm within {row.X_d:2d} d before the onset: hits {row.hits:4.2f} of {n_insar_events} "
                       f"({row.hits_pct:3.0f} %) vs {row.null_mean:4.2f} ({row.null_pct:3.0f} %) for a random day in the same season; p = {row.p:.3f}")
sweep_lines.append(f"p of each setting (rows: Z, mm; columns: X, d):\n{p_table.to_string(float_format=lambda value: f'{value:.3f}')}")
sweep_lines.append(f"{rain_total_days}-day rain total >= Z mm within X d before the InSAR onset: best Z/X {best_sweep.Z_mm:.0f} mm / {best_sweep.X_d:.0f} d: "
                   f"hits {best_sweep.hits:.2f} of {n_insar_events} vs {best_sweep.null_mean:.2f} expected for a random day in the same season, "
                   f"p = {best_sweep.p:.3f}; p = {p_search_paid:.3f} once all {n_settings} settings are paid for")
stats_lines += sweep_lines
print("\n".join(sweep_lines))

# %% [markdown]
# ## [35] Cumulative rain of every water year, with the onsets
#
# Is it the accumulated rain? Every water year's cumulative rain (10's composite)
# as a grey line, the InSAR event years in colour with the onset window
# as a thick segment, the array event years in dark grey. If the onsets waited
# for a fixed amount of rain the segments would sit on one horizontal line;
# instead they span the shaded band, a factor of several in accumulated rain
# (and in fraction of the year's total). Whether the InSAR onsets follow a short
# spell of heavy rain instead is the 3-day rain-total test of [34]
# (`11_rain_sweep_*`), quoted under the figure.

# %%
# [36]
fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
# the band of accumulated rain at the InSAR onsets (window start to window end)
cum_lo = float(insar_rain.cum_at_lo.min())
cum_hi = float(insar_rain.cum_at_hi.max())
ax.axhspan(cum_lo, cum_hi, color=insar_colour, alpha=0.08, lw=0)
# every water year
for wy in range(wy_years[0], wy_years[1] + 1):
    this_year = wy_series(wy)
    ax.plot(this_year.dowy, this_year.cum, "-", color="0.82", lw=0.6, zorder=1)
# the array event years (bimonthly era), onset window as a dark segment
for event in aa_rain.itertuples():
    this_year = wy_series(event.wy)
    ax.plot(this_year.dowy, this_year.cum, "-", color="0.55", lw=0.8, zorder=2)
    segment = this_year[(this_year.dowy >= event.dowy_lo) & (this_year.dowy <= event.dowy_hi)]
    ax.plot(segment.dowy, segment.cum, "-", color="0.2", lw=2.6, solid_capstyle="butt", zorder=3)
# the InSAR event years in colour, onset window as a thick vermillion segment, labelled by water year
event_colours = ["#0072B2", "#E69F00", "#CC79A7", "#000000", "#56B4E9"]   # Okabe-Ito, no green (Dani)
assert len(event_colours) >= n_insar, "more InSAR events than distinct colours"
for colour, event in zip(event_colours, insar_rain.itertuples()):
    this_year = wy_series(event.wy)
    ax.plot(this_year.dowy, this_year.cum, "-", color=colour, lw=1.4, zorder=4, label=f"WY{event.wy} ({event.wy_total_mm:.0f} mm)")
    ax.plot([event.dowy_lo, event.dowy_hi], [event.cum_at_lo, event.cum_at_hi], "-", color=insar_colour, lw=4, solid_capstyle="butt", zorder=5)
    ax.text(event.dowy_hi + 3, event.cum_at_hi, f"{event.cum_at_hi:.0f} mm, {event.frac_at_hi:.0%} of WY", fontsize=6, color=colour, va="center", zorder=6,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
ax.set_xlim(1, 366)
ax.set_xticks([1, 62, 124, 182, 244, 305, 366])
ax.set_xticklabels(["Oct 1", "Dec 1", "Feb 1", "Apr 1", "Jun 1", "Aug 1", "Oct 1"])
ax.set_ylabel("Cumulative precipitation in the water year (mm)")
ax.set_xlabel("Day of water year")
array_years_text = f"{int(np.floor(aa_dense.onset_lo.min()))} to {int(np.floor(aa_dense.onset_hi.max()))}"
legend_handles = ax.get_legend_handles_labels()[0]
legend_handles += [Line2D([], [], color=insar_colour, lw=4, label="InSAR onset window"),
                   Line2D([], [], color="0.2", lw=2.6, label=f"Array onset window ({array_years_text})"),
                   Line2D([], [], color="0.82", lw=0.6, label=f"Every water year {wy_years[0]} to {wy_years[1]}"),
                   plt.Rectangle((0, 0), 1, 1, facecolor=insar_colour, alpha=0.15, edgecolor="none",
                                 label=f"Rain accumulated at the InSAR onsets: {cum_lo:.0f} to {cum_hi:.0f} mm")]
ax.legend(handles=legend_handles, fontsize=6, loc="upper left")
f_cumulative = os.path.join(fig_dir, "figS_wy_cumulative_rain_onsets.png")
fig.savefig(f_cumulative, dpi=300)
fig.savefig(f_cumulative.replace(".png", ".pdf"))
print("Saved figure to:", f_cumulative)
plt.show()
print(f"rain accumulated at the InSAR onsets: {cum_lo:.0f} to {cum_hi:.0f} mm, {insar_rain.frac_at_hi.min():.0%} to {insar_rain.frac_at_hi.max():.0%} of the water-year total; "
      f"the 3-day rain-total test (Fremont): best {best_sweep.Z_mm:.0f} mm within {best_sweep.X_d:.0f} d, hits {best_sweep.hits:.2f} of {n_insar_events} "
      f"vs {best_sweep.null_mean:.2f} for a random day in the same season, p = {best_sweep.p:.3f}, {p_search_paid:.3f} with all {n_settings} settings paid for")

# %% [markdown]
# ## [37] The prediction statement
#
# Built from the numbers above in a fixed template, so the statement the paper
# makes is generated, not typed.

# %%
# [38]
insar_forecast = forecasts[forecasts.catalog == "A_insar"].set_index("model")
array_forecast = forecasts[forecasts.catalog == "B_aa_dense_noLP"].set_index("model")
merged_forecast = forecasts[forecasts.catalog == "C_merged"].set_index("model")
insar_window_days = (insar_events.onset_hi - insar_events.onset_lo) * 365.25
array_first_year = int(np.floor(aa_dense.onset_lo.min()))
statement_lines = [
    "TULE POND CREEP-EVENT PREDICTION STATEMENT (generated by notebook 11_forecast)",
    f"first written {forecast_issue_date}; conditioned on no event through the last observation, {last_obs} "
    f"(descending CSLC + Dolphin extension check, extension_2026-09/RESULT.md); "
    f"data: OPERA DISP-S1 to {data_end} (the end of the paper's InSAR record), HRKT array events to {aa_dense.onset_hi.max():.2f} (bimonthly surveys to {dense_era_end:.2f}), "
    f"Fremont GHCN gauge to {rain.index[rain.prcp_mm.notna()].max().date()}",
    f"the renewal-model windows below run from the last observation; nothing is claimed for the months after {last_obs}", "",
    "LOCATION  Hayward fault, Tule Pond section (see notebook 04 for the measured extent)",
    f"MAGNITUDE surface slip {insar_events.amplitude_mm.min():.1f}-{insar_events.amplitude_mm.max():.1f} mm on the stack ({n_insar_events} InSAR events; "
    f"95% CIs {insar_events.amp_lo.min():.1f}-{insar_events.amp_hi.max():.1f} mm), within one "
    f"{insar_window_days.min():.0f}-{insar_window_days.max():.0f} d acquisition interval", "",
    "TIME WINDOW",
    f"  (1) InSAR clock alone ({n_intervals} intervals, mean {interval_mean_yr:.2f} yr, sd {interval_sd_yr * 365.25:.0f} d, over {n_timing} onset draws): "
    f"next = last + mean = {ymd(next_onset_yr)}; "
    f"+/-1 sd {ymd(next_onset_yr - interval_sd_yr)}..{ymd(next_onset_yr + interval_sd_yr)}; "
    f"95% prediction interval (t, n={n_intervals}) {ymd(next_onset_yr - half_width_yr)}..{ymd(next_onset_yr + half_width_yr)}",
    f"      renewal models (BPT, bootstrap): 5-95% {insar_forecast.loc['bpt', 'q05']} .. {insar_forecast.loc['bpt', 'q95']}, median {insar_forecast.loc['bpt', 'q50']}; "
    f"P(WY2027 wet season, Oct-Apr) = {insar_forecast.loc['bpt', 'p_wet_season']:.2f}",
    f"  (2) HRKT bimonthly-era record ({array_first_year}-{dense_era_end:.0f}, Loma Prieta interval dropped): BPT 5-95% "
    f"{array_forecast.loc['bpt', 'q05']} .. {array_forecast.loc['bpt', 'q95']}, median {array_forecast.loc['bpt', 'q50']}; "
    f"P(WY2027 wet season) = {array_forecast.loc['bpt', 'p_wet_season']:.2f}, P(by end 2027) = {array_forecast.loc['bpt', 'p_by_2027']:.2f}",
    f"      merged BPT 5-95% {merged_forecast.loc['bpt', 'q05']} .. {merged_forecast.loc['bpt', 'q95']}, median {merged_forecast.loc['bpt', 'q50']}; "
    f"P(WY2027 wet season) = {merged_forecast.loc['bpt', 'p_wet_season']:.2f}; Poisson null {merged_forecast.loc['exponential', 'p_wet_season']:.2f}",
    f"  (3) all {n_insar} InSAR onset windows fell {int(insar_rain.dowy_lo.min())}-{int(insar_rain.dowy_hi.max())} d into the water year "
    f"({dowy_label(insar_rain.dowy_lo.min())}-{dowy_label(insar_rain.dowy_hi.max())}), after {insar_rain.cum_at_lo.min():.0f}-{insar_rain.cum_at_hi.max():.0f} mm of WY rain "
    f"({insar_rain.frac_at_hi.min():.0%}-{insar_rain.frac_at_hi.max():.0%} of the WY total), in years spanning "
    f"P{(wy_total_series < insar_rain.wy_total_mm.min()).mean() * 100:.0f}-P{(wy_total_series < insar_rain.wy_total_mm.max()).mean() * 100:.0f} "
    f"of the {wy_years[0]}-{complete_wy_last} WY-total distribution",
    f"      a {rain_total_days}-day rain total of at least Z mm within X days before the onset (Fremont gauge, the {n_insar_events} InSAR onsets, "
    f"{n_settings} settings of Z and X): the best, {best_sweep.Z_mm:.0f} mm within {best_sweep.X_d:.0f} d, gives {best_sweep.hits:.2f} of "
    f"{n_insar_events} onsets against {best_sweep.null_mean:.2f} for a random day of the same season in the same water year, "
    f"p = {best_sweep.p:.3f}; p = {p_search_paid:.3f} once all {n_settings} settings are paid for",
    f"      rules A and B do not tighten the window beyond the calendar spread ({calendar_spread} d)"]
with open(os.path.join(out_dir, "prediction_statement.txt"), "w") as fh:
    fh.write("\n".join(statement_lines) + "\n")
print("\n".join(statement_lines))

# %% [markdown]
# ## [39] Supplementary figure: the rainfall conditioning
#
# WY accumulation with the onsets; the lag after each rule A threshold; onset
# day against the day the best rule A threshold was crossed; and the p of every
# setting of the 3-day rain-total test of [34] (rows Z, columns X, the best
# setting outlined), with the search-paid p in the panel title.

# %%
# [40]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
# a) every water year's accumulation, the InSAR event years in colour, the onset windows as thick segments
ax = axes[0, 0]
for wy in range(wy_years[0], wy_years[1] + 1):
    this_year = wy_series(wy)
    ax.plot(this_year.dowy, this_year.cum, "-", color="0.8", lw=0.5, zorder=1)
for colour, event in zip(event_colours, insar_rain.itertuples()):
    this_year = wy_series(event.wy)
    ax.plot(this_year.dowy, this_year.cum, "-", color=colour, lw=1.2, zorder=3, label=f"WY{event.wy}")
    ax.plot([event.dowy_lo, event.dowy_hi], [event.cum_at_lo, event.cum_at_hi], "-", color=insar_colour, lw=3, solid_capstyle="butt", zorder=4)
for event in aa_rain.itertuples():
    this_year = wy_series(event.wy)
    segment = this_year[(this_year.dowy >= event.dowy_lo) & (this_year.dowy <= event.dowy_hi)]
    ax.plot(segment.dowy, segment.cum, "-", color="0.25", lw=2.2, alpha=0.7, zorder=2)
ax.axhline(best_rule_a.thr, color=insar_colour, lw=0.7, ls="--")
ax.text(5, best_rule_a.thr + 8, f"{best_rule_a.thr:.0f} mm", fontsize=6, color=insar_colour)
ax.set_xlim(1, 366)
ax.set_xticks([1, 62, 124, 182, 244, 305, 366])
ax.set_xticklabels(["Oct 1", "Dec 1", "Feb 1", "Apr 1", "Jun 1", "Aug 1", "Oct 1"])
ax.set_ylabel("Water-year precipitation (mm)")
ax.legend(handles=[Line2D([], [], color=insar_colour, lw=3, label="InSAR onset window"),
                   Line2D([], [], color="0.25", lw=2.2, alpha=0.7, label=f"Array onset window ({array_years_text})"),
                   Line2D([], [], color="0.8", lw=0.5, label=f"All water years {wy_years[0]} to {wy_years[1]}")], fontsize=6, loc="upper left")
utils.panel_title(ax, "A", "Onsets against water-year accumulation")
# b) the lag of each onset after every rule A threshold
ax = axes[0, 1]
thresholds = list(threshold_scores[threshold_scores.kind == "Y"].thr)
for colour, event in zip(event_colours, insar_rain.itertuples()):
    lag_lo = []
    lag_hi = []
    for Y in thresholds:
        lag_lo.append(getattr(event, f"lagY{Y}_lo"))
        lag_hi.append(getattr(event, f"lagY{Y}_hi"))
    ax.fill_betweenx(thresholds, lag_lo, lag_hi, color=colour, alpha=0.5, lw=0)
    ax.plot(lag_hi, thresholds, "-", color=colour, lw=0.8)
for event in aa_rain.itertuples():
    lag_lo = []
    lag_hi = []
    for Y in thresholds:
        lag_lo.append(getattr(event, f"lagY{Y}_lo"))
        lag_hi.append(getattr(event, f"lagY{Y}_hi"))
    ax.fill_betweenx(thresholds, lag_lo, lag_hi, color="0.3", alpha=0.10, lw=0)
ax.axvline(0, color="0.3", lw=0.6)
ax.axhline(best_rule_a.thr, color=insar_colour, lw=0.7, ls="--")
ax.set_xlim(-120, 160)
ax.set_xlabel("Onset minus threshold crossing (d)")
ax.set_ylabel("Water-year threshold (mm)")
utils.panel_title(ax, "B", "Lag against threshold")
# c) onset day against the day the best rule A threshold was crossed
ax = axes[1, 0]
best_Y = int(best_rule_a.thr)
for colour, event in zip(event_colours, insar_rain.itertuples()):
    crossing_day = event.dowy_hi - getattr(event, f"lagY{best_Y}_hi")
    ax.plot([crossing_day, crossing_day], [event.dowy_lo, event.dowy_hi], "-", color=colour, lw=3, solid_capstyle="butt")
for event in aa_rain.itertuples():
    lag = getattr(event, f"lagY{best_Y}_hi")
    if np.isfinite(lag):
        crossing_day = event.dowy_hi - lag
        ax.plot([crossing_day, crossing_day], [event.dowy_lo, event.dowy_hi], "-", color="0.25", lw=1.8, alpha=0.7, solid_capstyle="butt")
diagonal = np.array([1, 366])
ax.plot(diagonal, diagonal, "-", color="0.5", lw=0.6)
ax.fill_between(diagonal, diagonal + best_rule_a.insar_lag_min, diagonal + best_rule_a.insar_lag_max, color=insar_colour, alpha=0.12, lw=0)
ax.set_xlim(30, 220)
ax.set_ylim(30, 260)
ticks = [62, 124, 182, 244]
tick_labels = ["Dec 1", "Feb 1", "Apr 1", "Jun 1"]
ax.set_xticks(ticks)
ax.set_xticklabels(tick_labels)
ax.set_yticks(ticks)
ax.set_yticklabels(tick_labels)
ax.set_xlabel(f"Day water-year rain passes {best_Y} mm")
ax.set_ylabel("Onset day of water year")
utils.panel_title(ax, "C", f"Onset {best_rule_a.insar_lag_min:+.0f} to {best_rule_a.insar_lag_max:+.0f} d after the threshold")
# d) the 3-day rain-total test: p of every setting (rows Z, columns X), the number written in each cell, the best outlined
ax = axes[1, 1]
p_cmap = plt.get_cmap("magma")                       # sequential, no blue or green
image = ax.imshow(p_table.values, cmap=p_cmap, vmin=0, vmax=1, aspect="auto")
for i, Z in enumerate(p_table.index):
    for j, X in enumerate(p_table.columns):
        p_value = float(p_table.loc[Z, X])
        red, green, blue, _ = p_cmap(p_value)
        text_colour = "white" if 0.299 * red + 0.587 * green + 0.114 * blue < 0.5 else "black"     # readable on the cell's own colour
        ax.text(j, i, f"{p_value:.3f}", ha="center", va="center", fontsize=7, color=text_colour)
best_row = list(p_table.index).index(best_sweep.Z_mm)
best_column = list(p_table.columns).index(best_sweep.X_d)
ax.add_patch(plt.Rectangle((best_column - 0.5, best_row - 0.5), 1, 1, facecolor="none", edgecolor="#56B4E9", lw=2))   # Okabe-Ito sky blue
ax.set_xticks(range(len(p_table.columns)))
ax.set_xticklabels([f"{X}" for X in p_table.columns])
ax.set_yticks(range(len(p_table.index)))
ax.set_yticklabels([f"{Z}" for Z in p_table.index])
ax.set_xlabel("X, days before the onset")
ax.set_ylabel(f"Z, {rain_total_days}-day rain total at Fremont (mm)")
colour_bar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
colour_bar.set_label("p (null: a random day of the season, same water year)", fontsize=7)
colour_bar.ax.tick_params(labelsize=7)
utils.panel_title(ax, "D", f"InSAR onsets, {n_settings} settings; search paid for: p = {p_search_paid:.3f}")
fig.tight_layout()
f_conditioning = os.path.join(fig_dir, f"figS_precip_conditioning_bestY{best_Y}mm_rain{rain_total_days}d_sweep{n_settings}_nnull{n_null}.png")
fig.savefig(f_conditioning, dpi=300)
fig.savefig(f_conditioning.replace(".png", ".pdf"))
print("Saved figure to:", f_conditioning)
plt.show()

# %% [markdown]
# ## [41] Registry: the stats sidecar `11_forecast_stats.txt` (with the prediction statement) and the `11_*` numbers

# %%
# [42]
sidecar = REG.sidecar("11_forecast", stats_lines + ["", *statement_lines])
REG.register("11_issue_date", forecast_issue_date, "date", 0, where="P10, M9", cell="constants", sidecar=sidecar)
REG.register("11_data_end", data_end, "date", 0, where="P10, M9", cell="load", sidecar=sidecar)
REG.register("11_last_obs_date", last_obs, "date", 0, where="P10, M9", cell="load", sidecar=sidecar)
REG.register("11_insar_interval_mean_yr", interval_mean_yr, "yr", 2, where="P10", cell="clock", sidecar=sidecar)
REG.register("11_insar_interval_sd_d", interval_sd_yr * 365.25, "d", 0, where="P10", cell="clock", sidecar=sidecar)
REG.register("11_insar_interval_sd_ci_lo_d", sd_ci_days[0], "d", 0, where="M9", cell="clock", sidecar=sidecar)
REG.register("11_insar_interval_sd_ci_hi_d", sd_ci_days[1], "d", 0, where="M9", cell="clock", sidecar=sidecar)
REG.register("11_insar_cov", interval_sd_yr / interval_mean_yr, "1", 2, where="P10", cell="clock", sidecar=sidecar)
REG.register("11_next_clock_date", ymd(next_onset_yr), "date", 0, where="P10, Fig3", cell="clock", sidecar=sidecar)
REG.register("11_next_1sd_lo", ymd(next_onset_yr - interval_sd_yr), "date", 0, where="P10", cell="clock", sidecar=sidecar)
REG.register("11_next_1sd_hi", ymd(next_onset_yr + interval_sd_yr), "date", 0, where="P10", cell="clock", sidecar=sidecar)
REG.register("11_pi95_lo", ymd(next_onset_yr - half_width_yr), "date", 0, where="P10, Fig3", cell="clock", sidecar=sidecar)
REG.register("11_pi95_hi", ymd(next_onset_yr + half_width_yr), "date", 0, where="P10, Fig3", cell="clock", sidecar=sidecar)
for cat_name, key in (("A_insar", "insar"), ("B_aa_dense", "aa_dense"), ("B_aa_dense_noLP", "aa_dense_nolp"), ("C_merged", "merged")):
    issue = forecasts[forecasts.catalog == cat_name].set_index("model")
    for model in ("bpt", "lognormal", "weibull", "exponential"):
        REG.register(f"11_{key}_{model}_q05", issue.loc[model, "q05"], "date", 0, where="P10, Fig3", cell="forecast", sidecar=sidecar)
        REG.register(f"11_{key}_{model}_q50", issue.loc[model, "q50"], "date", 0, where="P10, Fig3", cell="forecast", sidecar=sidecar)
        REG.register(f"11_{key}_{model}_q95", issue.loc[model, "q95"], "date", 0, where="P10, Fig3", cell="forecast", sidecar=sidecar)
        REG.register(f"11_{key}_{model}_p_wet_wy2027", issue.loc[model, "p_wet_season"], "1", 2, where="P10, Fig3", cell="forecast", sidecar=sidecar)
        REG.register(f"11_{key}_{model}_p_by_end_2027", issue.loc[model, "p_by_2027"], "1", 2, where="P10", cell="forecast", sidecar=sidecar)
        REG.register(f"11_{key}_{model}_daicc", issue.loc[model, "daicc"], "1", 2, where="M9", cell="forecast", sidecar=sidecar)
    REG.register(f"11_{key}_n_intervals", int(issue.n_intervals.iloc[0]), "count", 0, where="M9", cell="forecast", sidecar=sidecar)
    REG.register(f"11_{key}_interval_mean_yr", float(issue.loc["bpt", "mean_yr"]), "yr", 2, where="P10", cell="forecast", sidecar=sidecar)
    REG.register(f"11_{key}_interval_cov", float(issue.loc["bpt", "cov"]), "1", 2, where="P10", cell="forecast", sidecar=sidecar)
for cat_name, key in (("A_insar", "insar"), ("B_aa_dense_noLP", "aa_dense_nolp"), ("C_merged", "merged")):
    mean_yr, sd_yr, n_shown = fig3c_numbers[cat_name]
    REG.register(f"11_{key}_fig3c_interval_mean_yr", mean_yr, "yr", 2, where="Fig3", cell="fig3c", sidecar=sidecar)
    REG.register(f"11_{key}_fig3c_interval_sd_yr", sd_yr, "yr", 2, where="Fig3", cell="fig3c", sidecar=sidecar)
    REG.register(f"11_{key}_fig3c_n_intervals", n_shown, "count", 0, where="Fig3", cell="fig3c", sidecar=sidecar)
REG.register("11_onset_dowy_min", int(insar_rain.dowy_lo.min()), "d", 0, where="P10", cell="conditioning", sidecar=sidecar)
REG.register("11_onset_dowy_max", int(insar_rain.dowy_hi.max()), "d", 0, where="P10", cell="conditioning", sidecar=sidecar)
REG.register("11_onset_earliest", dowy_label(insar_rain.dowy_lo.min()), "date", 0, where="P10", cell="conditioning", sidecar=sidecar)
REG.register("11_onset_latest", dowy_label(insar_rain.dowy_hi.max()), "date", 0, where="P10", cell="conditioning", sidecar=sidecar)
REG.register("11_onset_calendar_spread_d", calendar_spread, "d", 0, where="P10", cell="conditioning", sidecar=sidecar)
REG.register("11_wy_total_min_mm", insar_rain.wy_total_mm.min(), "mm", 0, where="P10", cell="conditioning", sidecar=sidecar)
REG.register("11_wy_total_max_mm", insar_rain.wy_total_mm.max(), "mm", 0, where="P10", cell="conditioning", sidecar=sidecar)
REG.register("11_wy_percentile_min", (wy_total_series < insar_rain.wy_total_mm.min()).mean() * 100, "pct", 0, where="P10", cell="conditioning", sidecar=sidecar)
REG.register("11_wy_percentile_max", (wy_total_series < insar_rain.wy_total_mm.max()).mean() * 100, "pct", 0, where="P10", cell="conditioning", sidecar=sidecar)
REG.register("11_cum_at_onset_min_mm", insar_rain.cum_at_lo.min(), "mm", 0, where="P10", cell="conditioning", sidecar=sidecar)
REG.register("11_cum_at_onset_max_mm", insar_rain.cum_at_hi.max(), "mm", 0, where="P10", cell="conditioning", sidecar=sidecar)
REG.register("11_frac_at_onset_min", insar_rain.frac_at_hi.min(), "1", 2, where="P10", cell="conditioning", sidecar=sidecar)
REG.register("11_frac_at_onset_max", insar_rain.frac_at_hi.max(), "1", 2, where="P10", cell="conditioning", sidecar=sidecar)
REG.register("11_ruleA_best_Y_mm", best_rule_a.thr, "mm", 0, where="P10", cell="conditioning", sidecar=sidecar)
REG.register("11_ruleA_lag_min_d", best_rule_a.insar_lag_min, "d", 0, where="P10", cell="conditioning", sidecar=sidecar)
REG.register("11_ruleA_lag_max_d", best_rule_a.insar_lag_max, "d", 0, where="P10", cell="conditioning", sidecar=sidecar)
REG.register("11_ruleB_best_Z_mm", best_rule_b.thr, "mm", 0, where="P10", cell="conditioning", sidecar=sidecar)
REG.register("11_ruleB_lag_min_d", best_rule_b.insar_lag_min, "d", 0, where="P10", cell="conditioning", sidecar=sidecar)
REG.register("11_ruleB_lag_max_d", best_rule_b.insar_lag_max, "d", 0, where="P10", cell="conditioning", sidecar=sidecar)
REG.register("11_rain_sweep_best_Z_mm", int(best_sweep.Z_mm), "mm", 0, where="P10", cell="rain sweep", sidecar=sidecar)
REG.register("11_rain_sweep_best_X_d", int(best_sweep.X_d), "d", 0, where="P10", cell="rain sweep", sidecar=sidecar)
REG.register("11_rain_sweep_best_hits", best_sweep.hits, "count", 2, where="P10", cell="rain sweep", sidecar=sidecar)
REG.register("11_rain_sweep_best_null_hits", best_sweep.null_mean, "count", 2, where="P10", cell="rain sweep", sidecar=sidecar)
REG.register("11_rain_sweep_best_p", best_sweep.p, "1", 3, where="P10", cell="rain sweep", sidecar=sidecar)
REG.register("11_rain_sweep_p_search_paid", p_search_paid, "1", 3, where="P10", cell="rain sweep", sidecar=sidecar)
REG.register("11_rain_sweep_n_settings", n_settings, "count", 0, where="P10", cell="rain sweep", sidecar=sidecar)
REG.render()
