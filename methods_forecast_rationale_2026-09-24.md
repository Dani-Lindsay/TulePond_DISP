# The next-event forecast: why we did what we did (2026-09-24)

Scraped from the record on disk so the Methods can be written from it. The
email itself is not on disk; what survives is the request as it was written
into the script headers on the day, Roland's chat notes and Dani's voice notes
in `Hayward_DISP-S1/manuscript/`, and the assumptions register. Everything
below is now in notebook `11_forecast` (and the rain in `10_rain_seasonality`).

## 1. Where each choice came from

| Choice | Set in 10 | Origin | Source |
|---|---|---|---|
| A prediction must state WHERE, WHEN, HOW BIG (and, in the five-part PNSN/NEPEC form, HOW PROBABLE and WHY) | the statement cell | Roland, 2026-08-25: asked for a quantitative TIME window to go with the location and size windows the paper already had; USGS/NEPEC criteria | `run21` header; `tulepond_letter_notes_sorted_2026-09-16.md` lines 632-645 (the USGS FAQ three-part and the PNSN five-part wording; neither sentence is verbatim on either page, cite the FAQ as a web page with access date) |
| Forecast from the InSAR record ALONE (A) and separately from the 45-year array record (B, B', C) | `catalogues` | Roland, 2026-08-25: "from (A) the InSAR constraints alone and (B, C) the alignment-array record that reaches back to 1979" | `run21` header |
| Onsets drawn UNIFORMLY within their observation windows, 400 draws; no midpoint ever used | `n_timing`, `forecast()` step 1 | Dani: the event is equally likely anywhere inside its window (an acquisition interval for InSAR, a survey interval for the arrays) | `run21` header ("-- Dani --"); `run22` header; assumptions register row 17 |
| Array events enter only where the survey window is 130 d or less (the bimonthly era, 1983-2006) | `resolvable_days` | the annual-cadence array windows (300-390 d) do not constrain the season; same cut as the seasonality figure | `run21` header; Fig 3 caption |
| B': the interval spanning Loma Prieta (1989.80) dropped as a variant | `loma_prieta`, `drop_lp` | documented creep suppression after Loma Prieta (Lienkaemper et al. 1997) | `run21` header; assumptions register row 39 |
| C: the dense-era array events (to 2006) and the InSAR events, with 2006 to 2017 UNOBSERVED: no interval is formed across the gap and the clock restarts at the first InSAR detection; array detections after 2006 (annual surveys) enter no catalogue | `merge_catalogues` in 08, `interval_mask` in 10 | Dani, 2026-09-24: the 6.26 yr interval 2005.96 to 2012.22 in the legacy C spanned the change from bimonthly to annual surveys, a gap in observation, not in the clock. C now has 12 intervals (was 15) | 10 cell 8; 08 |
| Five renewal families: exponential, lognormal, Weibull, BPT, Gaussian | `models` | exponential = the Poisson, memoryless, no-clock NULL; lognormal, Weibull and BPT = the three standard earthquake-forecasting families; Gaussian = the plain "clock" for the four-interval InSAR case, where the family cannot be resolved anyway | `run21` header; assumptions register row 37 ("family irrelevant at n = 4") |
| Maximum-likelihood fits; parametric bootstrap, 5 refits per timing draw | `fit_interval_model`, `n_boot`, `forecast()` step 4 | so the quoted windows carry the estimation uncertainty of small samples, the four InSAR intervals above all | `run21` header |
| Conditional on no event through the DATA END, read from the fault-parallel series (the last acquisition seen by both tracks, 2025-12-15 now, 2026-03-21 after the refresh); the issue date is only the date the statement was written | `data_end`, `conditional_cdf` | a forecast can only condition on quiet time the data vouch for; conditioning on the later issue date claimed months nobody observed and raised the array null from 0.22 to 0.28 (Dani, 2026-09-24). The legacy run21 used the issue date and a typed data end of 2025-12-21, which was the ascending track's last date, not the common grid's | `run21` header; 10 cell 8 |
| Reported: the 5-50-95 % onset dates, P(onset in the WY2027 wet season, 2026-10-01 to 2027-04-30), P(by end 2027) | `forecast_rows` | the wet season is the paper's window; end-2027 is the horizon at which the InSAR clock and the 45-yr record disagree most | `run21` header |
| Families compared by AICc on the median timing draw | `aicc`, `median_draw` | AICc for n = 4 to 15; on one representative draw so the comparison is not itself a mixture | `run21` header |
| The InSAR clock in closed form: mean interval, sd with its chi-square CI, next = last + mean, t-based 95 % prediction interval | the clock cell | the plain arithmetic the text quotes beside the renewal mixture | `run21` (recurrence_forecast_stats) |
| The 45-yr record as the STATED NULL: P(WY2027) = 0.28 against the InSAR clock's ~0.99; "the 2026-27 wet season discriminates the two" | statement lines (2) | Dani and Roland, September 2026: the forecast is a test with a stated null, not a claim | quotebank "Forecast and hazard"; letter v0.1 P15; reuse A12 (reuse the sentence as is) |
| Rainfall conditioning rules A (WY cumulative > Y), B (first 3-day storm > Z), C (a storm > Z within X days) | `y_grid`, `z_grid`, `c_z`, `c_x` | Roland, 2026-08-25, verbatim: "within X days of annual WY precipitation > Y mm" or "within X days of the first storm > Z mm"; rule C is the Roeloffs (2001) storm-trigger test | `run22` header |
| Every rain quantity is a RANGE over the onset window, not a value at a midpoint | `event_rain` columns `_lo/_hi`, `_min/_max` | same uniform-in-window principle | `run22` header |
| Rule C scored against its base rate: the fraction of Dec-Mar days within X d after such a storm; P(chance) = the product over the five InSAR windows of the fraction of wet-season placements that would be hit | `rule_c` | so five hits out of five can be weighed against five hits by calendar alone | `run22` header |
| The calendar spread of the InSAR onsets is the baseline any rule has to beat | `calendar_spread` | a rule earns its place only if its lag spread is tighter than the onset spread itself; rules A and B do not | `run22` header; statement line (3) |
| Conclusion the conditioning supports: the amount of rain is not the control, storm proximity may be | `figS_wy_cumulative_rain_onsets` (cell 16) | Roland's chat note: "there is modulating of the hydrology ... a system that is resonant to the input forcing"; Dani's note: "rain amount is not the control, storm proximity might be" | quotebank "When"; the cumulative-rain figure (61 to 447 mm at onset, 37 to 73 % of the year) |
| The array-cadence caveat behind all of this | 08 (`run17`-`run19` not ported) | Roland, review comment 2026-08-06: the array cadence is too wide to DETECT the events independently, but a step comb with a fixed period can be FIT; hence the forced-period tests, kept out of the paper | `run17` header |

## 2. What Roland's guidance was, as recorded

- 2026-08-06 (review comment): the alignment arrays cannot detect the events
  on their own at their cadence, but a comb of steps with a fixed or tightly
  constrained period can be fitted to them. That produced the forced-period
  tests (`run17`, `run19`), which are not in the paper.
- 2026-08-25 (request): give the prediction its TIME window, from the InSAR
  record alone and from the 45-year array record, in the NEPEC form (where,
  when, how big); and test whether the onsets can be tied to rain as "within
  X days of the water-year total passing Y mm" or "within X days of the first
  storm larger than Z mm". That produced `run21` and `run22`, the same day.
- September 2026 (chat notes): the regularity has a forcing term, the
  hydrology; the system is resonant to it, which is why it is more
  predictable than other slow-slip systems; and the prediction, whether it
  succeeds or not, says nothing about predicting large destructive
  earthquakes. The 45-year record's P = 0.28 is the stated null; the 2026-27
  wet season is the test.
- Still open, in Roland's hands: which renewal model is quoted in the text
  (`manuscript_v0.2.md` line 642), and whether both the InSAR-clock window
  and the 45-yr probability are quoted or the clock only (notes section 7).

If the email is wanted verbatim, the Gmail connector in this session is not
authorised; it can be added through the claude.ai connector settings.

## 3. Draft Methods text (fill from `registry.txt`, keys `09_*` and `10_*`)

**Rainfall record and onset seasonality (notebook 09).** Daily precipitation
at Tule Pond is a day-by-day composite of the three nearest NOAA GHCN-daily
gauges, Fremont (USC00043244, 4 km), Newark (USC00046144, 7 km) and Hayward
Air Terminal (USW00093228, 18 km), taking the first reporting gauge in that
order without rescaling. Newark is a morning-observer cooperative gauge, so
its daily totals are shifted one day earlier onto the midnight basis of the
other two; the shift is confirmed on the storm of 11 December 2014, which
Newark as published books on the 12th. The series runs from 1978-10-01 to the
forecast issue date, water years named by the year they end (`09_wy_median_mm`,
`09_wy_p10_mm`, `09_wy_p90_mm` over the complete years). Onset seasonality
uses the season-resolved onsets only: the five InSAR windows (6 to 12 d) and
the `09_n_aa_season_resolved` array windows of 130 d or less, of which
`09_n_aa_dec_mar` fall between December and March, as do all five InSAR onsets.

**Next-event forecast (notebook 10).** Onsets are known only to an
acquisition or survey interval and are taken as uniformly distributed within
it; no midpoint is used anywhere. Repeat intervals are formed from four
catalogues: A, the five InSAR events (4 intervals); B, the dense-era array
events of 1983 to 2006 (8); B', the same without the interval spanning the
1989 Loma Prieta earthquake, after which creep on the section was suppressed
(7); and C, the dense-era array events followed by the InSAR events (12),
with the annually surveyed years 2006 to 2017 treated as unobserved: no
interval is formed across them. For each catalogue the onsets are drawn
400 times within their windows, the intervals of each draw fitted by maximum
likelihood with exponential, lognormal, Weibull, Brownian passage time and
Gaussian renewal models, and each fit re-sampled five times in a parametric
bootstrap. The next-onset distribution is the mixture over all draws and
refits of the model's conditional distribution given no event by the
last acquisition seen by both tracks (`10_data_end`); the date the statement
was written (`10_issue_date`) carries no conditioning, because no observation
vouches for the months between the two. We report the 5, 50 and 95 % onset dates and
the probability of an onset in the 2026-27 wet season (1 October to 30 April)
and by the end of 2027; families are compared by their small-sample Akaike
criterion on the median draw. The exponential model is the memoryless null.
In closed form the InSAR clock has mean interval `10_insar_interval_mean_yr`
yr with standard deviation `10_insar_interval_sd_d` d (95 % CI
`10_insar_interval_sd_ci_lo_d` to `10_insar_interval_sd_ci_hi_d` d), giving
next = last + mean = `10_next_clock_date` with a t-based 95 % prediction
interval of `10_pi95_lo` to `10_pi95_hi`. The 45-year array record gives
`10_aa_dense_bpt_p_wet_wy2027` for the same wet season, the stated null; the
2026-27 wet season discriminates the two.

**Rainfall conditioning (notebook 10).** For every season-resolved onset we
evaluate, over the whole onset window, the water-year precipitation
accumulated by then, the lag after the water-year total first exceeds Y mm
(rule A, Y from 50 to 400 mm), the lag after the first 3-day storm of at
least Z mm (rule B, Z from 15 to 75 mm), and whether a 3-day storm of at least
Z mm ended within X days of any day of the window (rule C, Z 20 to 50 mm, X 7
to 30 d). Rule C is scored against its own base rate, the fraction of
December-to-March days that lie within X d after such a storm, and against
the probability that all five InSAR windows would be hit by the calendar
alone. The spread of the InSAR onset dates themselves (`10_onset_calendar_spread_d`
d) is the baseline any rule must beat; rules A and B do not, and the rain
accumulated at onset ranges from `10_cum_at_onset_min_mm` to
`10_cum_at_onset_max_mm` mm (`10_frac_at_onset_min` to `10_frac_at_onset_max`
of the year's total), whereas a `10_ruleC_Z_mm` mm storm within `10_ruleC_X_d`
d precedes all five InSAR onsets (P by chance `10_ruleC_p_chance`) and
`10_ruleC_aa_hits` of the nine array onsets.

## 4. Open decisions for the text

1. Which renewal family is quoted (BPT is what the statement uses now):
   Roland to decide.
2. Clock window only, or clock plus the 45-yr null: the notes lean to both,
   with the null stated.
3. Vocabulary: "forecast" is the USGS/NEPEC word; "prediction" and "predict"
   are the intended words since 2026-09-16; "slow earthquake" stays out of the
   title and abstract.
4. The conditioning date is the data end and moves with the data refresh
   (2026-03-21 after the March 2026 mini-stacks); every `10_*` number moves
   with it (TULEPOND_SUPERSEDE=1 and an anchor update). Processing DISP-S1
   products beyond March 2026 locally would move it further, if the products
   exist.
5. DECIDED 2026-09-24: catalogue C treats 2006 to 2017 as unobserved (no
   interval across the annual-survey years; array detections after 2006 are not
   used for the clock). C's P(WY2027) is 0.26, the Poisson null on C 0.15.

## 5. The literature behind each step (from memory on 2026-09-24; verify every entry before citing)

| Step in notebook 10 | Precedent | Citations to check |
|---|---|---|
| Onsets drawn uniformly within their windows, 400 draws | Monte Carlo treatment of event-date uncertainty in paleoseismology; short catalogues | Rhoades, Van Dissen & Dowrick 1994 (JGR); Ogata 1999 (JGR, uncertain occurrence times); Biasi, Weldon, Fumal & Seitz 2002 (BSSA); Parsons 2008 (JGR, Monte Carlo recurrence parameters from short paleoseismic catalogs) |
| Exponential (Poisson null), lognormal, Weibull, BPT, Gaussian by maximum likelihood | the standard time-dependent recurrence families | Nishenko & Buland 1987 (BSSA); WGCEP 1988, 1990, 2003; Hagiwara 1974 (Tectonophysics); Utsu 1984; Matthews, Ellsworth & Reasenberg 2002 (BSSA); Ellsworth et al. 1999 (USGS OFR 99-522); Field et al. 2015 (BSSA, UCERF3); on repeaters: Rubinstein et al. 2012 (BSSA), Chen, Bürgmann & Nadeau 2013 (JGR) |
| Parametric bootstrap, 5 refits per draw | estimation uncertainty from few intervals | Efron & Tibshirani 1993; Savage 1991 and 1994 (BSSA), the NEPEC-forecast criticism and the t-based prediction interval used by the closed-form clock |
| Conditional on no event by the reference date | the hazard-function conditional probability | WGCEP 1988; Ellsworth et al. 1999; Field & Jordan 2015 (BSSA, uncertain date of last event) |
| Families compared by AICc on the median draw | small-sample information criterion | Hurvich & Tsai 1989; Burnham & Anderson 2002; Ogata 1999 (AIC on paleoseismic recurrence) |
| Rule C storm proximity against its base rate | storm-triggered creep at Parkfield | Roeloffs 2001 (JGR) |
| Stating a recurrence-based window and instrumenting the fault to test it | the Parkfield prediction experiment | Bakun & Lindh 1985 (Science); Bakun & McEvilly 1984 (JGR) |

## 6. Background and precedent, step by step (prose for the Methods and the supplement)

All citations from memory on 2026-09-24; check year, journal and claim before use.

**Why a renewal forecast at all.** Quasi-periodic recurrence is the working
assumption of time-dependent earthquake forecasting: the elastic-rebound idea
that a patch reloads after it slips, so the hazard rises with elapsed time
(Reid 1910; Shimazaki & Nakata 1980 for the time- and slip-predictable
models). The same statistics are applied to slow slip: Cascadia episodic
tremor and slip recurs every ~14 months (Miller et al. 2002, Science; Rogers &
Dragert 2003, Science) and has been forecast on that basis; Hikurangi slow
slip events recur on 2 to 5 yr cycles (Wallace & Beavan 2010, JGR); repeating
earthquakes at Parkfield are the classic small-scale clock (Nadeau & Johnson
1998, BSSA; Nadeau & McEvilly 1999, Science), and their recurrence has been
fitted with the same renewal families we use (Rubinstein et al. 2012, BSSA;
Chen, Bürgmann & Nadeau 2013, JGR). Surface creep events themselves were
described as recurring phenomena from the first creepmeter records (Wesson
1988, JGR, "Dynamics of fault creep"; Bilham et al. 2004, BSSA on creepmeter
design; Gittins & Hawthorne 2022, JGR, on the statistics of creep events on
the San Andreas, to check). Our contribution is a creep event with a CoV of
0.04 over five cycles, which is tighter than any of these.

**Why the catalogues are built the way they are.** A, the five InSAR events,
is the record the forecast rests on; four intervals is the whole basis, which
is why Savage's warnings (below) apply. B restricts the array record to the
surveys of 130 d or less, the bimonthly era of 1983 to 2006: an event known
only to a 300 to 390 d annual window carries no information about the season
and adds nothing but width to an interval distribution, so the cut is the
same one the seasonality figure uses. B' drops the one interval that spans
the 1989 Loma Prieta earthquake because creep on the southern Hayward fault
was suppressed for years afterwards by the coseismic stress change
(Lienkaemper, Galehouse & Simpson 1997, Science, "Creep response of the
Hayward fault to stress changes caused by the Loma Prieta earthquake"; also
Lienkaemper et al. 2001, 2012 on the recovery); an interval lengthened by an
external stress step is not a sample of the intrinsic clock. C joins the two
records so the 45-year trend can be examined on one axis; the array events are
taken up to 0.3 yr before the first InSAR onset so the 2017 event, which both
records see, is counted once.

**Why onsets are sampled uniformly within their windows.** An InSAR onset is
known to lie between two acquisitions 6 to 12 d apart, an array onset between
two surveys 43 to 124 d apart, and nothing in either record says where inside
the interval it fell. Paleoseismology met the same problem with radiocarbon
age ranges and answered it by Monte Carlo: sample every event's date from its
own distribution, form the intervals of each sample, and let the spread of
the samples carry the dating uncertainty (Rhoades, Van Dissen & Dowrick 1994,
JGR; Ogata 1999, JGR; Biasi, Weldon, Fumal & Seitz 2002, BSSA). Parsons (2008,
JGR) is the closest precedent for a short catalogue: uniform draws within the
age ranges, a renewal fit to every draw, and the parameter spread reported.
Using the midpoint instead would understate the interval scatter and would
put a spurious precision of days on the array record; the uniform draw is
Dani's rule and runs through the rain rules as well.

**Why these five families.** Lognormal recurrence is the original
time-dependent model of the Working Group on California Earthquake
Probabilities (Nishenko & Buland 1987, BSSA; WGCEP 1988, 1990); Weibull is the
engineering reliability distribution first applied to earthquakes by Hagiwara
(1974, Tectonophysics) and compared with the others by Utsu (1984); Brownian
passage time is the physically motivated model of a patch loaded at a
constant rate with Brownian perturbations (Matthews, Ellsworth & Reasenberg
2002, BSSA; Ellsworth et al. 1999, USGS Open-File 99-522) and is the family
used by WGCEP 2003 and UCERF3 (Field et al. 2015, BSSA). The exponential is
the Poisson process, memoryless, the "no clock" null every one of those
studies carries. The Gaussian is not a recurrence model in that literature;
it is included because with four intervals the families are indistinguishable
(the AICc differences are small) and the text quotes the plain mean and
standard deviation, so the Gaussian is that statement made explicit. Say this
in the text rather than let it look like a sixth family.

**Why maximum likelihood, and why the bootstrap.** Maximum likelihood is how
all the studies above fit their families; for the exponential, lognormal and
Gaussian it is closed form, for Weibull and BPT it is a one-dimensional
search. The parametric bootstrap (Efron & Tibshirani 1993) re-samples
intervals from the fitted model, refits, and takes the spread of the refits as
the estimation uncertainty. It is needed here because of Savage (1991, BSSA,
"Criticism of some forecasts of the National Earthquake Prediction Evaluation
Council", and 1994, BSSA, "Empirical earthquake probabilities from observed
recurrence intervals"): with a handful of intervals the recurrence parameters
are so poorly constrained that a point forecast is not meaningful, and the
prediction interval for the next event must come from the Student-t
distribution with n - 1 degrees of freedom. Our closed-form clock uses exactly
Savage's t-based interval; the bootstrap is the mixture-model counterpart for
the renewal families. Savage's target was NEPEC forecasts of the 1980s, which
makes the citation doubly apt.

**Why the forecast is conditioned on the elapsed time.** A forecast issued at
date T must use the probability of an event between T and t given that none
has occurred by T, the hazard-function form of WGCEP (1988) and Ellsworth et
al. (1999). Two reference dates are given: the end of the data, which is what
the record itself supports, and the pinned issue date, which is what the
paper states. Field & Jordan (2015, BSSA) treat the case where the date of
the last event is itself uncertain, which is our onset window, and the
uniform draw of the last onset in the mixture is their construction.

**Why the reported quantities are the 5-50-95 % dates, P(wet season) and
P(by end 2027).** The quantiles are the usual summary of a forecast density.
The wet-season probability is the number the paper can be judged on: every
onset in both records has fallen between December and March, so the
prediction is "in the 2026-27 wet season", and its probability under the
InSAR clock (near 1) and under the 45-year array record (0.28) are the two
hypotheses. The end-2027 horizon is where the two disagree most. Stating the
45-year probability as the null is the framing of a test rather than a claim
(cf. Jackson & Kagan 2006, BSSA, on what the Parkfield prediction did and did
not test).

**Why AICc on the median draw.** The corrected Akaike criterion (Hurvich &
Tsai 1989; Burnham & Anderson 2002) is the standard small-sample comparison
and was used by Ogata (1999) on paleoseismic recurrence. It is evaluated on
one representative draw, the one with the median mean interval, so the
comparison is between families and not between mixtures; on four intervals it
cannot separate them, and the paper should say so.

**Why the rain rules have the form they do.** Rule C is the storm-trigger
test of Roeloffs (2001, JGR, "Creep rate changes at Parkfield, California
1966-1999: seasonal, precipitation induced, and tectonic"), who showed that
Parkfield creep rate responds to individual storms with a lag of days to weeks
consistent with pore-pressure diffusion, and that the seasonal cycle of creep
follows rainfall (the older seasonal-creep observations are Schulz, Mavko,
Burford & Stuart 1982, JGR). Rules A and B are the two forms Roland proposed
on 2026-08-25, cumulative water-year rain and the first storm of the year, and
the point of scoring all three is that A and B fail and C passes: the onsets
do not wait for a fixed amount of rain (61 to 447 mm, 37 to 73 % of the
year's total, notebook 10 cell 16) but every one follows a 40 mm storm within
21 d. That is the observation behind "storm proximity, not rain amount", and
it is what makes the mechanism a pore-pressure or normal-stress perturbation
on a critically stressed patch rather than a slow seasonal loading.

**Why rule C is scored against a base rate.** A storm within 21 d of some day
of a 6 to 12 d window is a common event in a Bay Area winter, so five hits out
of five must be weighed against how often the calendar alone would produce
them: the base rate is the fraction of December-to-March days that lie within
21 d after a 40 mm storm, and P(chance) multiplies, over the five events, the
fraction of same-length windows placed anywhere in that wet season that would
have been hit. The array events are used as a consistency check only, because
their windows are wide enough to contain a storm by chance.

**Why the calendar spread is the baseline.** The five InSAR onsets already
fall within a 3-month band of the water year. A rain rule is only useful if
it narrows the timing beyond that, so each rule's lag spread is compared to
the spread of the onset dates themselves; rules A and B do not beat it.

**The precedent for issuing the statement.** The Parkfield experiment (Bakun
& Lindh 1985, Science; Bakun & McEvilly 1984, JGR) published a recurrence-based
time window, 1988 +/- 5 yr, with its assumptions, and instrumented the fault to
test it; the event came in 2004, and the retrospectives (Savage 1993, BSSA,
"The Parkfield prediction fallacy"; Jackson & Kagan 2006, BSSA) are the
lessons on what a stated window can and cannot claim. Roland's note calls the
Tule Pond forecast a smaller-scale version of Parkfield; the honest version of
that comparison is that ours has a 2-year cycle, a CoV of 0.04, a known
forcing, and a null hypothesis stated in advance.
