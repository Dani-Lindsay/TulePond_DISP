# Context: Parkmont (Hayward Fault, Fremont) creep events vs. groundwater levels

Author: Dani Lindsay (ESNZ). This file is the working context for coding up the comparison between repeating shallow creep events on the Hayward Fault and groundwater well records on either side of the fault. Section 1 is my own interpretation, in my words. Section 2 onward is the exploratory analysis done so far, its methods, its numbers, and known caveats. Do not add interpretation beyond what is written here.

---

## 1. My interpretation and framing (my words)

The project: repeating ~2-yr surface creep events on the ~4 km "Parkmont" segment of the Hayward Fault in Fremont (HRKT alignment array; InSAR FP stack). The question here is how the water-well data fit in with how we see these slip events.

Framing of the data:
- Precipitation and storm frequency indicate the delivery of surface water into the fault zone. The well data provide a measure of the water level on each side of the fault.
- The well data provide an indication of the pore-fluid fluctuations at depth in the fault.
- My understanding is that the wells represent aquifers on each side of the fault, and that the aquifers on either side act as coherent units, so what happens away from the fault is likely to represent what is happening at the fault surface on that side.
- The depth of the water table might interact with the depth of the creeping zone.

Working statements from the exploration:
- The events do not coincide with the peak aquifer levels.
- They happen regardless of drought conditions.
- They are not triggered by differential pore-fluid pressure across the fault.
- The fault is a barrier, but both aquifer systems recharge simultaneously.
- Events happen on the rising limb of the aquifer recharge, and are insensitive to the magnitude of the peak.
- Groundwater levels fluctuate seasonally by ~3.7 m (east) and ~2.0 m (west), with the lowest years ~2.7–3.3 m below the long-term mean (numbers in Section 4).
- From the casing head, the mean depth to water is ~9 m on the east and ~14 m on the west, which makes slip at the actual water-table depth below the resolution of the data.
- The seasonal swing is different on each side of the fault. These kPa-scale swings are the same order of magnitude as the stress-drop calculations we made.
- The rising limb is the unclamping phase (more water → higher pore-fluid pressure on the fault → lower effective normal stress); drawdown is the reverse, by the same magnitude.
- The order of stress from groundwater swings is comparable to tidal stresses that modulate tremor (e.g., tidal modulation of tremor in Cascadia).

Open questions (mine):
1. Does the fault "feel" the water on both sides?
2. Is stress loading the clock, modulated by the hydrology, or is it coincidental that all events occur during the wet season?
3. Where does the tectonic stressing rate come from, and how does it compare with the hydrological stressing rate? (See Section 6. It has not been calculated independently yet.)

Information from the water district (ACWD): they put water in on the east side to maintain a well-managed water level. This agrees with the east well staying steady through the recent drought cycles.

---

## 2. Data

### 2.1 Well records (uploaded as .xlsx)
| File | Well | Side of fault | Start | End | Rows |
|---|---|---|---|---|---|
| `WL_4S1W27D008.xlsx` | 4S/1W-27D008 | East ("Above Hayward Fault" sub-basin) | 1990-10-02 | 2026-09-01 | 1924 |
| `WL_4S1W29A006.xlsx` | 4S/1W-29A006 | West ("Below Hayward Fault" sub-basin) | 1991-10-09 | 2026-09-01 | 2192 |

- Sheet `WaterLevel`, columns: `MeasurementDate`, `ReferencePointElevation`, `H2OElevation`, `WellName`.
- pandas reads `MeasurementDate` directly as datetime (not Excel serials).
- Sampling is roughly weekly, with some denser periods (29A006 in 2005–06, 2009–11, 2016, 2023). No gaps > 120 days.
- A few NaN `H2OElevation` rows (3 east, 1 west), so drop them. There are duplicate dates (e.g. 29A006, 2000-08-29), so average them.
- **Units are assumed to be feet (not confirmed).** Elevation datum not confirmed.
- Reference-point elevation changes: 27D008 at 1991-10-23, 1994-08-23, 2004-05-18; 29A006 at 1991-10-23, 1994-08-23, and Aug–Oct 2000 (alternates 61.23/61.47). There are no jumps in `H2OElevation` at these changes.
- Depth to water = `ReferencePointElevation − H2OElevation` (below casing head; not necessarily ground surface).
- **Screen depths of both wells are unknown.** The measured values are heads in the screened interval, not necessarily the water table.

### 2.2 Well locations (`WOI_map.pdf`)
- Both wells are roughly 0.8–1 km from the mapped trace (estimated from the 1,000 ft scale bar), and about 1.9 km apart.
- Other wells on the map, with data not yet obtained: 4S/1W-21R007, -28D001/008/011/012, -28F018/024, -28R003. The 28D/28F/28R wells appear to be roughly 250–500 m west of the trace. No well on the map is comparably close on the east side.
- The Quarry Lakes recharge ponds (top left of the map) span the fault. ACWD operates them as separate "Above Hayward Fault" (east) and "Below Hayward Fault" (west) recharge basins, separated by a levee. They are repurposed gravel quarries, used since the 1970s for recharge with imported water and to limit saline intrusion. ACWD manages each sub-basin to its own operating range. The ponds lie roughly 1–1.5 km north of the two wells (map estimate).
- Reference: Teague et al. (2018), USGS SIR 2018-5003 (Niles Cone recharge, including InSAR over Quarry Lakes).

### 2.3 Creep events (`fig3_history_forecast_...png`, my existing figure)
- A: HRKT alignment array 1980–2025 plus InSAR FP stack (n = 10) since 2015.
- B: onset windows on the water year, over 7-day rainfall.
- C: repeat intervals. InSAR 1.97 ± 0.08 yr (n = 4); alignment array 2.55 ± 0.98 yr (n = 7); combined 2.34 ± 0.81 yr (n = 11).
- D: probability of the next event. The high-probability window is roughly Oct 2026 to May 2027.

**Onset windows used so far were digitised by eye from Fig. 3B. They must be replaced with the real values.**
```python
AA  = [('1983-01-25','1983-05-20'),('1985-12-10','1986-03-15'),('1988-04-20','1988-06-01'),
       ('1993-03-20','1993-05-01'),('1997-04-25','1997-06-10'),('2000-01-25','2000-03-25'),
       ('2002-11-20','2003-01-25'),('2005-01-01','2005-03-20'),('2005-11-25','2006-02-05')]
INS = [('2017-01-25','2017-02-05'),('2019-03-08','2019-03-16'),('2021-02-03','2021-02-10'),
       ('2023-01-22','2023-02-08'),('2024-12-18','2025-01-05')]
FORECAST = ('2026-10-01','2027-05-01')  # approx. from Fig. 3D
```
11 events overlap the well records: 6 alignment array (1993 onward) and 5 InSAR. The "onset" used in the analyses is the midpoint of each window. The alignment-array windows are 42–78 days wide; the InSAR windows are 7–18 days.

---

## 3. Processing conventions used

- **Water year**: Oct 1 – Sep 30, labelled by the ending year.
- **Common grid**: weekly (`freq='7D'`) from 1991-10-09 to 2026-09-01. Linear time interpolation with `limit=6` (≤ 6 weeks).
- **Smoothing** for peaks, lows and rates: 3-sample centred rolling mean (about 3 weeks).
- **Cross-fault head difference**: dH = east − west on the common grid.
- **Pressure conversion**: 1 m head = 9.81 kPa.
- **Known pitfall**: pandas datetimes here are `datetime64[us]`. `Timestamp.value` is in ns but `DatetimeIndex.asi8` is in µs, so mixing them in `np.interp` silently clamps to the last value. Convert both with `.as_unit('ns')` first.

Existing scripts (from the exploration): `plot_wells.py`, `peaks.py`, `thresh.py`, `swing.py`.

---

## 4. Results of the exploration so far

### 4.1 Depth to water (below casing head)
| | Mean | Median | Range |
|---|---|---|---|
| East 27D008 | 9.4 m | 9.1 m | 4.8–15.5 m |
| West 29A006 | 14.3 m | 14.3 m | 11.2–18.3 m |

### 4.2 Seasonal swing, recharge vs drawdown (WY1993–2025)
Rise = autumn low to spring peak; fall = spring peak to the following low.

| | Rise (median, IQR, range) | Fall (median) | Rise duration | Fall duration | Rise rate | Fall rate | Rise in kPa |
|---|---|---|---|---|---|---|---|
| East | 3.65 m (2.79–4.67; 0.97–7.13) | 3.48 m | 189 d | 175 d | 0.63 m/mo | 0.57 m/mo | ~36 kPa |
| West | 1.99 m (1.66–2.66; 0–6.36) | 2.04 m | 161 d | 203 d | 0.42 m/mo | 0.33 m/mo | ~20 kPa |

Rise and fall are similar in magnitude on each side. The east swing is about twice the west swing.

### 4.3 Low years (annual-mean anomaly relative to the WY1992–2025 mean)
- East lowest: 1993 −3.3 m, 2000 −3.1, 1992 −2.5, 2013 −2.5, 1994 −2.3, 2007 −1.9. Drought years: 2014 −0.9, 2015 +1.5, 2021 +0.6, 2022 +0.4. The east well does not show the recent droughts.
- West lowest: 2014 −2.7 m, 1997 −2.3, 2013 −2.2, 1996 −1.9, 2022 −1.2, 2021 −1.0. The west well shows the 2013–15 and 2021–22 droughts. Its 1996–97 lows are not drought years.
- Lowest weekly level below the long-term mean: east −6.1 m (1999-11-03); west −3.9 m (2014-01-22).
- The east record has a long-term rise (annual means of roughly 25–30 ft in the early 1990s, roughly 40–46 ft in the 2020s).

### 4.4 Cross-fault head difference
- East is typically 20–30 ft (6–9 m) higher than west.
- The 1-yr running mean of dH was about 10–20 ft in the 1990s and about 25–30 ft through the InSAR era.

### 4.5 Timing of the seasonal peak and cross-fault phase
- Median peak day of water year: east 198, west 188 (about mid-April). IQR east 164–222, west 167–213 (early March to early May).
- Drought years (2013–15, 2020–21) have weak or autumn "peaks". Wet years (1995, 1998, 2011, 2017) peak late May to early July.
- Peak date, west minus east: median −14 d (IQR −26 to +12 d). Peak picking is noisy for flat-topped years.
- Cross-correlation of the series with a 1-yr running mean removed: max r = 0.778 at 0-week lag (0.777 at +1 week). The two sides are in phase at weekly resolution.

### 4.6 Onsets relative to the seasonal cycle
- All onset windows fall between December and June. InSAR onsets fall between late December and mid-March.
- Onsets are on the rising limb, typically 1–3 months before the seasonal peak.

### 4.7 Water-level state at onset (values in ft; onset = window midpoint)
| Onset | Source | E level | E rise since low | E frac of swing | E 4-wk rate (ft/wk) | W level | W rise | W frac | W rate | dH |
|---|---|---|---|---|---|---|---|---|---|---|
| 1993-04-10 | AA | 30.57 | 10.20 | 0.86 | 0.46 | 18.22 | 9.09 | 0.93 | −0.14 | 12.35 |
| 1997-05-18 | AA | 36.16 | 7.96 | 0.58 | −0.66 | 5.18 | −0.50 | −0.08 | −0.98 | 30.97 |
| 2000-02-24 | AA | 29.65 | 13.88 | 0.75 | 1.41 | 19.12 | 2.03 | 0.37 | 0.46 | 10.53 |
| 2002-12-23 | AA | 33.30 | 2.46 | 0.27 | 0.13 | 13.43 | 1.49 | 0.22 | 0.27 | 19.87 |
| 2005-02-09 | AA | 35.13 | 9.20 | 0.54 | 1.13 | 16.93 | 6.57 | 0.73 | 0.43 | 18.20 |
| 2005-12-31 | AA | 32.86 | 1.48 | 0.13 | 0.18 | 15.36 | 0.59 | 0.14 | 0.12 | 17.51 |
| 2017-01-30 | InSAR | 47.19 | 9.08 | 0.76 | 0.40 | 19.54 | 6.34 | 0.97 | 0.45 | 27.64 |
| 2019-03-12 | InSAR | 44.71 | 11.56 | 0.75 | 1.03 | 19.88 | 9.23 | 0.96 | 0.48 | 24.84 |
| 2021-02-06 | InSAR | 41.75 | 6.66 | 0.91 | 0.38 | 11.91 | 1.19 | 0.56 | 0.15 | 29.84 |
| 2023-01-30 | InSAR | 41.95 | 13.63 | 0.71 | 0.86 | 15.68 | 8.98 | 0.68 | 1.03 | 26.27 |
| 2024-12-27 | InSAR | 43.26 | 1.03 | 0.19 | 0.08 | 13.93 | 1.42 | 0.24 | 0.32 | 29.33 |

The null comparison uses the same calendar dates in non-event water years. Values are medians, with CV in brackets where computed.

| Variable | Events | Null |
|---|---|---|
| E level (ft) | 36.16 (CV 0.16) | 37.96 (CV 0.16) |
| E rise since low (ft) | 9.08 | 6.50 |
| E 4-wk rate (ft/wk) | 0.40 | 0.37 |
| W level (ft) | 15.68 | 15.03 |
| W rise since low (ft) | 2.03 | 3.78 |
| W 4-wk rate (ft/wk) | 0.32 | 0.23 |
| dH (ft) | 24.8 (range 10.5–31.0) | 22.8 (IQR 17.2–27.3) |

Event values fall within the null distributions for level, rise, rate and dH.

### 4.8 InSAR-era water years: event vs no event (ft)
| WY | Event | E peak | E swing | E max 4-wk rate | E Jan–Mar rise | W peak | W swing | W max 4-wk rate |
|---|---|---|---|---|---|---|---|---|
| 2016 | no | 49.53 | 10.90 | 0.80 | 7.14 | 19.81 | 9.26 | 1.24 |
| 2017 | yes | 49.99 | 11.88 | 0.87 | 2.75 | 19.75 | 6.54 | 0.86 |
| 2018 | no | 41.44 | 3.19 | 0.37 | −0.26 | 19.66 | 6.04 | 0.78 |
| 2019 | yes | 48.47 | 15.31 | 1.22 | 10.84 | 20.25 | 9.60 | 0.94 |
| 2020 | no | 44.62 | 6.41 | 1.17 | −2.71 | 16.52 | 5.45 | 0.53 |
| 2021 | yes | 42.38 | 7.29 | 0.79 | 1.91 | 12.84 | 2.12 | 0.26 |
| 2022 | no | 43.84 | 14.02 | 1.22 | 5.95 | 12.86 | 5.62 | 0.61 |
| 2023 | yes | 47.39 | 19.07 | 1.22 | 8.93 | 19.82 | 13.12 | 1.03 |
| 2024 | no | 50.65 | 6.92 | 0.77 | 5.36 | 20.07 | 4.45 | 0.44 |
| 2025 | yes | 47.55 | 5.33 | 0.56 | 3.67 | 18.36 | 5.85 | 0.60 |
| 2026 | no | 44.34 | 7.05 | 0.63 | 0.88 | 17.74 | 6.31 | 0.77 |

### 4.9 Superposed epoch (±365 d around onset midpoint)
- Change in level relative to onset, per well, compared with same-season windows (onset day ± 30 d) in non-event years. 400 random draws; 10–90% band shown.
- The median event curves for both wells lie within the 10–90% band.

### 4.10 Percentile-at-onset caveat
- Ranked against the same season in **all** years: InSAR onsets give east percentiles of 70–98 and alignment-array onsets give 7–45, a clean split.
- Ranked against the same season **within ±5 yr** of each event: the split disappears and the points scatter.
- The all-years split reflects the long-term rise in the east well (era effect). Any percentile or anomaly metric must use a local baseline or detrended series.

---

## 5. Stress conversions (as discussed)

- Pore-pressure change per seasonal swing: east about 36 kPa, west about 20 kPa. Drawdown is the same magnitude with the opposite sign.
- If the full Δp reaches the slip surface, ΔCFS = μ′·Δp with μ′ = 0.4–0.6. That gives east about 15–22 kPa peak-to-peak and west about 8–12 kPa.
- The fraction of well head change that reaches the slip surface is unknown.
- Reference values for tidal stresses that modulate tremor or LFEs: about 1–5 kPa (Rubinstein et al., 2008, Cascadia; Royer et al., 2015, Cascadia; Thomas et al., 2009, Parkfield). Seasonal hydrological loading of a few kPa that modulates seismicity: Johnson, Fu & Bürgmann (2017), California; Bettinelli et al. (2008), Himalaya.
- My stress-drop values for the Parkmont events: **TBD (insert)**.

---

## 6. Stressing-rate comparison (not yet done properly)

- The hydrological stressing rate during the rising limb, from the numbers above, is about (μ′·Δp)/(rise duration). That is roughly 30–44 kPa/yr east and 16–24 kPa/yr west.
- The tectonic stressing rate has **not** been calculated independently. The placeholder used in discussion was stress drop / recurrence interval (~2 yr), assuming each event releases the stress accumulated since the previous one. With stress drops similar to the groundwater swings, that gave about 10–20 kPa/yr. This is an assumption, not a result.
- To do: an independent estimate of the tectonic loading rate on the Parkmont patch, then the ratio to the hydrological rate.

---

## 7. Outstanding data and tasks

1. Replace the digitised onset windows with the real values, then rerun everything.
2. Confirm units (ft) and datum for `H2OElevation`.
3. Obtain screen and completion depths for 27D008 and 29A006 (ACWD or DWR well completion reports).
4. Obtain records for the other wells on the map (28D/28F/28R series, 21R007).
5. Obtain ACWD pond operation records (inflow to Above and Below Hayward Fault ponds) to separate managed recharge from natural recharge.
6. Add stress-drop values and compute the tectonic stressing rate independently (Section 6).
7. Address the open questions in Section 1 (fault "feeling" both sides; clock vs wet-season coincidence).

### Figures produced so far (to reproduce or extend)
- `hayward_wells_vs_SSE.png`: A hydrographs with onset windows; B cross-fault dH; C seasonal climatology (anomaly from water-year mean, median and IQR) with onset windows; D superposed epoch; E percentile at onset (±5 yr baseline).
- `hayward_wells_depth_peaks.png`: A depth to water (m, with ft on the secondary axis); B date of seasonal peak per water year, east vs west.
- `hayward_wells_trigger_tests.png`: level, rise since low, and 4-wk rate at onset vs null violins; InSAR-era swing in event vs no-event years.

Colour conventions (matching Fig. 3): InSAR `#c8202f`, alignment array `#0b2d6b`, east well `#0b6e99`, west well `#d4600a`, forecast window `#d6ebf7`.

---

## Update, 2 October 2026 (local session; facts only, the text above is unchanged)

- Item 1 done: the real onset windows (notebooks 04 and 09, `event_table.csv` of notebook 10) replace the digitised ones; onsets are drawn uniformly within their windows, never midpoints.
- Items 2 and 3 done (ACWD, Eileen, email 2 October 2026): levels in feet above NGVD29; screens 27D008 (east) 60–110 ft and 29A006 (west) 67–127 ft below ground surface. ACWD's other wells near Tule Pond have mostly semiannual readings (offered, not yet requested).
- The analysis code is `drafts/10x_hydrology/10x_rain_wells.py`; the report is
  `<data>/outputs/drafts_10x_hydrology/report/TulePond_hydrology_report_2026-10-02.pdf`.
