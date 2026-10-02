"""Build the hydrology report PDF from the numbers report_numbers.py wrote (run 10x_rain_wells.py, then
report_numbers.py, first). Needs reportlab, which is not in opera_disp_feb13: run it in any env that has it,
e.g. a scratch pixi env (pixi init; pixi add reportlab pillow pandas; pixi run python build_report.py)."""
import os
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether

OUT = "/Users/dlindsay/work/proj_TulePond_DISP/outputs/drafts_10x_hydrology"
HERE = os.path.join(OUT, "report")          # the report's tables, figures and the PDF live with the data, not in the repo
FIG = "/Users/dlindsay/work/proj_TulePond_DISP/Figures_2026-10-02"
swing = pd.read_csv(os.path.join(HERE, "seasonal_swing_by_wy.csv"), index_col="wy")
state = pd.read_csv(os.path.join(HERE, "state_at_onset.csv"))
lag = pd.read_csv(os.path.join(HERE, "rain_well_lag.csv")).set_index("side")
storm = pd.read_csv(os.path.join(OUT, "storm_test.csv"))
st_other, st_own, st_insar, st_array = (storm.iloc[i] for i in range(4))
tests = pd.read_csv(os.path.join(OUT, "hydrology_tests.csv"))
tq = tests.set_index("quantity")
fmt_p = lambda p: "< 0.005" if p < 0.005 else f"{p:.3f}"

# Verdana: wide, open letterforms and generous spacing (easier to read); larger size and line spacing, black text,
# left-aligned, no small grey print
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
FONT_DIR = "/System/Library/Fonts/Supplemental"
pdfmetrics.registerFont(TTFont("Verdana", os.path.join(FONT_DIR, "Verdana.ttf")))
pdfmetrics.registerFont(TTFont("Verdana-Bold", os.path.join(FONT_DIR, "Verdana Bold.ttf")))
pdfmetrics.registerFontFamily("Verdana", normal="Verdana", bold="Verdana-Bold", italic="Verdana", boldItalic="Verdana-Bold")
ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["Normal"], fontName="Verdana", fontSize=10.5, leading=16.5, spaceAfter=8, alignment=0)
small = ParagraphStyle("small", parent=body, fontSize=9.5, leading=15)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Verdana-Bold", fontSize=14, leading=19, spaceBefore=16, spaceAfter=8)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Verdana-Bold", fontSize=11.5, leading=16, spaceBefore=12, spaceAfter=5)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=16, bulletIndent=3, spaceAfter=7)
cap = ParagraphStyle("cap", parent=small, spaceBefore=4, spaceAfter=12)


def P(text, style=body):
    return Paragraph(text, style)


def B(items):
    return [Paragraph(t, bullet, bulletText="•") for t in items]


def table(rows, col_widths=None, font=8.5, header=True):
    t = Table(rows, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [("FONT", (0, 0), (-1, -1), "Verdana", font), ("LEADING", (0, 0), (-1, -1), font + 4),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.black),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
             ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")])]   # banded rows: easier to follow across
    if header:
        style += [("FONT", (0, 0), (-1, 0), "Verdana-Bold", font), ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
                  ("LINEABOVE", (0, 0), (-1, 0), 0.6, colors.black)]
    t.setStyle(TableStyle(style))
    return t


def fig(path, width_cm, caption):
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    return [Image(path, width=width_cm * cm, height=width_cm * cm * h / w), P(caption, cap)]


east_med, west_med = swing.east_rise_m.median(), swing.west_rise_m.median()
insar = state[state.source == "InSAR"]
arr = state[(state.source != "InSAR") & state.east_rise_lo_kpa.notna()]
# values quoted in the text, all derived from the tables
frac_lo, frac_hi = (insar.rain_to_date_lo_mm / insar.wy_rain_mm).min(), (insar.rain_to_date_hi_mm / insar.wy_rain_mm).max()
ins_e = (min(insar.east_rise_lo_kpa.min(), insar.east_rise_hi_kpa.min()), max(insar.east_rise_lo_kpa.max(), insar.east_rise_hi_kpa.max()))
ins_w = (min(insar.west_rise_lo_kpa.min(), insar.west_rise_hi_kpa.min()), max(insar.west_rise_lo_kpa.max(), insar.west_rise_hi_kpa.max()))
ins_rate = (min(insar.east_rate_lo_kpa_wk.min(), insar.east_rate_hi_kpa_wk.min()), max(insar.east_rate_lo_kpa_wk.max(), insar.east_rate_hi_kpa_wk.max()))
ins_hd = (min(insar.head_diff_lo_m.min(), insar.head_diff_hi_m.min()), max(insar.head_diff_lo_m.max(), insar.head_diff_hi_m.max()))
covered = state[state.east_rise_lo_kpa.notna()].copy()
rain_daily = pd.read_csv("/Users/dlindsay/work/proj_TulePond_DISP/outputs/10_rain_seasonality/tulepond_daily_precip.csv", index_col="date", parse_dates=True)
covered["dowy_lo"] = rain_daily.dowy.reindex(pd.to_datetime(covered.window_lo)).values
covered["dowy_hi"] = rain_daily.dowy.reindex(pd.to_datetime(covered.window_hi)).values
covered["e_peak"] = swing.east_peak_dowy.reindex(covered.wy).values
covered["w_peak"] = swing.west_peak_dowy.reindex(covered.wy).values
opens_before = ((covered.dowy_lo < covered.e_peak) & (covered.dowy_lo < covered.w_peak))
closes_before = ((covered.dowy_hi < covered.e_peak) & (covered.dowy_hi < covered.w_peak))
wy_totals = rain_daily.groupby("wy").prcp_mm.sum()
wy_totals = wy_totals[(wy_totals.index >= 1980) & (wy_totals.index <= 2025)]
pct = lambda wy: 100 * (wy_totals < wy_totals[wy]).mean()
print(f"InSAR rain fraction {frac_lo:.2f}-{frac_hi:.2f}; east rise {ins_e}; west {ins_w}; rate {ins_rate}; head diff {ins_hd}; "
      f"windows opening before both peaks {opens_before.sum()} of {len(covered)}, closing before {closes_before.sum()}; "
      f"WY2021 {wy_totals[2021]:.0f} mm pct {pct(2021):.0f}, WY2023 {wy_totals[2023]:.0f} mm pct {pct(2023):.0f}")

story = []
story += [P("Tule Pond slow slip events and the hydrology: what has been tested", ParagraphStyle("t", parent=h1, fontSize=17, leading=23, spaceAfter=6)),
          P("Working report for Danielle Lindsay and Roland Bürgmann. Prepared 2 October 2026 from the experimental notebook "
            "drafts/10x_hydrology/10x_rain_wells in the TulePond_DISP repository. Nothing in it is registered "
            "for the paper; every number here is reproducible from that notebook and report_numbers.py.", small),
          Spacer(1, 6)]

story += [P("1. The short version", h1)]
story += B([
    "All 5 InSAR onsets and 6 of the 9 bimonthly-era array onsets fall between December and March; the InSAR onsets fall between 13 December and 9 March. "
    "The events are a wet-season phenomenon. That is the observation the forecast uses, and it needs no mechanism.",
    f"Storms: a 3-day rain total of at least {st_own.Z_mm:.0f} mm precedes the onset more often than chance ({st_own.hits:.1f} of {st_own.n:.0f} onsets within "
    f"{st_own.X_d:.0f} days of one, against {st_own.expected:.1f} expected at a random time in the same year and season; p = {st_own.p_best:.3f}, or "
    f"{st_own.p_search_paid:.2f} once the search over storm size and lag is paid for). The InSAR events carry this ({st_insar.hits:.1f} of {st_insar.n:.0f}); "
    f"the array windows are too wide to test it. Storms are not rare: {st_own.storms_per_wy_median:.0f} per water year (range {st_own.storms_per_wy_min:.0f} to "
    f"{st_own.storms_per_wy_max:.0f}), and {st_own.wet_days_within_21d_pct:.0f} % of December-to-March days lie within 21 days after one.",
    f"No threshold: at the InSAR onsets the water-year rain to date spans {insar.rain_to_date_lo_mm.min():.0f} to {insar.rain_to_date_hi_mm.max():.0f} mm "
    f"({100 * frac_lo:.0f} to {100 * frac_hi:.0f} % of that year's total), and the aquifer rise since its autumn low spans "
    f"{ins_e[0]:.0f} to {ins_e[1]:.0f} kPa (east) and {ins_w[0]:.0f} to {ins_w[1]:.0f} kPa (west).",
    f"No dependence on how wet the year is: events occurred in WY2021 ({wy_totals[2021]:.0f} mm, the driest water year since 1980; west-well rise "
    f"{swing.west_rise_kpa[2021]:.0f} kPa) and WY2023 ({wy_totals[2023]:.0f} mm, wetter than {pct(2023):.0f} % of years; {swing.west_rise_kpa[2023]:.0f} kPa).",
    f"Recharge limb: {opens_before.sum()} of the {len(covered)} onset windows covered by the wells open before that year's peak on both sides of the fault "
    f"({closes_before.sum()} also close before it; the 1993 window runs past the west peak). The exception is the 1997 array event (window 27 April to 21 June), "
    "which falls after both peaks, with heads falling on both sides.",
    "Cross-fault pressure difference: no signal. The east well sits 6 to 9 m above the west; the 1-yr mean difference was 6.4 to 9.0 m through the whole InSAR era "
    f"and {ins_hd[0]:.1f} to {ins_hd[1]:.1f} m at the five InSAR onsets.",
    f"The wells follow the rain with a lag of 1 to 2 weeks (weekly level change against weekly rain, best lag {int(lag.loc['east', 'best_lag_weeks'])} weeks east, "
    f"{int(lag.loc['west', 'best_lag_weeks'])} week west, r = {lag.loc['east', 'r_best']:.2f} and {lag.loc['west', 'r_best']:.2f}). The aquifer is a smoothed, "
    "partly managed version of the rain record, so rain and aquifer pressure cannot be separated as causes with these data.",
    f"The seasonal pore-pressure cycle in the screened interval (18 to 39 m deep) is {east_med * 9.81:.0f} kPa east (median rise {east_med:.1f} m) and "
    f"{west_med * 9.81:.0f} kPa west ({west_med:.1f} m), falling by about the same amount each summer. In the InSAR era the rise varied six-fold between years "
    f"({swing.loc[2017:2026, 'east_rise_kpa'].min():.0f} to {swing.loc[2017:2026, 'east_rise_kpa'].max():.0f} kPa east) and the events did not track it.",
])
story += [P("Conclusion for the paper: the hydrology narrows the forecast window to the wet season; it does not pick the year, and nothing in the wells adds to what the "
            "rain record says. The wells are consistent with a patch that is unclamped a little every recharge season and fails on the clock set by tectonic loading. "
            "That is a statement about timing, not about mechanism, and the report stops there on purpose.")]

story += [P("2. The data", h1), P("2.1 Events", h2)]
story += B(["InSAR: the five Tule Pond events of notebook 04, onset windows of 6 to 12 days (the acquisition spacing).",
            "Alignment array HRKT (km 61.2, inside the section): the nine events of the bimonthly era 1979.7 to 2007.4, notebook 09, windows of 43 to 124 days "
            "(the survey interval that contains the step). Three of them (1983, 1986, 1988) predate the wells.",
            "The onset is taken as equally likely anywhere in its window. Every test below draws it uniformly within the window (2000 draws per event); "
            "every quantity at onset is given at both ends of the window, never at a midpoint."])
story += [P("2.2 Rain", h2),
          P("The daily composite of notebook 10: the two nearest NOAA GHCN-daily gauges (Fremont 4 km, Newark 7 km), Fremont where it reported and Newark "
            "otherwise, the morning-observer Newark gauge shifted one day earlier (checked against Hayward Air Terminal, which is not in the composite); "
            "1 October 1978 to 25 August 2026; 99 days have no reading. Water years run 1 October to 30 September "
            "and are named by the year they end. The median water-year total is 362 mm (10 to 90 %: 219 to 567 mm).")]
story += [P("2.3 Groundwater (ACWD indicator wells, received 2 October 2026)", h2)]
story += [table([["", "4S/1W-27D008", "4S/1W-29A006"],
                 ["Side of the fault", "east ('above Hayward Fault')", "west ('below Hayward Fault')"],
                 ["Distance from the trace", "0.8 to 1 km (map estimate)", "0.8 to 1 km (map estimate)"],
                 ["Screen (below ground)", "60 to 110 ft (18 to 34 m)", "67 to 127 ft (20 to 39 m)"],
                 ["Record", "2 Oct 1990 to 1 Sep 2026", "9 Oct 1991 to 1 Sep 2026"],
                 ["Readings", "1880, about weekly", "2147, about weekly"],
                 ["Head (m above NGVD29)", f"mean {10.8:.1f}, range 4.8 to 15.5", f"mean {4.3:.1f}, range 0.4 to 7.6"],
                 ["Depth to water below casing", "about 9 m", "about 14 m"],
                 ["Seasonal rise, median", f"{east_med:.2f} m = {east_med * 9.81:.0f} kPa", f"{west_med:.2f} m = {west_med * 9.81:.0f} kPa"],
                 ["Seasonal rise, range", f"{swing.east_rise_kpa.min():.0f} to {swing.east_rise_kpa.max():.0f} kPa", f"{swing.west_rise_kpa.min():.0f} to {swing.west_rise_kpa.max():.0f} kPa"],
                 ["Peak of the cycle", f"day {swing.east_peak_dowy.median():.0f} (mid-April)", f"day {swing.west_peak_dowy.median():.0f} (early April)"]],
                col_widths=[4.6 * cm, 6.4 * cm, 6.4 * cm], font=8)]
story += [P("The levels are heads above the 1929 national datum (NGVD29), in feet in the files and converted to metres here. Both screens sit below the water level, "
            "so each well reads the pressure head in a screened interval 18 to 39 m deep, not the water table. A change of 1 m of head is a change of 9.81 kPa in pore "
            "pressure at the screen. Readings were linearly interpolated to daily values across gaps of up to 6 weeks; duplicate dates were averaged. "
            f"The east well rises over the record (annual means {swing.loc[1992:1994, 'east_mean_m'].min():.1f} to {swing.loc[1992:1994, 'east_mean_m'].max():.1f} m in 1992-94, "
            f"{swing.loc[2020:2026, 'east_mean_m'].min():.1f} to {swing.loc[2020:2026, 'east_mean_m'].max():.1f} m since 2020) and does not show the 2013-15 and 2021-22 droughts; "
            "ACWD recharges that side deliberately. The west well shows both droughts. Any comparison across years therefore uses within-year quantities (rise since the "
            "autumn low, anomaly from the water-year mean) rather than absolute levels.", small)]

story += [P("3. The pore-pressure numbers", h1),
          P("Rise = the head at the water-year peak minus its lowest value from the preceding 1 August (3-week smoothed), in metres and in kPa (x 9.81). "
            "Water years with an event are marked. A rise of 0 means the level never climbed above its autumn value that year.")]
rows = [["Water\nyear", "Event", "East rise\n(m)", "East rise\n(kPa)", "East\npeak day", "West rise\n(m)", "West rise\n(kPa)", "West\npeak day"]]
for wy, r in swing.iterrows():
    rows.append([str(wy), "yes" if r.event else "", f"{r.east_rise_m:.2f}", f"{r.east_rise_kpa:.0f}", f"{r.east_peak_dowy:.0f}",
                 f"{r.west_rise_m:.2f}" if pd.notna(r.west_rise_m) else "", f"{r.west_rise_kpa:.0f}" if pd.notna(r.west_rise_kpa) else "",
                 f"{r.west_peak_dowy:.0f}" if pd.notna(r.west_peak_dowy) else ""])
story += [table(rows, font=8.5, col_widths=[1.6 * cm, 1.5 * cm] + [2.2 * cm] * 6)]
story += [Spacer(1, 8), P("The state of the system at each onset. Each quantity is given at the start and end of the onset window (lo / hi). "
                          "Rise = head gained since that water year's autumn low; rate = change over the previous 4 weeks. kPa throughout.", body)]
rows_a = [["Event", "Onset window", "Year's rain\n(mm)", "Rain to date\n(mm)", "East rise\n(kPa)", "East rate\n(kPa/wk)"]]
rows_b = [["Event", "Onset window", "West rise\n(kPa)", "West rate\n(kPa/wk)", "East minus west\nhead (m)", "Year's full rise\nE / W (kPa)"]]
for r in state.itertuples():
    has = pd.notna(r.east_rise_lo_kpa)
    name, window = ("InSAR" if r.source == "InSAR" else "array"), f"{r.window_lo} to {r.window_hi}"
    rows_a.append([name, window, f"{r.wy_rain_mm:.0f}", f"{r.rain_to_date_lo_mm:.0f} / {r.rain_to_date_hi_mm:.0f}",
                   f"{r.east_rise_lo_kpa:.0f} / {r.east_rise_hi_kpa:.0f}" if has else "before the wells",
                   f"{r.east_rate_lo_kpa_wk:+.1f} / {r.east_rate_hi_kpa_wk:+.1f}" if has else ""])
    if has:
        rows_b.append([name, window, f"{r.west_rise_lo_kpa:.0f} / {r.west_rise_hi_kpa:.0f}", f"{r.west_rate_lo_kpa_wk:+.1f} / {r.west_rate_hi_kpa_wk:+.1f}",
                       f"{r.head_diff_lo_m:.1f} / {r.head_diff_hi_m:.1f}", f"{r.east_wy_rise_kpa:.0f} / {r.west_wy_rise_kpa:.0f}"])
widths = [1.5 * cm, 4.6 * cm, 2.4 * cm, 2.8 * cm, 2.9 * cm, 2.8 * cm]
story += [P("Rain and the east well:", body), table(rows_a, font=8.5, col_widths=widths), Spacer(1, 10),
          P("The west well, the cross-fault head and the year's full rise (events since the wells began):", body),
          table(rows_b, font=8.5, col_widths=widths), Spacer(1, 6)]
story += [P(f"Reading the table: the InSAR onsets fall after {ins_e[0]:.0f} to {ins_e[1]:.0f} kPa of the east well's seasonal rise (that year's full rise "
            f"{insar.east_wy_rise_kpa.min():.0f} to {insar.east_wy_rise_kpa.max():.0f} kPa) and {ins_w[0]:.0f} to {ins_w[1]:.0f} kPa of the west's "
            f"({insar.west_wy_rise_kpa.min():.0f} to {insar.west_wy_rise_kpa.max():.0f} kPa). The east rate at the InSAR onsets is {ins_rate[0]:+.1f} to {ins_rate[1]:+.1f} kPa per week: "
            "rising at four, flat at the December 2024 onset, which came at the very start of that year's recharge. The 1997 array event is the only one with heads falling "
            "on both sides through its window.", body)]

story += [P("4. The tests", h1),
          P("Each quantity is compared, at the drawn onset days, with (a) the same day of the water year in every water year without an event (\"other years\": is the "
            "time of year unusual in this respect?) and (b) the event's own water year at a random day drawn from the season the events occur in (\"own year\": does the "
            "timing within the year matter, or only the kind of year?). Two-sided p from 4000 realisations.")]
names = {"rain_wy_to_date": "Rain since 1 Oct (mm)", "days_since_storm": "Days since a 40 mm storm", "rain_previous": "Rain in previous 30 d (mm)",
         "east_anomaly": "East head minus WY mean (m)", "east_rise": "East rise since low (m)", "east_rate": "East 4-week change (m/wk)",
         "west_anomaly": "West head minus WY mean (m)", "west_rise": "West rise since low (m)", "west_rate": "West 4-week change (m/wk)",
         "east_days_since_low": "East: days since the low", "west_days_since_low": "West: days since the low", "head_difference": "East minus west head (m)"}
rows = [["Quantity", "Events", "Event\nmean", "Other years\nmedian", "p\nother years", "Own year\nmedian", "p\nown year"]]
for r in tests.itertuples():
    rows.append([names.get(r.quantity, r.quantity), str(r.n_events), f"{r.event_mean:.2f}", f"{r.other_years_median:.2f}", f"{r.p_other_years:.3f}",
                 f"{r.own_year_median:.2f}", f"{r.p_own_year:.3f}"])
story += [table(rows, font=8.5, col_widths=[5.6 * cm, 1.5 * cm, 1.8 * cm, 2.3 * cm, 2.2 * cm, 2.0 * cm, 1.9 * cm])]
story += [P(f"Twelve tests at p < 0.05 would produce about one false positive; the west 'days since the low' at {tq.p_other_years['west_days_since_low']:.3f} / "
            f"{tq.p_own_year['west_days_since_low']:.3f} is treated as that. The two rain results "
            "(rain to date, other years; rain in the previous 30 days) say that event years are wet years: within their own year the onsets sit at an ordinary point of the "
            f"season's total (p = {tq.p_own_year['rain_wy_to_date']:.2f}). The 30-day rain does not separate event by event either: only 4 of 14 onsets sit above the 95 % line of event-free years for most "
            "of their window; the 2019, 2021 and 2024 events sit well inside the band.", body)]
story += [P("4.1 The storm test", h2),
          P("Notebook 11's rule C (the paper's Fig. 3B): an onset 'hits' if a 3-day storm of at least Z mm ended within X days before it; Z and X were searched over "
            "20 to 50 mm and 7 to 30 days and the best pair (40 mm, 21 d) reported with p = 0.02 against the wet-season base rate. Redone here with the onsets drawn "
            "within their windows and against the own-year null, and with the search over the 16 (Z, X) pairs paid for (how often the null's own best pair does as well):")]
story += [table([["Null", "Hits", "Expected", "p, best pair", "p, search paid for"]] +
                [[r.null, f"{r.hits:.1f} of {r.n:.0f}", f"{r.expected:.1f}", fmt_p(r.p_best), fmt_p(r.p_search_paid) if pd.notna(r.p_search_paid) else ""]
                 for r in storm.itertuples()], col_widths=[6.6 * cm, 2.4 * cm, 2.0 * cm, 2.4 * cm, 3.2 * cm])]
story += [P(f"So: against random timing within the same wet season, storm proximity survives at p = {st_own.p_search_paid:.2f}, carried by the five InSAR events. It is modest evidence from "
            "five events, and the 1 to 2 week lag of the aquifer behind the rain means 'storm within 21 days' and 'aquifer rising' are not independent descriptions.", body)]

story += [PageBreak(), P("5. Figures", h1)]
story += [KeepTogether(fig(os.path.join(HERE, "fig_cumulative_rain.png"), 15.0, "Figure 1. Cumulative rain through the water year. Grey: every water year without an event (1979 to 2026). "
             "Coloured: the event years, with the onset window thick (vermillion InSAR, labelled; navy array). The InSAR onsets fall between "
             f"{insar.rain_to_date_lo_mm.min():.0f} and {insar.rain_to_date_hi_mm.max():.0f} mm of accumulated rain, in the driest water year since 1980 (WY2021) "
             f"and in one wetter than {pct(2023):.0f} % of years (WY2023)."))]
story += [PageBreak()]
story += fig(os.path.join(HERE, "fig_wells.png"), 16.5, "Figure 2. The two ACWD wells. A: heads, with the onset windows shaded. B: the cross-fault head difference. "
             "C: the seasonal cycle as the anomaly from each water year's mean (median and interquartile range over all years), with the onset windows drawn below. "
             "D: the seasonal rise in kPa for each water year of the InSAR era, event years in vermillion.")
story += [PageBreak()]
story += fig(os.path.join(FIG, "qc10x_hydrology_tests.png"), 16.5, "Figure 3. Each quantity through the water year, event years against the rest. Grey band and line: "
             "10 to 90 % and median of the water years without an event. Coloured: each event's own year, the onset window thick. The storm panel is one row per event: "
             "its 40 mm storms as ticks, its window as a bar, the 21 days before the window as a pale box. The p in each corner is the 'other years' test of section 4.")

story += [PageBreak(), P("6. Not tested, or not testable with these data", h1),
          P("Listed so that anything missing can be picked up.")]
story += B([
    "Pore pressure at the slip depth. The wells read 18 to 39 m; the slip is in the top 300 m. How much of the 20 to 36 kPa seasonal cycle reaches the fault depends on the "
    "hydraulic diffusivity of the fault-zone sediments (a periodic signal decays as exp(-z/d) with d = sqrt(2D/omega)), which is unknown here. No Coulomb stress change on the "
    "fault is computed for that reason; the report stays with the aquifer numbers.",
    "Tectonic stressing rate on the patch, and its ratio to the hydrological rate (about 20 to 36 kPa over a 5 to 6 month rise). Not calculated.",
    "The undrained (instantaneous, poroelastic) response to the water load, as distinct from diffusion. Not separated.",
    "ACWD's recharge-pond operations (volumes delivered to the 'above' and 'below' basins). Would separate managed from natural recharge; not obtained.",
    "The semiannual wells closer to the trace (28D, 28F, 28R, 21R series; some within 250 to 500 m west of the trace). ACWD offered them; not requested.",
    "Groundwater pumping records and the summer drawdown as a possible clamping signal (no event has occurred on the falling limb except 1997).",
    "A lagged version of the rain tests: rain in the 1 to 2 weeks before the drawn onset (the aquifer lag), as distinct from the 30-day total and the 21-day storm rule.",
    "Soil moisture or evapotranspiration (the surface water that does not reach the aquifer).",
    "Whether the array events of the 1980s and 1990s were preceded by storms at a shorter lag than 21 days: their windows (43 to 124 days) are wider than any plausible lag.",
    "Events on the other creeping sections (Oakland COZ1 events of 2019 and 2022, notebook 04) against the same hydrology, as a control.",
])
story += [Spacer(1, 10), P("Files: the notebook in drafts/10x_hydrology/ (its HTML in outputs/drafts_10x_hydrology/); this report's tables (seasonal_swing_by_wy.csv, state_at_onset.csv, rain_well_lag.csv) "
                           "and figures in outputs/drafts_10x_hydrology/report/; the test table in outputs/drafts_10x_hydrology/hydrology_tests.csv. ACWD data in "
                           "external_inputs/ACWD_water_well_data/. The paper's rain products are notebooks 10 and 11.", small)]

doc = SimpleDocTemplate(os.path.join(HERE, "TulePond_hydrology_report_2026-10-02.pdf"), pagesize=A4, leftMargin=1.9 * cm, rightMargin=1.9 * cm,
                        topMargin=1.9 * cm, bottomMargin=1.9 * cm, title="Tule Pond slow slip events and the hydrology", author="Danielle Lindsay")
doc.build(story)
print("wrote", os.path.join(HERE, "TulePond_hydrology_report_2026-10-02.pdf"))
