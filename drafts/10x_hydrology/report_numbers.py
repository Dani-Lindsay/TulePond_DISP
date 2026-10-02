"""Numbers and figures for the hydrology report (written to <data>/outputs/drafts_10x_hydrology/report/). Reads the draft notebook's cells
up to the daily quantities, then: the seasonal swing per water year in m and kPa, the state at every onset,
the lag of the wells behind the rain, the cross-fault head at the events, and three figures."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
src = open("10x_rain_wells.py").read()
exec(src.split("# %% [markdown]\n# ## [9] Onset draws")[0].replace("import matplotlib.pyplot as plt", ""))
R = os.path.join(out_dir, "report"); os.makedirs(R, exist_ok=True)     # products live with the data, not in the repo
kpa_per_m = 9.81
event_wys = set(rain.wy[events.window_lo].values)
utils.mpl_paper_style(base=8)
C = dict(insar="#D55E00", aa="#0B2D6B", east="#0072B2", west="#E69F00")

# 1. seasonal swing per water year: autumn low (1 Aug before the WY to the peak) to the WY peak, 3-week smoothed
rows = []
for wy in range(1992, 2027):
    r = dict(wy=wy, event=wy in event_wys)
    for side in ("east", "west"):
        s = levels[side].rolling(21, center=True, min_periods=7).mean()
        year = s[f"{wy-1}-10-01":f"{wy}-09-30"].dropna()
        if len(year) < 300: continue
        peak_day = year.idxmax(); low = s[f"{wy-1}-08-01":peak_day].min()
        r[f"{side}_low_m"], r[f"{side}_peak_m"], r[f"{side}_peak_dowy"] = low, year.max(), int(rain.dowy[peak_day])
        r[f"{side}_rise_m"] = year.max() - low; r[f"{side}_rise_kpa"] = (year.max() - low) * kpa_per_m
        r[f"{side}_mean_m"] = year.mean()
    rows.append(r)
swing = pd.DataFrame(rows).set_index("wy")
swing.round(2).to_csv(os.path.join(R, "seasonal_swing_by_wy.csv"))
print(swing.round(2).to_string())
for side in ("east", "west"):
    ok = swing[f"{side}_rise_m"].dropna()
    print(f"{side}: rise median {ok.median():.2f} m ({ok.median()*kpa_per_m:.0f} kPa), range {ok.min():.2f}-{ok.max():.2f} m "
          f"({ok.min()*kpa_per_m:.0f}-{ok.max()*kpa_per_m:.0f} kPa); peak day median {swing[f'{side}_peak_dowy'].median():.0f}")

# 2. the state at every onset: rise since the low at both ends of the window, in m and kPa; level anomaly; head difference
state = []
for e in events.itertuples():
    r = dict(source=e.source, window_lo=e.window_lo.date(), window_hi=e.window_hi.date(), wy=int(rain.wy[e.window_lo]))
    for side in ("east", "west"):
        for end, day in (("lo", e.window_lo), ("hi", e.window_hi)):
            rise = daily[f"{side}_rise"].get(day, np.nan)
            r[f"{side}_rise_{end}_kpa"] = rise * kpa_per_m
            r[f"{side}_anom_{end}_m"] = daily[f"{side}_anomaly"].get(day, np.nan)
            r[f"{side}_rate_{end}_kpa_wk"] = daily[f"{side}_rate"].get(day, np.nan) * kpa_per_m
        r[f"{side}_wy_rise_kpa"] = swing[f"{side}_rise_kpa"].get(r["wy"], np.nan)
    r["head_diff_lo_m"], r["head_diff_hi_m"] = daily.head_difference.get(e.window_lo, np.nan), daily.head_difference.get(e.window_hi, np.nan)
    r["rain_to_date_lo_mm"], r["rain_to_date_hi_mm"] = daily.rain_wy_to_date[e.window_lo], daily.rain_wy_to_date[e.window_hi]
    r["wy_rain_mm"] = float(rain_mm[rain.wy == r["wy"]].sum())
    state.append(r)
state = pd.DataFrame(state)
state.round(1).to_csv(os.path.join(R, "state_at_onset.csv"), index=False)
print(state.round(1).to_string(index=False))

# 3. lag of the wells behind the rain (weekly), and long-term level statistics
weekly_rain = rain_mm.resample("7D").sum()
lag_rows = []
for side in ("east", "west"):
    dl = levels[side].reindex(weekly_rain.index).diff()
    both = pd.concat([weekly_rain, dl], axis=1).dropna(); both.columns = ["rain", "dlevel"]
    cc = {lag: both.rain.corr(both.dlevel.shift(-lag)) for lag in range(0, 13)}
    best = max(cc, key=cc.get)
    lag_rows.append(dict(side=side, best_lag_weeks=best, r_best=cc[best], r_lag0=cc[0], n_weeks=len(both)))
    lv = levels[side].dropna()
    print(f"{side}: lag {best} wk (r {cc[best]:.2f}); level mean {lv.mean():.2f} m, range {lv.min():.2f}-{lv.max():.2f}; "
          f"InSAR era (2016-10 on) annual means {swing.loc[2017:2026, f'{side}_mean_m'].min():.2f}-{swing.loc[2017:2026, f'{side}_mean_m'].max():.2f} m")
pd.DataFrame(lag_rows).round(3).to_csv(os.path.join(R, "rain_well_lag.csv"), index=False)
hd = daily.head_difference.rolling(365, center=True, min_periods=200).mean()
print(f"cross-fault head, 1-yr mean: 1990s {hd['1993':'1999'].min():.1f}-{hd['1993':'1999'].max():.1f} m; InSAR era {hd['2016-10':].min():.1f}-{hd['2016-10':].max():.1f} m")

# 4. figures: (a) wells 4-panel, (b) cumulative rain by water year, (c) the storm raster is in qc10x
def shade_windows(ax):
    for e in events.itertuples():
        col = C["insar"] if e.source == "InSAR" else C["aa"]
        a, b = e.window_lo, e.window_hi
        ax.axvspan(a, max(b, a + pd.Timedelta(days=14)), color=col, alpha=0.35, lw=0)
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True, width_ratios=[1.6, 1])
ax = axes[0, 0]
for side in ("east", "west"):
    ax.plot(levels[side].index, levels[side].values, color=C[side], lw=0.7, label=f"{'27D008 (east)' if side == 'east' else '29A006 (west)'}")
shade_windows(ax); ax.set_xlim(pd.Timestamp("1991-01-01"), pd.Timestamp("2027-01-01")); ax.set_ylabel("Head (m above NGVD29)")
ax.legend(fontsize=6, loc="upper left", frameon=False); utils.panel_title(ax, "A", "Well heads either side of the fault; onset windows shaded")
ax = axes[1, 0]
ax.plot(daily.index, daily.head_difference, color="0.5", lw=0.5); ax.plot(hd.index, hd, color="k", lw=1.4)
shade_windows(ax); ax.set_xlim(pd.Timestamp("1991-01-01"), pd.Timestamp("2027-01-01")); ax.set_ylabel("East minus west head (m)")
utils.panel_title(ax, "B", "Cross-fault head difference (thin daily, thick 1-yr mean)")
ax = axes[0, 1]
for side in ("east", "west"):
    by = daily[f"{side}_anomaly"].to_frame("v").assign(wy=rain.wy.values, dowy=rain.dowy.values).pivot(index="dowy", columns="wy", values="v")
    ax.fill_between(by.index, by.quantile(0.25, axis=1), by.quantile(0.75, axis=1), color=C[side], alpha=0.2, lw=0)
    ax.plot(by.index, by.median(axis=1), color=C[side], lw=1.4, label="east" if side == "east" else "west")
y0 = ax.get_ylim()[0]
for k, e in enumerate(events.sort_values("window_lo").itertuples()):
    col = C["insar"] if e.source == "InSAR" else C["aa"]
    ax.plot([rain.dowy[e.window_lo], rain.dowy[e.window_hi]], [y0 - 0.3 - 0.22 * k] * 2, color=col, lw=2.2, solid_capstyle="butt")
ax.set_xticks([1, 62, 124, 183, 244, 305]); ax.set_xticklabels(["Oct", "Dec", "Feb", "Apr", "Jun", "Aug"]); ax.set_xlim(1, 366)
ax.axhline(0, color="0.6", lw=0.5, ls=":"); ax.set_ylabel("Head minus water-year mean (m)"); ax.legend(fontsize=6, frameon=False, loc="upper right")
utils.panel_title(ax, "C", "Seasonal cycle (median, IQR); onset windows below")
ax = axes[1, 1]
w = swing.loc[2016:2026]
x = np.arange(len(w))
for k, side in enumerate(("east", "west")):
    colours = [C["insar"] if ev else "0.7" for ev in w.event]
    ax.bar(x + (k - 0.5) * 0.4, w[f"{side}_rise_kpa"], 0.38, color=colours, edgecolor="k", lw=0.4, hatch="" if side == "east" else "///")
ax.set_xticks(x); ax.set_xticklabels([str(y)[2:] for y in w.index]); ax.set_xlabel("Water year (20xx)"); ax.set_ylabel("Seasonal rise (kPa)")
ax.legend([plt.Rectangle((0, 0), 1, 1, color=C["insar"]), plt.Rectangle((0, 0), 1, 1, color="0.7"), plt.Rectangle((0, 0), 1, 1, fc="w", ec="k"),
           plt.Rectangle((0, 0), 1, 1, fc="w", ec="k", hatch="///")], ["InSAR event year", "no event", "east", "west"], fontsize=6, frameon=False, ncol=2)
utils.panel_title(ax, "D", "Seasonal pore-pressure rise, InSAR era")
fig.savefig(os.path.join(R, "fig_wells.png"), dpi=200)

fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
for wy in np.unique(rain.wy):
    cum = daily.rain_wy_to_date[rain.wy == wy]; d = rain.dowy[rain.wy == wy]
    if wy in event_wys: continue
    ax.plot(d, cum, color="0.75", lw=0.6)
for e in events.itertuples():
    wy = int(rain.wy[e.window_lo]); col = C["insar"] if e.source == "InSAR" else C["aa"]
    cum = daily.rain_wy_to_date[rain.wy == wy]; d = rain.dowy[rain.wy == wy]
    ax.plot(d, cum, color=col, lw=0.9)
    m = (cum.index >= e.window_lo) & (cum.index <= e.window_hi)
    ax.plot(d[m], cum[m], color=col, lw=4, solid_capstyle="butt")
    if e.source == "InSAR":                      # label the InSAR years only (the array labels crowd the panel)
        ax.text(d[m][-1] + 3, cum[m].iloc[-1], f"WY{wy}", fontsize=6, color=col, va="center")
ax.set_xticks([1, 32, 62, 93, 124, 152, 183, 213, 244, 274, 305, 336]); ax.set_xticklabels(["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"])
ax.set_xlim(1, 366); ax.set_ylabel("Rain since 1 Oct (mm)")
ax.legend([plt.Line2D([], [], color="0.75"), plt.Line2D([], [], color=C["insar"], lw=3), plt.Line2D([], [], color=C["aa"], lw=3)],
          ["water years without an event (1979-2026)", "InSAR event year, onset window thick", "array event year, onset window thick"], fontsize=6.5, frameon=False, loc="upper left")
utils.panel_title(ax, "", "Cumulative rain through the water year, Tule Pond composite gauge")
fig.savefig(os.path.join(R, "fig_cumulative_rain.png"), dpi=200)
print("figures written to", R)
