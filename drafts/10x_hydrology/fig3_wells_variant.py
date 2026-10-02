"""Fig 3 variant (Dani 2026-10-03): panel C = the aquifer seasonal cycle either side of the fault with the onset
windows, in place of the repeat intervals. Everything else is 11_forecast's Fig 3, drawn by 11's own code.

How: 11_forecast's cells up to and including Fig 3 are exec'd with its output and figure folders pointed at this
draft's scratch folder and its registry neutralised, so nothing of the paper chain is overwritten or registered.
Panel C is then cleared and redrawn from the experimental hydrology notebook's daily well series.
Writes <fig_dir>/qc10x_fig3_wells_variant.{png,pdf}.  Run in opera_disp_feb13 (about a minute)."""
import os
import sys

REPO = "/Users/dlindsay/tools/internal/TulePond_DISP"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts")]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from TulePond_filepaths import common_paths

SCRATCH = os.path.join(common_paths["out_dir"], "drafts_10x_hydrology", "fig3_variant_scratch")
os.makedirs(SCRATCH, exist_ok=True)


class NoReg:                                   # nothing is registered from here
    def inputs(self, *a): return self
    def register(self, key, value, *a, **k): return value
    def sidecar(self, name, lines): return name
    def render(self): return None


# --- 11_forecast up to and including Fig 3, writing only into SCRATCH
src = open(os.path.join(REPO, "scripts", "11_forecast.py")).read()
head = src.split("# %% [markdown]\n# ## [31]")[0]
for old, new in (('out_dir = os.path.join(common_paths["out_dir"], notebook)', f'out_dir = {SCRATCH!r}'),
                 ('fig_dir = common_paths["fig_dir"]', f'fig_dir = {SCRATCH!r}'),
                 ('REG = utils.Registry(notebook, common_paths, repo_dir)', 'REG = NoReg()')):
    assert head.count(old) == 1, f"11_forecast.py layout changed: {old!r}"
    head = head.replace(old, new)
ns = {"__name__": "fig3_variant", "NoReg": NoReg}
os.chdir(os.path.join(REPO, "scripts"))
exec(compile(head, "11_forecast_head", "exec"), ns)
fig, ax_c, label_pt = ns["fig"], ns["ax_c"], ns["label_pt"]

# --- the daily well series and the event windows, from the experimental hydrology notebook (cells up to [8])
hydro_src = open(os.path.join(HERE, "10x_rain_wells.py")).read()
hydro = {"__name__": "hydro"}
exec(compile(hydro_src.split("# %% [markdown]\n# ## [9] Onset draws")[0].replace("import matplotlib.pyplot as plt", ""), "10x_head", "exec"), hydro)
daily, rain, events = hydro["daily"], hydro["rain"], hydro["events"]
kpa_per_m = 9.81

# --- C) pore pressure through the water year (head minus its water-year mean, x 9.81), median and IQR, and the windows
ax_c.clear()
side_style = {"east": ("#0072B2", "27D008 (east)"), "west": ("#E69F00", "29A006 (west)")}
lowest = 0.0
for side, (colour, label) in side_style.items():
    by_day = (daily[f"{side}_anomaly"] * kpa_per_m).to_frame("v").assign(wy=rain.wy.values, dowy=rain.dowy.values)
    table = by_day.pivot(index="dowy", columns="wy", values="v")
    table = table.loc[:, table.notna().mean() > 0.9]              # complete water years only
    q25, q50, q75 = table.quantile(0.25, axis=1), table.median(axis=1), table.quantile(0.75, axis=1)
    ax_c.fill_between(table.index, q25, q75, color=colour, alpha=0.22, lw=0)
    ax_c.plot(table.index, q50, color=colour, lw=1.3, label=f"{label}")
    lowest = min(lowest, float(q25.min()))
    print(f"{side}: {table.shape[1]} water years; median cycle {q50.min():.1f} to {q50.max():.1f} kPa, peak day {int(q50.idxmax())}")
ax_c.axhline(0, color="0.6", lw=0.5, ls=":")
# the onset windows, one row each, oldest at the top, just under the curves
row_gap = 1.45                                                      # kPa between rows: compact, the curves keep the panel
top = lowest - 2.2
ordered = events.sort_values("window_lo").reset_index(drop=True)
for k, e in enumerate(ordered.itertuples()):
    y = top - k * row_gap
    colour = "#C1272D" if e.source == "InSAR" else "#08306B"
    ax_c.plot([rain.dowy[e.window_lo], rain.dowy[e.window_hi]], [y, y], "-", color=colour, lw=2.0, solid_capstyle="butt")
bottom = top - (len(ordered) - 1) * row_gap - 1.5
ax_c.text(4, top + 0.8, f"Onset windows ({ordered.window_lo.dt.year.min()} to {ordered.window_lo.dt.year.max()})", fontsize=6, color="0.3", va="bottom")
ax_c.set_ylim(bottom, 21)
ax_c.set_yticks([v for v in (-20, -10, 0, 10, 20) if v > lowest])     # pressure ticks only, none beside the window rows
ax_c.set_xlim(1, 366)
ax_c.set_xticks([1, 62, 124, 183, 244, 305])
ax_c.set_xticklabels(["Oct", "Dec", "Feb", "Apr", "Jun", "Aug"], fontsize=label_pt)
ax_c.tick_params(labelsize=label_pt)
ax_c.set_ylabel("Pore pressure minus\nwater-year mean (kPa)", fontsize=label_pt, labelpad=1)
ax_c.legend(fontsize=6, loc="upper left", handlelength=1.2, frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
ax_c.set_title(r"$\bf{C}$  Aquifer pressure through the year", loc="left", fontsize=8)
print(f"C: {len(ordered)} onset windows ({(ordered.source == 'InSAR').sum()} InSAR); y range {bottom:.0f} to {ax_c.get_ylim()[1]:.0f} kPa")

f_out = os.path.join(common_paths["fig_dir"], "qc10x_fig3_wells_variant.png")
fig.savefig(f_out, dpi=600)
fig.savefig(f_out.replace(".png", ".pdf"))
print("Saved", f_out)
