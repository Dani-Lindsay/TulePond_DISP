# Data S1: the data behind the figures and tables

Science asks for one multi-sheet Excel file with a sheet per figure panel or table and the P values. Two steps, both
read-only on the data root (`TULEPOND_DATA_DIR`) and writing only into `OUT_DIR` (refused inside a `FROZEN_*` or
`SUBMITTED_*` folder or the data root):

```
TULEPOND_DATA_DIR=<data root> python scripts/data_S1/build_data_S1.py OUT_DIR      # OUT_DIR/Data_S1/*.csv + README.csv
TULEPOND_DATA_DIR=<data root> python scripts/data_S1/make_workbook.py OUT_DIR <manuscript .tex>   # OUT_DIR/Data_S1.xlsx
```

- `build_data_S1.py`: one csv per figure panel from the saved products of a full run (`outputs/`, `GPS/`,
  `sensitivity/`, `external_inputs/`). Values the notebooks compute but do not save (fig. S6 statistic and Oakland
  stack, fig. S5 per-creepmeter stacks, fig. S3 interpolation flags, Fig. 3B 7-day rain) are recomputed with notebook
  04's own cells (found by content, output redirected to `OUT_DIR/_sandbox/`, no registry writes). Fig 1B's
  earthquakes are notebook 14's (`outputs/14_paper_figures/fig1_earthquakes_ncedc_dd.csv`, NCEDC DD catalogue);
  Fig 1D's creepmeter bars the larger side of notebook 05's bootstrap 95 % CI (`insar_utils.creepmeter_bar_half_widths`).
  `--fig1-earthquakes CSV` replaces notebook 14's file, only for a data root whose 14 predates the DD catalogue
  (the frozen v1.0.0 release).
- `make_workbook.py`: README sheet first (with the deposit file of every gridded map, which are not tabulated), one sheet
  per csv, tables S1-S5 parsed from the manuscript .tex (values as printed), and `P_values` (the 16 storm rules and the
  best rule's search-paid P from the registry, `12_rain_sweep_p_search_paid`). Needs openpyxl (environment.yml).

Gridded maps (Figs 1A, 2A, 2B, 2E, 2F, figs S1, S2C, S5A, S7, S11A) are not tabulated: the README sheet gives their file
in the data deposit.

**Check (25 Nov 2026).** Rebuilt from the frozen v1.0.0 release (`--fig1-earthquakes` = notebook 14's DD selection on
the release inputs) and the submitted .tex: the 60 sheets equal the `Data_S1.xlsx` uploaded with the submission on
24 Nov 2026 (every value to 1e-12, every text cell), except 10 provenance cells of the README sheet, which now name
the repository sources (`04/fig1_inputs.npz`, `14/fig1_earthquakes_ncedc_dd.csv`) instead of the 14 Nov Fig 1 script.

## Which saved file feeds each panel (fig. numbers of the submission, detector figure = fig. S6)

| Panel | Source (outputs/ of the run unless stated) |
|---|---|
| Fig 1A, 1D creepmeters | 04 fig1_inputs.npz cm_*; bars from 05 station_agreement.csv |
| Fig 1B | 04 fig1_inputs.npz hovmoller, hov_km, t_regular, cre_*; 14 fig1_earthquakes_ncedc_dd.csv |
| Fig 1C, 1D, 1E | 04 fig1_inputs.npz cum_prof(_raw), km_window, rate_*, aa_*, dates, stack_*, model_full |
| Fig 2C, D, G, H | 06 fig2_inputs.npz prof_* |
| Fig 2I, J, K, L | 09 selected_model.npz, 08 homogeneous_model.npz |
| Fig 3A | external_inputs/AA_BayArea_v3.0 HRKT.csv; 11 aa_insar_stacks.csv; 12 fig3_onset_windows.csv |
| Fig 3B | 12 tulepond_daily_precip.csv (7-day sum computed) |
| Fig 3C, D | 13 fig3_inputs.npz interval_*, grid, bpt_cdf |
| fig S2 | GPS/gps_insar_validation.csv |
| fig S3, S4 | 03 acquisition_times.csv, fp_ts.csv, creep_ts_LOS_*.csv, creep_geom_*.csv |
| fig S5 | 02 fp_velocity_xfault_swath200m_dist1000m.csv; 05 station_agreement.csv; 03 fp_ts.csv (stacks recomputed) |
| fig S6 | 04 creep_events.csv, creepmeter_events_COZ1.csv; stacks and statistic recomputed (04 cells) |
| fig S8 | 07 semivariogram_bins_5km.csv, noise_model_5km.csv |
| figs S9, S10 | sensitivity/ dip_rows, sensitivity_rows, depth_scan(_noise), basement_depth |
| fig S12 | 12 wateryear_totals.csv, precip_conditioning_thresholds.csv, rain_total_sweep.csv |
| Tables S1-S5 | the manuscript .tex (values as printed) |
