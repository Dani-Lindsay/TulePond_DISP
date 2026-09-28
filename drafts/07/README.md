# drafts/07 -- exploratory tests behind the 07 slip model

Every script here was run once to answer one question; the answers are in
`07_sweep_lessons_2026-09-28.md` (with source tags). The decisions they led
to are carried by `scripts/07_slip_model.py` (the homogeneous baseline, the
one to rerun) and `scripts/07_slip_model_tests.py` (the sensitivity suite).
Most drafts `exec` the machinery cells of `07_slip_model_tests.py` and write
to `outputs/07_slip_model_tests/`. Run from the repo root in
`opera_disp_feb13`, with PROJ_LIB/PROJ_DATA pointed at the env.

Kept as the record of how each choice was made; none is needed to rerun
the model on new data.

| Script | Question | Status |
|---|---|---|
| `seismicity_dip_sections.py` | Does the seismicity show a dipping fault at Tule Pond? | context for the vertical-fault choice |
| `trace_versions_check.py` | Which trace does the mesh follow? | done |
| `prealignment_check.py` | Far-field plane levelling, shown per track | adopted (levelling) |
| `surface_interp_test.py` | Surface constraint on every surface triangle vs CSI's nearest-triangle link | adopted (interp) |
| `fault_parallel_inversion_test.py` | Fault-parallel input vs LOS tracks | adopted (fault-parallel) |
| `residual_maps_los_vs_fp.py` | Residual maps, LOS vs fault-parallel | figure behind that decision |
| `los_we_residual_metric.py` | West/east residual metric, LOS | superseded by the residual maps |
| `sfcvm_extract.py` | USGS SFCVM v21.1 Vs/rigidity across the creep section | input to Mw with SFCVM rigidity and to the PyLith 1D model |
| `interim_results_2026-09-28.py` | One-figure summary of the morning's results | record |
| `top_tie_test.py` | Tie slip uniform 0-D (synthetic + real, 0.7 field) | adopted (0.2 km) |
| `tie_depth_0.9_test.py` | Tie depth on the 0.9 field | confirms 0.2 km |
| `tie_vs_crossings_test.py` | Tie alone vs crossings alone vs both | both kept |
| `fig2_model_row.py` | The sigma 24 draft model for Fig 2 row 3 | superseded by the notebook; its output is the notebook's CHECK reference (`fig2_model_row_draft_sigma24_reference.npz`) |
| `sigma_final_config.py` | Sigma 8-64, plateau rule | superseded by `sigma_lcurve.py` (no plateau exists) |
| `sigma_lcurve.py` | Sigma 4-128, L-curve terms (full chi2, prior-weighted roughness) | adopted: corner 10.1 -> sigma 12; now inside the notebook |
| `sigma_lcurve_figure.py` | What the sigma choice looks like (slip, surface, depth) | figure `test_sigma_lcurve.png` |
