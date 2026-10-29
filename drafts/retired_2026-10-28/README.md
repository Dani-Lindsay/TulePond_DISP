# Retired 2026-10-30: the pre-lock slip-model notebooks (rewire, code_audit_2026-10-28)

Moved here (git mv, history kept) when the slip-model notebooks were rewritten from scratch on the locked method
(`model_config.py`; `~/work/proj_TulePond_DISP/code_audit_2026-10-28/REWIRE_BRIEF.md`). Reference only: they encode the
16-17 Oct choices (order 1, default mesh, Vs-floor grid, N^0.5 weighting, 1 km noise, no cut, constant offset, sigma 12)
and several read PyLith runs that were moved to `backup_vsfloor150_2026-10-25/`, so they do not run any more.

| Notebook | Was | Replaced by |
|---|---|---|
| 07a_uniform_model | data side + CSI smooth12 homogeneous model (surface constraint ON) | new 07a (data side + analytic homogeneous model on the PyLith tents) |
| 07b_uniform_model_tests | uniform-model test families (retired 2026-10-05) | - |
| 07c_fig2 | Fig 2 from the 17 Oct model | new 07c (plane-aware, 60-150 m panel K) |
| 08a_3d_model | 3D model, order 1, PyLith forward runs inside | new 08a (archived G of gf_shallow_sfcvm_order2, no PyLith) |
| 08b_3d_model_tests | recovery, depth scan, dip scan (pre-lock) | 08b_sensitivity (plots only) + scripts/sensitivity/ |
| 08c_3d_model_sensitivity | 32-row order-1 RUN_TABLE | scripts/sensitivity/sensitivity_rows.py |

`scripts/model_plots/plot_test_summary.py` and `plot_tests_figures.py` served only 08b/08c.
