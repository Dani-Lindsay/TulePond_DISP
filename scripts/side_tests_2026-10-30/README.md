# Side tests of 30 Oct 2026 (copies for backup; the outputs live in the data root)

Each script was run from its data-root folder (~/work/proj_TulePond_DISP/<folder>/), which also holds its README, csv
and figures. These are COPIES so the code is on GitHub; edit and run the data-root versions.

| Script | Data-root folder | What |
|---|---|---|
| dipslip_test.py | dipslip_2026-10-30 | Roland's dip-slip test, homogeneous half-space |
| dip_deflection.py | dip_deflection_2026-10-30 | M7 deflection of the dip-sweep models |
| fig2_homog60w.py | dip_deflection_2026-10-30 | Fig 2 with the homogeneous 60 W model (comparison only) |
| product_3d_dip70w.py | dip_deflection_2026-10-30 | 3D fine-mesh 70 W / 70 E products (DIP env var) |
| product_homog_dip70w.py | dip_deflection_2026-10-30 | homogeneous 70 W product |
| alongstrike_dip_preference.py | dip_deflection_2026-10-30 | near-field RMS by km for vertical / 70 W / 70 E |
| sfsvm_boundary_dip.py | dip_deflection_2026-10-30 | dip of the SFSVM Hayward boundary beside the section |
| stress_drop.py | stress_drop_2026-10-30 | stress drop, Knopoff / Kanamori & Anderson (locked) |
| model_comparison.py | fig_redraws_2026-10-30 | SM model-comparison figure (FOUR_ROWS=1) |
| redraw_model_slip.py | fig_redraws_2026-10-30 | slip figure redraw without re-running 08a |
| make_table.py | sensitivity_table_2026-10-30 | LaTeX sensitivity table |
