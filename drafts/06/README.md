# drafts/06: material cut from notebook 06

Cut on 2026-09-28 so that `06_event_field_geometry` reads as the recipe for the
paper's event field. Each file is copied verbatim from HEAD `f8bdcea` and says
where in 06 it ran. Only `ascending_deficit.py` registered numbers (`06_asym_*`).

| file | what it was | why it left |
|---|---|---|
| `strike143_reference_check.py` | The SSE-section strike change reproduces the MintPy-mode `fp_maps.npz` at the old whole-box strike | One-off proof for the 2026-09-28 strike change and saved-products mode. |
| `test_fig2_fp_up.py` | Fig 2 redrawn with fault-parallel / vertical columns | "Just a test" (Dani, 2026-09-23); not a paper figure. |
| `ascending_deficit.py` | CHECK 6: strike-slip + vertical / fault-normal end members that close the ascending deficit, with its `06_asym_*` registry lines (commented) | Exploration (Dani, 2026-09-28). `methods_qc_2026-09-23.md` quotes its numbers for S2. |
| `fp_supplement_figure.py` | The fault-parallel maps and profiles supplement figure | Dropped: the fault-parallel field becomes Fig 2 (Dani, 2026-09-28); kept as the starting layout. |

Dropped outright (git history has them, commit before this one): CHECK 3 (the
symmetric / antisymmetric split, `symmetry_table.csv`, `figS_event_symmetry`,
`06_sym_*` keys) and CHECKs 4 and 5 (legacy parity). The paper has no legacy
data to check against.
