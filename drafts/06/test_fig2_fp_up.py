# Cut from scripts/06_event_field_geometry.py on 2026-09-28 (HEAD f8bdcea, lines 1150-1172).
# Exploration, not part of the paper recipe. Runs only in the namespace of
# notebook 06 after the cell named below; kept verbatim for the record.
# Paste after the Fig 2 cell (needs draw_fig2, zeroed_profiles, block_amplitudes, common_ok).

# %% [markdown]
# ## Test: the same figure with the fault-parallel and vertical fields
#
# Not a paper figure (Dani, 2026-09-23: "just a test"). The two columns are the
# decomposed fault-parallel (positive right-lateral: the east block moving
# south-east) and vertical (positive up) fields from `asc_desc2horz_vert`,
# in place of the two tracks; same masks, profiles and block amplitudes.

# %%
fpup_rate = {"fault-parallel": fp_rate, "vertical": up_rate}
fpup_evt = {"fault-parallel": fp_evt, "vertical": up_evt}
ok_fpup = common_ok & np.all([np.isfinite(f[c]) for f in (fpup_rate, fpup_evt) for c in fpup_rate], axis=0)
fpup_rate = {c: np.where(ok_fpup, m_, np.nan) for c, m_ in fpup_rate.items()}
fpup_evt = {c: np.where(ok_fpup, m_, np.nan) for c, m_ in fpup_evt.items()}
fpup_profiles = {c: {n_: zeroed_profiles(fpup_rate[c], fpup_evt[c], ll) for n_, ll in profile_pts.items()} for c in fpup_rate}
for c in fpup_profiles:
    for n_, p in fpup_profiles[c].items():
        p["blocks"] = block_amplitudes(p["de"], p["ve"])
        print(f"[{c}] {n_}-{n_}': rate step {p['rate_step']:+.2f} mm/yr, event step {p['event_step']:+.2f} mm; "
              f"per block W {p['blocks']['W']['amp']:+.2f}, E {p['blocks']['E']['amp']:+.2f} mm")
draw_fig2(fpup_evt, fpup_profiles, f"Mean event ({n_mean} events)", os.path.join(fig_dir, f"test_fig2_fp_up_coh{coh_maps}_{n_mean}ev"),
          rate_maps_in=fpup_rate, cols=list(fpup_rate), ylabs=("Rate (mm/yr)", "Displacement (mm)"), lims={"rate": 10.0, "event": 10.0})

