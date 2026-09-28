# Cut from scripts/06_event_field_geometry.py on 2026-09-28 (HEAD f8bdcea).
# Exploration, not part of the paper recipe (Dani, 2026-09-28). Runs only in the
# namespace of notebook 06 after the profiles cell (needs L, s_hat, m_asc, m_desc, tracks); kept verbatim for the record.

# %% [markdown]
# ## CHECK 6: what the ascending deficit needs
#
# Pure strike-slip puts the ascending near-fault step at `geom_ratio` times the
# descending one; the mean event's ratio is smaller in magnitude (the ascending
# step is short). Two LOS observations cannot separate three components, so
# we solve the two end members exactly: strike-slip plus VERTICAL (east side up
# positive) and strike-slip plus FAULT-NORMAL (east block moving away from the
# fault, i.e. opening, positive). Either a few mm of east-side-up or of
# east-side contraction closes the gap; the two-block forward model in the
# supplement (TwoBlock_LOS_test) shows the same thing panel by panel. This is
# the reason notebook 07 inverts the descending track alone.

# %%
u_hat = np.array([0.0, 0.0, 1.0])
n_hat = utils.horizontal_unit_vector(strike - 90.0)          # fault-normal, toward the east (NE) block
d_obs = np.array([m_asc, m_desc])                              # mean near-fault LOS steps (CHECK 2), mm
asym = {}
for comp, c_hat in (("up", u_hat), ("normal", n_hat)):
    G2 = np.array([[L[o] @ s_hat, L[o] @ c_hat] for o in tracks])
    fp_c, x_c = np.linalg.solve(G2, d_obs)
    asym[comp] = (float(fp_c), float(x_c))
fp_only = float(np.linalg.lstsq(np.array([[L[o] @ s_hat] for o in tracks]), d_obs, rcond=None)[0][0])
print(f"mean near-fault LOS steps: asc {m_asc:+.2f}, desc {m_desc:+.2f} mm (ratio {m_asc / m_desc:+.2f}, geometric {geom_ratio:+.2f})")
print(f"strike-slip only (least squares on both tracks): {fp_only:.2f} mm")
print(f"strike-slip + vertical:     FP {asym['up'][0]:.2f} mm, east side up {asym['up'][1]:+.2f} mm")
print(f"strike-slip + fault-normal: FP {asym['normal'][0]:.2f} mm, east block opening {asym['normal'][1]:+.2f} mm (negative = contraction)")
print(f"LOS sensitivities: up asc {L['ascending'] @ u_hat:+.3f} desc {L['descending'] @ u_hat:+.3f}; "
      f"normal asc {L['ascending'] @ n_hat:+.3f} desc {L['descending'] @ n_hat:+.3f}")

# --- its sidecar and registry lines (removed from 06's REGISTER cell)
# L_.append(f"ascending deficit (mean steps asc {m_asc:+.2f}, desc {m_desc:+.2f} mm): strike-slip only {fp_only:.2f} mm; "
#           f"with vertical FP {asym['up'][0]:.2f} mm + east side up {asym['up'][1]:+.2f} mm; "
#           f"with fault-normal FP {asym['normal'][0]:.2f} mm + east block opening {asym['normal'][1]:+.2f} mm")
# REG.register("06_asym_fp_only_mm", fp_only, "mm", 2, where="S2", cell="asymmetry", sidecar=sidecar)
# REG.register("06_asym_fp_with_up_mm", asym["up"][0], "mm", 2, where="S2", cell="asymmetry", sidecar=sidecar)
# REG.register("06_asym_east_up_mm", asym["up"][1], "mm", 2, where="S2", cell="asymmetry", sidecar=sidecar)
# REG.register("06_asym_fp_with_normal_mm", asym["normal"][0], "mm", 2, where="S2", cell="asymmetry", sidecar=sidecar)
# REG.register("06_asym_east_opening_mm", asym["normal"][1], "mm", 2, where="S2", cell="asymmetry", sidecar=sidecar)
