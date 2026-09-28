# Cut from scripts/06_event_field_geometry.py on 2026-09-28 (HEAD f8bdcea, lines 654-670).
# Exploration, not part of the paper recipe. Runs only in the namespace of
# notebook 06 after the cell named below; kept verbatim for the record.
# Paste inside 'RUN: fault-parallel and vertical maps', after decompose() is defined.

# CHECK (2026-09-28): at the whole-box strike this path must reproduce the
# fp_maps.npz written in MintPy mode (kept as fp_maps_strike143_reference.npz)
# exactly; everything after it differs from that file by the strike alone
f_ref = os.path.join(out_dir, "fp_maps_strike143_reference.npz")
if os.path.exists(f_ref):
    ref = np.load(f_ref)
    assert abs(float(ref["strike"]) - strike_box) < 1e-9, "reference was made at another strike"
    chk_fp, chk_up = decompose(mean_event["ascending"]["mean"], mean_event["descending"]["mean"], strike_box)
    chk_fpr, chk_upr = decompose(event_maps["ascending"]["rate"], event_maps["descending"]["rate"], strike_box)
    for name_, new_, old_ in (("fp_event", chk_fp, ref["fp_event"]), ("up_event", chk_up, ref["up_event"]),
                              ("fp_rate", chk_fpr, ref["fp_rate"]), ("up_rate", chk_upr, ref["up_rate"])):
        same_nan = np.array_equal(np.isnan(new_), np.isnan(old_))
        diff = float(np.nanmax(np.abs(new_ - old_)))
        print(f"{name_} at the box strike ({strike_box:.2f} deg): NaN pattern identical {same_nan}, max |difference| {diff:.2e}")
        assert same_nan and diff == 0.0, f"{name_} differs from the MintPy-mode product"
    print("PASS: this path reproduces the MintPy-mode decomposition exactly; now the SSE-section strike")

