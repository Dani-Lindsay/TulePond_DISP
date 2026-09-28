# Cut from scripts/06_event_field_geometry.py on 2026-09-28 (HEAD f8bdcea).
# Exploration, not part of the paper recipe (Dani, 2026-09-28). Runs only in the
# namespace of notebook 06 after the Fig 2 cell (needs map_base, colour_bar, to_grid, running_median, profiles); kept verbatim for the record.

# Dropped as a supplement figure: the fault-parallel maps and profiles become Fig 2
# (Dani, 2026-09-28); this layout is the starting point for it.

# %% [markdown]
# ## Supplement: the fault-parallel maps and profiles
#
# The decomposed field (MintPy `asc_desc2horz_vert`, both tracks) that Fig 2
# showed until 2026-09-23: X-X' and Y-Y' through the inter-event rate (top pair)
# and the mean event (bottom pair), grey pixels, blue per-side running
# median, and the two fault-parallel maps. The numbers quoted in the text
# (the near-fault steps) come from these profiles.

# %%
Wp_s, gap_x = 4.4, 1.2                # the supplement keeps its own panel width
names = list(profiles)

fig = pygmt.Figure()                  # the helpers draw into the module-level figure
pygmt.config(FONT="8p,Helvetica", FONT_ANNOT_PRIMARY="7p,Helvetica", FONT_LABEL="7p,Helvetica", MAP_FRAME_TYPE="plain",
             MAP_FRAME_PEN="0.6p", MAP_TICK_LENGTH_PRIMARY="2p", MAP_LABEL_OFFSET="4p", MAP_ANNOT_OFFSET="3p", FORMAT_GEO_MAP="ddd.xx")


def profile_pair(ylab, lim, label, vkey, dkey):
    """Two stacked panels (X above Y) for one field, drawn from the current origin
    (the top panel's bottom-left). Leaves the origin at the LOWER panel."""
    for j, n_ in enumerate(names):
        p = profiles[n_]
        d, v = p[dkey], p[vkey]                                  # zeroed above: the rate at the fault, the event on its far field
        if j > 0:
            fig.shift_origin(yshift=f"-{Hp + gap_pair}c")
        xfr = "xa1000f500+lCross-fault distance (m)" if j == 1 else "xa1000f500"
        fig.basemap(region=[-cross_m, cross_m, -lim, lim], projection=f"X{Wp_s}c/{Hp}c", frame=["WSrt" if j == 1 else "Wsrt", xfr, f"ya{lim / 2:g}"])
        fig.plot(x=d, y=v, style="c0.03c", fill=grey, transparency=30)      # the data
        for xm, vm in running_median(d, v):                                  # the per-side running median
            fig.plot(x=xm, y=vm, pen="1.0p," + median_col)
        fig.plot(x=[0, 0], y=[-lim, lim], pen="0.5p,gray50")
        fig.text(position="TL", text=n_, font="9p,Helvetica-Bold", offset="0.1c/-0.08c")
        fig.text(position="TR", text=n_ + "'", font="9p,Helvetica-Bold", offset="-0.1c/-0.08c")
        if j == 0:
            fig.text(x=-cross_m, y=lim, text=label, justify="BL", offset="0c/0.12c", font="9p,Helvetica", no_clip=True)
    # one y label for the pair, centred on the pair (drawn from the lower panel)
    fig.text(position="ML", text=ylab, angle=90, justify="BC", offset=f"-0.95c/{0.5 * (Hp + gap_pair)}c", font="9p,Helvetica", no_clip=True)


def map_panel(grid, cpt, lim, label, cb_label):
    """A map with its own colour bar."""
    map_base(grid, cpt, label)
    colour_bar(cpt, lim, cb_label, Wm + 0.25)


g_rate, lim_rate = to_grid(fp_rate, True)
g_evt, lim_evt = to_grid(fp_evt, False)
cpt_rate = utils.write_cpt(os.path.join(cpt_dir, "rate.cpt"), "RdBu_r", -lim_rate, lim_rate)
cpt_evt = utils.write_cpt(os.path.join(cpt_dir, "event.cpt"), "RdBu_r", -lim_evt, lim_evt)
# top row: A the inter-event profiles, B the inter-event map (the map is Hpair tall, so its bottom is the pair's bottom)
profile_pair("Fault parallel (mm/yr)", 10.0, "@%Helvetica-Bold%A@%% @:7p:Inter-event rate@::", "v", "d")
fig.shift_origin(xshift=f"{Wp_s + gap_x}c")
map_panel(g_rate, cpt_rate, lim_rate, "@%Helvetica-Bold%B@%% @:7p:Inter-event rate (FP)@::", "mm/yr")
# bottom row: C the mean-event profiles, D the mean-event map
fig.shift_origin(xshift=f"-{Wp_s + gap_x}c", yshift=f"-{gap_row + Hp}c")
profile_pair("Fault parallel (mm)", 10.0, f"@%Helvetica-Bold%C@%% @:7p:Mean event ({n_mean} events)@::", "ve", "de")
fig.shift_origin(xshift=f"{Wp_s + gap_x}c")
map_panel(g_evt, cpt_evt, lim_evt, "@%Helvetica-Bold%D@%% @:7p:Mean event (FP)@::", "mm")

f_figs = os.path.join(fig_dir, f"figS_fp_maps_coh{coh_maps}_{n_mean}ev")
fig.savefig(f_figs + ".png", dpi=300, transparent=False, crop="+gwhite", anti_alias=True, show=False)
fig.savefig(f_figs + ".pdf", transparent=False, crop="+gwhite", anti_alias=True, show=False)
print("Saved the fault-parallel maps and profiles (supplement) to:", f_figs + ".png")
fig.show()

