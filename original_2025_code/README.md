# original_2025_code: Danielle Lindsay's pre-repository code (February 2025)

Kept verbatim as an independent statement of the cross-fault method, and run by `15_check_original_2025_code.ipynb`
on the repository's inputs. It is the code behind the first version of this result:

> Lindsay, D., Taira, T., & Bürgmann, R. (2025). Seasonal to Multiannual Creep Rate Changes Along the Hayward,
> Rodgers Creek, and Maacama Faults. Poster 079, 2025 SCEC Annual Meeting, SCEC Contribution 14951.
> https://central.scec.org/publication/14951

The poster PDF (`2025_SCSC_Sep3_Final.pdf`, 17 MB) is in the Zenodo data record of this paper, not in the repository.
The code was written on a drive that later failed; this script, one output table and one figure were recovered.

| File | What it is |
|---|---|
| `HW1_creeprate_2025-02-14.py` | The script, unchanged (805 lines, dated 14 Feb 2025 on disk). Reads a MintPy velocity and time series, splits the pixels within 0.1 to 0.3 km of each 100 m trace crossing (`pygmt.project`, half-width 0.2 km, at least 10 pixels per side) into west and east, takes the median of each side at every epoch and differences them (east minus west). Then projects each side to 'fault parallel' with `los2fault_projection`. |
| `HW3_west_east_fault_parallel_ts_2025-02.csv` | Its output for the HW3 frame crop (37.52 to 37.69 N, descending track 11115, OPERA DISP-S1 as downloaded in early 2025): the per-crossing series, 300 epochs, 2016.61 to 2024.59. The pre-AI Tule Pond result. |
| `Hayward_0.1km_xfault_2025.csv` | The 2025 trace crossings (`ft_lon, ft_lat, profile_az`; `profile_az` is the cross-fault azimuth toward the east side). |
| `COZ1_creepmeter_vs_insar_2025-02.png` | The February 2025 creepmeter validation at COZ1 (`read_plot_creepmeter_COZ1.py`, not recovered in full). |

Two things in the 2025 script that the repository does differently, found while writing notebook 15:

1. `read_insar_data` never loaded the geometry: `east = 0.65`, `north = -0.11` are placeholders ("fix when I know how
   to load east and north"), so one approximate descending look vector was used for every pixel.
2. `los2fault_projection` converts its `aspect` argument to anticlockwise-from-east and then uses a formula that
   expects anticlockwise-from-north, and the Hayward loop passed it `profile_az + 90` (the strike). The projection
   vector therefore pointed along bearing 54, across the fault, not along it.

Together these make the 2025 'fault-parallel' values a constant factor off (notebook 15 computes it; a few per cent on
this descending track). The east-minus-west LOS difference, which is what carries the events, is unaffected, and it is
that series notebook 15 compares with notebook 03.
