#!/usr/bin/env python3
"""The Spring and Fall 2025 well tables of the ACWD 2025 Groundwater Monitoring Report as csv.

Source: Alameda County Water District, 2025 Groundwater Monitoring Report (E. Chen, 6 Feb 2026),
https://acwd.org/DocumentCenter/View/7973, Appendix E (Spring 2025, pdf pages 80-86) and Appendix F (Fall 2025,
pdf pages 88-94). The tables are fixed-width; this reads the text that `pdftotext -layout` writes for those pages:

    pdftotext -layout -f 80 -l 86 "2025 GWMR.pdf" appendixE_spring2025.txt
    pdftotext -layout -f 88 -l 94 "2025 GWMR.pdf" appendixF_fall2025.txt
    python acwd_gwmr_tables.py appendixE_spring2025.txt spring_2025.csv
    python acwd_gwmr_tables.py appendixF_fall2025.txt fall_2025.csv

Columns: well (State Well Number), alt_id, aquifer (ACWD code: AHF above-fault, N Newark, C Centerville, F Fremont,
D Deep; combinations are multi-aquifer wells), owner, date, reference_elevation_ft, water_elevation_ft (NGVD 1929;
blank where no level was measured, "artesian" where the well was flowing). Every row of the table is kept; the water-level rows are the ones with a value.
"""
import csv
import re
import sys

# The table is fixed-width but its columns shift by a few characters between pages, so each row is read as tokens:
# the State Well Number, then (optionally) an alternate id, the aquifer code, the owner, the measurement date, the
# reference-point elevation and the water elevation ("--" or blank when no level was measured).
FIELDS = ["well", "alt_id", "aquifer", "owner", "date", "reference_elevation_ft", "water_elevation_ft"]
AQUIFERS = {"AHF", "N", "C", "F", "D", "CF", "FD", "CFD", "CD", "NC", "ND", "NCF", "NF", "DF"}   # ACWD codes (Appendix G)
WELL = re.compile(r"^\dS/\dW-\d{2}[A-Z]\d{3}$")
DATE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")
NUM = re.compile(r"^-?\d+(\.\d+)?$")


def parse(text_path):
    rows = []
    for line in open(text_path, encoding="utf-8"):
        toks = line.split()
        if not toks or not WELL.match(toks[0]):
            continue
        dates = [i for i, t in enumerate(toks) if DATE.match(t)]
        dashes = [i for i, t in enumerate(toks) if t == "--"]
        assert dates or dashes, f"no date and no '--' in: {line!r}"
        i = dates[0] if dates else dashes[0]              # a well with no measurement has "--" where the date would be
        codes = [j for j, t in enumerate(toks[1:i], start=1) if t in AQUIFERS]
        assert codes, f"no aquifer code in: {line!r}"
        j = codes[-1]                                    # the last code before the date (an alternate id like "Well N" comes first)
        assert j + 1 < i, f"no owner between the aquifer code and the date in: {line!r}"
        if dates and i + 1 < len(toks):
            ref = toks[i + 1]
            assert NUM.match(ref), f"reference elevation not a number in: {line!r}"
            water = toks[i + 2] if i + 2 < len(toks) else "--"
            if not NUM.match(water):
                # "--": not measured; "A": artesian (flowing at the surface, no level); a date: the column is empty
                assert water in ("--", "A") or DATE.match(water), f"water elevation not a number, '--' or 'A' in: {line!r}"
                water = "artesian" if water == "A" else ""
            date = toks[i]
        else:                                            # no measurement, or a date with nothing after it
            date, ref, water = (toks[i] if dates else ""), "", ""
        rows.append({"well": toks[0], "alt_id": " ".join(toks[1:j]), "aquifer": toks[j], "owner": " ".join(toks[j + 1:i]),
                     "date": date, "reference_elevation_ft": ref, "water_elevation_ft": water})
    assert rows, f"no well rows found in {text_path}"
    return rows


def main(text_path, csv_path):
    rows = parse(text_path)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    n_level = sum(1 for r in rows if r["water_elevation_ft"])
    print(f"{csv_path}: {len(rows)} wells, {n_level} with a water level")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
