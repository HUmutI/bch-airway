#!/usr/bin/env python
"""Emit both benchmark tables, best in bold and second best underlined.

Run it after hd_test.csv and hd_ped.csv exist. It prints the two tabular bodies
to paste into the paper, and the rank our strategy holds in every column, so a
claim in the text can be checked against the table it refers to.
"""
import csv, collections, sys
import statistics as st

ORDER = ["AirMorph", "MEDPSeg", "TfeNet",
         "AirMorph u MEDPSeg", "AirMorph u TfeNet", "MEDPSeg u TfeNet",
         "All three members", "Majority vote", "Majority vote, selected"]
TEX = {"AirMorph u MEDPSeg": r"AirMorph $\cup$ MEDPSeg",
       "AirMorph u TfeNet": r"AirMorph $\cup$ TfeNet",
       "MEDPSeg u TfeNet": r"MEDPSeg $\cup$ TfeNet"}
OURS = "All three members"
RULE_AFTER = {"TfeNet", OURS}
LOWER_BETTER = {"HD95"}


def means(path, key, keep=None, rename=None):
    rows = [r for r in csv.DictReader(open(path)) if keep is None or keep(r)]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        name = (rename or {}).get(r[key], r[key])
        for m, x in r.items():
            if m in (key, "case", "lcc_rule"):
                continue
            try:
                by[name][m].append(float(x))
            except (TypeError, ValueError):
                pass
    return {s: {m: st.mean(v) for m, v in d.items()} for s, d in by.items()}


def emit(values, cols, dec):
    """LaTeX rows. Marks are decided on the printed value, not the full float."""
    marks = {}
    for m in cols:
        vals = sorted({round(values[s][m], dec[m]) for s in ORDER},
                      reverse=m not in LOWER_BETTER)
        marks[m] = (vals[0], vals[1] if len(vals) > 1 else None)
    out = []
    for s in ORDER:
        cells = []
        for m in cols:
            v = round(values[s][m], dec[m])
            t = f"{v:.{dec[m]}f}"
            if v == marks[m][0]:
                t = r"\textbf{" + t + "}"
            elif marks[m][1] is not None and v == marks[m][1]:
                t = r"\underline{" + t + "}"
            cells.append(t)
        out.append(f"{TEX.get(s, s)} & " + " & ".join(cells) + r" \\")
        if s in RULE_AFTER:
            out.append(r"\midrule")
    return "\n".join(out)


def ranks(values, cols):
    for m in cols:
        r = sorted(ORDER, key=lambda s: values[s][m],
                   reverse=m not in LOWER_BETTER)
        print(f"    {m:<12} ours {values[OURS][m]:7.2f}  rank {r.index(OURS)+1}/9")


B = "/path/to/airway-data/paper"
COLS = ["DSC", "clDice", "TLD", "BND", "Sensitivity", "Precision", "HD95"]
DEC = {c: 2 for c in COLS}

adult = means(f"{B}/recompute_2026-09-16/recomputed_atm26_test.csv", "model",
              keep=lambda r: r["lcc_rule"] in ("none", "seeded"),
              rename={"airmorph_raw": "AirMorph", "medpseg_raw": "MEDPSeg",
                      "tfenet_raw": "TfeNet",
                      "or_airmorph_medpseg_lcc": "AirMorph u MEDPSeg",
                      "or_airmorph_tfenet_lcc": "AirMorph u TfeNet",
                      "or_medpseg_tfenet_lcc": "MEDPSeg u TfeNet",
                      "or_all3_lcc": OURS, "vote2of3": "Majority vote",
                      "vote2of3_lcc": "Majority vote, selected"})
hd_a = means(f"{B}/pediatric_benchmark/hd_test.csv", "strategy")
for s in ORDER:
    adult[s]["HD95"] = hd_a[s]["HD95"]

ped = means(f"{B}/pediatric_benchmark/pediatric_benchmark_9.csv", "strategy")
hd_p = means(f"{B}/pediatric_benchmark/hd_ped.csv", "strategy")
for s in ORDER:
    ped[s]["HD95"] = hd_p[s]["HD95"]

for label, v in (("ADULT, ATM26 test, 30 volumes", adult),
                 ("PEDIATRIC, 9 annotated volumes", ped)):
    print(f"% ===== {label} =====")
    print(emit(v, COLS, DEC))
    print(f"\n  ranks for {OURS}:")
    ranks(v, COLS)
    print()
