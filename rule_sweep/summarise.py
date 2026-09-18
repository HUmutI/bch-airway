#!/usr/bin/env python
"""Rank the swept rules by how many of the six reported metrics they lead.

The paper reports DSC, clDice, TLD, BND, sensitivity and precision. A rule is
only worth adopting if it leads several of them at once, so the ranking here is
the count of columns a rule tops, with the mean of the six as the tie-break.
Solo members are listed for context but excluded from the leader count, since
the comparison the paper makes is between combination rules.
"""
import csv, sys, collections
import statistics as st

METRICS = ["DSC", "clDice", "TLD", "BND", "Sensitivity", "Precision"]

rows = list(csv.DictReader(open(sys.argv[1])))
by = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    for m in METRICS + ["FPR"]:
        by[r["model"]][m].append(float(r[m]))

models = list(by)
mean = {mo: {m: st.mean(by[mo][m]) for m in METRICS + ["FPR"]} for mo in models}
combos = [mo for mo in models if not mo.startswith("solo_")]
best = {m: max(mean[mo][m] for mo in combos) for m in METRICS}

def leads(mo):
    return sum(1 for m in METRICS if mean[mo][m] >= best[m] - 1e-9)

order = sorted(combos, key=lambda mo: (-leads(mo), -st.mean([mean[mo][m] for m in METRICS])))
n = len(set(r["case"] for r in rows))
print(f"{n} validation cases, {len(models)} rules\n")
head = f"{'rule':<26}" + "".join(f"{m:>12}" for m in METRICS) + f"{'FPR':>9}{'leads':>7}"
print(head); print("-" * len(head))
for mo in order + [m for m in models if m.startswith("solo_")]:
    star = "".join("*" if not mo.startswith("solo_") and mean[mo][m] >= best[m] - 1e-9
                   else " " for m in METRICS)
    cells = "".join(f"{mean[mo][m]:>11.2f}{'*' if not mo.startswith('solo_') and mean[mo][m] >= best[m]-1e-9 else ' '}"
                    for m in METRICS)
    lead = leads(mo) if not mo.startswith("solo_") else 0
    print(f"{mo:<26}{cells}{mean[mo]['FPR']:>9.3f}{lead:>7}")
