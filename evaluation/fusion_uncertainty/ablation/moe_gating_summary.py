"""Summaries for the moe_gating ablation: candidate table on val with the
selection rule (max DSC s.t. BND loss <= 1 vs union), paired differences on test."""
import os, sys
import numpy as np, pandas as pd

OUT = "/fileserver/external/quin/ege/fusion_uncertainty_run/ablation/moe_gating"
REF = "/fileserver/external/quin/ege/bch-airway/evaluation/fusion_uncertainty/results"
COLS = ["DSC", "clDice", "TLD", "BND", "Sensitivity", "Precision"]


def ref(split):
    return pd.read_csv(os.path.join(REF, f"whole_{split}.csv"))


def candidates(tags, split="val"):
    df = pd.concat([pd.read_csv(os.path.join(OUT, f"percase_{t}_{split}.csv")) for t in tags])
    r = ref(split); union = r[r.strategy == "union_seeded"].set_index("case")
    g = df.groupby("config")[["DSC", "TLD", "BND", "Sensitivity", "Precision", "sec_desc", "sec_apply"]].mean()
    g["BND_loss_vs_union"] = union["BND"].mean() - g["BND"]
    g["dDSC_vs_union"] = g["DSC"] - union["DSC"].mean()
    g["feasible"] = g["BND_loss_vs_union"] <= 1.0
    g = g.sort_values("DSC", ascending=False)
    return g


def paired(split, tag, config):
    df = pd.read_csv(os.path.join(OUT, f"percase_{tag}_{split}.csv"))
    df = df[df.config == config].set_index("case")
    r = ref(split)
    out = {}
    for strat in ("union_seeded", "topofuse", "vote2of3_seeded"):
        b = r[r.strategy == strat].set_index("case").loc[df.index]
        d = {}
        for col in COLS:
            diff = df[col] - b[col]
            d[col] = (diff.mean(), int((diff > 0).sum()), int((diff < 0).sum()))
        out[strat] = d
    return df, out


if __name__ == "__main__":
    print(candidates(sys.argv[1:] or ["sweep1"]).round(2).to_string())


def report_tables(config, tags):
    """Markdown tables for RESULTS.md: means per split + paired diffs on test."""
    lines = []
    rows = []
    for split, tag in tags.items():
        f = os.path.join(OUT, f"percase_{tag}_{split}.csv")
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f); df = df[df.config == config]
        r = ref(split)
        for strat, lab in (("union_seeded", "union, seeded"), ("topofuse", "union, refined")):
            m = r[r.strategy == strat][COLS].mean()
            rows.append([split, lab] + [m[c] for c in COLS])
        m = df[COLS].mean()
        rows.append([split, f"moe_gating ({config})"] + [m[c] for c in COLS])
    lines.append("| split | method | DSC | clDice | TLD | BND | Sens | Prec |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r_ in rows:
        lines.append("| " + " | ".join([r_[0], r_[1]] + [f"{v:.2f}" for v in r_[2:]]) + " |")
    out = "\n".join(lines)
    if os.path.exists(os.path.join(OUT, f"percase_{tags['test']}_test.csv")):
        df, pr = paired("test", tags["test"], config)
        out += "\n\nPaired per-case differences on test (moe_gating minus reference; mean, #cases better / worse):\n\n"
        out += "| vs | " + " | ".join(COLS) + " |\n|---|" + "---|" * len(COLS) + "\n"
        for strat, lab in (("union_seeded", "union, seeded"), ("topofuse", "union, refined"), ("vote2of3_seeded", "majority, seeded")):
            out += f"| {lab} | " + " | ".join(f"{m:+.2f} ({b}/{w})" for m, b, w in [pr[strat][c] for c in COLS]) + " |\n"
    return out


def percase_csv(config, tags):
    dfs = []
    for split, tag in tags.items():
        f = os.path.join(OUT, f"percase_{tag}_{split}.csv")
        if os.path.exists(f):
            d = pd.read_csv(f); dfs.append(d[d.config == config])
    df = pd.concat(dfs)
    r = pd.concat([ref(s) for s in tags])
    r = r[r.strategy.isin(["union_seeded", "topofuse"])].rename(columns={"strategy": "config"})
    allr = pd.concat([df, r], ignore_index=True)
    allr.to_csv(os.path.join(OUT, "percase_final.csv"), index=False)
    return allr
