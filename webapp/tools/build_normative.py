#!/usr/bin/env python3
"""Build web/normative.json: age-banded percentile grids of the airway
measurements the site reports, from the 90-scan normative cohort.

The cohort tables live on izmir (FINAL_AIRWAY_90_RESULTS_2026-09-16, see
cohort90/README.md). Only aggregates leave this script: for every metric and
every two-year age band, the percentiles of the scans in that band and its
two neighbours (about 30 scans), never a per-scan value. The browser turns a
measurement into a percentile by interpolating on that grid (web/app.js).

    AIRWAY_ROOT=/fileserver/external/quin/ege python tools/build_normative.py
"""
import csv, json, os, sys
from collections import defaultdict
import numpy as np

ROOT = os.environ.get("AIRWAY_ROOT", "/fileserver/external/quin/ege")
R90 = os.path.join(ROOT, "FINAL_AIRWAY_90_RESULTS_2026-09-16")
CSV = os.path.join(R90, "AIRWAY_RESULTS_90", "csv")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "normative.json")

BANDS = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14), (14, 16), (16, 18)]
GRID = [2.5, 5, 10, 25, 50, 75, 90, 95, 97.5]
MAX_GEN = 10


def read(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def band_of(age):
    for lo, hi in BANDS:
        if lo <= age < hi or (hi == 18 and age == 18):
            return (lo, hi)
    return None


def main():
    cases = {r["case_id"]: r for r in read(os.path.join(CSV, "case_level_measurements.csv"))}
    vol = {r["case_id"]: r for r in read(os.path.join(R90, "data", "airway_fractal_volume_case_level_90.csv"))}
    fd = {r["case_id"]: r for r in read(os.path.join(R90, "data", "airway_fractal_dimension_corrected_90.csv"))}
    branches = read(os.path.join(CSV, "branch_level_generation_1_to_14.csv"))
    pergen = read(os.path.join(CSV, "per_scan_generation_metrics.csv"))

    # per-case scalar metrics, keyed exactly as morphometry.json names them
    per_case = {}
    for cid, c in cases.items():
        age = f(c["age_years"])
        if age is None:
            continue
        g = lambda k, src=c: f(src.get(k))                                   # noqa: E731
        tr, lmb, rmb = g("trachea_mean_axis_diameter_mm"), g("lmb_mean_axis_diameter_mm"), g("rmb_mean_axis_diameter_mm")
        dist, prox = g("trachea_distal_mean_axis_diameter_mm"), g("trachea_proximal_mean_axis_diameter_mm")
        intra = g("intrathoracic_tracheal_length_mm")
        v, d = vol.get(cid, {}), fd.get(cid, {})
        m = {
            "age_years": age,
            "trachea_length_intrathoracic_mm": intra,
            "trachea_mean_diameter_mm": tr,
            "trachea_short_axis_mm": g("trachea_short_axis_mm"),
            "trachea_long_axis_mm": g("trachea_long_axis_mm"),
            "trachea_eccentricity": g("trachea_eccentricity_ratio"),
            "trachea_taper_distal_over_proximal": (dist / prox) if dist and prox else None,
            "trachea_length_over_diameter": (intra / tr) if intra and tr else None,
            "lmb_mean_diameter_mm": lmb, "rmb_mean_diameter_mm": rmb,
            "lmb_length_mm": g("lmb_centerline_length_mm"), "rmb_length_mm": g("rmb_centerline_length_mm"),
            "daughter_over_parent_diameter": ((lmb + rmb) / (2 * tr)) if lmb and rmb and tr else None,
            "right_left_asymmetry": ((rmb - lmb) / (rmb + lmb)) if lmb and rmb else None,
            "lung_volume_ml": g("total_segmented_lung_volume_mL"),
            "airway_lumen_volume_ml": f(v.get("airway_lumen_volume_mL")),
            "airway_to_lung_volume_percent": f(v.get("airway_to_lung_volume_percent")),
            "total_centreline_length_mm": f(v.get("total_centerline_length_mm")),
            "skeleton_fractal_dimension": f(d.get("whole_tree_skeleton_fractal_dimension")),
        }
        per_case[cid] = m

    # from the branch table: count, deepest generation, mean tortuosity
    by_case = defaultdict(list)
    for r in branches:
        by_case[r["case_id"]].append(r)
    for cid, rows in by_case.items():
        if cid not in per_case:
            continue
        gens = [int(float(r["generation"])) for r in rows]
        tort = [f(r["tortuosity_length_ratio"]) for r in rows]
        tort = [t for t in tort if t is not None]
        per_case[cid]["branch_count"] = len(rows)
        per_case[cid]["max_generation"] = max(gens) if gens else None
        per_case[cid]["mean_tortuosity"] = float(np.mean(tort)) if tort else None

    # per generation, per case
    pg = defaultdict(dict)
    for r in pergen:
        cid, g = r["case_id"], int(float(r["generation"]))
        if cid in per_case and 1 <= g <= MAX_GEN:
            pg[cid][g] = {"diameter_mean_mm": f(r["mean_diameter_mm"]),
                          "length_mean_mm": f(r["mean_centerline_length_mm"]),
                          "n": f(r["branch_count"])}

    def grid(values):
        vals = np.array([v for v in values if v is not None], dtype=float)
        if vals.size < 8:
            return None
        return {"n": int(vals.size), "p": GRID, "v": [round(float(x), 4) for x in np.percentile(vals, GRID)]}

    def window(band):
        i = BANDS.index(band)
        lo = BANDS[max(0, i - 1)][0]
        hi = BANDS[min(len(BANDS) - 1, i + 1)][1]
        return lo, hi

    metrics = sorted({k for m in per_case.values() for k in m if k != "age_years"})
    out = {"source": "90-scan normative pediatric cohort (FINAL_AIRWAY_90_RESULTS_2026-09-16), "
                     "10 scans per two-year band, one scan per patient; percentiles per band are "
                     "computed over that band and its two neighbours",
           "bands": [f"{lo}-{hi}" for lo, hi in BANDS], "grid": GRID,
           "metrics": {}, "per_generation": {}}
    for k in metrics:
        out["metrics"][k] = {}
        for band in BANDS:
            lo, hi = window(band)
            vals = [m[k] for m in per_case.values() if lo <= m["age_years"] < hi or (hi == 18 and m["age_years"] == 18)]
            gq = grid([m.get(k) for m in per_case.values()
                       if (lo <= m["age_years"] < hi or (hi == 18 and m["age_years"] == 18))])
            if gq:
                gq["ages"] = f"{lo}-{hi}"
                out["metrics"][k][f"{band[0]}-{band[1]}"] = gq
    for key in ("diameter_mean_mm", "length_mean_mm", "n"):
        out["per_generation"][key] = {}
        for g in range(1, MAX_GEN + 1):
            out["per_generation"][key][str(g)] = {}
            for band in BANDS:
                lo, hi = window(band)
                vals = [pg[cid][g][key] for cid, m in per_case.items()
                        if (lo <= m["age_years"] < hi or (hi == 18 and m["age_years"] == 18)) and g in pg.get(cid, {})]
                gq = grid(vals)
                if gq:
                    gq["ages"] = f"{lo}-{hi}"
                    out["per_generation"][key][str(g)][f"{band[0]}-{band[1]}"] = gq

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), separators=(",", ":"))
    print(f"{len(per_case)} cases, {len(metrics)} metrics, {len(out['per_generation']['n'])} generations -> {OUT} "
          f"({os.path.getsize(OUT) // 1024} KB)")
    for k in ("trachea_mean_diameter_mm", "branch_count", "max_generation"):
        print(k, {b: (v["n"], v["v"][4]) for b, v in out["metrics"][k].items()})


if __name__ == "__main__":
    sys.exit(main())
