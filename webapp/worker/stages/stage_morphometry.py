#!/usr/bin/env python
"""Measure the labelled airway tree of one upload.

Reuses the cohort's measurement code unchanged, so the numbers on the page are
the numbers in the cohort tables:

  measure_case()   or_all3_run/scripts/09_airmorph_morphometry.py
                   one row per branch: centreline length, chord, tortuosity,
                   EDT radius statistics, branching angle, generation, names
  axis_geometry    FINAL_AIRWAY_90_RESULTS_2026-09-16/reproducibility_scripts/
                   perpendicular cross sections of trachea and main bronchi,
                   intrathoracic tracheal length, lung volume

Writes results/morphometry_branches.csv (all branches, 40 columns) and
results/morphometry.json (the summary the page shows). The central-airway
block needs one root with two children; a sparse tree gets "not available"
there and everything else still comes out.

Runs in the AirMorph environment, CPU only. A few minutes on a deep tree.
"""
import argparse, csv, importlib.util, json, math, os, sys, time
from pathlib import Path
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config as C  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                  # dataclasses look the module up here
    spec.loader.exec_module(mod)
    return mod


def clean(x):
    """JSON has no NaN."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else round(float(x), 3)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def fnum(rows, key):
    v = np.array([r[key] for r in rows], float)
    return v[np.isfinite(v)]


def agg(rows, key, fn=np.mean):
    v = fnum(rows, key)
    return float(fn(v)) if len(v) else float("nan")


# --------------------------------------------------------- central topology
def central_topology(rows):
    """Root, left and right main bronchus. Mirrors extract_central_topology in
    or_all3_run/scripts/16_build_meeting_package.py for a single case."""
    by_id = {int(r["branch_instance_id"]): r for r in rows}
    children = {}
    roots = []
    for r in rows:
        p = r["parent_instance_id"]
        if p is None or (isinstance(p, float) and math.isnan(p)):
            roots.append(int(r["branch_instance_id"]))
        else:
            children.setdefault(int(p), []).append(int(r["branch_instance_id"]))
    if len(roots) != 1:
        raise ValueError(f"expected one tracheal root, found {len(roots)}")
    root = roots[0]

    def descendants(b):
        out, stack = [], [b]
        while stack:
            x = stack.pop(); out.append(x); stack.extend(children.get(x, []))
        return out

    def votes(r):
        left = right = 0
        for k in ("branch_name", "lobar_name", "segmental_name", "subsegmental_name"):
            v = str(r.get(k, ""))
            left += v.startswith(("Left", "LB")); right += v.startswith(("Right", "RB"))
        return left, right

    cands = []
    for c in children.get(root, []):
        sub = descendants(c)
        l = r_ = 0
        for b in sub:
            a, b_ = votes(by_id[b]); l += a; r_ += b_
        cands.append({"id": c, "n": len(sub), "l": l, "r": r_})
    cands.sort(key=lambda x: x["n"], reverse=True)
    sel = cands[:2]
    if len(sel) != 2:
        raise ValueError("fewer than two root children")
    for s in sel:
        s["side"] = "L" if s["l"] > s["r"] else "R"
    method = "anatomic-label-votes"
    if {s["side"] for s in sel} != {"L", "R"}:
        method = "physical-x-fallback"
        return root, sel, method          # caller resolves with the image
    side = {s["side"]: s["id"] for s in sel}
    return root, {"L": side["L"], "R": side["R"], "n": {s["id"]: s["n"] for s in sel}}, method


def resolve_sides_by_x(case, cid, root, sel):
    """LPS physical x increases to patient left."""
    img = sitk.ReadImage(str(case / f"{cid}_skel_parsing.nii.gz"))
    arr = sitk.GetArrayFromImage(img)
    for s in sel:
        z, y, x = np.argwhere(arr == s["id"]).mean(0)
        s["x"] = img.TransformContinuousIndexToPhysicalPoint((float(x), float(y), float(z)))[0]
    sel.sort(key=lambda s: s["x"])
    return {"R": sel[0]["id"], "L": sel[1]["id"], "n": {s["id"]: s["n"] for s in sel}}


# ------------------------------------------------------------------ fractal
def box_dimension(points_mm):
    """Whole-tree box-counting dimension of the skeleton in physical mm.
    Same recipe as compute_fractal_volume_and_workbook.py in the 90-case report."""
    points = np.asarray(points_mm, float)
    points = points - points.min(axis=0)
    scales = np.array([1., 2., 4., 8., 16., 32.])
    counts = np.array([len(np.unique(np.floor(points / e).astype(np.int32), axis=0)) for e in scales])
    use = (counts >= 8) & (counts <= max(8, int(.95 * len(points))))
    if use.sum() < 4:
        use = counts >= 4
    if use.sum() < 3:
        return float("nan"), float("nan")
    x = np.log(1 / scales[use]); y = np.log(counts[use])
    slope, icpt = np.polyfit(x, y, 1)
    pred = slope * x + icpt
    ssr = ((y - pred) ** 2).sum(); sst = ((y - y.mean()) ** 2).sum()
    return float(slope), float(1 - ssr / sst) if sst else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-dir", required=True)
    a = ap.parse_args()
    t0 = time.time()
    job = Path(a.job_dir)
    case = job / "case"
    cid = case.name
    res = job / "results"
    res.mkdir(exist_ok=True)
    summary = json.load(open(res / "summary.json")) if (res / "summary.json").exists() else {}

    morph = load_module("airmorph_morphometry", C.MORPHOMETRY_SCRIPT)
    print("measuring branches", flush=True)
    rows = morph.measure_case(case)
    for r in rows:
        r["case_id"] = "upload"; r["subject"] = job.name; r["age_group"] = "n/a"
        r["source_mask"] = "airway_mask.nii.gz"
    print(f"{len(rows)} branches in {time.time() - t0:.0f}s", flush=True)
    morph.write_csv(res / "morphometry_branches.csv", rows)

    # ---- whole tree ----------------------------------------------------------
    gens = [int(r["generation"]) for r in rows]
    out = {
        "branch_count": len(rows),
        "total_centreline_length_mm": agg(rows, "centerline_length_mm", np.sum),
        "max_generation": max(gens) if gens else None,
        "airway_lumen_volume_ml": summary.get("airway_volume_ml"),
        "mean_tortuosity": agg(rows, "tortuosity_length_ratio"),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    per_gen = []
    for g in sorted(set(gens)):
        rr = [r for r in rows if int(r["generation"]) == g]
        per_gen.append({"generation": g, "n": len(rr),
                        "diameter_mean_mm": agg(rr, "diameter_mean_mm"),
                        "diameter_median_mm": agg(rr, "diameter_median_mm", np.median),
                        "length_mean_mm": agg(rr, "centerline_length_mm"),
                        "length_total_mm": agg(rr, "centerline_length_mm", np.sum),
                        "angle_mean_deg": agg(rr, "branch_angle_deg")})
    out["per_generation"] = per_gen
    per_lobe = []
    for name in sorted(set(r["lobar_name"] for r in rows)):
        rr = [r for r in rows if r["lobar_name"] == name]
        per_lobe.append({"lobe": name, "n": len(rr),
                         "length_total_mm": agg(rr, "centerline_length_mm", np.sum),
                         "volume_ml": agg(rr, "volume_mm3", np.sum) / 1000,
                         "diameter_mean_mm": agg(rr, "diameter_mean_mm"),
                         "max_generation": max(int(r["generation"]) for r in rr)})
    out["per_lobe"] = per_lobe

    # ---- fractal dimension of the skeleton, physical mm ----------------------
    try:
        img = sitk.ReadImage(str(case / f"{cid}_skel_parsing.nii.gz"))
        arr = sitk.GetArrayFromImage(img)
        coords = np.argwhere(arr > 0)[:, ::-1].astype(float)
        D = np.asarray(img.GetDirection()).reshape(3, 3); S = np.diag(img.GetSpacing())
        pts = coords @ (D @ S).T + np.asarray(img.GetOrigin())
        fd, r2 = box_dimension(pts)
        out["skeleton_fractal_dimension"] = fd
        out["skeleton_fractal_r2"] = r2
        out["skeleton_points"] = int(len(pts))
    except Exception as e:                                        # noqa: BLE001
        print(f"fractal dimension skipped: {e}", flush=True)

    # ---- central airways --------------------------------------------------------
    central = {"status": "not available"}
    try:
        root, sel, method = central_topology(rows)
        if method == "physical-x-fallback":
            sel = resolve_sides_by_x(case, cid, root, sel)
        by_id = {int(r["branch_instance_id"]): r for r in rows}
        rr, lm, rm = by_id[root], by_id[sel["L"]], by_id[sel["R"]]
        central = {
            "status": "ok", "assignment": method,
            "trachea": {"id": root, "length_mm": rr["centerline_length_mm"],
                        "diameter_edt_mean_mm": rr["diameter_mean_mm"],
                        "diameter_edt_min_mm": rr["diameter_min_mm"],
                        "tortuosity": rr["tortuosity_length_ratio"]},
            "lmb": {"id": sel["L"], "length_mm": lm["centerline_length_mm"],
                    "diameter_edt_mean_mm": lm["diameter_mean_mm"],
                    "angle_deg": lm["branch_angle_deg"], "descendants": sel["n"][sel["L"]]},
            "rmb": {"id": sel["R"], "length_mm": rm["centerline_length_mm"],
                    "diameter_edt_mean_mm": rm["diameter_mean_mm"],
                    "angle_deg": rm["branch_angle_deg"], "descendants": sel["n"][sel["R"]]},
        }
        # sum of the two local parent-child angles: a proxy for the carinal
        # angle, not a measurement of it
        if all(np.isfinite([lm["branch_angle_deg"], rm["branch_angle_deg"]])):
            central["interbronchial_angle_proxy_deg"] = lm["branch_angle_deg"] + rm["branch_angle_deg"]
        # cross-sectional convention
        try:
            ag = load_module("axis_geometry", C.AXIS_GEOMETRY_SCRIPT)
            print("cross sections", flush=True)
            x = ag.measure_case(cid, case, root, sel["L"], sel["R"])
            g = lambda k: x.get(k, float("nan"))                               # noqa: E731
            central["cross_section"] = {
                "trachea_length_full_mm": g("full_segmented_tracheal_length_mm"),
                "trachea_length_intrathoracic_mm": g("intrathoracic_tracheal_length_mm"),
                "trachea_short_axis_mm": g("trachea_short_axis_mm"),
                "trachea_long_axis_mm": g("trachea_long_axis_mm"),
                "trachea_mean_diameter_mm": g("trachea_mean_axis_diameter_mm"),
                "trachea_eccentricity": g("trachea_eccentricity_ratio"),
                "trachea_taper_distal_over_proximal": (
                    g("trachea_distal_mean_axis_diameter_mm") / g("trachea_proximal_mean_axis_diameter_mm")
                    if g("trachea_proximal_mean_axis_diameter_mm") else float("nan")),
                "trachea_length_over_diameter": (
                    g("intrathoracic_tracheal_length_mm") / g("trachea_mean_axis_diameter_mm")
                    if g("trachea_mean_axis_diameter_mm") else float("nan")),
                "lmb_mean_diameter_mm": g("lmb_mean_axis_diameter_mm"),
                "rmb_mean_diameter_mm": g("rmb_mean_axis_diameter_mm"),
                "lmb_length_mm": g("lmb_centerline_length_mm"),
                "rmb_length_mm": g("rmb_centerline_length_mm"),
                "daughter_over_parent_diameter": (
                    (g("lmb_mean_axis_diameter_mm") + g("rmb_mean_axis_diameter_mm"))
                    / (2 * g("trachea_mean_axis_diameter_mm"))
                    if g("trachea_mean_axis_diameter_mm") else float("nan")),
                # (R - L) / (R + L), the definition in the 90-case report
                "right_left_asymmetry": (
                    (g("rmb_mean_axis_diameter_mm") - g("lmb_mean_axis_diameter_mm"))
                    / (g("rmb_mean_axis_diameter_mm") + g("lmb_mean_axis_diameter_mm"))
                    if (g("rmb_mean_axis_diameter_mm") + g("lmb_mean_axis_diameter_mm")) else float("nan")),
                "lung_volume_ml": g("total_segmented_lung_volume_mL"),
                "airway_to_lung_volume_percent": (
                    100 * summary["airway_volume_ml"] / g("total_segmented_lung_volume_mL")
                    if summary.get("airway_volume_ml") and g("total_segmented_lung_volume_mL") else float("nan")),
                "cross_sections_trachea": g("trachea_cross_sections"),
            }
        except Exception as e:                                    # noqa: BLE001
            central["cross_section"] = {"status": "not available", "reason": str(e)[:300]}
            print(f"cross sections skipped: {e}", flush=True)
    except Exception as e:                                        # noqa: BLE001
        central = {"status": "not available", "reason": str(e)[:300]}
        print(f"central airways skipped: {e}", flush=True)
    out["central"] = central

    # ---- sanity flags for the reader (the package stage adds its own) -------
    qc = []
    if central.get("status") != "ok":
        qc.append({"code": "central_airways_unresolved", "level": "warn",
                   "text": "Trachea and main bronchi could not be identified from the tree, so the "
                           "central-airway measurements are missing. The labelling near the carina is suspect."})
    if out["max_generation"] is not None and out["max_generation"] < 8:
        qc.append({"code": "shallow_tree", "level": "warn",
                   "text": f"The deepest branch is generation {out['max_generation']}; the normative cohort "
                           "reaches 12 to 13. Peripheral airways were probably not segmented."})
    broken = sum(1 for r in rows if "disconnected-centerline" in str(r.get("geometry_qc_flags", "")))
    if rows and broken > 0.1 * len(rows):
        qc.append({"code": "fragmented_centrelines", "level": "info",
                   "text": f"{broken} of {len(rows)} branches have a centreline in more than one piece; "
                           "their lengths and tortuosity are less reliable."})
    out["qc"] = qc
    out["seconds"] = round(time.time() - t0, 1)

    json.dump(clean(out), open(res / "morphometry.json", "w"), indent=1)
    # branch table for the page: the columns a reader wants, not all 40
    keep = ["branch_instance_id", "parent_instance_id", "generation", "branch_name", "lobar_name",
            "segmental_name", "centerline_length_mm", "diameter_mean_mm", "diameter_min_mm",
            "tortuosity_length_ratio", "branch_angle_deg", "volume_mm3", "geometry_qc_flags"]
    json.dump(clean([{k: r[k] for k in keep} for r in rows]),
              open(res / "morphometry_table.json", "w"))
    # the zip was closed by the packaging stage before these existed
    import zipfile
    zp = res / "airway_results.zip"
    if zp.exists():
        with zipfile.ZipFile(zp, "a", zipfile.ZIP_DEFLATED) as z:
            for f in ("morphometry.json", "morphometry_branches.csv"):
                if f not in z.namelist():
                    z.write(res / f, f)
    print(f"morphometry done in {out['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
