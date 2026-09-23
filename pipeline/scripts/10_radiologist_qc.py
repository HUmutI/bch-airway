#!/usr/bin/env python3
"""Create axial CT + branch-mask QC videos and a review workbook.

Each MP4 is deliberately a single axial view.  The branch-parsed airway mask is
overlaid on CT at 50% opacity.  Raw NIfTI outputs remain the source of truth.
"""
from __future__ import annotations

import os
import argparse
import csv
import json
import logging
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from matplotlib.backends.backend_agg import FigureCanvasAgg


BASE = Path(os.environ.get("OR3_BASE", "/fileserver/external/quin/ege"))
RUN = BASE / "or_all3_run"
STATUS = RUN / "normal_airway_morphometry_status.csv"
DEFAULT_OUT = RUN / "radiologist_qc"


def case_dirs(status_path: Path) -> list[Path]:
    with status_path.open(newline="") as f:
        return [Path(row["case_dir"]) for row in csv.DictReader(f) if not row["error"]]


def read_array(path: Path):
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image), image


def normalized_ct(image: np.ndarray) -> np.ndarray:
    return np.clip((image.astype(np.float32) + 1000.0) / 1400.0, 0, 1)


def review_label(names: list[str], is_root: bool) -> str:
    if all(str(x).lower() == "trachea" for x in names):
        return "trachea" if is_root else "proximal_or_unclassified"
    # Segment-level colors are deliberately used in the rapid-review video.
    # Subsegmental labels remain available in the NIfTI/JSON and spreadsheet;
    # showing ~80 subsegmental colors would make a video legend unreadable.
    for value in (names[1], names[0]):
        if value and str(value).lower() != "trachea":
            return str(value)
    return "proximal_or_unclassified"


def label_maps(case_dir: Path, case_id: str):
    annotation = json.loads((case_dir / f"{case_id}_anno.json").read_text())
    edge = np.load(case_dir / f"{case_id}_airway_graph.npy").astype(int)
    children = {int(child) + 1 for child in edge[1]}
    roots = set(map(int, annotation)) - children
    names = {int(k): review_label(v, int(k) in roots) for k, v in annotation.items()}
    canonical = sorted(set(names.values()))
    cmap = plt.get_cmap("turbo")
    colors = {name: cmap(i / max(1, len(canonical) - 1))[:3]
              for i, name in enumerate(canonical)}
    colors["proximal_or_unclassified"] = (0.65, 0.65, 0.65)
    colors["trachea"] = (1.0, 1.0, 1.0)
    return names, colors


def color_branch_mask(labels2d: np.ndarray, names, colors) -> np.ndarray:
    rgba = np.zeros(labels2d.shape + (4,), dtype=np.float32)
    for value in np.unique(labels2d):
        if value <= 0:
            continue
        pixels = labels2d == value
        rgba[pixels, :3] = colors[names.get(int(value), "proximal_or_unclassified")]
        rgba[pixels, 3] = 0.5
    return rgba


def render_frame(ct, branch_mask, names, colors, case_id, frame, frames,
                 z_bounds) -> np.ndarray:
    frac = frame / max(1, frames - 1)
    z0, z1 = z_bounds
    z = int(round(z0 + frac * (z1 - z0)))

    fig = plt.figure(figsize=(8, 8), facecolor="black")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes((0.02, 0.02, 0.96, 0.96), facecolor="black")
    ax.imshow(ct[z], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.imshow(color_branch_mask(branch_mask[z], names, colors), interpolation="nearest")
    ax.set_title(f"{case_id}  |  Axial {z + 1}/{ct.shape[0]}  |  Branch mask opacity 0.5",
                 fontsize=11, color="white")
    ax.axis("off")
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    result = rgba[:, :, :3].copy()
    plt.close(fig)
    return result


def make_video(case_dir: Path, out_dir: Path, frames: int, fps: int, force: bool):
    case_id = case_dir.name
    output = out_dir / "videos" / f"{case_id}_qc.mp4"
    if output.exists() and not force:
        return output, "skipped-existing"

    required = [case_dir / "image.nii.gz", case_dir / "union3_lcc.nii.gz",
                case_dir / f"{case_id}_parse.nii.gz",
                case_dir / f"{case_id}_anno.json",
                case_dir / f"{case_id}_airway_graph.npy"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("missing: " + ", ".join(missing))

    ct_raw, ct_img = read_array(required[0])
    mask_raw, mask_img = read_array(required[1])
    branch_mask, parse_img = read_array(required[2])
    if ct_raw.shape != mask_raw.shape or mask_raw.shape != branch_mask.shape:
        raise ValueError(f"geometry shape mismatch: {ct_raw.shape}, {mask_raw.shape}, {branch_mask.shape}")
    if tuple(ct_img.GetSpacing()) != tuple(mask_img.GetSpacing()):
        raise ValueError("CT/mask spacing mismatch")

    ct = normalized_ct(ct_raw)
    mask = mask_raw > 0
    branch_mask = branch_mask.astype(np.int32)
    names, colors = label_maps(case_dir, case_id)
    foreground = np.argwhere(mask)
    lo = foreground.min(axis=0)
    hi = foreground.max(axis=0)
    margin = np.maximum(3, np.rint((hi - lo) * 0.03).astype(int))
    lo = np.maximum(0, lo - margin)
    hi = np.minimum(np.asarray(mask.shape) - 1, hi + margin)
    z_bounds = (int(lo[0]), int(hi[0]))

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output, fps=fps, codec="libx264", quality=7,
                                macro_block_size=2)
    try:
        for frame in range(frames):
            rgb = render_frame(ct, branch_mask, names, colors, case_id, frame,
                               frames, z_bounds)
            writer.append_data(rgb)
    finally:
        writer.close()
    return output, "created"


def write_review_workbook(out_dir: Path, rows: list[dict]):
    path = out_dir / "radiologist_review.xlsx"
    instructions = [
        ["Purpose", "Rapid research QC of pediatric axial CT with branch-parsed airway-mask overlay."],
        ["Mask score", "1 unusable, 2 major errors, 3 usable with edits, 4 good, 5 excellent"],
        ["Branch-mask score", "1 unusable, 2 major errors, 3 usable with edits, 4 good, 5 excellent"],
        ["Anatomy score", "1 mostly wrong, 2 major errors, 3 mixed, 4 mostly correct, 5 correct"],
        ["Decision", "accept / accept_with_edits / reject"],
        ["Overlay", "Branch-parsed airway mask over axial CT at 50% opacity."],
        ["Gray branches", "Model returned trachea away from the root; shown as proximal_or_unclassified."],
        ["Detailed review", "Use NIfTI volumes in a medical image viewer; MP4 is a screening aid."],
    ]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Case Review", index=False)
        pd.DataFrame(instructions, columns=["item", "guidance"]).to_excel(
            writer, sheet_name="Instructions", index=False)
        ws = writer.book["Case Review"]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = min(
                60, max(12, max(len(str(c.value or "")) for c in col) + 2))
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=Path, default=STATUS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--case")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dirs = case_dirs(args.status)
    if args.case:
        dirs = [d for d in dirs if d.name == args.case]
    if args.max_cases:
        dirs = dirs[:args.max_cases]
    if not dirs:
        parser.error("no completed cases selected")

    automated_qc = {}
    analysis_path = RUN / "radiologist_qc/tables/analysis_ready_airway_branch_morphometry.csv"
    if analysis_path.exists():
        analysis = pd.read_csv(analysis_path, low_memory=False)
        for case_id, group in analysis.groupby("case_id"):
            reasons = []
            root_defects = group.segmentation_qc_flags.fillna("").astype(bool).sum()
            anatomy_flags = group.anatomy_review_required.fillna(False).astype(bool).sum()
            if root_defects:
                reasons.append("ROOT_TRACHEA_RADIUS_AT_GRID_FLOOR")
            if anatomy_flags:
                reasons.append(f"ANATOMY_HEAD_CONTRADICTIONS={anatomy_flags}")
            priority = "HIGH" if root_defects else ("MEDIUM" if anatomy_flags else "STANDARD")
            automated_qc[case_id] = (priority, ";".join(reasons))

    existing_reviews = {}
    existing_workbook = args.out_dir / "radiologist_review.xlsx"
    if existing_workbook.exists():
        try:
            previous = pd.read_excel(existing_workbook, sheet_name="Case Review", dtype=object)
            existing_reviews = {str(row["case_id"]): row.to_dict()
                                for _, row in previous.iterrows()}
        except Exception as exc:
            logging.warning("could not preserve existing review entries: %s", exc)

    rows, failures = [], 0
    for number, case_dir in enumerate(dirs, 1):
        logging.info("[%d/%d] %s", number, len(dirs), case_dir.name)
        row = {
            "case_id": case_dir.name, "video": "",
            "generation_status": "", "generation_error": "",
            "automated_qc_priority": automated_qc.get(case_dir.name, ("STANDARD", ""))[0],
            "automated_qc_reasons": automated_qc.get(case_dir.name, ("STANDARD", ""))[1],
            "reviewer_1": "", "reviewer_1_date": "",
            "reviewer_1_mask_quality_1_to_5": "",
            "reviewer_1_branch_mask_quality_1_to_5": "",
            "reviewer_1_anatomy_quality_1_to_5": "",
            "reviewer_1_decision": "", "reviewer_1_comments": "",
            "reviewer_2": "", "reviewer_2_date": "",
            "reviewer_2_mask_quality_1_to_5": "",
            "reviewer_2_branch_mask_quality_1_to_5": "",
            "reviewer_2_anatomy_quality_1_to_5": "",
            "reviewer_2_decision": "", "reviewer_2_comments": "",
        }
        previous = existing_reviews.get(case_dir.name, {})
        for key, value in previous.items():
            if key in row and key.startswith("reviewer_") and pd.notna(value):
                row[key] = value
        try:
            video, status = make_video(case_dir, args.out_dir, args.frames,
                                       args.fps, args.force)
            # Keep the workbook portable when videos are first rendered in a
            # validated staging directory and then moved into the delivery dir.
            row.update(video=str(Path("videos") / video.name), generation_status=status)
        except Exception as exc:
            failures += 1
            row["generation_status"] = "failed"
            row["generation_error"] = f"{type(exc).__name__}: {exc}"
            logging.exception("QC generation failed for %s", case_dir.name)
        rows.append(row)

    workbook = write_review_workbook(args.out_dir, rows)
    logging.info("wrote %s (%d cases, %d failures)", workbook, len(rows), failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
