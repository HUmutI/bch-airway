#!/usr/bin/env python3
"""Build tables, figures, and the workbook for the cross-sectional analysis."""
from __future__ import annotations

import shutil
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).resolve().parent))

import topology_qc

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "font.weight": "normal",
    "mathtext.fontset": "dejavusans",
    "axes.labelsize": 17,
    "axes.labelweight": "normal",
    "axes.titleweight": "normal",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})

# Figures 4 and 10 use a 12 x 16 canvas and are height-limited when LaTeX
# places them. The other report figures are width-limited. Compensating for
# that different placement scale makes their axis labels the same visible
# size on the PDF page as the 17 pt labels in Figures 4 and 10.
PAGE_MATCHED_AXIS_LABEL = {
    "fontsize": 14.5,
    "fontfamily": "DejaVu Sans",
    "fontweight": "normal",
}

RUN = Path(__file__).resolve().parents[1]
BASE = Path("/fileserver/external/quin/ege")
OUT = RUN / "AIRWAY_RESULTS_90"
CSV_DIR = OUT / "csv"
FIG_DIR = OUT / "figures"

BUCKETS = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12", "12-14", "14-16", "16-18"]
DISPLAY = {bucket: bucket.replace("-", "–") for bucket in BUCKETS} | {"All": "All"}
MAX_GENERATION = 14

# Nine visually distinct age-group colours. Marker and line style vary as well
# so the groups stay separable in greyscale and for colour-vision deficiency.
PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#4d4d4d",
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]
LINESTYLES = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-"]
STYLE = {
    bucket: {"color": PALETTE[i], "marker": MARKERS[i], "linestyle": LINESTYLES[i]}
    for i, bucket in enumerate(BUCKETS)
}

CENTRAL_METRICS = [
    "intrathoracic_tracheal_length_mm",
    "trachea_mean_axis_diameter_mm",
    "lmb_mean_axis_diameter_mm",
    "rmb_mean_axis_diameter_mm",
]
CENTRAL_LABELS = [
    "Intrathoracic Tracheal Length (mm)",
    "Tracheal Diameter (mm)",
    "LMB Diameter (mm)",
    "RMB Diameter (mm)",
]
SECTION_METRICS = [
    "trachea_distal_mean_axis_diameter_mm",
    "trachea_middle_mean_axis_diameter_mm",
    "trachea_proximal_mean_axis_diameter_mm",
]
AXIS_METRICS = [
    "trachea_short_axis_mm", "trachea_long_axis_mm",
    "lmb_short_axis_mm", "lmb_long_axis_mm",
    "rmb_short_axis_mm", "rmb_long_axis_mm",
]


# ---------------------------------------------------------------- statistics


def describe(values: pd.Series) -> dict:
    values = pd.to_numeric(values, errors="coerce").dropna()
    keys = ("n", "mean", "sd", "median", "q1", "q3", "iqr", "min", "max")
    if not len(values):
        return {key: np.nan for key in keys}
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def fmt_median_quartiles(values: pd.Series, expected: int | None = None) -> str:
    """Median with first and third quartiles."""
    stat = describe(values)
    if np.isnan(stat["median"]):
        return "NA"
    text = f"{stat['median']:.1f} ({stat['q1']:.1f}, {stat['q3']:.1f})"
    return text


def fmt_median_iqr(values: pd.Series) -> str:
    stat = describe(values)
    if np.isnan(stat["median"]):
        return "NA"
    return f"{stat['median']:.1f} ({stat['iqr']:.1f})"


def sex_counts(group: pd.DataFrame) -> tuple[int, int, int]:
    """Male, female, and unrecorded counts for one age group."""
    sex = group.sex.fillna("Unknown").astype(str).replace("", "Unknown").str.title()
    return (
        int(sex.eq("Male").sum()),
        int(sex.eq("Female").sum()),
        int(sex.eq("Unknown").sum()),
    )


def groups(case: pd.DataFrame, include_all: bool = True):
    for bucket in BUCKETS:
        yield bucket, case[case.age_bucket == bucket]
    if include_all:
        yield "All", case


def grouped_descriptive(case: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    records = []
    for bucket, group in groups(case):
        for metric in metrics:
            record = {
                "age_group": DISPLAY[bucket],
                "age_bucket_code": bucket,
                "metric": metric,
                "n_subjects": int(group.subject.nunique()),
                "n_scans": int(group.case_id.nunique()),
            }
            record.update(describe(group[metric]))
            records.append(record)
    return pd.DataFrame(records)


# -------------------------------------------------------------------- tables


def table_dataset_overview(case: pd.DataFrame) -> pd.DataFrame:
    """Everything descriptive about the cohort, in one table.

    This is the table the report opens with, so sex, age, size, and provenance
    are all stated once here and the airway tables that follow stay numeric.
    """
    rows = []
    for bucket, group in groups(case):
        male, female, unknown = sex_counts(group)
        age = describe(group.age_years)
        volume = describe(group.total_segmented_lung_volume_mL)
        sex_cell = f"{male}/{female}"
        if unknown:
            sex_cell += f" ({unknown} not recorded)"
        rows.append({
            "Age Group (y)": DISPLAY[bucket],
            "No. of Subjects": int(group.subject.nunique()),
            "No. of CT Scans": int(group.case_id.nunique()),
            "Sex (M/F)": sex_cell,
            "Male (%)": f"{100.0 * male / max(len(group), 1):.0f}%",
            "Age, mean (SD)": f"{age['mean']:.1f} ({age['sd']:.1f})",
            "Age, median (IQR)": f"{age['median']:.1f} ({age['iqr']:.1f})",
            "Age, range": f"{age['min']:.1f}-{age['max']:.1f}",
            "Lung Volume mL, mean (SD)": f"{volume['mean']:.0f} ({volume['sd']:.0f})",
            "Lung Volume mL, median": f"{volume['median']:.0f}",
            "Source (original/E3)": (
                f"{int((group.source_cohort == 'original_83').sum())}/"
                f"{int((group.source_cohort == 'E3').sum())}"
            ),
        })
    return pd.DataFrame(rows)


def table_central(case: pd.DataFrame, normalized: bool = False) -> pd.DataFrame:
    metrics = [f"{m}_per_L_cuberoot" for m in CENTRAL_METRICS] if normalized else CENTRAL_METRICS
    labels = [
        label.replace("(mm)", "(mm/L^(1/3))") for label in CENTRAL_LABELS
    ] if normalized else CENTRAL_LABELS
    rows = []
    for bucket, group in groups(case):
        row = {
            "Age Range (y)": DISPLAY[bucket],
            "No. of Subjects": int(group.subject.nunique()),
            "No. of CT Scans": int(group.case_id.nunique()),
            "Age (y)": fmt_median_iqr(group.age_years),
        }
        for metric, label in zip(metrics, labels):
            row[label] = fmt_median_quartiles(group[metric], expected=len(group))
        rows.append(row)
    return pd.DataFrame(rows)


def table_sections(case: pd.DataFrame, normalized: bool = False) -> pd.DataFrame:
    metrics = [f"{m}_per_L_cuberoot" for m in SECTION_METRICS] if normalized else SECTION_METRICS
    suffix = " (mm/L^(1/3))" if normalized else " (mm)"
    rows = []
    for bucket, group in groups(case):
        row = {
            "Age Range (y)": DISPLAY[bucket],
            "No. of Subjects": int(group.subject.nunique()),
            "No. of CT Scans": int(group.case_id.nunique()),
        }
        for metric, name in zip(metrics, ("Distal", "Middle", "Proximal")):
            row[name + suffix] = fmt_median_iqr(group[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def table_axes(case: pd.DataFrame) -> pd.DataFrame:
    """Short and long calliper axes, which the mean-axis diameter averages."""
    rows = []
    for bucket, group in groups(case):
        row = {
            "Age Range (y)": DISPLAY[bucket],
            "No. of Subjects": int(group.subject.nunique()),
            "No. of CT Scans": int(group.case_id.nunique()),
        }
        for structure, prefix in (("Trachea", "trachea"), ("LMB", "lmb"), ("RMB", "rmb")):
            row[f"{structure} Short Axis (mm)"] = fmt_median_quartiles(
                group[f"{prefix}_short_axis_mm"], expected=len(group)
            )
            row[f"{structure} Long Axis (mm)"] = fmt_median_quartiles(
                group[f"{prefix}_long_axis_mm"], expected=len(group)
            )
            row[f"{structure} Short/Long Ratio"] = fmt_median_iqr(
                group[f"{prefix}_eccentricity_ratio"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def table_convention_comparison(case: pd.DataFrame) -> pd.DataFrame:
    """Quantify how much each convention change moved the reported numbers."""
    rows = []
    for bucket, group in groups(case):
        row = {
            "Age Range (y)": DISPLAY[bucket],
            "No. of CT Scans": int(group.case_id.nunique()),
            "Full Segmented Tracheal Length (mm)": fmt_median_quartiles(
                group.full_segmented_tracheal_length_mm
            ),
            "Intrathoracic Tracheal Length (mm)": fmt_median_quartiles(
                group.intrathoracic_tracheal_length_mm
            ),
            "Cervical Portion Removed (mm)": fmt_median_quartiles(
                group.cervical_tracheal_length_mm
            ),
            "Length Reduction (%)": fmt_median_iqr(group.tracheal_length_reduction_percent),
            "Tracheal Diameter, EDT (mm)": fmt_median_quartiles(
                group.tracheal_edt_mean_diameter_mm
            ),
            "Tracheal Diameter, Mean Axis (mm)": fmt_median_quartiles(
                group.trachea_mean_axis_diameter_mm
            ),
            "Diameter Increase (%)": fmt_median_iqr(group.tracheal_diameter_increase_percent),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def table_exclusions(case: pd.DataFrame) -> pd.DataFrame:
    """Every measurement removed by the topology and plausibility checks."""
    flagged = case[
        case.lmb_diameter_excluded | case.rmb_diameter_excluded
        | case.fragmented_main_bronchus | case.detached_stump
    ]
    rows = []
    for row in flagged.itertuples():
        removed = [
            name for name, dropped in (("LMB", row.lmb_diameter_excluded),
                                       ("RMB", row.rmb_diameter_excluded)) if dropped
        ]
        rows.append({
            "Case": row.case_id,
            "Age Group (y)": DISPLAY[row.age_bucket],
            "QC Status": "retained in inclusive model-output analysis",
            "Finding": "; ".join(dict.fromkeys(
                text for text in (row.topology_note, row.exclusion_reason) if text
            )),
        })
    return pd.DataFrame(rows)


def table_lung_volume(case: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bucket, group in groups(case):
        stat = describe(group.total_segmented_lung_volume_mL)
        rows.append({
            "Age Range (y)": DISPLAY[bucket],
            "No. of Subjects": int(group.subject.nunique()),
            "No. of CT Scans": int(group.case_id.nunique()),
            "Mean": f"{stat['mean']:.0f}",
            "SD": f"{stat['sd']:.0f}",
            "Median": f"{stat['median']:.0f}",
            "Q1": f"{stat['q1']:.0f}",
            "Q3": f"{stat['q3']:.0f}",
            "Min": f"{stat['min']:.0f}",
            "Max": f"{stat['max']:.0f}",
        })
    return pd.DataFrame(rows)


def table_sex_distribution(case: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bucket, group in groups(case):
        sex = group.sex.fillna("Unknown").astype(str).str.title()
        total = max(len(sex), 1)
        known = int(sex.isin(["Male", "Female"]).sum())
        male = int(sex.eq("Male").sum())
        female = int(sex.eq("Female").sum())
        rows.append({
            "Age Range (y)": DISPLAY[bucket],
            "No. of CT Scans": len(group),
            "Male, n": male,
            "Female, n": female,
            "Unknown, n": int(sex.eq("Unknown").sum()),
            "Male, % of all scans": f"{100.0 * male / total:.1f}%",
            "Female, % of all scans": f"{100.0 * female / total:.1f}%",
            "Male, % of scans with known sex": f"{100.0 * male / known:.1f}%" if known else "NA",
            "Female, % of scans with known sex": f"{100.0 * female / known:.1f}%" if known else "NA",
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------- generation data


def generation_tables(branches: pd.DataFrame, case: pd.DataFrame):
    """Per-scan and per-age-group generation summaries, capped at generation 14."""
    metadata = case[
        ["case_id", "subject", "age_years", "age_bucket",
         "total_segmented_lung_volume_mL", "lung_linear_scale_L_cuberoot"]
    ].drop_duplicates("case_id")
    data = branches.drop(columns=["subject"], errors="ignore").merge(
        metadata, on="case_id", validate="many_to_one"
    )
    data = data[data.generation <= MAX_GENERATION].copy()
    data["diameter_mean_mm"] = 2.0 * data.radius_mean_mm
    data["radius_mean_mm_per_L_cuberoot"] = data.radius_mean_mm / data.lung_linear_scale_L_cuberoot
    data["diameter_mean_mm_per_L_cuberoot"] = data.diameter_mean_mm / data.lung_linear_scale_L_cuberoot
    data["centerline_length_mm_per_L_cuberoot"] = (
        data.centerline_length_mm / data.lung_linear_scale_L_cuberoot
    )

    per_case = data.groupby(
        ["age_bucket", "case_id", "subject", "generation"], as_index=False
    ).agg(
        branch_count=("branch_instance_id", "size"),
        mean_radius_mm=("radius_mean_mm", "mean"),
        median_radius_mm=("radius_mean_mm", "median"),
        mean_diameter_mm=("diameter_mean_mm", "mean"),
        median_diameter_mm=("diameter_mean_mm", "median"),
        mean_centerline_length_mm=("centerline_length_mm", "mean"),
        mean_tortuosity=("tortuosity_length_ratio", "mean"),
        mean_branch_angle_deg=("branch_angle_deg", "mean"),
        total_parsed_branch_volume_mm3=("volume_mm3", "sum"),
        mean_radius_mm_per_L_cuberoot=("radius_mean_mm_per_L_cuberoot", "mean"),
        mean_diameter_mm_per_L_cuberoot=("diameter_mean_mm_per_L_cuberoot", "mean"),
        mean_centerline_length_mm_per_L_cuberoot=("centerline_length_mm_per_L_cuberoot", "mean"),
    )

    metrics = [
        "branch_count", "mean_radius_mm", "median_radius_mm", "mean_diameter_mm",
        "median_diameter_mm", "mean_centerline_length_mm", "mean_tortuosity",
        "mean_branch_angle_deg", "total_parsed_branch_volume_mm3",
        "mean_radius_mm_per_L_cuberoot", "mean_diameter_mm_per_L_cuberoot",
        "mean_centerline_length_mm_per_L_cuberoot",
    ]
    roster = data[["age_bucket", "case_id", "subject"]].drop_duplicates()
    rows = []
    for bucket in BUCKETS:
        bucket_data = per_case[per_case.age_bucket == bucket]
        bucket_cases = roster[roster.age_bucket == bucket]
        for generation in range(1, MAX_GENERATION + 1):
            group = bucket_data[bucket_data.generation == generation]
            # Scans without this generation contribute zero branches, otherwise
            # sparse deep generations would look better detected than they are.
            counts = bucket_cases.merge(
                group[["case_id", "branch_count"]], on="case_id", how="left"
            ).branch_count.fillna(0)
            count_stat = describe(counts)
            row = {
                "age_group": DISPLAY[bucket],
                "age_bucket_code": bucket,
                "AirMorph_generation": generation,
                "n_scans_in_age_group": int(bucket_cases.case_id.nunique()),
                "n_scans_with_generation": int(group.case_id.nunique()),
                "total_branches": int(group.branch_count.sum()),
                "branch_count_all_scans_mean": count_stat["mean"],
                "branch_count_all_scans_sd": count_stat["sd"],
                "branch_count_all_scans_median": count_stat["median"],
                "branch_count_all_scans_q1": count_stat["q1"],
                "branch_count_all_scans_q3": count_stat["q3"],
            }
            for metric in metrics:
                stat = describe(group[metric])
                for name in ("mean", "sd", "median", "q1", "q3", "iqr", "min", "max"):
                    row[f"{metric}_{name}"] = stat[name]
            rows.append(row)
    return data, per_case, pd.DataFrame(rows)


# ------------------------------------------------------------------- figures


def age_axis_stats(case: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for bucket in BUCKETS:
        group = case[case.age_bucket == bucket]
        row = {"bucket": bucket, "label": DISPLAY[bucket], "x": float(group.age_years.median())}
        for metric in metrics:
            stat = describe(group[metric])
            row[f"{metric}_mean"] = stat["mean"]
            row[f"{metric}_sd"] = stat["sd"]
            row[f"{metric}_median"] = stat["median"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_generation_dimensions(summary: pd.DataFrame, out: Path, normalized: bool = False):
    if normalized:
        specs = [
            ("mean_radius_mm_per_L_cuberoot", "Radius / lung scale"),
            ("mean_diameter_mm_per_L_cuberoot", "Diameter / lung scale"),
        ]
        title = "Lung-volume-normalized airway dimensions by age group and generation"
    else:
        specs = [
            ("mean_radius_mm", "Radius (mm)"),
            ("mean_diameter_mm", "Diameter (mm)"),
        ]
        title = "Airway dimensions by age group and AirMorph generation"

    # The 1-unit y ticks need more physical separation than the original
    # 2-unit scale, so these two generation figures use a taller canvas.
    fig, axes = plt.subplots(2, 1, figsize=(12, 16), sharex=True)
    for ax, (metric, ylabel) in zip(axes, specs):
        for bucket in BUCKETS:
            group = summary[summary.age_bucket_code == bucket].sort_values("AirMorph_generation")
            ax.errorbar(
                group.AirMorph_generation, group[f"{metric}_median"],
                yerr=group[f"{metric}_sd"].fillna(0),
                label=f"{DISPLAY[bucket]} years", markersize=5.5, linewidth=1.8,
                elinewidth=0.9, capsize=2.5, alpha=0.95, **STYLE[bucket],
            )
        ax.set_ylabel(ylabel + "\nMedian ± SD")
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Generation")
    axes[-1].set_xticks(np.arange(1, MAX_GENERATION + 1))
    axes[-1].set_xlim(0.5, MAX_GENERATION + 0.5)
    axes[0].legend(ncol=3, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_generation_counts(summary: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(12, 8))
    for bucket in BUCKETS:
        group = summary[summary.age_bucket_code == bucket].sort_values("AirMorph_generation")
        ax.errorbar(
            group.AirMorph_generation, group.branch_count_all_scans_mean,
            yerr=group.branch_count_all_scans_sd.fillna(0),
            label=f"{DISPLAY[bucket]} years", markersize=5.5, linewidth=1.8,
            elinewidth=0.9, capsize=2.5, **STYLE[bucket],
        )
    ax.set_xlabel("Generation", **PAGE_MATCHED_AXIS_LABEL)
    ax.set_ylabel("Branches per scan\nMean ± SD", **PAGE_MATCHED_AXIS_LABEL)
    ax.set_xticks(np.arange(1, MAX_GENERATION + 1))
    ax.set_xlim(0.5, MAX_GENERATION + 0.5)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, loc="upper left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_central(case: pd.DataFrame, out: Path, normalized: bool = False):
    metrics = [f"{m}_per_L_cuberoot" for m in CENTRAL_METRICS] if normalized else CENTRAL_METRICS
    stats = age_axis_stats(case, metrics)
    unit = " (mm/L$^{1/3}$)" if normalized else " (mm)"

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    axes[0].errorbar(
        stats.x, stats[f"{metrics[0]}_mean"], yerr=stats[f"{metrics[0]}_sd"],
        color="#1f77b4", marker="o", linewidth=2.2, capsize=4, markersize=7,
    )
    axes[0].set_ylabel("Tracheal length" + unit + "\nMean ± SD", **PAGE_MATCHED_AXIS_LABEL)

    for metric, label, color, marker in zip(
        metrics[1:], ("Trachea", "Left main bronchus", "Right main bronchus"),
        ("#d62728", "#2ca02c", "#9467bd"), ("s", "^", "D"),
    ):
        axes[1].errorbar(
            stats.x, stats[f"{metric}_mean"], yerr=stats[f"{metric}_sd"],
            color=color, marker=marker, label=label, linewidth=2.2, capsize=4, markersize=7,
        )
    axes[1].set_ylabel("Airway diameter" + unit + "\nMean ± SD", **PAGE_MATCHED_AXIS_LABEL)
    axes[1].set_xlabel("Age (years)", **PAGE_MATCHED_AXIS_LABEL)
    axes[1].legend()
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_sections(case: pd.DataFrame, out: Path, normalized: bool = False):
    metrics = [f"{m}_per_L_cuberoot" for m in SECTION_METRICS] if normalized else SECTION_METRICS
    stats = age_axis_stats(case, metrics)
    fig, ax = plt.subplots(figsize=(12, 8))
    for metric, label, color, marker in zip(
        metrics, ("Distal (carina side)", "Middle", "Proximal (inlet side)"),
        ("#1b9e77", "#7570b3", "#d95f02"), ("o", "s", "^"),
    ):
        ax.errorbar(
            stats.x, stats[f"{metric}_mean"], yerr=stats[f"{metric}_sd"],
            color=color, marker=marker, label=label, linewidth=2.2, capsize=4, markersize=7,
        )
    ax.set_xlabel("Age (years)", **PAGE_MATCHED_AXIS_LABEL)
    ax.set_ylabel(
        "Tracheal diameter"
        + (" (mm/L$^{1/3}$)" if normalized else " (mm)")
        + "\nMean ± SD",
        **PAGE_MATCHED_AXIS_LABEL,
    )
    title = "Tracheal diameter by intrathoracic section and age group"
    if normalized:
        title += " after lung-volume normalization"
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_lung_volume(case: pd.DataFrame, out: Path):
    stats = age_axis_stats(case, ["total_segmented_lung_volume_mL"])
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.errorbar(
        stats.x, stats["total_segmented_lung_volume_mL_mean"] / 1000.0,
        yerr=stats["total_segmented_lung_volume_mL_sd"] / 1000.0,
        marker="o", linewidth=2.2, capsize=4, color="#1f77b4", markersize=7,
    )
    ax.set_xlabel("Age (years)", **PAGE_MATCHED_AXIS_LABEL)
    ax.set_ylabel("Lung volume (L)\nMean ± SD", **PAGE_MATCHED_AXIS_LABEL)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_axes(case: pd.DataFrame, out: Path):
    """Short and long calliper axes side by side, with the mean the tables use."""
    stats = age_axis_stats(case, AXIS_METRICS + ["trachea_mean_axis_diameter_mm"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharex=True)
    specs = [
        ("Trachea", "trachea"), ("Left main bronchus", "lmb"), ("Right main bronchus", "rmb"),
    ]
    for ax, (title, prefix) in zip(axes, specs):
        for suffix, label, color, marker in (
            ("long_axis_mm", "Long axis", "#d62728", "^"),
            ("short_axis_mm", "Short axis", "#1f77b4", "v"),
        ):
            metric = f"{prefix}_{suffix}"
            ax.errorbar(
                stats.x, stats[f"{metric}_mean"], yerr=stats[f"{metric}_sd"],
                color=color, marker=marker, label=label, linewidth=2, capsize=3.5, markersize=6,
            )
        ax.fill_between(
            stats.x, stats[f"{prefix}_short_axis_mm_mean"], stats[f"{prefix}_long_axis_mm_mean"],
            color="#999999", alpha=0.18, label="Range averaged into the reported diameter",
        )
        ax.set_title(title)
        ax.set_xlabel("Median age within age group (years)")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Calliper width (mm), mean ± SD")
    axes[0].legend(fontsize=9)
    fig.suptitle(
        "Shortest and longest perpendicular calliper widths. The reported diameter is their mean.",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_convention_comparison(case: pd.DataFrame, out: Path):
    """Show how much the two convention changes move the reported numbers."""
    stats = age_axis_stats(
        case,
        ["full_segmented_tracheal_length_mm", "intrathoracic_tracheal_length_mm",
         "tracheal_edt_mean_diameter_mm", "trachea_mean_axis_diameter_mm"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].errorbar(
        stats.x, stats["full_segmented_tracheal_length_mm_mean"],
        yerr=stats["full_segmented_tracheal_length_mm_sd"],
        color="#d62728", marker="^", linestyle="--", linewidth=2, capsize=4,
        markersize=7, label="Full segmented root (previous)",
    )
    axes[0].errorbar(
        stats.x, stats["intrathoracic_tracheal_length_mm_mean"],
        yerr=stats["intrathoracic_tracheal_length_mm_sd"],
        color="#1f77b4", marker="o", linewidth=2, capsize=4, markersize=7,
        label="Carina to lung-apex inlet proxy (current)",
    )
    axes[0].set_ylabel("Tracheal length (mm), mean ± SD")
    axes[0].set_title("Tracheal length definition")

    axes[1].errorbar(
        stats.x, stats["tracheal_edt_mean_diameter_mm_mean"],
        yerr=stats["tracheal_edt_mean_diameter_mm_sd"],
        color="#d62728", marker="^", linestyle="--", linewidth=2, capsize=4,
        markersize=7, label="Twice inscribed EDT radius (previous)",
    )
    axes[1].errorbar(
        stats.x, stats["trachea_mean_axis_diameter_mm_mean"],
        yerr=stats["trachea_mean_axis_diameter_mm_sd"],
        color="#1f77b4", marker="o", linewidth=2, capsize=4, markersize=7,
        label="Mean of short and long axes (current)",
    )
    axes[1].set_ylabel("Tracheal diameter (mm), mean ± SD")
    axes[1].set_title("Tracheal diameter definition")

    for ax in axes:
        ax.set_xlabel("Median age within age group (years)")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9)
    fig.suptitle("Effect of the two measurement-convention changes", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sex_distribution(case: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(12, 8))
    counts = []
    for bucket in BUCKETS:
        sex = case[case.age_bucket == bucket].sex.fillna("Unknown").str.title()
        counts.append([int(sex.eq(label).sum()) for label in ("Male", "Female", "Unknown")])
    counts = np.asarray(counts)
    bottom = np.zeros(len(BUCKETS))
    for index, (label, color) in enumerate(
        (("Male", "#1f77b4"), ("Female", "#d62728"), ("Unknown", "#bdbdbd"))
    ):
        ax.bar(
            [DISPLAY[b] for b in BUCKETS], counts[:, index], bottom=bottom,
            color=color, label=label, edgecolor="white", linewidth=0.8,
        )
        for x, (value, base) in enumerate(zip(counts[:, index], bottom)):
            if value:
                ax.text(x, base + value / 2.0, str(value), ha="center", va="center",
                        fontsize=9, color="white" if label != "Unknown" else "#333333")
        bottom += counts[:, index]
    ax.set_xlabel("Age group (years)", **PAGE_MATCHED_AXIS_LABEL)
    ax.set_ylabel("CT scans", **PAGE_MATCHED_AXIS_LABEL)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------ workbook


def style_workbook(path: Path):
    from openpyxl import load_workbook

    workbook = load_workbook(path)
    thin = Side(style="thin", color="A6A6A6")
    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        sheet.sheet_view.showGridLines = False
        for cell in sheet[1]:
            cell.fill = PatternFill("solid", fgColor="17365D")
            cell.font = Font(color="FFFFFF", bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = Border(bottom=thin)
        sheet.row_dimensions[1].height = 42
        for column_cells in sheet.columns:
            values = [len(str(cell.value or "")) for cell in column_cells[:150]]
            width = min(44, max(11, max(values, default=10) + 2))
            sheet.column_dimensions[get_column_letter(column_cells[0].column)].width = width
        if sheet.title.startswith("Table"):
            for row in range(2, sheet.max_row + 1):
                if str(sheet.cell(row, 1).value) == "All":
                    for cell in sheet[row]:
                        cell.fill = PatternFill("solid", fgColor="DCE6F1")
                        cell.font = Font(bold=True)
    workbook.save(path)


# ---------------------------------------------------------------------- main


def build_case_table() -> pd.DataFrame:
    cohort = pd.read_csv(RUN / "cohort_90.csv")
    measurements = pd.read_csv(RUN / "cross_section_measurements.csv")
    case = cohort.merge(
        measurements.drop(columns=["measurement_status"], errors="ignore"),
        on="case_id", validate="one_to_one",
    )
    if len(case) != 90:
        raise SystemExit(f"expected 90 measured scans, found {len(case)}")

    case["lung_linear_scale_L_cuberoot"] = np.cbrt(case.total_segmented_lung_volume_mL / 1000.0)
    for metric in CENTRAL_METRICS + SECTION_METRICS:
        case[f"{metric}_per_L_cuberoot"] = case[metric] / case.lung_linear_scale_L_cuberoot

    # Validate the central-airway topology and remove impossible measurements
    # before anything is summarized.
    findings = topology_qc.check_cohort(
        [
            BASE / "or_all3_run/normal_airway_branch_morphometry.csv",
            BASE / "E3_PEDIATRIC_AIRWAY_RESULTS/normal_airway_branch_morphometry.csv",
        ],
        case,
    )
    case = case.merge(findings, on="case_id", validate="one_to_one")
    pre_qc_values = case.copy()
    case = topology_qc.apply_plausibility(case)
    # Requested inclusive model-output analysis: QC findings are documented,
    # while every available automated measurement remains in the summaries.
    for column in pre_qc_values.columns:
        if column.endswith("_mm") or column.endswith("_ratio"):
            case[column] = pre_qc_values[column]
    # Recompute normalized values after QC so excluded bronchial measurements
    # cannot survive in the normalized tables through an earlier copy.
    for metric in CENTRAL_METRICS + SECTION_METRICS:
        case[f"{metric}_per_L_cuberoot"] = case[metric] / case.lung_linear_scale_L_cuberoot
    flagged = case[
        case.fragmented_main_bronchus | case.detached_stump
        | case.lmb_diameter_excluded | case.rmb_diameter_excluded
    ]
    if len(flagged):
        print(f"topology QC flagged {len(flagged)} scan(s):")
        for row in flagged.itertuples():
            print(f"  {row.case_id}: {row.topology_note or row.exclusion_reason}")
    findings.to_csv(RUN / "topology_qc.csv", index=False)

    case["tracheal_length_reduction_percent"] = (
        100.0 * case.cervical_tracheal_length_mm / case.full_segmented_tracheal_length_mm
    )
    # The previous convention, recomputed from the live branch table so the
    # comparison reflects the current segmentation of every case.
    case["tracheal_edt_mean_diameter_mm"] = case.raw_root_edt_mean_diameter_mm
    case["tracheal_diameter_increase_percent"] = 100.0 * (
        case.trachea_mean_axis_diameter_mm / case.tracheal_edt_mean_diameter_mm - 1.0
    )
    return case.sort_values(["age_years", "case_id"]).reset_index(drop=True)


def main() -> None:
    for directory in (OUT, CSV_DIR, FIG_DIR, OUT / "supporting_files"):
        directory.mkdir(parents=True, exist_ok=True)

    case = build_case_table()
    # Branch-level morphometry comes from the two source cohorts and is then
    # restricted to the selected scans, so a newly selected case is never missing.
    branches = pd.concat(
        [
            pd.read_csv(
                path, low_memory=False,
                dtype={"branch_instance_id": "Int64", "parent_instance_id": "Int64"},
            )
            for path in (
                BASE / "or_all3_run/normal_airway_branch_morphometry.csv",
                BASE / "E3_PEDIATRIC_AIRWAY_RESULTS/normal_airway_branch_morphometry.csv",
            )
        ],
        ignore_index=True,
    )
    branches = branches[branches.case_id.isin(set(case.case_id))].copy()
    if branches.case_id.nunique() != len(case):
        missing = sorted(set(case.case_id) - set(branches.case_id))
        raise SystemExit(f"no branch morphometry for: {missing}")
    branch_data, per_case_generation, generation_summary = generation_tables(branches, case)

    tables = {
        "Table 1 Dataset Overview": table_dataset_overview(case),
        "Table 2 Central": table_central(case),
        "Table 3 Tracheal Sections": table_sections(case),
        "Table 4 Short Long Axes": table_axes(case),
        "Table 5 Central Normalized": table_central(case, normalized=True),
        "Table 6 Sections Normalized": table_sections(case, normalized=True),
        "Table 7 Lung Volume": table_lung_volume(case),
        "Table 8 Sex Distribution": table_sex_distribution(case),
        "Table 9 Convention Change": table_convention_comparison(case),
        "Table 10 Excluded Measurements": table_exclusions(case),
    }
    filenames = {
        "Table 1 Dataset Overview": "table_1_dataset_overview.csv",
        "Table 2 Central": "table_2_central_airway_by_age.csv",
        "Table 3 Tracheal Sections": "table_3_tracheal_sections_by_age.csv",
        "Table 4 Short Long Axes": "table_4_short_long_axes.csv",
        "Table 5 Central Normalized": "table_5_central_lung_normalized.csv",
        "Table 6 Sections Normalized": "table_6_sections_lung_normalized.csv",
        "Table 7 Lung Volume": "table_7_lung_volume.csv",
        "Table 8 Sex Distribution": "table_8_sex_distribution.csv",
        "Table 9 Convention Change": "table_9_convention_comparison.csv",
        "Table 10 Excluded Measurements": "table_10_excluded_measurements.csv",
    }
    for name, frame in tables.items():
        frame.to_csv(CSV_DIR / filenames[name], index=False)

    complete_metrics = [
        "age_years", "total_segmented_lung_volume_mL",
        "full_segmented_tracheal_length_mm", "cervical_tracheal_length_mm",
        *CENTRAL_METRICS, *SECTION_METRICS, *AXIS_METRICS,
        "trachea_eccentricity_ratio", "lmb_eccentricity_ratio", "rmb_eccentricity_ratio",
        "trachea_area_equivalent_diameter_mm",
        "trachea_protocol10mm_mean_axis_diameter_mm",
        "lmb_protocol5mm_mean_axis_diameter_mm",
        "rmb_protocol5mm_mean_axis_diameter_mm",
        "tracheal_edt_mean_diameter_mm", "lmb_edt_mean_diameter_mm", "rmb_edt_mean_diameter_mm",
        "trachea_cross_sections", "lmb_cross_sections", "rmb_cross_sections",
    ]
    complete = grouped_descriptive(case, complete_metrics)
    normalized_complete = grouped_descriptive(
        case, [f"{m}_per_L_cuberoot" for m in CENTRAL_METRICS + SECTION_METRICS]
    )

    complete.to_csv(CSV_DIR / "central_airway_complete_statistics.csv", index=False)
    normalized_complete.to_csv(CSV_DIR / "lung_normalized_complete_statistics.csv", index=False)
    case.drop(columns=["case_dir"]).to_csv(
        CSV_DIR / "case_level_measurements.csv", index=False
    )
    per_case_generation.to_csv(CSV_DIR / "per_scan_generation_metrics.csv", index=False)
    generation_summary.to_csv(CSV_DIR / "age_group_generation_statistics.csv", index=False)
    branch_data.to_csv(CSV_DIR / "branch_level_generation_1_to_14.csv", index=False)

    base_columns = [
        "age_group", "age_bucket_code", "AirMorph_generation",
        "n_scans_in_age_group", "n_scans_with_generation", "total_branches",
    ]
    stat_names = ("mean", "sd", "median", "q1", "q3", "iqr", "min", "max")
    generation_radius = generation_summary[
        base_columns + [f"mean_radius_mm_{s}" for s in stat_names]
    ]
    generation_diameter = generation_summary[
        base_columns + [f"mean_diameter_mm_{s}" for s in stat_names]
    ]
    generation_counts = generation_summary[
        base_columns + [f"branch_count_all_scans_{s}" for s in ("mean", "sd", "median", "q1", "q3")]
    ]

    methods = pd.DataFrame(
        [
            ("Cohort", "90 CT scans from 90 unique patient identifiers, 10 scans in each of nine 2-year age groups."),
            ("Tracheal length", "Physical centerline path from the carina to the thoracic-inlet proxy. The proxy is the most superior segmented lung voxel; centerline above that level is treated as cervical trachea and excluded."),
            ("Tracheal diameter", "Planes perpendicular to the local centerline tangent are resampled from the fused airway mask by trilinear interpolation at 0.25 mm. The lumen contour's shortest and longest calliper widths are measured and averaged, following Kuo et al."),
            ("Tracheal sampling", "Every 2 mm from 5 mm above the carina to 2 mm below the inlet proxy. A 10-mm subset reproducing the source paper's interval is reported separately in the complete statistics."),
            ("Main bronchi", "Same cross-sectional method, sampled every 1 mm between 2 mm margins. A 5-mm protocol subset is reported separately."),
            ("Tracheal sections", "The intrathoracic tracheal centerline is split into equal distal, middle, and proximal thirds, with the distal third at the carina."),
            ("Previous convention retained for comparison", "Twice the Euclidean distance-transform radius on the centerline, and the full segmented root length. Both appear in the convention-comparison table."),
            ("Generation analysis", f"AirMorph topologic generation with the tracheal root as generation 1, reported to generation {MAX_GENERATION} only. Deeper generations are excluded from every table, figure, and exported branch file."),
            ("Generation dimensions", "Generation-level radius and diameter remain EDT-derived. Cross-sectional calliper fitting is unreliable for peripheral airways whose lumen approaches the voxel size, so the stabler inscribed-radius measure is used there."),
            ("Measurement accuracy", "On elliptical-cylinder phantoms at 0.3-0.5 mm voxel spacing, short-axis error was within 0.36 mm, long-axis error within 0.50 mm, and mean-axis error within 0.43 mm."),
            ("Topology and plausibility checks", "The trachea must have two substantive children, one per side. A main bronchus split across fragments is measured only if the dominant fragment was selected. Any main-bronchus diameter exceeding its own tracheal diameter is anatomically impossible and is removed, because it indicates the cross-section sampled the carinal confluence. Removed values are listed in the excluded-measurements table and are never imputed."),
            ("Main bronchus sampling", "Cross-sections start at 30 percent of the bronchial length from the carina, with a 3 mm floor, so the carinal confluence is not sampled. The distal margin is 10 percent of the length with a 2 mm floor."),
            ("Sex", "Resolved from the source cohort records through the anonymization mapping, retaining only anonymized identifiers and the sex value. Unresolved scans are reported as Unknown and were not imputed."),
            ("Lung volume", "Nonzero voxels of AirMorph's lunglobe.nii.gz, produced by the embedded LungMask U-Net R231 model. It is a left/right whole-lung mask, not a five-lobe mask."),
            ("Normalization", "Each scan's measurement divided by the cube root of its total segmented lung volume in liters, before age-group summaries."),
            ("Not reported", "Outer bronchial diameter, wall thickness, bronchus-artery ratios, contrast status, and a manually identified thoracic-inlet landmark are unavailable and were not imputed."),
        ],
        columns=["Item", "Definition"],
    )

    qc = pd.DataFrame(
        [
            ("CT scans", len(case)),
            ("Unique patient identifiers", case.subject.nunique()),
            ("Age groups with exactly 10 scans", int(case.groupby("age_bucket").size().eq(10).sum())),
            ("Scans from the original cohort", int(case.source_cohort.eq("original_83").sum())),
            ("Scans from the E3 cohort", int(case.source_cohort.eq("E3").sum())),
            ("Scans with sex recorded", int(case.sex.isin(["Male", "Female"]).sum())),
            ("Thoracic-inlet clip applied", int(case.thoracic_inlet_clip_applied.sum())),
            ("Minimum tracheal cross-sections in one scan", int(case.trachea_cross_sections.min())),
            ("Median tracheal cross-sections per scan", float(case.trachea_cross_sections.median())),
            ("Minimum LMB cross-sections in one scan", int(case.lmb_cross_sections.min())),
            ("Minimum RMB cross-sections in one scan", int(case.rmb_cross_sections.min())),
            ("Scans missing a tracheal diameter", int(case.trachea_mean_axis_diameter_mm.isna().sum())),
            ("Scans missing an LMB diameter", int(case.lmb_mean_axis_diameter_mm.isna().sum())),
            ("Scans missing an RMB diameter", int(case.rmb_mean_axis_diameter_mm.isna().sum())),
            ("Scans with a tracheal diameter", int(case.trachea_mean_axis_diameter_mm.notna().sum())),
            ("LMB diameters removed by QC", int(case.lmb_diameter_excluded.sum())),
            ("RMB diameters removed by QC", int(case.rmb_diameter_excluded.sum())),
            ("Scans with a fragmented main bronchus", int(case.fragmented_main_bronchus.sum())),
            ("Scans with a detached tracheal stump", int(case.detached_stump.sum())),
            ("Maximum reported generation", int(branch_data.generation.max())),
            ("Branch rows at generations 1-14", len(branch_data)),
            ("Main bronchi assigned by spatial fallback", int(case.main_bronchus_assignment_method.eq("physical-x-fallback").sum())),
            ("Median tracheal length reduction from inlet clipping (%)", round(float(case.tracheal_length_reduction_percent.median()), 1)),
            ("Median tracheal diameter increase from axis averaging (%)", round(float(case.tracheal_diameter_increase_percent.median()), 1)),
        ],
        columns=["QC check", "Result"],
    )

    workbook = OUT / "airway_results_90.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for name, frame in tables.items():
            frame.to_excel(writer, sheet_name=name, index=False)
        complete.to_excel(writer, sheet_name="Central Complete Stats", index=False)
        normalized_complete.to_excel(writer, sheet_name="Normalized Complete Stats", index=False)
        generation_radius.to_excel(writer, sheet_name="Generation Radius", index=False)
        generation_diameter.to_excel(writer, sheet_name="Generation Diameter", index=False)
        generation_counts.to_excel(writer, sheet_name="Generation Counts", index=False)
        generation_summary.to_excel(writer, sheet_name="Generation Complete Stats", index=False)
        per_case_generation.to_excel(writer, sheet_name="Per Scan Generation", index=False)
        case.drop(columns=["case_dir"]).to_excel(writer, sheet_name="Case Level", index=False)
        methods.to_excel(writer, sheet_name="Methods", index=False)
        qc.to_excel(writer, sheet_name="QC", index=False)
    style_workbook(workbook)

    plot_generation_dimensions(generation_summary, FIG_DIR / "01_generation_dimensions.png")
    plot_generation_counts(generation_summary, FIG_DIR / "02_generation_branch_counts.png")
    plot_generation_dimensions(
        generation_summary, FIG_DIR / "03_generation_dimensions_normalized.png", normalized=True
    )
    plot_central(case, FIG_DIR / "04_central_airway_dimensions.png")
    plot_central(case, FIG_DIR / "05_central_airway_normalized.png", normalized=True)
    plot_sections(case, FIG_DIR / "06_tracheal_sections.png")
    plot_sections(case, FIG_DIR / "07_tracheal_sections_normalized.png", normalized=True)
    plot_lung_volume(case, FIG_DIR / "08_total_segmented_lung_volume.png")
    plot_axes(case, FIG_DIR / "09_short_and_long_axes.png")
    plot_convention_comparison(case, FIG_DIR / "10_measurement_convention_change.png")
    plot_sex_distribution(case, FIG_DIR / "11_sex_distribution.png")

    for source in (
        RUN / "cohort_selection_90.csv",
        RUN / "cohort_selection_report.csv",
    ):
        if source.exists():
            shutil.copy2(source, OUT / "supporting_files" / source.name)

    print(f"wrote workbook: {workbook}")
    print(f"wrote {len(list(FIG_DIR.glob('*.png')))} figures")
    print(f"cohort: {len(case)} scans, {case.subject.nunique()} identifiers")


if __name__ == "__main__":
    main()
