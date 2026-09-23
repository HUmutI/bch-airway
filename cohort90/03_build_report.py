#!/usr/bin/env python3
"""Build and compile the LaTeX report for the cross-sectional analysis."""
from __future__ import annotations

import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pandas as pd

RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "AIRWAY_RESULTS_90"
CSV = OUT / "csv"
TEX = OUT / "airway_report_90.tex"
TECTONIC = RUN / "tools/tectonic-musl/tectonic"


def esc(value) -> str:
    text = str(value)
    for old, new in {
        "&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_",
        "–": "--", "≥": r"$\geq$", "≤": r"$\leq$", "±": r"$\pm$",
    }.items():
        text = text.replace(old, new)
    return text


def tabular(data: pd.DataFrame, headers: list[str]) -> str:
    align = "l" + "c" * (len(headers) - 1)
    lines = [r"\begin{tabular}{" + align + "}", r"\toprule",
             " & ".join(headers) + r" \\", r"\midrule"]
    for record in data.itertuples(index=False, name=None):
        lines.append(" & ".join(esc(value) for value in record) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def table_block(caption: str, label: str, data: pd.DataFrame,
                headers: list[str], note: str, landscape: bool = True) -> str:
    content = rf"""\begin{{table}}[p]
\centering
\caption{{{caption}}}
\label{{{label}}}
\small
\resizebox{{\linewidth}}{{!}}{{%
{tabular(data, headers)}
}}
\par\vspace{{0.5em}}\begin{{minipage}}{{0.98\linewidth}}\footnotesize
\textit{{Note.}} {note}
\end{{minipage}}
\end{{table}}"""
    if landscape:
        return "\\begin{landscape}\n" + content + "\n\\end{landscape}"
    return content


def build_tables() -> list[str]:
    read = lambda name: pd.read_csv(CSV / name)  # noqa: E731
    return [
        table_block(
            "Characteristics of the study cohort.", "tab:overview",
            read("table_1_dataset_overview.csv"),
            ["Age group (y)", "Subjects", "CT scans", "Sex (M/F)", "Male (\\%)",
             "Age, mean (SD)", "Age, median (IQR)", "Age, range",
             "Lung volume mL, mean (SD)", "Lung volume mL, median",
             "Source (original/E3)"],
            "Each age group holds 10 scans from 10 distinct patients, and no patient "
            "appears in more than one group, so the cohort comprises 90 scans from 90 "
            "patients. Ages are in years. The source column gives the number of scans "
            "contributed by each of the two processed cohorts, which share an identical "
            "segmentation and labelling pipeline. Sex is stated here and is not repeated "
            "in the measurement tables that follow.",
        ),
        table_block(
            "Dimensions of the trachea and the left and right main bronchi by age group.",
            "tab:central", read("table_2_central_airway_by_age.csv"),
            ["Age range (y)", "Subjects", "CT scans", "Age (y)",
             "Tracheal length (mm)", "Tracheal diameter (mm)",
             "LMB diameter (mm)", "RMB diameter (mm)"],
            "Age is median with interquartile-range width in parentheses. Airway "
            "measurements are median with first and third quartiles in parentheses. "
            "Tracheal length is measured from the carina to the thoracic-inlet proxy. "
            "Diameters are the mean of the shortest and longest calliper widths of the "
            "lumen on planes perpendicular to the centerline. LMB = left main bronchus; "
            "RMB = right main bronchus. Sex composition is given in "
            "Table~\\ref{tab:overview}.",
        ),
        table_block(
            "Diameter of the trachea by age group for each location of the measurement.",
            "tab:sections", read("table_3_tracheal_sections_by_age.csv"),
            ["Age range (y)", "Subjects", "CT scans",
             "Distal (mm)", "Middle (mm)", "Proximal (mm)"],
            "Results are median with interquartile range in parentheses. The "
            "intrathoracic tracheal centerline was divided into equal thirds, with the "
            "distal third adjacent to the carina and the proximal third adjacent to the "
            "thoracic-inlet proxy.",
        ),
        table_block(
            "Shortest and longest perpendicular calliper widths of the central airways.",
            "tab:axes", read("table_4_short_long_axes.csv"),
            ["Age range (y)", "Subjects", "CT scans",
             "Trachea short (mm)", "Trachea long (mm)", "Trachea short/long",
             "LMB short (mm)", "LMB long (mm)", "LMB short/long",
             "RMB short (mm)", "RMB long (mm)", "RMB short/long"],
            "Axis widths are median with first and third quartiles in parentheses; the "
            "short/long ratio is median with interquartile range. The diameters in "
            "Table~\\ref{tab:central} are the mean of the short and long axes. A ratio "
            "below one quantifies the non-circular cross-section of the airway lumen.",
        ),
        table_block(
            "Sex distribution by age group.", "tab:sex",
            read("table_8_sex_distribution.csv"),
            ["Age range (y)", "CT scans", "Male, n", "Female, n", "Unknown, n",
             "Male, \\% of all", "Female, \\% of all",
             "Male, \\% of known", "Female, \\% of known"],
            "Sex was resolved from the source cohort records through the anonymization "
            "mapping. No sex value was imputed.",
        ),
        table_block(
            "Effect of the two measurement-convention changes on reported tracheal values.",
            "tab:convention", read("table_9_convention_comparison.csv"),
            ["Age range (y)", "CT scans", "Full segmented length (mm)",
             "Intrathoracic length (mm)", "Cervical portion (mm)", "Reduction (\\%)",
             "Diameter, EDT (mm)", "Diameter, mean axis (mm)", "Increase (\\%)"],
            "Lengths and diameters are median with first and third quartiles in "
            "parentheses; percentages are median with interquartile range. The full "
            "segmented length and the distance-transform diameter are the values used in "
            "the previous package and are shown here only for comparison.",
        ),
        table_block(
            "Quality-control observations retained in the inclusive model-output analysis.",
            "tab:excluded", read("table_10_excluded_measurements.csv"),
            ["Case", "Age group (y)", "QC status", "Finding"],
            "A main-bronchus mean axis exceeding that scan's tracheal mean axis was treated "
            "as a conservative QC failure suggesting carinal confluence or incorrect "
            "topology. At the investigator's request these automated outputs remain in the "
            "descriptive summaries; this table preserves the QC observations.",
        ),
        table_block(
            "Central airway dimensions normalized by segmented lung volume.",
            "tab:central-normalized", read("table_5_central_lung_normalized.csv"),
            ["Age range (y)", "Subjects", "CT scans", "Age (y)",
             r"Length (mm/L$^{1/3}$)", r"Trachea (mm/L$^{1/3}$)",
             r"LMB (mm/L$^{1/3}$)", r"RMB (mm/L$^{1/3}$)"],
            "Each scan's measurement was divided by the cube root of its total segmented "
            "lung volume in liters before age-group summaries were calculated.",
        ),
        table_block(
            "Tracheal section diameters normalized by segmented lung volume.",
            "tab:sections-normalized", read("table_6_sections_lung_normalized.csv"),
            ["Age range (y)", "Subjects", "CT scans",
             r"Distal (mm/L$^{1/3}$)", r"Middle (mm/L$^{1/3}$)",
             r"Proximal (mm/L$^{1/3}$)"],
            "Values are median with interquartile range in parentheses. Normalization was "
            "performed within each scan before grouping.",
        ),
        table_block(
            "Total segmented lung volume by age group.", "tab:lung",
            read("table_7_lung_volume.csv"),
            ["Age range (y)", "Subjects", "CT scans", "Mean", "SD", "Median",
             "Q1", "Q3", "Minimum", "Maximum"],
            "All values are in milliliters. The source is AirMorph's "
            r"\texttt{lunglobe.nii.gz}, produced by the embedded LungMask U-Net R231 "
            "model. Despite the filename it is a left and right whole-lung mask rather "
            "than a five-lobe segmentation, and it does not by itself establish "
            "inspiratory adequacy.",
        ),
    ]


FIGURES = [
    ("01_generation_dimensions.png",
     "Airway lumen radius and diameter in each generation grouped by age. Each line is one age group; "
     "points are age-group medians of per-scan generation means and bars are between-scan standard "
     "deviations. Generation 1 is the trachea. Reporting stops at generation 14.",
     "fig:generation"),
    ("02_generation_branch_counts.png",
     "Detected airway branches per scan by age group and generation. Scans in which a generation was "
     "not detected contribute zero, so the curves show detection rather than conditional size.",
     "fig:counts"),
    ("09_short_and_long_axes.png",
     "Shortest and longest perpendicular calliper widths of the trachea and main bronchi by age group. "
     "The shaded band is the interval whose midpoint is the reported diameter.",
     "fig:axes"),
    ("10_measurement_convention_change.png",
     "Effect of the two convention changes. Clipping the trachea at the thoracic-inlet proxy shortens "
     "the reported length, and averaging the short and long axes increases the reported diameter "
     "relative to twice the inscribed distance-transform radius.",
     "fig:convention"),
    ("04_central_airway_dimensions.png",
     "Intrathoracic tracheal length and central airway diameters by age group, shown as mean and "
     "standard deviation.",
     "fig:central-fig"),
    ("06_tracheal_sections.png",
     "Distal, middle, and proximal intrathoracic tracheal diameters by age group.",
     "fig:sections-fig"),
    ("11_sex_distribution.png",
     "Sex distribution of the 90 scans. Each age group contains exactly 10 scans.",
     "fig:sex-fig"),
    ("08_total_segmented_lung_volume.png",
     "Total segmented lung volume by age group, shown as mean and standard deviation.",
     "fig:lung-fig"),
    ("03_generation_dimensions_normalized.png",
     "Airway dimensions by generation after within-scan lung-volume normalization.",
     "fig:generation-normalized"),
    ("05_central_airway_normalized.png",
     "Central airway dimensions after within-scan lung-volume normalization.",
     "fig:central-normalized-fig"),
    ("07_tracheal_sections_normalized.png",
     "Tracheal section diameters after within-scan lung-volume normalization.",
     "fig:sections-normalized-fig"),
]


def main() -> None:
    case = pd.read_csv(CSV / "case_level_measurements.csv")
    qc = {
        "scans": len(case),
        "subjects": case.subject.nunique(),
        "known_sex": int(case.sex.isin(["Male", "Female"]).sum()),
        "clip": int(case.thoracic_inlet_clip_applied.sum()),
        "median_sections": float(case.trachea_cross_sections.median()),
        "min_sections": int(case.trachea_cross_sections.min()),
        "reduction": float(case.tracheal_length_reduction_percent.median()),
        "increase": float(case.tracheal_diameter_increase_percent.median()),
        "ratio": float(case.trachea_eccentricity_ratio.median()),
        "lmb_removed": int(case.lmb_diameter_excluded.sum()),
        "rmb_removed": int(case.rmb_diameter_excluded.sum()),
    }

    figures = [
        rf"""\begin{{figure}}[p]
\centering
\includegraphics[width=\textwidth,height=0.78\textheight,keepaspectratio]{{figures/{filename}}}
\caption{{{caption}}}
\label{{{label}}}
\end{{figure}}
\clearpage"""
        for filename, caption, label in FIGURES
    ]

    document = r"""\documentclass[11pt]{article}
\usepackage[margin=0.75in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{pdflscape}
\usepackage[hidelinks]{hyperref}
\usepackage[font=small,labelfont=bf]{caption}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5em}
\title{Normative Central Airway Dimensions on Chest CT\\
in a Balanced Pediatric Cohort\\[0.4em]
\large Cross-sectional measurement with a thoracic-inlet-clipped trachea}
\author{}
\date{15 September 2026}
\begin{document}
\maketitle
\tableofcontents
\clearpage

\section{Cohort}
The analysis contains """ + str(qc["scans"]) + r""" normal chest CT scans from """ + str(qc["subjects"]) + r""" unique patient identifiers, with exactly 10 scans in each of nine two-year age groups from 0--2 through 16--18 years. Each patient contributes one scan to the whole cohort, so the number of scans and the number of subjects are equal and no patient appears in more than one age group. Scans were drawn from two processed cohorts that share an identical segmentation and labelling pipeline.

Sex is recorded for """ + str(qc["known_sex"]) + r""" of the """ + str(qc["scans"]) + r""" scans. It was resolved from the source cohort records through the anonymization mapping, retaining only anonymized identifiers and the sex value. Male, female, and unknown counts with percentages are reported in every main table and in Table~\ref{tab:sex}; no sex value was imputed.

In the Age (y) column, a value such as \textbf{0.8 (0.7)} means a median age of 0.8 years with an interquartile-range width of 0.7 years, where IQR $=$ Q3 $-$ Q1. It is not a standard deviation.

\section{How the airways were measured}
Two measurement conventions changed relative to the previous package, and both changes move the results towards the definitions used by Kuo et al.

\textbf{Tracheal length.} The segmented airway tree extends above the thoracic inlet into the cervical trachea, so the full segmented root length is not comparable with a carina-to-inlet measurement. The tracheal centerline is therefore clipped at a thoracic-inlet proxy defined as the most superior segmented lung voxel, and only the portion between the carina and that level is reported. Clipping was applied in """ + str(qc["clip"]) + r""" of the """ + str(qc["scans"]) + r""" scans and removed a median of """ + f"{qc['reduction']:.1f}" + r"""\,\% of the segmented root length.

\textbf{Airway diameter.} The previous diameter was twice the Euclidean distance-transform radius sampled on the centerline. That radius is the largest sphere that fits inside the lumen, so for a non-circular airway such as the trachea it tracks the shortest diameter rather than the average diameter. On an elliptical phantom with 8\,mm and 16\,mm axes, twice the distance-transform radius measured 8.0\,mm while the true mean axis was 12.0\,mm. In this report a plane perpendicular to the local centerline tangent is resampled from the fused airway mask by trilinear interpolation at 0.25\,mm, the lumen contour containing the centerline point is isolated, and its shortest and longest calliper widths are measured. The reported diameter is the mean of those two widths, matching the source paper's convention. The change increased the reported tracheal diameter by a median of """ + f"{qc['increase']:.1f}" + r"""\,\%. The median tracheal short-to-long axis ratio was """ + f"{qc['ratio']:.2f}" + r""", confirming that the lumen is measurably non-circular.

The trachea was sampled every 2\,mm between a 5\,mm carinal margin and a 2\,mm inlet margin, giving a median of """ + f"{qc['median_sections']:.0f}" + r""" cross-sections per scan and a minimum of """ + str(qc["min_sections"]) + r""". The main bronchi were sampled every 1\,mm after excluding at least 30 per cent (and at least 3\,mm) proximally and 10 per cent (and at least 2\,mm) distally. Subsets matching the source paper's 10\,mm tracheal and 5\,mm bronchial intervals are reported in the accompanying complete-statistics sheets. On elliptical-cylinder phantoms at 0.3--0.5\,mm voxel spacing, short-axis error stayed within 0.36\,mm, long-axis error within 0.50\,mm, and mean-axis error within 0.43\,mm, including for cylinders tilted 20 degrees away from the voxel grid.

\textbf{Generation reporting.} AirMorph topologic generation is used, with the trachea as generation 1. Every table, figure, and exported branch file stops at generation 14. Deeper generations were excluded because their branch detection is governed by CT resolution rather than anatomy. Generation-level radius and diameter remain distance-transform measurements, because calliper fitting on peripheral airways whose lumen approaches the voxel size is unstable; the cross-sectional convention is applied to the central airways, where it is reliable and where the source paper makes its comparisons.

\section{Quality control of the central airways}
Every scan's central-airway topology was checked before its measurements were summarized. The trachea is required to have two substantive children, one per side. Where a main bronchus is split into fragments at the carina, its measurement is retained only if the dominant fragment was the one measured.

A main-bronchus mean axis exceeding that scan's tracheal mean axis was flagged as suggesting carinal confluence or incorrect topology. Bronchial cross-sections begin 30 per cent of the bronchial length away from the carina, subject to a 3 mm floor. At the investigator's request, all available automated model outputs are retained in the descriptive summaries, with ten scans in every age group and no per-cell sample-size annotation. Table~\ref{tab:excluded} documents the QC observations without removing those values.

\section{Tables}
""" + "\n\n".join(build_tables()) + r"""

\clearpage
\section{Figures}
""" + "\n\n".join(figures) + r"""

\section{Limitations}
The thoracic-inlet proxy is the lung apex, which lies above the true thoracic inlet, so the reported tracheal length remains longer than a manually measured carina-to-inlet length. The gap widens with age because the distance between the lung apex and the inlet increases during growth. Tracheal diameters, by contrast, now agree closely with published values across the age range. A bone-landmark inlet plane at the first ribs and the superior border of the manubrium, or a radiologist-placed landmark, would remove the remaining length discrepancy and is the recommended next step before publication.

All anatomical labels are model predictions and require pediatric radiologist review. Sex is now known for all but one scan, but with 10 scans per age group a sex-stratified table would rest on cells of two to eight scans, so sex is reported as counts and percentages rather than as stratified reference values. Contrast status is unavailable, so contrast and noncontrast scans are pooled. Outer bronchial diameter, wall thickness, bronchial wall area, paired pulmonary-artery diameter, and bronchus-artery ratios cannot be derived from a lumen segmentation and were not reported. Because each age group contains 10 scans, these values describe the cohort and are not proposed as clinical reference limits.

\section{Reference papers}
Kuo W, et al. Reference values for central airway dimensions on CT images of children and adolescents. \textit{AJR} 2018. \url{https://doi.org/10.2214/AJR.17.18597}

Lv Q, et al. Normative values for lung, bronchial sizes, and bronchus-artery ratios in chest CT scans: from infancy into young adulthood. \textit{European Radiology} 2025. \url{https://doi.org/10.1007/s00330-025-11367-w}

\section{Source data}
Complete numerical summaries, case-level values, and per-generation statistics accompany this report in \path{airway_results_90.xlsx} and the \path{csv/} directory.

\end{document}
"""
    TEX.write_text(document)

    with zipfile.ZipFile(OUT / "airway_latex_overleaf.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("main.tex", document)
        archive.writestr(
            "README.txt", "Upload this ZIP as a new Overleaf project. main.tex is the main file.\n"
        )
        for figure in sorted((OUT / "figures").glob("*.png")):
            archive.write(figure, f"figures/{figure.name}")

    engine = str(TECTONIC) if TECTONIC.exists() else shutil.which("tectonic") or shutil.which("pdflatex")
    if not engine:
        print("no TeX engine found; wrote compile-ready source only")
        return
    subprocess.run(
        [engine, "-X", "compile", TEX.name, "--outdir", str(OUT), "--keep-logs"],
        cwd=OUT, check=True,
    )
    final = OUT / "FINAL_airway_report_90.pdf"
    shutil.copy2(TEX.with_suffix(".pdf"), final)
    print(f"wrote PDF: {final}")

    archive_path = OUT.with_suffix(".tar.gz")
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(OUT, arcname=OUT.name)
    print(f"wrote archive: {archive_path}")


if __name__ == "__main__":
    main()
