#!/usr/bin/env python3
"""Build a self-contained LaTeX report from the final meeting package."""

from pathlib import Path
import shutil
import subprocess
import tarfile
import zipfile

import pandas as pd


RUN = Path(__file__).resolve().parents[1]
OUT = RUN / "AIRWAY_RADIOLOGIST_PROF_MEETING_2026-09-11"
CSV = OUT / "csv"
TEX = OUT / "airway_meeting_report.tex"
AGE_ORDER = ["0-3", "3-6", "6-9", "10-12", "12-14", "14-16", "16-18"]


def esc(value) -> str:
    text = str(value)
    replacements = {
        "&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_",
        "–": "--", "≥": r"$\geq$", "≤": r"$\leq$",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def tabular(data: pd.DataFrame, headers: list[str], align: str | None = None) -> str:
    if align is None:
        align = "l" + "c" * (len(headers) - 1)
    rows = [r"\begin{tabular}{" + align + "}", r"\toprule"]
    rows.append(" & ".join(headers) + r" \\")
    rows.append(r"\midrule")
    for record in data.itertuples(index=False, name=None):
        rows.append(" & ".join(esc(value) for value in record) + r" \\")
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(rows)


def table_block(caption: str, label: str, data: pd.DataFrame,
                headers: list[str], note: str, landscape: bool = True) -> str:
    body = tabular(data, headers)
    content = rf"""\begin{{table}}[p]
\centering
\caption{{{caption}}}
\label{{{label}}}
\small
\resizebox{{\linewidth}}{{!}}{{%
{body}
}}
\par\vspace{{0.5em}}\begin{{minipage}}{{0.98\linewidth}}\footnotesize
\textit{{Note.}} {note}
\end{{minipage}}
\end{{table}}"""
    if landscape:
        return "\\begin{landscape}\n" + content + "\n\\end{landscape}"
    return content


def lung_volume_table(case: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bucket in AGE_ORDER + ["All"]:
        group = case if bucket == "All" else case[case.age_bucket == bucket]
        x = group.total_segmented_lung_volume_mL
        rows.append({
            "Age Range (y)": bucket.replace("-", "–"),
            "Subjects": group.subject.nunique(),
            "CT Scans": group.case_id.nunique(),
            "Mean": f"{x.mean():.0f}",
            "SD": f"{x.std(ddof=1):.0f}",
            "Median": f"{x.median():.0f}",
            "Q1": f"{x.quantile(.25):.0f}",
            "Q3": f"{x.quantile(.75):.0f}",
            "Min": f"{x.min():.0f}",
            "Max": f"{x.max():.0f}",
        })
    return pd.DataFrame(rows)


def main() -> None:
    t1 = pd.read_csv(CSV / "table_1_central_airway_by_age.csv")
    t2 = pd.read_csv(CSV / "table_2_tracheal_sections_by_age.csv")
    t1n = pd.read_csv(CSV / "table_1_lung_normalized.csv")
    t2n = pd.read_csv(CSV / "table_2_lung_normalized.csv")
    case = pd.read_csv(CSV / "case_level_central_airway_and_lung_volume.csv")
    lung = lung_volume_table(case)

    tables = [
        table_block(
            "Central airway dimensions by age group.", "tab:central", t1,
            ["Age range (y)", "Subjects", "CT scans", "Age (y)",
             "Full tracheal length (mm)", "Tracheal diameter (mm)",
             "LMB diameter (mm)", "RMB diameter (mm)"],
            "Age is median (interquartile-range width). Airway measurements are median "
            "(first quartile, third quartile). Tracheal length covers the full segmented "
            "root path and can include extrathoracic trachea. Diameters are EDT-derived "
            "lumen-caliber surrogates. LMB = left main bronchus; RMB = right main bronchus."
        ),
        table_block(
            "Tracheal lumen-diameter surrogate by section and age group.", "tab:sections", t2,
            ["Age range (y)", "Subjects", "CT scans", "Distal (mm)",
             "Middle (mm)", "Proximal (mm)"],
            "Values are median (interquartile-range width). The full segmented tracheal "
            "centerline was divided into equal distal, middle, and proximal thirds."
        ),
        table_block(
            "Central airway dimensions normalized by segmented lung volume.",
            "tab:central-normalized", t1n,
            ["Age range (y)", "Subjects", "CT scans", "Age (y)",
             r"Full length (mm/L$^{1/3}$)", r"Trachea (mm/L$^{1/3}$)",
             r"LMB (mm/L$^{1/3}$)", r"RMB (mm/L$^{1/3}$)"],
            r"Age is median (interquartile-range width). Measurements are median "
            r"(first quartile, third quartile). Each scan's measurement was divided by "
            r"the cube root of its total segmented lung volume in liters before grouping."
        ),
        table_block(
            "Tracheal section diameters normalized by segmented lung volume.",
            "tab:sections-normalized", t2n,
            ["Age range (y)", "Subjects", "CT scans", r"Distal (mm/L$^{1/3}$)",
             r"Middle (mm/L$^{1/3}$)", r"Proximal (mm/L$^{1/3}$)"],
            r"Values are median (interquartile-range width). Normalization was performed "
            r"within each scan before age-group summaries were calculated."
        ),
        table_block(
            "Total segmented lung volume by age group.", "tab:lung-volume", lung,
            ["Age range (y)", "Subjects", "CT scans", "Mean", "SD", "Median",
             "Q1", "Q3", "Minimum", "Maximum"],
            "All volume values are in milliliters. The source is AirMorph's "
            r"\texttt{lunglobe.nii.gz}, generated by the embedded LungMask U-Net R231 "
            "model. Despite the filename, it is a left/right whole-lung mask rather than "
            "a five-lobe segmentation. These volumes do not independently establish "
            "inspiratory adequacy."
        ),
    ]

    figure_specs = [
        ("01_generation_dimensions_median_with_sd.png",
         "Airway lumen radius and diameter across every detected AirMorph generation. Points are age-group medians of per-scan generation means; error bars are between-scan standard deviations.", "fig:generation-dimensions"),
        ("02_generation_branch_counts_mean_sd.png",
         "Detected airway branches per scan by age group and AirMorph generation. Scans without a given generation contribute zero to the count summary.", "fig:generation-counts"),
        ("03_generation_dimensions_lung_normalized.png",
         "Airway dimensions by generation after within-scan lung-volume normalization.", "fig:generation-normalized"),
        ("04_central_airway_dimensions_mean_sd.png",
         "Full segmented tracheal length and central airway lumen-diameter surrogates by age group. Error bars show standard deviations.", "fig:central"),
        ("05_central_airway_dimensions_lung_normalized.png",
         "Central airway dimensions after within-scan lung-volume normalization.", "fig:central-normalized"),
        ("06_tracheal_sections_mean_sd.png",
         "Distal, middle, and proximal tracheal lumen-diameter surrogates by age group. Error bars show standard deviations.", "fig:sections"),
        ("07_tracheal_sections_lung_normalized.png",
         "Tracheal section diameters after within-scan lung-volume normalization.", "fig:sections-normalized"),
        ("08_total_segmented_lung_volume_mean_sd.png",
         "Total segmented lung volume by age group, shown as mean and standard deviation.", "fig:lung-volume"),
    ]
    figures = []
    for filename, caption, label in figure_specs:
        figures.append(rf"""\begin{{figure}}[p]
\centering
\includegraphics[width=\textwidth,height=0.78\textheight,keepaspectratio]{{figures/{filename}}}
\caption{{{caption}}}
\label{{{label}}}
\end{{figure}}
\clearpage""")

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
\title{Normal Pediatric Airway Morphometry\\Meeting Tables and Figures}
\author{}
\date{11 September 2026}
\begin{document}
\maketitle
\tableofcontents
\clearpage

\section{Cohort and reporting conventions}
The analysis contains 83 normal CT scans from 65 unique anonymized subjects. The requested age groups are 0--3, 3--6, 6--9, 10--12, 12--14, 14--16, and 16--18 years. Multiple scans from the same subject are retained, and each table reports both subject and scan counts.

In the Age (y) column, a value such as \textbf{0.9 (1.1)} means a median age of 0.9 years with an interquartile-range width of 1.1 years, where IQR = Q3 $-$ Q1. The all-cohort value \textbf{10.0 (10.0)} means median 10.0 years, Q1 4.0 years, Q3 14.0 years, and therefore IQR 10.0 years. It does not represent a standard deviation or an age range of 10 years around the median.

\section{Primary tables}
""" + "\n\n".join(tables) + r"""

\clearpage
\section{Figures}
""" + "\n\n".join(figures) + r"""

\section{Measurement and interpretation notes}
The full segmented tracheal length is the physical centerline path from the carina-side endpoint to the superior segmented endpoint. It may include segmented extrathoracic trachea and is therefore not directly interchangeable with a manually measured carina-to-thoracic-inlet length. EDT-derived radius is the Euclidean distance from the centerline to the closest segmented airway boundary; diameter is twice that value. These are lumen-caliber surrogates rather than outer-wall dimensions or averages of manually measured perpendicular axes.

AirMorph topologic generation uses the tracheal root as generation 1. Every detected generation through generation 24 is retained. The deepest generations occur in very few scans and should not be interpreted as stable biological age trends. Outer diameter, wall thickness, pulmonary artery diameter, bronchus--artery ratios, iodine-contrast subgroups, true short/long axes, and a manually identified thoracic-inlet plane are unavailable and were not imputed.

Lung-volume normalization is exploratory. Total segmented lung volume was calculated from AirMorph's \texttt{lunglobe.nii.gz}, which is generated by its embedded LungMask U-Net R231 model. Although the file is named ``lunglobe,'' its labels represent the left and right whole lungs rather than five anatomical lobes. Volume equals the number of nonzero mask voxels multiplied by physical voxel volume and divided by 1000 to obtain milliliters. Each airway length or diameter in millimeters was divided within the same scan by the cube root of this volume in liters. The segmented volume helps evaluate differences in lung inflation but cannot itself confirm inspiratory or expiratory acquisition.

\section{Source data}
Complete numerical summaries and case-level source values are supplied beside this report in \path{airway_meeting_results.xlsx} and the \path{csv/} directory.

\end{document}
"""
    TEX.write_text(document)

    overleaf_zip = OUT / "airway_latex_overleaf.zip"
    with zipfile.ZipFile(overleaf_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("main.tex", document)
        zf.writestr(
            "README.txt",
            "Upload this ZIP as a new Overleaf project. main.tex is the main document.\n",
        )
        for figure in sorted((OUT / "figures").glob("*.png")):
            zf.write(figure, f"figures/{figure.name}")

    engine = shutil.which("pdflatex")
    if engine:
        for _ in range(2):
            subprocess.run(
                [engine, "-interaction=nonstopmode", "-halt-on-error", TEX.name],
                cwd=OUT, check=True,
            )
        print(f"wrote PDF: {TEX.with_suffix('.pdf')}")
    else:
        print("No pdflatex executable found; wrote compile-ready TeX source.")

    archive = OUT.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(OUT, arcname=OUT.name)
    print(f"wrote LaTeX report: {TEX}")
    print(f"wrote Overleaf ZIP: {overleaf_zip}")
    print(f"updated archive: {archive}")


if __name__ == "__main__":
    main()
