#!/usr/bin/env python3
"""Retypeset the report so its tables fit the page without being rescaled.

Every table in the source was wrapped in \\resizebox inside a landscape page.
That scales each table by whatever factor it happens to need, so no two tables
share a font size and the document reads as a set of screenshots rather than a
paper. Here each age-indexed table is transposed instead -- age groups become
columns, measures become rows -- and values of the form "median (q1, q3)" are
stacked onto two lines. Both changes cut the natural width enough that every
table sets in portrait at one consistent size, with no scaling anywhere.

Numbers are carried across verbatim from the source .tex. Nothing is recomputed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

BASE = Path("/fileserver/external/quin/ege")
RUN = BASE / "AIRWAY_BALANCED90_DATASET_OVERVIEW_CORRECTED_2026-09-15"
RESULTS = RUN / "AIRWAY_RESULTS_90"
TARGET = RESULTS / "FINAL_INCLUSIVE_N10_airway_report_90.tex"
# The untouched generated source, kept aside on first run so this stays re-runnable.
SOURCE = RESULTS / "FINAL_INCLUSIVE_N10_airway_report_90.source.tex"
FIGURES = RESULTS / "figures"
NEW_FIGURES = BASE / "_archive_2026-09-26" / "BALANCED_COMBINED_90_2026-09-15"

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[letterpaper,left=0.9in,right=0.9in,top=0.75in,bottom=0.75in]{geometry}
\usepackage{pdflscape}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{makecell}
\usepackage{threeparttable}
\usepackage{float}
\PassOptionsToPackage{hyphens}{url}
\usepackage[hidelinks]{hyperref}
\sloppy
\usepackage[font=small,labelfont=bf,skip=6pt]{caption}
\renewcommand{\arraystretch}{1.18}
\renewcommand\theadfont{\normalsize}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5em}
\setcounter{topnumber}{1}
\setcounter{bottomnumber}{1}
\setcounter{totalnumber}{3}
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.7}
\renewcommand{\textfraction}{0.1}
\renewcommand{\floatpagefraction}{0.6}
"""

FIGURE_TREES = r"""
\begin{figure}[tbp]
\centering
\includegraphics[width=\textwidth]{figures/00_branching_by_age.png}
\caption{Representative segmented airway tree for each age group, one scan per
group, drawn at a common millimetre scale and hung from the thoracic-inlet clip
so that calibre, length and branching are compared directly. The scan shown for
each group is the one whose frontal extent is closest to its group median,
subject to extent not decreasing with age.}
\label{fig:trees}
\end{figure}
"""

FIGURE_GENERATIONS = r"""
\begin{figure}[tbp]
\centering
\includegraphics[width=0.78\textwidth]{figures/00_generation_tree.png}
\caption{PLACEHOLDER FOR NOW}
\label{fig:generations}
\end{figure}
"""


def matched_braces(text: str, start: int) -> tuple[str, int]:
    """Return the balanced-brace group beginning at `start`, and the index after it."""
    depth, index = 0, start
    while index < len(text):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:index], index + 1
        index += 1
    raise ValueError("unbalanced braces")


def field(block: str, command: str) -> str:
    position = block.index("\\" + command + "{") + len(command) + 1
    return matched_braces(block, position)[0]


def parse_table(block: str) -> dict:
    caption = field(block, "caption")
    label = field(block, "label")
    body = block[block.index("\\toprule") + len("\\toprule"):block.index("\\bottomrule")]
    body = body.replace("\\midrule", "")
    rows = [line.strip() for line in body.split("\\\\") if line.strip()]
    grid = [[cell.strip() for cell in row.split("&")] for row in rows]
    note = ""
    if "\\textit{Note.}" in block:
        note = block.split("\\textit{Note.}")[1].split("\\end{minipage}")[0].strip()
    return {"caption": caption, "label": label, "header": grid[0], "rows": grid[1:], "note": note}


VALUE = re.compile(r"^(-?[\d.]+)\s*\(([^)]*)\)(?:\s*\[n=(\d+)\])?$")


def split_cell(cell: str) -> tuple[str, str | None]:
    """Stack "median (q1, q3)" onto two lines and peel off any [n=k] annotation."""
    match = VALUE.match(cell)
    if not match:
        return cell, None
    value, spread, count = match.groups()
    return f"\\makecell[c]{{{value}\\\\[-1pt]\\scriptsize({spread})}}", count


def stack_header(text: str) -> str:
    """Break a column heading over two lines at the most even word boundary.

    Headings such as "Trachea short/long" are wider than the numbers beneath
    them and are what actually sets the column width, so wrapping them buys
    more room than anything done to the data.
    """
    words = text.split()
    if len(words) < 2:
        return text
    best = min(range(1, len(words)),
               key=lambda cut: abs(len(" ".join(words[:cut])) - len(" ".join(words[cut:]))))
    top, bottom = " ".join(words[:best]), " ".join(words[best:])
    return f"\\makecell[c]{{{top}\\\\{bottom}}}"


def stack_cell(cell: str) -> str:
    """Stack "median (q1, q3)" onto two lines, keeping any [n=k] with its value."""
    match = VALUE.match(cell)
    if not match:
        return cell
    value, spread, count = match.groups()
    tail = f"({spread})" + (f" [n={count}]" if count else "")
    return f"\\makecell[c]{{{value}\\\\[-1pt]\\scriptsize {tail}}}"


def transpose(table: dict, landscape: bool = False, colsep: float = 3.0) -> str:
    """Age groups become columns and measures become rows.

    A handful of tables carry quartile pairs in all ten columns and are wider
    than the text block whatever is done to them in portrait. Those are turned
    on their side rather than shrunk, so every table in the document still sets
    at one size.
    """
    measures = table["header"][1:]
    groups = [row[0] for row in table["rows"]]
    columns = len(groups)

    lines, counts_needed = [], {}
    for index, measure in enumerate(measures, start=1):
        cells, counts = [], []
        for row in table["rows"]:
            text, count = split_cell(row[index])
            cells.append(text)
            counts.append(count)
        lines.append((measure, cells))
        if any(count is not None for count in counts):
            counts_needed[measure] = [count or "10" for count in counts]

    spec = "@{}l" + "c" * (columns - 1) + "@{\\hspace{10pt}}c@{}"
    out = [f"% TABLE {table['label']}"]
    if landscape:
        out.append(r"\begin{landscape}")
    out += [r"\begin{table}[p]" if landscape else r"\begin{table}[H]", r"\centering",
            f"\\caption{{{table['caption']}}}", f"\\label{{{table['label']}}}",
            r"\begin{threeparttable}", r"\footnotesize",
            f"\\setlength{{\\tabcolsep}}{{{colsep}pt}}",
           f"\\begin{{tabular}}{{{spec}}}", r"\toprule",
           " & " + f"\\multicolumn{{{columns - 1}}}{{c}}{{Age group (y)}}" + " & \\\\",
           f"\\cmidrule(lr){{2-{columns}}}",
           "Measure & " + " & ".join(groups) + r" \\", r"\midrule"]
    for measure, cells in lines:
        out.append(f"{measure} & " + " & ".join(cells) + r" \\")
        if measure in counts_needed:
            out.append(f"\\quad scans contributing, $n$ & "
                       + " & ".join(counts_needed[measure]) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    if table["note"]:
        out += [r"\begin{tablenotes}[flushleft]\footnotesize",
                f"\\item \\textit{{Note.}} {table['note']}", r"\end{tablenotes}"]
    out += [r"\end{threeparttable}", r"\end{table}"]
    if landscape:
        out.append(r"\end{landscape}")
    return "\n".join(out)


def as_written(table: dict, landscape: bool = False, colsep: float = 4.0) -> str:
    """Age groups stay in the rows, measures in the columns, as originally laid out.

    Only the typesetting is rebuilt: quartile pairs are stacked onto a second
    line so the table narrows enough to set at the document's one font size,
    with no \\resizebox anywhere.
    """
    header = table["header"]
    spec = "@{}l" + "c" * (len(header) - 1) + "@{}"

    out = [f"% TABLE {table['label']}"]
    if landscape:
        out.append(r"\begin{landscape}")
    out += [r"\begin{table}[p]" if landscape else r"\begin{table}[H]", r"\centering",
            f"\\caption{{{table['caption']}}}", f"\\label{{{table['label']}}}",
            r"\begin{threeparttable}", r"\footnotesize",
            f"\\setlength{{\\tabcolsep}}{{{colsep}pt}}",
            f"\\begin{{tabular}}{{{spec}}}", r"\toprule",
            " & ".join(stack_header(cell) for cell in header) + r" \\", r"\midrule"]
    for row in table["rows"]:
        if row[0].strip() == "All":                  # summary row set apart
            out.append(r"\midrule")
        out.append(" & ".join(stack_cell(cell) for cell in row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    if table["note"]:
        out += [r"\begin{tablenotes}[flushleft]\footnotesize",
                f"\\item \\textit{{Note.}} {table['note']}", r"\end{tablenotes}"]
    out += [r"\end{threeparttable}", r"\end{table}"]
    if landscape:
        out.append(r"\end{landscape}")
    return "\n".join(out)


def as_paragraph_table(table: dict) -> str:
    """The QC listing is not age-indexed, so it keeps its shape and wraps instead."""
    widths = ["l", "c", "p{0.24\\textwidth}", "p{0.40\\textwidth}"]
    out = [f"% TABLE {table['label']}", r"\begin{table}[H]", r"\centering",
           f"\\caption{{{table['caption']}}}", f"\\label{{{table['label']}}}",
           r"\begin{threeparttable}", r"\footnotesize",
           r"\setlength{\tabcolsep}{5pt}",
           "\\begin{tabular}{@{}" + "".join(widths) + "@{}}", r"\toprule",
           " & ".join(f"\\textbf{{{cell}}}" for cell in table["header"]) + r" \\",
           r"\midrule"]
    for row in table["rows"]:
        out.append(" & ".join(row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    if table["note"]:
        out += [r"\begin{tablenotes}[flushleft]\footnotesize",
                f"\\item \\textit{{Note.}} {table['note']}", r"\end{tablenotes}"]
    out += [r"\end{threeparttable}", r"\end{table}"]
    return "\n".join(out)


OVERFULL = re.compile(r"Overfull \\hbox \(([\d.]+)pt too wide\) in paragraph at lines (\d+)--(\d+)")


ROTATE_ABOVE_PT = 25.0        # below this, tightening the columns is enough
TRANSPOSE = False             # True puts age groups in the columns instead


def assemble(source: str, sideways: set[str], tight: set[str]) -> str:
    """Rebuild the document, turning the named tables onto their side."""
    head, body = source.split(r"\begin{document}", 1)
    title = head[head.index(r"\title{"):]
    document = PREAMBLE + title + r"\begin{document}" + body

    pattern = re.compile(r"\\begin\{landscape\}\s*(\\begin\{table\}.*?\\end\{table\})\s*"
                         r"\\end\{landscape\}", re.DOTALL)

    def convert(match: re.Match) -> str:
        table = parse_table(match.group(1))
        if table["header"][0].startswith(("Age group", "Age range")):
            emit = transpose if TRANSPOSE else as_written
            return emit(table, landscape=table["label"] in sideways,
                        colsep=1.5 if table["label"] in tight else 3.0)
        return as_paragraph_table(table)

    document = pattern.sub(convert, document)

    # The overview figures belong with the cohort and the measurement section,
    # both near the front, rather than in the figure appendix at the back.
    document = document.replace("\n\\section{How the airways were measured}",
                                FIGURE_TREES + "\n\\section{How the airways were measured}", 1)
    document = document.replace("\n\\section{Quality control of the central airways}",
                                FIGURE_GENERATIONS + "\n\\section{Quality control of the central airways}", 1)
    return document


def build() -> str:
    engine = RUN / "tools/tectonic-musl/tectonic"
    subprocess.run([str(engine), "-X", "compile", "--keep-logs", TARGET.name],
                   cwd=RESULTS, check=True, capture_output=True, text=True)
    return TARGET.with_suffix(".log").read_text(errors="replace")


def too_wide(log: str, document: str) -> dict[str, float]:
    """Labels of tables that ran past the text block, with the worst overflow."""
    marks = [(index + 1, line.split()[2])
             for index, line in enumerate(document.splitlines())
             if line.startswith("% TABLE ")]
    worst: dict[str, float] = {}
    for amount, _, end in OVERFULL.findall(log):
        preceding = [label for start, label in marks if start <= int(end)]
        if preceding:
            label = preceding[-1]
            worst[label] = max(worst.get(label, 0.0), float(amount))
    return worst


def main() -> None:
    shutil.copy(NEW_FIGURES / "airway_branching_by_age_90.png",
                FIGURES / "00_branching_by_age.png")
    shutil.copy(NEW_FIGURES / "airway_generation_convention_tree.png",
                FIGURES / "00_generation_tree.png")

    if not SOURCE.exists():
        shutil.copy(TARGET, SOURCE)
    source = SOURCE.read_text()

    # Typeset once in portrait, ask the engine which tables did not fit, then
    # turn only those onto their side. Measuring beats guessing at column widths.
    sideways: set[str] = set()
    tight: set[str] = set()
    for attempt in range(5):
        document = assemble(source, sideways, tight)
        TARGET.write_text(document)
        overflowing = too_wide(build(), document)
        if not overflowing:
            break
        newly = {label: amount for label, amount in overflowing.items()
                 if label not in sideways and not (label in tight and amount <= ROTATE_ABOVE_PT)}
        if not newly:
            print("still overfull:", {k: round(v, 1) for k, v in overflowing.items()})
            break
        for label, amount in sorted(newly.items()):
            if amount > ROTATE_ABOVE_PT:
                print(f"  {label} overflows by {amount:.0f}pt -> landscape")
                sideways.add(label)
                tight.discard(label)
            else:
                print(f"  {label} overflows by {amount:.0f}pt -> tighter columns")
                tight.add(label)

    print(f"tables turned sideways: {sorted(sideways) or 'none'}")
    print(f"tables with tightened columns: {sorted(tight) or 'none'}")
    print(f"wrote {TARGET}")
    print(f"wrote {RESULTS / 'FINAL_INCLUSIVE_N10_airway_report_90.pdf'}")


if __name__ == "__main__":
    main()
