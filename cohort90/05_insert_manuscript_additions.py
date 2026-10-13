#!/usr/bin/env python
"""Insert the continuous-age / lung-volume / dysanapsis analysis into the
manuscript draft (Word), as tracked changes, with new tables and figures.

Reads the outputs of 04_continuous_age_analysis.py and an unpacked copy of
the draft, and writes a new .docx. Every addition is a tracked insertion
(author below) so the co-authors can accept or edit it; nothing existing is
deleted. Anchors are the first words of existing paragraphs, checked unique.

    python 05_insert_manuscript_additions.py --unpacked <dir> --results <dir> --out <file.docx>
"""
import argparse, datetime, os, re, shutil, subprocess, zipfile
import numpy as np
import pandas as pd
from PIL import Image

AUTHOR = "Umut Isik"
DATE = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
_id = [9000]


def nid():
    _id[0] += 1
    return _id[0]


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fmt_p(p):
    return "<0.001" if p < 0.001 else (f"{p:.3f}" if p < 0.01 else f"{p:.2f}")


# ----------------------------------------------------------------- XML builders

FONT = '<w:rFonts w:ascii="Arial" w:eastAsia="Arial" w:hAnsi="Arial" w:cs="Arial"/>'
BODY_PPR = ('<w:pPr><w:spacing w:before="240" w:after="240" w:line="480" w:lineRule="auto"/><w:jc w:val="both"/>'
            '<w:rPr><w:ins w:id="{id}" w:author="' + AUTHOR + '" w:date="' + DATE + '"/>' + FONT + '</w:rPr></w:pPr>')
HEAD_PPR = ('<w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before="240" w:after="240" w:line="480" w:lineRule="auto"/><w:jc w:val="both"/>'
            '<w:rPr><w:ins w:id="{id}" w:author="' + AUTHOR + '" w:date="' + DATE + '"/>' + FONT + '</w:rPr></w:pPr>')
REF_PPR = ('<w:pPr><w:spacing w:after="0"/><w:ind w:left="640" w:hanging="640"/>'
           '<w:rPr><w:ins w:id="{id}" w:author="' + AUTHOR + '" w:date="' + DATE + '"/>' + FONT + '</w:rPr></w:pPr>')


def ins_wrap(inner):
    return f'<w:ins w:id="{nid()}" w:author="{AUTHOR}" w:date="{DATE}">{inner}</w:ins>'


def run(text, bold=False, italic=False, sup=False):
    rpr = FONT + ("<w:b/><w:bCs/>" if bold else "") + ("<w:i/><w:iCs/>" if italic else "") + ('<w:vertAlign w:val="superscript"/>' if sup else "")
    return f'<w:r><w:rPr>{rpr}</w:rPr><w:t xml:space="preserve">{esc(text)}</w:t></w:r>'


def rich(parts):
    """parts: list of str or (str, dict) with bold/italic/sup flags; handles ^{...} superscripts in str via markers."""
    out = []
    for p in parts:
        if isinstance(p, tuple):
            out.append(run(p[0], **p[1]))
        else:
            # split on L^1/3 style superscripts written as |sup:...|
            for seg in re.split(r"(\|sup:[^|]*\|)", p):
                if seg.startswith("|sup:"):
                    out.append(run(seg[5:-1], sup=True))
                elif seg:
                    out.append(run(seg))
    return "".join(out)


def para(parts, ppr=BODY_PPR):
    return f'<w:p>{ppr.format(id=nid())}{ins_wrap(rich(parts))}</w:p>'


def heading(text):
    return f'<w:p>{HEAD_PPR.format(id=nid())}{ins_wrap(run(text, bold=True))}</w:p>'


def ref_para(text):
    return f'<w:p>{REF_PPR.format(id=nid())}{ins_wrap(run(text))}</w:p>'


def append_runs(paragraph_xml, parts):
    """Append a tracked insertion of runs at the end of an existing paragraph."""
    i = paragraph_xml.rfind("</w:p>")
    return paragraph_xml[:i] + ins_wrap(rich(parts)) + paragraph_xml[i:]


def table(header, rows, widths=None, footnote=None):
    ncol = len(header)
    total = 9016
    widths = widths or [total // ncol] * ncol
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)

    def cell(text, w, bold=False):
        pp = ('<w:pPr><w:spacing w:before="36" w:after="36" w:line="276" w:lineRule="auto"/><w:jc w:val="center"/>'
              f'<w:rPr><w:ins w:id="{nid()}" w:author="{AUTHOR}" w:date="{DATE}"/>{FONT}<w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr></w:pPr>')
        rpr = FONT + '<w:sz w:val="18"/><w:szCs w:val="18"/>' + ("<w:b/><w:bCs/>" if bold else "")
        SUP = '<w:vertAlign w:val="superscript"/>'
        segs = [seg for seg in re.split(r"(\|sup:[^|]*\|)", str(text)) if seg]
        runs = "".join('<w:r><w:rPr>' + rpr + (SUP if seg.startswith("|sup:") else "") + '</w:rPr><w:t xml:space="preserve">'
                       + esc(seg[5:-1] if seg.startswith("|sup:") else seg) + '</w:t></w:r>' for seg in segs)
        return (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/><w:tcMar><w:left w:w="60" w:type="dxa"/><w:right w:w="60" w:type="dxa"/></w:tcMar>'
                f'<w:vAlign w:val="center"/></w:tcPr><w:p>{pp}{ins_wrap(runs)}</w:p></w:tc>')

    def tr(cells, bold=False):
        return (f'<w:tr><w:trPr><w:ins w:id="{nid()}" w:author="{AUTHOR}" w:date="{DATE}"/></w:trPr>'
                + "".join(cell(c, w, bold) for c, w in zip(cells, widths)) + "</w:tr>")

    xml = ('<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/>'
           '<w:tblLook w:val="06A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" w:lastColumn="0" w:noHBand="1" w:noVBand="1"/></w:tblPr>'
           f'<w:tblGrid>{grid}</w:tblGrid>' + tr(header, bold=True) + "".join(tr(r) for r in rows) + "</w:tbl>")
    if footnote:
        xml += para([footnote])
    return xml


def image_para(path, rid, docpr, width_emu=5334000):
    w, h = Image.open(path).size
    cy = int(width_emu * h / w)
    return (f'<w:p>{BODY_PPR.format(id=nid())}{ins_wrap(f"""<w:r><w:rPr><w:noProof/></w:rPr><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{width_emu}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/><wp:docPr id="{docpr}" name="Figure {docpr}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr><a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="{docpr}" name="Figure {docpr}"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{width_emu}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>""")}</w:p>')


# ----------------------------------------------------------------- document surgery

def find_para(doc, startswith):
    paras = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r"<w:p[ >].*?</w:p>", doc, flags=re.S)]
    hits = [p for p in paras if "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", p[2])).startswith(startswith)]
    assert len(hits) == 1, (startswith, len(hits))
    return hits[0]


def insert_after(doc, startswith, xml):
    s, e, _ = find_para(doc, startswith)
    return doc[:e] + xml + doc[e:]


def append_to(doc, startswith, parts):
    s, e, p = find_para(doc, startswith)
    return doc[:s] + append_runs(p, parts) + doc[e:]


def ci(r, nd=3, unit=""):
    return f"{r.coef:+.{nd}f} ({r.ci_low:+.{nd}f} to {r.ci_high:+.{nd}f}){unit}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unpacked", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = a.results
    cr = pd.read_csv(os.path.join(R, "central_regressions.csv"))
    gr = pd.read_csv(os.path.join(R, "generation_regressions.csv"))
    joint = pd.read_csv(os.path.join(R, "joint_model.csv"), index_col=0).iloc[:, 0]
    J = lambda k: joint[k]
    Jf = lambda k: float(joint[k])
    # CI cells are stored as "(np.float64(a), np.float64(b))": drop the type name before parsing
    Jci = lambda k: [float(v) for v in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", str(joint[k]).replace("np.float64", ""))]
    by = pd.read_csv(os.path.join(R, "alr_by_age_group.csv"))
    assoc = pd.read_csv(os.path.join(R, "alr_associations.csv"), index_col=0)
    t = pd.read_csv(os.path.join(R, "case_table.csv"))

    def crow(outcome_prefix, model, term):
        s = cr[(cr.outcome.str.startswith(outcome_prefix)) & (cr.model == model) & (cr.term == term)]
        assert len(s) == 1, (outcome_prefix, model, term, len(s))
        return s.iloc[0]

    A = {k: {"age": crow(lab, f"{k} ~ age", "age"), "vol": crow(lab, f"{k} ~ tlv_l_cbrt", "tlv_l_cbrt"),
             "allo": crow(lab, "logd ~ logv", "logv"), "age_sex": crow(lab, f"{k} ~ age + C(sex)", "age"),
             "sex": crow(lab, f"{k} ~ age + C(sex)", "C(sex)[T.Male]"), "age_adj": crow(lab, f"{k} ~ age + tlv_l_cbrt", "age"),
             "vol_adj": crow(lab, f"{k} ~ age + tlv_l_cbrt", "tlv_l_cbrt"), "norm": crow(lab, f"{k}_norm ~ age", "age"),
             "norm_sex": crow(lab, f"{k}_norm ~ age + C(sex)", "C(sex)[T.Male]")}
         for k, lab in (("trachea", "Trachea"), ("lmb", "LMB"), ("rmb", "RMB"))}
    G = {(int(r.generation), r.kind): r for _, r in gr.iterrows()}
    alr3, alr10 = assoc.loc["alr_central3"], assoc.loc["alr_10"]
    med_norm = {k: t[f"{k}_norm"].median() for k in ("trachea", "lmb", "rmb")}
    norm_pct = max(abs(A[k]["norm"].coef) / med_norm[k] * 100 for k in A)
    n_all10 = int((t.alr_n_branches == 10).sum()); min_br = int(t.alr_n_branches.min())
    age_ci = Jci("age_pct_ci"); nage_ci = Jci("norm_age_pct_ci")

    doc = open(os.path.join(a.unpacked, "word/document.xml"), encoding="utf8").read()
    L13 = "|sup:1/3|"

    # ---- Abstract
    doc = append_to(doc, "Results: The cohort included", [
        f" In continuous-age regression, tracheal, left and right main-bronchial diameters increased by "
        f"{A['trachea']['age'].coef:.2f}, {A['lmb']['age'].coef:.2f} and {A['rmb']['age'].coef:.2f} mm per year (all P<0.001) and scaled with "
        f"lung volume to the power {min(A[k]['allo'].coef for k in A):.2f}–{max(A[k]['allo'].coef for k in A):.2f}, whereas lung-volume-normalized "
        f"diameters were not associated with age (all P≥{min(A[k]['norm'].p for k in A):.2f}). Across G1–G10, diameter increased by "
        f"{Jf('age_pct_per_year'):.1f}% per year; after normalization it decreased by {abs(Jf('norm_age_pct_per_year')):.1f}% per year, "
        f"driven by generations 5–10. The airway-to-lung ratio of the trachea and main bronchi was unrelated to age (ρ={alr3.spearman_rho:+.2f})."])

    # ---- Methods: in-plane resolution
    doc = append_to(doc, "All included examinations were noncontrast", [
        " In-plane pixel spacing ranged from 0.31 to 0.88 mm (median, 0.54 mm), with a 512 × 512 reconstruction matrix in 87 of 90 examinations."])

    # ---- Methods: how the generation diameter is computed
    doc = append_to(doc, "Generation-specific airway diameter was summarized", [
        " Specifically, the lumen radius was sampled at every centerline voxel of a branch as the Euclidean distance from that voxel to the "
        "nearest non-airway voxel, and branch diameter was defined as twice the mean of these radii along the branch. The subject-level "
        "diameter of a generation was the unweighted mean of the branch diameters of all branches assigned to that generation, so that each "
        "branch contributed equally irrespective of its length; cohort summaries were then calculated across subjects."])

    # ---- Methods: ALR definition
    doc = insert_after(doc, "where RMB and LMB denote", para([
        ("Airway-to-lung ratio. ", {"bold": True}),
        "As an index of airway–lung dysanapsis [21, 22], the airway-to-lung ratio (ALR) was calculated as the geometric mean of central-airway "
        "lumen diameters (cm) divided by the cube root of total segmented lung volume (cm", "|sup:3|", "), with the geometric mean computed as "
        "exp[(1/n) Σ ln d", "|sup:i|", "]. Two branch sets were used. ALR-3 comprised the trachea and the right and left main bronchi, measured "
        "with the calliper convention described above. ALR-10 added the seven segmental bronchi that are labelled in every examination "
        "(RB1, RB4, RB6, RB10, LB1+2, LB6 and LB10), taking the most proximal labelled branch of each; because segmental bronchi were measured "
        "by distance transform, all ten branches of ALR-10 were measured with that single convention. Branches not detected in a scan were "
        f"omitted from the mean (all ten were available in {n_all10} of 90 examinations; minimum, {min_br})."]))

    # ---- Methods: statistics
    doc = insert_after(doc, "Continuous variables were summarized", para([
        "Continuous-age and lung-volume associations were examined with ordinary least-squares regression of each central-airway diameter "
        "on age, on the cube root of lung volume, on age and sex, and on age and lung volume together. The allometric relationship between "
        "diameter and lung volume was estimated as the slope of log diameter on log lung volume, for which proportional (isometric) scaling "
        "predicts 1/3. Normalized diameters were regressed on age and on age and sex. For G1–G10, subject-level generation diameter was "
        "regressed on age separately for each generation, and all generations were analyzed jointly in a linear mixed-effects model of log "
        "diameter with age and generation as fixed effects and a random intercept per subject; an age-by-generation interaction was tested "
        "with a likelihood-ratio test. ALR was summarized by age group, its association with age was assessed with Spearman correlation and "
        "linear regression, and sex differences with the Mann–Whitney U test. Regression coefficients are reported with 95% confidence "
        "intervals (CI). Mixed-effects models were fitted with statsmodels (version 0.15.0)."]))

    # ---- Results: continuous-age and lung-volume associations (after the normalized-caliber section)
    tr_, lm_, rm_ = A["trachea"], A["lmb"], A["rmb"]
    res1 = heading("Continuous-age and lung-volume associations of central-airway caliber") + para([
        f"Across the continuous age range, tracheal diameter increased by {tr_['age'].coef:.3f} mm per year (95% CI, {tr_['age'].ci_low:.3f}–{tr_['age'].ci_high:.3f}; "
        f"R² = {tr_['age'].r2:.2f}), left main-bronchial diameter by {lm_['age'].coef:.3f} mm per year ({lm_['age'].ci_low:.3f}–{lm_['age'].ci_high:.3f}; R² = {lm_['age'].r2:.2f}), "
        f"and right main-bronchial diameter by {rm_['age'].coef:.3f} mm per year ({rm_['age'].ci_low:.3f}–{rm_['age'].ci_high:.3f}; R² = {rm_['age'].r2:.2f}) (Table 4; Figure 5). "
        f"Sex was not associated with absolute diameter after adjustment for age (male–female difference, {min(A[k]['sex'].coef for k in A):.2f}–{max(A[k]['sex'].coef for k in A):.2f} mm; "
        f"all P ≥ {min(A[k]['sex'].p for k in A):.2f}). Diameter scaled with segmented lung volume as a power law with exponents of {tr_['allo'].coef:.2f} "
        f"(trachea; 95% CI, {tr_['allo'].ci_low:.2f}–{tr_['allo'].ci_high:.2f}), {lm_['allo'].coef:.2f} (left main bronchus; {lm_['allo'].ci_low:.2f}–{lm_['allo'].ci_high:.2f}) and "
        f"{rm_['allo'].coef:.2f} (right main bronchus; {rm_['allo'].ci_low:.2f}–{rm_['allo'].ci_high:.2f}), each compatible with the cube-root scaling used for normalization (Figure 7). "
        f"Lung volume explained {100*min(A[k]['vol'].r2 for k in A):.0f}–{100*max(A[k]['vol'].r2 for k in A):.0f}% of the variance in central-airway diameter and age "
        f"{100*min(A[k]['age'].r2 for k in A):.0f}–{100*max(A[k]['age'].r2 for k in A):.0f}%. When age and lung volume were entered together, both remained independently associated "
        f"with diameter (age: {tr_['age_adj'].coef:.2f}, {lm_['age_adj'].coef:.2f} and {rm_['age_adj'].coef:.2f} mm per year for the trachea, left and right main bronchus; "
        f"P ≤ {max(A[k]['age_adj'].p for k in A):.2f}), indicating that, at a given lung volume, older children had somewhat larger central airways."]) + para([
        f"After normalization to the cube root of lung volume, diameter was not associated with age: slopes of {tr_['norm'].coef:+.3f} "
        f"(95% CI, {tr_['norm'].ci_low:+.3f} to {tr_['norm'].ci_high:+.3f}; P = {tr_['norm'].p:.2f}), {lm_['norm'].coef:+.3f} ({lm_['norm'].ci_low:+.3f} to {lm_['norm'].ci_high:+.3f}; "
        f"P = {lm_['norm'].p:.2f}) and {rm_['norm'].coef:+.3f} ({rm_['norm'].ci_low:+.3f} to {rm_['norm'].ci_high:+.3f}; P = {rm_['norm'].p:.2f}) mm/L", L13,
        f" per year for the trachea, left and right main bronchus (Figure 6), that is, less than {norm_pct:.1f}% of the cohort median per year. Normalized tracheal and right "
        f"main-bronchial diameters were slightly smaller in males than in females of the same age ({tr_['norm_sex'].coef:+.2f} and {rm_['norm_sex'].coef:+.2f} mm/L", L13,
        f"; P = {tr_['norm_sex'].p:.2f} and {rm_['norm_sex'].p:.2f}), consistent with the larger segmented lung volume of males at a given age (+446 mL; P = 0.002); "
        f"normalized left main-bronchial diameter did not differ by sex (P = {lm_['norm_sex'].p:.2f})."])

    gabs = [G[(g, "absolute")] for g in range(1, 11)]; gnorm = [G[(g, "normalised")] for g in range(1, 11)]
    neg = [g for g in range(1, 11) if G[(g, "normalised")].coef < 0 and G[(g, "normalised")].p < 0.05]
    res2 = heading("Generation-specific age associations") + para([
        f"Within each generation G1–G10, subject-level diameter increased with age (Spearman ρ, {min(r.spearman_rho for r in gabs):.2f}–{max(r.spearman_rho for r in gabs):.2f}; "
        f"all P < 0.001; Table 5; Figure 8). The absolute slope decreased from {gabs[0].coef:.2f} mm per year at G1 to {min(r.coef for r in gabs[6:]):.3f}–{max(r.coef for r in gabs[6:]):.3f} mm per year at G7–G10, "
        f"but relative to each generation's mean diameter the increase was {gabs[0].slope_pct_per_year:.1f}% per year at G1–G2, "
        f"{min(r.slope_pct_per_year for r in gabs[2:5]):.1f}–{max(r.slope_pct_per_year for r in gabs[2:5]):.1f}% at G3–G5 and "
        f"{min(r.slope_pct_per_year for r in gabs[5:]):.1f}–{max(r.slope_pct_per_year for r in gabs[5:]):.1f}% at G6–G10. In the joint mixed-effects model, diameter increased by "
        f"{Jf('age_pct_per_year'):.2f}% per year (95% CI, {age_ci[0]:.2f}–{age_ci[1]:.2f}%; P < 0.001) and decreased by a factor of {Jf('generation_ratio_mean'):.2f} per generation; "
        f"the age effect differed among generations (likelihood-ratio χ² = {Jf('interaction_lrt_chi2'):.1f}, {int(Jf('interaction_lrt_df'))} df; P < 0.001), being steeper proximally. "
        f"Sex was not associated with generation diameter after adjustment for age ({Jf('sex_male_pct'):+.1f}%; P = {Jf('sex_p'):.2f}). After lung-volume normalization, the age "
        f"association was positive at G1 ({gnorm[0].coef:+.3f} mm/L", L13, f" per year; P = {gnorm[0].p:.2f}), absent at G2–G4, and negative from G{neg[0]} to G{neg[-1]} "
        f"({min(G[(g,'normalised')].coef for g in neg):+.3f} to {max(G[(g,'normalised')].coef for g in neg):+.3f} mm/L", L13,
        f" per year; all P ≤ {max(G[(g,'normalised')].p for g in neg):.3f}); in the joint model, normalized diameter decreased by {abs(Jf('norm_age_pct_per_year')):.2f}% per year "
        f"(95% CI, {abs(nage_ci[1]):.2f}–{abs(nage_ci[0]):.2f}%; P < 0.001). Relative to concurrent lung volume, the central airways therefore kept pace with lung growth, whereas "
        "airways from the fifth generation onward became relatively narrower with age."])
    doc = insert_after(doc, "Across all nine age groups, median normalized", res1 + res2)

    # ---- Results: ALR (after proportional geometry)
    res3 = heading("Airway-to-lung ratio") + para([
        f"ALR based on the trachea and main bronchi (ALR-3) had a median of {alr3['median']:.4f} (IQR, {alr3.q1:.4f}–{alr3.q3:.4f}) and was not associated with age "
        f"(Spearman ρ = {alr3.spearman_rho:+.2f}, P = {alr3.spearman_p:.2f}; regression slope, {alr3.slope_per_year:+.5f} per year, P = {alr3.slope_p:.2f}); age-group medians ranged "
        f"from {by.alr3_median.min():.3f} to {by.alr3_median.max():.3f} (Table 6; Figure 9). The ten-branch ALR, which adds seven segmental bronchi, had a median of "
        f"{alr10['median']:.4f} (IQR, {alr10.q1:.4f}–{alr10.q3:.4f}) and decreased with age (ρ = {alr10.spearman_rho:+.2f}, P {fmt_p(alr10.spearman_p)}; slope, "
        f"{alr10.slope_per_year:+.5f} per year, 95% CI {alr10.slope_ci}), from an age-group median of {by.alr10_median.max():.3f} to {by.alr10_median.min():.3f}, "
        f"about {100*abs(alr10.slope_per_year)/alr10['median']:.1f}% of the cohort median per year. Both indices were slightly lower in males (median {alr3.male_median:.4f} vs "
        f"{alr3.female_median:.4f}, P = {alr3.sex_p:.2f}; {alr10.male_median:.4f} vs {alr10.female_median:.4f}, P = {alr10.sex_p:.2f})."])
    doc = insert_after(doc, "None of the three measures showed", res3)

    # ---- Discussion
    doc = insert_after(doc, "Similar attenuation was observed", para([
        f"Continuous-age regression quantifies this attenuation. Absolute central-airway diameter rose by {lm_['age'].coef:.2f}–{tr_['age'].coef:.2f} mm per year with "
        f"R² of {min(A[k]['age'].r2 for k in A):.2f}–{max(A[k]['age'].r2 for k in A):.2f}, whereas the normalized diameters changed by less than {norm_pct:.1f}% of their "
        f"median per year with confidence intervals that include zero. The observed allometric exponents of {min(A[k]['allo'].coef for k in A):.2f}–{max(A[k]['allo'].coef for k in A):.2f} "
        "provide an empirical basis for the cube-root normalization, which was adopted a priori on dimensional grounds. Two further observations qualify the simple "
        "proportional picture. At a given lung volume, older children had slightly larger central airways, and from the fifth generation onward the lung-volume-normalized "
        f"diameter declined by about {abs(Jf('norm_age_pct_per_year')):.0f}% per year, so that the ten-branch airway-to-lung ratio fell with age while the central-airway ratio "
        "did not. Both are consistent with dysanapsis, the long-standing observation that airway caliber and lung volume do not grow in fixed proportion [21], now measured "
        "in adults as the airway-to-lung ratio and associated with later obstructive disease [22]. Our branch set and measurement convention differ from the 13- to "
        "19-branch protocols of those studies, so absolute ALR values are not comparable across studies, and the age dependence of the peripheral index cannot be "
        "separated here from the resolution limit on distance-transform diameters in small children, where a 1.5-mm bronchus spans only two to three voxels; the "
        "cross-sectional design and uncontrolled inspiration apply to these analyses as to the normalization above."]))

    # ---- References
    doc = insert_after(doc, "20. ", ref_para(
        "21. Mead J (1980) Dysanapsis in normal lungs assessed by the relationship between maximal flow, static recoil, and vital capacity. "
        "Am Rev Respir Dis 121:339–342. https://doi.org/10.1164/arrd.1980.121.2.339") + ref_para(
        "22. Smith BM, Kirby M, Hoffman EA, et al (2020) Association of Dysanapsis With Chronic Obstructive Pulmonary Disease Among Older Adults. "
        "JAMA 323:2268–2280. https://doi.org/10.1001/jama.2020.6918"))

    # ---- Tables 4-6 after Table 3's footnote
    def fmt_ci(r, nd):
        return f"{r.coef:+.{nd}f} ({r.ci_low:+.{nd}f} to {r.ci_high:+.{nd}f})"
    t4_rows = []
    for k, lab in (("trachea", "Trachea"), ("lmb", "Left main bronchus"), ("rmb", "Right main bronchus")):
        x = A[k]
        t4_rows.append([lab, fmt_ci(x["age"], 3), f"{x['age'].r2:.2f}", f"{x['allo'].coef:.2f} ({x['allo'].ci_low:.2f} to {x['allo'].ci_high:.2f})", f"{x['allo'].r2:.2f}",
                        fmt_ci(x["age_adj"], 3), f"{fmt_ci(x['sex'], 2)}; P = {x['sex'].p:.2f}", f"{fmt_ci(x['norm'], 3)}; P = {x['norm'].p:.2f}",
                        f"{fmt_ci(x['norm_sex'], 2)}; P = {x['norm_sex'].p:.2f}"])
    t4 = heading("Table 4. Regression of central-airway diameter on age, lung volume and sex (n = 90)") + table(
        ["Airway", "Age slope, mm/y (95% CI)", "R²", "Lung-volume exponent b (95% CI)", "R²", "Age slope adjusted for lung volume, mm/y (95% CI)",
         "Male − female, mm (95% CI)", "Normalized age slope, mm/L" + L13 + " per y (95% CI)", "Normalized male − female, mm/L" + L13 + " (95% CI)"],
        t4_rows, widths=[1000, 1150, 500, 1150, 500, 1250, 1150, 1200, 1116],
        footnote="Footnote: Ordinary least-squares regression; all age slopes and lung-volume exponents P < 0.001. The lung-volume exponent b is the slope of log diameter on "
                 "log total segmented lung volume (d ∝ TLV^b); proportional scaling predicts b = 1/3. Sex differences are adjusted for age. Normalized diameter is "
                 "diameter divided by the cube root of lung volume in liters.")
    t5_rows = []
    for g in range(1, 11):
        a_, n_ = G[(g, "absolute")], G[(g, "normalised")]
        t5_rows.append([f"G{g}", f"{a_['mean']:.2f}", fmt_ci(a_, 3), f"{a_.slope_pct_per_year:+.1f}", f"{a_.spearman_rho:+.2f}", fmt_ci(n_, 3), fmt_p(n_.p)])
    t5 = heading("Table 5. Age association of subject-level airway diameter by topologic generation (n = 90)") + table(
        ["Generation", "Mean diameter, mm", "Age slope, mm/y (95% CI)", "Age slope, % of mean per y", "Spearman ρ", "Normalized age slope, mm/L" + L13 + " per y (95% CI)", "P (normalized)"],
        t5_rows, widths=[900, 1100, 1700, 1100, 900, 2000, 1316],
        footnote=f"Footnote: Diameter is the distance-transform diameter averaged over the branches of each generation within a subject. All absolute slopes and Spearman "
                 f"correlations P < 0.001. Joint linear mixed-effects model of log diameter (900 generation-level observations in 90 subjects, random intercept per subject): "
                 f"age {Jf('age_pct_per_year'):+.2f}% per year (95% CI, {age_ci[0]:+.2f} to {age_ci[1]:+.2f}%; P < 0.001); mean diameter ratio between consecutive generations "
                 f"{Jf('generation_ratio_mean'):.3f}; age × generation interaction likelihood-ratio χ² = {Jf('interaction_lrt_chi2'):.1f} on {int(Jf('interaction_lrt_df'))} df (P < 0.001); "
                 f"male vs female {Jf('sex_male_pct'):+.1f}% (P = {Jf('sex_p'):.2f}). Normalized diameter in the joint model: age {Jf('norm_age_pct_per_year'):+.2f}% per year "
                 f"(95% CI, {nage_ci[0]:+.2f} to {nage_ci[1]:+.2f}%; P < 0.001).")
    t6_rows = [[r.age_group.replace("-", "–"), str(int(r.n)), f"{r.alr3_median:.4f} ({r.alr3_q1:.4f}, {r.alr3_q3:.4f})", f"{r.alr10_median:.4f} ({r.alr10_q1:.4f}, {r.alr10_q3:.4f})"]
               for _, r in by.iterrows()]
    t6_rows.append(["All", "90", f"{alr3['median']:.4f} ({alr3.q1:.4f}, {alr3.q3:.4f})", f"{alr10['median']:.4f} ({alr10.q1:.4f}, {alr10.q3:.4f})"])
    t6_rows.append(["Spearman ρ with age", "", f"{alr3.spearman_rho:+.2f}", f"{alr10.spearman_rho:+.2f}"])
    t6_rows.append(["P value", "", fmt_p(alr3.spearman_p), fmt_p(alr10.spearman_p)])
    t6 = heading("Table 6. Airway-to-lung ratio by age group and association with age") + table(
        ["Age range (y)", "Scans", "ALR-3: trachea and main bronchi", "ALR-10: ten central branches"], t6_rows, widths=[1800, 1000, 3108, 3108],
        footnote="Footnote: Values are medians (first quartile, third quartile). ALR is the geometric mean of lumen diameters (cm) divided by the cube root of total segmented "
                 "lung volume (cm³), dimensionless. ALR-3 uses the calliper diameters of the trachea and the right and left main bronchi. ALR-10 adds the most proximal "
                 "labelled branch of RB1, RB4, RB6, RB10, LB1+2, LB6 and LB10 and uses distance-transform diameters for all ten branches; branches not detected were omitted "
                 f"(all ten present in {n_all10} of 90 scans; minimum, {min_br}). Spearman correlations were calculated across all 90 subjects.")
    doc = insert_after(doc, "Footnote:  Values are medians", t4 + t5 + t6)

    # ---- Figures 5-9: images + captions after Figure 4's caption
    media = os.path.join(a.unpacked, "word", "media")
    rels_p = os.path.join(a.unpacked, "word", "_rels", "document.xml.rels")
    rels = open(rels_p, encoding="utf8").read()
    figs = [("Figure_5_central_diameter_vs_age.png",
             "Figure 5. Central-airway diameter against continuous age. Tracheal, left main-bronchial and right main-bronchial diameter of each subject against age, "
             "colored by sex, with the ordinary least-squares regression line and its 95% confidence band; slopes, R² and Spearman ρ are given in Table 4."),
            ("Figure_6_normalised_diameter_vs_age.png",
             "Figure 6. Lung-volume-normalized central-airway diameter against age. Diameter divided by the cube root of total segmented lung volume (mm/L^1/3), with the "
             "regression line and 95% confidence band; none of the slopes differs from zero."),
            ("Figure_7_diameter_vs_lung_volume.png",
             "Figure 7. Central-airway diameter against segmented lung volume. Logarithmic axes; the solid line is the fitted power law d ∝ TLV^b with its 95% confidence "
             "band and the dashed line the isometric slope of 1/3 through the same data. The fitted exponents (0.30–0.33) do not differ from 1/3."),
            ("Figure_8_generation_age_slopes.png",
             "Figure 8. Age association of airway diameter by topologic generation. Left: ordinary least-squares slope of subject-level generation diameter on age "
             "(mm per year, 95% CI) for G1–G10. Right: the same for lung-volume-normalized diameter. "
             "Percent change per year, P values and the joint mixed-effects estimates are given in Table 5 and the text."),
            ("Figure_9_airway_to_lung_ratio.png",
             "Figure 9. Airway-to-lung ratio against age. Left: ten-branch ALR (trachea, main bronchi and seven segmental bronchi, distance-transform diameters). "
             "Right: ALR of the trachea and main bronchi (calliper diameters). Points are subjects colored by sex, the black line the regression on age with its 95% "
             "confidence band, and the green squares the age-group medians.")]
    xml = ""
    for i, (fn, cap) in enumerate(figs):
        src = os.path.join(R, fn); dst_name = f"image{100 + i}.png"
        shutil.copy2(src, os.path.join(media, dst_name))
        rid = f"rId{900 + i}"
        rels = rels.replace("</Relationships>", f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{dst_name}"/></Relationships>')
        xml += image_para(src, rid, 910000 + i)
        xml += para([cap.replace("^1/3", "|sup:1/3|").replace("^b", "|sup:b|")])
    doc = insert_after(doc, "Figure 4. Central-airway diameter normalized", xml)
    open(rels_p, "w", encoding="utf8").write(rels)
    open(os.path.join(a.unpacked, "word/document.xml"), "w", encoding="utf8").write(doc)

    # ---- zip
    if os.path.exists(a.out):
        os.remove(a.out)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(a.unpacked):
            for f in files:
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, a.unpacked))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
