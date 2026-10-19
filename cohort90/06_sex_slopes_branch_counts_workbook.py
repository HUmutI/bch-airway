#!/usr/bin/env python
"""Requests from the pulmonology co-author (2026-10-16) for the 90-scan cohort.

1. Sex-specific growth: ALR-10 and ALR-3 against age with a separate regression
   line per sex, and the per-generation age slopes (absolute and lung-volume-
   normalised diameter) fitted separately for males and females; age x sex
   interaction tests in the CSV.
2. Branches by generation: number of detected branches per generation per age
   group against the dichotomous expectation 2^(g-1), and the same as a
   percentage of that expectation, to show where detection levels off.
3. A per-patient workbook: central diameters (calliper and distance transform),
   lung volume, normalised diameters, ALR, segmental diameters, and per
   generation G1-G14 the branch count and mean/SD/median diameter.

Generation 1 is the trachea, as in the manuscript. Same data and definitions as
04_continuous_age_analysis.py (case_table is imported from it).

    python 06_sex_slopes_branch_counts_workbook.py --out <dir>   (stats venv)
"""
import argparse, importlib.util, os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("cont", os.path.join(HERE, "04_continuous_age_analysis.py"))
cont = importlib.util.module_from_spec(spec); spec.loader.exec_module(cont)
CSV, GENS, SEX_COL, ALR_SEGMENTAL = cont.CSV, cont.GENS, cont.SEX_COL, cont.ALR_SEGMENTAL
ALL_GENS = list(range(1, 15))
GROUPS = ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12", "12-14", "14-16", "16-18"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def slope(d, y):
    m = smf.ols(f"{y} ~ age", d).fit()
    lo, hi = m.conf_int().loc["age"]
    return m.params["age"], lo, hi, m.pvalues["age"], int(m.nobs)


def interaction_p(d, y):
    return smf.ols(f"{y} ~ age * C(sex)", d).fit().pvalues["age:C(sex)[T.Male]"]


# ------------------------------------------------------------ 1. sex-specific slopes

def sex_tables(t):
    rows = []
    for k, lab in (("alr_10", "ALR-10"), ("alr_central3", "ALR-3")):
        d = t.dropna(subset=[k])
        for s in ("Female", "Male"):
            b, lo, hi, p, n = slope(d[d.sex == s], k)
            rows.append({"outcome": lab, "sex": s, "n": n, "slope_per_year": b, "ci_low": lo, "ci_high": hi, "p": p,
                         "age_x_sex_p": interaction_p(d, k)})
    for kind, suf in (("absolute", ""), ("normalised", "_norm")):
        for g in GENS:
            y = f"g{g}{suf}"; d = t.dropna(subset=[y])
            ip = interaction_p(d, y)
            for s in ("Female", "Male"):
                b, lo, hi, p, n = slope(d[d.sex == s], y)
                rows.append({"outcome": f"G{g} diameter, {kind}", "sex": s, "n": n, "slope_per_year": b,
                             "ci_low": lo, "ci_high": hi, "p": p, "age_x_sex_p": ip})
    return pd.DataFrame(rows)


def fig_alr_by_sex(t, st, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, k, lab, key in ((axes[0], "alr_10", "Ten central branches", "ALR-10"),
                            (axes[1], "alr_central3", "Trachea and main bronchi", "ALR-3")):
        d = t.dropna(subset=[k])
        for s, c in SEX_COL.items():
            sel = d[d.sex == s]
            ax.scatter(sel.age, sel[k], s=16, c=c, alpha=0.75, edgecolor="white", lw=0.4)
            cont.fit_line(ax, sel.age.values, sel[k].values, color=c, label=s)   # slopes: workbook, "Slopes by sex"
        ax.set_title(lab, fontsize=10); ax.set_xlabel("Age (years)"); ax.set_xlim(0, 18.5); ax.set_xticks([0, 5, 10, 15])
        lo_, hi_ = d[k].min(), d[k].max()
        ax.set_ylim(lo_ - 0.05 * (hi_ - lo_), hi_ + 0.30 * (hi_ - lo_))   # head-room for the legend
        ax.legend(loc="upper right", fontsize=9.5, frameon=False)
    axes[0].set_ylabel("Airway-to-lung ratio (dimensionless)")
    fig.tight_layout(); fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300); plt.close(fig)


def fig_generation_by_sex(st, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, kind, ylabel in ((axes[0], "absolute", "Age slope (mm per year)"),
                             (axes[1], "normalised", "Age slope (mm/L$^{1/3}$ per year)")):
        for s, c, dx in (("Female", SEX_COL["Female"], -0.12), ("Male", SEX_COL["Male"], 0.12)):
            d = st[(st.sex == s) & st.outcome.str.endswith(kind)].copy()
            d["g"] = d.outcome.str.extract(r"G(\d+)").astype(int)
            d = d.sort_values("g")
            ax.errorbar(d.g + dx, d.slope_per_year, yerr=[d.slope_per_year - d.ci_low, d.ci_high - d.slope_per_year],
                        fmt="o", color=c, ecolor=c, capsize=2, ms=4, label=s)
        ax.axhline(0, color="#999", lw=0.8, ls="--")
        ax.set_xticks(GENS); ax.set_xlabel("Topologic generation"); ax.set_ylabel(ylabel)
    axes[0].set_title("Absolute diameter", fontsize=10); axes[1].set_title("Lung-volume-normalised diameter", fontsize=10)
    axes[0].legend(loc="upper right", fontsize=10, frameon=False)
    fig.tight_layout(); fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300); plt.close(fig)


# ------------------------------------------------------------ 2. branches by generation

def branch_counts(t):
    gen = pd.read_csv(os.path.join(CSV, "per_scan_generation_metrics.csv"))
    gen = gen[gen.generation.isin(ALL_GENS)]
    c = gen.pivot(index="case_id", columns="generation", values="branch_count").reindex(columns=ALL_GENS).fillna(0)
    c = c.merge(t[["case_id", "age_group"]].set_index("case_id"), left_index=True, right_index=True)
    long = c.reset_index().melt(id_vars=["case_id", "age_group"], var_name="generation", value_name="branches")
    long["generation"] = long.generation.astype(int)
    long["expected"] = 2.0 ** (long.generation - 1)
    long["pct_of_expected"] = 100 * long.branches / long.expected
    summ = long.groupby(["age_group", "generation"], observed=True).agg(
        median_branches=("branches", "median"), q1=("branches", lambda s: s.quantile(.25)),
        q3=("branches", lambda s: s.quantile(.75)), median_pct_of_expected=("pct_of_expected", "median")).reset_index()
    return long, summ


def fig_branch_counts(summ, out):
    cmap = plt.get_cmap("viridis")
    cols = {g: cmap(i / (len(GROUPS) - 1) * 0.9) for i, g in enumerate(GROUPS)}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    gx = np.array(ALL_GENS)
    axes[0].plot(gx, 2.0 ** (gx - 1), "--", color="#777", lw=1, label="Expected, 2$^{g-1}$")
    for g in GROUPS:
        d = summ[summ.age_group == g].sort_values("generation")
        nz = d[d.median_branches > 0]                    # a median of zero cannot sit on a log axis
        axes[0].plot(nz.generation, nz.median_branches, "o-", color=cols[g], ms=3, lw=1.2, label=f"{g} y")
        axes[1].plot(d.generation, d.median_pct_of_expected, "o-", color=cols[g], ms=3, lw=1.2)
    axes[0].set_yscale("log", base=2)
    ticks = 2.0 ** np.arange(0, 14, 2)
    axes[0].set_yticks(ticks); axes[0].set_yticklabels([f"{int(v)}" for v in ticks]); axes[0].minorticks_off()
    axes[0].set_ylabel("Branches per scan (median, log$_2$ scale)")
    axes[1].set_ylabel("Detected branches (% of 2$^{g-1}$)")
    axes[1].set_ylim(0, 115); axes[1].set_yticks([0, 25, 50, 75, 100])
    axes[1].axhline(100, color="#777", lw=1, ls="--")
    for ax in axes:
        ax.set_xticks(ALL_GENS); ax.set_xlabel("Topologic generation (1 = trachea)")
    axes[0].legend(loc="upper left", fontsize=9, frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300); plt.close(fig)


# ------------------------------------------------------------ 3. workbook

def generation_long(t):
    br = pd.read_csv(os.path.join(CSV, "branch_level_generation_1_to_14.csv"), low_memory=False)
    br = br[br.generation.isin(ALL_GENS) & br.case_id.isin(t.case_id)]
    g = br.groupby(["case_id", "generation"]).agg(
        branch_count=("diameter_mean_mm", "size"), diameter_mean_mm=("diameter_mean_mm", "mean"),
        diameter_sd_mm=("diameter_mean_mm", "std"), diameter_median_mm=("diameter_mean_mm", "median"),
        length_mean_mm=("centerline_length_mm", "mean")).reset_index()
    g = g.merge(t[["case_id", "tlv_l_cbrt"]], on="case_id")
    g["diameter_mean_norm_mm_per_L_cbrt"] = g.diameter_mean_mm / g.tlv_l_cbrt
    return g.drop(columns="tlv_l_cbrt")


def workbook(t, gl, st, long, summ, path):
    pat = pd.DataFrame({
        "case_id": t.case_id, "age_years": t.age, "age_group": t.age_group.astype(str), "sex": t.sex,
        "lung_volume_mL": t.tlv_ml, "lung_scale_L_cuberoot": t.tlv_l_cbrt,
        "trachea_calliper_mm": t.trachea, "LMB_calliper_mm": t.lmb, "RMB_calliper_mm": t.rmb,
        "trachea_calliper_norm": t.trachea_norm, "LMB_calliper_norm": t.lmb_norm, "RMB_calliper_norm": t.rmb_norm,
        "trachea_EDT_mm": t.trachea_edt, "LMB_EDT_mm": t.lmb_edt, "RMB_EDT_mm": t.rmb_edt,
        **{f"{s}_EDT_mm": t[s] for s in ALR_SEGMENTAL},
        "ALR_3": t.alr_central3, "ALR_10": t.alr_10, "ALR_10_n_branches": t.alr_n_branches})
    wide = gl.pivot(index="case_id", columns="generation")
    wide.columns = [f"G{g}_{m}" for m, g in wide.columns]
    order = [f"G{g}_{m}" for g in ALL_GENS for m in ("branch_count", "diameter_mean_mm", "diameter_sd_mm",
                                                       "diameter_median_mm", "diameter_mean_norm_mm_per_L_cbrt")]
    wide = wide.reindex(columns=order).reset_index()
    wide = pat[["case_id", "age_years", "sex"]].merge(wide, on="case_id")
    readme = pd.DataFrame({"item": [
        "Cohort", "Generation", "Calliper diameter", "EDT diameter", "Normalised diameter", "ALR-3", "ALR-10",
        "Generation statistics", "Missing values", "Correction", "Ages"], "definition": [
        "90 CT scans, one per patient, 10 per 2-year age group, 0-18 years.",
        "Topologic generation from the AirMorph-based tree, generation 1 = trachea, 2 = main bronchi, 3 = lobar bronchi.",
        "Mean of short and long axes on planes perpendicular to the centreline (trachea, main bronchi), as in the manuscript.",
        "Distance-transform diameter: twice the mean lumen radius sampled along the branch centreline.",
        "Diameter divided by the cube root of total segmented lung volume in litres (mm/L^1/3).",
        "Geometric mean of the calliper diameters of trachea, LMB and RMB (cm) / cube root of lung volume (cm^3).",
        "Geometric mean of the EDT diameters of trachea, LMB, RMB and the proximal-most labelled RB1, RB4, RB6, RB10, LB1+2, LB6, LB10 (cm) / cube root of lung volume (cm^3); undetected branches omitted.",
        "Per scan and generation: number of branches, and mean / SD / median of the branch EDT diameters (each branch counts once); SD is empty when the generation has one branch.",
        "Empty = not detected or not measurable.",
        "sub-0261_acc-0081 and sub-0534_acc-0193: the LMB was re-assigned from a carina stub to the true left main bronchus (manual QC).",
        "Ages are whole years for 2 years and older, months/12 below 2 years (as recorded). Two scans are recorded as 18 years."]})
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        pat.to_excel(xw, sheet_name="Per patient", index=False)
        wide.to_excel(xw, sheet_name="Per generation (wide)", index=False)
        gl.merge(pat[["case_id", "age_years", "age_group", "sex"]], on="case_id").to_excel(xw, sheet_name="Per generation (long)", index=False)
        st.to_excel(xw, sheet_name="Slopes by sex", index=False)
        summ.to_excel(xw, sheet_name="Branch counts by age group", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "B2"
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(48, max(10, len(str(col[0].value)) + 2))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t = cont.case_table()
    st = sex_tables(t)
    fig_alr_by_sex(t, st, os.path.join(a.out, "Figure_A_ALR_by_sex"))
    fig_generation_by_sex(st, os.path.join(a.out, "Figure_B_generation_slopes_by_sex"))
    long, summ = branch_counts(t)
    fig_branch_counts(summ, os.path.join(a.out, "Figure_C_branches_by_generation"))
    gl = generation_long(t)
    workbook(t, gl, st, long, summ, os.path.join(a.out, "airway_cohort90_per_patient.xlsx"))
    print(st[st.outcome.str.startswith("ALR")].round(6).to_string(index=False))
    print(summ.pivot(index="generation", columns="age_group", values="median_pct_of_expected").round(0).to_string())


if __name__ == "__main__":
    main()
