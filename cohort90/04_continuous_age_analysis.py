#!/usr/bin/env python
"""Continuous-age, lung-volume and dysanapsis analysis of the 90-scan cohort.

Requested for the morphometry manuscript (2026-10-06): scatter plots and
regressions of central-airway diameter against age and lung volume, the same
for lung-volume-normalised diameter, age association per generation G1-G10, a
joint model with generation and age, and the airway-to-lung ratio (ALR, the
dysanapsis index).

Inputs are the tables the report was built from (AIRWAY_RESULTS_90/csv), so
every number here is on the same measurements as Tables 1-3 of the manuscript:
central-airway diameters are the calliper (mean-axis) diameters, generation
diameters the distance-transform diameters, lung volume the LungMask volume.

    python 04_continuous_age_analysis.py --out <dir>

Writes case_table.csv, regression tables (CSV + a Word-ready .txt), and the
figures as PDF + PNG.
"""
import argparse, os, warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("AIRWAY_ROOT", "/fileserver/external/quin/ege")
CSV = os.path.join(ROOT, "cohort90_results", "AIRWAY_RESULTS_90", "csv")
GENS = list(range(1, 11))
# the central-airway set for the dysanapsis index: the trachea, both main
# bronchi and the segmental bronchi that are labelled in every scan and sit
# proximally in their lobe (the branch set of the MESA Lung protocol, as far as
# our labels allow; no separate bronchus-intermedius or lobar label exists)
ALR_SEGMENTAL = ["RB1", "RB4", "RB6", "RB10", "LB1+2", "LB6", "LB10"]
SEX_COL = {"Male": "#2A78D6", "Female": "#D9822B"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


# ----------------------------------------------------------------- data

def case_table():
    case = pd.read_csv(os.path.join(CSV, "case_level_measurements.csv"))
    gen = pd.read_csv(os.path.join(CSV, "per_scan_generation_metrics.csv"))
    br = pd.read_csv(os.path.join(CSV, "branch_level_generation_1_to_14.csv"), low_memory=False)
    t = pd.DataFrame({
        "case_id": case.case_id, "age": case.age_years, "age_group": case.age_bucket, "sex": case.sex,
        "tlv_ml": case.total_segmented_lung_volume_mL,
        "trachea": case.trachea_mean_axis_diameter_mm, "lmb": case.lmb_mean_axis_diameter_mm,
        "rmb": case.rmb_mean_axis_diameter_mm,
        "trachea_edt": case.raw_root_edt_mean_diameter_mm, "lmb_edt": case.lmb_edt_mean_diameter_mm,
        "rmb_edt": case.rmb_edt_mean_diameter_mm,
    })
    t["tlv_l_cbrt"] = (t.tlv_ml / 1000.0) ** (1 / 3)              # L^(1/3), the manuscript's scale
    for k in ("trachea", "lmb", "rmb"):
        t[f"{k}_norm"] = t[k] / t.tlv_l_cbrt                       # mm / L^(1/3)
    # generation diameters: unweighted mean over the branches of that generation
    g = gen[gen.generation.isin(GENS)].pivot(index="case_id", columns="generation", values="mean_diameter_mm")
    g.columns = [f"g{c}" for c in g.columns]
    t = t.merge(g, left_on="case_id", right_index=True, how="left")
    for c in GENS:
        t[f"g{c}_norm"] = t[f"g{c}"] / t.tlv_l_cbrt
    # dysanapsis: proximal-most labelled branch of each segmental bronchus
    seg = br[br.segmental_name.isin(ALR_SEGMENTAL) & (br.branch_name == br.segmental_name)]
    seg = seg.sort_values("generation").groupby(["case_id", "segmental_name"]).first().reset_index()
    sp = seg.pivot(index="case_id", columns="segmental_name", values="diameter_mean_mm")
    t = t.merge(sp, left_on="case_id", right_index=True, how="left")
    cm = lambda mm: mm / 10.0
    tlv_cm = t.tlv_ml ** (1 / 3)                                   # cm, TLV in cm^3 = mL
    t["alr_central3"] = np.exp(np.log(cm(t[["trachea", "lmb", "rmb"]])).mean(axis=1)) / tlv_cm
    ten = ["trachea_edt", "lmb_edt", "rmb_edt"] + ALR_SEGMENTAL
    logs = np.log(cm(t[ten]))
    t["alr_n_branches"] = logs.notna().sum(axis=1)
    t["alr_10"] = np.exp(logs.mean(axis=1, skipna=True)) / tlv_cm
    t["age_group"] = pd.Categorical(t.age_group, ["0-2", "2-4", "4-6", "6-8", "8-10", "10-12", "12-14", "14-16", "16-18"], ordered=True)
    return t


# ----------------------------------------------------------------- statistics

def ols(df, formula):
    m = smf.ols(formula, data=df).fit()
    return m


def row(name, m, term, df=None, y=None, x=None):
    ci = m.conf_int().loc[term]
    r = {"outcome": name, "model": m.model.formula, "term": term, "coef": m.params[term],
         "ci_low": ci[0], "ci_high": ci[1], "p": m.pvalues[term], "r2": m.rsquared, "n": int(m.nobs)}
    if df is not None and y is not None and x is not None:
        rho, p = stats.spearmanr(df[x], df[y], nan_policy="omit")
        r["spearman_rho"] = rho; r["spearman_p"] = p
    return r


def central_regressions(t):
    rows = []
    for k, lab in (("trachea", "Trachea"), ("lmb", "LMB"), ("rmb", "RMB")):
        d = t.dropna(subset=[k])
        rows.append(row(f"{lab} diameter (mm)", ols(d, f"{k} ~ age"), "age", d, k, "age"))
        rows.append(row(f"{lab} diameter (mm)", ols(d, f"{k} ~ tlv_l_cbrt"), "tlv_l_cbrt", d, k, "tlv_ml"))
        m = ols(d.assign(logd=np.log(d[k]), logv=np.log(d.tlv_ml)), "logd ~ logv")
        rows.append(row(f"{lab} diameter, allometric exponent (log d ~ log TLV)", m, "logv"))
        m = ols(d, f"{k} ~ age + C(sex)")
        rows.append(row(f"{lab} diameter (mm), age + sex", m, "age"))
        rows.append(row(f"{lab} diameter (mm), age + sex", m, "C(sex)[T.Male]"))
        m = ols(d, f"{k} ~ age + tlv_l_cbrt")
        rows.append(row(f"{lab} diameter (mm), age + lung scale", m, "age"))
        rows.append(row(f"{lab} diameter (mm), age + lung scale", m, "tlv_l_cbrt"))
        rows.append(row(f"{lab} normalised (mm/L^1/3)", ols(d, f"{k}_norm ~ age"), "age", d, f"{k}_norm", "age"))
        m = ols(d, f"{k}_norm ~ age + C(sex)")
        rows.append(row(f"{lab} normalised (mm/L^1/3), age + sex", m, "age"))
        rows.append(row(f"{lab} normalised (mm/L^1/3), age + sex", m, "C(sex)[T.Male]"))
    return pd.DataFrame(rows)


def generation_regressions(t):
    rows = []
    for g in GENS:
        d = t.dropna(subset=[f"g{g}"])
        for kind, y in (("absolute", f"g{g}"), ("normalised", f"g{g}_norm")):
            m = ols(d, f"{y} ~ age")
            r = row(f"G{g} {kind}", m, "age", d, y, "age")
            r["generation"] = g; r["kind"] = kind
            r["mean"] = d[y].mean(); r["slope_pct_per_year"] = 100 * m.params["age"] / d[y].mean()
            rows.append(r)
    return pd.DataFrame(rows)


def joint_models(t):
    """Long format: one row per scan x generation; log diameter so that the age
    effect is a percentage and the generation effect a constant ratio."""
    long = t.melt(id_vars=["case_id", "age", "sex", "tlv_l_cbrt"], value_vars=[f"g{g}" for g in GENS],
                  var_name="gen", value_name="d").dropna()
    long["generation"] = long.gen.str[1:].astype(int)
    long["logd"] = np.log(long.d); long["lognorm"] = np.log(long.d / long.tlv_l_cbrt)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m1 = smf.mixedlm("logd ~ age + C(generation)", long, groups=long.case_id).fit(reml=False)
        m2 = smf.mixedlm("logd ~ age * C(generation)", long, groups=long.case_id).fit(reml=False)
        m3 = smf.mixedlm("lognorm ~ age + C(generation)", long, groups=long.case_id).fit(reml=False)
        m4 = smf.mixedlm("logd ~ age + C(sex) + C(generation)", long, groups=long.case_id).fit(reml=False)
    lrt = 2 * (m2.llf - m1.llf); df = m2.df_modelwc - m1.df_modelwc
    out["age_pct_per_year"] = 100 * (np.exp(m1.params["age"]) - 1)
    ci = m1.conf_int().loc["age"]; out["age_pct_ci"] = (100 * (np.exp(ci[0]) - 1), 100 * (np.exp(ci[1]) - 1))
    out["age_p"] = m1.pvalues["age"]
    gcoef = m1.params[[k for k in m1.params.index if k.startswith("C(generation)")]]
    out["generation_ratio_mean"] = float(np.exp(np.diff(np.r_[0, gcoef.values]).mean()))   # mean ratio between consecutive generations
    out["interaction_lrt_chi2"] = lrt; out["interaction_lrt_df"] = int(df); out["interaction_lrt_p"] = 1 - stats.chi2.cdf(lrt, df)
    out["norm_age_pct_per_year"] = 100 * (np.exp(m3.params["age"]) - 1)
    ci = m3.conf_int().loc["age"]; out["norm_age_pct_ci"] = (100 * (np.exp(ci[0]) - 1), 100 * (np.exp(ci[1]) - 1))
    out["norm_age_p"] = m3.pvalues["age"]
    out["sex_male_pct"] = 100 * (np.exp(m4.params["C(sex)[T.Male]"]) - 1); out["sex_p"] = m4.pvalues["C(sex)[T.Male]"]
    out["n_obs"] = int(m1.nobs); out["n_subjects"] = long.case_id.nunique()
    return out, long


def alr_tables(t):
    by = t.groupby("age_group", observed=True).agg(
        n=("alr_10", "size"),
        alr10_median=("alr_10", "median"), alr10_q1=("alr_10", lambda s: s.quantile(.25)), alr10_q3=("alr_10", lambda s: s.quantile(.75)),
        alr3_median=("alr_central3", "median"), alr3_q1=("alr_central3", lambda s: s.quantile(.25)), alr3_q3=("alr_central3", lambda s: s.quantile(.75)),
        branches_median=("alr_n_branches", "median")).reset_index()
    assoc = {}
    for k in ("alr_10", "alr_central3"):
        rho, p = stats.spearmanr(t.age, t[k], nan_policy="omit")
        m = ols(t, f"{k} ~ age")
        u = stats.mannwhitneyu(t[t.sex == "Male"][k].dropna(), t[t.sex == "Female"][k].dropna())
        assoc[k] = {"median": t[k].median(), "q1": t[k].quantile(.25), "q3": t[k].quantile(.75),
                    "spearman_rho": rho, "spearman_p": p, "slope_per_year": m.params["age"],
                    "slope_ci": tuple(m.conf_int().loc["age"]), "slope_p": m.pvalues["age"], "r2": m.rsquared,
                    "male_median": t[t.sex == "Male"][k].median(), "female_median": t[t.sex == "Female"][k].median(), "sex_p": u.pvalue}
    return by, assoc


# ----------------------------------------------------------------- figures

def fit_line(ax, x, y, color="#333", label=None):
    m = sm.OLS(y, sm.add_constant(x)).fit()
    xs = np.linspace(x.min(), x.max(), 100)
    pred = m.get_prediction(sm.add_constant(xs)).summary_frame(alpha=0.05)
    ax.plot(xs, pred["mean"], color=color, lw=1.4, label=label)
    ax.fill_between(xs, pred["mean_ci_lower"], pred["mean_ci_upper"], color=color, alpha=0.15, lw=0)
    return m


def scatter_sex(ax, d, x, y):
    for s, c in SEX_COL.items():
        sel = d[d.sex == s]
        ax.scatter(sel[x], sel[y], s=18, c=c, alpha=0.8, edgecolor="white", lw=0.4, label=s)


def fig_vs_age(t, cols, labels, ylabel, out, norm=False):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharex=True, sharey=True)
    for ax, k, lab in zip(axes, cols, labels):
        d = t.dropna(subset=[k])
        scatter_sex(ax, d, "age", k)
        m = fit_line(ax, d.age.values, d[k].values)
        rho, p = stats.spearmanr(d.age, d[k])
        slope, (lo, hi) = m.params[1], m.conf_int()[1]
        unit = "mm/L$^{1/3}$" if norm else "mm"
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("Age (years)"); ax.set_xlim(0, 18.5); ax.set_xticks([0, 5, 10, 15])
    # one linear y range for the three panels, from zero, so that they compare directly
    axes[0].set_ylim(0, np.ceil(1.08 * np.nanmax(t[list(cols)].values)))
    axes[0].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, integer=True))
    for ax in axes:
        ax.tick_params(labelleft=True)            # shared range, but values on every panel
    axes[0].set_ylabel(ylabel); axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300); plt.close(fig)


def fig_vs_volume(t, out):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharex=True, sharey=True)
    for ax, k, lab in zip(axes, ("trachea", "lmb", "rmb"), ("Trachea", "Left main bronchus", "Right main bronchus")):
        d = t.dropna(subset=[k])
        scatter_sex(ax, d, "tlv_ml", k)
        lx, ly = np.log(d.tlv_ml.values), np.log(d[k].values)
        m = sm.OLS(ly, sm.add_constant(lx)).fit()
        xs = np.log(np.linspace(d.tlv_ml.min(), d.tlv_ml.max(), 200))   # evenly spaced on the linear axis
        pred = m.get_prediction(sm.add_constant(xs)).summary_frame(alpha=0.05)
        ax.plot(np.exp(xs), np.exp(pred["mean"]), color="#333", lw=1.4)
        ax.fill_between(np.exp(xs), np.exp(pred["mean_ci_lower"]), np.exp(pred["mean_ci_upper"]), color="#333", alpha=0.15, lw=0)
        ax.plot(np.exp(xs), np.exp(m.params[0]) * np.exp(xs) ** (1 / 3) * np.exp(np.mean(ly - (m.params[0] + lx / 3))), "--", color="#999", lw=1, label="isometric, b = 1/3")
        b, (lo, hi) = m.params[1], m.conf_int()[1]
        rho, p = stats.spearmanr(d.tlv_ml, d[k])
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("Segmented lung volume (mL)")
    # linear axes, one y range for the three panels; the fitted power law is drawn as a curve
    axes[0].set_xlim(0, 2000 * np.ceil(t.tlv_ml.max() / 2000)); axes[0].set_xticks(np.arange(0, axes[0].get_xlim()[1] + 1, 2000))
    axes[0].set_ylim(0, np.ceil(1.08 * np.nanmax(t[["trachea", "lmb", "rmb"]].values)))
    axes[0].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, integer=True))
    for ax in axes:
        ax.tick_params(labelleft=True)
    axes[0].set_ylabel("Diameter (mm)"); axes[0].legend(loc="lower right", fontsize=7.5, frameon=False)
    fig.tight_layout(); fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300); plt.close(fig)


def fig_generation(gr, joint, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, kind, ylabel in ((axes[0], "absolute", "Age slope (mm per year)"), (axes[1], "normalised", "Age slope (mm/L$^{1/3}$ per year)")):
        d = gr[gr.kind == kind].sort_values("generation")
        ax.errorbar(d.generation, d.coef, yerr=[d.coef - d.ci_low, d.ci_high - d.coef], fmt="o", color="#2A78D6", ecolor="#2A78D6", capsize=2, ms=4)
        ax.axhline(0, color="#999", lw=0.8, ls="--")
        ax.set_xticks(GENS); ax.set_xlabel("Topologic generation"); ax.set_ylabel(ylabel)
    axes[0].set_title("Absolute diameter", fontsize=10)
    axes[1].set_title("Lung-volume-normalised diameter", fontsize=10)
    fig.tight_layout(); fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300); plt.close(fig)


def fig_alr(t, assoc, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, k, lab in ((axes[0], "alr_10", "Ten central branches"),
                       (axes[1], "alr_central3", "Trachea and main bronchi")):
        d = t.dropna(subset=[k])
        scatter_sex(ax, d, "age", k)
        fit_line(ax, d.age.values, d[k].values)
        med = d.groupby("age_group", observed=True)[k].median()
        centres = [1, 3, 5, 7, 9, 11, 13, 15, 17]
        ax.plot(centres, med.values, "s-", color="#1B9E6A", ms=4, lw=1, label="age-group median")
        ax.set_title(lab, fontsize=10); ax.set_xlabel("Age (years)"); ax.set_xlim(0, 18.5); ax.set_xticks([0, 5, 10, 15])
        lo_, hi_ = d[k].min(), d[k].max(); ax.set_ylim(lo_ - 0.1 * (hi_ - lo_), hi_ + 0.12 * (hi_ - lo_))
    axes[0].set_ylabel("Airway-to-lung ratio (dimensionless)"); axes[0].legend(loc="upper right", fontsize=7.5, frameon=False)
    fig.tight_layout(); fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=300); plt.close(fig)


# ----------------------------------------------------------------- main

def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "cohort90_results", "continuous_age_2026-10"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t = case_table()
    t.to_csv(os.path.join(a.out, "case_table.csv"), index=False)
    cr = central_regressions(t); cr.to_csv(os.path.join(a.out, "central_regressions.csv"), index=False)
    gr = generation_regressions(t); gr.to_csv(os.path.join(a.out, "generation_regressions.csv"), index=False)
    joint, long = joint_models(t)
    pd.Series(joint).to_csv(os.path.join(a.out, "joint_model.csv"))
    by, assoc = alr_tables(t); by.to_csv(os.path.join(a.out, "alr_by_age_group.csv"), index=False)
    pd.DataFrame(assoc).T.to_csv(os.path.join(a.out, "alr_associations.csv"))

    fig_vs_age(t, ["trachea", "lmb", "rmb"], ["Trachea", "Left main bronchus", "Right main bronchus"], "Diameter (mm)", os.path.join(a.out, "Figure_5_central_diameter_vs_age"))
    fig_vs_age(t, ["trachea_norm", "lmb_norm", "rmb_norm"], ["Trachea", "Left main bronchus", "Right main bronchus"], "Normalised diameter (mm/L$^{1/3}$)", os.path.join(a.out, "Figure_6_normalised_diameter_vs_age"), norm=True)
    fig_vs_volume(t, os.path.join(a.out, "Figure_7_diameter_vs_lung_volume"))
    fig_generation(gr, joint, os.path.join(a.out, "Figure_8_generation_age_slopes"))
    fig_alr(t, assoc, os.path.join(a.out, "Figure_9_airway_to_lung_ratio"))

    # ---- a Word-ready summary
    L = []
    L.append("CENTRAL AIRWAYS: regression of diameter on age, lung volume and sex (n = 90)")
    L.append("outcome | model | term | coefficient (95% CI) | P | R2 | Spearman rho (P)")
    for _, r in cr.iterrows():
        sp = f"{r.spearman_rho:+.2f} ({fmt_p(r.spearman_p)})" if pd.notna(r.get("spearman_rho", np.nan)) else ""
        L.append(f"{r.outcome} | {r.model} | {r.term} | {r.coef:+.3f} ({r.ci_low:+.3f} to {r.ci_high:+.3f}) | {fmt_p(r.p)} | {r.r2:.2f} | {sp}")
    L.append("")
    L.append("GENERATIONS G1-G10: age slope per generation")
    L.append("generation | mean diameter mm | slope mm/y (95% CI) | %/y | P | Spearman rho | normalised slope mm/L^1/3 per y (95% CI) | P")
    for g in GENS:
        A = gr[(gr.generation == g) & (gr.kind == "absolute")].iloc[0]; N = gr[(gr.generation == g) & (gr.kind == "normalised")].iloc[0]
        L.append(f"G{g} | {A['mean']:.2f} | {A.coef:+.3f} ({A.ci_low:+.3f} to {A.ci_high:+.3f}) | {A.slope_pct_per_year:+.1f} | {fmt_p(A.p)} | {A.spearman_rho:+.2f} | {N.coef:+.3f} ({N.ci_low:+.3f} to {N.ci_high:+.3f}) | {fmt_p(N.p)}")
    L.append("")
    L.append("JOINT MIXED MODEL (log diameter ~ age + generation, random intercept per subject, %d observations in %d subjects)" % (joint["n_obs"], joint["n_subjects"]))
    L.append(f"age: {joint['age_pct_per_year']:+.2f}% per year (95% CI {joint['age_pct_ci'][0]:+.2f} to {joint['age_pct_ci'][1]:+.2f}), P {fmt_p(joint['age_p'])}")
    L.append(f"mean diameter ratio between consecutive generations: {joint['generation_ratio_mean']:.3f}")
    L.append(f"age x generation interaction: LRT chi2 {joint['interaction_lrt_chi2']:.1f} on {joint['interaction_lrt_df']} df, P {fmt_p(joint['interaction_lrt_p'])}")
    L.append(f"normalised: age {joint['norm_age_pct_per_year']:+.2f}% per year (95% CI {joint['norm_age_pct_ci'][0]:+.2f} to {joint['norm_age_pct_ci'][1]:+.2f}), P {fmt_p(joint['norm_age_p'])}")
    L.append(f"sex (male vs female, adjusted for age and generation): {joint['sex_male_pct']:+.1f}%, P {fmt_p(joint['sex_p'])}")
    L.append("")
    L.append("AIRWAY-TO-LUNG RATIO (ALR = geometric mean lumen diameter [cm] / TLV^(1/3) [cm])")
    L.append("age group | n | ALR-10 median (Q1, Q3) | ALR-3 median (Q1, Q3)")
    for _, r in by.iterrows():
        L.append(f"{r.age_group} | {int(r.n)} | {r.alr10_median:.4f} ({r.alr10_q1:.4f}, {r.alr10_q3:.4f}) | {r.alr3_median:.4f} ({r.alr3_q1:.4f}, {r.alr3_q3:.4f})")
    for k, lab in (("alr_10", "ALR-10"), ("alr_central3", "ALR-3")):
        a_ = assoc[k]
        L.append(f"{lab}: all median {a_['median']:.4f} (IQR {a_['q1']:.4f}-{a_['q3']:.4f}); Spearman rho with age {a_['spearman_rho']:+.2f} (P {fmt_p(a_['spearman_p'])}); "
                 f"OLS slope {a_['slope_per_year']:+.5f} per year (95% CI {a_['slope_ci'][0]:+.5f} to {a_['slope_ci'][1]:+.5f}), P {fmt_p(a_['slope_p'])}; "
                 f"male median {a_['male_median']:.4f} vs female {a_['female_median']:.4f}, Mann-Whitney P {fmt_p(a_['sex_p'])}")
    L.append(f"branches available for ALR-10 per scan: median {t.alr_n_branches.median():.0f}, min {t.alr_n_branches.min()}, scans with all 10: {(t.alr_n_branches == 10).sum()}")
    open(os.path.join(a.out, "SUMMARY_FOR_MANUSCRIPT.txt"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
