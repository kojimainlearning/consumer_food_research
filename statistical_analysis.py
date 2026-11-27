import os
from contextlib import redirect_stdout
import pandas as pd
import numpy as np
from itertools import combinations
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_TXT_PATH = os.path.join(RESULTS_DIR, "statistical_analysis_results.txt")

# Load cohort CSVs
df_comp = pd.read_csv(os.path.join(RESULTS_DIR, "cohort_survey_eeg_comparison.csv"))
df_stat = pd.read_csv(os.path.join(RESULTS_DIR, "cohort_stationary_summary.csv"))

# SHAPIRO-WILK NORMALITY TEST (Per metric, per condition)
def check_normality(df, metric, conditions):
    print(f"\n--- Shapiro-Wilk Normality: {metric} ---")
    all_normal = True
    for cond in conditions:
        vals = df.loc[df["Condition"] == cond, metric].dropna()
        w_stat, p_val = stats.shapiro(vals)
        is_norm = p_val > 0.05
        all_normal = all_normal and is_norm
        print(f"  {cond:10s}: W = {w_stat:.4f}, p = {p_val:.4f} "
              f"({'Normal' if is_norm else 'NON-NORMAL'})")
    return all_normal


# OMNIBUS TEST: RM-ANOVA (Parametric) vs FRIEDMAN (Non-Parametric)

def run_omnibus_tests(df, metric, conditions):
    sub = df[df["Condition"].isin(conditions)].dropna(subset=[metric]).copy()
    
    # Parametric: One-Way Repeated-Measures ANOVA
    aov = AnovaRM(data=sub, depvar=metric, subject="Participant", within=["Condition"]).fit()
    f_val = aov.anova_table["F Value"].iloc[0]
    df1   = aov.anova_table["Num DF"].iloc[0]
    df2   = aov.anova_table["Den DF"].iloc[0]
    p_aov = aov.anova_table["Pr > F"].iloc[0]
    
    # Non-Parametric: Friedman Test + Kendall's W effect size
    piv = sub.pivot(index="Participant", columns="Condition", values=metric)[conditions]
    chi2, p_fr = stats.friedmanchisquare(*[piv[c] for c in conditions])
    n_subj, k_cond = piv.shape
    kendall_w = chi2 / (n_subj * (k_cond - 1))
    
    print(f"  RM-ANOVA : F({df1:.0f}, {df2:.0f}) = {f_val:.3f}, p = {p_aov:.4f}")
    print(f"  Friedman : chi2 = {chi2:.3f}, p = {p_fr:.4f}, Kendall's W = {kendall_w:.3f}")
    return piv

# WILCOXON SIGNED-RANK + BENJAMINI-HOCHBERG FDR & BONFERRONI CORRECTION
def run_wilcoxon_vs_zero(df, metric, conditions):
    # Tests whether each condition's Delta shifts significantly from 0 (Pre-Touch Rest)
    rows = []
    for cond in conditions:
        vals = df.loc[df["Condition"] == cond, metric].dropna().values
        w_stat, p_raw = stats.wilcoxon(vals)
        rows.append({
            "Condition": cond,
            "Mean": np.mean(vals),
            "SEM": stats.sem(vals),
            "Median": np.median(vals),
            "IQR": np.percentile(vals, 75) - np.percentile(vals, 25),
            "W": w_stat,
            "p_raw": p_raw
        })
    res = pd.DataFrame(rows)

    # Apply Multiple-Comparison Corrections across the family of conditions
    res["p_BH_FDR"]     = multipletests(res["p_raw"], alpha=0.05, method="fdr_bh")[1]
    res["p_Bonferroni"] = multipletests(res["p_raw"], alpha=0.05, method="bonferroni")[1]
    print(f"\n--- One-Sample Wilcoxon vs. 0: {metric} ---")
    print(res.round(4).to_string(index=False))
    return res

def run_pairwise_wilcoxon(piv, metric):
    # Post-hoc pairwise comparisons between every pair of conditions
    rows = []
    for c1, c2 in combinations(piv.columns, 2):
        w_stat, p_raw = stats.wilcoxon(piv[c1], piv[c2])
        rows.append({"Pair": f"{c1} vs {c2}", "W": w_stat, "p_raw": p_raw})
    res = pd.DataFrame(rows)
    res["p_BH_FDR"]     = multipletests(res["p_raw"], alpha=0.05, method="fdr_bh")[1]
    res["p_Bonferroni"] = multipletests(res["p_raw"], alpha=0.05, method="bonferroni")[1]
    print(f"\n--- Post-Hoc Pairwise Wilcoxon: {metric} ---")
    print(res.round(4).to_string(index=False))
    return res


with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as log_file, redirect_stdout(log_file):
    eeg_conds = ["eau", "the_vert", "biscuit", "chocolat"]
    for m in ["PostClean_Delta_FAA_F4_F3", "PostClean_Delta_AB_FC", "PostClean_Delta_EI"]:
        is_norm = check_normality(df_comp, m, eeg_conds)
        piv = run_omnibus_tests(df_comp, m, eeg_conds)
        run_wilcoxon_vs_zero(df_comp, m, eeg_conds)
        run_pairwise_wilcoxon(piv, m)

    survey_conds = ["the_vert", "biscuit", "chocolat"]
    for m in ["Survey_Delicious", "Survey_Happy", "Survey_Calm"]:
        check_normality(df_comp, m, survey_conds)
        piv = run_omnibus_tests(df_comp, m, survey_conds)
        run_pairwise_wilcoxon(piv, m)

    # Pre-Touch vs Post-Swallow Alpha Surge + FDR
    pre  = df_stat[df_stat["Phase"] == "2_Pre_Touch_Rest"].set_index(["Participant", "Condition"])
    post = df_stat[df_stat["Phase"] == "4b_Post_Swallow_Clean_Recovery"].set_index(["Participant", "Condition"])

    alpha_rows = []
    for cond in eeg_conds:
        v_pre  = pre.xs(cond, level="Condition")["Alpha_FC_uV2"]
        v_post = post.xs(cond, level="Condition")["Alpha_FC_uV2"]
        w_stat, p_raw = stats.wilcoxon(v_post, v_pre)
        med_pct = np.median((v_post - v_pre) / v_pre * 100)
        alpha_rows.append({"Condition": cond, "Median_%_Change": med_pct, "W": w_stat, "p_raw": p_raw})

    df_alpha = pd.DataFrame(alpha_rows)
    df_alpha["p_BH_FDR"]     = multipletests(df_alpha["p_raw"], method="fdr_bh")[1]
    df_alpha["p_Bonferroni"] = multipletests(df_alpha["p_raw"], method="bonferroni")[1]
    print("\n--- Frontal Alpha Power (Pre-Touch vs Post-Swallow) ---")
    print(df_alpha.round(4).to_string(index=False))

    # CORRELATION: Spearman Rank (Survey vs EEG)
    non_water = df_comp[df_comp["Condition"] != "eau"]
    rho, p_corr = stats.spearmanr(non_water["Survey_Happy"], non_water["PostClean_Delta_FAA_F4_F3"])
    print(f"\nSpearman (Survey_Happy vs PostClean_Delta_FAA): rho = {rho:+.3f}, p = {p_corr:.4f}")

print(f"\nSaved statistical analysis results to: {OUTPUT_TXT_PATH}")