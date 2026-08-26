"""
Analysis for "Differences in Explanations for Advertisers versus Viewers".

Reproduces the paper's Results section:
  - RQ1 (Perspective) & RQ2 (Source): Mentioned / Explicit feature counts
  - RQ3 (Source) & RQ4 (scenario user attributes): per-label logistic models
  - Inter-rater agreement (Cohen's kappa, % agreement, Gwet's AC1)
  - Appendix robustness checks (Benjamini-Yekutieli, Mann-Whitney U)

Run:
    pip install -r requirements.txt
    python analysis.py

All input data lives in ./data (see README.md for the column dictionary).
Participant identifiers have been anonymized.
"""
import ast
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import cohen_kappa_score

DATA_DIR = "data"

# The eight framing labels, named as in the paper.
LABELS = ["content_based", "user_based", "qualities", "uniqueness",
          "emotions", "lifestyle", "advertising", "demographics"]
# Balanced labels analysed in the main models; the rest have skewed base rates
# and are reported as exploratory.
PRIMARY_LABELS = ["lifestyle", "qualities", "uniqueness", "emotions", "advertising"]

# Reference categories for the (treatment-coded) logistic models.
REF = dict(source="gemma2", perspective="advertiser", gender="F", age="18_24")

_parse = lambda x: ast.literal_eval(x) if isinstance(x, str) else []


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_explanations():
    """All 1,152 explanations (both human sessions + both LLMs) with labels."""
    df = pd.read_csv(f"{DATA_DIR}/explanations.csv")
    df["features_num"] = df["features_num"].apply(ast.literal_eval)
    df["explicit"] = df["explicit"].apply(ast.literal_eval)
    # 'Mentioned' for the LLM rows = number of coded features present in the text.
    df["mentioned"] = df["features_num"].apply(len)
    df["explicit_n"] = df["explicit"].apply(sum)
    return df


def human_feature_counts():
    """Human 'Mentioned'/'Explicit' counts, per creation session.

    'Mentioned' for humans is the CORRESPONDENCE count (features the participant
    both selected AND verbalised), i.e. the sum of the per-feature correspondence
    flags, NOT the number of features selected.
    """
    h = pd.read_csv(f"{DATA_DIR}/human_study.csv")
    out = {}
    for sfx, source in [("s1", "human_session_1"), ("s2", "human_session_2")]:
        d = pd.DataFrame()
        d["perspective"] = h[f"perspective_{sfx}"].str.lower()
        d["mentioned"] = h[f"corresponding_{sfx}"].apply(lambda x: int(sum(_parse(x))))
        d["explicit_n"] = h[f"explicit_{sfx}"].apply(lambda x: int(sum(_parse(x))))
        out[source] = d
    return out


def feature_groups(df):
    """The four sources, each with 'mentioned' and 'explicit_n' columns."""
    hc = human_feature_counts()
    return {
        "gemma2": df[df.source == "gemma2"].assign(perspective=df.perspective),
        "gpt-4o": df[df.source == "gpt-4o"].assign(perspective=df.perspective),
        "human_session_1": hc["human_session_1"],
        "human_session_2": hc["human_session_2"],
    }


# --------------------------------------------------------------------------- #
# RQ1 / RQ2: feature-count comparisons (Welch t-tests)
# --------------------------------------------------------------------------- #
def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else 0.0


def source_comparisons(groups, col):
    """Pairwise Welch t-tests across the four sources, BH-corrected."""
    src = list(groups)
    rows, pvals = [], []
    for i in range(len(src)):
        for j in range(i + 1, len(src)):
            a, b = groups[src[i]][col], groups[src[j]][col]
            t, p = ttest_ind(a, b, equal_var=False)          # Welch
            rows.append((src[i], src[j], a.mean() - b.mean(), cohens_d(a, b), t))
            pvals.append(p)
    out = pd.DataFrame(rows, columns=["A", "B", "delta", "d", "t"])
    out["p_BH"] = multipletests(pvals, method="fdr_bh")[1]
    return out


def perspective_within_llm(groups, col):
    """Advertiser vs viewer within each LLM, BH-corrected across the two models."""
    rows, pvals = [], []
    for model in ["gemma2", "gpt-4o"]:
        d = groups[model]
        a = d[d.perspective == "advertiser"][col]
        b = d[d.perspective == "viewer"][col]
        t, p = ttest_ind(a, b, equal_var=False)
        rows.append((model, t, cohens_d(a, b)))
        pvals.append(p)
    out = pd.DataFrame(rows, columns=["model", "t", "d"])
    out["p_BH"] = multipletests(pvals, method="fdr_bh")[1]
    return out


# --------------------------------------------------------------------------- #
# RQ3 / RQ4: pre-specified per-label logistic models
# --------------------------------------------------------------------------- #
def logistic_models(df):
    """One fixed logistic model per label (no stepwise selection):

        label ~ source + perspective + gender + age + source:perspective

    p-values are BH-corrected jointly across every non-intercept term of every
    label. Returns (results_dict, bh_map).
    """
    d = df.copy()
    d["age"] = d["age"].str.replace("-", "_").str.replace("+", "plus")  # patsy-safe
    formula = (
        "{y} ~ C(source, Treatment('{s}')) + C(perspective, Treatment('{p}')) "
        "+ C(gender, Treatment('{g}')) + C(age, Treatment('{a}')) "
        "+ C(source, Treatment('{s}')):C(perspective, Treatment('{p}'))"
    ).format(y="{y}", s=REF["source"], p=REF["perspective"],
             g=REF["gender"], a=REF["age"])

    results, keys, pvals = {}, [], []
    for y in LABELS:
        m = smf.logit(formula.format(y=y), data=d).fit(disp=0)
        results[y] = m
        for term in m.params.index:
            if term == "Intercept":
                continue
            keys.append((y, term))
            pvals.append(m.pvalues[term])
    bh = multipletests(pvals, method="fdr_bh")[1]
    bh_map = dict(zip(keys, bh))
    return results, bh_map


# --------------------------------------------------------------------------- #
# Inter-rater agreement
# --------------------------------------------------------------------------- #
def gwet_ac1(x, y):
    """Gwet's AC1 for two binary raters (prevalence-robust)."""
    x, y = np.asarray(x), np.asarray(y)
    po = (x == y).mean()
    pi = (x.mean() + y.mean()) / 2.0
    pe = 2 * pi * (1 - pi)
    return (po - pe) / (1 - pe) if (1 - pe) > 0 else 1.0


def inter_rater():
    a = pd.read_csv(f"{DATA_DIR}/annotator_a.csv")
    b = pd.read_csv(f"{DATA_DIR}/annotator_b.csv")
    m = a.merge(b, on=["batch", "participant_id"], suffixes=("_1", "_2"))
    rows = []
    for lab in LABELS:
        x, y = m[f"{lab}_1"].astype(int), m[f"{lab}_2"].astype(int)
        rows.append((lab, (x == y).mean(), cohen_kappa_score(x, y), gwet_ac1(x, y)))
    return pd.DataFrame(rows, columns=["label", "pct_agree", "kappa", "AC1"])


# --------------------------------------------------------------------------- #
# Appendix robustness checks
# --------------------------------------------------------------------------- #
def by_robustness(df, bh_map):
    """Terms significant under BH but not under Benjamini-Yekutieli."""
    _, _ = df, bh_map
    d = df.copy()
    d["age"] = d["age"].str.replace("-", "_").str.replace("+", "plus")
    formula = (
        "{y} ~ C(source, Treatment('{s}')) + C(perspective, Treatment('{p}')) "
        "+ C(gender, Treatment('{g}')) + C(age, Treatment('{a}')) "
        "+ C(source, Treatment('{s}')):C(perspective, Treatment('{p}'))"
    ).format(y="{y}", s=REF["source"], p=REF["perspective"],
             g=REF["gender"], a=REF["age"])
    keys, pvals = [], []
    for y in LABELS:
        m = smf.logit(formula.format(y=y), data=d).fit(disp=0)
        for term in m.params.index:
            if term == "Intercept":
                continue
            keys.append((y, term))
            pvals.append(m.pvalues[term])
    by = dict(zip(keys, multipletests(pvals, method="fdr_by")[1]))
    flagged = [(k, bh_map[k], by[k]) for k in keys
               if bh_map[k] < 0.05 <= by[k]]
    return pd.DataFrame(flagged, columns=["label_term", "p_BH", "p_BY"])


def mannwhitney(groups, col):
    """Rank-based robustness check for the feature-count comparisons."""
    src = list(groups)
    rows = []
    for i in range(len(src)):
        for j in range(i + 1, len(src)):
            a, b = groups[src[i]][col], groups[src[j]][col]
            _, p_w = ttest_ind(a, b, equal_var=False)
            _, p_u = mannwhitneyu(a, b, alternative="two-sided")
            rows.append((f"{src[i]} vs {src[j]}", p_w, p_u,
                         "disagree" if (p_w < .05) != (p_u < .05) else ""))
    return pd.DataFrame(rows, columns=["comparison", "welch_p", "mwu_p", "flag"])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    pd.set_option("display.width", 120)
    df = load_explanations()
    groups = feature_groups(df)

    print(f"Loaded {len(df)} explanations "
          f"({', '.join(sorted(df.source.unique()))}).\n")

    print("Group means (Mentioned / Explicit):")
    for k, g in groups.items():
        print(f"  {k:16} {g['mentioned'].mean():.2f} / {g['explicit_n'].mean():.2f}")

    print("\n=== RQ1/RQ2  Mentioned features (source contrasts) ===")
    print(source_comparisons(groups, "mentioned").round(3).to_string(index=False))
    print("\n=== RQ1/RQ2  Explicit features (source contrasts) ===")
    print(source_comparisons(groups, "explicit_n").round(3).to_string(index=False))

    print("\n=== RQ1  Perspective within each LLM ===")
    for col in ["mentioned", "explicit_n"]:
        print(f"[{col}]")
        print(perspective_within_llm(groups, col).round(3).to_string(index=False))

    print("\n=== RQ3/RQ4  Logistic models (significant terms, BH<.05) ===")
    results, bh_map = logistic_models(df)
    for y in LABELS:
        m = results[y]
        sig = [(t, np.exp(m.params[t]), bh_map[(y, t)])
               for t in m.params.index
               if t != "Intercept" and bh_map[(y, t)] < 0.05]
        if sig:
            tag = "primary" if y in PRIMARY_LABELS else "exploratory"
            print(f"[{y}] ({tag}, McFadden R2={m.prsquared:.3f})")
            for t, orr, p in sig:
                print(f"    {t:52} OR={orr:5.2f}  p_BH={p:.3f}")

    print("\n=== Inter-rater agreement ===")
    print(inter_rater().round(2).to_string(index=False))

    print("\n=== Appendix: terms significant under BH but NOT under BY ===")
    by = by_robustness(df, bh_map)
    print("  none" if by.empty else by.round(3).to_string(index=False))

    print("\n=== Appendix: Welch vs Mann-Whitney U (Mentioned) ===")
    print(mannwhitney(groups, "mentioned").round(4).to_string(index=False))

    print("\n=== Appendix: Welch vs Mann-Whitney U (Explicit) ===")
    print(mannwhitney(groups, "explicit_n").round(4).to_string(index=False))
 


if __name__ == "__main__":
    main()
