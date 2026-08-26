"""
Figure generation for "Differences in Explanations for Advertisers versus Viewers".

Produces the paper's figures from the anonymized data in ./data and writes
PNG + PDF into ./figures:

  feature_counts_source     Mentioned / Explicit feature means by source
  feature_sessions          per-feature counts across stages (selected -> corresponding -> explicit)
  lifestyle_interactions    Lifestyle label by source x {perspective, day type, device}
  uniqueness_emotions       Uniqueness and Emotions labels by source x perspective
  advertising_interactions  Advertising label by perspective x age and source x gender

Count figures use group mean +/- 95% CI; binary-label figures use the observed
proportion +/- 95% Wilson CI. Depends on numpy + pandas + matplotlib only.

NOTE: for the human sources, 'Mentioned' is the CORRESPONDENCE count taken from
human_study.csv (features selected AND verbalised), not the number of features
selected, so the source figure matches the Results section.

Run:  python figures.py
"""
import ast
import math
import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "data"
OUT_DIR = "figures"

# ---------------- consistent theme ----------------
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
SRC_ORDER = ["gemma2", "gpt-4o", "human_session_1", "human_session_2"]
SRC_LABEL = {"gemma2": "Gemma2", "gpt-4o": "GPT-4o",
             "human_session_1": "Human S1\n(generate)", "human_session_2": "Human S2\n(revise)"}
NICE = {"advertiser": "Advertiser", "viewer": "Viewer",
        "Weekend": "Weekend", "Workday": "Workday",
        "Phone": "Phone", "PC": "PC", "TV": "TV",
        "F": "Female", "M": "Male", "U": "Unspecified",
        "18-24": "18–24", "25-44": "25–44", "45+": "45+"}
FEATURES = {1: "Gender", 2: "Age", 3: "Content type", 4: "Genres", 5: "Programs/wk",
            6: "Hours/day", 7: "Type of day", 8: "Time of day", 9: "Device"}

_parse = lambda x: ast.literal_eval(x) if isinstance(x, str) else []


def theme():
    matplotlib.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.titleweight": "bold", "axes.titlesize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight", "savefig.dpi": 300})


def palette(levels):
    return {lv: OKABE[i % len(OKABE)] for i, lv in enumerate(levels)}


def save(fig, base):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{base}.{ext}"))
    plt.close(fig)


# ---------------- stats ----------------
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return ph, max(0, c - h), min(1, c + h)


def mean_ci(a, z=1.96):
    a = np.asarray(a, float)
    m = a.mean()
    se = a.std(ddof=1) / math.sqrt(len(a))
    return m, m - z * se, m + z * se


# ---------------- data ----------------
def load():
    df = pd.read_csv(f"{DATA_DIR}/explanations.csv")
    df["features_num"] = df["features_num"].apply(ast.literal_eval)
    df["explicit"] = df["explicit"].apply(ast.literal_eval)
    df["explicit_n"] = df["explicit"].apply(sum)
    df["mentioned_llm"] = df["features_num"].apply(len)   # valid for LLM rows only
    human = pd.read_csv(f"{DATA_DIR}/human_study.csv")
    return df, human


def mentioned_by_source(df, human):
    """{source: array of Mentioned counts}, human rows using CORRESPONDENCE."""
    out = {
        "gemma2": df[df.source == "gemma2"]["mentioned_llm"].values,
        "gpt-4o": df[df.source == "gpt-4o"]["mentioned_llm"].values,
        "human_session_1": human["corresponding_s1"].apply(lambda x: int(sum(_parse(x)))).values,
        "human_session_2": human["corresponding_s2"].apply(lambda x: int(sum(_parse(x)))).values,
    }
    return out


def explicit_by_source(df, human):
    return {
        "gemma2": df[df.source == "gemma2"]["explicit_n"].values,
        "gpt-4o": df[df.source == "gpt-4o"]["explicit_n"].values,
        "human_session_1": human["explicit_s1"].apply(lambda x: int(sum(_parse(x)))).values,
        "human_session_2": human["explicit_s2"].apply(lambda x: int(sum(_parse(x)))).values,
    }


# ---------------- generic panels ----------------
def panel_counts_from_dict(ax, series, order, title, ylab):
    xs = np.arange(len(order))
    m, lo, hi = [], [], []
    for k in order:
        mm, l, h = mean_ci(series[k])
        m.append(mm); lo.append(mm - l); hi.append(h - mm)
    ax.bar(xs, m, 0.62, yerr=[lo, hi], capsize=3, color=OKABE[0],
           edgecolor="white", linewidth=.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([SRC_LABEL.get(x, x) for x in order], fontsize=8)
    ax.set_ylabel(ylab); ax.set_title(title); ax.grid(axis="y", alpha=.25)
    ax.set_ylim(0, max(4.2, max(np.array(m) + np.array(hi)) * 1.15))


def panel_prop(ax, df, label, gcol, mcol, gorder, morder, title):
    pal = palette(morder); xs = np.arange(len(gorder)); k = len(morder); w = 0.8 / k
    for j, mv in enumerate(morder):
        pt, lo, hi = [], [], []
        for gv in gorder:
            sub = df[(df[gcol] == gv) & (df[mcol] == mv)]
            p, l, h = wilson(int(sub[label].sum()), len(sub))
            pt.append(p); lo.append(p - l); hi.append(h - p)
        off = (j - (k - 1) / 2) * w
        ax.bar(xs + off, pt, w, yerr=[lo, hi], capsize=2.5, color=pal[mv],
               edgecolor="white", linewidth=.5, label=NICE.get(mv, mv))
    ax.set_xticks(xs)
    ax.set_xticklabels([SRC_LABEL.get(x, NICE.get(x, x)) for x in gorder], fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("Proportion with label"); ax.set_title(title)
    ax.grid(axis="y", alpha=.25); ax.legend(frameon=False, fontsize=8, loc="upper right")


# ---------------- figures ----------------
def fig_feature_counts_source(df, human):
    ment = mentioned_by_source(df, human)
    expl = explicit_by_source(df, human)
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3))
    panel_counts_from_dict(axes[0], ment, SRC_ORDER, "(a) Mentioned features", "Mean count")
    panel_counts_from_dict(axes[1], expl, SRC_ORDER, "(b) Explicit features", "Mean count")
    fig.suptitle("Feature use by source", fontsize=10, fontweight="bold")
    fig.tight_layout(); save(fig, "feature_counts_source")


def fig_feature_sessions(human):
    order = list(range(1, 10))

    def counts(sel_col, flag_col):
        c = Counter()
        for _, r in human.iterrows():
            fs = _parse(r[sel_col])
            fl = _parse(r[flag_col]) if flag_col else None
            if flag_col is None:
                for f in fs:
                    c[f] += 1
            else:
                for f, g in zip(fs, fl):
                    if g == 1:
                        c[f] += 1
        return c

    panels = [
        ("Selected", ("selected_s1", None), ("selected_s2", None)),
        ("Corresponding", ("selected_s1", "corresponding_s1"), ("selected_s2", "corresponding_s2")),
        ("Explicit", ("selected_s1", "explicit_s1"), ("selected_s2", "explicit_s2")),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    x = np.arange(len(order)); w = 0.4
    for i, (ax, (title, s1, s2)) in enumerate(zip(axes, panels)):
        c1, c2 = counts(*s1), counts(*s2)
        ax.bar(x - w / 2, [c1[k] for k in order], w, color=OKABE[0],
               edgecolor="white", linewidth=.5, label="Session 1 (Generate)")
        ax.bar(x + w / 2, [c2[k] for k in order], w, color=OKABE[1],
               edgecolor="white", linewidth=.5, label="Session 2 (Revise)")
        ax.set_xticks(x)
        ax.set_xticklabels([FEATURES[k] for k in order], fontsize=8, rotation=40, ha="right")
        ax.set_title(f"({'abc'[i]}) {title}"); ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Feature count")
    axes[2].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Per-feature counts across creation stages: selected → corresponding → explicit",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(); save(fig, "feature_sessions")


def fig_lifestyle_interactions(df):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    panel_prop(axes[0], df, "lifestyle", "source", "perspective", SRC_ORDER,
               ["advertiser", "viewer"], "(a) source × perspective")
    panel_prop(axes[1], df, "lifestyle", "source", "day_type", SRC_ORDER,
               ["Weekend", "Workday"], "(b) source × day type")
    panel_prop(axes[2], df, "lifestyle", "source", "device", SRC_ORDER,
               ["Phone", "PC", "TV"], "(c) source × device")
    for a in axes[1:]:
        a.set_ylabel("")
    fig.suptitle("Lifestyle assumptions: interaction effects", fontsize=10, fontweight="bold")
    fig.tight_layout(); save(fig, "lifestyle_interactions")


def fig_uniqueness_emotions(df):
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4))
    panel_prop(axes[0], df, "uniqueness", "source", "perspective", SRC_ORDER,
               ["advertiser", "viewer"], "(a) Uniqueness")
    panel_prop(axes[1], df, "emotions", "source", "perspective", SRC_ORDER,
               ["advertiser", "viewer"], "(b) Emotions")
    axes[1].set_ylabel("")
    fig.suptitle("Framing labels: source × perspective", fontsize=10, fontweight="bold")
    fig.tight_layout(); save(fig, "uniqueness_emotions")


def fig_advertising_interactions(df):
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4))
    panel_prop(axes[0], df, "advertising", "perspective", "age",
               ["advertiser", "viewer"], ["18-24", "25-44", "45+"], "(a) perspective × age")
    panel_prop(axes[1], df, "advertising", "source", "gender", SRC_ORDER,
               ["F", "M", "U"], "(b) source × gender")
    axes[0].set_xticklabels(["Advertiser", "Viewer"], fontsize=8)
    axes[1].set_ylabel("")
    fig.suptitle("Advertising-practice assumptions: interaction effects", fontsize=10, fontweight="bold")
    fig.tight_layout(); save(fig, "advertising_interactions")


def main():
    theme()
    df, human = load()
    fig_feature_counts_source(df, human)
    fig_feature_sessions(human)
    fig_lifestyle_interactions(df)
    fig_uniqueness_emotions(df)
    fig_advertising_interactions(df)
    print(f"Wrote figures (PNG + PDF) to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
