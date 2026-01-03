# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# 01_eda.ipynb
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# -----------------------------
# Config
# -----------------------------
TARGET = "default"

NUM_FEATURES = [
    "age", "income", "employment_length", "dti",
    "previous_loans", "delinquencies", "credit_history"
]

CLIP_QUANTILES = (0.01, 0.99)
SUMMARY_QUANTILES = (0.01, 0.5, 0.99)
CORR_THRESHOLD = 0.2
N_BINS = 10

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (6, 4)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv("../data/raw/credit_data.csv")


# -----------------------------
# Basic checks
# -----------------------------
def basic_data_checks(df, target, features):
    assert target in df.columns, "Target column not found"
    missing = set(features) - set(df.columns)
    assert not missing, f"Missing features: {missing}"

    print("Shape:", df.shape)
    print("\nTarget distribution:")
    print(df[target].value_counts(normalize=True))


basic_data_checks(df, TARGET, NUM_FEATURES)


# -----------------------------
# Quantile summary
# -----------------------------
def quantile_summary(df, features, q):
    return (
        df[features]
        .quantile(q)
        .T
        .rename(columns={q[0]: "q01", q[1]: "median", q[2]: "q99"})
    )


quantiles = quantile_summary(df, NUM_FEATURES, SUMMARY_QUANTILES)
display(quantiles)


# -----------------------------
# Missing by target
# -----------------------------
def missing_by_target(df, features, target):
    return (
        df.groupby(target)[features]
        .apply(lambda x: x.isnull().mean())
        .T
        .rename(columns={0: "miss_rate_0", 1: "miss_rate_1"})
    )


missing_stats = missing_by_target(df, NUM_FEATURES, TARGET)
display(missing_stats)


# -----------------------------
# Numeric distributions (clipped)
# -----------------------------
def plot_numeric_distributions(df, features, clip_q):
    n_cols = 3
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()

    for i, col in enumerate(features):
        low, high = df[col].quantile(clip_q)
        sns.histplot(df[col].clip(low, high), bins=40, ax=axes[i])
        axes[i].set_title(f"{col} (clipped)")

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


plot_numeric_distributions(df, NUM_FEATURES, CLIP_QUANTILES)


# -----------------------------
# Feature vs target
# -----------------------------
def plot_vs_target(df, features, target):
    n_cols = 3
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()

    for i, col in enumerate(features):
        sns.boxplot(x=target, y=col, data=df, showfliers=False, ax=axes[i])
        axes[i].set_title(col)

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


plot_vs_target(df, NUM_FEATURES, TARGET)


# -----------------------------
# Target correlations
# -----------------------------
def target_correlations(df, features, target, method="spearman"):
    return (
        df[features + [target]]
        .corr(method=method)[target]
        .drop(target)
        .sort_values(key=lambda x: x.abs(), ascending=False)
    )


corr_target = target_correlations(df, NUM_FEATURES, TARGET)
# display(corr_target[corr_target.abs() > CORR_THRESHOLD])
display(corr_target)


# -----------------------------
# Binning
# -----------------------------
def quantile_binning(df, feature, target, n_bins):
    tmp = df[[feature, target]].copy()
    tmp["bin"] = pd.qcut(tmp[feature], q=n_bins, duplicates="drop")

    agg = (
        tmp.groupby("bin", observed=True)
        .agg(
            total=(target, "count"),
            bads=(target, "sum"),
            bad_rate=(target, "mean")
        )
        .reset_index()
    )
    return agg


def is_monotonic(series):
    diffs = series.diff().dropna()
    return (diffs >= 0).all() or (diffs <= 0).all()


def plot_badrate_binning(df, features, target, n_bins=N_BINS):
    n_cols = 3
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()

    for i, col in enumerate(features):
        agg = quantile_binning(df, col, target, n_bins)

        axes[i].plot(range(len(agg)), agg["bad_rate"], marker="o", linestyle="-", color="b")
        axes[i].set_xticks(range(len(agg)))
        axes[i].set_xticklabels(agg.index, rotation=45, ha="right")
        axes[i].set_ylabel("Bad Rate")
        axes[i].set_xlabel(col)
        axes[i].set_title(f"{col} vs Bad Rate")

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


plot_badrate_binning(df, NUM_FEATURES, TARGET)


# -----------------------------
# IV calculation
# -----------------------------
def calculate_iv(agg):
    eps = 1e-6
    total_good = (agg["total"] - agg["bads"]).sum()
    total_bad = agg["bads"].sum()

    agg = agg.copy()
    agg["dist_good"] = (agg["total"] - agg["bads"]) / (total_good + eps)
    agg["dist_bad"] = agg["bads"] / (total_bad + eps)
    agg["woe"] = np.log((agg["dist_good"] + eps) / (agg["dist_bad"] + eps))
    agg["iv"] = (agg["dist_good"] - agg["dist_bad"]) * agg["woe"]

    return agg["iv"].sum()


# -----------------------------
# Feature quality report
# -----------------------------
def feature_quality_report(df, features, target, n_bins):
    rows = []

    for feat in features:
        agg = quantile_binning(df, feat, target, n_bins)
        rows.append({
            "feature": feat,
            "iv": calculate_iv(agg),
            "monotonic": is_monotonic(agg["bad_rate"]),
            "bins": agg.shape[0]
        })

    return (
        pd.DataFrame(rows)
        .sort_values("iv", ascending=False)
        .reset_index(drop=True)
    )


quality_report = feature_quality_report(df, NUM_FEATURES, TARGET, N_BINS)
display(quality_report)



# %% [markdown]
# ### Feature diagnostics summary:
# * признаки с IV < 0.02 (кандидаты на удаление)
# * немонотонные (требуют биннинга / трансформации)
# * стабильные и монотонные (можно отдавать в модель)

# %%
