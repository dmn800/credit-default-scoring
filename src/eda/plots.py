import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .binning import (
    coarse_quantile_binning,
    enforce_monotonicity,
    extract_bin_edges,
    apply_bins
)
from .iv import calculate_iv


def plot_kde_by_target(df, features, target, n_cols=3):
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()

    for i, col in enumerate(features):
        sns.kdeplot(
            data=df,
            x=col,
            hue=target,
            common_norm=False,
            fill=True,
            alpha=0.4,
            ax=axes[i]
        )
        axes[i].set_title(col)

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def matrix_correlation(df, features, method="spearman"):
    corr = df[features].corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_title("Correlation Matrix")

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha="right"
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0
    )

    plt.show()


def plot_badrate_binning(df, features, target, n_bins, min_bin_share):
    n_cols = 3
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        agg = coarse_quantile_binning(df, feature, target, n_bins, min_bin_share)
        final_bins = enforce_monotonicity(agg)

        axes[i].plot(range(len(final_bins)), final_bins["bad_rate"], marker="o", linestyle="-", color="b")
        axes[i].set_xticks(range(len(final_bins)))
        axes[i].set_xticklabels(final_bins.index, rotation=45, ha="right")
        axes[i].set_ylabel("Bad Rate")
        axes[i].set_xlabel(feature)
        axes[i].set_title(f"{feature} vs Bad Rate")

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_badrate_stability_grid(df_train, df_test, features, target, n_bins, min_bin_share, n_cols=3):
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows)
    )
    axes = axes.flatten()

    for i, feat in enumerate(features):
        coarse = coarse_quantile_binning(
            df_train, feat, target, n_bins, min_bin_share
        )
        final_bins = enforce_monotonicity(coarse)
        edges = extract_bin_edges(final_bins)

        # train
        train_agg = final_bins.copy()

        # test
        test_bins = apply_bins(df_test, feat, edges)
        tmp = df_test[[feat, target]].copy()
        tmp["bin"] = test_bins

        test_agg = (
            tmp.groupby("bin", observed=True)
            .agg(total=(target, "count"), bads=(target, "sum"))
            .reset_index()
        )
        test_agg["bad_rate"] = test_agg["bads"] / test_agg["total"]

        x = np.arange(len(train_agg))

        axes[i].plot(
            x,
            train_agg["bad_rate"],
            marker="o",
            label="train"
        )
        axes[i].plot(
            x,
            test_agg["bad_rate"],
            marker="o",
            linestyle="--",
            label="test"
        )

        axes[i].set_title(feat)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(
            train_agg["bin"].astype(str),
            rotation=45,
            ha="right",
            fontsize=8,
        )
        axes[i].legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
