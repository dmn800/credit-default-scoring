import numpy as np
import pandas as pd


def coarse_quantile_binning(df, feature, target, n_bins, min_bin_share):
    tmp = df[[feature, target]].copy()
    tmp["bin"] = pd.qcut(tmp[feature], q=n_bins, duplicates="drop")

    agg = (
        tmp.groupby("bin", observed=False)
        .agg(total=(target, "count"), bads=(target, "sum"))
        .reset_index()
    )

    agg["bad_rate"] = agg["bads"] / agg["total"]
    agg["bin_share"] = agg["total"] / agg["total"].sum()

    return agg[agg["bin_share"] >= min_bin_share].reset_index(drop=True)


def enforce_monotonicity(agg):
    agg = agg.copy().reset_index(drop=True)

    corr = np.corrcoef(np.arange(len(agg)), agg["bad_rate"])[0, 1]
    direction = "increasing" if corr >= 0 else "decreasing"

    def is_monotonic(x):
        diffs = x.diff().dropna()
        return (diffs >= 0).all() if direction == "increasing" else (diffs <= 0).all()

    while not is_monotonic(agg["bad_rate"]):
        diffs = agg["bad_rate"].diff().abs()
        idx = diffs[1:].idxmin()

        agg.loc[idx - 1, "total"] += agg.loc[idx, "total"]
        agg.loc[idx - 1, "bads"] += agg.loc[idx, "bads"]

        agg = agg.drop(idx).reset_index(drop=True)
        agg["bad_rate"] = agg["bads"] / agg["total"]

    return agg


def extract_bin_edges(agg):
    """
    Extract numerical bin edges from final bins.
    """
    edges = [agg["bin"].iloc[0].left]
    edges += [b.right for b in agg["bin"]]
    return edges


def apply_bins(df, feature, bin_edges):
    """
    Apply precomputed bin edges to new data.
    """
    return pd.cut(
        df[feature],
        bins=bin_edges,
        include_lowest=True,
        right=True,
    )
