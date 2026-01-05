import pandas as pd
from .binning import (
    coarse_quantile_binning,
    enforce_monotonicity,
    extract_bin_edges,
    apply_bins
)
from .iv import calculate_iv


def feature_quality_report(df, features, target, n_bins, min_bin_share):
    rows = []

    for feat in features:
        agg = coarse_quantile_binning(
            df, feat, target, n_bins, min_bin_share
        )
        final_bins = enforce_monotonicity(agg)

        rows.append(
            {
                "feature": feat,
                "iv": calculate_iv(final_bins),
                "bins": final_bins.shape[0],
                "direction": "↑"
                if final_bins["bad_rate"].iloc[-1]
                > final_bins["bad_rate"].iloc[0]
                else "↓",
            }
        )

    return pd.DataFrame(rows).sort_values("iv", ascending=False)


def iv_train_test_report(df_train, df_test, features, target, n_bins, min_bin_share):
    rows = []

    for feat in features:
        # train binning
        coarse = coarse_quantile_binning(
            df_train, feat, target, n_bins, min_bin_share
        )
        final_bins = enforce_monotonicity(coarse)
        iv_train = calculate_iv(final_bins)

        # extract bin edges
        edges = extract_bin_edges(final_bins)

        # apply to test
        test_bins = apply_bins(df_test, feat, edges)

        tmp = df_test[[feat, target]].copy()
        tmp["bin"] = test_bins

        agg_test = (
            tmp.groupby("bin", observed=True)
            .agg(total=(target, "count"), bads=(target, "sum"))
            .reset_index()
        )

        agg_test["bad_rate"] = agg_test["bads"] / agg_test["total"]
        iv_test = calculate_iv(agg_test)

        rows.append(
            {
                "feature": feat,
                "iv_train": iv_train,
                "iv_test": iv_test,
                "iv_drop_pct": 1 - iv_test / iv_train
                if iv_train > 0
                else None,
            }
        )

    return pd.DataFrame(rows).sort_values("iv_train", ascending=False)