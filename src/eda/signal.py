import pandas as pd
from scipy.stats import ks_2samp


def ks_report(df, features, target):
    rows = []
    for col in features:
        good = df.loc[df[target] == 0, col]
        bad = df.loc[df[target] == 1, col]
        ks, _ = ks_2samp(good, bad)
        rows.append({"feature": col, "ks": ks})
    return pd.DataFrame(rows).sort_values("ks", ascending=False)


def target_correlations(df, features, target, method="spearman"):
    return (
        df[features + [target]]
        .corr(method=method)[target]
        .drop(target)
        .sort_values(key=lambda x: x.abs(), ascending=False)
    )