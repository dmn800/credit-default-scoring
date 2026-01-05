import pandas as pd


def duplicate_rate(df):
    return df.duplicated().mean()


def low_variance_report(df, features, threshold=0.99):
    rows = []

    for feat in features:
        top_share = df[feat].value_counts(normalize=True).iloc[0]
        rows.append(
            {
                "feature": feat,
                "top_value_share": top_share,
                "low_variance": top_share >= threshold
            }
        )

    return pd.DataFrame(rows)


def business_rules_violations(df, features):
    rules = {
        "employment_gt_age": (df["employment_length"] > (df["age"] - 18)),
        "credit_history_gt_age": (df["credit_history"] > (df["age"] - 18)),
        "negative_income": (df["income"] <= 0),
        "dti_gt_1": (df["dti"] > 1),
        "negative_values": ((df[features] < 0).any(axis=1))
    }

    report = {
        rule: condition.mean()
        for rule, condition in rules.items()
    }

    return pd.Series(report, name="violation_rate").to_frame()


def target_sanity(df, target):
    data = {
        'Value': [df[target].unique(), df[target].mean()]
    }
    index = ['Target unique values', 'Target mean']
    return pd.DataFrame(data, index=index)