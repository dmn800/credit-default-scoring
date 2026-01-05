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
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent  # notebooks -> project root
sys.path.insert(0, str(PROJECT_ROOT))

# %% [markdown]
# # Imports, Config, Load, Split

# %%
# 01_eda.ipynb
# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.model_selection import train_test_split
from IPython.display import display

from src.eda.audit import (
    duplicate_rate,
    low_variance_report,
    business_rules_violations,
    target_sanity
)
from src.eda.signal import (
    ks_report,
    target_correlations
)
from src.eda.reports import (
    feature_quality_report,
    iv_train_test_report
)
from src.eda.plots import (
    plot_kde_by_target,
    matrix_correlation,
    plot_badrate_binning,
    plot_badrate_stability_grid
)

TARGET = "default"

NUM_FEATURES = [
    "age",
    "income",
    "employment_length",
    "dti",
    "previous_loans",
    "delinquencies",
    "credit_history"
]

df = pd.read_csv("../data/raw/credit_data.csv")

df_train, df_test = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=df[TARGET]
)

# %% [markdown]
# # Data Quality Audit

# %%
print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)

print("\nTarget rate:")
display(
    pd.DataFrame({
        "train": df_train[TARGET].mean(),
        "test": df_test[TARGET].mean()
    }, index=["bad_rate"])
)

display(duplicate_rate(df_train))
display(low_variance_report(df_train, NUM_FEATURES))
display(business_rules_violations(df_train, NUM_FEATURES))
display(target_sanity(df_train, TARGET))


# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ---
# ### Выводы по data quality
#
# Данные прошли базовую проверку качества и логической согласованности.
#
# **Target:**
# - Bad rate в train-сэмпле составляет ~6.7%, что соответствует степени риска в сегменте потребительских кредитов, имеющих массовый и необеспеченный характер.
#
# **Логические несоответствия:**
# - Выявлена существенная доля наблюдений, в которых:
#   - стаж работы превышает возраст заемщика;
#   - длина кредитной истории превышает возраст заемщика.
# - Доля таких наблюдений составляет 22–34% в зависимости от правила, что классифицируется как критическое логическое нарушение.
#
# **Принятое решение:**
# - Для синтетического датасета данные нарушения рассматриваются как артефакт генерации данных;
# - Коррекция выполняется на уровне data generation / preprocessing и не затрагивает этап EDA;
# - В дальнейшем используется ограничение максимальной длины кредитной истории, согласованное с возрастом заемщика.
#
# Других критических проблем (дубликаты, константные признаки, отрицательные значения) не выявлено.
#

# %% [markdown]
# # Signal EDA

# %%
plot_kde_by_target(df_train, NUM_FEATURES, TARGET, n_cols=3)
display(ks_report(df_train, NUM_FEATURES, TARGET))
matrix_correlation(df_train, NUM_FEATURES, method="spearman")
target_correlations(df_train, NUM_FEATURES, TARGET, method="spearman")

# %% [markdown]
# ---
# ### Выводы по Signal EDA
#
# Для большинства числовых признаков наблюдается различие распределений между good и bad сегментами, однако степень overlap варьируется.
#
# Наиболее выраженный separation (по KDE и KS-statistic) демонстрируют:
# - `delinquencies`
# - `dti`
#
# Признаки с минимальным различием распределений (KS < 0.05) рассматриваются как слабые кандидаты и подлежат дополнительной проверке на этапе биннинга и IV.
#
# Correlation analysis использовался исключительно как вспомогательный инструмент и не рассматривался как критерий отбора признаков. Можно отметить высокую корреляцию переменной `age` с `employment_length` и `credit_history`, которая потенциально может быть исключена из моделирования.
#

# %% [markdown]
# # Feature Quality Summary

# %%
config_quality = {
    "df": df_train,
    "features": NUM_FEATURES,
    "target": TARGET,
    "n_bins": 10,
    "min_bin_share": 0.01
}

display(feature_quality_report(**config_quality))
plot_badrate_binning(**config_quality)


# %% [markdown]
# ---
# ### Выводы по Feature Quality Summary:
# * признаки с IV < 0.02 (кандидаты на удаление);
# * число бинов трансформировано с учетом требования монотонности bad rate;
# * переменная `previous_loans` потенциально может быть исключена из моделирования.

# %% [markdown]
# # Stability check (train vs test)

# %%
config_stability = {
    "df_train": df_train,
    "df_test": df_test,
    "features": NUM_FEATURES,
    "target": TARGET,
    "n_bins": 10,
    "min_bin_share": 0.01
}

display(iv_train_test_report(**config_stability))
plot_badrate_stability_grid(**config_stability)


# %% [markdown]
# ---
# ### Итоговые выводы по EDA
#
# EDA выполнен исключительно на train-сэмпле с последующей проверкой устойчивости на отложенной выборке.
#
# Для всех признаков выполнен корректный скоринговый биннинг с контролем монотонности bad rate.
#
# Stability-check показал, что большинство признаков сохраняют форму риска и сопоставимые значения IV на test.
#
# На основании IV-stability и визуального анализа bad rate сформирован финальный список признаков для этапа моделирования:
# * `delinquencies`;
# * `dti`;
# * `employment_length`;
# * `credit_history`;
# * `income`.
#

# %%
