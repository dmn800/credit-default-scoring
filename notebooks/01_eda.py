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

# Настройки отображения
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (6, 4)

# Загрузка данных
df = pd.read_csv("../data/raw/credit_data.csv")

# -----------------------------
# 1. Обзор данных
# -----------------------------
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# -----------------------------
# 2. Распределения числовых признаков
# -----------------------------
numerical_features = ["age", "income", "employment_length", "dti",
                      "previous_loans", "delinquencies", "credit_history"]

n_cols = 3
n_rows = int(np.ceil(len(numerical_features) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(numerical_features):
    sns.histplot(df[col], kde=True, ax=axes[i])
    axes[i].set_title(col)

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# -----------------------------
# 3. Числовые признаки vs целевая переменная
# -----------------------------
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(numerical_features):
    sns.boxplot(x="default", y=col, data=df, ax=axes[i])
    axes[i].set_title(f"{col} vs default")

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# -----------------------------
# 4. Категориальные признаки (если есть)
# -----------------------------
# В данном случае у нас нет явных категориальных признаков,
# но если они появятся, можно сделать так:
categorical_features = []  # пример: ["gender", "region"]

if categorical_features:
    n_cols = 3
    n_rows = int(np.ceil(len(categorical_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(categorical_features):
        sns.countplot(x=col, hue="default", data=df, ax=axes[i])
        axes[i].set_title(col)

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# -----------------------------
# 5. Матрица корреляций
# -----------------------------
corr = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


# %%
