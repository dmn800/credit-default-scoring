# Credit Default Scoring

End-to-end credit default scoring project: data generation, feature engineering, modeling, evaluation, and interpretation.

## Problem Statement
The goal of this project is to predict the probability of a borrower defaulting on a loan.  
This is framed as a **binary classification problem** with imbalanced classes, reflecting real-world credit risk scenarios.

## Project Structure
credit-default-scoring/
|
|-- data/                            # Raw and processed datasets
|   |-- raw/
|   |-- processed/
|-- notebooks/                       # Jupyter notebooks
|   |-- 01_eda.ipynb
|   |-- 02_feature_engineering.ipynb
|   |-- 03_modeling.ipynb
|-- src/                             # Python modules
|   |-- data_generation.py
|   |-- preprocessing.py
|   |-- features.py
|   |-- models.py
|   |-- metrics.py
|-- experiments/                     # Model results, baseline comparisons
|-- reports/                         # Conclusions and analysis
|-- README.md
|-- requirements.txt
>>>>>>> f467cda (Add README for ML credit scoring project)

## Data Generation
Synthetic dataset is used for demonstration purposes. Features include:

- Age
- Income
- Employment length
- Debt-to-income ratio
- Number of previous loans
- Number of delinquencies
- Credit history length
- Target: default (0/1)

## Modeling Approach

1. Baseline: Logistic Regression  
2. Advanced: Gradient Boosting / LightGBM (future)

## Evaluation Metrics

- ROC AUC
- Gini coefficient
- KS statistic
- Precision / Recall
- Confusion Matrix
- Lift / Decile analysis

> Accuracy is not the primary metric due to class imbalance.

## Interpretation

- Coefficients of logistic regression are analyzed
- Feature importance visualized for tree-based models
- Sanity checks to ensure relationships are reasonable (e.g., higher DTI → higher PD)

## Limitations

- Synthetic data may not fully capture real-world distributions  
- No external validation dataset yet  
- Advanced risk measures (e.g., PD curves, calibration) not implemented yet

## Next Steps

- Implement feature engineering and preprocessing  
- Train baseline and advanced models  
- Evaluate metrics and compare performance  
- Visualize model interpretation (SHAP, coefficients)  
- Prepare final report with conclusions