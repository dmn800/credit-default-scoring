import numpy as np
import pandas as pd


def generate_synthetic_credit_data(
    n_samples: int = 10000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic credit default dataset^

    Characteristics:
    - latent risk segmentation (low / medium / high)
    - bad rate controlled mainly via intercepts
    - monotonic, interpretable drivers

    Target:
        default (1 = default, 0 = non-default)
    """

    rng = np.random.default_rng(random_state)

    # ----------------------------
    # 1. Latent risk segmentation
    # ----------------------------
    risk_segments = rng.choice(
        ['low', 'medium', 'high'],
        size=n_samples,
        p=[0.65, 0.25, 0.10]
    )

    segment_intercepts = {
        'low': -3.2,
        'medium': -2.0,
        'high': -1.0
    }

    intercept = np.array([segment_intercepts[s] for s in risk_segments])
    
    # ----------------------------
    # 2. Feature generation
    # ----------------------------
    age = rng.integers(21, 70, size=n_samples)
    income = rng.normal(80_000, 30_000, size=n_samples).clip(15_000, None)

    max_employment = np.maximum(age - 18, 0)
    employment_length = rng.integers(0, max_employment + 1)

    dti = rng.beta(2, 5, size=n_samples)  # debt-to-income
    prev_loans = rng.integers(0, 10, size=n_samples)
    delinquencies = rng.poisson(0.5, size=n_samples)

    max_credit_history = np.maximum(age - 18, 1)
    credit_history = rng.integers(1, max_credit_history + 1)

    # ----------------------------
    # Logistic PD model
    # ----------------------------
    score = (
        intercept
        + 1.1 * delinquencies
        + 0.9 * (dti > 0.4).astype(int)
        - 0.02 * credit_history
        - 0.00001 * income
        - 0.03 * employment_length
    )

    pd_default = 1 / (1 + np.exp(-score))
    default = rng.binomial(1, pd_default)

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "employment_length": employment_length,
        "dti": dti,
        "previous_loans": prev_loans,
        "delinquencies": delinquencies,
        "credit_history": credit_history,
        "default": default,
    })

    return df


if __name__ == "__main__":
    df = generate_synthetic_credit_data()
    df.to_csv("data/raw/credit_data.csv", index=False)
