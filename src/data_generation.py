import numpy as np
import pandas as pd


def generate_synthetic_credit_data(
    n_samples: int = 10000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic credit default dataset.

    Target:
        default (1 = default, 0 = non-default)
    """

    rng = np.random.default_rng(random_state)

    age = rng.integers(21, 70, size=n_samples)
    income = rng.normal(80_000, 30_000, size=n_samples).clip(15_000, None)
    employment_length = rng.integers(0, 40, size=n_samples)

    dti = rng.beta(2, 5, size=n_samples)  # debt-to-income
    prev_loans = rng.integers(0, 10, size=n_samples)
    delinquencies = rng.poisson(0.5, size=n_samples)
    credit_history = rng.integers(1, 30, size=n_samples)

    # Linear PD model (latent)
    score = (
        -3.0
        + 0.04 * dti * 10
        + 0.6 * delinquencies
        - 0.00001 * income
        - 0.03 * employment_length
        - 0.02 * credit_history
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
