"""Synthetic, reproducible application and outcome generation."""

import numpy as np
import pandas as pd

FEATURES = [
    "annual_income_k",
    "debt_to_income",
    "credit_utilization",
    "payment_misses_12m",
    "employment_years",
    "requested_amount_k",
]
OFFERED_RATE_COLUMN = "historical_offered_rate"
OUTCOME_COLUMN = "defaulted"
RATE_GRID = (0.08, 0.12, 0.16, 0.20, 0.24, 0.28)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic transform."""
    values = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-values))


def synthetic_default_probability(
    records: pd.DataFrame,
    offered_rate: np.ndarray | float,
) -> np.ndarray:
    """Known simulator probability used to create and evaluate synthetic data."""
    rate = np.broadcast_to(np.asarray(offered_rate, dtype=float), len(records))
    log_odds = (
        -3.3
        + 3.0 * records["debt_to_income"].to_numpy(dtype=float)
        + 1.15 * records["credit_utilization"].to_numpy(dtype=float)
        + 0.43 * records["payment_misses_12m"].to_numpy(dtype=float)
        - 0.045 * records["employment_years"].to_numpy(dtype=float)
        + 0.018 * records["requested_amount_k"].to_numpy(dtype=float)
        / np.maximum(records["annual_income_k"].to_numpy(dtype=float), 1.0)
        + 0.85 * (rate - 0.16) / 0.04
    )
    return _sigmoid(log_odds)


def synthetic_acceptance_probability(
    preferred_rate: np.ndarray,
    offered_rate: np.ndarray | float,
) -> np.ndarray:
    """Illustrative response curve; not estimated from real customer behavior."""
    preferred, offered = np.broadcast_arrays(
        np.asarray(preferred_rate, dtype=float),
        np.asarray(offered_rate, dtype=float),
    )
    return _sigmoid(14.0 * (preferred - offered))


def make_synthetic_applications(
    n: int = 6000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate fake applicants and defaults observed at randomized offer rates."""
    if n < 200:
        raise ValueError("n must be at least 200 for train/test evaluation.")

    rng = np.random.default_rng(random_state)
    annual_income = np.clip(rng.lognormal(4.05, 0.48, size=n), 18.0, 180.0)
    dti = np.clip(rng.beta(2.0, 4.0, size=n) * 0.9, 0.01, 0.78)
    utilization = np.clip(rng.beta(2.0, 2.2, size=n), 0.01, 0.99)
    missed_payments = rng.poisson(0.65, size=n).clip(0, 6)
    employment_years = np.clip(rng.gamma(2.2, 2.5, size=n), 0.0, 30.0)
    requested_amount = np.clip(
        rng.lognormal(2.35, 0.65, size=n),
        1.0,
        np.minimum(40.0, annual_income * 0.8),
    )
    preferred_rate = rng.uniform(0.10, 0.28, size=n)
    historical_rate = rng.choice(RATE_GRID, size=n)

    applications = pd.DataFrame(
        {
            "application_id": [f"SYN-CREDIT-{index:06d}" for index in range(n)],
            "annual_income_k": annual_income,
            "debt_to_income": dti,
            "credit_utilization": utilization,
            "payment_misses_12m": missed_payments.astype(int),
            "employment_years": employment_years,
            "requested_amount_k": requested_amount,
            "preferred_rate": preferred_rate,
            OFFERED_RATE_COLUMN: historical_rate,
        }
    )
    probability = synthetic_default_probability(applications, historical_rate)
    applications[OUTCOME_COLUMN] = rng.binomial(1, probability, size=n)
    return applications
