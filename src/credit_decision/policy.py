"""Explainable fixed-rate and risk-aware pricing policies."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import RATE_GRID, synthetic_acceptance_probability
from .model import DefaultRiskModel

POLICY_VERSION = "synthetic-policy-v1"


@dataclass(frozen=True)
class PolicyConfig:
    """Explicit, configurable assumptions for the toy decision policies."""

    risk_limit: float = 0.24
    human_review_band: float = 0.025
    maximum_automatic_dti: float = 0.58
    funding_rate: float = 0.06
    loss_given_default: float = 0.65
    rate_grid: tuple[float, ...] = RATE_GRID
    fixed_rate: float = 0.16

    def __post_init__(self) -> None:
        if not 0 < self.risk_limit < 1:
            raise ValueError("risk_limit must be between 0 and 1.")
        if not 0 <= self.human_review_band < self.risk_limit:
            raise ValueError("human_review_band must be non-negative and below risk_limit.")
        if not 0 < self.maximum_automatic_dti <= 1:
            raise ValueError("maximum_automatic_dti must be in (0, 1].")
        if not 0 <= self.funding_rate < 1:
            raise ValueError("funding_rate must be in [0, 1).")
        if not 0 <= self.loss_given_default <= 1:
            raise ValueError("loss_given_default must be in [0, 1].")
        if not self.rate_grid or any(not 0 < rate < 1 for rate in self.rate_grid):
            raise ValueError("rate_grid must contain rates strictly between 0 and 1.")
        if self.fixed_rate not in self.rate_grid:
            raise ValueError("fixed_rate must be present in rate_grid.")


def _expected_profit(
    default_probability: np.ndarray,
    acceptance_probability: np.ndarray,
    principal_k: np.ndarray,
    offered_rate: np.ndarray,
    config: PolicyConfig,
) -> np.ndarray:
    expected_loan_contribution = principal_k * (
        (1.0 - default_probability) * offered_rate
        - default_probability * config.loss_given_default
        - config.funding_rate
    )
    return acceptance_probability * expected_loan_contribution


def _base_decisions(
    applications: pd.DataFrame,
    config: PolicyConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare decisions and apply shared synthetic affordability safeguards."""
    if "application_id" not in applications:
        raise ValueError("applications must include application_id.")
    required = {
        "annual_income_k",
        "debt_to_income",
        "preferred_rate",
        "requested_amount_k",
    }
    missing = required - set(applications.columns)
    if missing:
        raise ValueError(f"Applications are missing columns: {sorted(missing)}")
    if applications.empty:
        raise ValueError("applications must not be empty.")

    dti = applications["debt_to_income"].to_numpy(dtype=float)
    principal = applications["requested_amount_k"].to_numpy(dtype=float)
    preferred_rate = applications["preferred_rate"].to_numpy(dtype=float)
    if (
        not np.isfinite(dti).all()
        or not np.isfinite(principal).all()
        or not np.isfinite(preferred_rate).all()
    ):
        raise ValueError(
            "debt_to_income, preferred_rate, and requested_amount_k must be finite."
        )
    if ((dti < 0) | (dti > 1)).any() or (principal <= 0).any():
        raise ValueError("debt_to_income must be in [0, 1] and amount must be positive.")
    if ((preferred_rate < 0) | (preferred_rate > 1)).any():
        raise ValueError("preferred_rate must be between 0 and 1.")

    decisions = np.full(len(applications), "decline", dtype=object)
    reasons = np.full(len(applications), "no_profitable_offer_below_risk_limit", dtype=object)
    offered_rates = np.full(len(applications), np.nan)
    manual_review = dti > config.maximum_automatic_dti
    decisions[manual_review] = "human_review"
    reasons[manual_review] = "debt_to_income_requires_human_review"
    return decisions, reasons, offered_rates


def decide_fixed_rate(
    applications: pd.DataFrame,
    model: DefaultRiskModel,
    config: PolicyConfig,
) -> pd.DataFrame:
    """Apply one fixed rate and route borderline risk for human review."""
    decisions, reasons, offered_rates = _base_decisions(applications, config)
    predicted_risks = np.full(len(applications), np.nan)
    expected_profits = np.full(len(applications), np.nan)
    eligible = decisions != "human_review"
    if not eligible.any():
        return _decision_frame(
            applications,
            decisions,
            reasons,
            offered_rates,
            predicted_risks,
            expected_profits,
        )

    positions = np.flatnonzero(eligible)
    candidates = applications.iloc[positions]
    rate = config.fixed_rate
    risk = model.predict_default_probability(candidates, rate)
    profit = _expected_profit(
        risk,
        synthetic_acceptance_probability(
            candidates["preferred_rate"].to_numpy(dtype=float),
            rate,
        ),
        candidates["requested_amount_k"].to_numpy(dtype=float),
        np.full(len(candidates), rate),
        config,
    )
    predicted_risks[positions] = risk
    expected_profits[positions] = profit

    borderline = np.abs(risk - config.risk_limit) <= config.human_review_band
    for local_position, original_position in enumerate(positions):
        if borderline[local_position]:
            decisions[original_position] = "human_review"
            reasons[original_position] = "risk_near_automatic_decision_limit"
        elif risk[local_position] <= config.risk_limit and profit[local_position] > 0:
            requested_rate = float(
                candidates.iloc[local_position]["preferred_rate"]
            )
            decisions[original_position] = (
                "approve"
                if abs(rate - requested_rate) < 1e-9
                else "counteroffer"
            )
            reasons[original_position] = "fixed_rate_within_risk_and_profit_limits"
            offered_rates[original_position] = rate
        else:
            reasons[original_position] = "fixed_rate_exceeds_risk_or_profit_limit"
    return _decision_frame(
        applications,
        decisions,
        reasons,
        offered_rates,
        predicted_risks,
        expected_profits,
    )


def decide_risk_aware_pricing(
    applications: pd.DataFrame,
    model: DefaultRiskModel,
    config: PolicyConfig,
) -> pd.DataFrame:
    """Choose the most profitable feasible rate, preserving human review."""
    decisions, reasons, offered_rates = _base_decisions(applications, config)
    predicted_risks = np.full(len(applications), np.nan)
    expected_profits = np.full(len(applications), np.nan)
    eligible_positions = np.flatnonzero(decisions != "human_review")
    if not len(eligible_positions):
        return _decision_frame(
            applications,
            decisions,
            reasons,
            offered_rates,
            predicted_risks,
            expected_profits,
        )

    candidates = applications.iloc[eligible_positions]
    risk_by_rate = np.column_stack(
        [
            model.predict_default_probability(candidates, rate)
            for rate in config.rate_grid
        ]
    )
    rates = np.asarray(config.rate_grid, dtype=float)
    acceptance_by_rate = synthetic_acceptance_probability(
        candidates["preferred_rate"].to_numpy(dtype=float)[:, None],
        rates[None, :],
    )
    principal = candidates["requested_amount_k"].to_numpy(dtype=float)[:, None]
    profit_by_rate = _expected_profit(
        risk_by_rate,
        acceptance_by_rate,
        principal,
        rates[None, :],
        config,
    )
    feasible = (risk_by_rate <= config.risk_limit) & (profit_by_rate > 0)
    feasible_profit = np.where(feasible, profit_by_rate, -np.inf)
    best_rate_indexes = np.argmax(feasible_profit, axis=1)
    has_feasible_offer = feasible.any(axis=1)
    fixed_index = config.rate_grid.index(config.fixed_rate)

    for local_position, position in enumerate(eligible_positions):
        if not has_feasible_offer[local_position]:
            predicted_risks[position] = risk_by_rate[local_position, fixed_index]
            expected_profits[position] = profit_by_rate[local_position, fixed_index]
            reasons[position] = "no_profitable_offer_below_risk_limit"
            continue

        best_index = best_rate_indexes[local_position]
        expected_profit = profit_by_rate[local_position, best_index]
        chosen_rate = rates[best_index]
        predicted_default = risk_by_rate[local_position, best_index]
        if abs(predicted_default - config.risk_limit) <= config.human_review_band:
            decisions[position] = "human_review"
            reasons[position] = "risk_near_automatic_decision_limit"
            predicted_risks[position] = predicted_default
            expected_profits[position] = expected_profit
        else:
            requested_rate = float(applications.iloc[position]["preferred_rate"])
            decisions[position] = (
                "approve"
                if abs(chosen_rate - requested_rate) < 1e-9
                else "counteroffer"
            )
            reasons[position] = (
                "max_expected_profit_subject_to_synthetic_risk_limit"
            )
            offered_rates[position] = chosen_rate
            predicted_risks[position] = predicted_default
            expected_profits[position] = expected_profit

    return _decision_frame(
        applications,
        decisions,
        reasons,
        offered_rates,
        predicted_risks,
        expected_profits,
    )


def _decision_frame(
    applications: pd.DataFrame,
    decisions: np.ndarray,
    reasons: np.ndarray,
    offered_rates: np.ndarray,
    predicted_risks: np.ndarray,
    expected_profits: np.ndarray,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "application_id": applications["application_id"].to_numpy(),
            "decision": decisions,
            "reason_code": reasons,
            "offered_rate": offered_rates,
            "predicted_default_probability": predicted_risks,
            "expected_profit_k": expected_profits,
            "policy_version": POLICY_VERSION,
        }
    )
    return result
