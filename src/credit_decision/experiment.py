"""Train, compare, and evaluate synthetic credit decision policies."""

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from .data import (
    FEATURES,
    OFFERED_RATE_COLUMN,
    OUTCOME_COLUMN,
    make_synthetic_applications,
    synthetic_acceptance_probability,
    synthetic_default_probability,
)
from .model import DefaultRiskModel
from .policy import (
    PolicyConfig,
    decide_fixed_rate,
    decide_risk_aware_pricing,
)


def evaluate_policy(
    applications: pd.DataFrame,
    decisions: pd.DataFrame,
    config: PolicyConfig,
    policy_name: str,
) -> dict[str, float | int | str]:
    """Estimate outcomes using simulator probabilities at each proposed price."""
    aligned = applications.reset_index(drop=True)
    actions = decisions.reset_index(drop=True)
    if len(aligned) != len(actions):
        raise ValueError("applications and decisions must have equal lengths.")
    if not aligned["application_id"].equals(actions["application_id"]):
        raise ValueError("application rows must align by application_id.")

    accepted_offers = actions["decision"].isin(["approve", "counteroffer"])
    rates = actions["offered_rate"].to_numpy(dtype=float)
    if actions.loc[accepted_offers, "offered_rate"].isna().any():
        raise ValueError("Approved and counteroffer decisions must include an offer rate.")

    expected_defaults = 0.0
    expected_acceptances = 0.0
    expected_profit = 0.0
    expected_principal = 0.0
    if accepted_offers.any():
        accepted_records = aligned.loc[accepted_offers].reset_index(drop=True)
        offer_rates = rates[accepted_offers.to_numpy()]
        default_probability = synthetic_default_probability(
            accepted_records,
            offer_rates,
        )
        acceptance_probability = synthetic_acceptance_probability(
            accepted_records["preferred_rate"].to_numpy(dtype=float),
            offer_rates,
        )
        principal = accepted_records["requested_amount_k"].to_numpy(dtype=float)
        expected_acceptances = float(acceptance_probability.sum())
        expected_defaults = float(np.sum(acceptance_probability * default_probability))
        expected_principal = float(np.sum(acceptance_probability * principal))
        expected_profit = float(
            np.sum(
                acceptance_probability
                * principal
                * (
                    (1.0 - default_probability) * offer_rates
                    - default_probability * config.loss_given_default
                    - config.funding_rate
                )
            )
        )

    approved_count = int(accepted_offers.sum())
    total_applications = len(aligned)
    return {
        "policy": policy_name,
        "applications": total_applications,
        "approved_or_counteroffered": approved_count,
        "human_review": int((actions["decision"] == "human_review").sum()),
        "declined": int((actions["decision"] == "decline").sum()),
        "counteroffers": int((actions["decision"] == "counteroffer").sum()),
        "expected_acceptances": expected_acceptances,
        "expected_defaults": expected_defaults,
        "expected_default_rate_among_acceptances": (
            expected_defaults / expected_acceptances
            if expected_acceptances
            else 0.0
        ),
        "expected_profit_k": expected_profit,
        "expected_profit_per_application_k": expected_profit / total_applications,
        "expected_principal_k": expected_principal,
        "expected_loss_rate_on_accepted_principal": (
            expected_defaults
            * config.loss_given_default
            / expected_principal
            if expected_principal
            else 0.0
        ),
    }


def run_experiment(
    n: int = 6000,
    random_state: int = 42,
    config: PolicyConfig | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Run a deterministic train/test experiment with synthetic policies."""
    config = config or PolicyConfig()
    applications = make_synthetic_applications(
        n=n,
        random_state=random_state,
    )
    train, test = train_test_split(
        applications,
        test_size=0.30,
        random_state=random_state,
        stratify=applications[OUTCOME_COLUMN],
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    model = DefaultRiskModel.fit(train)

    decisions_by_policy = {
        "fixed_rate_baseline": decide_fixed_rate(test, model, config),
        "risk_aware_pricing": decide_risk_aware_pricing(test, model, config),
    }
    policy_metrics = [
        evaluate_policy(test, decisions, config, name)
        for name, decisions in decisions_by_policy.items()
    ]

    observed_rate = test[OFFERED_RATE_COLUMN].to_numpy(dtype=float)
    predicted_default = model.predict_default_probability(test, observed_rate)
    classification_metrics = {
        "roc_auc": float(roc_auc_score(test[OUTCOME_COLUMN], predicted_default)),
        "brier_score": float(
            brier_score_loss(test[OUTCOME_COLUMN], predicted_default)
        ),
        "test_records": len(test),
    }

    detail = test[
        [
            "application_id",
            *FEATURES,
            "preferred_rate",
            OFFERED_RATE_COLUMN,
            OUTCOME_COLUMN,
        ]
    ].copy()
    for policy_name, decisions in decisions_by_policy.items():
        renamed = decisions.drop(columns="application_id").add_prefix(
            f"{policy_name}_"
        )
        detail = pd.concat(
            [detail.reset_index(drop=True), renamed.reset_index(drop=True)],
            axis=1,
        )

    report: dict[str, object] = {
        "notice": (
            "Synthetic teaching simulation only. Not for real lending, pricing, "
            "credit eligibility, or customer decisions."
        ),
        "seed": random_state,
        "training_records": len(train),
        "test_records": len(test),
        "assumptions": asdict(config),
        "model_quality_at_randomized_historical_rates": classification_metrics,
        "policy_evaluation": {
            "method": (
                "Expected outcomes use the known synthetic simulator at each "
                "policy offer rate; these are not observed outcomes or real-world "
                "causal estimates."
            ),
            "results": policy_metrics,
        },
    }
    return report, detail


def report_as_json(report: dict[str, object]) -> str:
    """Serialize the report without relying on non-standard JSON values."""
    return json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
