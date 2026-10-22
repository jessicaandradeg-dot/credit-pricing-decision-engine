import json

import pandas as pd
import pytest

from credit_decision.experiment import (
    evaluate_policy,
    report_as_json,
    run_experiment,
)
from credit_decision.policy import PolicyConfig


def test_experiment_is_reproducible():
    first_report, first_details = run_experiment(n=800, random_state=23)
    second_report, second_details = run_experiment(n=800, random_state=23)

    assert first_report == second_report
    pd.testing.assert_frame_equal(first_details, second_details)


def test_experiment_produces_expected_summary_shape():
    report, details = run_experiment(n=600, random_state=3)

    assert report["test_records"] == len(details)
    assert set(report["policy_evaluation"]["results"][0]) >= {
        "expected_profit_k",
        "expected_defaults",
        "human_review",
        "counteroffers",
    }
    assert len(report["policy_evaluation"]["results"]) == 2
    assert report["policy_evaluation"]["results"][0]["counteroffers"] > 0
    comparison = report["model_comparison_at_randomized_historical_rates"]
    assert set(comparison) == {
        "logistic_regression_baseline",
        "mlp_neural_network",
        "policy_model",
    }
    assert comparison["mlp_neural_network"]["test_records"] == len(details)
    assert comparison["policy_model"] == "logistic_regression_baseline"
    recovery_scenarios = report["recovery_assumption_sensitivity"]["scenarios"]
    assert [scenario["assumed_recovery_rate"] for scenario in recovery_scenarios] == [
        0.20,
        0.40,
        0.60,
    ]
    assert all(len(scenario["results"]) == 2 for scenario in recovery_scenarios)
    causal_experiment = report["randomized_pricing_experiment"]
    assert causal_experiment["arms"]["control"]["applications"] + (
        causal_experiment["arms"]["treatment"]["applications"]
    ) == len(details)
    assert "bootstrap_95pct_ci_lower" in causal_experiment[
        "intent_to_treat_effects"
    ]["acceptance_rate"]
    json.loads(report_as_json(report))


def test_evaluation_rejects_misaligned_decisions():
    applications = pd.DataFrame(
        {
            "application_id": ["SYN-CREDIT-1"],
            "requested_amount_k": [10.0],
            "preferred_rate": [0.16],
            "annual_income_k": [60.0],
            "debt_to_income": [0.3],
            "credit_utilization": [0.2],
            "payment_misses_12m": [0],
            "employment_years": [5.0],
            "historical_offered_rate": [0.16],
            "defaulted": [0],
        }
    )
    decisions = pd.DataFrame(
        {
            "application_id": ["SYN-CREDIT-2"],
            "decision": ["approve"],
            "offered_rate": [0.16],
        }
    )

    with pytest.raises(ValueError, match="align by application_id"):
        evaluate_policy(applications, decisions, PolicyConfig(), "test")
