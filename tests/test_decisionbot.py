import sqlite3

import pandas as pd

from credit_decision.decisionbot import (
    analyze_exception_layer,
    route_human_review,
    save_decision_audit,
)
from credit_decision.data import make_synthetic_applications
from credit_decision.model import DefaultRiskModel
from credit_decision.policy import PolicyConfig, decide_risk_aware_pricing


def test_review_routing_optimizes_net_value_within_capacity():
    decisions = pd.DataFrame(
        {
            "application_id": ["a", "b", "c", "d"],
            "decision": ["human_review", "human_review", "human_review", "approve"],
            "reason_code": ["risk"] * 4,
            "expected_profit_k": [1.0, 1.5, None, 9.0],
        }
    )

    routed, summary = route_human_review(
        decisions,
        capacity_fraction=0.5,
        review_cost_k=0.25,
    )

    assert routed["decision"].tolist() == [
        "human_review",
        "human_review",
        "decline",
        "approve",
    ]
    assert summary["review_capacity"] == 2
    assert summary["reviews_selected"] == 2
    assert summary["total_review_cost_k"] == 0.5
    assert routed.loc[:1, "expected_net_contribution_k"].tolist() == [0.75, 1.25]


def test_exception_tree_is_shallow_and_reports_held_out_risk_bands():
    applications = make_synthetic_applications(n=800, random_state=31)
    model = DefaultRiskModel.fit(applications.iloc[:560].reset_index(drop=True))
    evaluation = applications.iloc[560:].reset_index(drop=True)
    decisions = decide_risk_aware_pricing(evaluation, model, PolicyConfig())

    tree, risk_bands, summary = analyze_exception_layer(
        evaluation,
        decisions,
        review_cost_k=0.1,
        random_state=31,
    )

    assert tree.get_depth() <= 3
    assert not risk_bands.empty
    assert {
        "actual_exception_rate",
        "predicted_exception_rate",
        "decisions_without_next_layer",
        "estimated_contribution_loss_k",
    } <= set(risk_bands.columns)
    assert 0 <= summary["pass_without_next_layer_rate"] <= 1
    assert summary["contribution_loss_is_counterfactual_proxy"] == 1


def test_sqlite_decision_audit_can_be_queried(tmp_path):
    path = tmp_path / "decisionbot.sqlite"
    decisions = pd.DataFrame(
        {
            "application_id": ["a", "b"],
            "decision": ["approve", "counteroffer"],
            "reason_code": ["baseline", "risk"],
            "policy_version": ["synthetic-policy-v1"] * 2,
        }
    )

    save_decision_audit(decisions, path)

    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT decision, COUNT(*) FROM decision_audit GROUP BY decision"
        ).fetchall()
    assert rows == [("approve", 1), ("counteroffer", 1)]
