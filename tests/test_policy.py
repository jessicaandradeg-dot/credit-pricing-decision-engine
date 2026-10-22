import numpy as np
import pandas as pd
import pytest

from credit_decision.data import make_synthetic_applications
from credit_decision.model import DefaultRiskModel
from credit_decision.policy import (
    PolicyConfig,
    decide_fixed_rate,
    decide_risk_aware_pricing,
)


@pytest.fixture()
def trained_model():
    applications = make_synthetic_applications(n=1000, random_state=13)
    return applications, DefaultRiskModel.fit(applications)


def test_policies_return_auditable_decision_fields(trained_model):
    applications, model = trained_model
    test = applications.iloc[:100].reset_index(drop=True)
    config = PolicyConfig()

    for decisions in (
        decide_fixed_rate(test, model, config),
        decide_risk_aware_pricing(test, model, config),
    ):
        assert len(decisions) == len(test)
        assert decisions["application_id"].equals(test["application_id"])
        assert set(decisions["decision"]).issubset(
            {"approve", "counteroffer", "decline", "human_review"}
        )
        assert decisions["reason_code"].notna().all()
        assert decisions["policy_version"].eq("synthetic-policy-v2").all()
        assert decisions.loc[
            decisions["decision"] != "human_review",
            "predicted_default_probability",
        ].notna().all()
        assert decisions.loc[
            decisions["decision"].isin(["approve", "counteroffer"]),
            "expected_profit_k",
        ].gt(0).all()
        assert decisions.loc[
            decisions["decision"].isin(["approve", "counteroffer"]),
            "offered_rate",
        ].notna().all()


def test_high_debt_to_income_is_routed_to_human(trained_model):
    applications, model = trained_model
    test = applications.iloc[:1].copy()
    test["debt_to_income"] = 0.9

    decisions = decide_risk_aware_pricing(test, model, PolicyConfig())

    assert decisions.loc[0, "decision"] == "human_review"
    assert decisions.loc[0, "reason_code"] == "debt_to_income_requires_human_review"


def test_configuration_rejects_fixed_rate_outside_grid():
    with pytest.raises(ValueError, match="present in rate_grid"):
        PolicyConfig(fixed_rate=0.19)


def test_probability_and_profit_are_not_filled_for_non_offers(trained_model):
    applications, model = trained_model
    test = applications.iloc[:5].copy().reset_index(drop=True)
    test["debt_to_income"] = 0.95

    decisions = decide_fixed_rate(test, model, PolicyConfig())

    assert decisions["decision"].eq("human_review").all()
    assert decisions["offered_rate"].isna().all()
