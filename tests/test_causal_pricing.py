import numpy as np
import pandas as pd
import pytest

from credit_decision.causal_pricing import run_randomized_pricing_experiment
from credit_decision.data import make_synthetic_applications


def test_randomized_pricing_experiment_is_reproducible_and_balanced():
    applications = make_synthetic_applications(n=500, random_state=21)

    first_report, first_detail = run_randomized_pricing_experiment(
        applications,
        random_state=10,
        bootstrap_samples=300,
    )
    second_report, second_detail = run_randomized_pricing_experiment(
        applications,
        random_state=10,
        bootstrap_samples=300,
    )

    assert first_report == second_report
    pd.testing.assert_frame_equal(first_detail, second_detail)
    assert first_report["arms"]["control"]["applications"] == 250
    assert first_report["arms"]["treatment"]["applications"] == 250
    assert set(first_report["intent_to_treat_effects"]) == {
        "acceptance_rate",
        "contribution_k_per_offer",
    }


def test_randomized_pricing_effects_include_bootstrap_and_simulator_benchmark():
    applications = make_synthetic_applications(n=600, random_state=6)
    report, detail = run_randomized_pricing_experiment(
        applications,
        random_state=15,
        bootstrap_samples=300,
    )

    for effect in report["intent_to_treat_effects"].values():
        assert np.isfinite(list(effect.values())).all()
        assert effect["bootstrap_95pct_ci_lower"] <= effect[
            "bootstrap_95pct_ci_upper"
        ]
        assert "simulator_finite_population_ate" in effect
    assert len(detail) == len(applications)
    assert "default_rate_among_accepted_descriptive" in report["arms"]["control"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"control_rate": 0.12, "treatment_rate": 0.12}, "distinct values"),
        ({"control_rate": 1.1, "treatment_rate": 0.2}, "between 0 and 1"),
        ({"bootstrap_samples": 100}, "at least 200"),
    ],
)
def test_randomized_pricing_experiment_rejects_invalid_design(kwargs, message):
    applications = make_synthetic_applications(n=200)

    with pytest.raises(ValueError, match=message):
        run_randomized_pricing_experiment(applications, **kwargs)
