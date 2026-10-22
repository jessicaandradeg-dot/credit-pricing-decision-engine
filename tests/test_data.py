import pandas as pd
import pytest

from credit_decision.data import (
    make_synthetic_applications,
    synthetic_acceptance_probability,
    synthetic_default_probability,
)


def test_synthetic_data_is_reproducible_and_uses_fake_ids():
    first = make_synthetic_applications(n=300, random_state=7)
    second = make_synthetic_applications(n=300, random_state=7)

    pd.testing.assert_frame_equal(first, second)
    assert first["application_id"].str.startswith("SYN-CREDIT-").all()
    assert set(first["defaulted"].unique()).issubset({0, 1})


def test_synthetic_data_requires_train_test_sample_size():
    with pytest.raises(ValueError, match="at least 200"):
        make_synthetic_applications(n=199)


def test_default_and_acceptance_probabilities_are_bounded():
    applications = make_synthetic_applications(n=200)
    default_probability = synthetic_default_probability(
        applications,
        applications["historical_offered_rate"].to_numpy(),
    )
    acceptance_probability = synthetic_acceptance_probability(
        applications["preferred_rate"].to_numpy(),
        0.16,
    )

    assert ((default_probability >= 0) & (default_probability <= 1)).all()
    assert ((acceptance_probability >= 0) & (acceptance_probability <= 1)).all()
