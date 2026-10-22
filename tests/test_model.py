import numpy as np
import pytest

from credit_decision.data import make_synthetic_applications
from credit_decision.model import DefaultRiskModel


def test_risk_model_returns_reproducible_probabilities():
    applications = make_synthetic_applications(n=500, random_state=8)
    model = DefaultRiskModel.fit(applications)

    risk = model.predict_default_probability(applications.iloc[:20], 0.16)

    assert len(risk) == 20
    assert np.isfinite(risk).all()
    assert ((risk >= 0) & (risk <= 1)).all()


def test_model_rejects_invalid_offer_rate():
    applications = make_synthetic_applications(n=300)
    model = DefaultRiskModel.fit(applications)

    with pytest.raises(ValueError, match="between 0 and 1"):
        model.predict_default_probability(applications.iloc[:2], 1.5)


def test_model_requires_both_outcome_classes():
    applications = make_synthetic_applications(n=300)
    applications["defaulted"] = 0

    with pytest.raises(ValueError, match="both default classes"):
        DefaultRiskModel.fit(applications)
