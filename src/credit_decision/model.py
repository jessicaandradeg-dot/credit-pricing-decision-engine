"""Default-risk estimation trained only on synthetic training records."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, OFFERED_RATE_COLUMN, OUTCOME_COLUMN


@dataclass
class DefaultRiskModel:
    """Scaled logistic model for default conditional on a proposed interest rate."""

    estimator: object

    @classmethod
    def fit(cls, applications: pd.DataFrame) -> "DefaultRiskModel":
        missing = set(FEATURES + [OFFERED_RATE_COLUMN, OUTCOME_COLUMN]) - set(
            applications.columns
        )
        if missing:
            raise ValueError(f"Training data is missing columns: {sorted(missing)}")
        if applications[OUTCOME_COLUMN].nunique() != 2:
            raise ValueError("Training outcomes must contain both default classes.")

        features = applications[FEATURES + [OFFERED_RATE_COLUMN]].copy()
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=0),
        )
        estimator.fit(features, applications[OUTCOME_COLUMN].astype(int))
        return cls(estimator=estimator)

    def predict_default_probability(
        self,
        applications: pd.DataFrame,
        offered_rate: np.ndarray | float,
    ) -> np.ndarray:
        """Estimate default probability for each record at the given rate."""
        missing = set(FEATURES) - set(applications.columns)
        if missing:
            raise ValueError(f"Prediction data is missing columns: {sorted(missing)}")
        rates = np.broadcast_to(
            np.asarray(offered_rate, dtype=float),
            (len(applications),),
        )
        if not np.isfinite(rates).all() or ((rates < 0) | (rates > 1)).any():
            raise ValueError("offered_rate must be finite and between 0 and 1.")
        features = applications[FEATURES].copy()
        features[OFFERED_RATE_COLUMN] = rates
        return self.estimator.predict_proba(features)[:, 1]
