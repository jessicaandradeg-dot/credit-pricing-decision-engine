"""Default-risk estimation trained only on synthetic training records."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, OFFERED_RATE_COLUMN, OUTCOME_COLUMN


def _training_data(applications: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = set(FEATURES + [OFFERED_RATE_COLUMN, OUTCOME_COLUMN]) - set(
        applications.columns
    )
    if missing:
        raise ValueError(f"Training data is missing columns: {sorted(missing)}")
    if applications[OUTCOME_COLUMN].nunique() != 2:
        raise ValueError("Training outcomes must contain both default classes.")
    features = applications[FEATURES + [OFFERED_RATE_COLUMN]].copy()
    target = applications[OUTCOME_COLUMN].astype(int)
    return features, target


def _prediction_features(
    applications: pd.DataFrame,
    offered_rate: np.ndarray | float,
) -> pd.DataFrame:
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
    return features


@dataclass
class DefaultRiskModel:
    """Scaled logistic model for default conditional on a proposed interest rate."""

    estimator: object

    @classmethod
    def fit(cls, applications: pd.DataFrame) -> "DefaultRiskModel":
        features, target = _training_data(applications)
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=0),
        )
        estimator.fit(features, target)
        return cls(estimator=estimator)

    def predict_default_probability(
        self,
        applications: pd.DataFrame,
        offered_rate: np.ndarray | float,
    ) -> np.ndarray:
        """Estimate default probability for each record at the given rate."""
        features = _prediction_features(applications, offered_rate)
        return self.estimator.predict_proba(features)[:, 1]


@dataclass
class NeuralDefaultRiskModel:
    """Small scaled multilayer perceptron for a synthetic model comparison."""

    estimator: object

    @classmethod
    def fit(cls, applications: pd.DataFrame) -> "NeuralDefaultRiskModel":
        features, target = _training_data(applications)
        estimator = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                solver="adam",
                alpha=1e-3,
                max_iter=400,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=0,
            ),
        )
        estimator.fit(features, target)
        return cls(estimator=estimator)

    def predict_default_probability(
        self,
        applications: pd.DataFrame,
        offered_rate: np.ndarray | float,
    ) -> np.ndarray:
        """Estimate default probability for each record at the given rate."""
        features = _prediction_features(applications, offered_rate)
        return self.estimator.predict_proba(features)[:, 1]
