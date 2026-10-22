"""Capacity-aware human review and shallow exception-layer distillation."""

from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

_TREE_FEATURES = [
    "predicted_default_probability",
    "debt_to_income",
    "preferred_rate",
    "expected_profit_k",
]
_RISK_BINS = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 1.000001]


def route_human_review(
    decisions: pd.DataFrame,
    capacity_fraction: float,
    review_cost_k: float,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Choose reviews by maximizing modeled contribution minus review cost.

    The objective is the sum of expected contribution less the explicit review
    cost, subject to a fixed capacity. Cases without a finite projected value
    are declined rather than routed on an unquantified assumption.
    """
    if not 0 <= capacity_fraction <= 1:
        raise ValueError("capacity_fraction must be in [0, 1].")
    if not np.isfinite(review_cost_k) or review_cost_k < 0:
        raise ValueError("review_cost_k must be finite and non-negative.")
    required = {"decision", "expected_profit_k", "reason_code"}
    if required - set(decisions.columns):
        raise ValueError(f"Decisions are missing columns: {sorted(required - set(decisions.columns))}")

    routed = decisions.copy()
    routed["human_review_cost_k"] = 0.0
    modeled_contribution = pd.to_numeric(
        routed["expected_profit_k"], errors="coerce"
    ).fillna(0.0)
    accepted = routed["decision"].isin(["approve", "counteroffer"])
    routed["expected_net_contribution_k"] = np.where(
        accepted,
        modeled_contribution,
        0.0,
    )
    review_positions = np.flatnonzero(
        routed["decision"].to_numpy() == "human_review"
    )
    capacity = int(np.floor(len(routed) * capacity_fraction))
    projected = pd.to_numeric(
        routed["expected_profit_k"], errors="coerce"
    ).to_numpy(dtype=float)
    net_review_value = projected[review_positions] - review_cost_k
    eligible_mask = np.isfinite(net_review_value) & (net_review_value > 0)
    eligible_positions = review_positions[eligible_mask]
    eligible_values = net_review_value[eligible_mask]
    order = np.argsort(-eligible_values, kind="stable")
    selected_positions = eligible_positions[order[:capacity]]
    selected = np.zeros(len(routed), dtype=bool)
    selected[selected_positions] = True

    unselected_positions = review_positions[~selected[review_positions]]
    routed.loc[unselected_positions, "decision"] = "decline"
    routed.loc[unselected_positions, "reason_code"] = "review_capacity_or_value_limit"
    routed.loc[selected_positions, "human_review_cost_k"] = review_cost_k
    routed.loc[selected_positions, "expected_net_contribution_k"] = (
        projected[selected_positions] - review_cost_k
    )

    metrics: dict[str, float | int] = {
        "review_capacity": capacity,
        "reviews_selected": int(selected.sum()),
        "review_capacity_fraction": capacity_fraction,
        "review_cost_per_case_k": review_cost_k,
        "total_review_cost_k": float(selected.sum() * review_cost_k),
        "net_expected_contribution_k": float(
            routed["expected_net_contribution_k"].sum()
        ),
    }
    return routed, metrics


def analyze_exception_layer(
    applications: pd.DataFrame,
    decisions: pd.DataFrame,
    review_cost_k: float,
    random_state: int = 42,
) -> tuple[DecisionTreeClassifier, pd.DataFrame, dict[str, float | int]]:
    """Fit a shallow held-out surrogate for counteroffer/review exceptions.

    The reported contribution loss is a modeled upper-bound proxy: when the
    tree bypasses a true exception, the calculation assumes its expected
    contribution is lost. It is not realized or causal loss.
    """
    if not 0 <= review_cost_k or not np.isfinite(review_cost_k):
        raise ValueError("review_cost_k must be finite and non-negative.")
    if not {"application_id", "debt_to_income", "preferred_rate"} <= set(applications):
        raise ValueError("Applications are missing required identifiers or features.")
    if not {
        "application_id",
        "decision",
        "predicted_default_probability",
        "expected_profit_k",
    } <= set(decisions):
        raise ValueError("Decisions are missing fields required for exception analysis.")
    records = applications[
        ["application_id", "debt_to_income", "preferred_rate"]
    ].merge(
        decisions[
            [
                "application_id",
                "decision",
                "predicted_default_probability",
                "expected_profit_k",
            ]
        ],
        on="application_id",
        how="inner",
        validate="one_to_one",
    )
    if len(records) != len(applications) or records.empty:
        raise ValueError("Applications and decisions must align one-to-one.")

    target = records["decision"].isin(["counteroffer", "human_review"]).astype(int)
    if target.nunique() != 2 or target.value_counts().min() < 2:
        raise ValueError("At least two examples of each exception class are required.")
    features = records[_TREE_FEATURES].copy()
    features["predicted_default_probability"] = features[
        "predicted_default_probability"
    ].fillna(1.0)
    features["expected_profit_k"] = features["expected_profit_k"].fillna(0.0)
    train_positions, test_positions = train_test_split(
        np.arange(len(records)),
        test_size=0.30,
        random_state=random_state,
        stratify=target,
    )
    tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=max(5, int(np.ceil(0.02 * len(train_positions)))),
        class_weight="balanced",
        random_state=random_state,
    )
    tree.fit(features.iloc[train_positions], target.iloc[train_positions])
    predicted_exception = tree.predict(features.iloc[test_positions]).astype(bool)
    held_out = records.iloc[test_positions].copy()
    held_out["actual_exception"] = target.iloc[test_positions].to_numpy(dtype=bool)
    held_out["predicted_exception"] = predicted_exception
    held_out["risk_for_band"] = held_out[
        "predicted_default_probability"
    ].fillna(1.0)
    held_out["risk_band"] = pd.cut(
        held_out["risk_for_band"],
        bins=_RISK_BINS,
        include_lowest=True,
        right=False,
    ).astype(str)
    contribution = pd.to_numeric(
        held_out["expected_profit_k"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    review_mask = held_out["decision"].eq("human_review")
    contribution.loc[review_mask] = (
        contribution.loc[review_mask] - review_cost_k
    ).clip(lower=0.0)
    bypassed = held_out["actual_exception"] & ~held_out["predicted_exception"]
    held_out["estimated_contribution_loss_k"] = np.where(
        bypassed, contribution, 0.0
    )

    by_band = (
        held_out.groupby("risk_band", observed=True)
        .agg(
            applications=("decision", "size"),
            actual_exception_rate=("actual_exception", "mean"),
            predicted_exception_rate=("predicted_exception", "mean"),
            decisions_without_next_layer=("predicted_exception", lambda values: int((~values).sum())),
            estimated_contribution_loss_k=("estimated_contribution_loss_k", "sum"),
        )
        .reset_index()
    )
    summary: dict[str, float | int] = {
        "training_records": int(len(train_positions)),
        "held_out_records": int(len(test_positions)),
        "tree_depth": int(tree.get_depth()),
        "pass_without_next_layer_rate": float((~predicted_exception).mean()),
        "exception_recall": float(
            (
                predicted_exception
                & held_out["actual_exception"].to_numpy(dtype=bool)
            ).sum()
            / max(1, held_out["actual_exception"].sum())
        ),
        "estimated_contribution_loss_k": float(
            held_out["estimated_contribution_loss_k"].sum()
        ),
        "contribution_loss_is_counterfactual_proxy": 1,
    }
    return tree, by_band, summary


def save_decision_audit(decisions: pd.DataFrame, path: Path) -> None:
    """Persist synthetic decision records to a local SQLite audit table."""
    if decisions.empty:
        raise ValueError("decisions must not be empty.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        decisions.to_sql("decision_audit", connection, if_exists="replace", index=False)
