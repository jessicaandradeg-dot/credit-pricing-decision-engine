"""Randomized synthetic experiment for causal pricing practice."""

import numpy as np
import pandas as pd

from .data import (
    synthetic_acceptance_probability,
    synthetic_default_probability,
)
from .policy import PolicyConfig

_REQUIRED_COLUMNS = {"application_id", "preferred_rate", "requested_amount_k"}
_PRIMARY_OUTCOMES = ("acceptance_rate", "contribution_k_per_offer")
_BOOTSTRAP_CONFIDENCE_LEVEL = 0.95


def _difference_with_bootstrap_interval(
    control: np.ndarray,
    treatment: np.ndarray,
    oracle_effect: float,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    control_draws = rng.integers(
        0,
        len(control),
        size=(bootstrap_samples, len(control)),
    )
    treatment_draws = rng.integers(
        0,
        len(treatment),
        size=(bootstrap_samples, len(treatment)),
    )
    bootstrap_effects = (
        treatment[treatment_draws].mean(axis=1)
        - control[control_draws].mean(axis=1)
    )
    tail = (1.0 - _BOOTSTRAP_CONFIDENCE_LEVEL) / 2.0
    lower, upper = np.quantile(bootstrap_effects, [tail, 1.0 - tail])
    return {
        "estimate_treatment_minus_control": float(
            treatment.mean() - control.mean()
        ),
        "bootstrap_95pct_ci_lower": float(lower),
        "bootstrap_95pct_ci_upper": float(upper),
        "simulator_finite_population_ate": oracle_effect,
    }


def run_randomized_pricing_experiment(
    applications: pd.DataFrame,
    control_rate: float = 0.12,
    treatment_rate: float = 0.24,
    config: PolicyConfig | None = None,
    random_state: int = 42,
    bootstrap_samples: int = 1000,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Estimate offer-rate effects on acceptance and contribution per offer.

    Applicants are randomly assigned to one of two rates. The synthetic
    simulator makes both potential outcomes available only for an educational
    finite-population benchmark; treatment effects are estimated from assigned
    arms, not from that benchmark.
    """
    missing = _REQUIRED_COLUMNS - set(applications.columns)
    if missing:
        raise ValueError(f"Applications are missing columns: {sorted(missing)}")
    if len(applications) < 40:
        raise ValueError("At least 40 applications are required.")
    if applications["application_id"].duplicated().any():
        raise ValueError("application_id values must be unique.")
    if not np.isfinite([control_rate, treatment_rate]).all() or any(
        rate <= 0 or rate >= 1 for rate in (control_rate, treatment_rate)
    ):
        raise ValueError("Offer rates must be strictly between 0 and 1.")
    if control_rate == treatment_rate:
        raise ValueError(
            "control_rate and treatment_rate must be distinct values."
        )
    if bootstrap_samples < 200:
        raise ValueError("bootstrap_samples must be at least 200.")

    preferred_rate = applications["preferred_rate"].to_numpy(dtype=float)
    principal = applications["requested_amount_k"].to_numpy(dtype=float)
    if not np.isfinite(preferred_rate).all() or (
        (preferred_rate < 0) | (preferred_rate > 1)
    ).any():
        raise ValueError("preferred_rate must be finite and between 0 and 1.")
    if not np.isfinite(principal).all() or (principal <= 0).any():
        raise ValueError("requested_amount_k must be finite and positive.")

    config = config or PolicyConfig()
    assignment_seed, outcome_seed, bootstrap_seed = np.random.SeedSequence(
        random_state
    ).spawn(3)
    assignment_rng = np.random.default_rng(assignment_seed)
    outcome_rng = np.random.default_rng(outcome_seed)
    bootstrap_rng = np.random.default_rng(bootstrap_seed)

    n = len(applications)
    treatment_assignment = np.zeros(n, dtype=bool)
    treatment_assignment[
        assignment_rng.permutation(n)[: n // 2]
    ] = True
    common_acceptance_draw = outcome_rng.random(n)
    common_default_draw = outcome_rng.random(n)

    def potential_outcomes(rate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        acceptance_probability = synthetic_acceptance_probability(
            preferred_rate,
            rate,
        )
        default_probability = synthetic_default_probability(applications, rate)
        accepted = common_acceptance_draw < acceptance_probability
        defaulted = accepted & (common_default_draw < default_probability)
        contribution = np.where(
            accepted,
            principal
            * (
                (1.0 - defaulted.astype(float)) * rate
                - defaulted.astype(float) * config.loss_given_default
                - config.funding_rate
            ),
            0.0,
        )
        return accepted.astype(float), defaulted.astype(float), contribution

    control_acceptance, control_defaults, control_contribution = potential_outcomes(
        control_rate
    )
    treatment_acceptance, treatment_defaults, treatment_contribution = (
        potential_outcomes(treatment_rate)
    )

    observed_acceptance = np.where(
        treatment_assignment,
        treatment_acceptance,
        control_acceptance,
    )
    observed_defaults = np.where(
        treatment_assignment,
        treatment_defaults,
        control_defaults,
    )
    observed_contribution = np.where(
        treatment_assignment,
        treatment_contribution,
        control_contribution,
    )
    control_mask = ~treatment_assignment
    treatment_mask = treatment_assignment

    control_acceptance_observed = observed_acceptance[control_mask]
    treatment_acceptance_observed = observed_acceptance[treatment_mask]
    control_contribution_observed = observed_contribution[control_mask]
    treatment_contribution_observed = observed_contribution[treatment_mask]

    effects = {
        "acceptance_rate": _difference_with_bootstrap_interval(
            control_acceptance_observed,
            treatment_acceptance_observed,
            float((treatment_acceptance - control_acceptance).mean()),
            bootstrap_samples,
            bootstrap_rng,
        ),
        "contribution_k_per_offer": _difference_with_bootstrap_interval(
            control_contribution_observed,
            treatment_contribution_observed,
            float((treatment_contribution - control_contribution).mean()),
            bootstrap_samples,
            bootstrap_rng,
        ),
    }

    def arm_summary(
        mask: np.ndarray,
        rate: float,
    ) -> dict[str, float | int]:
        accepted = observed_acceptance[mask]
        defaults = observed_defaults[mask]
        contribution = observed_contribution[mask]
        accepted_count = int(accepted.sum())
        return {
            "applications": int(mask.sum()),
            "offer_rate": rate,
            "acceptance_rate": float(accepted.mean()),
            "mean_contribution_k_per_offer": float(contribution.mean()),
            "default_rate_among_accepted_descriptive": (
                float(defaults.sum() / accepted_count) if accepted_count else 0.0
            ),
        }

    report: dict[str, object] = {
        "design": "individually randomized two-arm synthetic pricing experiment",
        "seed": random_state,
        "bootstrap_samples": bootstrap_samples,
        "confidence_level": _BOOTSTRAP_CONFIDENCE_LEVEL,
        "outcomes": {
            "primary": list(_PRIMARY_OUTCOMES),
            "default_rate_among_accepted": (
                "Descriptive only; conditioning on acceptance, a post-treatment "
                "variable, does not identify a causal effect on default."
            ),
        },
        "arms": {
            "control": arm_summary(control_mask, control_rate),
            "treatment": arm_summary(treatment_mask, treatment_rate),
        },
        "intent_to_treat_effects": effects,
        "method_notes": [
            "Treatment was assigned by a seeded 1:1 random permutation.",
            "Primary effects are treatment-minus-control differences in assigned-arm means.",
            "Confidence intervals use a stratified nonparametric bootstrap within arms.",
            "The finite-population ATE is available only because both synthetic potential outcomes are generated; it is an oracle benchmark, not observed evidence.",
            "All outcomes, prices, and economic assumptions are invented; results do not transfer to actual customers or portfolios.",
        ],
    }
    detail = pd.DataFrame(
        {"application_id": applications["application_id"].to_numpy()}
    )
    detail["assigned_arm"] = np.where(treatment_assignment, "treatment", "control")
    detail["offered_rate"] = np.where(
        treatment_assignment,
        treatment_rate,
        control_rate,
    )
    detail["accepted"] = observed_acceptance.astype(int)
    detail["defaulted_if_accepted"] = observed_defaults.astype(int)
    detail["contribution_k_per_offer"] = observed_contribution
    return report, detail
