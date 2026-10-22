"""Run the synthetic pricing-policy comparison."""

import argparse
from pathlib import Path

from .decisionbot import save_decision_audit
from .experiment import report_as_json, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the synthetic credit-pricing decision experiment."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=6000,
        help="Number of synthetic applications.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for synthetic reports.",
    )
    parser.add_argument(
        "--database-path",
        type=Path,
        help="Optional SQLite path for the synthetic decision audit table.",
    )
    args = parser.parse_args()

    report, details = run_experiment(n=args.n, random_state=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "experiment_summary.json").write_text(
        report_as_json(report),
        encoding="utf-8",
    )
    details.to_csv(args.output_dir / "synthetic_test_decisions.csv", index=False)
    database_path = args.database_path or args.output_dir / "decisionbot.sqlite"
    decision_columns = {
        "application_id": "application_id",
        "risk_aware_pricing_decision": "decision",
        "risk_aware_pricing_reason_code": "reason_code",
        "risk_aware_pricing_offered_rate": "offered_rate",
        "risk_aware_pricing_predicted_default_probability": (
            "predicted_default_probability"
        ),
        "risk_aware_pricing_expected_profit_k": "expected_profit_k",
        "risk_aware_pricing_human_review_cost_k": "human_review_cost_k",
        "risk_aware_pricing_expected_net_contribution_k": (
            "expected_net_contribution_k"
        ),
        "risk_aware_pricing_policy_version": "policy_version",
    }
    save_decision_audit(
        details[list(decision_columns)].rename(columns=decision_columns),
        database_path,
    )
    print(report_as_json(report))
    print(f"\nSynthetic decision details saved to: {args.output_dir.resolve()}")
    print(f"SQLite decision audit saved to: {database_path.resolve()}")


if __name__ == "__main__":
    main()
