"""Run the synthetic pricing-policy comparison."""

import argparse
from pathlib import Path

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
    args = parser.parse_args()

    report, details = run_experiment(n=args.n, random_state=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "experiment_summary.json").write_text(
        report_as_json(report),
        encoding="utf-8",
    )
    details.to_csv(args.output_dir / "synthetic_test_decisions.csv", index=False)
    print(report_as_json(report))
    print(f"\nSynthetic decision details saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
