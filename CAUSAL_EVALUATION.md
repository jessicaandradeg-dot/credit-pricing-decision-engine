# Randomized pricing evaluation

## Objective

This module adds a controlled, synthetic pricing experiment to the existing credit-risk and policy prototype. It demonstrates how random assignment can estimate the **intention-to-treat (ITT)** effect of an offer-rate policy on acceptance and simulated contribution per offer.

## Design

- Population: synthetic applications held out from model training in the experiment run.
- Assignment: seeded, balanced 1:1 random permutation to the control or treatment offer rate.
- Default rates: 12% control and 24% treatment.
- Primary outcomes: acceptance (binary) and simulated contribution per offered application, including zero when an offer is not accepted.
- Effect: treatment-arm mean minus control-arm mean.
- Uncertainty: 95% percentile confidence interval from a nonparametric bootstrap, resampling independently within each randomized arm.
- Validation benchmark: because the synthetic generator creates potential outcomes under both rates, the report includes the finite-population simulator ATE. This is an oracle benchmark for the simulation, not a value observable in an actual experiment.

The same applicant-level random draws are used to construct both synthetic potential outcomes. Assignment is generated independently. Only the outcome under the assigned rate is included in the observed arm estimate.

## Interpretation safeguards

- Acceptance and contribution per randomized offer are the prespecified primary outcomes.
- Default rate among accepted offers is reported descriptively only. Acceptance is affected by the offer rate, so comparing default only among accepted applicants conditions on a post-treatment variable and is not interpreted as a causal effect.
- The experiments use invented response and default functions; the confidence intervals quantify sampling variability in that simulation, not uncertainty about real customers or a live portfolio.
- `loss_given_default` is a simplified unrecovered-loss assumption. It is not a collections strategy or a recovery model estimated from real repayment data.
- The experiment report also recalculates policy outcomes under several assumed recovery rates by changing loss-given-default. This is sensitivity analysis, not measured collections effectiveness.
- The original policy comparison still uses synthetic-oracle expected outcomes. It is not a substitute for randomized evaluation or off-policy estimation in real data.

## Run and inspect

```powershell
$env:PYTHONPATH = "src"
python -m credit_decision --n 1000 --seed 42
python -m pytest tests/test_causal_pricing.py
```

The JSON summary contains arm sizes, outcome means, ITT estimates, bootstrap intervals, and the simulator benchmark. The local CSV contains each synthetic application's randomized arm and assigned-rate outcomes; generated reports remain ignored by Git.

## Example result

For `--n 1000 --seed 42`, the experiment used 150 synthetic applications in each arm. Assignment to the 24% offer resulted in an acceptance ITT estimate of `-38.0` percentage points compared with the 12% offer (95% bootstrap CI: `-48.0` to `-28.7` points); the finite-population simulator ATE was `-34.0` points. The contribution ITT estimate was `-0.498` thousand units per offer (95% CI: `-1.361` to `0.316`), so the interval includes zero. This is one seeded realization, not evidence about real customers.

## Moving toward a real study

A real pricing pilot would require a legitimate eligible population, a qualified risk and legal review, explicit safety/affordability constraints, prespecified outcomes and stopping rules, power analysis, privacy controls, monitoring, human oversight, and approval by the responsible institution. The present exercise supplies none of these approvals or operational safeguards and must not be used to set real prices or credit eligibility.
