# Architecture and algorithm notes

## Scope

This repository is an offline, synthetic research prototype. The system compares a fixed-rate policy with a rule-based risk-aware pricing policy. It has no production API, persistent customer store, or connection to a lender.

## Components and data flow

```text
Synthetic data generator
  ├─ feature records and randomized historical offers
  └─ simulator-generated default outcome
       ↓
Deterministic train/test split (stratified by outcome)
  ├─ Logistic regression pipeline (standardization + classifier)
  └─ MLP pipeline (standardization + two ReLU hidden layers)
       ↓
Held-out predictions at observed historical offer rates
       ↓
Model metrics: ROC AUC and Brier score

Logistic regression prediction at candidate rates
       ↓
Input safeguards → fixed-rate / risk-aware policy
       ↓
Reason-coded decisions → synthetic-oracle policy evaluation
       ↓
Recovery-rate assumption sensitivity
       ↓
Seeded randomized pricing experiment → ITT and bootstrap intervals
       ↓
JSON summary and CSV detail
```

The neural network is a comparison model only. The pricing policies continue using logistic regression, preserving the original policy behavior and making the baseline explicit.

## Module responsibilities

| Module | Responsibility |
|---|---|
| `data.py` | Generate fictional application records and known simulator probabilities. |
| `model.py` | Validate training/prediction inputs; fit the standardized logistic and MLP estimators. |
| `policy.py` | Validate decision inputs, apply fixed-rate or risk-aware rules, and attach reasons/version. |
| `experiment.py` | Split data, fit models, compare held-out classification metrics, evaluate policies with the synthetic oracle, and serialize reports. |
| `causal_pricing.py` | Randomize a held-out synthetic cohort, simulate assigned-arm outcomes, estimate ITT differences, bootstrap intervals, and compare against the simulator's known finite-population effect. |
| `security_demo.py` | Demonstrate authenticated symmetric encryption and asymmetric wrapping of a one-time key. |
| `crypto_demo.py` | Run a local round trip over a fixed fictional sample and print metadata only. |

## Data and model contract

One row is one generated application. `application_id` is a synthetic identifier, not a predictor. Numeric features, the randomized historical offer, and the simulated default outcome are defined in [`src/credit_decision/data.py`](./src/credit_decision/data.py). The model inputs are the six fields in `FEATURES` plus `historical_offered_rate`; `defaulted` is the binary training target. `preferred_rate` is reserved for the synthetic acceptance curve and is not used as a risk-model feature.

Training and held-out evaluation use a reproducible, stratified split. Standardization is fitted inside each scikit-learn pipeline using training records only. ROC AUC evaluates ranking of held-out records; Brier score evaluates probabilistic accuracy. The classifier comparison uses identical features and held-out records. Results are conditional on the invented simulator and do not establish external validity.

## Algorithms and computational shape

Let `n` be the number of applications, `p` the number of model features, `m` the number of candidate rates, `h1` and `h2` the MLP hidden-layer widths, and `e` the number of training epochs/iterations.

- Synthetic record generation and feature validation are linear in the record count: approximately `O(n p)`.
- Logistic model training is iterative and depends on `e`, `n`, and `p`; prediction is approximately `O(n p)`.
- MLP training is approximately `O(e n (p h1 + h1 h2))`; inference is approximately `O(n (p h1 + h1 h2))`.
- Risk-aware pricing evaluates the model across `m` candidate rates and computes a feasible maximum per application: approximately `O(m n p)` for the logistic model, plus `O(n m)` policy selection.
- The MLP comparison is not called by policy decisions; it adds one training fit and one held-out evaluation.
- The causal experiment simulates two potential outcomes per application to expose an oracle finite-population effect, but estimates ITT from the independently randomized assigned arms. Bootstrap work is approximately `O(B n)` where `B` is the number of resamples.
- Recovery sensitivity reruns the policies for a small fixed grid of assumed recovery rates. It changes loss-given-default and is not a fitted model of collection outcomes.

These are high-level operation counts for explaining design, not performance benchmarks.

## Error handling and reproducibility

Input schemas, binary target classes, offered-rate ranges, row alignment, and policy assumptions are checked explicitly. Experiments use a caller-supplied random seed; both estimators use fixed random states. Invalid encrypted payloads fail authentication instead of returning plaintext. The crypto demonstration creates keys in memory and does not persist them.

## Extension points

- Add schema/version metadata to exported synthetic reports.
- Add calibration curves, threshold-independent uncertainty intervals, and repeated-seed model comparisons.
- Add temporal splits only when a meaningful synthetic time process is defined.
- Keep deployment, real customer data, production keys, and lending decisions out of this portfolio prototype.
