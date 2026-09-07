# Offline Policy Optimization from Logged Feedback

A reproducible contextual-bandit project for choosing actions when every training example comes
from a historical policy and counterfactual outcomes are unavailable.

The central design is deliberately hybrid:

```text
logged trajectories
    -> decision-time features
    -> value model q(s, a) -----------\
    -> behavior support pi_old(a|s) ---+-> constrained action ranking
    -> explicit safety rules ---------/
                                         |
                                         v
                         IPS / SNIPS / doubly robust / FQE
```

The model ranks; explicit code owns authority. An action must be structurally valid, sufficiently
supported by historical behavior, and outside forced-escalation conditions before its predicted
value can matter.

## Why this project is interesting

- **Selection bias:** high observed reward can mean the old policy chose an action only in easy
  states. Raw action averages are not a policy evaluation.
- **Delayed value:** a low-reward action can improve the state for a later action, so the project
  reports fitted-Q trajectory value alongside turn-level estimators.
- **Coverage:** effective sample size makes extrapolation visible when a candidate policy moves
  beyond the historical policy's support.
- **Safety:** deterministic constraints cover invalid and high-risk decisions that sparse logs
  cannot reliably teach.
- **Reproducibility:** the public demo generates its own trajectories, fixes every random seed,
  splits by episode, and requires no credentials or network calls.

## Run it

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"     # Windows
.venv/Scripts/python -m pytest
.venv/Scripts/offline-policy-demo --episodes 3000
```

On macOS or Linux, replace `.venv/Scripts/` with `.venv/bin/`.

The CLI prints a JSON report containing IPS, SNIPS, doubly robust value, fitted-Q episode value,
and effective sample fraction. All output is labeled as synthetic demonstration evidence rather
than business impact.

## Repository map

```text
src/offline_policy/
  synthetic.py    generated logged trajectories with known propensities
  features.py     decision-time-only feature boundary
  policy.py       value model, behavior support, and safety envelope
  evaluation.py   IPS, SNIPS, doubly robust estimation, ESS, and FQE
  experiment.py   deterministic end-to-end train/evaluate split
tests/            estimator, safety, determinism, and integration checks
docs/             compact methodology and limitations
```

Read [the methodology](docs/methodology.md) for the estimator assumptions and failure modes.

## Public-work-sample boundary

This is a clean-room portfolio reconstruction of methodology developed during an interview
exercise. It contains no original prompt, supplied dataset, starter code, trained weights,
company-identifying material, or results from the private exercise. The generated dataset and
all code in this repository are self-contained demonstration material.

## License

MIT
