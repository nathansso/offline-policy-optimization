"""End-to-end, credential-free demonstration on generated logged feedback."""

from __future__ import annotations

import argparse
import json

import numpy as np

from .evaluation import (
    doubly_robust,
    effective_sample_size,
    fitted_q_value,
    importance_weights,
    ips,
    snips,
)
from .policy import FittedPolicy
from .synthetic import generate_logged_feedback


def run(episodes: int = 3_000, seed: int = 2026) -> dict[str, float | int | str]:
    logs = generate_logged_feedback(episodes, seed=seed)
    episode_ids = logs["episode_id"].drop_duplicates().to_numpy(copy=True)
    rng = np.random.default_rng(seed + 1)
    rng.shuffle(episode_ids)
    split = int(0.70 * len(episode_ids))
    train_ids = set(episode_ids[:split])
    train = logs.loc[logs["episode_id"].isin(train_ids)].reset_index(drop=True)
    evaluation = logs.loc[~logs["episode_id"].isin(train_ids)].reset_index(drop=True)

    policy = FittedPolicy.fit(train, random_state=seed)
    target = policy.probabilities(evaluation)
    actions = evaluation["action_id"].to_numpy(dtype=int)
    rewards = evaluation["reward"].to_numpy(dtype=float)
    weights = importance_weights(
        target,
        actions,
        evaluation["action_prob"].to_numpy(dtype=float),
    )
    q_values = policy.q_values(evaluation)
    return {
        "note": "Synthetic demonstration only; these values are not business-impact estimates.",
        "episodes": episodes,
        "logged_steps": len(logs),
        "evaluation_steps": len(evaluation),
        "ips": ips(weights, rewards),
        "snips": snips(weights, rewards),
        "doubly_robust": doubly_robust(weights, rewards, q_values, target, actions),
        "fitted_q_episode_value": fitted_q_value(evaluation, policy),
        "effective_sample_fraction": effective_sample_size(weights),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3_000)
    parser.add_argument("--seed", type=int, default=2026)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.episodes, arguments.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
