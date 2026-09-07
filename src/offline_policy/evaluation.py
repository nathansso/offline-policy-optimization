"""Off-policy estimators with overlap diagnostics and trajectory-aware uncertainty."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .domain import ACTION_NAMES
from .features import all_action_matrix, next_state_frame, state_action_matrix
from .policy import FittedPolicy


def importance_weights(
    target_probabilities: np.ndarray,
    logged_actions: np.ndarray,
    behavior_propensities: np.ndarray,
    *,
    clip: float = 20.0,
) -> np.ndarray:
    """Compute clipped target/behavior ratios on the logged action."""

    actions = np.asarray(logged_actions, dtype=int)
    propensities = np.asarray(behavior_propensities, dtype=float)
    if target_probabilities.shape[0] != len(actions) or len(actions) != len(propensities):
        raise ValueError("probabilities, actions, and propensities must have equal length")
    if np.any(propensities <= 0):
        raise ValueError("logged propensities must be positive")
    chosen = target_probabilities[np.arange(len(actions)), actions]
    return np.minimum(chosen / propensities, clip)


def effective_sample_size(weights: np.ndarray) -> float:
    """Return ESS as a fraction of the observed rows."""

    values = np.asarray(weights, dtype=float)
    denominator = np.square(values).sum()
    if len(values) == 0 or denominator == 0:
        return 0.0
    return float(np.square(values.sum()) / denominator / len(values))


def ips(weights: np.ndarray, rewards: np.ndarray) -> float:
    return float(np.mean(np.asarray(weights) * np.asarray(rewards)))


def snips(weights: np.ndarray, rewards: np.ndarray) -> float:
    values = np.asarray(weights, dtype=float)
    if values.sum() == 0:
        return float("nan")
    return float(np.sum(values * np.asarray(rewards)) / values.sum())


def doubly_robust(
    weights: np.ndarray,
    rewards: np.ndarray,
    q_values: np.ndarray,
    target_probabilities: np.ndarray,
    logged_actions: np.ndarray,
) -> float:
    """Combine a direct value model with an importance-weighted residual."""

    actions = np.asarray(logged_actions, dtype=int)
    direct = np.sum(target_probabilities * q_values, axis=1)
    logged_q = q_values[np.arange(len(actions)), actions]
    corrected = direct + np.asarray(weights) * (np.asarray(rewards) - logged_q)
    return float(np.mean(corrected))


def clustered_interval(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    repetitions: int = 300,
    seed: int = 11,
) -> tuple[float, float]:
    """Bootstrap whole trajectories rather than correlated individual decisions."""

    episode_ids = frame["episode_id"].drop_duplicates().to_numpy()
    if len(episode_ids) < 2:
        raise ValueError("at least two episodes are required")
    by_episode = {episode: frame.loc[frame["episode_id"] == episode] for episode in episode_ids}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repetitions):
        selected = rng.choice(episode_ids, size=len(episode_ids), replace=True)
        sample = pd.concat([by_episode[episode] for episode in selected], ignore_index=True)
        estimates.append(statistic(sample))
    return tuple(float(value) for value in np.quantile(estimates, [0.05, 0.95]))


def fitted_q_value(
    logs: pd.DataFrame,
    policy: FittedPolicy,
    *,
    discount: float = 0.95,
    iterations: int = 6,
    random_state: int = 23,
) -> float:
    """Estimate trajectory value under the target policy with fitted-Q evaluation."""

    actions = logs["action_id"].to_numpy(dtype=int)
    current = state_action_matrix(logs, actions)
    following = next_state_frame(logs)
    target_next = policy.probabilities(following)
    rewards = logs["reward"].to_numpy(dtype=float)
    not_done = 1.0 - logs["done"].to_numpy(dtype=float)
    targets = rewards.copy()
    evaluator: HistGradientBoostingRegressor | None = None

    for iteration in range(iterations):
        evaluator = HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.06,
            max_leaf_nodes=20,
            l2_regularization=2.0,
            random_state=random_state + iteration,
        )
        evaluator.fit(current, targets)
        q_next = evaluator.predict(all_action_matrix(following)).reshape(-1, len(ACTION_NAMES))
        next_value = np.sum(target_next * q_next, axis=1)
        targets = rewards + discount * not_done * next_value

    assert evaluator is not None
    initial = logs.loc[logs["step"] == 0]
    q_initial = evaluator.predict(all_action_matrix(initial)).reshape(-1, len(ACTION_NAMES))
    target_initial = policy.probabilities(initial)
    return float(np.mean(np.sum(target_initial * q_initial, axis=1)))
