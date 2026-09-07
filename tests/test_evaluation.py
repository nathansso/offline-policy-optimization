import numpy as np

from offline_policy.evaluation import (
    doubly_robust,
    effective_sample_size,
    importance_weights,
    ips,
    snips,
)


def test_estimators_match_a_small_hand_calculation() -> None:
    target = np.array([[0.8, 0.2], [0.4, 0.6]])
    actions = np.array([0, 1])
    behavior = np.array([0.4, 0.3])
    rewards = np.array([2.0, 4.0])
    q_values = np.array([[1.0, 0.0], [1.5, 3.0]])

    weights = importance_weights(target, actions, behavior)

    np.testing.assert_allclose(weights, [2.0, 2.0])
    assert ips(weights, rewards) == 6.0
    assert snips(weights, rewards) == 3.0
    assert effective_sample_size(weights) == 1.0
    assert doubly_robust(weights, rewards, q_values, target, actions) == 3.6


def test_zero_target_support_is_visible() -> None:
    target = np.array([[0.0, 1.0], [1.0, 0.0]])
    weights = importance_weights(target, np.array([0, 1]), np.array([0.5, 0.5]))

    np.testing.assert_array_equal(weights, [0.0, 0.0])
    assert effective_sample_size(weights) == 0.0
