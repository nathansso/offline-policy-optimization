from offline_policy.experiment import run


def test_end_to_end_experiment_reports_overlap_and_value() -> None:
    result = run(episodes=240, seed=41)

    assert result["episodes"] == 240
    assert result["evaluation_steps"] > 0
    assert 0.0 <= result["effective_sample_fraction"] <= 1.0
    for key in ("ips", "snips", "doubly_robust", "fitted_q_episode_value"):
        assert isinstance(result[key], float)
