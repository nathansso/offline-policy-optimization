import pandas as pd

from offline_policy.synthetic import generate_logged_feedback


def test_generation_is_deterministic_and_propensities_are_valid() -> None:
    first = generate_logged_feedback(30, seed=17)
    second = generate_logged_feedback(30, seed=17)

    pd.testing.assert_frame_equal(first, second)
    assert first["action_prob"].between(0, 1, inclusive="neither").all()
    assert first.groupby("episode_id")["step"].min().eq(0).all()
