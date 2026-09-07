from offline_policy.domain import Action
from offline_policy.policy import FittedPolicy
from offline_policy.synthetic import generate_logged_feedback


def test_forced_escalation_and_structural_masks() -> None:
    logs = generate_logged_feedback(400, seed=9)
    policy = FittedPolicy.fit(logs, random_state=9)
    states = logs.iloc[:3].copy()
    states.loc[states.index[0], ["unhappy", "service_risk"]] = [1, 0.1]
    states.loc[states.index[1], ["open_opportunity", "member"]] = [0, 0]
    states.loc[states.index[2], "patience"] = 0.0

    assert policy.choose(states.iloc[[0]]) == (Action.ESCALATE.name,)
    eligible = policy.eligible_actions(states)
    assert not eligible[1, Action.OFFER_NEXT_STEP]
    assert not eligible[1, Action.MEMBERSHIP_TOUCH]
    assert not eligible[2, Action.NUDGE]


def test_training_and_inference_are_reproducible() -> None:
    logs = generate_logged_feedback(350, seed=29)
    sample = logs.iloc[:25]
    first = FittedPolicy.fit(logs, random_state=31)
    second = FittedPolicy.fit(logs, random_state=31)

    assert first.choose(sample) == second.choose(sample)
