"""Deterministic synthetic logged feedback for a runnable public demonstration."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .domain import ACTION_NAMES, Action
from .features import STATE_COLUMNS


@dataclass
class State:
    turn: int
    intent: float
    urgency: float
    patience: float
    service_risk: float
    ignored_count: int
    open_opportunity: int
    member: int
    unhappy: int
    prior_nudge: int


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + np.exp(-value))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    values = np.exp(shifted)
    return values / values.sum()


def _behavior_probabilities(state: State) -> np.ndarray:
    """A stochastic historical policy with broad but uneven support."""

    logits = np.array(
        [
            0.1 + 0.8 * (1 - state.urgency),
            0.5 + 0.4 * state.patience,
            0.1 + 0.9 * state.open_opportunity + 0.7 * state.intent,
            0.2 + 0.8 * (1 - state.intent),
            -0.8 + 1.8 * state.intent + 1.0 * state.open_opportunity,
            -0.7 + 1.6 * state.member + 0.4 * state.intent,
            -0.8 + 1.5 * (state.patience < 0.25) + 0.35 * state.ignored_count,
            -1.2 + 2.4 * state.unhappy + 2.0 * state.service_risk,
        ],
        dtype=float,
    )
    learned = _softmax(logits)
    return 0.92 * learned + 0.08 / len(ACTION_NAMES)


def _reward_and_done(state: State, action: Action, rng: np.random.Generator) -> tuple[float, bool]:
    aggressive = action in {Action.NUDGE, Action.OFFER_NEXT_STEP, Action.MEMBERSHIP_TOUCH}
    opt_out = _sigmoid(-4.2 + 0.65 * state.turn + 2.2 * (1 - state.patience))
    if aggressive:
        opt_out *= 1.6 - 0.8 * state.intent
    if rng.random() < min(opt_out, 0.75):
        return -30.0, True

    if action == Action.ESCALATE:
        return (5.0 if state.unhappy or state.service_risk >= 0.8 else -8.0), True
    if action == Action.PARK:
        return (-1.0 if state.patience <= 0.25 else -4.0), True
    if action == Action.OFFER_NEXT_STEP:
        chance = _sigmoid(-2.7 + 3.7 * state.intent + 1.1 * state.prior_nudge)
        if state.open_opportunity and rng.random() < chance:
            return 40.0, True
        return -4.0, False
    if action == Action.MEMBERSHIP_TOUCH:
        chance = _sigmoid(-3.0 + 2.7 * state.intent)
        if state.member and rng.random() < chance:
            return 16.0, True
        return -3.0, False
    if action == Action.NUDGE:
        return (3.0 if state.open_opportunity else -5.0), False
    if action == Action.ASK_CONSTRAINT:
        return (2.5 if state.intent < 0.65 else 0.5), False
    if action == Action.CHECK_IN:
        return (2.0 if state.patience > 0.35 else -1.0), False
    return -0.5, False


def _advance(state: State, action: Action, rng: np.random.Generator) -> State:
    outreach = action not in {Action.WAIT, Action.PARK, Action.ESCALATE}
    ignored = state.ignored_count + int(outreach and rng.random() < 0.30)
    intent_shift = 0.10 if action == Action.NUDGE and state.open_opportunity else 0.0
    intent_shift += 0.05 if action == Action.ASK_CONSTRAINT else 0.0
    patience_cost = 0.12 if outreach else 0.03
    return State(
        turn=state.turn + 1,
        intent=float(np.clip(state.intent + intent_shift + rng.normal(0, 0.04), 0, 1)),
        urgency=state.urgency,
        patience=float(np.clip(state.patience - patience_cost, 0, 1)),
        service_risk=state.service_risk,
        ignored_count=ignored,
        open_opportunity=state.open_opportunity,
        member=state.member,
        unhappy=state.unhappy,
        prior_nudge=int(state.prior_nudge or action == Action.NUDGE),
    )


def generate_logged_feedback(
    episodes: int = 3_000,
    *,
    horizon: int = 4,
    seed: int = 2026,
) -> pd.DataFrame:
    """Generate short logged trajectories with known behavior propensities."""

    if episodes < 1 or horizon < 1:
        raise ValueError("episodes and horizon must be positive")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int | str | bool]] = []
    for episode_id in range(episodes):
        state = State(
            turn=0,
            intent=float(rng.beta(2.2, 2.8)),
            urgency=float(rng.beta(2.0, 2.0)),
            patience=float(rng.beta(4.0, 1.5)),
            service_risk=float(rng.beta(1.2, 5.0)),
            ignored_count=0,
            open_opportunity=int(rng.random() < 0.62),
            member=int(rng.random() < 0.38),
            unhappy=int(rng.random() < 0.06),
            prior_nudge=0,
        )
        for step in range(horizon):
            probabilities = _behavior_probabilities(state)
            action = Action(int(rng.choice(len(ACTION_NAMES), p=probabilities)))
            reward, terminal = _reward_and_done(state, action, rng)
            next_state = _advance(state, action, rng)
            done = terminal or step == horizon - 1
            row = {
                "episode_id": episode_id,
                "step": step,
                **asdict(state),
                "action": action.name,
                "action_id": int(action),
                "action_prob": float(probabilities[int(action)]),
                "reward": reward,
                "done": done,
                **{f"next_{key}": value for key, value in asdict(next_state).items()},
            }
            rows.append(row)
            if done:
                break
            state = next_state
    frame = pd.DataFrame(rows)
    return frame.loc[
        :,
        [
            "episode_id",
            "step",
            *STATE_COLUMNS,
            "action",
            "action_id",
            "action_prob",
            "reward",
            "done",
            *(f"next_{column}" for column in STATE_COLUMNS),
        ],
    ]
