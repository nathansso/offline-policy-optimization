"""Decision-time-only feature transformations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .domain import ACTION_NAMES

STATE_COLUMNS = (
    "turn",
    "intent",
    "urgency",
    "patience",
    "service_risk",
    "ignored_count",
    "open_opportunity",
    "member",
    "unhappy",
    "prior_nudge",
)

NEXT_STATE_COLUMNS = tuple(f"next_{column}" for column in STATE_COLUMNS)


def state_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Return the exact state observed at decision time."""

    missing = sorted(set(STATE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"missing decision-time features: {', '.join(missing)}")
    return frame.loc[:, STATE_COLUMNS].to_numpy(dtype=float, copy=True)


def next_state_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Project logged next-state columns back into the decision-time schema."""

    missing = sorted(set(NEXT_STATE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"missing next-state features: {', '.join(missing)}")
    return frame.loc[:, NEXT_STATE_COLUMNS].rename(
        columns=dict(zip(NEXT_STATE_COLUMNS, STATE_COLUMNS, strict=True))
    )


def state_action_matrix(frame: pd.DataFrame, actions: np.ndarray) -> np.ndarray:
    """Append a one-hot action to each state without admitting outcome fields."""

    action_ids = np.asarray(actions, dtype=int)
    if action_ids.shape != (len(frame),):
        raise ValueError("one action is required for every state")
    if np.any((action_ids < 0) | (action_ids >= len(ACTION_NAMES))):
        raise ValueError("action id outside the declared action vocabulary")
    one_hot = np.eye(len(ACTION_NAMES), dtype=float)[action_ids]
    return np.concatenate((state_matrix(frame), one_hot), axis=1)


def all_action_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Expand each state once per candidate action, state-major."""

    repeated = frame.loc[frame.index.repeat(len(ACTION_NAMES))].reset_index(drop=True)
    actions = np.tile(np.arange(len(ACTION_NAMES)), len(frame))
    return state_action_matrix(repeated, actions)
