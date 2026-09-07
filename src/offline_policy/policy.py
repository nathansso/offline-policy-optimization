"""Learned action ranking with explicit support and safety boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .domain import ACTION_NAMES, Action, PolicyThresholds
from .features import all_action_matrix, state_action_matrix, state_matrix


@dataclass
class FittedPolicy:
    """Contextual-bandit value model plus an auditable deterministic envelope."""

    value_model: HistGradientBoostingRegressor
    behavior_model: HistGradientBoostingClassifier
    behavior_classes: np.ndarray
    thresholds: PolicyThresholds

    @classmethod
    def fit(
        cls,
        logs: pd.DataFrame,
        *,
        thresholds: PolicyThresholds | None = None,
        random_state: int = 7,
    ) -> FittedPolicy:
        required = {"action_id", "reward", "episode_id"}
        missing = sorted(required.difference(logs.columns))
        if missing:
            raise ValueError(f"missing training columns: {', '.join(missing)}")

        action_ids = logs["action_id"].to_numpy(dtype=int)
        value_model = HistGradientBoostingRegressor(
            max_iter=160,
            learning_rate=0.06,
            max_leaf_nodes=24,
            l2_regularization=1.0,
            random_state=random_state,
        )
        value_model.fit(
            state_action_matrix(logs, action_ids),
            logs["reward"].clip(logs["reward"].quantile(0.01), logs["reward"].quantile(0.99)),
        )

        behavior_model = HistGradientBoostingClassifier(
            max_iter=140,
            learning_rate=0.06,
            max_leaf_nodes=24,
            l2_regularization=1.0,
            random_state=random_state + 1,
        )
        behavior_model.fit(state_matrix(logs), action_ids)
        return cls(
            value_model=value_model,
            behavior_model=behavior_model,
            behavior_classes=behavior_model.classes_.astype(int),
            thresholds=thresholds or PolicyThresholds(),
        )

    def q_values(self, states: pd.DataFrame) -> np.ndarray:
        flat = self.value_model.predict(all_action_matrix(states))
        return np.asarray(flat, dtype=float).reshape(len(states), len(ACTION_NAMES))

    def behavior_probabilities(self, states: pd.DataFrame) -> np.ndarray:
        observed = self.behavior_model.predict_proba(state_matrix(states))
        probabilities = np.zeros((len(states), len(ACTION_NAMES)), dtype=float)
        probabilities[:, self.behavior_classes] = observed
        return probabilities

    def eligible_actions(self, states: pd.DataFrame) -> np.ndarray:
        """Return the pre-support safety mask; forced escalation is applied later."""

        n = len(states)
        allow = np.ones((n, len(ACTION_NAMES)), dtype=bool)
        allow[:, Action.ESCALATE] = False

        no_opportunity = states["open_opportunity"].to_numpy(dtype=int) == 0
        allow[
            np.ix_(no_opportunity, [Action.NUDGE, Action.ASK_CONSTRAINT, Action.OFFER_NEXT_STEP])
        ] = False

        not_member = states["member"].to_numpy(dtype=int) == 0
        allow[np.ix_(not_member, [Action.MEMBERSHIP_TOUCH])] = False

        low_patience = states["patience"].to_numpy(dtype=float) <= self.thresholds.patience_floor
        proactive = [
            Action.NUDGE,
            Action.ASK_CONSTRAINT,
            Action.OFFER_NEXT_STEP,
            Action.MEMBERSHIP_TOUCH,
        ]
        allow[np.ix_(low_patience, proactive)] = False

        cold_start = (states["turn"].to_numpy(dtype=int) == 0) & (
            states["intent"].to_numpy(dtype=float) < self.thresholds.cold_start_intent
        )
        allow[np.ix_(cold_start, [Action.OFFER_NEXT_STEP])] = False
        return allow

    def probabilities(self, states: pd.DataFrame) -> np.ndarray:
        q_values = self.q_values(states)
        allow = self.eligible_actions(states)
        support = self.behavior_probabilities(states) >= self.thresholds.support_floor
        allowed = allow & support
        empty = ~allowed.any(axis=1)
        allowed[empty, Action.WAIT] = True

        choices = np.where(allowed, q_values, -np.inf).argmax(axis=1)
        forced = (states["unhappy"].to_numpy(dtype=int) == 1) | (
            states["service_risk"].to_numpy(dtype=float) >= self.thresholds.escalation_risk
        )
        choices[forced] = Action.ESCALATE
        probabilities = np.zeros_like(q_values)
        probabilities[np.arange(len(states)), choices] = 1.0
        return probabilities

    def choose(self, states: pd.DataFrame) -> tuple[str, ...]:
        choices = self.probabilities(states).argmax(axis=1)
        return tuple(ACTION_NAMES[index] for index in choices)
