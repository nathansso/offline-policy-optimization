"""Small domain vocabulary shared by training, serving, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Action(IntEnum):
    WAIT = 0
    CHECK_IN = 1
    NUDGE = 2
    ASK_CONSTRAINT = 3
    OFFER_NEXT_STEP = 4
    MEMBERSHIP_TOUCH = 5
    PARK = 6
    ESCALATE = 7


ACTION_NAMES = tuple(action.name for action in Action)


@dataclass(frozen=True)
class PolicyThresholds:
    """Auditable controls kept outside the learned value model."""

    escalation_risk: float = 0.80
    cold_start_intent: float = 0.45
    patience_floor: float = 0.15
    support_floor: float = 0.02
