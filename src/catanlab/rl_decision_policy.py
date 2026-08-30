from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from catanlab.rl_special_actions import (
    CategoricalDecisionInput,
)
from catanlab.rl_teacher import (
    TeacherDecisionKind,
)


@dataclass(frozen=True)
class LearnedDecisionRequest:
    """
    Model-facing description of one realism-v2 decision.

    The policy predicts a categorical action ID. Conversion
    back to a simulator-facing value remains the
    responsibility of CategoricalDecisionInput.

    This keeps ordinary and phase-specific action spaces
    separate while giving learned policies one common API.
    """

    decision_kind: TeacherDecisionKind
    observation: tuple[float, ...]
    decision_input: CategoricalDecisionInput

    def __post_init__(self):
        if not self.observation:
            raise ValueError(
                "Learned decision observation cannot be empty."
            )

        if self.decision_input.action_dim <= 0:
            raise ValueError(
                "Learned decision must contain at least "
                "one categorical choice."
            )

        if not self.decision_input.legal_action_ids:
            raise ValueError(
                "Learned decision must contain at least "
                "one legal categorical choice."
            )

    @property
    def action_dim(self) -> int:
        return self.decision_input.action_dim

    @property
    def legal_mask(self) -> tuple[bool, ...]:
        return self.decision_input.legal_mask

    @property
    def legal_action_ids(self) -> tuple[int, ...]:
        return self.decision_input.legal_action_ids

    def decode(self, action_id: int):
        """
        Decode one model-selected categorical action ID.

        Illegal choices are rejected here even if a buggy
        learned policy attempts to return one.
        """
        if (
            action_id < 0
            or action_id >= self.action_dim
        ):
            raise ValueError(
                f"Invalid learned decision action ID: "
                f"{action_id}"
            )

        if not self.legal_mask[action_id]:
            raise ValueError(
                "Learned policy selected an illegal "
                "categorical action: "
                f"kind={self.decision_kind.value}, "
                f"action_id={action_id}"
            )

        return self.decision_input.decode(
            action_id
        )


class LearnedDecisionPolicy(Protocol):
    """
    Common realism-v2 learned-policy boundary.

    Implementations may use fixed output heads, dynamic
    candidate scoring, or another architecture, but must
    return an action ID in the request's categorical space.
    """

    def choose_decision(
        self,
        request: LearnedDecisionRequest,
    ) -> int:
        ...


def validate_decision_action_id(
    request: LearnedDecisionRequest,
    action_id: int,
) -> int:
    """
    Validate a policy-produced action ID without decoding it.
    """
    if (
        action_id < 0
        or action_id >= request.action_dim
    ):
        raise ValueError(
            f"Invalid learned decision action ID: "
            f"{action_id}"
        )

    if not request.legal_mask[
        action_id
    ]:
        raise ValueError(
            "Learned policy selected an illegal "
            "categorical action: "
            f"kind={request.decision_kind.value}, "
            f"action_id={action_id}"
        )

    return action_id


def choose_decision_value(
    policy: LearnedDecisionPolicy,
    request: LearnedDecisionRequest,
):
    """
    Execute the common learned-policy contract and return
    the simulator-facing categorical value.
    """
    action_id = policy.choose_decision(
        request
    )

    validate_decision_action_id(
        request,
        action_id,
    )

    return request.decision_input.decode(
        action_id
    )
