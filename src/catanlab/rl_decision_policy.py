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


class TorchFixedHeadDecisionPolicy:
    """
    Adapter from RealismV2ActorCritic fixed decision heads
    to the LearnedDecisionPolicy action-ID contract.

    Supported kinds:
        MONOPOLY_RESOURCE
        YEAR_OF_PLENTY
        TRADE_RESPONSE
    """

    def __init__(
        self,
        model,
        *,
        deterministic: bool = True,
        seed: int | None = None,
    ):
        import torch

        self.model = model
        self.deterministic = deterministic

        self.generator = torch.Generator()

        if seed is not None:
            self.generator.manual_seed(
                seed
            )

    def choose_decision(
        self,
        request: LearnedDecisionRequest,
    ) -> int:
        import torch

        from catanlab.rl_model import (
            mask_policy_logits,
        )

        if not self.model.is_fixed_decision_kind(
            request.decision_kind
        ):
            raise ValueError(
                "TorchFixedHeadDecisionPolicy does not "
                "support dynamic decision kind: "
                f"{request.decision_kind.value}"
            )

        expected_dim = (
            self.model.fixed_decision_dim(
                request.decision_kind
            )
        )

        if request.action_dim != expected_dim:
            raise ValueError(
                "Decision-space dimension does not match "
                "the model head: "
                f"kind={request.decision_kind.value}, "
                f"request_dim={request.action_dim}, "
                f"model_dim={expected_dim}"
            )

        observation = torch.tensor(
            request.observation,
            dtype=torch.float32,
        ).unsqueeze(0)

        legal_mask = torch.tensor(
            request.legal_mask,
            dtype=torch.bool,
        ).unsqueeze(0)

        self.model.eval()

        with torch.no_grad():
            logits = (
                self.model.fixed_decision_logits(
                    observation,
                    request.decision_kind,
                )
            )

            masked_logits = mask_policy_logits(
                logits,
                legal_mask,
            )

            if self.deterministic:
                action_id = int(
                    torch.argmax(
                        masked_logits,
                        dim=-1,
                    ).item()
                )

            else:
                probabilities = torch.softmax(
                    masked_logits,
                    dim=-1,
                )

                action_id = int(
                    torch.multinomial(
                        probabilities,
                        num_samples=1,
                        generator=self.generator,
                    ).item()
                )

        return validate_decision_action_id(
            request,
            action_id,
        )


class TorchRealismV2DecisionPolicy:
    """
    LearnedDecisionPolicy adapter for the complete
    realism-v2 categorical decision model.

    Fixed-size decision kinds route through the model's
    fixed heads.

    Variable-size decision kinds first encode the current
    categorical vocabulary into candidate feature vectors,
    then route through the dynamic candidate scorer.

    The returned value is always a categorical action ID.
    Decoding back to a simulator-facing object remains the
    responsibility of CategoricalDecisionInput.
    """

    def __init__(
        self,
        model,
        *,
        deterministic: bool = True,
        seed: int | None = None,
    ):
        import torch

        self.model = model
        self.deterministic = deterministic

        self.generator = torch.Generator()

        if seed is not None:
            self.generator.manual_seed(
                seed
            )

    def _select_action_id(
        self,
        logits,
        legal_mask,
    ) -> int:
        import torch

        from catanlab.rl_model import (
            mask_policy_logits,
        )

        masked_logits = mask_policy_logits(
            logits,
            legal_mask,
        )

        if self.deterministic:
            return int(
                torch.argmax(
                    masked_logits,
                    dim=-1,
                ).item()
            )

        probabilities = torch.softmax(
            masked_logits,
            dim=-1,
        )

        return int(
            torch.multinomial(
                probabilities,
                num_samples=1,
                generator=self.generator,
            ).item()
        )

    def choose_decision(
        self,
        request: LearnedDecisionRequest,
    ) -> int:
        import torch

        observation = torch.tensor(
            request.observation,
            dtype=torch.float32,
        ).unsqueeze(0)

        legal_mask = torch.tensor(
            request.legal_mask,
            dtype=torch.bool,
        ).unsqueeze(0)

        self.model.eval()

        with torch.no_grad():
            if self.model.is_fixed_decision_kind(
                request.decision_kind
            ):
                expected_dim = (
                    self.model.fixed_decision_dim(
                        request.decision_kind
                    )
                )

                if (
                    request.action_dim
                    != expected_dim
                ):
                    raise ValueError(
                        "Decision-space dimension does "
                        "not match the fixed model head: "
                        f"kind="
                        f"{request.decision_kind.value}, "
                        f"request_dim="
                        f"{request.action_dim}, "
                        f"model_dim="
                        f"{expected_dim}"
                    )

                logits = (
                    self.model.fixed_decision_logits(
                        observation,
                        request.decision_kind,
                    )
                )

            elif self.model.is_dynamic_decision_kind(
                request.decision_kind
            ):
                from catanlab.rl_candidate_features import (
                    encode_dynamic_decision_input,
                )

                encoded_candidates = (
                    encode_dynamic_decision_input(
                        request.decision_kind,
                        request.decision_input,
                    )
                )

                if (
                    len(encoded_candidates)
                    != request.action_dim
                ):
                    raise RuntimeError(
                        "Dynamic candidate encoding "
                        "changed categorical dimension: "
                        f"kind="
                        f"{request.decision_kind.value}, "
                        f"request_dim="
                        f"{request.action_dim}, "
                        f"encoded_dim="
                        f"{len(encoded_candidates)}"
                    )

                candidate_features = torch.tensor(
                    encoded_candidates,
                    dtype=torch.float32,
                )

                logits = (
                    self.model.dynamic_decision_logits(
                        observation,
                        request.decision_kind,
                        candidate_features,
                    )
                )

            else:
                raise ValueError(
                    "Realism-v2 model does not support "
                    "decision kind: "
                    f"{request.decision_kind!r}"
                )

            if logits.shape != legal_mask.shape:
                raise RuntimeError(
                    "Realism-v2 decision logits do not "
                    "match the categorical legal mask: "
                    f"logits={logits.shape}, "
                    f"mask={legal_mask.shape}"
                )

            action_id = self._select_action_id(
                logits,
                legal_mask,
            )

        return validate_decision_action_id(
            request,
            action_id,
        )
