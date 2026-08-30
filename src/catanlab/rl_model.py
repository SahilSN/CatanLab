from __future__ import annotations

import torch
from torch import nn


class CatanActorCritic(nn.Module):
    """
    Small feed-forward actor-critic network for the
    fixed CatanLab RL interface.

    Input:
        observation vector

    Outputs:
        policy logits over fixed action IDs
        scalar state value
    """

    def __init__(
        self,
        observation_dim: int = 1138,
        action_dim: int = 202,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.backbone = nn.Sequential(
            nn.Linear(
                observation_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(
            hidden_dim,
            action_dim,
        )

        self.value_head = nn.Linear(
            hidden_dim,
            1,
        )

    def forward(
        self,
        observation: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        features = self.backbone(
            observation
        )

        logits = self.policy_head(
            features
        )

        value = self.value_head(
            features
        ).squeeze(-1)

        return logits, value


def mask_policy_logits(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Force illegal actions to have effectively zero
    probability after softmax.

    `legal_mask` must have the same final dimension as
    `logits`.
    """
    if logits.shape != legal_mask.shape:
        raise ValueError(
            "logits and legal_mask must have "
            f"matching shapes: "
            f"{logits.shape} != "
            f"{legal_mask.shape}"
        )

    if legal_mask.dtype != torch.bool:
        raise ValueError(
            "legal_mask must be a boolean tensor."
        )

    if not torch.all(
        legal_mask.any(
            dim=-1
        )
    ):
        raise ValueError(
            "Every policy row must contain at least "
            "one legal action."
        )

    return logits.masked_fill(
        ~legal_mask,
        float("-inf"),
    )



class FactorizedCatanActorCritic(nn.Module):
    """
    Actor-critic with an explicitly factorized policy.

    The policy first represents action-type preference,
    then parameter preference within parameterized
    action families.

    The final output is still a flat 202-logit vector,
    so existing RL agents and legal masks remain
    compatible.
    """

    PASS_START = 0
    PASS_COUNT = 1

    SETTLEMENT_START = 1
    SETTLEMENT_COUNT = 54

    CITY_START = 55
    CITY_COUNT = 54

    ROAD_START = 109
    ROAD_COUNT = 72

    DEV_START = 181
    DEV_COUNT = 1

    TRADE_START = 182
    TRADE_COUNT = 20

    TYPE_PASS = 0
    TYPE_SETTLEMENT = 1
    TYPE_CITY = 2
    TYPE_ROAD = 3
    TYPE_DEV = 4
    TYPE_TRADE = 5

    NUM_TYPES = 6

    def __init__(
        self,
        observation_dim: int = 1138,
        action_dim: int = 202,
        hidden_dim: int = 256,
    ):
        super().__init__()

        if action_dim != 202:
            raise ValueError(
                "FactorizedCatanActorCritic currently "
                "requires action_dim=202."
            )

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.backbone = nn.Sequential(
            nn.Linear(
                observation_dim,
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),
        )

        self.type_head = nn.Linear(
            hidden_dim,
            self.NUM_TYPES,
        )

        self.settlement_head = nn.Linear(
            hidden_dim,
            self.SETTLEMENT_COUNT,
        )

        self.city_head = nn.Linear(
            hidden_dim,
            self.CITY_COUNT,
        )

        self.road_head = nn.Linear(
            hidden_dim,
            self.ROAD_COUNT,
        )

        self.trade_head = nn.Linear(
            hidden_dim,
            self.TRADE_COUNT,
        )

        self.value_head = nn.Linear(
            hidden_dim,
            1,
        )

    def policy_components(
        self,
        observation: torch.Tensor,
    ):
        features = self.backbone(
            observation
        )

        type_logits = self.type_head(
            features
        )

        settlement_logits = (
            self.settlement_head(
                features
            )
        )

        city_logits = self.city_head(
            features
        )

        road_logits = self.road_head(
            features
        )

        trade_logits = self.trade_head(
            features
        )

        value = self.value_head(
            features
        ).squeeze(-1)

        return (
            type_logits,
            settlement_logits,
            city_logits,
            road_logits,
            trade_logits,
            value,
        )

    def forward(
        self,
        observation: torch.Tensor,
    ):
        (
            type_logits,
            settlement_logits,
            city_logits,
            road_logits,
            trade_logits,
            value,
        ) = self.policy_components(
            observation
        )

        batch_shape = (
            observation.shape[:-1]
        )

        flat_logits = torch.empty(
            *batch_shape,
            self.action_dim,
            dtype=observation.dtype,
            device=observation.device,
        )

        flat_logits[
            ...,
            self.PASS_START,
        ] = type_logits[
            ...,
            self.TYPE_PASS,
        ]

        flat_logits[
            ...,
            self.SETTLEMENT_START:
            self.SETTLEMENT_START
            + self.SETTLEMENT_COUNT
        ] = (
            type_logits[
                ...,
                self.TYPE_SETTLEMENT,
            ].unsqueeze(-1)
            + settlement_logits
        )

        flat_logits[
            ...,
            self.CITY_START:
            self.CITY_START
            + self.CITY_COUNT
        ] = (
            type_logits[
                ...,
                self.TYPE_CITY,
            ].unsqueeze(-1)
            + city_logits
        )

        flat_logits[
            ...,
            self.ROAD_START:
            self.ROAD_START
            + self.ROAD_COUNT
        ] = (
            type_logits[
                ...,
                self.TYPE_ROAD,
            ].unsqueeze(-1)
            + road_logits
        )

        flat_logits[
            ...,
            self.DEV_START,
        ] = type_logits[
            ...,
            self.TYPE_DEV,
        ]

        flat_logits[
            ...,
            self.TRADE_START:
            self.TRADE_START
            + self.TRADE_COUNT
        ] = (
            type_logits[
                ...,
                self.TYPE_TRADE,
            ].unsqueeze(-1)
            + trade_logits
        )

        return flat_logits, value


class RealismV2ActorCritic(
    FactorizedCatanActorCritic
):
    """
    Factorized ordinary-action actor-critic extended with
    the fixed-size realism-v2 categorical decision heads.

    Ordinary `forward()` behavior is inherited unchanged:
        observation -> (202 ordinary logits, value)

    Phase-specific decisions are queried explicitly through
    `fixed_decision_logits()`.

    Dynamic decision spaces are deliberately not handled
    here. They will be added through candidate scoring in
    R2.6b2.
    """

    MONOPOLY_RESOURCE_DIM = 5
    YEAR_OF_PLENTY_DIM = 15
    TRADE_RESPONSE_DIM = 2

    def __init__(
        self,
        observation_dim: int = 1138,
        action_dim: int = 202,
        hidden_dim: int = 256,
    ):
        super().__init__(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )

        self.monopoly_resource_head = nn.Linear(
            hidden_dim,
            self.MONOPOLY_RESOURCE_DIM,
        )

        self.year_of_plenty_head = nn.Linear(
            hidden_dim,
            self.YEAR_OF_PLENTY_DIM,
        )

        self.trade_response_head = nn.Linear(
            hidden_dim,
            self.TRADE_RESPONSE_DIM,
        )

        # Dynamic realism-v2 decisions have a variable
        # number of candidates. Each decision family first
        # projects its fixed-width candidate representation
        # into the shared hidden width, then scores the
        # observation/candidate pair.
        from catanlab.rl_candidate_features import (
            DYNAMIC_CANDIDATE_FEATURE_DIMS,
        )

        self.dynamic_candidate_projectors = (
            nn.ModuleDict(
                {
                    decision_kind.value: nn.Linear(
                        candidate_dim,
                        hidden_dim,
                    )
                    for (
                        decision_kind,
                        candidate_dim,
                    ) in (
                        DYNAMIC_CANDIDATE_FEATURE_DIMS
                        .items()
                    )
                }
            )
        )

        self.dynamic_candidate_scorers = (
            nn.ModuleDict(
                {
                    decision_kind.value: nn.Sequential(
                        nn.Linear(
                            hidden_dim * 2,
                            hidden_dim,
                        ),
                        nn.ReLU(),
                        nn.Linear(
                            hidden_dim,
                            1,
                        ),
                    )
                    for decision_kind in (
                        DYNAMIC_CANDIDATE_FEATURE_DIMS
                    )
                }
            )
        )

    @staticmethod
    def is_fixed_decision_kind(
        decision_kind,
    ) -> bool:
        """
        Return whether this model currently owns a fixed
        realism-v2 categorical head for `decision_kind`.
        """
        from catanlab.rl_teacher import (
            TeacherDecisionKind,
        )

        return decision_kind in {
            TeacherDecisionKind.MONOPOLY_RESOURCE,
            TeacherDecisionKind.YEAR_OF_PLENTY,
            TeacherDecisionKind.TRADE_RESPONSE,
        }

    def fixed_decision_dim(
        self,
        decision_kind,
    ) -> int:
        """
        Return the fixed output dimension for one supported
        realism-v2 decision kind.
        """
        from catanlab.rl_teacher import (
            TeacherDecisionKind,
        )

        if (
            decision_kind
            == TeacherDecisionKind
            .MONOPOLY_RESOURCE
        ):
            return self.MONOPOLY_RESOURCE_DIM

        if (
            decision_kind
            == TeacherDecisionKind
            .YEAR_OF_PLENTY
        ):
            return self.YEAR_OF_PLENTY_DIM

        if (
            decision_kind
            == TeacherDecisionKind
            .TRADE_RESPONSE
        ):
            return self.TRADE_RESPONSE_DIM

        raise ValueError(
            "Decision kind does not have a fixed "
            "realism-v2 head: "
            f"{decision_kind!r}"
        )

    @staticmethod
    def is_dynamic_decision_kind(
        decision_kind,
    ) -> bool:
        from catanlab.rl_candidate_features import (
            is_dynamic_decision_kind,
        )

        return is_dynamic_decision_kind(
            decision_kind
        )

    def dynamic_candidate_feature_dim(
        self,
        decision_kind,
    ) -> int:
        from catanlab.rl_candidate_features import (
            dynamic_candidate_feature_dim,
        )

        return dynamic_candidate_feature_dim(
            decision_kind
        )

    def dynamic_decision_logits(
        self,
        observation: torch.Tensor,
        decision_kind,
        candidate_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score a variable-size realism-v2 candidate set.

        Supported shapes:

            observation:
                (..., observation_dim)

            candidate_features:
                (K, candidate_dim)
                or
                (..., K, candidate_dim)

        Returns:
                (..., K)

        A single candidate matrix may therefore be scored
        against one observation, while batched observations
        may use one candidate matrix per batch element.
        """
        if not self.is_dynamic_decision_kind(
            decision_kind
        ):
            raise ValueError(
                "Decision kind does not use dynamic "
                "candidate scoring: "
                f"{decision_kind!r}"
            )

        expected_dim = (
            self.dynamic_candidate_feature_dim(
                decision_kind
            )
        )

        if candidate_features.ndim < 2:
            raise ValueError(
                "candidate_features must have at least "
                "two dimensions."
            )

        if (
            candidate_features.shape[-1]
            != expected_dim
        ):
            raise ValueError(
                "Dynamic candidate feature dimension "
                "does not match decision kind: "
                f"kind={decision_kind.value}, "
                f"expected={expected_dim}, "
                f"actual={candidate_features.shape[-1]}"
            )

        if candidate_features.shape[-2] <= 0:
            raise ValueError(
                "Dynamic decision must contain at least "
                "one candidate."
            )

        observation_features = self.backbone(
            observation
        )

        key = decision_kind.value

        candidate_embeddings = (
            self.dynamic_candidate_projectors[
                key
            ](
                candidate_features
            )
        )

        # Unbatched candidate matrix:
        #
        #     observation_features: (..., H)
        #     candidates:           (K, H)
        #
        # This is primarily the inference path used by one
        # LearnedDecisionRequest.
        if candidate_features.ndim == 2:
            candidate_count = (
                candidate_embeddings.shape[-2]
            )

            expanded_observation = (
                observation_features
                .unsqueeze(-2)
                .expand(
                    *observation_features.shape[:-1],
                    candidate_count,
                    observation_features.shape[-1],
                )
            )

            expanded_candidates = (
                candidate_embeddings
            )

            for _ in range(
                observation_features.ndim - 1
            ):
                expanded_candidates = (
                    expanded_candidates.unsqueeze(0)
                )

            expanded_candidates = (
                expanded_candidates.expand(
                    *observation_features.shape[:-1],
                    candidate_count,
                    candidate_embeddings.shape[-1],
                )
            )

        else:
            # Batched candidate sets must have the same
            # leading batch shape as the observations.
            if (
                candidate_features.shape[:-2]
                != observation.shape[:-1]
            ):
                raise ValueError(
                    "Batched candidate features must "
                    "match the observation batch shape: "
                    f"{candidate_features.shape[:-2]} "
                    "!= "
                    f"{observation.shape[:-1]}"
                )

            candidate_count = (
                candidate_embeddings.shape[-2]
            )

            expanded_observation = (
                observation_features
                .unsqueeze(-2)
                .expand(
                    *observation_features.shape[:-1],
                    candidate_count,
                    observation_features.shape[-1],
                )
            )

            expanded_candidates = (
                candidate_embeddings
            )

        joint = torch.cat(
            (
                expanded_observation,
                expanded_candidates,
            ),
            dim=-1,
        )

        logits = (
            self.dynamic_candidate_scorers[
                key
            ](
                joint
            )
            .squeeze(-1)
        )

        return logits

    def fixed_decision_logits(
        self,
        observation: torch.Tensor,
        decision_kind,
    ) -> torch.Tensor:
        """
        Compute logits for one supported fixed-size
        realism-v2 decision family.

        The observation uses the same shared backbone as
        ordinary actions and the value function.
        """
        from catanlab.rl_teacher import (
            TeacherDecisionKind,
        )

        features = self.backbone(
            observation
        )

        if (
            decision_kind
            == TeacherDecisionKind
            .MONOPOLY_RESOURCE
        ):
            return self.monopoly_resource_head(
                features
            )

        if (
            decision_kind
            == TeacherDecisionKind
            .YEAR_OF_PLENTY
        ):
            return self.year_of_plenty_head(
                features
            )

        if (
            decision_kind
            == TeacherDecisionKind
            .TRADE_RESPONSE
        ):
            return self.trade_response_head(
                features
            )

        raise ValueError(
            "Decision kind does not have a fixed "
            "realism-v2 head: "
            f"{decision_kind!r}"
        )
