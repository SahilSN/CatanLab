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
