import pytest
import torch

from catanlab.rl_model import (
    RealismV2ActorCritic,
)
from catanlab.rl_teacher import (
    TeacherDecisionKind,
)


def test_realism_v2_model_preserves_ordinary_forward_contract():
    model = RealismV2ActorCritic()

    observation = torch.zeros(
        4,
        1138,
    )

    logits, value = model(
        observation
    )

    assert logits.shape == (
        4,
        202,
    )

    assert value.shape == (
        4,
    )


@pytest.mark.parametrize(
    (
        "decision_kind",
        "expected_dim",
    ),
    [
        (
            TeacherDecisionKind
            .MONOPOLY_RESOURCE,
            5,
        ),
        (
            TeacherDecisionKind
            .YEAR_OF_PLENTY,
            15,
        ),
        (
            TeacherDecisionKind
            .TRADE_RESPONSE,
            2,
        ),
    ],
)
def test_realism_v2_fixed_head_dimensions(
    decision_kind,
    expected_dim,
):
    model = RealismV2ActorCritic()

    observation = torch.zeros(
        3,
        1138,
    )

    logits = model.fixed_decision_logits(
        observation,
        decision_kind,
    )

    assert logits.shape == (
        3,
        expected_dim,
    )

    assert (
        model.fixed_decision_dim(
            decision_kind
        )
        == expected_dim
    )


@pytest.mark.parametrize(
    "decision_kind",
    [
        TeacherDecisionKind
        .MONOPOLY_RESOURCE,
        TeacherDecisionKind
        .YEAR_OF_PLENTY,
        TeacherDecisionKind
        .TRADE_RESPONSE,
    ],
)
def test_realism_v2_model_reports_fixed_kinds(
    decision_kind,
):
    assert (
        RealismV2ActorCritic
        .is_fixed_decision_kind(
            decision_kind
        )
    )


@pytest.mark.parametrize(
    "decision_kind",
    [
        TeacherDecisionKind.ROBBER_TILE,
        TeacherDecisionKind.ROBBER_VICTIM,
        TeacherDecisionKind.DISCARD,
        TeacherDecisionKind.ROAD_BUILDING,
        TeacherDecisionKind.TRADE_PROPOSAL,
        TeacherDecisionKind.TRADE_COUNTER,
    ],
)
def test_realism_v2_model_rejects_dynamic_kind_from_fixed_router(
    decision_kind,
):
    model = RealismV2ActorCritic()

    assert not model.is_fixed_decision_kind(
        decision_kind
    )

    observation = torch.zeros(
        1,
        1138,
    )

    with pytest.raises(
        ValueError,
        match="does not have a fixed",
    ):
        model.fixed_decision_logits(
            observation,
            decision_kind,
        )


def test_realism_v2_fixed_heads_share_observation_backbone():
    model = RealismV2ActorCritic(
        hidden_dim=32,
    )

    assert (
        model.monopoly_resource_head.in_features
        == 32
    )

    assert (
        model.year_of_plenty_head.in_features
        == 32
    )

    assert (
        model.trade_response_head.in_features
        == 32
    )


def test_realism_v2_model_works_with_existing_neural_agent():
    from catanlab.rl_agent import (
        NeuralPolicyAgent,
    )
    from catanlab.strategies import (
        StrategyType,
    )

    model = RealismV2ActorCritic()

    agent = NeuralPolicyAgent(
        StrategyType.FIVE_RESOURCE,
        model=model,
        deterministic=True,
        seed=1,
    )

    # Existing agent construction requires only the
    # ordinary forward contract.
    assert agent.model is model


def test_realism_v2_fixed_head_backprop_reaches_shared_backbone():
    model = RealismV2ActorCritic(
        hidden_dim=32,
    )

    observation = torch.randn(
        2,
        1138,
    )

    logits = model.fixed_decision_logits(
        observation,
        TeacherDecisionKind
        .MONOPOLY_RESOURCE,
    )

    loss = logits.sum()
    loss.backward()

    first_linear = model.backbone[0]

    assert first_linear.weight.grad is not None

    assert torch.any(
        first_linear.weight.grad != 0
    )
