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


@pytest.mark.parametrize(
    (
        "decision_kind",
        "candidate_dim",
    ),
    [
        (
            TeacherDecisionKind.ROBBER_TILE,
            1,
        ),
        (
            TeacherDecisionKind.ROBBER_VICTIM,
            1,
        ),
        (
            TeacherDecisionKind.DISCARD,
            5,
        ),
        (
            TeacherDecisionKind.ROAD_BUILDING,
            5,
        ),
        (
            TeacherDecisionKind.TRADE_PROPOSAL,
            13,
        ),
        (
            TeacherDecisionKind.TRADE_COUNTER,
            13,
        ),
    ],
)
def test_realism_v2_dynamic_candidate_dimensions(
    decision_kind,
    candidate_dim,
):
    model = RealismV2ActorCritic(
        hidden_dim=32,
    )

    assert model.is_dynamic_decision_kind(
        decision_kind
    )

    assert (
        model.dynamic_candidate_feature_dim(
            decision_kind
        )
        == candidate_dim
    )


@pytest.mark.parametrize(
    (
        "decision_kind",
        "candidate_dim",
        "candidate_count",
    ),
    [
        (
            TeacherDecisionKind.ROBBER_TILE,
            1,
            19,
        ),
        (
            TeacherDecisionKind.ROBBER_VICTIM,
            1,
            4,
        ),
        (
            TeacherDecisionKind.DISCARD,
            5,
            7,
        ),
        (
            TeacherDecisionKind.ROAD_BUILDING,
            5,
            11,
        ),
        (
            TeacherDecisionKind.TRADE_PROPOSAL,
            13,
            9,
        ),
        (
            TeacherDecisionKind.TRADE_COUNTER,
            13,
            5,
        ),
    ],
)
def test_realism_v2_dynamic_logits_match_candidate_count(
    decision_kind,
    candidate_dim,
    candidate_count,
):
    model = RealismV2ActorCritic(
        hidden_dim=32,
    )

    observation = torch.zeros(
        3,
        1138,
    )

    candidates = torch.zeros(
        candidate_count,
        candidate_dim,
    )

    logits = model.dynamic_decision_logits(
        observation,
        decision_kind,
        candidates,
    )

    assert logits.shape == (
        3,
        candidate_count,
    )


def test_realism_v2_dynamic_logits_support_batched_candidates():
    model = RealismV2ActorCritic(
        hidden_dim=32,
    )

    observation = torch.zeros(
        4,
        1138,
    )

    candidates = torch.zeros(
        4,
        6,
        5,
    )

    logits = model.dynamic_decision_logits(
        observation,
        TeacherDecisionKind.DISCARD,
        candidates,
    )

    assert logits.shape == (
        4,
        6,
    )


def test_realism_v2_dynamic_logits_depend_on_candidate_features():
    model = RealismV2ActorCritic(
        hidden_dim=32,
    )

    observation = torch.zeros(
        1,
        1138,
    )

    candidates = torch.tensor(
        [
            [
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                2.0,
            ],
        ]
    )

    logits = model.dynamic_decision_logits(
        observation,
        TeacherDecisionKind.DISCARD,
        candidates,
    )

    assert logits.shape == (
        1,
        2,
    )

    # The candidates must enter the computation graph
    # independently rather than being collapsed before
    # scoring.
    loss = (
        logits[0, 0]
        - logits[0, 1]
    )

    loss.backward()

    projector = (
        model.dynamic_candidate_projectors[
            TeacherDecisionKind
            .DISCARD
            .value
        ]
    )

    assert projector.weight.grad is not None


def test_realism_v2_dynamic_head_backprop_reaches_shared_backbone():
    model = RealismV2ActorCritic(
        hidden_dim=32,
    )

    observation = torch.randn(
        2,
        1138,
    )

    candidates = torch.randn(
        5,
        5,
    )

    logits = model.dynamic_decision_logits(
        observation,
        TeacherDecisionKind.DISCARD,
        candidates,
    )

    logits.sum().backward()

    first_linear = model.backbone[0]

    assert first_linear.weight.grad is not None

    assert torch.any(
        first_linear.weight.grad != 0
    )


def test_realism_v2_dynamic_head_rejects_wrong_feature_width():
    model = RealismV2ActorCritic()

    observation = torch.zeros(
        1,
        1138,
    )

    candidates = torch.zeros(
        3,
        4,
    )

    with pytest.raises(
        ValueError,
        match="feature dimension",
    ):
        model.dynamic_decision_logits(
            observation,
            TeacherDecisionKind.DISCARD,
            candidates,
        )


def test_realism_v2_dynamic_head_rejects_fixed_kind():
    model = RealismV2ActorCritic()

    observation = torch.zeros(
        1,
        1138,
    )

    candidates = torch.zeros(
        5,
        5,
    )

    with pytest.raises(
        ValueError,
        match="does not use dynamic",
    ):
        model.dynamic_decision_logits(
            observation,
            TeacherDecisionKind
            .MONOPOLY_RESOURCE,
            candidates,
        )


def test_realism_v2_dynamic_head_rejects_mismatched_batch_shape():
    model = RealismV2ActorCritic()

    observation = torch.zeros(
        2,
        1138,
    )

    candidates = torch.zeros(
        3,
        5,
        5,
    )

    with pytest.raises(
        ValueError,
        match="batch shape",
    ):
        model.dynamic_decision_logits(
            observation,
            TeacherDecisionKind.DISCARD,
            candidates,
        )
