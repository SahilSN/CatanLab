import pytest

from catanlab.resources import Resource
from catanlab.rl_candidate_features import (
    DYNAMIC_CANDIDATE_FEATURE_DIMS,
    dynamic_candidate_feature_dim,
    encode_dynamic_candidate,
    encode_dynamic_decision_input,
    is_dynamic_decision_kind,
)
from catanlab.rl_special_actions import (
    CategoricalDecisionInput,
)
from catanlab.rl_teacher import (
    TeacherDecisionKind,
)
from catanlab.trading import TradeOffer


def test_dynamic_candidate_kind_registry_is_complete():
    expected = {
        TeacherDecisionKind.ROBBER_TILE,
        TeacherDecisionKind.ROBBER_VICTIM,
        TeacherDecisionKind.DISCARD,
        TeacherDecisionKind.ROAD_BUILDING,
        TeacherDecisionKind.TRADE_PROPOSAL,
        TeacherDecisionKind.TRADE_COUNTER,
    }

    assert set(
        DYNAMIC_CANDIDATE_FEATURE_DIMS
    ) == expected

    assert all(
        is_dynamic_decision_kind(kind)
        for kind in expected
    )


@pytest.mark.parametrize(
    ("decision_kind", "value", "expected_dim"),
    [
        (
            TeacherDecisionKind.ROBBER_TILE,
            30,
            1,
        ),
        (
            TeacherDecisionKind.ROBBER_VICTIM,
            3,
            1,
        ),
        (
            TeacherDecisionKind.DISCARD,
            (2, 1, 0, 0, 1),
            5,
        ),
        (
            TeacherDecisionKind.ROAD_BUILDING,
            ((0, 1),),
            5,
        ),
        (
            TeacherDecisionKind.ROAD_BUILDING,
            (
                (0, 1),
                (1, 2),
            ),
            5,
        ),
        (
            TeacherDecisionKind.TRADE_PROPOSAL,
            None,
            13,
        ),
        (
            TeacherDecisionKind.TRADE_COUNTER,
            None,
            13,
        ),
    ],
)
def test_dynamic_candidate_dimensions(
    decision_kind,
    value,
    expected_dim,
):
    features = encode_dynamic_candidate(
        decision_kind,
        value,
    )

    assert len(features) == expected_dim

    assert (
        dynamic_candidate_feature_dim(
            decision_kind
        )
        == expected_dim
    )


def test_discard_candidate_preserves_canonical_counts():
    features = encode_dynamic_candidate(
        TeacherDecisionKind.DISCARD,
        (
            2,
            1,
            0,
            3,
            1,
        ),
    )

    assert features == (
        2.0,
        1.0,
        0.0,
        3.0,
        1.0,
    )


def test_road_building_candidate_marks_second_edge():
    single = encode_dynamic_candidate(
        TeacherDecisionKind.ROAD_BUILDING,
        (
            (0, 1),
        ),
    )

    double = encode_dynamic_candidate(
        TeacherDecisionKind.ROAD_BUILDING,
        (
            (0, 1),
            (1, 2),
        ),
    )

    assert single[-1] == 0.0
    assert double[-1] == 1.0

    assert single != double


def test_trade_candidate_encodes_public_offer_only():
    offer = TradeOffer(
        proposer_id=0,
        recipient_id=2,
        give=((Resource.WOOD, 1),),
        receive=((Resource.ORE, 1),),
    )

    features = encode_dynamic_candidate(
        TeacherDecisionKind.TRADE_PROPOSAL,
        offer,
    )

    assert len(features) == 13

    # Not the no-trade candidate.
    assert features[0] == 0.0

    # Give-resource one-hot.
    assert features[3:8] == (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    # Receive-resource one-hot.
    assert features[8:13] == (
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def test_none_trade_candidate_has_distinct_marker():
    features = encode_dynamic_candidate(
        TeacherDecisionKind.TRADE_COUNTER,
        None,
    )

    assert features == (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def test_dynamic_trade_encoder_rejects_non_search_v2_bundle():
    offer = TradeOffer(
        proposer_id=0,
        recipient_id=1,
        give=(
            (Resource.WOOD, 1),
            (Resource.BRICK, 1),
        ),
        receive=((Resource.ORE, 1),),
    )

    with pytest.raises(
        ValueError,
        match="one give resource",
    ):
        encode_dynamic_candidate(
            TeacherDecisionKind.TRADE_PROPOSAL,
            offer,
        )


def test_dynamic_decision_input_preserves_candidate_order():
    decision_input = CategoricalDecisionInput(
        vocabulary=(
            (2, 0, 0, 0, 0),
            (1, 1, 0, 0, 0),
            (0, 1, 0, 0, 1),
        ),
        legal_mask=(
            True,
            True,
            True,
        ),
    )

    encoded = encode_dynamic_decision_input(
        TeacherDecisionKind.DISCARD,
        decision_input,
    )

    assert encoded == (
        (
            2.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
        ),
    )


@pytest.mark.parametrize(
    "decision_kind",
    [
        TeacherDecisionKind.ORDINARY_ACTION,
        TeacherDecisionKind.MONOPOLY_RESOURCE,
        TeacherDecisionKind.YEAR_OF_PLENTY,
        TeacherDecisionKind.TRADE_RESPONSE,
    ],
)
def test_dynamic_encoder_rejects_non_dynamic_kind(
    decision_kind,
):
    assert not is_dynamic_decision_kind(
        decision_kind
    )

    with pytest.raises(
        ValueError,
        match="does not use dynamic",
    ):
        encode_dynamic_candidate(
            decision_kind,
            None,
        )
