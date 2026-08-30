import pytest

from catanlab.rl_decision_policy import (
    LearnedDecisionRequest,
    choose_decision_value,
    validate_decision_action_id,
)
from catanlab.rl_special_actions import (
    CategoricalDecisionInput,
)
from catanlab.rl_teacher import (
    TeacherDecisionKind,
)


def make_request():
    return LearnedDecisionRequest(
        decision_kind=(
            TeacherDecisionKind
            .MONOPOLY_RESOURCE
        ),
        observation=(
            0.0,
            1.0,
            2.0,
        ),
        decision_input=(
            CategoricalDecisionInput(
                vocabulary=(
                    "wood",
                    "brick",
                    "ore",
                ),
                legal_mask=(
                    True,
                    False,
                    True,
                ),
            )
        ),
    )


def test_learned_decision_request_exposes_action_space():
    request = make_request()

    assert request.action_dim == 3

    assert request.legal_mask == (
        True,
        False,
        True,
    )

    assert request.legal_action_ids == (
        0,
        2,
    )


def test_learned_decision_request_decodes_legal_action():
    request = make_request()

    assert request.decode(2) == "ore"


def test_learned_decision_request_rejects_illegal_action():
    request = make_request()

    with pytest.raises(
        ValueError,
        match="illegal",
    ):
        request.decode(1)


def test_learned_decision_request_rejects_out_of_range_action():
    request = make_request()

    with pytest.raises(
        ValueError,
        match="Invalid learned decision action ID",
    ):
        request.decode(3)


def test_learned_decision_request_requires_legal_choice():
    with pytest.raises(
        ValueError,
        match="at least one legal",
    ):
        LearnedDecisionRequest(
            decision_kind=(
                TeacherDecisionKind
                .TRADE_RESPONSE
            ),
            observation=(1.0,),
            decision_input=(
                CategoricalDecisionInput(
                    vocabulary=(
                        False,
                        True,
                    ),
                    legal_mask=(
                        False,
                        False,
                    ),
                )
            ),
        )


def test_validate_decision_action_id_returns_legal_id():
    request = make_request()

    assert (
        validate_decision_action_id(
            request,
            2,
        )
        == 2
    )


def test_validate_decision_action_id_rejects_masked_id():
    request = make_request()

    with pytest.raises(
        ValueError,
        match="illegal",
    ):
        validate_decision_action_id(
            request,
            1,
        )


def test_choose_decision_value_decodes_policy_choice():
    request = make_request()

    class Policy:
        def choose_decision(
            self,
            request,
        ):
            assert (
                request.decision_kind
                == TeacherDecisionKind
                .MONOPOLY_RESOURCE
            )

            return 2

    assert (
        choose_decision_value(
            Policy(),
            request,
        )
        == "ore"
    )


def test_choose_decision_value_rejects_buggy_policy():
    request = make_request()

    class Policy:
        def choose_decision(
            self,
            request,
        ):
            return 1

    with pytest.raises(
        ValueError,
        match="illegal",
    ):
        choose_decision_value(
            Policy(),
            request,
        )


def test_all_realism_v2_categorical_kinds_fit_request_contract():
    from catanlab.board import Board, Edge, Tile, Vertex
    from catanlab.economy import (
        PlayerInventory,
        ResourceBank,
    )
    from catanlab.graph import HexCoord
    from catanlab.resources import Resource
    from catanlab.rl_decision_policy import (
        LearnedDecisionRequest,
    )
    from catanlab.rl_special_actions import (
        discard_decision_input,
        monopoly_resource_decision_input,
        road_building_decision_input,
        robber_tile_decision_input,
        robber_victim_decision_input,
        trade_counter_decision_input,
        trade_proposal_decision_input,
        trade_response_decision_input,
        year_of_plenty_decision_input,
    )
    from catanlab.rl_teacher import (
        TeacherDecisionKind,
    )
    from catanlab.simulation import PlayerState
    from catanlab.trading import TradeOffer

    observation = (
        0.0,
        1.0,
        2.0,
    )

    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=HexCoord(0, 0),
                resource=Resource.WOOD,
                number=5,
            ),
            Tile(
                id=1,
                coord=HexCoord(1, 0),
                resource=Resource.BRICK,
                number=6,
            ),
        ],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
                adjacent_tiles=[0],
            ),
            Vertex(
                id=1,
                position=(1.0, 0.0),
                adjacent_tiles=[0, 1],
            ),
            Vertex(
                id=2,
                position=(2.0, 0.0),
                adjacent_tiles=[1],
            ),
            Vertex(
                id=3,
                position=(0.0, 1.0),
                adjacent_tiles=[0],
            ),
        ],
        edges=[
            Edge(
                vertex_a=0,
                vertex_b=1,
            ),
            Edge(
                vertex_a=1,
                vertex_b=2,
            ),
        ],
        robber_tile_id=0,
    )

    players = [
        PlayerState(
            player_id=0,
            roads=[(0, 1)],
        ),
        PlayerState(
            player_id=1,
            settlements=[3],
        ),
    ]

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    inventories[0].add(
        Resource.WOOD,
        2,
    )
    inventories[0].add(
        Resource.BRICK,
        2,
    )
    inventories[0].add(
        Resource.ORE,
        1,
    )

    inventories[1].add(
        Resource.SHEEP,
        1,
    )

    bank = ResourceBank()

    incoming = TradeOffer(
        proposer_id=1,
        recipient_id=0,
        give=((Resource.ORE, 1),),
        receive=((Resource.WOOD, 1),),
    )

    decision_inputs = {
        TeacherDecisionKind.ROBBER_TILE: (
            robber_tile_decision_input(
                board
            )
        ),
        TeacherDecisionKind.ROBBER_VICTIM: (
            robber_victim_decision_input(
                board,
                players,
                inventories,
                players[0],
            )
        ),
        TeacherDecisionKind.DISCARD: (
            discard_decision_input(
                inventories[0],
                2,
            )
        ),
        TeacherDecisionKind.MONOPOLY_RESOURCE: (
            monopoly_resource_decision_input()
        ),
        TeacherDecisionKind.YEAR_OF_PLENTY: (
            year_of_plenty_decision_input(
                bank
            )
        ),
        TeacherDecisionKind.ROAD_BUILDING: (
            road_building_decision_input(
                board,
                players,
                players[0],
            )
        ),
        TeacherDecisionKind.TRADE_PROPOSAL: (
            trade_proposal_decision_input(
                players,
                players[0],
                inventories[0],
            )
        ),
        TeacherDecisionKind.TRADE_RESPONSE: (
            trade_response_decision_input(
                incoming,
                inventories,
            )
        ),
        TeacherDecisionKind.TRADE_COUNTER: (
            trade_counter_decision_input(
                players,
                players[0],
                inventories[0],
                incoming,
            )
        ),
    }

    expected = {
        TeacherDecisionKind.ROBBER_TILE,
        TeacherDecisionKind.ROBBER_VICTIM,
        TeacherDecisionKind.DISCARD,
        TeacherDecisionKind.MONOPOLY_RESOURCE,
        TeacherDecisionKind.YEAR_OF_PLENTY,
        TeacherDecisionKind.ROAD_BUILDING,
        TeacherDecisionKind.TRADE_PROPOSAL,
        TeacherDecisionKind.TRADE_RESPONSE,
        TeacherDecisionKind.TRADE_COUNTER,
    }

    assert set(
        decision_inputs
    ) == expected

    for decision_kind, decision_input in (
        decision_inputs.items()
    ):
        request = LearnedDecisionRequest(
            decision_kind=decision_kind,
            observation=observation,
            decision_input=decision_input,
        )

        assert request.action_dim > 0
        assert request.legal_action_ids

        for action_id in (
            request.legal_action_ids
        ):
            value = request.decode(
                action_id
            )

            assert (
                decision_input.encode(
                    value
                )
                == action_id
            )
