from catanlab.board import (
    Board,
    Tile,
    Vertex,
)
from catanlab.economy import PlayerInventory
from catanlab.resources import Resource
from catanlab.rl_special_actions import (
    CategoricalDecisionInput,
    monopoly_resource_decision_input,
    robber_tile_decision_input,
    robber_victim_decision_input,
)
from catanlab.simulation import PlayerState


def test_categorical_decision_round_trip():
    decision = CategoricalDecisionInput(
        vocabulary=(
            "a",
            "b",
            "c",
        ),
        legal_mask=(
            True,
            False,
            True,
        ),
    )

    for action_id, value in enumerate(
        decision.vocabulary
    ):
        assert decision.encode(
            value
        ) == action_id

        assert decision.decode(
            action_id
        ) == value

    assert decision.action_dim == 3

    assert decision.legal_action_ids == (
        0,
        2,
    )


def test_categorical_decision_rejects_bad_mask_length():
    import pytest

    with pytest.raises(ValueError):
        CategoricalDecisionInput(
            vocabulary=(
                "a",
                "b",
            ),
            legal_mask=(
                True,
            ),
        )


def test_categorical_decision_rejects_unknown_value():
    import pytest

    decision = CategoricalDecisionInput(
        vocabulary=("a",),
        legal_mask=(True,),
    )

    with pytest.raises(ValueError):
        decision.encode("missing")


def test_categorical_decision_rejects_invalid_id():
    import pytest

    decision = CategoricalDecisionInput(
        vocabulary=("a",),
        legal_mask=(True,),
    )

    with pytest.raises(ValueError):
        decision.decode(-1)

    with pytest.raises(ValueError):
        decision.decode(1)


def test_robber_tile_codec_masks_current_tile():
    board = Board(
        tiles=[
            Tile(
                id=10,
                coord=(0, 0),
                resource=Resource.WOOD,
                number=6,
            ),
            Tile(
                id=20,
                coord=(1, 0),
                resource=Resource.BRICK,
                number=8,
            ),
            Tile(
                id=30,
                coord=(2, 0),
                resource=Resource.SHEEP,
                number=5,
            ),
        ],
        vertices=[],
        edges=[],
        robber_tile_id=20,
    )

    decision = robber_tile_decision_input(
        board
    )

    assert decision.vocabulary == (
        10,
        20,
        30,
    )

    assert decision.legal_mask == (
        True,
        False,
        True,
    )

    assert decision.decode(
        decision.encode(30)
    ) == 30


def test_robber_victim_codec_masks_by_adjacency_and_hand_size():
    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=(0, 0),
                resource=Resource.WOOD,
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
                adjacent_tiles=[0],
            ),
            Vertex(
                id=2,
                position=(2.0, 0.0),
                adjacent_tiles=[0],
            ),
            Vertex(
                id=3,
                position=(3.0, 0.0),
                adjacent_tiles=[],
            ),
        ],
        edges=[],
        robber_tile_id=0,
    )

    players = [
        PlayerState(
            player_id=0,
            settlements=[0],
        ),
        PlayerState(
            player_id=1,
            settlements=[1],
        ),
        PlayerState(
            player_id=2,
            settlements=[2],
        ),
        PlayerState(
            player_id=3,
            settlements=[3],
        ),
    ]

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
        PlayerInventory(),
        PlayerInventory(),
    ]

    # Adjacent player 1 has a card and is eligible.
    inventories[1].add(
        Resource.WOOD,
        1,
    )

    # Adjacent player 2 has an empty public hand and is
    # therefore ineligible.
    #
    # Player 3 has cards but is not adjacent.
    inventories[3].add(
        Resource.ORE,
        2,
    )

    decision = (
        robber_victim_decision_input(
            board,
            players,
            inventories,
            players[0],
        )
    )

    assert decision.vocabulary == (
        0,
        1,
        2,
        3,
    )

    assert decision.legal_mask == (
        False,
        True,
        False,
        False,
    )

    assert decision.encode(1) == 1
    assert decision.decode(1) == 1


def test_robber_victim_codec_has_no_legal_choice_without_robber():
    board = Board(
        tiles=[],
        vertices=[],
        edges=[],
        robber_tile_id=None,
    )

    players = [
        PlayerState(player_id=0),
        PlayerState(player_id=1),
    ]

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    inventories[1].add(
        Resource.WOOD,
        1,
    )

    decision = (
        robber_victim_decision_input(
            board,
            players,
            inventories,
            players[0],
        )
    )

    assert decision.legal_mask == (
        False,
        False,
    )


def test_monopoly_resource_codec_round_trip():
    decision = (
        monopoly_resource_decision_input()
    )

    expected = (
        Resource.WOOD,
        Resource.BRICK,
        Resource.SHEEP,
        Resource.WHEAT,
        Resource.ORE,
    )

    assert decision.vocabulary == expected

    assert decision.legal_mask == (
        True,
        True,
        True,
        True,
        True,
    )

    for resource in expected:
        action_id = decision.encode(
            resource
        )

        assert decision.decode(
            action_id
        ) == resource
