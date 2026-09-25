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


def test_year_of_plenty_codec_has_fifteen_canonical_pairs():
    from catanlab.rl_special_actions import (
        YEAR_OF_PLENTY_VOCABULARY,
    )

    assert len(
        YEAR_OF_PLENTY_VOCABULARY
    ) == 15

    assert len(
        set(YEAR_OF_PLENTY_VOCABULARY)
    ) == 15

    assert (
        Resource.ORE,
        Resource.ORE,
    ) in YEAR_OF_PLENTY_VOCABULARY

    assert (
        Resource.WHEAT,
        Resource.ORE,
    ) in YEAR_OF_PLENTY_VOCABULARY

    # Reverse ordering is deliberately not a second class.
    assert (
        Resource.ORE,
        Resource.WHEAT,
    ) not in YEAR_OF_PLENTY_VOCABULARY


def test_year_of_plenty_codec_round_trip():
    from catanlab.rl_special_actions import (
        year_of_plenty_decision_input,
    )

    class FullBank:
        def can_supply(
            self,
            resource,
            amount,
        ):
            return True

    decision = year_of_plenty_decision_input(
        FullBank()
    )

    assert decision.action_dim == 15

    assert all(decision.legal_mask)

    for action_id, pair in enumerate(
        decision.vocabulary
    ):
        assert decision.encode(
            pair
        ) == action_id

        assert decision.decode(
            action_id
        ) == pair


def test_year_of_plenty_codec_masks_bank_shortages():
    from catanlab.rl_special_actions import (
        year_of_plenty_decision_input,
    )

    class LimitedBank:
        counts = {
            Resource.WOOD: 1,
            Resource.BRICK: 0,
            Resource.SHEEP: 3,
            Resource.WHEAT: 1,
            Resource.ORE: 1,
        }

        def can_supply(
            self,
            resource,
            amount,
        ):
            return (
                self.counts[
                    resource
                ]
                >= amount
            )

    decision = year_of_plenty_decision_input(
        LimitedBank()
    )

    # One wood exists: WOOD + WOOD is illegal.
    assert not decision.is_legal_value(
        (
            Resource.WOOD,
            Resource.WOOD,
        )
    )

    # One wood and one wheat exist.
    assert decision.is_legal_value(
        (
            Resource.WOOD,
            Resource.WHEAT,
        )
    )

    # Brick has zero supply.
    assert not decision.is_legal_value(
        (
            Resource.BRICK,
            Resource.ORE,
        )
    )

    # Sheep has enough supply for a duplicate pair.
    assert decision.is_legal_value(
        (
            Resource.SHEEP,
            Resource.SHEEP,
        )
    )


def test_road_building_codec_enumerates_sequential_edges():
    from catanlab.board import (
        Board,
        Edge,
        Vertex,
    )
    from catanlab.rl_special_actions import (
        road_building_decision_input,
    )
    from catanlab.simulation import PlayerState

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
                neighbors=[1],
            ),
            Vertex(
                id=1,
                position=(1.0, 0.0),
                neighbors=[0, 2],
            ),
            Vertex(
                id=2,
                position=(2.0, 0.0),
                neighbors=[1],
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
    )

    player = PlayerState(
        player_id=0,
        settlements=[0],
    )

    players = [player]

    decision = (
        road_building_decision_input(
            board,
            players,
            player,
        )
    )

    # The first edge is immediately legal.
    assert (
        ((0, 1),)
        in decision.vocabulary
    )

    # The second edge is not initially connected, but
    # becomes legal after placing (0, 1).
    assert (
        (
            (0, 1),
            (1, 2),
        )
        in decision.vocabulary
    )

    assert all(decision.legal_mask)


def test_road_building_codec_round_trip():
    from catanlab.board import (
        Board,
        Edge,
        Vertex,
    )
    from catanlab.rl_special_actions import (
        road_building_decision_input,
    )
    from catanlab.simulation import PlayerState

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
                neighbors=[1],
            ),
            Vertex(
                id=1,
                position=(1.0, 0.0),
                neighbors=[0, 2],
            ),
            Vertex(
                id=2,
                position=(2.0, 0.0),
                neighbors=[1],
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
    )

    player = PlayerState(
        player_id=0,
        settlements=[0],
    )

    decision = (
        road_building_decision_input(
            board,
            [player],
            player,
        )
    )

    for action_id, sequence in enumerate(
        decision.vocabulary
    ):
        assert (
            decision.encode(sequence)
            == action_id
        )

        assert (
            decision.decode(action_id)
            == sequence
        )


def test_road_building_codec_does_not_mutate_players():
    from catanlab.board import (
        Board,
        Edge,
        Vertex,
    )
    from catanlab.rl_special_actions import (
        road_building_decision_input,
    )
    from catanlab.simulation import PlayerState

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
                neighbors=[1],
            ),
            Vertex(
                id=1,
                position=(1.0, 0.0),
                neighbors=[0, 2],
            ),
            Vertex(
                id=2,
                position=(2.0, 0.0),
                neighbors=[1],
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
    )

    player = PlayerState(
        player_id=0,
        settlements=[0],
    )

    players = [player]

    before = list(player.roads)

    road_building_decision_input(
        board,
        players,
        player,
    )

    assert player.roads == before


def test_discard_codec_enumerates_only_exact_legal_multisets():
    from catanlab.economy import (
        PlayerInventory,
    )
    from catanlab.resources import Resource
    from catanlab.rl_special_actions import (
        discard_decision_input,
    )

    inventory = PlayerInventory()

    inventory.add(
        Resource.WOOD,
        2,
    )
    inventory.add(
        Resource.BRICK,
        1,
    )
    inventory.add(
        Resource.ORE,
        1,
    )

    decision = discard_decision_input(
        inventory,
        2,
    )

    for counts in decision.vocabulary:
        assert sum(counts) == 2

        assert counts[0] <= 2
        assert counts[1] <= 1
        assert counts[2] == 0
        assert counts[3] == 0
        assert counts[4] <= 1

    assert (
        2,
        0,
        0,
        0,
        0,
    ) in decision.vocabulary

    assert (
        1,
        1,
        0,
        0,
        0,
    ) in decision.vocabulary

    assert (
        0,
        1,
        0,
        0,
        1,
    ) in decision.vocabulary


def test_discard_codec_round_trip():
    from catanlab.economy import (
        PlayerInventory,
    )
    from catanlab.resources import Resource
    from catanlab.rl_special_actions import (
        discard_decision_input,
    )

    inventory = PlayerInventory()

    inventory.add(Resource.WOOD, 2)
    inventory.add(Resource.BRICK, 2)
    inventory.add(Resource.WHEAT, 1)

    decision = discard_decision_input(
        inventory,
        2,
    )

    for action_id, counts in enumerate(
        decision.vocabulary
    ):
        assert (
            decision.encode(counts)
            == action_id
        )

        assert (
            decision.decode(action_id)
            == counts
        )

    assert all(decision.legal_mask)


def test_discard_counts_uses_canonical_resource_order():
    from catanlab.resources import Resource
    from catanlab.rl_special_actions import (
        discard_counts,
    )

    discarded = [
        Resource.ORE,
        Resource.WOOD,
        Resource.BRICK,
        Resource.WOOD,
    ]

    assert discard_counts(
        discarded
    ) == (
        2,
        1,
        0,
        0,
        1,
    )


def test_discard_codec_matches_search_city_fixture_choice():
    from catanlab.economy import (
        PlayerInventory,
    )
    from catanlab.resources import Resource
    from catanlab.rl_special_actions import (
        discard_decision_input,
    )

    inventory = PlayerInventory()

    inventory.add(Resource.ORE, 2)
    inventory.add(Resource.WHEAT, 2)
    inventory.add(Resource.WOOD, 2)
    inventory.add(Resource.BRICK, 2)

    decision = discard_decision_input(
        inventory,
        4,
    )

    # Search's validated city-preservation fixture
    # discards the two wood and two brick while keeping
    # all ore and wheat.
    expected = (
        2,
        2,
        0,
        0,
        0,
    )

    assert expected in decision.vocabulary

    action_id = decision.encode(
        expected
    )

    assert decision.decode(
        action_id
    ) == expected

    assert decision.legal_mask[
        action_id
    ]


def test_discard_codec_rejects_impossible_count():
    from catanlab.economy import (
        PlayerInventory,
    )
    from catanlab.resources import Resource
    from catanlab.rl_special_actions import (
        discard_decision_input,
    )

    inventory = PlayerInventory()
    inventory.add(Resource.SHEEP, 2)

    try:
        discard_decision_input(
            inventory,
            3,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected impossible discard count to fail."
        )


def test_trade_proposal_codec_matches_search_one_for_one_space():
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.rl_special_actions import (
        trade_proposal_decision_input,
    )
    from catanlab.simulation import PlayerState
    from catanlab.trading import TradeOffer

    players = [
        PlayerState(player_id=0),
        PlayerState(player_id=1),
        PlayerState(player_id=2),
    ]

    inventory = PlayerInventory()
    inventory.add(Resource.WOOD, 1)

    decision = trade_proposal_decision_input(
        players,
        players[0],
        inventory,
        excluded_recipients={2},
    )

    assert decision.vocabulary[0] is None

    expected = TradeOffer(
        proposer_id=0,
        recipient_id=1,
        give=((Resource.WOOD, 1),),
        receive=((Resource.ORE, 1),),
    )

    assert expected in decision.vocabulary

    # Excluded player 2 must never appear.
    assert all(
        offer is None
        or offer.recipient_id != 2
        for offer in decision.vocabulary
    )

    # Search-v2's proposal head is strictly 1-for-1.
    for offer in decision.vocabulary[1:]:
        assert sum(
            amount
            for _, amount in offer.give
        ) == 1

        assert sum(
            amount
            for _, amount in offer.receive
        ) == 1

    action_id = decision.encode(expected)

    assert decision.decode(
        action_id
    ) == expected


def test_trade_proposal_codec_does_not_peek_at_recipient_hand():
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.rl_special_actions import (
        trade_proposal_decision_input,
    )
    from catanlab.simulation import PlayerState
    from catanlab.trading import TradeOffer

    players = [
        PlayerState(player_id=0),
        PlayerState(player_id=1),
    ]

    own = PlayerInventory()
    own.add(Resource.WOOD, 1)

    # No opponent inventory is supplied at all.
    decision = trade_proposal_decision_input(
        players,
        players[0],
        own,
    )

    request_ore = TradeOffer(
        proposer_id=0,
        recipient_id=1,
        give=((Resource.WOOD, 1),),
        receive=((Resource.ORE, 1),),
    )

    assert request_ore in decision.vocabulary


def test_trade_response_codec_masks_infeasible_accept():
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.rl_special_actions import (
        trade_response_decision_input,
    )
    from catanlab.trading import TradeOffer

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    inventories[0].add(Resource.ORE, 1)

    offer = TradeOffer(
        proposer_id=0,
        recipient_id=1,
        give=((Resource.ORE, 1),),
        receive=((Resource.WOOD, 1),),
    )

    decision = trade_response_decision_input(
        offer,
        inventories,
    )

    assert decision.vocabulary == (
        False,
        True,
    )

    assert decision.legal_mask == (
        True,
        False,
    )

    inventories[1].add(
        Resource.WOOD,
        1,
    )

    feasible = trade_response_decision_input(
        offer,
        inventories,
    )

    assert feasible.legal_mask == (
        True,
        True,
    )


def test_trade_counter_codec_matches_search_one_for_one_space():
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.rl_special_actions import (
        trade_counter_decision_input,
    )
    from catanlab.simulation import PlayerState
    from catanlab.trading import TradeOffer

    players = [
        PlayerState(player_id=0),
        PlayerState(player_id=1),
    ]

    incoming = TradeOffer(
        proposer_id=0,
        recipient_id=1,
        give=((Resource.SHEEP, 1),),
        receive=((Resource.ORE, 1),),
    )

    inventory = PlayerInventory()
    inventory.add(Resource.WOOD, 1)

    decision = trade_counter_decision_input(
        players,
        players[1],
        inventory,
        incoming,
    )

    assert decision.vocabulary[0] is None

    expected = TradeOffer(
        proposer_id=1,
        recipient_id=0,
        give=((Resource.WOOD, 1),),
        receive=((Resource.ORE, 1),),
    )

    assert expected in decision.vocabulary

    action_id = decision.encode(
        expected
    )

    assert decision.decode(
        action_id
    ) == expected


def test_trade_counter_codec_excludes_attempted_offers():
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.rl_special_actions import (
        trade_counter_decision_input,
    )
    from catanlab.simulation import PlayerState
    from catanlab.trading import TradeOffer

    players = [
        PlayerState(player_id=0),
        PlayerState(player_id=1),
    ]

    incoming = TradeOffer(
        proposer_id=0,
        recipient_id=1,
        give=((Resource.SHEEP, 1),),
        receive=((Resource.ORE, 1),),
    )

    attempted = TradeOffer(
        proposer_id=1,
        recipient_id=0,
        give=((Resource.WOOD, 1),),
        receive=((Resource.ORE, 1),),
    )

    inventory = PlayerInventory()
    inventory.add(Resource.WOOD, 1)

    decision = trade_counter_decision_input(
        players,
        players[1],
        inventory,
        incoming,
        attempted_offers={attempted},
    )

    assert attempted not in decision.vocabulary
    assert decision.vocabulary[0] is None
