import pytest

from catanlab.economy import (
    PlayerInventory,
)
from catanlab.resources import Resource
from catanlab.trading import (
    TradeOffer,
    execute_player_trade,
    make_bundle,
    validate_trade_offer,
)


def make_inventories():
    return [
        PlayerInventory(),
        PlayerInventory(),
    ]


def test_valid_one_for_one_trade():
    inventories = make_inventories()

    inventories[0].add(
        Resource.WOOD
    )

    inventories[1].add(
        Resource.ORE
    )

    offer = TradeOffer(
        proposer_id=0,
        recipient_id=1,
        give=make_bundle(
            (Resource.WOOD, 1),
        ),
        receive=make_bundle(
            (Resource.ORE, 1),
        ),
    )

    assert validate_trade_offer(
        offer,
        inventories,
    )

    execute_player_trade(
        offer,
        inventories,
    )

    assert inventories[0].count(
        Resource.WOOD
    ) == 0

    assert inventories[0].count(
        Resource.ORE
    ) == 1

    assert inventories[1].count(
        Resource.WOOD
    ) == 1

    assert inventories[1].count(
        Resource.ORE
    ) == 0


def test_valid_two_for_one_trade():
    inventories = make_inventories()

    inventories[0].add(
        Resource.WOOD,
        2,
    )

    inventories[1].add(
        Resource.WHEAT,
        1,
    )

    offer = TradeOffer(
        proposer_id=0,
        recipient_id=1,
        give=make_bundle(
            (Resource.WOOD, 2),
        ),
        receive=make_bundle(
            (Resource.WHEAT, 1),
        ),
    )

    assert validate_trade_offer(
        offer,
        inventories,
    )


def test_trade_supports_mixed_bundle_for_single_card():
    inventories = make_inventories()

    inventories[0].add(
        Resource.WOOD,
    )
    inventories[0].add(
        Resource.SHEEP,
    )

    inventories[1].add(
        Resource.ORE,
    )

    offer = TradeOffer(
        proposer_id=0,
        recipient_id=1,
        give=make_bundle(
            (Resource.WOOD, 1),
            (Resource.SHEEP, 1),
        ),
        receive=make_bundle(
            (Resource.ORE, 1),
        ),
    )

    assert validate_trade_offer(
        offer,
        inventories,
    )

    execute_player_trade(
        offer,
        inventories,
    )

    assert inventories[0].count(
        Resource.WOOD
    ) == 0

    assert inventories[0].count(
        Resource.SHEEP
    ) == 0

    assert inventories[0].count(
        Resource.ORE
    ) == 1

    assert inventories[1].count(
        Resource.WOOD
    ) == 1

    assert inventories[1].count(
        Resource.SHEEP
    ) == 1


def test_trade_supports_mixed_bundles_on_both_sides():
    inventories = make_inventories()

    inventories[0].add(
        Resource.WOOD,
    )
    inventories[0].add(
        Resource.BRICK,
    )

    inventories[1].add(
        Resource.ORE,
    )
    inventories[1].add(
        Resource.WHEAT,
    )

    offer = TradeOffer(
        proposer_id=0,
        recipient_id=1,
        give=make_bundle(
            (Resource.WOOD, 1),
            (Resource.BRICK, 1),
        ),
        receive=make_bundle(
            (Resource.ORE, 1),
            (Resource.WHEAT, 1),
        ),
    )

    assert validate_trade_offer(
        offer,
        inventories,
    )

    execute_player_trade(
        offer,
        inventories,
    )

    assert inventories[0].count(
        Resource.ORE
    ) == 1

    assert inventories[0].count(
        Resource.WHEAT
    ) == 1

    assert inventories[1].count(
        Resource.WOOD
    ) == 1

    assert inventories[1].count(
        Resource.BRICK
    ) == 1


def test_trade_rejects_missing_proposer_cards():
    inventories = make_inventories()

    inventories[1].add(
        Resource.ORE,
    )

    offer = TradeOffer(
        proposer_id=0,
        recipient_id=1,
        give=make_bundle(
            (Resource.WOOD, 1),
        ),
        receive=make_bundle(
            (Resource.ORE, 1),
        ),
    )

    assert not validate_trade_offer(
        offer,
        inventories,
    )


def test_trade_rejects_missing_recipient_cards():
    inventories = make_inventories()

    inventories[0].add(
        Resource.WOOD,
    )

    offer = TradeOffer(
        proposer_id=0,
        recipient_id=1,
        give=make_bundle(
            (Resource.WOOD, 1),
        ),
        receive=make_bundle(
            (Resource.ORE, 1),
        ),
    )

    assert not validate_trade_offer(
        offer,
        inventories,
    )


def test_trade_rejects_self_trade():
    inventories = make_inventories()

    inventories[0].add(
        Resource.WOOD,
    )
    inventories[0].add(
        Resource.ORE,
    )

    offer = TradeOffer(
        proposer_id=0,
        recipient_id=0,
        give=make_bundle(
            (Resource.WOOD, 1),
        ),
        receive=make_bundle(
            (Resource.ORE, 1),
        ),
    )

    assert not validate_trade_offer(
        offer,
        inventories,
    )


def test_bundle_rejects_zero_amount():
    with pytest.raises(
        ValueError
    ):
        make_bundle(
            (Resource.WOOD, 0),
        )


def test_illegal_trade_is_atomic():
    inventories = make_inventories()

    inventories[0].add(
        Resource.WOOD,
    )

    offer = TradeOffer(
        proposer_id=0,
        recipient_id=1,
        give=make_bundle(
            (Resource.WOOD, 1),
        ),
        receive=make_bundle(
            (Resource.ORE, 1),
        ),
    )

    with pytest.raises(
        ValueError
    ):
        execute_player_trade(
            offer,
            inventories,
        )

    assert inventories[0].count(
        Resource.WOOD
    ) == 1

    assert inventories[1].count(
        Resource.WOOD
    ) == 0


def test_generate_trade_bundles_supports_mixed_resources():
    from catanlab.trading import (
        generate_trade_bundles,
    )

    inventory = PlayerInventory()

    inventory.add(
        Resource.WOOD,
        1,
    )
    inventory.add(
        Resource.SHEEP,
        1,
    )
    inventory.add(
        Resource.BRICK,
        1,
    )

    bundles = generate_trade_bundles(
        inventory
    )

    assert make_bundle(
        (Resource.WOOD, 1),
        (Resource.SHEEP, 1),
    ) in bundles

    assert make_bundle(
        (Resource.WOOD, 1),
        (Resource.SHEEP, 1),
        (Resource.BRICK, 1),
    ) in bundles


def test_trade_bundle_generation_respects_four_card_cap():
    from catanlab.trading import (
        bundle_size,
        generate_trade_bundles,
    )

    inventory = PlayerInventory()

    for resource in (
        Resource.WOOD,
        Resource.BRICK,
        Resource.SHEEP,
        Resource.WHEAT,
        Resource.ORE,
    ):
        inventory.add(
            resource,
            5,
        )

    bundles = generate_trade_bundles(
        inventory,
        max_cards=4,
        max_types=3,
    )

    assert bundles

    assert all(
        bundle_size(bundle)
        <= 4
        for bundle in bundles
    )

    assert all(
        len(bundle) <= 3
        for bundle in bundles
    )


def test_trade_rejects_same_resource_on_both_sides():
    inventories = make_inventories()

    inventories[0].add(
        Resource.WOOD,
        2,
    )

    inventories[1].add(
        Resource.WOOD,
        2,
    )
    inventories[1].add(
        Resource.ORE,
        1,
    )

    offer = TradeOffer(
        proposer_id=0,
        recipient_id=1,
        give=make_bundle(
            (Resource.WOOD, 1),
        ),
        receive=make_bundle(
            (Resource.WOOD, 1),
            (Resource.ORE, 1),
        ),
    )

    assert not validate_trade_offer(
        offer,
        inventories,
    )


def test_adaptive_initial_trade_can_use_mixed_bundle():
    from catanlab.board import (
        Board,
        Edge,
        Vertex,
    )
    from catanlab.simulation import (
        PlayerState,
    )
    from catanlab.strategies import (
        StrategyType,
    )
    from catanlab.turns import (
        AdaptiveStrategyAgent,
    )

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
                neighbors=[0],
            ),
        ],
        edges=[
            Edge(
                vertex_a=0,
                vertex_b=1,
            ),
        ],
    )

    players = [
        PlayerState(
            player_id=0,
            settlements=[0],
        ),
        PlayerState(
            player_id=1,
        ),
    ]

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    # Give the active player several expendable
    # resources while leaving it short of resources
    # useful to its strategy.
    inventories[0].add(
        Resource.SHEEP,
        3,
    )
    inventories[0].add(
        Resource.WOOD,
        1,
    )

    inventories[1].add(
        Resource.BRICK,
        2,
    )
    inventories[1].add(
        Resource.ORE,
        2,
    )

    agent = AdaptiveStrategyAgent(
        StrategyType.ROAD_BUILDING
    )

    offer = agent.propose_player_trade(
        board,
        players,
        players[0],
        inventories,
    )

    assert offer is not None

    assert (
        offer.proposer_id
        == 0
    )

    assert (
        offer.recipient_id
        == 1
    )

    assert len(
        offer.give
    ) >= 1

    assert len(
        offer.receive
    ) >= 1
