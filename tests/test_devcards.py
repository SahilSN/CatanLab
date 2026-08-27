from collections import Counter

from catanlab.devcards import (
    DevCardType,
    STANDARD_DEV_CARD_COUNTS,
    build_dev_card_deck,
    buy_dev_card,
    draw_dev_card,
)
from catanlab.economy import PlayerInventory
from catanlab.resources import Resource
from catanlab.simulation import PlayerState


def test_standard_deck_has_25_cards():
    deck = build_dev_card_deck(
        seed=42
    )

    assert len(deck.cards) == 25


def test_standard_deck_distribution():
    deck = build_dev_card_deck(
        seed=42
    )

    counts = Counter(
        deck.cards
    )

    assert counts == (
        STANDARD_DEV_CARD_COUNTS
    )


def test_deck_is_reproducible():
    deck_a = build_dev_card_deck(
        seed=123
    )

    deck_b = build_dev_card_deck(
        seed=123
    )

    assert deck_a.cards == deck_b.cards


def test_draw_removes_card():
    deck = build_dev_card_deck(
        seed=42
    )

    before = len(
        deck.cards
    )

    card = draw_dev_card(
        deck
    )

    assert isinstance(
        card,
        DevCardType,
    )

    assert len(
        deck.cards
    ) == before - 1


def test_buy_dev_card_spends_resources():
    deck = build_dev_card_deck(
        seed=42
    )

    player = PlayerState(
        player_id=0
    )

    inventory = PlayerInventory()

    inventory.add(
        Resource.SHEEP
    )
    inventory.add(
        Resource.WHEAT
    )
    inventory.add(
        Resource.ORE
    )

    card = buy_dev_card(
        player,
        inventory,
        deck,
    )

    assert card.value in (
        player.dev_cards
    )

    assert inventory.total() == 0
    assert len(deck.cards) == 24


def test_victory_point_card_counts_for_vp():
    player = PlayerState(
        player_id=0,
        settlements=[1, 2],
        dev_cards=[
            DevCardType.VICTORY_POINT.value,
        ],
    )

    assert player.victory_points == 3


def test_play_knight():
    from catanlab.devcards import (
        play_knight,
    )

    player = PlayerState(
        player_id=0,
        dev_cards=[
            DevCardType.KNIGHT.value,
        ],
    )

    play_knight(
        player
    )

    assert (
        DevCardType.KNIGHT.value
        not in player.dev_cards
    )

    assert player.knights_played == 1


def test_play_knight_requires_card():
    from catanlab.devcards import (
        play_knight,
    )

    player = PlayerState(
        player_id=0
    )

    try:
        play_knight(
            player
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Expected Knight play without "
            "a card to fail"
        )


def test_largest_army_requires_three_knights():
    from catanlab.devcards import (
        update_largest_army,
    )

    players = [
        PlayerState(
            player_id=0,
            knights_played=2,
        ),
        PlayerState(
            player_id=1,
            knights_played=1,
        ),
    ]

    holder = update_largest_army(
        players
    )

    assert holder is None

    assert not any(
        player.has_largest_army
        for player in players
    )


def test_largest_army_awarded_at_three():
    from catanlab.devcards import (
        update_largest_army,
    )

    players = [
        PlayerState(
            player_id=0,
            knights_played=3,
        ),
        PlayerState(
            player_id=1,
            knights_played=2,
        ),
    ]

    holder = update_largest_army(
        players
    )

    assert holder == 0
    assert players[0].has_largest_army
    assert players[0].victory_points == 2


def test_largest_army_transfers_when_overtaken():
    from catanlab.devcards import (
        update_largest_army,
    )

    players = [
        PlayerState(
            player_id=0,
            knights_played=3,
            has_largest_army=True,
        ),
        PlayerState(
            player_id=1,
            knights_played=4,
        ),
    ]

    holder = update_largest_army(
        players
    )

    assert holder == 1
    assert not players[0].has_largest_army
    assert players[1].has_largest_army


def test_largest_army_holder_keeps_on_tie():
    from catanlab.devcards import (
        update_largest_army,
    )

    players = [
        PlayerState(
            player_id=0,
            knights_played=4,
            has_largest_army=True,
        ),
        PlayerState(
            player_id=1,
            knights_played=4,
        ),
    ]

    holder = update_largest_army(
        players
    )

    assert holder == 0
    assert players[0].has_largest_army
    assert not players[1].has_largest_army


def test_move_robber():
    from catanlab.board import build_random_board
    from catanlab.devcards import move_robber

    board = build_random_board(
        seed=42
    )

    old_tile = board.robber_tile_id

    new_tile = next(
        tile.id
        for tile in board.tiles
        if tile.id != old_tile
    )

    move_robber(
        board,
        new_tile,
    )

    assert board.robber_tile_id == new_tile


def test_knight_moves_robber():
    from catanlab.board import build_random_board
    from catanlab.devcards import (
        play_knight_and_move_robber,
    )

    board = build_random_board(
        seed=42
    )

    player = PlayerState(
        player_id=0,
        dev_cards=[
            DevCardType.KNIGHT.value,
        ],
    )

    old_tile = board.robber_tile_id

    new_tile = next(
        tile.id
        for tile in board.tiles
        if tile.id != old_tile
    )

    play_knight_and_move_robber(
        player,
        board,
        new_tile,
    )

    assert player.knights_played == 1
    assert board.robber_tile_id == new_tile
    assert (
        DevCardType.KNIGHT.value
        not in player.dev_cards
    )


def test_robber_must_move():
    from catanlab.board import build_random_board
    from catanlab.devcards import move_robber

    board = build_random_board(
        seed=42
    )

    try:
        move_robber(
            board,
            board.robber_tile_id,
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Expected robber to require "
            "a different tile"
        )


def test_seven_discards_half_hand():
    import random

    from catanlab.devcards import (
        discard_for_seven,
    )

    inventory = PlayerInventory()

    inventory.add(
        Resource.WOOD,
        5,
    )

    inventory.add(
        Resource.WHEAT,
        4,
    )

    discarded = discard_for_seven(
        inventory,
        random.Random(42),
    )

    assert len(discarded) == 4
    assert inventory.total() == 5


def test_seven_does_not_discard_at_seven():
    import random

    from catanlab.devcards import (
        discard_for_seven,
    )

    inventory = PlayerInventory()

    inventory.add(
        Resource.WOOD,
        7,
    )

    discarded = discard_for_seven(
        inventory,
        random.Random(42),
    )

    assert discarded == []
    assert inventory.total() == 7


def test_players_adjacent_to_robber_tile():
    from catanlab.board import (
        Board,
        Tile,
        Vertex,
    )
    from catanlab.devcards import (
        players_adjacent_to_tile,
    )
    from catanlab.graph import HexCoord

    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=HexCoord(0, 0),
                resource=Resource.WOOD,
                number=6,
            )
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
    ]

    eligible = players_adjacent_to_tile(
        board,
        players,
        0,
        exclude_player_id=0,
    )

    assert eligible == [1]


def test_steal_random_resource():
    import random

    from catanlab.devcards import (
        steal_random_resource,
    )

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    inventories[1].add(
        Resource.ORE,
        2,
    )

    stolen = steal_random_resource(
        0,
        1,
        inventories,
        random.Random(42),
    )

    assert stolen == Resource.ORE

    assert inventories[0].count(
        Resource.ORE
    ) == 1

    assert inventories[1].count(
        Resource.ORE
    ) == 1


def test_rob_adjacent_player():
    import random

    from catanlab.board import (
        Board,
        Tile,
        Vertex,
    )
    from catanlab.devcards import (
        rob_adjacent_player,
    )
    from catanlab.graph import HexCoord

    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=HexCoord(0, 0),
                resource=Resource.WHEAT,
                number=9,
            )
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
    ]

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    inventories[1].add(
        Resource.WHEAT
    )

    result = rob_adjacent_player(
        board,
        players,
        inventories,
        thief_id=0,
        rng=random.Random(42),
    )

    assert result == (
        1,
        Resource.WHEAT,
    )

    assert inventories[0].count(
        Resource.WHEAT
    ) == 1

    assert inventories[1].total() == 0


def test_year_of_plenty_adds_two_resources():
    from catanlab.devcards import (
        play_year_of_plenty,
    )

    player = PlayerState(
        player_id=0,
        dev_cards=[
            DevCardType.YEAR_OF_PLENTY.value,
        ],
    )

    inventory = PlayerInventory()

    play_year_of_plenty(
        player,
        inventory,
        Resource.ORE,
        Resource.WHEAT,
    )

    assert inventory.count(
        Resource.ORE
    ) == 1

    assert inventory.count(
        Resource.WHEAT
    ) == 1

    assert (
        DevCardType.YEAR_OF_PLENTY.value
        not in player.dev_cards
    )


def test_monopoly_collects_resource():
    from catanlab.devcards import (
        play_monopoly,
    )

    player = PlayerState(
        player_id=0,
        dev_cards=[
            DevCardType.MONOPOLY.value,
        ],
    )

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
        PlayerInventory(),
    ]

    inventories[1].add(
        Resource.ORE,
        2,
    )

    inventories[2].add(
        Resource.ORE,
        3,
    )

    collected = play_monopoly(
        player,
        inventories,
        Resource.ORE,
    )

    assert collected == 5

    assert inventories[0].count(
        Resource.ORE
    ) == 5

    assert inventories[1].count(
        Resource.ORE
    ) == 0

    assert inventories[2].count(
        Resource.ORE
    ) == 0

    assert (
        DevCardType.MONOPOLY.value
        not in player.dev_cards
    )


def test_monopoly_requires_card():
    from catanlab.devcards import (
        play_monopoly,
    )

    player = PlayerState(
        player_id=0
    )

    inventories = [
        PlayerInventory()
    ]

    try:
        play_monopoly(
            player,
            inventories,
            Resource.WOOD,
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Expected Monopoly play without card to fail"
        )


def test_road_building_places_two_roads():
    from catanlab.board import (
        Board,
        Edge,
        Vertex,
    )
    from catanlab.devcards import (
        play_road_building,
    )

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
            ),
            Vertex(
                id=1,
                position=(1.0, 0.0),
            ),
            Vertex(
                id=2,
                position=(2.0, 0.0),
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
        dev_cards=[
            DevCardType.ROAD_BUILDING.value,
        ],
    )

    play_road_building(
        player,
        board,
        [player],
        first_edge=(0, 1),
        second_edge=(1, 2),
    )

    assert player.roads == [
        (0, 1),
        (1, 2),
    ]

    assert (
        DevCardType.ROAD_BUILDING.value
        not in player.dev_cards
    )

    assert player.played_dev_cards == [
        DevCardType.ROAD_BUILDING.value
    ]


def test_road_building_second_road_can_extend_first():
    from catanlab.board import (
        Board,
        Edge,
        Vertex,
    )
    from catanlab.devcards import (
        play_road_building,
    )

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
            ),
            Vertex(
                id=1,
                position=(1.0, 0.0),
            ),
            Vertex(
                id=2,
                position=(2.0, 0.0),
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
        dev_cards=[
            DevCardType.ROAD_BUILDING.value,
        ],
    )

    play_road_building(
        player,
        board,
        [player],
        first_edge=(0, 1),
        second_edge=(1, 2),
    )

    assert (1, 2) in player.roads


def test_road_building_rolls_back_illegal_second_road():
    from catanlab.board import (
        Board,
        Edge,
        Vertex,
    )
    from catanlab.devcards import (
        play_road_building,
    )

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
            ),
            Vertex(
                id=1,
                position=(1.0, 0.0),
            ),
            Vertex(
                id=2,
                position=(2.0, 0.0),
            ),
        ],
        edges=[
            Edge(
                vertex_a=0,
                vertex_b=1,
            )
        ],
    )

    player = PlayerState(
        player_id=0,
        settlements=[0],
        dev_cards=[
            DevCardType.ROAD_BUILDING.value,
        ],
    )

    try:
        play_road_building(
            player,
            board,
            [player],
            first_edge=(0, 1),
            second_edge=(1, 2),
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Expected illegal second road to fail"
        )

    assert player.roads == []

    assert (
        DevCardType.ROAD_BUILDING.value
        in player.dev_cards
    )


def test_road_building_requires_card():
    from catanlab.board import (
        Board,
        Edge,
        Vertex,
    )
    from catanlab.devcards import (
        play_road_building,
    )

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
            ),
            Vertex(
                id=1,
                position=(1.0, 0.0),
            ),
            Vertex(
                id=2,
                position=(2.0, 0.0),
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

    try:
        play_road_building(
            player,
            board,
            [player],
            first_edge=(0, 1),
            second_edge=(1, 2),
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Expected Road Building without card to fail"
        )


def test_newly_bought_knight_is_not_playable():
    from catanlab.devcards import (
        DevCardDeck,
        DevCardType,
        buy_dev_card,
        play_knight,
    )
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState

    player = PlayerState(
        player_id=0
    )

    inventory = PlayerInventory()

    inventory.add(
        Resource.SHEEP
    )
    inventory.add(
        Resource.WHEAT
    )
    inventory.add(
        Resource.ORE
    )

    deck = DevCardDeck(
        cards=[
            DevCardType.KNIGHT,
        ]
    )

    buy_dev_card(
        player,
        inventory,
        deck,
    )

    assert (
        DevCardType.KNIGHT.value
        in player.dev_cards
    )

    assert (
        DevCardType.KNIGHT.value
        in player.new_dev_cards
    )

    try:
        play_knight(
            player
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Newly purchased Knight should not "
            "be playable this turn."
        )


def test_old_copy_is_playable_when_same_card_was_just_bought():
    from catanlab.devcards import (
        DevCardType,
        play_knight,
    )
    from catanlab.simulation import PlayerState

    player = PlayerState(
        player_id=0,
        dev_cards=[
            DevCardType.KNIGHT.value,
            DevCardType.KNIGHT.value,
        ],
        new_dev_cards=[
            DevCardType.KNIGHT.value,
        ],
    )

    play_knight(
        player
    )

    assert player.knights_played == 1

    assert player.dev_cards == [
        DevCardType.KNIGHT.value,
    ]


def test_new_card_becomes_playable_after_new_marker_clears():
    from catanlab.devcards import (
        DevCardType,
        play_knight,
    )
    from catanlab.simulation import PlayerState

    player = PlayerState(
        player_id=0,
        dev_cards=[
            DevCardType.KNIGHT.value,
        ],
        new_dev_cards=[
            DevCardType.KNIGHT.value,
        ],
    )

    # This is what the turn engine will do at the
    # start of the player's next turn.
    player.new_dev_cards.clear()

    play_knight(
        player
    )

    assert player.knights_played == 1
    assert player.dev_cards == []


def test_newly_bought_road_building_is_not_playable():
    from catanlab.board import (
        Board,
        Edge,
        Vertex,
    )
    from catanlab.devcards import (
        DevCardType,
        play_road_building,
    )
    from catanlab.simulation import PlayerState

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=i,
                position=(float(i), 0.0),
            )
            for i in range(3)
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
        dev_cards=[
            DevCardType.ROAD_BUILDING.value,
        ],
        new_dev_cards=[
            DevCardType.ROAD_BUILDING.value,
        ],
    )

    try:
        play_road_building(
            player,
            board,
            [player],
            (0, 1),
            (1, 2),
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "New Road Building card should not "
            "be playable this turn."
        )

    assert player.roads == []


def test_road_building_can_place_only_one_legal_road():
    from catanlab.board import (
        Board,
        Edge,
        Vertex,
    )
    from catanlab.devcards import (
        play_road_building,
    )

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
            ),
            Vertex(
                id=1,
                position=(1.0, 0.0),
            ),
        ],
        edges=[
            Edge(
                vertex_a=0,
                vertex_b=1,
            ),
        ],
    )

    player = PlayerState(
        player_id=0,
        settlements=[0],
        dev_cards=[
            DevCardType.ROAD_BUILDING.value,
        ],
    )

    play_road_building(
        player,
        board,
        [player],
        first_edge=(0, 1),
    )

    assert player.roads == [
        (0, 1),
    ]

    assert (
        DevCardType.ROAD_BUILDING.value
        not in player.dev_cards
    )


def test_buy_dev_card_empty_deck_does_not_spend_resources():
    from catanlab.devcards import (
        DevCardDeck,
        buy_dev_card,
    )
    from catanlab.economy import (
        PlayerInventory,
        ResourceBank,
    )
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState

    player = PlayerState(
        player_id=0
    )

    inventory = PlayerInventory()

    inventory.add(
        Resource.SHEEP
    )
    inventory.add(
        Resource.WHEAT
    )
    inventory.add(
        Resource.ORE
    )

    bank = ResourceBank()

    # These cards are currently in the player's
    # hand, so remove them from the bank to create
    # a conservation-consistent state.
    bank.remove(
        Resource.SHEEP
    )
    bank.remove(
        Resource.WHEAT
    )
    bank.remove(
        Resource.ORE
    )

    deck = DevCardDeck(
        cards=[]
    )

    before_hand = {
        Resource.SHEEP:
            inventory.count(Resource.SHEEP),
        Resource.WHEAT:
            inventory.count(Resource.WHEAT),
        Resource.ORE:
            inventory.count(Resource.ORE),
    }

    before_bank = {
        Resource.SHEEP:
            bank.count(Resource.SHEEP),
        Resource.WHEAT:
            bank.count(Resource.WHEAT),
        Resource.ORE:
            bank.count(Resource.ORE),
    }

    try:
        buy_dev_card(
            player,
            inventory,
            deck,
            bank=bank,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected purchase from empty deck to fail."
        )

    assert {
        resource:
            inventory.count(resource)
        for resource in before_hand
    } == before_hand

    assert {
        resource:
            bank.count(resource)
        for resource in before_bank
    } == before_bank

    assert player.dev_cards == []
    assert player.new_dev_cards == []
    assert deck.cards == []


def test_knight_invalid_robber_tile_does_not_consume_card():
    from catanlab.board import build_random_board
    from catanlab.devcards import (
        DevCardType,
        play_knight_and_move_robber,
    )
    from catanlab.simulation import PlayerState

    board = build_random_board(
        seed=123
    )

    player = PlayerState(
        player_id=0,
        dev_cards=[
            DevCardType.KNIGHT.value
        ],
    )

    original_robber = (
        board.robber_tile_id
    )

    try:
        play_knight_and_move_robber(
            player,
            board,
            tile_id=len(board.tiles),
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Invalid robber tile should fail."
        )

    assert (
        DevCardType.KNIGHT.value
        in player.dev_cards
    )
    assert player.knights_played == 0
    assert (
        board.robber_tile_id
        == original_robber
    )


def test_knight_same_robber_tile_does_not_consume_card():
    from catanlab.board import build_random_board
    from catanlab.devcards import (
        DevCardType,
        play_knight_and_move_robber,
    )
    from catanlab.simulation import PlayerState

    board = build_random_board(
        seed=123
    )

    player = PlayerState(
        player_id=0,
        dev_cards=[
            DevCardType.KNIGHT.value
        ],
    )

    original_robber = (
        board.robber_tile_id
    )

    try:
        play_knight_and_move_robber(
            player,
            board,
            tile_id=original_robber,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Robber must move to another tile."
        )

    assert (
        DevCardType.KNIGHT.value
        in player.dev_cards
    )
    assert player.knights_played == 0
    assert (
        board.robber_tile_id
        == original_robber
    )


def test_play_knight_records_public_history():
    from catanlab.devcards import (
        DevCardType,
        play_knight,
    )
    from catanlab.simulation import PlayerState

    player = PlayerState(
        player_id=0,
        dev_cards=[
            DevCardType.KNIGHT.value,
        ],
    )

    play_knight(player)

    assert player.played_dev_cards == [
        DevCardType.KNIGHT.value
    ]


def test_year_of_plenty_records_public_history():
    from catanlab.devcards import (
        DevCardType,
        play_year_of_plenty,
    )
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState

    player = PlayerState(
        player_id=0,
        dev_cards=[
            DevCardType.YEAR_OF_PLENTY.value,
        ],
    )

    inventory = PlayerInventory()

    play_year_of_plenty(
        player,
        inventory,
        Resource.WOOD,
        Resource.BRICK,
    )

    assert player.played_dev_cards == [
        DevCardType.YEAR_OF_PLENTY.value
    ]


def test_monopoly_records_public_history():
    from catanlab.devcards import (
        DevCardType,
        play_monopoly,
    )
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState

    player = PlayerState(
        player_id=0,
        dev_cards=[
            DevCardType.MONOPOLY.value,
        ],
    )

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    inventories[1].add(
        Resource.ORE,
        2,
    )

    play_monopoly(
        player,
        inventories,
        Resource.ORE,
    )

    assert player.played_dev_cards == [
        DevCardType.MONOPOLY.value
    ]


def test_failed_road_building_does_not_record_history():
    import pytest

    from catanlab.board import build_random_board
    from catanlab.devcards import (
        DevCardType,
        play_road_building,
    )
    from catanlab.simulation import PlayerState

    board = build_random_board(seed=123)

    players = [
        PlayerState(player_id=i)
        for i in range(4)
    ]

    player = players[0]

    player.dev_cards.append(
        DevCardType.ROAD_BUILDING.value
    )

    with pytest.raises(ValueError):
        play_road_building(
            player,
            board,
            players,
            first_edge=(-1, -2),
        )

    assert player.played_dev_cards == []

    assert (
        DevCardType.ROAD_BUILDING.value
        in player.dev_cards
    )
