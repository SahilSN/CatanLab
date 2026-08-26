from catanlab.game import (
    VICTORY_POINTS_TO_WIN,
    run_game,
    setup_game,
    winner,
)
from catanlab.simulation import PlayerState
from catanlab.strategies import StrategyType


def standard_strategies():
    return [
        StrategyType.FULL_OWS,
        StrategyType.ROAD_BUILDING,
        StrategyType.FIVE_RESOURCE,
        StrategyType.HYBRID_OWS,
    ]


def test_setup_game_creates_four_players():
    (
        board,
        draft,
        inventories,
        agents,
        dev_deck,
    ) = setup_game(
        standard_strategies(),
        board_seed=42,
        dev_seed=42,
    )

    assert len(
        draft.players
    ) == 4

    assert len(
        inventories
    ) == 4

    assert len(
        agents
    ) == 4


def test_setup_players_have_two_settlements_and_roads():
    (
        board,
        draft,
        inventories,
        agents,
        dev_deck,
    ) = setup_game(
        standard_strategies(),
        board_seed=42,
        dev_seed=42,
    )

    for player in draft.players:
        assert len(
            player.settlements
        ) == 2

        assert len(
            player.roads
        ) == 2


def test_setup_grants_starting_resources():
    (
        board,
        draft,
        inventories,
        agents,
        dev_deck,
    ) = setup_game(
        standard_strategies(),
        board_seed=42,
        dev_seed=42,
    )

    assert any(
        inventory.total() > 0
        for inventory in inventories
    )


def test_winner_requires_ten_points():
    player = PlayerState(
        player_id=0,
        settlements=list(
            range(
                VICTORY_POINTS_TO_WIN
                - 1
            )
        ),
    )

    assert winner(
        [player]
    ) is None

    player.settlements.append(
        99
    )

    assert winner(
        [player]
    ) == 0


def test_game_records_award_history():
    result = run_game(
        standard_strategies(),
        seed=42,
        max_turns=8,
    )

    assert len(
        result.award_history
    ) == result.turns_played

    first = result.award_history[0]

    assert first.turn_number == 1
    assert len(
        first.knights_played
    ) == 4
    assert len(
        first.road_lengths
    ) == 4


def test_game_preserves_opening_draft_metadata():
    result = run_game(
        standard_strategies(),
        seed=42,
        max_turns=8,
    )

    assert len(
        result.opening_placements
    ) == 8

    assert len(
        result.opening_roads
    ) == 8

    placement_counts = {
        player_id: 0
        for player_id in range(4)
    }

    road_counts = {
        player_id: 0
        for player_id in range(4)
    }

    for player_id, _ in (
        result.opening_placements
    ):
        placement_counts[
            player_id
        ] += 1

    for player_id, _ in (
        result.opening_roads
    ):
        road_counts[
            player_id
        ] += 1

    assert all(
        count == 2
        for count
        in placement_counts.values()
    )

    assert all(
        count == 2
        for count
        in road_counts.values()
    )


def test_game_preserves_internal_seeds():
    import random

    seed = 42

    rng = random.Random(
        seed
    )

    expected_board_seed = (
        rng.randrange(
            2**31
        )
    )

    expected_dev_seed = (
        rng.randrange(
            2**31
        )
    )

    result = run_game(
        standard_strategies(),
        seed=seed,
        max_turns=8,
    )

    assert (
        result.board_seed
        == expected_board_seed
    )

    assert (
        result.dev_seed
        == expected_dev_seed
    )


def test_public_victory_points_hide_vp_dev_cards():
    from catanlab.simulation import PlayerState

    player = PlayerState(
        player_id=0,
        settlements=[1, 2],
        cities=[3],
        dev_cards=[
            "victory_point",
            "victory_point",
        ],
        has_largest_army=True,
    )

    # 2 settlements + city worth 2 + Largest Army.
    assert player.public_victory_points == 6

    # Two hidden VP cards count toward the true score.
    assert player.victory_points == 8


def test_public_victory_points_include_public_awards():
    from catanlab.simulation import PlayerState

    player = PlayerState(
        player_id=0,
        settlements=[1],
        cities=[2],
        has_largest_army=True,
        has_longest_road=True,
    )

    assert player.public_victory_points == 7
    assert player.victory_points == 7


def test_hidden_vp_cards_still_count_for_winning():
    from catanlab.game import winner
    from catanlab.simulation import PlayerState

    player = PlayerState(
        player_id=0,
        settlements=[1, 2],
        cities=[3, 4],
        dev_cards=[
            "victory_point",
            "victory_point",
        ],
        has_longest_road=True,
    )

    # Public:
    # 2 settlements + 2 cities * 2 + LR = 8
    assert player.public_victory_points == 8

    # True:
    # 8 public + 2 hidden VP = 10
    assert player.victory_points == 10

    assert winner([player]) == 0


def test_opening_draft_uses_standard_snake_order():
    """
    Four-player setup must proceed:
        0, 1, 2, 3, 3, 2, 1, 0.
    """

    (
        board,
        draft,
        inventories,
        agents,
        dev_deck,
    ) = setup_game(
        standard_strategies(),
        board_seed=42,
        dev_seed=42,
    )

    assert [
        player_id
        for player_id, _ in draft.placement_order
    ] == [
        0,
        1,
        2,
        3,
        3,
        2,
        1,
        0,
    ]

    assert [
        player_id
        for player_id, _ in draft.road_order
    ] == [
        0,
        1,
        2,
        3,
        3,
        2,
        1,
        0,
    ]


def test_each_setup_road_touches_just_placed_settlement():
    """
    Each free setup road must connect directly to the
    settlement placed in the same setup action.
    """

    (
        board,
        draft,
        inventories,
        agents,
        dev_deck,
    ) = setup_game(
        standard_strategies(),
        board_seed=42,
        dev_seed=42,
    )

    assert len(draft.placement_order) == len(
        draft.road_order
    )

    for (
        placement,
        road_placement,
    ) in zip(
        draft.placement_order,
        draft.road_order,
    ):
        (
            settlement_player_id,
            vertex_id,
        ) = placement

        (
            road_player_id,
            road,
        ) = road_placement

        assert (
            road_player_id
            == settlement_player_id
        )

        assert vertex_id in road


def test_setup_resources_come_from_second_settlement_only():
    """
    At the end of setup, each player's entire hand
    must equal one resource from every non-desert
    hex adjacent to their second settlement.
    """

    from collections import Counter

    from catanlab.resources import Resource

    (
        board,
        draft,
        inventories,
        agents,
        dev_deck,
    ) = setup_game(
        standard_strategies(),
        board_seed=42,
        dev_seed=42,
    )

    producing_resources = (
        Resource.WOOD,
        Resource.BRICK,
        Resource.SHEEP,
        Resource.WHEAT,
        Resource.ORE,
    )

    for player, inventory in zip(
        draft.players,
        inventories,
    ):
        second_vertex_id = (
            player.settlements[1]
        )

        second_vertex = board.vertices[
            second_vertex_id
        ]

        expected = Counter()

        for tile_id in (
            second_vertex.adjacent_tiles
        ):
            resource = board.tiles[
                tile_id
            ].resource

            if resource != Resource.DESERT:
                expected[resource] += 1

        actual = Counter(
            {
                resource: inventory.count(
                    resource
                )
                for resource
                in producing_resources
                if inventory.count(
                    resource
                ) > 0
            }
        )

        assert actual == expected
