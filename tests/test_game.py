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
