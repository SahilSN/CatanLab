from catanlab.action_scoring import (
    score_actions,
)
from catanlab.simulation import PlayerState
from catanlab.strategies import StrategyType


def test_full_ows_baseline_prefers_city_and_dev():
    player = PlayerState(
        player_id=0
    )

    utilities = score_actions(
        StrategyType.FULL_OWS,
        player,
        [player],
    )

    assert (
        utilities.build_city
        >
        utilities.build_road
    )

    assert (
        utilities.buy_dev_card
        >
        utilities.build_road
    )


def test_road_strategy_prefers_expansion():
    player = PlayerState(
        player_id=0
    )

    utilities = score_actions(
        StrategyType.ROAD_BUILDING,
        player,
        [player],
    )

    assert (
        utilities.build_road
        >
        utilities.buy_dev_card
    )

    assert (
        utilities.build_settlement
        >
        utilities.build_city
    )


def test_dev_card_priority_spikes_near_largest_army_win():
    player = PlayerState(
        player_id=0,
        settlements=[
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ],
        knights_played=2,
    )

    opponent = PlayerState(
        player_id=1,
        knights_played=1,
    )

    utilities = score_actions(
        StrategyType.FULL_OWS,
        player,
        [
            player,
            opponent,
        ],
    )

    assert (
        utilities.buy_dev_card
        >
        utilities.build_city
    )


def test_road_priority_spikes_near_longest_road_win():
    player = PlayerState(
        player_id=0,
        settlements=[
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
        ],
        roads=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
        ],
    )

    utilities = score_actions(
        StrategyType.ROAD_BUILDING,
        player,
        [player],
    )

    assert (
        utilities.build_road
        >
        utilities.build_settlement
    )


def test_largest_army_holder_does_not_get_proximity_bonus():
    player = PlayerState(
        player_id=0,
        knights_played=4,
        has_largest_army=True,
    )

    utilities = score_actions(
        StrategyType.FULL_OWS,
        player,
        [player],
    )

    assert (
        utilities.buy_dev_card
        == 5.5
    )


def test_road_strategy_pivots_after_comfortable_longest_road_lead():
    from catanlab.action_scoring import (
        score_actions,
    )
    from catanlab.simulation import PlayerState
    from catanlab.strategies import StrategyType

    player = PlayerState(
        player_id=0,
        roads=[
            (i, i + 1)
            for i in range(7)
        ],
        has_longest_road=True,
    )

    opponent = PlayerState(
        player_id=1,
        roads=[
            (10 + i, 11 + i)
            for i in range(5)
        ],
    )

    utilities = score_actions(
        StrategyType.ROAD_BUILDING,
        player,
        [player, opponent],
    )

    assert (
        utilities.build_settlement
        > utilities.build_road
    )

    assert (
        utilities.build_city
        > utilities.build_road
    )


def test_road_strategy_defends_threatened_longest_road():
    from catanlab.action_scoring import (
        score_actions,
    )
    from catanlab.simulation import PlayerState
    from catanlab.strategies import StrategyType

    player = PlayerState(
        player_id=0,
        roads=[
            (i, i + 1)
            for i in range(6)
        ],
        has_longest_road=True,
    )

    opponent = PlayerState(
        player_id=1,
        roads=[
            (10 + i, 11 + i)
            for i in range(5)
        ],
    )

    utilities = score_actions(
        StrategyType.ROAD_BUILDING,
        player,
        [player, opponent],
    )

    assert (
        utilities.build_road
        > utilities.build_city
    )


def test_road_strategy_late_game_pivot_favors_city():
    from catanlab.action_scoring import (
        score_actions,
    )
    from catanlab.simulation import PlayerState
    from catanlab.strategies import StrategyType

    player = PlayerState(
        player_id=0,
        settlements=[20],
        cities=[21, 22],
        roads=[
            (i, i + 1)
            for i in range(7)
        ],
        dev_cards=[
            "victory_point",
        ],
        has_longest_road=True,
    )

    opponent = PlayerState(
        player_id=1,
        roads=[
            (10 + i, 11 + i)
            for i in range(5)
        ],
    )

    # 1 settlement + 4 city VP + 1 dev VP
    # + 2 Longest Road VP = 8 VP.
    assert player.victory_points == 8

    utilities = score_actions(
        StrategyType.ROAD_BUILDING,
        player,
        [player, opponent],
    )

    assert (
        utilities.build_city
        > utilities.build_road
    )


def test_road_strategy_prefers_pass_to_redundant_secure_road():
    from catanlab.action_scoring import score_actions
    from catanlab.simulation import PlayerState
    from catanlab.strategies import StrategyType

    player = PlayerState(
        player_id=0,
        roads=[
            (i, i + 1)
            for i in range(8)
        ],
        has_longest_road=True,
    )

    opponent = PlayerState(
        player_id=1,
        roads=[
            (20 + i, 21 + i)
            for i in range(5)
        ],
    )

    utilities = score_actions(
        StrategyType.ROAD_BUILDING,
        player,
        [player, opponent],
    )

    assert (
        utilities.build_road
        < utilities.pass_turn
    )
