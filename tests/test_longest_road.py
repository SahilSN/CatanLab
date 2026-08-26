from catanlab.longest_road import (
    longest_road_length,
    update_longest_road,
)
from catanlab.simulation import PlayerState


def test_no_roads_has_length_zero():
    player = PlayerState(
        player_id=0
    )

    assert longest_road_length(
        player,
        [player],
    ) == 0


def test_straight_road():
    player = PlayerState(
        player_id=0,
        roads=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
        ],
    )

    assert longest_road_length(
        player,
        [player],
    ) == 4


def test_fork_does_not_count_all_edges():
    """
        0
        |
    1 - 2 - 3

    Three roads exist, but no single trail can
    use all three without reusing the central
    connection pattern incorrectly.

    Longest trail = 2.
    """

    player = PlayerState(
        player_id=0,
        roads=[
            (1, 0),
            (1, 2),
            (1, 3),
        ],
    )

    assert longest_road_length(
        player,
        [player],
    ) == 2


def test_longer_branching_network():
    """
    0 - 1 - 2 - 3
            |
            4 - 5

    Longest trail is 4:
    0-1-2-4-5 or 3-2-1-0 depending on topology.
    """

    player = PlayerState(
        player_id=0,
        roads=[
            (0, 1),
            (1, 2),
            (2, 3),
            (2, 4),
            (4, 5),
        ],
    )

    assert longest_road_length(
        player,
        [player],
    ) == 4


def test_loop_counts_each_edge_once():
    player = PlayerState(
        player_id=0,
        roads=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
        ],
    )

    assert longest_road_length(
        player,
        [player],
    ) == 4


def test_loop_with_branch():
    player = PlayerState(
        player_id=0,
        roads=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (0, 4),
            (4, 5),
        ],
    )

    assert longest_road_length(
        player,
        [player],
    ) == 6


def test_opponent_settlement_interrupts_road():
    player = PlayerState(
        player_id=0,
        roads=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
        ],
    )

    opponent = PlayerState(
        player_id=1,
        settlements=[3],
    )

    assert longest_road_length(
        player,
        [
            player,
            opponent,
        ],
    ) == 3


def test_longest_road_requires_five():
    player = PlayerState(
        player_id=0,
        roads=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
        ],
    )

    holder = update_longest_road(
        [player]
    )

    assert holder is None
    assert not player.has_longest_road


def test_longest_road_awarded_at_five():
    player = PlayerState(
        player_id=0,
        roads=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
        ],
    )

    holder = update_longest_road(
        [player]
    )

    assert holder == 0
    assert player.has_longest_road
    assert player.victory_points == 2


def test_current_holder_keeps_on_tie():
    player_a = PlayerState(
        player_id=0,
        roads=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
        ],
        has_longest_road=True,
    )

    player_b = PlayerState(
        player_id=1,
        roads=[
            (10, 11),
            (11, 12),
            (12, 13),
            (13, 14),
            (14, 15),
        ],
    )

    holder = update_longest_road(
        [
            player_a,
            player_b,
        ]
    )

    assert holder == 0
    assert player_a.has_longest_road
    assert not player_b.has_longest_road


def test_longest_road_transfers_when_overtaken():
    player_a = PlayerState(
        player_id=0,
        roads=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
        ],
        has_longest_road=True,
    )

    player_b = PlayerState(
        player_id=1,
        roads=[
            (10, 11),
            (11, 12),
            (12, 13),
            (13, 14),
            (14, 15),
            (15, 16),
        ],
    )

    holder = update_longest_road(
        [
            player_a,
            player_b,
        ]
    )

    assert holder == 1
    assert not player_a.has_longest_road
    assert player_b.has_longest_road
