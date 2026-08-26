from collections import defaultdict

from catanlab.simulation import PlayerState


def opponent_blocked_vertices(
    players: list[PlayerState],
    player_id: int,
) -> set[int]:
    """
    Return vertices occupied by an opponent's
    settlement or city.

    A player's road may reach such a vertex but
    may not continue through it.
    """

    blocked: set[int] = set()

    for player in players:
        if player.player_id == player_id:
            continue

        blocked.update(
            player.settlements
        )

        blocked.update(
            player.cities
        )

    return blocked


def longest_road_length(
    player: PlayerState,
    players: list[PlayerState],
) -> int:
    """
    Compute the longest continuous trail of roads
    owned by the player.

    An edge may not be reused within a trail.

    Opponent settlements and cities interrupt
    continuation through their occupied vertex.
    """

    if not player.roads:
        return 0

    adjacency: dict[
        int,
        list[tuple[int, tuple[int, int]]],
    ] = defaultdict(list)

    edges: set[
        tuple[int, int]
    ] = set()

    for a, b in player.roads:
        edge = tuple(
            sorted(
                (
                    a,
                    b,
                )
            )
        )

        edges.add(
            edge
        )

    for a, b in edges:
        adjacency[a].append(
            (
                b,
                (a, b),
            )
        )

        adjacency[b].append(
            (
                a,
                (a, b),
            )
        )

    blocked = opponent_blocked_vertices(
        players,
        player.player_id,
    )

    def search(
        vertex: int,
        used_edges: set[
            tuple[int, int]
        ],
    ) -> int:
        best = len(
            used_edges
        )

        # You may finish a road at an opponent
        # building, but may not continue through it.
        if (
            vertex in blocked
            and used_edges
        ):
            return best

        for neighbor, edge in adjacency[
            vertex
        ]:
            if edge in used_edges:
                continue

            used_edges.add(
                edge
            )

            best = max(
                best,
                search(
                    neighbor,
                    used_edges,
                ),
            )

            used_edges.remove(
                edge
            )

        return best

    best = 0

    for start_vertex in adjacency:
        best = max(
            best,
            search(
                start_vertex,
                set(),
            ),
        )

    return best


def update_longest_road(
    players: list[PlayerState],
) -> int | None:
    """
    Update Longest Road ownership.

    A player needs a road length of at least 5.

    The current holder keeps Longest Road on a tie.
    Another player must strictly exceed the
    current holder's road length to take it.

    Returns the holder's player ID, or None.
    """

    lengths = {
        player.player_id:
            longest_road_length(
                player,
                players,
            )
        for player in players
    }

    current_holder = next(
        (
            player
            for player in players
            if player.has_longest_road
        ),
        None,
    )

    if current_holder is not None:
        holder_length = lengths[
            current_holder.player_id
        ]

        # If the current holder has fallen below
        # five because an opponent settlement split
        # the road, the award may need to change.
        challengers = [
            player
            for player in players
            if (
                player.player_id
                != current_holder.player_id
                and lengths[
                    player.player_id
                ] >= 5
            )
        ]

        if holder_length >= 5:
            stronger = [
                player
                for player in challengers
                if lengths[
                    player.player_id
                ] > holder_length
            ]

            if not stronger:
                return (
                    current_holder.player_id
                )

            best_length = max(
                lengths[
                    player.player_id
                ]
                for player in stronger
            )

            winners = [
                player
                for player in stronger
                if lengths[
                    player.player_id
                ] == best_length
            ]

            if len(winners) != 1:
                return (
                    current_holder.player_id
                )

            winner = winners[0]

            current_holder.has_longest_road = False
            winner.has_longest_road = True

            return winner.player_id

        current_holder.has_longest_road = False

    eligible = [
        player
        for player in players
        if lengths[
            player.player_id
        ] >= 5
    ]

    if not eligible:
        return None

    best_length = max(
        lengths[
            player.player_id
        ]
        for player in eligible
    )

    leaders = [
        player
        for player in eligible
        if lengths[
            player.player_id
        ] == best_length
    ]

    if len(leaders) != 1:
        return None

    winner = leaders[0]
    winner.has_longest_road = True

    return winner.player_id
