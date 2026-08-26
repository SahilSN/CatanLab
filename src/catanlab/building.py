from catanlab.board import Board
from catanlab.economy import (
    BuildType,
    PlayerInventory,
    ResourceBank,
)
from catanlab.longest_road import (
    update_longest_road,
)
from catanlab.simulation import PlayerState

MAX_ROADS = 15
MAX_SETTLEMENTS = 5
MAX_CITIES = 4




def build_road(
    board: Board,
    players: list[PlayerState],
    player: PlayerState,
    inventory: PlayerInventory,
    vertex_a: int,
    vertex_b: int,
    bank: ResourceBank | None = None,
) -> None:
    edge = canonical_edge(
        vertex_a,
        vertex_b,
    )

    if not can_build_road(
        board,
        players,
        player,
        vertex_a,
        vertex_b,
    ):
        raise ValueError(
            "Road placement is not legal."
        )

    inventory.spend(
        BuildType.ROAD,
        bank=bank,
    )

    player.roads.append(
        edge
    )

    update_longest_road(
        players
    )

    update_longest_road(
        players
    )


def build_settlement(
    board: Board,
    players: list[PlayerState],
    player: PlayerState,
    inventory: PlayerInventory,
    vertex_id: int,
    bank: ResourceBank | None = None,
) -> None:
    if len(player.settlements) >= MAX_SETTLEMENTS:
        raise ValueError(
            "Player has no settlement pieces remaining."
        )

    if vertex_id in player.settlements:
        raise ValueError(
            "Player already has a settlement "
            "on this vertex."
        )

    if vertex_id in player.cities:
        raise ValueError(
            "Player already has a city "
            "on this vertex."
        )

    for existing in (
        player.settlements
        + player.cities
    ):
        if (
            vertex_id
            in board.vertices[
                existing
            ].neighbors
        ):
            raise ValueError(
                "Settlement violates "
                "the distance rule."
            )

    inventory.spend(
        BuildType.SETTLEMENT,
        bank=bank,
    )

    player.settlements.append(
        vertex_id
    )

    update_longest_road(
        players
    )


def build_city(
    player: PlayerState,
    inventory: PlayerInventory,
    vertex_id: int,
    bank: ResourceBank | None = None,
) -> None:
    if len(player.cities) >= MAX_CITIES:
        raise ValueError(
            "Player has no city pieces remaining."
        )

    if vertex_id not in player.settlements:
        raise ValueError(
            "A city must upgrade "
            "an existing settlement."
        )

    inventory.spend(
        BuildType.CITY,
        bank=bank,
    )

    player.settlements.remove(
        vertex_id
    )

    player.cities.append(
        vertex_id
    )


def occupied_building_vertices(
    players: list[PlayerState],
) -> set[int]:
    """
    Return all vertices occupied by settlements
    or cities across all players.
    """

    occupied: set[int] = set()

    for player in players:
        occupied.update(
            player.settlements
        )

        occupied.update(
            player.cities
        )

    return occupied


def can_build_settlement(
    board: Board,
    players: list[PlayerState],
    vertex_id: int,
) -> bool:
    """
    Return whether a vertex satisfies global
    occupancy and distance-rule constraints.
    """

    occupied = occupied_building_vertices(
        players
    )

    if vertex_id in occupied:
        return False

    for occupied_vertex in occupied:
        if (
            vertex_id
            in board.vertices[
                occupied_vertex
            ].neighbors
        ):
            return False

    return True



def canonical_edge(
    vertex_a: int,
    vertex_b: int,
) -> tuple[int, int]:
    """Return an edge in canonical sorted form."""

    return tuple(
        sorted(
            (
                vertex_a,
                vertex_b,
            )
        )
    )


def board_edges(
    board: Board,
) -> set[tuple[int, int]]:
    """Return every legal road edge on the board."""

    return {
        canonical_edge(
            edge.vertex_a,
            edge.vertex_b,
        )
        for edge in board.edges
    }


def occupied_roads(
    players: list[PlayerState],
) -> set[tuple[int, int]]:
    """Return all road edges owned by any player."""

    occupied: set[
        tuple[int, int]
    ] = set()

    for player in players:
        occupied.update(
            canonical_edge(
                a,
                b,
            )
            for a, b in player.roads
        )

    return occupied


def player_network_vertices(
    player: PlayerState,
) -> set[int]:
    """
    Return every vertex currently connected to
    the player's settlements, cities, or roads.
    """

    vertices = set(
        player.settlements
    )

    vertices.update(
        player.cities
    )

    for a, b in player.roads:
        vertices.add(a)
        vertices.add(b)

    return vertices


def opponent_building_vertices(
    players: list[PlayerState],
    player: PlayerState,
) -> set[int]:
    """
    Return vertices occupied by another player's
    settlement or city.
    """

    blocked: set[int] = set()

    for other in players:
        if other.player_id == player.player_id:
            continue

        blocked.update(
            other.settlements
        )
        blocked.update(
            other.cities
        )

    return blocked


def can_build_road(
    board: Board,
    players: list[PlayerState],
    player: PlayerState,
    vertex_a: int,
    vertex_b: int,
) -> bool:
    """
    Return whether the player may legally build
    a road on the requested edge.
    """

    edge = canonical_edge(
        vertex_a,
        vertex_b,
    )

    if len(player.roads) >= MAX_ROADS:
        return False

    if edge not in board_edges(
        board
    ):
        return False

    if edge in occupied_roads(
        players
    ):
        return False

    network = player_network_vertices(
        player
    )

    if not network:
        return False

    blocked = opponent_building_vertices(
        players,
        player,
    )

    connects_at_a = (
        vertex_a in network
        and vertex_a not in blocked
    )

    connects_at_b = (
        vertex_b in network
        and vertex_b not in blocked
    )

    if (
        not connects_at_a
        and not connects_at_b
    ):
        return False

    return True


def can_build_connected_settlement(
    board: Board,
    players: list[PlayerState],
    player: PlayerState,
    vertex_id: int,
) -> bool:
    """
    Return whether a player may build a settlement
    during normal gameplay.

    The vertex must satisfy the global distance
    rule and connect to one of the player's roads.
    """

    if len(player.settlements) >= MAX_SETTLEMENTS:
        return False

    if not can_build_settlement(
        board,
        players,
        vertex_id,
    ):
        return False

    return any(
        vertex_id in (
            vertex_a,
            vertex_b,
        )
        for vertex_a, vertex_b
        in player.roads
    )


def build_road_free(
    board: Board,
    players: list[PlayerState],
    player: PlayerState,
    vertex_a: int,
    vertex_b: int,
) -> None:
    """
    Build a road without paying resources.

    Used for effects such as the Road Building
    development card. Normal road-placement
    legality still applies.
    """

    edge = canonical_edge(
        vertex_a,
        vertex_b,
    )

    if not can_build_road(
        board,
        players,
        player,
        vertex_a,
        vertex_b,
    ):
        raise ValueError(
            "Road placement is not legal."
        )

    player.roads.append(
        edge
    )
