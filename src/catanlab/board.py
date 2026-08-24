from dataclasses import dataclass, field
import random
from catanlab.dice import STANDARD_NUMBER_TOKENS
from catanlab.resources import (
    Resource,
    STANDARD_RESOURCE_COUNTS,
)

from catanlab.graph import (
    HexCoord,
    STANDARD_HEX_COORDS,
    hex_corners,
    are_hexes_adjacent,
)
from catanlab.resources import Resource


@dataclass
class Tile:
    id: int
    coord: HexCoord
    resource: Resource
    number: int | None


@dataclass
class Vertex:
    id: int
    position: tuple[float, float]
    adjacent_tiles: list[int] = field(
        default_factory=list
    )
    neighbors: list[int] = field(
        default_factory=list
    )


@dataclass(frozen=True)
class Edge:
    vertex_a: int
    vertex_b: int


@dataclass
class Board:
    tiles: list[Tile]
    vertices: list[Vertex]
    edges: list[Edge]


def build_standard_graph() -> Board:
    """
    Build the topology of a standard 19-hex
    Catan board.

    Resources and number tokens are placeholders
    for now. This function focuses only on
    geometry and connectivity.
    """

    tiles: list[Tile] = []

    vertices: list[Vertex] = []

    vertex_lookup: dict[
        tuple[float, float],
        int,
    ] = {}

    edge_set: set[
        tuple[int, int]
    ] = set()

    for tile_id, coord in enumerate(
        STANDARD_HEX_COORDS
    ):
        tile = Tile(
            id=tile_id,
            coord=coord,
            resource=Resource.DESERT,
            number=None,
        )

        tiles.append(tile)

        corners = hex_corners(coord)

        corner_vertex_ids = []

        for position in corners:
            if position not in vertex_lookup:
                vertex_id = len(vertices)

                vertex_lookup[position] = (
                    vertex_id
                )

                vertices.append(
                    Vertex(
                        id=vertex_id,
                        position=position,
                    )
                )

            vertex_id = vertex_lookup[
                position
            ]

            corner_vertex_ids.append(
                vertex_id
            )

            vertices[
                vertex_id
            ].adjacent_tiles.append(
                tile_id
            )

        for i in range(6):
            a = corner_vertex_ids[i]
            b = corner_vertex_ids[
                (i + 1) % 6
            ]

            edge = tuple(
                sorted((a, b))
            )

            edge_set.add(edge)

    edges = [
        Edge(
            vertex_a=a,
            vertex_b=b,
        )
        for a, b in sorted(edge_set)
    ]

    for edge in edges:
        vertices[
            edge.vertex_a
        ].neighbors.append(
            edge.vertex_b
        )

        vertices[
            edge.vertex_b
        ].neighbors.append(
            edge.vertex_a
        )

    return Board(
        tiles=tiles,
        vertices=vertices,
        edges=edges,
    )


def build_random_board(
    seed: int | None = None,
) -> Board:
    """
    Build a randomized standard Catan board.

    The board uses the standard resource counts
    and number-token distribution.

    The desert receives no number token.
    """

    rng = random.Random(seed)

    board = build_standard_graph()

    resources: list[Resource] = []

    for resource, count in (
        STANDARD_RESOURCE_COUNTS.items()
    ):
        resources.extend(
            [resource] * count
        )

    rng.shuffle(resources)

    for tile, resource in zip(
        board.tiles,
        resources,
    ):
        tile.resource = resource
        tile.number = None

    assign_balanced_numbers(
        board,
        rng,
    )

    return board


def tile_neighbors(
    board: Board,
    tile_id: int,
) -> list[int]:
    """Return tile IDs sharing an edge with a tile."""

    tile = board.tiles[tile_id]

    return [
        other.id
        for other in board.tiles
        if (
            other.id != tile.id
            and are_hexes_adjacent(
                tile.coord,
                other.coord,
            )
        )
    ]


HOT_NUMBERS = {
    6,
    8,
}


def assign_balanced_numbers(
    board: Board,
    rng: random.Random,
) -> None:
    """
    Assign standard number tokens while ensuring
    that no 6 or 8 borders another 6 or 8.
    """

    playable_tiles = [
        tile.id
        for tile in board.tiles
        if tile.resource != Resource.DESERT
    ]

    tokens = STANDARD_NUMBER_TOKENS.copy()

    assignments: dict[int, int] = {}

    # Tiles with more neighbors are slightly harder
    # to assign, so process them first.
    playable_tiles.sort(
        key=lambda tile_id: len(
            tile_neighbors(
                board,
                tile_id,
            )
        ),
        reverse=True,
    )

    # Randomize within the topology enough that the
    # same seed still determines the layout.
    rng.shuffle(playable_tiles)

    def valid(
        tile_id: int,
        number: int,
    ) -> bool:
        if number not in HOT_NUMBERS:
            return True

        for neighbor_id in tile_neighbors(
            board,
            tile_id,
        ):
            neighbor_number = assignments.get(
                neighbor_id
            )

            if neighbor_number in HOT_NUMBERS:
                return False

        return True

    def backtrack(
        index: int,
        remaining: list[int],
    ) -> bool:
        if index == len(playable_tiles):
            return True

        tile_id = playable_tiles[index]

        candidates = list(
            set(remaining)
        )

        rng.shuffle(candidates)

        # Try hot numbers first occasionally so that
        # backtracking has to solve the actual
        # constrained portion of the problem.
        candidates.sort(
            key=lambda value: (
                value not in HOT_NUMBERS
            )
        )

        for number in candidates:
            if not valid(
                tile_id,
                number,
            ):
                continue

            assignments[
                tile_id
            ] = number

            next_remaining = (
                remaining.copy()
            )

            next_remaining.remove(
                number
            )

            if backtrack(
                index + 1,
                next_remaining,
            ):
                return True

            del assignments[
                tile_id
            ]

        return False

    if not backtrack(
        0,
        tokens,
    ):
        raise RuntimeError(
            "Unable to generate balanced "
            "number-token layout."
        )

    for tile_id, number in (
        assignments.items()
    ):
        board.tiles[
            tile_id
        ].number = number

    for tile in board.tiles:
        if tile.resource == Resource.DESERT:
            tile.number = None
