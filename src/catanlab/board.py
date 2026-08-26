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


@dataclass(frozen=True)
class Port:
    vertex_a: int
    vertex_b: int
    resource: Resource | None

    @property
    def ratio(self) -> int:
        return (
            3
            if self.resource is None
            else 2
        )


@dataclass
class Board:
    tiles: list[Tile]
    vertices: list[Vertex]
    edges: list[Edge]
    robber_tile_id: int | None = None
    ports: list[Port] = field(
        default_factory=list
    )


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
        robber_tile_id=None,
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

    assign_standard_numbers(
        board
    )

    desert = next(
        tile
        for tile in board.tiles
        if tile.resource == Resource.DESERT
    )

    board.robber_tile_id = desert.id

    from catanlab.ports import (
        assign_standard_ports,
    )

    assign_standard_ports(
        board,
        seed=seed,
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


STANDARD_NUMBER_SPIRAL = [
    HexCoord(0, -2),
    HexCoord(-1, -1),
    HexCoord(-2, 0),
    HexCoord(-2, 1),
    HexCoord(-2, 2),
    HexCoord(-1, 2),
    HexCoord(0, 2),
    HexCoord(1, 1),
    HexCoord(2, 0),
    HexCoord(2, -1),
    HexCoord(2, -2),
    HexCoord(1, -2),
    HexCoord(0, -1),
    HexCoord(-1, 0),
    HexCoord(-1, 1),
    HexCoord(0, 1),
    HexCoord(1, 0),
    HexCoord(1, -1),
    HexCoord(0, 0),
]


STANDARD_NUMBER_SEQUENCE = [
    5,
    2,
    6,
    3,
    8,
    10,
    9,
    12,
    11,
    4,
    8,
    10,
    9,
    4,
    5,
    6,
    3,
    11,
]


def assign_standard_numbers(
    board: Board,
) -> None:
    """
    Assign number tokens using the standard CATAN
    counter-clockwise spiral placement.

    The sequence begins at one coastal corner and
    spirals inward. The desert is skipped, exactly
    as in the standard variable-board setup.
    """

    tile_by_coord = {
        tile.coord: tile
        for tile in board.tiles
    }

    if set(tile_by_coord) != set(
        STANDARD_NUMBER_SPIRAL
    ):
        raise ValueError(
            "Board geometry does not match the "
            "standard 19-hex CATAN layout."
        )

    token_index = 0

    for coord in STANDARD_NUMBER_SPIRAL:
        tile = tile_by_coord[coord]

        if tile.resource == Resource.DESERT:
            tile.number = None
            continue

        if token_index >= len(
            STANDARD_NUMBER_SEQUENCE
        ):
            raise RuntimeError(
                "Too many non-desert tiles for "
                "standard number-token sequence."
            )

        tile.number = (
            STANDARD_NUMBER_SEQUENCE[
                token_index
            ]
        )

        token_index += 1

    if token_index != len(
        STANDARD_NUMBER_SEQUENCE
    ):
        raise RuntimeError(
            "Standard number-token sequence was "
            "not completely assigned."
        )

