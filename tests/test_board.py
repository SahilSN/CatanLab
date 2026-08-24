from catanlab.board import (
    Board,
    Edge,
    Tile,
    Vertex,
    build_random_board,
    build_standard_graph,
)
from catanlab.graph import HexCoord
from catanlab.resources import Resource


def test_tile_creation():
    tile = Tile(
        id=0,
        coord=HexCoord(0, 0),
        resource=Resource.WOOD,
        number=6,
    )

    assert tile.id == 0
    assert tile.resource == Resource.WOOD
    assert tile.number == 6


def test_vertex_defaults():
    vertex = Vertex(
        id=0,
        position=(0.0, 0.0),
    )

    assert vertex.adjacent_tiles == []
    assert vertex.neighbors == []


def test_edge_creation():
    edge = Edge(
        vertex_a=1,
        vertex_b=2,
    )

    assert edge.vertex_a == 1
    assert edge.vertex_b == 2


def test_empty_board():
    board = Board(
        tiles=[],
        vertices=[],
        edges=[],
    )

    assert board.tiles == []
    assert board.vertices == []
    assert board.edges == []


def test_standard_board_tile_count():
    board = build_standard_graph()

    assert len(board.tiles) == 19


def test_standard_board_vertex_count():
    board = build_standard_graph()

    assert len(board.vertices) == 54


def test_standard_board_edge_count():
    board = build_standard_graph()

    assert len(board.edges) == 72


def test_vertex_has_at_most_three_tiles():
    board = build_standard_graph()

    assert all(
        1 <= len(vertex.adjacent_tiles) <= 3
        for vertex in board.vertices
    )


def test_vertex_has_at_most_three_neighbors():
    board = build_standard_graph()

    assert all(
        2 <= len(vertex.neighbors) <= 3
        for vertex in board.vertices
    )


def test_edges_are_unique():
    board = build_standard_graph()

    pairs = {
        tuple(
            sorted(
                (
                    edge.vertex_a,
                    edge.vertex_b,
                )
            )
        )
        for edge in board.edges
    }

    assert len(pairs) == len(
        board.edges
    )


def test_random_board_resource_counts():
    from collections import Counter

    from catanlab.resources import (
        STANDARD_RESOURCE_COUNTS,
    )

    board = build_random_board(
        seed=42
    )

    counts = Counter(
        tile.resource
        for tile in board.tiles
    )

    assert counts == (
        STANDARD_RESOURCE_COUNTS
    )


def test_random_board_has_18_number_tokens():
    board = build_random_board(
        seed=42
    )

    numbers = [
        tile.number
        for tile in board.tiles
        if tile.number is not None
    ]

    assert len(numbers) == 18


def test_desert_has_no_number():
    board = build_random_board(
        seed=42
    )

    desert_tiles = [
        tile
        for tile in board.tiles
        if tile.resource
        == Resource.DESERT
    ]

    assert len(desert_tiles) == 1
    assert desert_tiles[0].number is None


def test_random_board_number_distribution():
    from collections import Counter

    from catanlab.dice import (
        STANDARD_NUMBER_TOKENS,
    )

    board = build_random_board(
        seed=42
    )

    actual = Counter(
        tile.number
        for tile in board.tiles
        if tile.number is not None
    )

    expected = Counter(
        STANDARD_NUMBER_TOKENS
    )

    assert actual == expected


def test_random_board_is_reproducible():
    board_a = build_random_board(
        seed=123
    )

    board_b = build_random_board(
        seed=123
    )

    layout_a = [
        (
            tile.resource,
            tile.number,
        )
        for tile in board_a.tiles
    ]

    layout_b = [
        (
            tile.resource,
            tile.number,
        )
        for tile in board_b.tiles
    ]

    assert layout_a == layout_b


def test_hot_number_constraint_across_many_boards():
    from catanlab.board import (
        HOT_NUMBERS,
        tile_neighbors,
    )

    for seed in range(100):
        board = build_random_board(
            seed=seed
        )

        for tile in board.tiles:
            if tile.number not in HOT_NUMBERS:
                continue

            neighbor_numbers = [
                board.tiles[
                    neighbor_id
                ].number
                for neighbor_id
                in tile_neighbors(
                    board,
                    tile.id,
                )
            ]

            assert not any(
                number in HOT_NUMBERS
                for number
                in neighbor_numbers
            )
