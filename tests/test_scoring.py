from catanlab.board import (
    Board,
    Tile,
    Vertex,
)
from catanlab.graph import HexCoord
from catanlab.resources import Resource
from catanlab.scoring import (
    rank_vertices,
    score_vertex,
)


def test_score_vertex():
    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=HexCoord(0, 0),
                resource=Resource.WOOD,
                number=6,
            ),
            Tile(
                id=1,
                coord=HexCoord(1, 0),
                resource=Resource.WHEAT,
                number=9,
            ),
            Tile(
                id=2,
                coord=HexCoord(0, 1),
                resource=Resource.ORE,
                number=3,
            ),
        ],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
                adjacent_tiles=[
                    0,
                    1,
                    2,
                ],
            )
        ],
        edges=[],
    )

    result = score_vertex(
        board,
        board.vertices[0],
    )

    assert result.production_score == 11
    assert result.production_probability == 11 / 36
    assert set(result.resources) == {
        Resource.WOOD,
        Resource.WHEAT,
        Resource.ORE,
    }
    assert set(result.numbers) == {
        6,
        9,
        3,
    }


def test_desert_does_not_contribute():
    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=HexCoord(0, 0),
                resource=Resource.DESERT,
                number=None,
            )
        ],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
                adjacent_tiles=[0],
            )
        ],
        edges=[],
    )

    result = score_vertex(
        board,
        board.vertices[0],
    )

    assert result.production_score == 0
    assert result.production_probability == 0
    assert result.resources == ()
    assert result.numbers == ()


def test_rank_vertices_descending():
    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=HexCoord(0, 0),
                resource=Resource.WOOD,
                number=6,
            ),
            Tile(
                id=1,
                coord=HexCoord(1, 0),
                resource=Resource.BRICK,
                number=3,
            ),
        ],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
                adjacent_tiles=[1],
            ),
            Vertex(
                id=1,
                position=(1.0, 0.0),
                adjacent_tiles=[0],
            ),
        ],
        edges=[],
    )

    ranked = rank_vertices(
        board
    )

    assert ranked[0].vertex_id == 1
    assert ranked[0].production_score == 5
    assert ranked[1].vertex_id == 0
    assert ranked[1].production_score == 2


def test_adjacent_opening_pair_is_rejected():
    from catanlab.board import Edge
    from catanlab.scoring import (
        score_opening_pair,
    )

    vertex_a = Vertex(
        id=0,
        position=(0.0, 0.0),
        neighbors=[1],
    )

    vertex_b = Vertex(
        id=1,
        position=(1.0, 0.0),
        neighbors=[0],
    )

    board = Board(
        tiles=[],
        vertices=[
            vertex_a,
            vertex_b,
        ],
        edges=[
            Edge(
                vertex_a=0,
                vertex_b=1,
            )
        ],
    )

    try:
        score_opening_pair(
            board,
            vertex_a,
            vertex_b,
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Expected adjacent pair to fail"
        )


def test_opening_pair_score():
    from catanlab.scoring import (
        score_opening_pair,
    )

    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=HexCoord(0, 0),
                resource=Resource.WOOD,
                number=6,
            ),
            Tile(
                id=1,
                coord=HexCoord(1, 0),
                resource=Resource.WHEAT,
                number=9,
            ),
            Tile(
                id=2,
                coord=HexCoord(2, 0),
                resource=Resource.ORE,
                number=5,
            ),
        ],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
                adjacent_tiles=[
                    0,
                    1,
                ],
            ),
            Vertex(
                id=1,
                position=(3.0, 0.0),
                adjacent_tiles=[
                    2,
                ],
            ),
        ],
        edges=[],
    )

    result = score_opening_pair(
        board,
        board.vertices[0],
        board.vertices[1],
    )

    assert result.production_score == 13
    assert result.unique_resources == 3
    assert result.composite_score == 17.5


def test_rank_opening_pairs_returns_results():
    from catanlab.board import (
        build_random_board,
    )
    from catanlab.scoring import (
        rank_opening_pairs,
    )

    board = build_random_board(
        seed=42
    )

    ranked = rank_opening_pairs(
        board
    )

    assert len(ranked) > 0

    assert (
        ranked[0].composite_score
        >= ranked[-1].composite_score
    )


def test_resource_production_breakdown():
    from catanlab.scoring import (
        resource_production,
    )

    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=HexCoord(0, 0),
                resource=Resource.ORE,
                number=6,
            ),
            Tile(
                id=1,
                coord=HexCoord(1, 0),
                resource=Resource.WHEAT,
                number=9,
            ),
        ],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
                adjacent_tiles=[
                    0,
                    1,
                ],
            )
        ],
        edges=[],
    )

    production = resource_production(
        board,
        board.vertices[0],
    )

    assert production[
        Resource.ORE
    ] == 5

    assert production[
        Resource.WHEAT
    ] == 4

    assert production[
        Resource.WOOD
    ] == 0


def test_specific_port_rewards_matching_production():
    from catanlab.board import Port
    from catanlab.scoring import (
        port_synergy_score,
    )

    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=HexCoord(0, 0),
                resource=Resource.ORE,
                number=6,
            )
        ],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
                adjacent_tiles=[0],
            )
        ],
        edges=[],
        ports=[
            Port(
                vertex_a=0,
                vertex_b=1,
                resource=Resource.ORE,
            )
        ],
    )

    score = port_synergy_score(
        board,
        [
            board.vertices[0],
        ],
    )

    assert score == 2.5


def test_specific_port_without_matching_production_has_no_bonus():
    from catanlab.board import Port
    from catanlab.scoring import (
        port_synergy_score,
    )

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
            )
        ],
        edges=[],
        ports=[
            Port(
                vertex_a=0,
                vertex_b=1,
                resource=Resource.ORE,
            )
        ],
    )

    assert port_synergy_score(
        board,
        [
            board.vertices[0],
        ],
    ) == 0.0


def test_generic_port_receives_smaller_production_bonus():
    from catanlab.board import Port
    from catanlab.scoring import (
        port_synergy_score,
    )

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
            )
        ],
        edges=[],
        ports=[
            Port(
                vertex_a=0,
                vertex_b=1,
                resource=None,
            )
        ],
    )

    assert port_synergy_score(
        board,
        [
            board.vertices[0],
        ],
    ) == 0.5
