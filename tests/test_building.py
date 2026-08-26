from catanlab.board import (
    Board,
    Vertex,
)
from catanlab.building import (
    build_city,
    build_road,
    build_settlement,
)
from catanlab.economy import (
    PlayerInventory,
)
from catanlab.resources import Resource
from catanlab.simulation import PlayerState


def test_build_road_spends_resources():
    from catanlab.board import Edge

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
            ),
            Vertex(
                id=1,
                position=(1.0, 0.0),
            ),
        ],
        edges=[
            Edge(
                vertex_a=0,
                vertex_b=1,
            )
        ],
    )

    player = PlayerState(
        player_id=0,
        settlements=[0],
    )

    players = [
        player,
    ]

    inventory = PlayerInventory()

    inventory.add(
        Resource.WOOD
    )

    inventory.add(
        Resource.BRICK
    )

    build_road(
        board,
        players,
        player,
        inventory,
        0,
        1,
    )

    assert player.roads == [
        (0, 1)
    ]

    assert inventory.total() == 0


def test_build_settlement_spends_resources():
    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
            )
        ],
        edges=[],
    )

    player = PlayerState(
        player_id=0
    )

    inventory = PlayerInventory()

    for resource in (
        Resource.WOOD,
        Resource.BRICK,
        Resource.SHEEP,
        Resource.WHEAT,
    ):
        inventory.add(
            resource
        )

    build_settlement(
        board,
        [player],
        player,
        inventory,
        0,
    )

    assert player.settlements == [
        0
    ]

    assert player.victory_points == 1
    assert inventory.total() == 0


def test_build_city_upgrades_settlement():
    player = PlayerState(
        player_id=0,
        settlements=[4],
    )

    inventory = PlayerInventory()

    inventory.add(
        Resource.WHEAT,
        2,
    )

    inventory.add(
        Resource.ORE,
        3,
    )

    build_city(
        player,
        inventory,
        4,
    )

    assert 4 not in player.settlements
    assert player.cities == [
        4
    ]

    assert player.victory_points == 2
    assert inventory.total() == 0


def test_city_requires_existing_settlement():
    player = PlayerState(
        player_id=0
    )

    inventory = PlayerInventory()

    inventory.add(
        Resource.WHEAT,
        2,
    )

    inventory.add(
        Resource.ORE,
        3,
    )

    try:
        build_city(
            player,
            inventory,
            5,
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Expected invalid city upgrade to fail"
        )


def test_settlement_distance_rule():
    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
                neighbors=[1],
            ),
            Vertex(
                id=1,
                position=(1.0, 0.0),
                neighbors=[0],
            ),
        ],
        edges=[],
    )

    player = PlayerState(
        player_id=0,
        settlements=[0],
    )

    inventory = PlayerInventory()

    for resource in (
        Resource.WOOD,
        Resource.BRICK,
        Resource.SHEEP,
        Resource.WHEAT,
    ):
        inventory.add(
            resource
        )

    try:
        build_settlement(
            board,
            [player],
            player,
            inventory,
            1,
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Expected adjacent settlement "
            "to be rejected"
        )


def test_global_occupied_vertices():
    from catanlab.building import (
        occupied_building_vertices,
    )

    players = [
        PlayerState(
            player_id=0,
            settlements=[1],
            cities=[2],
        ),
        PlayerState(
            player_id=1,
            settlements=[5],
        ),
    ]

    occupied = occupied_building_vertices(
        players
    )

    assert occupied == {
        1,
        2,
        5,
    }


def test_cannot_build_on_other_players_vertex():
    from catanlab.building import (
        can_build_settlement,
    )

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
            ),
        ],
        edges=[],
    )

    players = [
        PlayerState(
            player_id=0,
            settlements=[0],
        ),
        PlayerState(
            player_id=1,
        ),
    ]

    assert not can_build_settlement(
        board,
        players,
        0,
    )


def test_cannot_build_adjacent_to_other_player():
    from catanlab.building import (
        can_build_settlement,
    )

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
                neighbors=[1],
            ),
            Vertex(
                id=1,
                position=(1.0, 0.0),
                neighbors=[0],
            ),
        ],
        edges=[],
    )

    players = [
        PlayerState(
            player_id=0,
            settlements=[0],
        ),
        PlayerState(
            player_id=1,
        ),
    ]

    assert not can_build_settlement(
        board,
        players,
        1,
    )


def test_cannot_build_road_on_nonexistent_edge():
    from catanlab.board import Edge
    from catanlab.building import can_build_road

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
            ),
            Vertex(
                id=1,
                position=(1.0, 0.0),
            ),
            Vertex(
                id=2,
                position=(2.0, 0.0),
            ),
        ],
        edges=[
            Edge(
                vertex_a=0,
                vertex_b=1,
            )
        ],
    )

    player = PlayerState(
        player_id=0,
        settlements=[0],
    )

    assert not can_build_road(
        board,
        [player],
        player,
        0,
        2,
    )


def test_cannot_build_on_occupied_road():
    from catanlab.board import Edge
    from catanlab.building import can_build_road

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
            ),
            Vertex(
                id=1,
                position=(1.0, 0.0),
            ),
        ],
        edges=[
            Edge(
                vertex_a=0,
                vertex_b=1,
            )
        ],
    )

    player_a = PlayerState(
        player_id=0,
        settlements=[0],
        roads=[
            (0, 1),
        ],
    )

    player_b = PlayerState(
        player_id=1,
        settlements=[1],
    )

    players = [
        player_a,
        player_b,
    ]

    assert not can_build_road(
        board,
        players,
        player_b,
        0,
        1,
    )


def test_road_must_connect_to_player_network():
    from catanlab.board import Edge
    from catanlab.building import can_build_road

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
            ),
            Vertex(
                id=1,
                position=(1.0, 0.0),
            ),
            Vertex(
                id=2,
                position=(2.0, 0.0),
            ),
            Vertex(
                id=3,
                position=(3.0, 0.0),
            ),
        ],
        edges=[
            Edge(
                vertex_a=0,
                vertex_b=1,
            ),
            Edge(
                vertex_a=2,
                vertex_b=3,
            ),
        ],
    )

    player = PlayerState(
        player_id=0,
        settlements=[0],
    )

    assert not can_build_road(
        board,
        [player],
        player,
        2,
        3,
    )


def test_road_can_extend_existing_road():
    from catanlab.board import Edge
    from catanlab.building import can_build_road

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
            ),
            Vertex(
                id=1,
                position=(1.0, 0.0),
            ),
            Vertex(
                id=2,
                position=(2.0, 0.0),
            ),
        ],
        edges=[
            Edge(
                vertex_a=0,
                vertex_b=1,
            ),
            Edge(
                vertex_a=1,
                vertex_b=2,
            ),
        ],
    )

    player = PlayerState(
        player_id=0,
        settlements=[0],
        roads=[
            (0, 1),
        ],
    )

    assert can_build_road(
        board,
        [player],
        player,
        1,
        2,
    )


def test_connected_settlement_requires_player_road():
    from catanlab.board import Edge
    from catanlab.building import (
        can_build_connected_settlement,
    )

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
                neighbors=[1],
            ),
            Vertex(
                id=1,
                position=(1.0, 0.0),
                neighbors=[0, 2],
            ),
            Vertex(
                id=2,
                position=(2.0, 0.0),
                neighbors=[1],
            ),
        ],
        edges=[
            Edge(
                vertex_a=0,
                vertex_b=1,
            ),
            Edge(
                vertex_a=1,
                vertex_b=2,
            ),
        ],
    )

    player = PlayerState(
        player_id=0,
        settlements=[0],
    )

    assert not can_build_connected_settlement(
        board,
        [player],
        player,
        2,
    )


def test_connected_settlement_allowed_at_road_endpoint():
    from catanlab.board import Edge
    from catanlab.building import (
        can_build_connected_settlement,
    )

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
                neighbors=[1],
            ),
            Vertex(
                id=1,
                position=(1.0, 0.0),
                neighbors=[0, 2],
            ),
            Vertex(
                id=2,
                position=(2.0, 0.0),
                neighbors=[1],
            ),
        ],
        edges=[
            Edge(
                vertex_a=0,
                vertex_b=1,
            ),
            Edge(
                vertex_a=1,
                vertex_b=2,
            ),
        ],
    )

    player = PlayerState(
        player_id=0,
        settlements=[0],
        roads=[
            (0, 1),
            (1, 2),
        ],
    )

    assert can_build_connected_settlement(
        board,
        [player],
        player,
        2,
    )


def test_build_road_free_does_not_spend_resources():
    from catanlab.board import Edge
    from catanlab.building import (
        build_road_free,
    )

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
            ),
            Vertex(
                id=1,
                position=(1.0, 0.0),
            ),
        ],
        edges=[
            Edge(
                vertex_a=0,
                vertex_b=1,
            )
        ],
    )

    player = PlayerState(
        player_id=0,
        settlements=[0],
    )

    players = [
        player,
    ]

    build_road_free(
        board,
        players,
        player,
        0,
        1,
    )

    assert player.roads == [
        (0, 1)
    ]


def test_building_fifth_road_awards_longest_road():
    from catanlab.board import Edge

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=i,
                position=(float(i), 0.0),
            )
            for i in range(6)
        ],
        edges=[
            Edge(
                vertex_a=i,
                vertex_b=i + 1,
            )
            for i in range(5)
        ],
    )

    player = PlayerState(
        player_id=0,
        settlements=[0],
        roads=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
        ],
    )

    inventory = PlayerInventory()

    inventory.add(
        Resource.WOOD
    )

    inventory.add(
        Resource.BRICK
    )

    build_road(
        board,
        [player],
        player,
        inventory,
        4,
        5,
    )

    assert player.has_longest_road
    assert player.victory_points == 3


def test_settlement_can_break_opponents_longest_road():
    from catanlab.board import Edge
    from catanlab.longest_road import (
        update_longest_road,
    )

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=i,
                position=(float(i), 0.0),
                neighbors=(
                    ([i - 1] if i > 0 else [])
                    + ([i + 1] if i < 5 else [])
                ),
            )
            for i in range(6)
        ],
        edges=[
            Edge(
                vertex_a=i,
                vertex_b=i + 1,
            )
            for i in range(5)
        ],
    )

    player_a = PlayerState(
        player_id=0,
        roads=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
        ],
    )

    player_b = PlayerState(
        player_id=1,
    )

    players = [
        player_a,
        player_b,
    ]

    update_longest_road(
        players
    )

    assert player_a.has_longest_road

    inventory = PlayerInventory()

    for resource in (
        Resource.WOOD,
        Resource.BRICK,
        Resource.SHEEP,
        Resource.WHEAT,
    ):
        inventory.add(
            resource
        )

    build_settlement(
        board,
        players,
        player_b,
        inventory,
        3,
    )

    assert not player_a.has_longest_road


def test_cannot_build_more_than_fifteen_roads():
    from catanlab.board import Edge
    from catanlab.building import (
        MAX_ROADS,
        can_build_road,
    )

    player = PlayerState(
        player_id=0,
        settlements=[0],
        roads=[
            (i, i + 1)
            for i in range(
                MAX_ROADS
            )
        ],
    )

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=i,
                position=(float(i), 0.0),
            )
            for i in range(
                MAX_ROADS + 2
            )
        ],
        edges=[
            Edge(
                vertex_a=i,
                vertex_b=i + 1,
            )
            for i in range(
                MAX_ROADS + 1
            )
        ],
    )

    assert not can_build_road(
        board,
        [player],
        player,
        MAX_ROADS,
        MAX_ROADS + 1,
    )


def test_cannot_build_sixth_settlement():
    from catanlab.board import Edge
    from catanlab.building import (
        MAX_SETTLEMENTS,
        can_build_connected_settlement,
    )

    player = PlayerState(
        player_id=0,
        settlements=list(
            range(
                MAX_SETTLEMENTS
            )
        ),
        roads=[
            (10, 11),
        ],
    )

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=i,
                position=(float(i), 0.0),
                neighbors=[],
            )
            for i in range(12)
        ],
        edges=[
            Edge(
                vertex_a=10,
                vertex_b=11,
            )
        ],
    )

    assert not can_build_connected_settlement(
        board,
        [player],
        player,
        11,
    )


def test_city_limit_is_four():
    from catanlab.building import MAX_CITIES

    player = PlayerState(
        player_id=0,
        settlements=[10],
        cities=[
            0,
            1,
            2,
            3,
        ],
    )

    inventory = PlayerInventory()

    inventory.add(
        Resource.WHEAT,
        2,
    )

    inventory.add(
        Resource.ORE,
        3,
    )

    try:
        build_city(
            player,
            inventory,
            10,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected fifth city to be rejected."
        )


def test_city_upgrade_returns_settlement_piece():
    player = PlayerState(
        player_id=0,
        settlements=[
            0,
            1,
            2,
            3,
            4,
        ],
    )

    inventory = PlayerInventory()

    inventory.add(
        Resource.WHEAT,
        2,
    )

    inventory.add(
        Resource.ORE,
        3,
    )

    build_city(
        player,
        inventory,
        0,
    )

    assert len(
        player.settlements
    ) == 4

    assert len(
        player.cities
    ) == 1


def test_cannot_build_road_through_opponent_settlement():
    from catanlab.board import Board, Edge, Vertex
    from catanlab.building import can_build_road

    player = PlayerState(
        player_id=0,
        settlements=[0],
        roads=[
            (0, 1),
        ],
    )

    opponent = PlayerState(
        player_id=1,
        settlements=[1],
    )

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
            ),
            Vertex(
                id=1,
                position=(1.0, 0.0),
            ),
            Vertex(
                id=2,
                position=(2.0, 0.0),
            ),
        ],
        edges=[
            Edge(
                vertex_a=0,
                vertex_b=1,
            ),
            Edge(
                vertex_a=1,
                vertex_b=2,
            ),
        ],
    )

    assert not can_build_road(
        board,
        [player, opponent],
        player,
        1,
        2,
    )


def test_can_build_road_through_own_settlement():
    from catanlab.board import Board, Edge, Vertex
    from catanlab.building import can_build_road

    player = PlayerState(
        player_id=0,
        settlements=[0, 1],
        roads=[
            (0, 1),
        ],
    )

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
            ),
            Vertex(
                id=1,
                position=(1.0, 0.0),
            ),
            Vertex(
                id=2,
                position=(2.0, 0.0),
            ),
        ],
        edges=[
            Edge(
                vertex_a=0,
                vertex_b=1,
            ),
            Edge(
                vertex_a=1,
                vertex_b=2,
            ),
        ],
    )

    assert can_build_road(
        board,
        [player],
        player,
        1,
        2,
    )


def test_can_extend_road_through_empty_vertex():
    from catanlab.board import Board, Edge, Vertex
    from catanlab.building import can_build_road

    player = PlayerState(
        player_id=0,
        settlements=[0],
        roads=[
            (0, 1),
        ],
    )

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
            ),
            Vertex(
                id=1,
                position=(1.0, 0.0),
            ),
            Vertex(
                id=2,
                position=(2.0, 0.0),
            ),
        ],
        edges=[
            Edge(
                vertex_a=0,
                vertex_b=1,
            ),
            Edge(
                vertex_a=1,
                vertex_b=2,
            ),
        ],
    )

    assert can_build_road(
        board,
        [player],
        player,
        1,
        2,
    )
