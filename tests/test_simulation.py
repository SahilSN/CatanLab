from catanlab.board import build_random_board
from catanlab.simulation import (
    BalancedAgent,
    ProductionAgent,
    blocked_vertices,
    legal_vertices,
    run_opening_draft,
)


def test_occupied_vertex_and_neighbors_are_blocked():
    board = build_random_board(
        seed=42
    )

    occupied = {
        0,
    }

    blocked = blocked_vertices(
        board,
        occupied,
    )

    assert 0 in blocked

    for neighbor in board.vertices[
        0
    ].neighbors:
        assert neighbor in blocked


def test_legal_vertices_exclude_blocked():
    board = build_random_board(
        seed=42
    )

    occupied = {
        0,
    }

    legal = legal_vertices(
        board,
        occupied,
    )

    assert 0 not in legal

    for neighbor in board.vertices[
        0
    ].neighbors:
        assert neighbor not in legal


def test_opening_draft_places_eight_settlements():
    board = build_random_board(
        seed=42
    )

    agents = [
        BalancedAgent(),
        ProductionAgent(),
        BalancedAgent(),
        ProductionAgent(),
    ]

    result = run_opening_draft(
        board,
        agents,
    )

    assert len(
        result.placement_order
    ) == 8

    assert all(
        len(player.settlements) == 2
        for player in result.players
    )


def test_opening_draft_has_unique_vertices():
    board = build_random_board(
        seed=42
    )

    agents = [
        BalancedAgent(),
        BalancedAgent(),
        BalancedAgent(),
        BalancedAgent(),
    ]

    result = run_opening_draft(
        board,
        agents,
    )

    vertices = [
        vertex_id
        for _, vertex_id
        in result.placement_order
    ]

    assert len(vertices) == len(
        set(vertices)
    )


def test_opening_draft_obeys_distance_rule():
    board = build_random_board(
        seed=42
    )

    agents = [
        BalancedAgent(),
        BalancedAgent(),
        BalancedAgent(),
        BalancedAgent(),
    ]

    result = run_opening_draft(
        board,
        agents,
    )

    chosen = [
        vertex_id
        for _, vertex_id
        in result.placement_order
    ]

    for vertex_id in chosen:
        neighbors = set(
            board.vertices[
                vertex_id
            ].neighbors
        )

        assert not (
            neighbors
            & (
                set(chosen)
                - {
                    vertex_id,
                }
            )
        )


def test_opening_draft_places_eight_roads():
    board = build_random_board(
        seed=42
    )

    agents = [
        BalancedAgent(),
        BalancedAgent(),
        BalancedAgent(),
        BalancedAgent(),
    ]

    result = run_opening_draft(
        board,
        agents,
    )

    assert len(
        result.road_order
    ) == 8

    for player in result.players:
        assert len(
            player.roads
        ) == 2


def test_setup_roads_touch_corresponding_settlements():
    board = build_random_board(
        seed=42
    )

    agents = [
        BalancedAgent(),
        BalancedAgent(),
        BalancedAgent(),
        BalancedAgent(),
    ]

    result = run_opening_draft(
        board,
        agents,
    )

    for (
        placement,
        road_placement,
    ) in zip(
        result.placement_order,
        result.road_order,
    ):
        player_id, vertex_id = (
            placement
        )

        road_player_id, road = (
            road_placement
        )

        assert (
            road_player_id
            == player_id
        )

        assert vertex_id in road


def test_setup_roads_are_unique():
    board = build_random_board(
        seed=42
    )

    agents = [
        BalancedAgent(),
        BalancedAgent(),
        BalancedAgent(),
        BalancedAgent(),
    ]

    result = run_opening_draft(
        board,
        agents,
    )

    roads = [
        road
        for _, road
        in result.road_order
    ]

    assert len(
        roads
    ) == len(
        set(roads)
    )


def test_second_settlement_grants_starting_resources():
    from catanlab.board import (
        Board,
        Tile,
        Vertex,
    )
    from catanlab.economy import (
        PlayerInventory,
    )
    from catanlab.graph import HexCoord
    from catanlab.resources import Resource
    from catanlab.simulation import (
        PlayerState,
        grant_second_settlement_resources,
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
                number=8,
            ),
            Tile(
                id=2,
                coord=HexCoord(0, 1),
                resource=Resource.ORE,
                number=5,
            ),
        ],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
                adjacent_tiles=[0],
            ),
            Vertex(
                id=1,
                position=(1.0, 0.0),
                adjacent_tiles=[
                    0,
                    1,
                    2,
                ],
            ),
        ],
        edges=[],
    )

    player = PlayerState(
        player_id=0,
        settlements=[
            0,
            1,
        ],
    )

    inventory = PlayerInventory()

    grant_second_settlement_resources(
        board,
        [player],
        [inventory],
    )

    assert inventory.count(
        Resource.WOOD
    ) == 1

    assert inventory.count(
        Resource.WHEAT
    ) == 1

    assert inventory.count(
        Resource.ORE
    ) == 1

    assert inventory.total() == 3


def test_first_settlement_does_not_grant_resources():
    from catanlab.board import (
        Board,
        Tile,
        Vertex,
    )
    from catanlab.economy import (
        PlayerInventory,
    )
    from catanlab.graph import HexCoord
    from catanlab.resources import Resource
    from catanlab.simulation import (
        PlayerState,
        grant_second_settlement_resources,
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
                resource=Resource.ORE,
                number=8,
            ),
        ],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
                adjacent_tiles=[0],
            ),
            Vertex(
                id=1,
                position=(1.0, 0.0),
                adjacent_tiles=[1],
            ),
        ],
        edges=[],
    )

    player = PlayerState(
        player_id=0,
        settlements=[
            0,
            1,
        ],
    )

    inventory = PlayerInventory()

    grant_second_settlement_resources(
        board,
        [player],
        [inventory],
    )

    assert inventory.count(
        Resource.WOOD
    ) == 0

    assert inventory.count(
        Resource.ORE
    ) == 1


def test_desert_does_not_grant_starting_resource():
    from catanlab.board import (
        Board,
        Tile,
        Vertex,
    )
    from catanlab.economy import (
        PlayerInventory,
    )
    from catanlab.graph import HexCoord
    from catanlab.resources import Resource
    from catanlab.simulation import (
        PlayerState,
        grant_second_settlement_resources,
    )

    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=HexCoord(0, 0),
                resource=Resource.DESERT,
                number=None,
            ),
            Tile(
                id=1,
                coord=HexCoord(1, 0),
                resource=Resource.SHEEP,
                number=9,
            ),
        ],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
            ),
            Vertex(
                id=1,
                position=(1.0, 0.0),
                adjacent_tiles=[
                    0,
                    1,
                ],
            ),
        ],
        edges=[],
    )

    player = PlayerState(
        player_id=0,
        settlements=[
            0,
            1,
        ],
    )

    inventory = PlayerInventory()

    grant_second_settlement_resources(
        board,
        [player],
        [inventory],
    )

    assert inventory.total() == 1

    assert inventory.count(
        Resource.SHEEP
    ) == 1
