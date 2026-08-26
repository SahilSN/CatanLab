from catanlab.board import (
    Board,
    Port,
    Tile,
    Vertex,
)
from catanlab.graph import HexCoord
from catanlab.resources import Resource
from catanlab.simulation import (
    PlayerState,
    StrategyOpeningAgent,
)
from catanlab.strategies import (
    StrategyType,
)


def test_full_ows_prefers_ore_over_wood():
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
                resource=Resource.WOOD,
                number=6,
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
                position=(2.0, 0.0),
                adjacent_tiles=[1],
            ),
        ],
        edges=[],
    )

    agent = StrategyOpeningAgent(
        StrategyType.FULL_OWS
    )

    player = PlayerState(
        player_id=0
    )

    chosen = agent.choose_vertex(
        board,
        [0, 1],
        player,
    )

    assert chosen == 0


def test_road_strategy_prefers_wood_over_ore():
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
                resource=Resource.WOOD,
                number=6,
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
                position=(2.0, 0.0),
                adjacent_tiles=[1],
            ),
        ],
        edges=[],
    )

    agent = StrategyOpeningAgent(
        StrategyType.ROAD_BUILDING
    )

    player = PlayerState(
        player_id=0
    )

    chosen = agent.choose_vertex(
        board,
        [0, 1],
        player,
    )

    assert chosen == 1


def test_port_strategy_values_matching_port():
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
                resource=Resource.ORE,
                number=6,
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
                position=(2.0, 0.0),
                adjacent_tiles=[1],
            ),
        ],
        edges=[],
        ports=[
            Port(
                vertex_a=1,
                vertex_b=2,
                resource=Resource.ORE,
            )
        ],
    )

    agent = StrategyOpeningAgent(
        StrategyType.PORT
    )

    player = PlayerState(
        player_id=0
    )

    chosen = agent.choose_vertex(
        board,
        [0, 1],
        player,
    )

    assert chosen == 1


def test_strategy_setup_road_is_deterministic():
    from catanlab.board import build_random_board
    from catanlab.simulation import (
        StrategyOpeningAgent,
        choose_setup_road,
    )
    from catanlab.strategies import StrategyType

    board = build_random_board(
        seed=42
    )

    agent = StrategyOpeningAgent(
        StrategyType.ROAD_BUILDING
    )

    settlement_vertex = 0

    first = choose_setup_road(
        board,
        settlement_vertex,
        set(),
        agent=agent,
        occupied_vertices={
            settlement_vertex,
        },
    )

    second = choose_setup_road(
        board,
        settlement_vertex,
        set(),
        agent=agent,
        occupied_vertices={
            settlement_vertex,
        },
    )

    assert first == second


def test_generic_setup_road_api_still_works():
    from catanlab.board import build_random_board
    from catanlab.simulation import (
        choose_setup_road,
    )

    board = build_random_board(
        seed=42
    )

    road = choose_setup_road(
        board,
        0,
        set(),
    )

    assert 0 in road
