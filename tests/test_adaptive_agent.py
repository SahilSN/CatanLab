from catanlab.board import (
    Board,
    Edge,
    Tile,
    Vertex,
)
from catanlab.economy import (
    PlayerInventory,
)
from catanlab.graph import HexCoord
from catanlab.resources import Resource
from catanlab.simulation import PlayerState
from catanlab.strategies import StrategyType
from catanlab.turns import (
    ActionType,
    AdaptiveStrategyAgent,
)


def make_simple_board():
    return Board(
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


def test_full_ows_builds_city_when_available():
    board = make_simple_board()

    player = PlayerState(
        player_id=0,
        settlements=[0],
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

    agent = AdaptiveStrategyAgent(
        StrategyType.FULL_OWS
    )

    action = agent.choose_action(
        board,
        [player],
        player,
        inventory,
    )

    assert (
        action.action_type
        == ActionType.BUILD_CITY
    )

    assert action.vertex_id == 0


def test_road_agent_builds_road_when_available():
    board = make_simple_board()

    player = PlayerState(
        player_id=0,
        settlements=[0],
    )

    inventory = PlayerInventory()

    inventory.add(
        Resource.WOOD
    )

    inventory.add(
        Resource.BRICK
    )

    agent = AdaptiveStrategyAgent(
        StrategyType.ROAD_BUILDING
    )

    action = agent.choose_action(
        board,
        [player],
        player,
        inventory,
    )

    assert (
        action.action_type
        == ActionType.BUILD_ROAD
    )

    assert action.edge == (
        0,
        1,
    )


def test_agent_passes_when_no_action_affordable():
    board = make_simple_board()

    player = PlayerState(
        player_id=0,
        settlements=[0],
    )

    inventory = PlayerInventory()

    agent = AdaptiveStrategyAgent(
        StrategyType.FIVE_RESOURCE
    )

    action = agent.choose_action(
        board,
        [player],
        player,
        inventory,
    )

    assert (
        action.action_type
        == ActionType.PASS
    )


def test_full_ows_buys_dev_near_largest_army_win():
    board = Board(
        tiles=[],
        vertices=[],
        edges=[],
    )

    player = PlayerState(
        player_id=0,
        settlements=list(
            range(8)
        ),
        knights_played=2,
    )

    opponent = PlayerState(
        player_id=1,
        knights_played=1,
    )

    inventory = PlayerInventory()

    inventory.add(
        Resource.SHEEP
    )

    inventory.add(
        Resource.WHEAT
    )

    inventory.add(
        Resource.ORE
    )

    agent = AdaptiveStrategyAgent(
        StrategyType.FULL_OWS
    )

    action = agent.choose_action(
        board,
        [
            player,
            opponent,
        ],
        player,
        inventory,
    )

    assert (
        action.action_type
        == ActionType.BUY_DEV_CARD
    )


def test_adaptive_agent_does_not_choose_fifth_city():
    from catanlab.board import build_random_board
    from catanlab.building import MAX_CITIES
    from catanlab.economy import PlayerInventory
    from catanlab.simulation import PlayerState
    from catanlab.strategies import StrategyType
    from catanlab.turns import AdaptiveStrategyAgent

    board = build_random_board(
        seed=0
    )

    player = PlayerState(
        player_id=0,
        settlements=[0],
        cities=list(
            range(
                1,
                MAX_CITIES + 1,
            )
        ),
    )

    inventory = PlayerInventory()

    agent = AdaptiveStrategyAgent(
        StrategyType.FULL_OWS
    )

    assert (
        agent._best_city_vertex(
            board,
            player,
        )
        is None
    )


def test_maritime_trade_reports_target_build():
    from catanlab.board import build_random_board
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState
    from catanlab.strategies import StrategyType
    from catanlab.turns import AdaptiveStrategyAgent
    from catanlab.economy import BuildType

    board = build_random_board(
        seed=0
    )

    # Give the player a settlement on one endpoint
    # of a real board edge so that building a road
    # is actually legal.
    edge = board.edges[0]

    player = PlayerState(
        player_id=0,
        settlements=[
            edge.vertex_a,
        ],
    )

    inventory = PlayerInventory()

    # One card short of a road: we already have the
    # wood, and a maritime trade can supply brick.
    inventory.add(
        Resource.WOOD,
        1,
    )
    inventory.add(
        Resource.SHEEP,
        4,
    )

    agent = AdaptiveStrategyAgent(
        StrategyType.ROAD_BUILDING
    )

    from catanlab.action_scoring import (
        score_actions,
    )

    utilities = score_actions(
        StrategyType.ROAD_BUILDING,
        player,
        [player],
    ).as_dict()

    result = agent._best_maritime_trade(
        board,
        player,
        inventory,
        players=[player],
        utilities=utilities,
    )

    assert result is not None

    _, build_type = result

    assert build_type == BuildType.ROAD


def test_maritime_trade_can_progress_toward_multi_card_deficit():
    from catanlab.board import build_random_board
    from catanlab.economy import (
        BuildType,
        PlayerInventory,
    )
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState
    from catanlab.strategies import StrategyType
    from catanlab.turns import AdaptiveStrategyAgent

    board = build_random_board(
        seed=0
    )

    # Give the player an upgradable settlement so a
    # city is a legal target.
    player = PlayerState(
        player_id=0,
        settlements=[0],
    )

    inventory = PlayerInventory()

    # Two ore short of a city, with enough surplus
    # wood to make progress through maritime trade.
    inventory.add(
        Resource.WHEAT,
        2,
    )
    inventory.add(
        Resource.ORE,
        1,
    )
    inventory.add(
        Resource.WOOD,
        8,
    )

    agent = AdaptiveStrategyAgent(
        StrategyType.FULL_OWS
    )

    result = agent._best_maritime_trade(
        board,
        player,
        inventory,
    )

    assert result is not None

    action, build_type = result

    assert build_type == BuildType.CITY
    assert action.receive_resource == Resource.ORE


def test_secure_longest_road_agent_will_not_trade_for_redundant_road():
    from catanlab.action_scoring import score_actions
    from catanlab.board import build_random_board
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState
    from catanlab.strategies import StrategyType
    from catanlab.turns import (
        ActionType,
        AdaptiveStrategyAgent,
    )

    board = build_random_board(
        seed=0
    )

    player = PlayerState(
        player_id=0,
        settlements=[0],
        roads=[
            (i, i + 1)
            for i in range(8)
        ],
        has_longest_road=True,
    )

    opponent = PlayerState(
        player_id=1,
        roads=[
            (20 + i, 21 + i)
            for i in range(5)
        ],
    )

    inventory = PlayerInventory()

    inventory.add(
        Resource.WOOD,
        1,
    )
    inventory.add(
        Resource.SHEEP,
        4,
    )

    agent = AdaptiveStrategyAgent(
        StrategyType.ROAD_BUILDING
    )

    utilities = score_actions(
        StrategyType.ROAD_BUILDING,
        player,
        [player, opponent],
    ).as_dict()

    assert (
        utilities[
            ActionType.BUILD_ROAD
        ]
        < utilities[
            ActionType.PASS
        ]
    )

    result = agent._best_maritime_trade(
        board,
        player,
        inventory,
        players=[
            player,
            opponent,
        ],
        utilities=utilities,
    )

    if result is not None:
        _, build_type = result

        from catanlab.economy import BuildType

        assert build_type != BuildType.ROAD
