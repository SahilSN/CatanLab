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
from catanlab.turns import (
    ActionType,
    GreedyBuildAgent,
    TurnAction,
    execute_action,
    legal_road_edges,
    run_turn,
)


def test_greedy_agent_passes_when_broke():
    board = Board(
        tiles=[],
        vertices=[],
        edges=[],
    )

    player = PlayerState(
        player_id=0
    )

    inventory = PlayerInventory()

    agent = GreedyBuildAgent()

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


def test_execute_city_action():
    board = Board(
        tiles=[],
        vertices=[],
        edges=[],
    )

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

    execute_action(
        board,
        [player],
        player,
        inventory,
        TurnAction(
            action_type=(
                ActionType.BUILD_CITY
            ),
            vertex_id=4,
        ),
    )

    assert player.settlements == []
    assert player.cities == [4]
    assert player.victory_points == 2


def test_legal_road_edges():
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
    )

    legal = legal_road_edges(
        board,
        [player],
        player,
    )

    assert (0, 1) in legal
    assert (1, 2) not in legal


def test_run_turn_produces_then_passes():
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
    )

    players = [
        PlayerState(
            player_id=0,
            settlements=[0],
        )
    ]

    inventories = [
        PlayerInventory()
    ]

    agents = [
        GreedyBuildAgent()
    ]

    result = run_turn(
        board,
        players,
        inventories,
        agents,
        player_id=0,
        roll=6,
    )

    assert inventories[0].count(
        Resource.WOOD
    ) == 1

    assert (
        result.action.action_type
        == ActionType.PASS
    )


def test_legal_settlement_requires_connected_road():
    from catanlab.turns import (
        legal_settlement_vertices,
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

    legal = legal_settlement_vertices(
        board,
        [player],
        player,
    )

    assert 2 in legal


def test_execute_buy_dev_card():
    from catanlab.devcards import (
        build_dev_card_deck,
    )

    board = Board(
        tiles=[],
        vertices=[],
        edges=[],
    )

    player = PlayerState(
        player_id=0
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

    deck = build_dev_card_deck(
        seed=42
    )

    execute_action(
        board,
        [player],
        player,
        inventory,
        TurnAction(
            action_type=(
                ActionType.BUY_DEV_CARD
            )
        ),
        dev_deck=deck,
    )

    assert len(
        player.dev_cards
    ) == 1

    assert len(
        deck.cards
    ) == 24

    assert inventory.total() == 0


def test_run_turn_seven_discards_large_hand():
    import random

    board = Board(
        tiles=[],
        vertices=[],
        edges=[],
    )

    players = [
        PlayerState(
            player_id=0
        )
    ]

    inventory = PlayerInventory()

    inventory.add(
        Resource.WOOD,
        8,
    )

    inventories = [
        inventory
    ]

    agents = [
        GreedyBuildAgent()
    ]

    result = run_turn(
        board,
        players,
        inventories,
        agents,
        player_id=0,
        roll=7,
        rng=random.Random(42),
    )

    assert inventories[0].total() == 4
    assert len(
        result.discards[0]
    ) == 4


def test_run_turn_seven_does_not_produce():
    import random

    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=HexCoord(0, 0),
                resource=Resource.WOOD,
                number=7,
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

    players = [
        PlayerState(
            player_id=0,
            settlements=[0],
        )
    ]

    inventories = [
        PlayerInventory()
    ]

    agents = [
        GreedyBuildAgent()
    ]

    result = run_turn(
        board,
        players,
        inventories,
        agents,
        player_id=0,
        roll=7,
        rng=random.Random(42),
    )

    assert inventories[0].total() == 0
    assert result.discards == {}


def test_run_turn_records_no_discards_on_normal_roll():
    board = Board(
        tiles=[],
        vertices=[],
        edges=[],
    )

    players = [
        PlayerState(
            player_id=0
        )
    ]

    inventories = [
        PlayerInventory()
    ]

    agents = [
        GreedyBuildAgent()
    ]

    result = run_turn(
        board,
        players,
        inventories,
        agents,
        player_id=0,
        roll=5,
    )

    assert result.discards == {}


def test_turn_offers_pre_and_post_roll_dev_windows():
    from catanlab.devcard_policy import (
        DevCardDecision,
        DevCardPhase,
    )
    from catanlab.simulation import PlayerState
    from catanlab.turns import (
        ActionType,
        TurnAction,
        TurnAgent,
        run_turn,
    )

    class RecordingAgent(TurnAgent):
        def __init__(self):
            self.phases = []

        def choose_dev_card_play(
            self,
            board,
            players,
            player,
            inventories,
            phase,
        ):
            self.phases.append(
                phase
            )

            return DevCardDecision(
                card=None,
                utility=0.0,
            )

        def choose_action(
            self,
            board,
            players,
            player,
            inventory,
        ):
            return TurnAction(
                action_type=(
                    ActionType.PASS
                )
            )

    board = Board(
        tiles=[],
        vertices=[],
        edges=[],
    )

    player = PlayerState(
        player_id=0
    )

    inventory = PlayerInventory()

    agent = RecordingAgent()

    run_turn(
        board,
        [player],
        [inventory],
        [agent],
        player_id=0,
        roll=2,
    )

    assert agent.phases == [
        DevCardPhase.PRE_ROLL,
        DevCardPhase.POST_ROLL,
    ]


def test_pre_roll_dev_card_blocks_post_roll_dev_card():
    from catanlab.board import (
        Tile,
        Vertex,
    )
    from catanlab.devcard_policy import (
        DevCardDecision,
        DevCardPhase,
    )
    from catanlab.devcards import (
        DevCardType,
    )
    from catanlab.graph import HexCoord
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState
    from catanlab.turns import (
        ActionType,
        TurnAction,
        TurnAgent,
        run_turn,
    )

    class KnightAgent(TurnAgent):
        def __init__(self):
            self.phases = []

        def choose_dev_card_play(
            self,
            board,
            players,
            player,
            inventories,
            phase,
        ):
            self.phases.append(
                phase
            )

            return DevCardDecision(
                card=DevCardType.KNIGHT,
                utility=10.0,
            )

        def choose_action(
            self,
            board,
            players,
            player,
            inventory,
        ):
            return TurnAction(
                action_type=(
                    ActionType.PASS
                )
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
                resource=Resource.BRICK,
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
        robber_tile_id=0,
    )

    player = PlayerState(
        player_id=0,
        dev_cards=[
            DevCardType.KNIGHT.value,
            DevCardType.KNIGHT.value,
        ],
    )

    inventory = PlayerInventory()

    agent = KnightAgent()

    run_turn(
        board,
        [player],
        [inventory],
        [agent],
        player_id=0,
        roll=2,
    )

    assert agent.phases == [
        DevCardPhase.PRE_ROLL,
    ]

    assert player.knights_played == 1

    assert player.dev_cards == [
        DevCardType.KNIGHT.value,
    ]


def test_previous_turn_new_card_becomes_playable():
    from catanlab.board import (
        Tile,
    )
    from catanlab.devcard_policy import (
        DevCardDecision,
    )
    from catanlab.devcards import (
        DevCardType,
    )
    from catanlab.graph import HexCoord
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState
    from catanlab.turns import (
        ActionType,
        TurnAction,
        TurnAgent,
        run_turn,
    )

    class KnightAgent(TurnAgent):
        def choose_dev_card_play(
            self,
            board,
            players,
            player,
            inventories,
            phase,
        ):
            return DevCardDecision(
                card=DevCardType.KNIGHT,
                utility=10.0,
            )

        def choose_action(
            self,
            board,
            players,
            player,
            inventory,
        ):
            return TurnAction(
                action_type=(
                    ActionType.PASS
                )
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
        vertices=[],
        edges=[],
        robber_tile_id=0,
    )

    player = PlayerState(
        player_id=0,
        dev_cards=[
            DevCardType.KNIGHT.value,
        ],
        new_dev_cards=[
            DevCardType.KNIGHT.value,
        ],
    )

    run_turn(
        board,
        [player],
        [PlayerInventory()],
        [KnightAgent()],
        player_id=0,
        roll=2,
    )

    assert player.knights_played == 1
    assert player.dev_cards == []
    assert player.new_dev_cards == []


def test_run_turn_seven_moves_robber():
    from catanlab.board import Tile
    from catanlab.graph import HexCoord
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState
    from catanlab.turns import (
        GreedyBuildAgent,
        run_turn,
    )

    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=HexCoord(0, 0),
                resource=Resource.WOOD,
                number=2,
            ),
            Tile(
                id=1,
                coord=HexCoord(1, 0),
                resource=Resource.ORE,
                number=6,
            ),
        ],
        vertices=[],
        edges=[],
        robber_tile_id=0,
    )

    player = PlayerState(
        player_id=0
    )

    run_turn(
        board,
        [player],
        [PlayerInventory()],
        [GreedyBuildAgent()],
        player_id=0,
        roll=7,
    )

    assert board.robber_tile_id == 1


def test_run_turn_seven_steals_from_adjacent_opponent():
    import random

    from catanlab.board import (
        Tile,
        Vertex,
    )
    from catanlab.graph import HexCoord
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState
    from catanlab.turns import (
        GreedyBuildAgent,
        run_turn,
    )

    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=HexCoord(0, 0),
                resource=Resource.WOOD,
                number=2,
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
                position=(1.0, 0.0),
                adjacent_tiles=[1],
            ),
        ],
        edges=[],
        robber_tile_id=0,
    )

    player_a = PlayerState(
        player_id=0
    )

    player_b = PlayerState(
        player_id=1,
        settlements=[1],
    )

    inventory_a = PlayerInventory()

    inventory_b = PlayerInventory()
    inventory_b.add(
        Resource.WHEAT
    )

    run_turn(
        board,
        [
            player_a,
            player_b,
        ],
        [
            inventory_a,
            inventory_b,
        ],
        [
            GreedyBuildAgent(),
            GreedyBuildAgent(),
        ],
        player_id=0,
        roll=7,
        rng=random.Random(0),
    )

    assert board.robber_tile_id == 1

    assert inventory_a.count(
        Resource.WHEAT
    ) == 1

    assert inventory_b.count(
        Resource.WHEAT
    ) == 0


def test_run_turn_seven_discards_before_robber_steal():
    import random

    from catanlab.board import (
        Tile,
        Vertex,
    )
    from catanlab.graph import HexCoord
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState
    from catanlab.turns import (
        GreedyBuildAgent,
        run_turn,
    )

    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=HexCoord(0, 0),
                resource=Resource.WOOD,
                number=2,
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
                adjacent_tiles=[1],
            ),
        ],
        edges=[],
        robber_tile_id=0,
    )

    player_a = PlayerState(
        player_id=0
    )

    player_b = PlayerState(
        player_id=1,
        settlements=[0],
    )

    inventory_a = PlayerInventory()
    inventory_b = PlayerInventory()

    inventory_b.add(
        Resource.WHEAT,
        8,
    )

    result = run_turn(
        board,
        [
            player_a,
            player_b,
        ],
        [
            inventory_a,
            inventory_b,
        ],
        [
            GreedyBuildAgent(),
            GreedyBuildAgent(),
        ],
        player_id=0,
        roll=7,
        rng=random.Random(0),
    )

    # Eight cards -> discard four first.
    assert len(
        result.discards[1]
    ) == 4

    # Then one of the remaining four is stolen.
    assert inventory_b.total() == 3
    assert inventory_a.total() == 1


def test_run_turn_allows_multiple_normal_actions():
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState
    from catanlab.turns import (
        ActionType,
        TurnAction,
        TurnAgent,
        run_turn,
    )

    class MultiActionAgent(TurnAgent):
        def __init__(self):
            self.actions = [
                TurnAction(
                    action_type=(
                        ActionType.MARITIME_TRADE
                    ),
                    give_resource=Resource.WOOD,
                    receive_resource=Resource.ORE,
                ),
                TurnAction(
                    action_type=(
                        ActionType.PASS
                    )
                ),
            ]

        def choose_action(
            self,
            board,
            players,
            player,
            inventory,
        ):
            return self.actions.pop(
                0
            )

    board = Board(
        tiles=[],
        vertices=[],
        edges=[],
    )

    player = PlayerState(
        player_id=0
    )

    inventory = PlayerInventory()

    inventory.add(
        Resource.WOOD,
        4,
    )

    result = run_turn(
        board,
        [player],
        [inventory],
        [MultiActionAgent()],
        player_id=0,
        roll=2,
    )

    assert inventory.count(
        Resource.WOOD
    ) == 0

    assert inventory.count(
        Resource.ORE
    ) == 1

    assert (
        result.action.action_type
        == ActionType.PASS
    )


def test_run_turn_can_trade_then_buy_dev_card():
    from catanlab.devcards import (
        DevCardDeck,
        DevCardType,
    )
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState
    from catanlab.turns import (
        ActionType,
        TurnAction,
        TurnAgent,
        run_turn,
    )

    class TradeThenBuyAgent(TurnAgent):
        def __init__(self):
            self.actions = [
                TurnAction(
                    action_type=(
                        ActionType.MARITIME_TRADE
                    ),
                    give_resource=Resource.WOOD,
                    receive_resource=Resource.ORE,
                ),
                TurnAction(
                    action_type=(
                        ActionType.BUY_DEV_CARD
                    )
                ),
                TurnAction(
                    action_type=(
                        ActionType.PASS
                    )
                ),
            ]

        def choose_action(
            self,
            board,
            players,
            player,
            inventory,
        ):
            return self.actions.pop(
                0
            )

    board = Board(
        tiles=[],
        vertices=[],
        edges=[],
    )

    player = PlayerState(
        player_id=0
    )

    inventory = PlayerInventory()

    inventory.add(
        Resource.WOOD,
        4,
    )
    inventory.add(
        Resource.SHEEP,
    )
    inventory.add(
        Resource.WHEAT,
    )

    deck = DevCardDeck(
        cards=[
            DevCardType.KNIGHT,
        ]
    )

    run_turn(
        board,
        [player],
        [inventory],
        [TradeThenBuyAgent()],
        player_id=0,
        roll=2,
        dev_deck=deck,
    )

    assert inventory.count(
        Resource.WOOD
    ) == 0

    assert inventory.count(
        Resource.ORE
    ) == 0

    assert inventory.count(
        Resource.SHEEP
    ) == 0

    assert inventory.count(
        Resource.WHEAT
    ) == 0

    assert player.dev_cards == [
        DevCardType.KNIGHT.value,
    ]

    assert player.new_dev_cards == [
        DevCardType.KNIGHT.value,
    ]


def test_turn_result_records_all_normal_actions():
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState
    from catanlab.turns import (
        ActionType,
        TurnAction,
        TurnAgent,
        run_turn,
    )

    class MultiActionAgent(TurnAgent):
        def __init__(self):
            self.pending = [
                TurnAction(
                    action_type=(
                        ActionType.MARITIME_TRADE
                    ),
                    give_resource=Resource.WOOD,
                    receive_resource=Resource.ORE,
                ),
                TurnAction(
                    action_type=(
                        ActionType.PASS
                    )
                ),
            ]

        def choose_action(
            self,
            board,
            players,
            player,
            inventory,
        ):
            return self.pending.pop(
                0
            )

    board = Board(
        tiles=[],
        vertices=[],
        edges=[],
    )

    player = PlayerState(
        player_id=0
    )

    inventory = PlayerInventory()

    inventory.add(
        Resource.WOOD,
        4,
    )

    result = run_turn(
        board,
        [player],
        [inventory],
        [MultiActionAgent()],
        player_id=0,
        roll=2,
    )

    assert len(
        result.actions
    ) == 2

    assert (
        result.actions[0].action_type
        == ActionType.MARITIME_TRADE
    )

    assert (
        result.actions[1].action_type
        == ActionType.PASS
    )

    # Compatibility field still points to the
    # final action.
    assert (
        result.action
        == result.actions[-1]
    )


def test_trade_sequence_allows_four_counteroffers():
    from catanlab.board import Board
    from catanlab.economy import (
        PlayerInventory,
    )
    from catanlab.resources import (
        Resource,
    )
    from catanlab.simulation import (
        PlayerState,
    )
    from catanlab.trading import (
        TradeOffer,
        make_bundle,
    )
    from catanlab.turns import (
        TurnAgent,
        _run_trade_sequence,
    )

    board = Board(
        tiles=[],
        vertices=[],
        edges=[],
    )

    players = [
        PlayerState(
            player_id=0
        ),
        PlayerState(
            player_id=1
        ),
    ]

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    inventories[0].add(
        Resource.WOOD,
        4,
    )

    inventories[1].add(
        Resource.ORE,
        4,
    )

    class Negotiator(TurnAgent):
        def __init__(
            self,
            accept_on,
        ):
            self.responses = 0
            self.accept_on = (
                accept_on
            )

        def choose_action(
            self,
            board,
            players,
            player,
            inventory,
            dev_deck=None,
        ):
            raise NotImplementedError

        def evaluate_player_trade(
            self,
            board,
            players,
            player,
            inventories,
            offer,
        ):
            self.responses += 1

            return (
                self.responses
                >= self.accept_on
            )

        def counter_player_trade(
            self,
            board,
            players,
            player,
            inventories,
            offer,
            attempted_offers=None,
        ):
            # Change the quantity so each offer is
            # distinct while remaining legal.
            amount = (
                len(
                    attempted_offers
                    or ()
                )
                % 4
                + 1
            )

            if (
                player.player_id
                == 0
            ):
                return TradeOffer(
                    proposer_id=0,
                    recipient_id=1,
                    give=make_bundle(
                        (
                            Resource.WOOD,
                            amount,
                        ),
                    ),
                    receive=make_bundle(
                        (
                            Resource.ORE,
                            1,
                        ),
                    ),
                )

            return TradeOffer(
                proposer_id=1,
                recipient_id=0,
                give=make_bundle(
                    (
                        Resource.ORE,
                        1,
                    ),
                ),
                receive=make_bundle(
                    (
                        Resource.WOOD,
                        amount,
                    ),
                ),
            )

    agents = [
        Negotiator(
            accept_on=99
        ),
        Negotiator(
            accept_on=99
        ),
    ]

    initial = TradeOffer(
        proposer_id=0,
        recipient_id=1,
        give=make_bundle(
            (
                Resource.WOOD,
                1,
            ),
        ),
        receive=make_bundle(
            (
                Resource.ORE,
                1,
            ),
        ),
    )

    accepted, offers = (
        _run_trade_sequence(
            board,
            players,
            inventories,
            agents,
            initial,
            remaining_offer_budget=12,
        )
    )

    assert accepted is None
    assert offers == 4


def test_domestic_trading_is_interleaved_with_build_actions():
    from catanlab.board import (
        Board,
        Edge,
        Vertex,
    )
    from catanlab.economy import (
        PlayerInventory,
    )
    from catanlab.resources import (
        Resource,
    )
    from catanlab.simulation import (
        PlayerState,
    )
    from catanlab.trading import (
        TradeOffer,
        make_bundle,
    )
    from catanlab.turns import (
        ActionType,
        TurnAction,
        TurnAgent,
        run_turn,
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
        edges=[
            Edge(
                vertex_a=0,
                vertex_b=1,
            ),
        ],
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

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    # Active player has wood but needs brick
    # for the road.
    inventories[0].add(
        Resource.WOOD,
        2,
    )

    inventories[0].add(
        Resource.SHEEP,
        1,
    )

    inventories[1].add(
        Resource.BRICK,
        2,
    )

    inventories[1].add(
        Resource.ORE,
        1,
    )

    class ActiveAgent(TurnAgent):
        def __init__(self):
            self.trade_calls = 0
            self.action_calls = 0

        def propose_player_trade(
            self,
            board,
            players,
            player,
            inventories,
            excluded_recipients=None,
        ):
            self.trade_calls += 1

            if self.trade_calls == 1:
                return TradeOffer(
                    proposer_id=0,
                    recipient_id=1,
                    give=make_bundle(
                        (
                            Resource.SHEEP,
                            1,
                        ),
                    ),
                    receive=make_bundle(
                        (
                            Resource.BRICK,
                            1,
                        ),
                    ),
                )

            return None

        def choose_action(
            self,
            board,
            players,
            player,
            inventory,
            dev_deck=None,
        ):
            self.action_calls += 1

            if self.action_calls == 1:
                return TurnAction(
                    action_type=(
                        ActionType.BUILD_ROAD
                    ),
                    edge=(
                        0,
                        1,
                    ),
                )

            return TurnAction(
                action_type=(
                    ActionType.PASS
                )
            )

    class RecipientAgent(TurnAgent):
        def choose_action(
            self,
            board,
            players,
            player,
            inventory,
            dev_deck=None,
        ):
            return TurnAction(
                action_type=(
                    ActionType.PASS
                )
            )

        def evaluate_player_trade(
            self,
            board,
            players,
            player,
            inventories,
            offer,
        ):
            return True

    active_agent = ActiveAgent()

    result = run_turn(
        board,
        players,
        inventories,
        [
            active_agent,
            RecipientAgent(),
        ],
        player_id=0,
        roll=2,
    )

    assert players[0].roads == [
        (
            0,
            1,
        )
    ]

    assert len(
        result.player_trades
    ) == 1

    assert (
        result.trade_sequence_count
        == 1
    )

    assert (
        result.trade_offer_count
        == 1
    )

    # The key assertion: after the road build,
    # the normal loop reached another iteration and
    # therefore offered another chance to negotiate.
    assert (
        active_agent.trade_calls
        >= 2
    )


def test_failed_trade_is_not_repeated_without_state_change():
    from catanlab.board import Board
    from catanlab.economy import (
        PlayerInventory,
    )
    from catanlab.resources import (
        Resource,
    )
    from catanlab.simulation import (
        PlayerState,
    )
    from catanlab.trading import (
        TradeOffer,
        make_bundle,
    )
    from catanlab.turns import (
        ActionType,
        TurnAction,
        TurnAgent,
        run_turn,
    )

    board = Board(
        tiles=[],
        vertices=[],
        edges=[],
    )

    players = [
        PlayerState(
            player_id=0
        ),
        PlayerState(
            player_id=1
        ),
    ]

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    inventories[0].add(
        Resource.WOOD,
        1,
    )

    inventories[1].add(
        Resource.ORE,
        1,
    )

    class ActiveAgent(TurnAgent):
        def __init__(self):
            self.proposals = 0
            self.actions = [
                TurnAction(
                    action_type=(
                        ActionType.PASS
                    )
                )
            ]

        def propose_player_trade(
            self,
            board,
            players,
            player,
            inventories,
            excluded_recipients=None,
            agents=None,
        ):
            self.proposals += 1

            return TradeOffer(
                proposer_id=0,
                recipient_id=1,
                give=make_bundle(
                    (
                        Resource.WOOD,
                        1,
                    ),
                ),
                receive=make_bundle(
                    (
                        Resource.ORE,
                        1,
                    ),
                ),
            )

        def choose_action(
            self,
            board,
            players,
            player,
            inventory,
            dev_deck=None,
        ):
            return self.actions.pop(
                0
            )

    class RejectAgent(TurnAgent):
        def choose_action(
            self,
            board,
            players,
            player,
            inventory,
            dev_deck=None,
        ):
            return TurnAction(
                action_type=(
                    ActionType.PASS
                )
            )

        def evaluate_player_trade(
            self,
            board,
            players,
            player,
            inventories,
            offer,
        ):
            return False

        def counter_player_trade(
            self,
            board,
            players,
            player,
            inventories,
            offer,
            attempted_offers=None,
        ):
            return None

    active = ActiveAgent()

    result = run_turn(
        board,
        players,
        inventories,
        [
            active,
            RejectAgent(),
        ],
        player_id=0,
        roll=2,
    )

    assert result.trade_sequence_count == 1
    assert result.trade_offer_count == 1
    assert len(
        result.player_trades
    ) == 0

    assert active.proposals == 1


def test_roll_seven_discards_exactly_half_of_large_hand():
    from catanlab.board import Board
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState
    from catanlab.turns import (
        ActionType,
        TurnAction,
        TurnAgent,
        run_turn,
    )

    board = Board(
        tiles=[],
        vertices=[],
        edges=[],
    )

    players = [
        PlayerState(player_id=0),
        PlayerState(player_id=1),
    ]

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    for _ in range(8):
        inventories[1].add(
            Resource.WOOD,
            1,
        )

    class PassAgent(TurnAgent):
        def choose_action(
            self,
            board,
            players,
            player,
            inventory,
            dev_deck=None,
        ):
            return TurnAction(
                action_type=ActionType.PASS
            )

    agents = [
        PassAgent(),
        PassAgent(),
    ]

    result = run_turn(
        board,
        players,
        inventories,
        agents,
        player_id=0,
        roll=7,
    )

    assert len(
        result.discards[1]
    ) == 4

    assert (
        inventories[1].total()
        == 4
    )


def test_roll_seven_does_not_discard_with_seven_cards():
    from catanlab.board import Board
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState
    from catanlab.turns import (
        ActionType,
        TurnAction,
        TurnAgent,
        run_turn,
    )

    board = Board(
        tiles=[],
        vertices=[],
        edges=[],
    )

    players = [
        PlayerState(player_id=0),
        PlayerState(player_id=1),
    ]

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    for _ in range(7):
        inventories[1].add(
            Resource.WHEAT,
            1,
        )

    class PassAgent(TurnAgent):
        def choose_action(
            self,
            board,
            players,
            player,
            inventory,
            dev_deck=None,
        ):
            return TurnAction(
                action_type=ActionType.PASS
            )

    result = run_turn(
        board,
        players,
        inventories,
        [
            PassAgent(),
            PassAgent(),
        ],
        player_id=0,
        roll=7,
    )

    assert 1 not in result.discards
    assert inventories[1].total() == 7


def test_roll_seven_uses_agent_discard_choice():
    from catanlab.board import Board
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState
    from catanlab.turns import (
        ActionType,
        TurnAction,
        TurnAgent,
        run_turn,
    )

    board = Board(
        tiles=[],
        vertices=[],
        edges=[],
    )

    players = [
        PlayerState(player_id=0),
        PlayerState(player_id=1),
    ]

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    inventories[1].add(
        Resource.WOOD,
        4,
    )
    inventories[1].add(
        Resource.ORE,
        4,
    )

    class PassAgent(TurnAgent):
        def choose_action(
            self,
            board,
            players,
            player,
            inventory,
            dev_deck=None,
        ):
            return TurnAction(
                action_type=ActionType.PASS
            )

    class OreDiscardAgent(PassAgent):
        def choose_discards(
            self,
            player,
            inventory,
            count,
        ):
            assert count == 4
            return [
                Resource.ORE,
                Resource.ORE,
                Resource.ORE,
                Resource.ORE,
            ]

    result = run_turn(
        board,
        players,
        inventories,
        [
            PassAgent(),
            OreDiscardAgent(),
        ],
        player_id=0,
        roll=7,
    )

    assert result.discards[1] == [
        Resource.ORE,
        Resource.ORE,
        Resource.ORE,
        Resource.ORE,
    ]

    assert (
        inventories[1].count(
            Resource.ORE
        )
        == 0
    )

    assert (
        inventories[1].count(
            Resource.WOOD
        )
        == 4
    )


def test_full_ows_discard_policy_preserves_ore_and_wheat():
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState
    from catanlab.strategies import StrategyType
    from catanlab.turns import AdaptiveStrategyAgent

    inventory = PlayerInventory()

    inventory.add(Resource.WOOD, 5)
    inventory.add(Resource.BRICK, 1)
    inventory.add(Resource.SHEEP, 1)
    inventory.add(Resource.WHEAT, 1)
    inventory.add(Resource.ORE, 1)

    player = PlayerState(
        player_id=0
    )

    agent = AdaptiveStrategyAgent(
        StrategyType.FULL_OWS
    )

    discarded = agent.choose_discards(
        player,
        inventory,
        4,
    )

    assert len(discarded) == 4

    # FULL_OWS strongly values ore/wheat and has a
    # large low-value wood surplus, so wood should be
    # discarded before those scarce premium cards.
    assert discarded == [
        Resource.WOOD,
        Resource.WOOD,
        Resource.WOOD,
        Resource.WOOD,
    ]


def test_robber_prefers_high_value_opponent_city():
    from catanlab.board import (
        Board,
        Tile,
        Vertex,
    )
    from catanlab.economy import (
        PlayerInventory,
    )
    from catanlab.resources import (
        Resource,
    )
    from catanlab.simulation import (
        PlayerState,
    )
    from catanlab.turns import (
        _knight_target_tile,
    )

    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=(0, 0),
                resource=Resource.ORE,
                number=6,
            ),
            Tile(
                id=1,
                coord=(1, 0),
                resource=Resource.WOOD,
                number=3,
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

    board.robber_tile_id = None

    players = [
        PlayerState(
            player_id=0,
        ),
        PlayerState(
            player_id=1,
            cities=[0],
        ),
        PlayerState(
            player_id=2,
            settlements=[1],
        ),
    ]

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
        PlayerInventory(),
    ]

    inventories[1].add(
        Resource.WHEAT,
        1,
    )
    inventories[2].add(
        Resource.WOOD,
        1,
    )

    target = _knight_target_tile(
        board,
        players,
        inventories,
        players[0],
    )

    assert target == 0


def test_robber_avoids_blocking_own_city_when_opponent_target_exists():
    from catanlab.board import (
        Board,
        Tile,
        Vertex,
    )
    from catanlab.economy import (
        PlayerInventory,
    )
    from catanlab.resources import (
        Resource,
    )
    from catanlab.simulation import (
        PlayerState,
    )
    from catanlab.turns import (
        _knight_target_tile,
    )

    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=(0, 0),
                resource=Resource.ORE,
                number=6,
            ),
            Tile(
                id=1,
                coord=(1, 0),
                resource=Resource.WOOD,
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
                adjacent_tiles=[1],
            ),
        ],
        edges=[],
    )

    board.robber_tile_id = None

    players = [
        PlayerState(
            player_id=0,
            cities=[0],
        ),
        PlayerState(
            player_id=1,
            settlements=[1],
        ),
    ]

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    inventories[1].add(
        Resource.BRICK,
        1,
    )

    target = _knight_target_tile(
        board,
        players,
        inventories,
        players[0],
    )

    assert target == 1


def test_action_dev_card_can_be_played_after_normal_action():
    from catanlab.board import Board
    from catanlab.devcard_policy import (
        DevCardDecision,
        DevCardPhase,
    )
    from catanlab.devcards import DevCardType
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState
    from catanlab.turns import (
        ActionType,
        TurnAction,
        TurnAgent,
        run_turn,
    )

    class InterleavingAgent(TurnAgent):
        def __init__(self):
            self.dev_calls = 0
            self.actions = [
                TurnAction(
                    action_type=(
                        ActionType.MARITIME_TRADE
                    ),
                    give_resource=Resource.WOOD,
                    receive_resource=Resource.BRICK,
                ),
                TurnAction(
                    action_type=ActionType.PASS
                ),
            ]

        def choose_dev_card_play(
            self,
            board,
            players,
            player,
            inventories,
            phase,
        ):
            self.dev_calls += 1

            # Decline:
            #   1. pre-roll
            #   2. immediate post-roll
            #
            # Play Monopoly at the next opportunity,
            # which occurs after the maritime trade.
            if self.dev_calls < 3:
                return DevCardDecision(
                    card=None,
                    utility=0.0,
                )

            return DevCardDecision(
                card=DevCardType.MONOPOLY,
                utility=10.0,
                resource=Resource.ORE,
            )

        def choose_action(
            self,
            board,
            players,
            player,
            inventory,
        ):
            return self.actions.pop(0)

    board = Board(
        tiles=[],
        vertices=[],
        edges=[],
    )

    player = PlayerState(
        player_id=0,
        dev_cards=[
            DevCardType.MONOPOLY.value
        ],
    )

    opponent = PlayerState(
        player_id=1
    )

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    # Ordinary 4:1 maritime trade.
    inventories[0].add(
        Resource.WOOD,
        4,
    )

    # Monopoly will collect these after the
    # maritime trade has executed.
    inventories[1].add(
        Resource.ORE,
        3,
    )

    agent = InterleavingAgent()

    result = run_turn(
        board,
        [
            player,
            opponent,
        ],
        inventories,
        [
            agent,
            TurnAgent(),
        ],
        player_id=0,
        roll=2,
    )

    # The normal action happened first.
    assert (
        inventories[0].count(
            Resource.WOOD
        )
        == 0
    )

    assert (
        inventories[0].count(
            Resource.BRICK
        )
        == 1
    )

    # Then Monopoly was played between normal
    # actions and transferred all opponent ore.
    assert (
        inventories[0].count(
            Resource.ORE
        )
        == 3
    )

    assert (
        inventories[1].count(
            Resource.ORE
        )
        == 0
    )

    assert (
        DevCardType.MONOPOLY.value
        not in player.dev_cards
    )

    # Pre-roll, immediate post-roll, and the
    # post-maritime-trade opportunity.
    assert agent.dev_calls == 3

    assert result.actions == (
        TurnAction(
            action_type=(
                ActionType.MARITIME_TRADE
            ),
            give_resource=Resource.WOOD,
            receive_resource=Resource.BRICK,
        ),
        TurnAction(
            action_type=ActionType.PASS
        ),
    )


def test_only_one_action_dev_card_can_be_played_with_interleaving():
    from catanlab.board import Board
    from catanlab.devcard_policy import DevCardDecision
    from catanlab.devcards import DevCardType
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState
    from catanlab.turns import (
        ActionType,
        TurnAction,
        TurnAgent,
        run_turn,
    )

    class OneDevOnlyAgent(TurnAgent):
        def __init__(self):
            self.dev_calls = 0
            self.actions = [
                TurnAction(
                    action_type=(
                        ActionType.MARITIME_TRADE
                    ),
                    give_resource=Resource.WOOD,
                    receive_resource=Resource.BRICK,
                ),
                TurnAction(
                    action_type=ActionType.PASS
                ),
            ]

        def choose_dev_card_play(
            self,
            board,
            players,
            player,
            inventories,
            phase,
        ):
            self.dev_calls += 1

            # Play Monopoly immediately pre-roll.
            return DevCardDecision(
                card=DevCardType.MONOPOLY,
                utility=10.0,
                resource=Resource.ORE,
            )

        def choose_action(
            self,
            board,
            players,
            player,
            inventory,
        ):
            return self.actions.pop(0)

    board = Board(
        tiles=[],
        vertices=[],
        edges=[],
    )

    player = PlayerState(
        player_id=0,
        dev_cards=[
            DevCardType.MONOPOLY.value,
            DevCardType.KNIGHT.value,
        ],
    )

    opponent = PlayerState(
        player_id=1
    )

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    inventories[0].add(
        Resource.WOOD,
        4,
    )

    inventories[1].add(
        Resource.ORE,
        2,
    )

    agent = OneDevOnlyAgent()

    run_turn(
        board,
        [
            player,
            opponent,
        ],
        inventories,
        [
            agent,
            TurnAgent(),
        ],
        player_id=0,
        roll=2,
    )

    # Monopoly was played.
    assert (
        inventories[0].count(
            Resource.ORE
        )
        == 2
    )

    assert (
        DevCardType.MONOPOLY.value
        not in player.dev_cards
    )

    # Knight remains untouched because no second
    # action development card may be played.
    assert (
        DevCardType.KNIGHT.value
        in player.dev_cards
    )

    # Once the pre-roll card succeeds, later
    # post-roll/interleaved opportunities are
    # suppressed entirely.
    assert agent.dev_calls == 1


def test_run_turn_stops_immediately_after_reaching_ten_vp():
    """
    Reaching 10 VP during a normal action should end
    the active player's turn immediately.

    In particular, the agent must not be asked to
    choose another normal action after the winning
    build.
    """

    class WinningCityAgent(
        GreedyBuildAgent
    ):
        def __init__(self):
            self.action_calls = 0

        def choose_action(
            self,
            board,
            players,
            player,
            inventory,
            dev_deck=None,
            bank=None,
        ):
            self.action_calls += 1

            if self.action_calls == 1:
                return TurnAction(
                    action_type=(
                        ActionType.BUILD_CITY
                    ),
                    vertex_id=0,
                )

            # This should never be reached.
            return TurnAction(
                action_type=ActionType.PASS
            )

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=i,
                position=(float(i), 0.0),
            )
            for i in range(6)
        ],
        edges=[],
    )

    # 3 cities = 6 VP
    # 3 settlements = 3 VP
    # Total before the turn = 9 VP.
    player = PlayerState(
        player_id=0,
        settlements=[
            0,
            1,
            2,
        ],
        cities=[
            3,
            4,
            5,
        ],
    )

    assert player.victory_points == 9

    inventory = PlayerInventory()

    inventory.add(
        Resource.WHEAT,
        2,
    )

    inventory.add(
        Resource.ORE,
        3,
    )

    agent = WinningCityAgent()

    result = run_turn(
        board,
        [player],
        [inventory],
        [agent],
        player_id=0,
        roll=2,
    )

    # Upgrading one settlement to a city adds
    # exactly one net victory point.
    assert player.victory_points == 10

    assert player.settlements == [
        1,
        2,
    ]

    assert 0 in player.cities

    # Most importantly, no second normal action was
    # requested after the winning build.
    assert agent.action_calls == 1

    assert len(
        result.actions
    ) == 1

    assert (
        result.actions[0].action_type
        == ActionType.BUILD_CITY
    )


def test_run_turn_stops_after_longest_road_reaches_ten_vp():
    """
    Acquiring Longest Road during a turn should
    immediately count toward victory.

    A player at 8 VP who builds their fifth
    consecutive road reaches 10 VP and must not be
    asked for another normal action.
    """

    class WinningRoadAgent(
        GreedyBuildAgent
    ):
        def __init__(self):
            self.action_calls = 0

        def choose_action(
            self,
            board,
            players,
            player,
            inventory,
            dev_deck=None,
            bank=None,
        ):
            self.action_calls += 1

            if self.action_calls == 1:
                return TurnAction(
                    action_type=(
                        ActionType.BUILD_ROAD
                    ),
                    edge=(4, 5),
                )

            # Victory should prevent this call.
            return TurnAction(
                action_type=ActionType.PASS
            )

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=i,
                position=(float(i), 0.0),
                neighbors=(
                    [1]
                    if i == 0
                    else (
                        [4]
                        if i == 5
                        else (
                            [
                                i - 1,
                                i + 1,
                            ]
                            if 1 <= i <= 4
                            else []
                        )
                    )
                ),
            )
            for i in range(13)
        ],
        edges=[
            Edge(
                vertex_a=i,
                vertex_b=i + 1,
            )
            for i in range(5)
        ],
    )

    # Four cities = 8 VP.
    #
    # The existing four-road chain is one road short
    # of Longest Road.
    player = PlayerState(
        player_id=0,
        cities=[
            0,
            10,
            11,
            12,
        ],
        roads=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
        ],
    )

    assert player.victory_points == 8
    assert not player.has_longest_road

    inventory = PlayerInventory()

    inventory.add(
        Resource.WOOD,
        1,
    )

    inventory.add(
        Resource.BRICK,
        1,
    )

    agent = WinningRoadAgent()

    result = run_turn(
        board,
        [player],
        [inventory],
        [agent],
        player_id=0,
        roll=2,
    )

    assert (4, 5) in player.roads

    assert player.has_longest_road

    assert player.victory_points == 10

    # No second normal action may occur after
    # Longest Road creates the winning score.
    assert agent.action_calls == 1

    assert len(
        result.actions
    ) == 1

    assert (
        result.actions[0].action_type
        == ActionType.BUILD_ROAD
    )


def test_buying_victory_point_card_can_win_immediately():
    """
    A Victory Point development card counts toward
    victory immediately when purchased.

    The normal same-turn restriction on newly bought
    action development cards must not delay the VP.
    """

    from catanlab.devcards import (
        DevCardDeck,
        DevCardType,
    )

    class WinningDevCardAgent(
        GreedyBuildAgent
    ):
        def __init__(self):
            self.action_calls = 0

        def choose_action(
            self,
            board,
            players,
            player,
            inventory,
            dev_deck=None,
            bank=None,
        ):
            self.action_calls += 1

            if self.action_calls == 1:
                return TurnAction(
                    action_type=(
                        ActionType.BUY_DEV_CARD
                    )
                )

            # The VP card should end the turn before
            # another normal action is requested.
            return TurnAction(
                action_type=ActionType.PASS
            )

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=i,
                position=(float(i), 0.0),
            )
            for i in range(6)
        ],
        edges=[],
    )

    # 3 cities = 6 VP
    # 3 settlements = 3 VP
    # Total = 9 VP.
    player = PlayerState(
        player_id=0,
        settlements=[
            0,
            1,
            2,
        ],
        cities=[
            3,
            4,
            5,
        ],
    )

    assert player.victory_points == 9

    inventory = PlayerInventory()

    inventory.add(
        Resource.SHEEP,
        1,
    )
    inventory.add(
        Resource.WHEAT,
        1,
    )
    inventory.add(
        Resource.ORE,
        1,
    )

    # Force the purchased card to be a VP card.
    deck = DevCardDeck(
        cards=[
            DevCardType.VICTORY_POINT,
        ]
    )

    agent = WinningDevCardAgent()

    result = run_turn(
        board,
        [player],
        [inventory],
        [agent],
        player_id=0,
        roll=2,
        dev_deck=deck,
    )

    assert (
        DevCardType.VICTORY_POINT.value
        in player.dev_cards
    )

    # It may still be marked as newly purchased,
    # which is fine; VP scoring is immediate.
    assert (
        DevCardType.VICTORY_POINT.value
        in player.new_dev_cards
    )

    assert player.victory_points == 10

    assert agent.action_calls == 1

    assert len(
        result.actions
    ) == 1

    assert (
        result.actions[0].action_type
        == ActionType.BUY_DEV_CARD
    )


def test_turn_can_end_immediately_when_player_already_has_ten_vp():
    from catanlab.board import Board
    from catanlab.economy import PlayerInventory
    from catanlab.simulation import PlayerState
    from catanlab.turns import (
        ActionType,
        TurnAgent,
        run_turn,
    )

    board = Board(
        tiles=[],
        vertices=[],
        edges=[],
    )

    player = PlayerState(
        player_id=0,
        cities=[0, 1, 2, 3],
        settlements=[4, 5],
    )

    assert player.victory_points == 10

    result = run_turn(
        board,
        [player],
        [PlayerInventory()],
        [TurnAgent()],
        player_id=0,
        roll=2,
    )

    assert (
        result.action.action_type
        == ActionType.PASS
    )


def test_post_roll_dev_card_victory_stops_before_normal_action():
    from catanlab.board import (
        build_random_board,
    )
    from catanlab.devcard_policy import (
        DevCardDecision,
    )
    from catanlab.devcards import DevCardType
    from catanlab.economy import PlayerInventory
    from catanlab.simulation import PlayerState
    from catanlab.turns import (
        ActionType,
        TurnAction,
        TurnAgent,
        run_turn,
    )

    class WinningPostRollAgent(TurnAgent):
        def __init__(self):
            self.dev_calls = 0
            self.action_calls = 0

        def choose_dev_card_play(
            self,
            board,
            players,
            player,
            inventories,
            phase,
        ):
            self.dev_calls += 1

            # Decline pre-roll, then play the third
            # Knight immediately after the dice roll.
            if self.dev_calls == 1:
                return DevCardDecision(
                    card=None,
                    utility=0.0,
                )

            return DevCardDecision(
                card=DevCardType.KNIGHT,
                utility=10.0,
            )

        def choose_action(
            self,
            board,
            players,
            player,
            inventory,
            dev_deck=None,
            bank=None,
        ):
            self.action_calls += 1
            return TurnAction(
                action_type=ActionType.PASS
            )

    board = build_random_board(
        seed=123
    )

    player = PlayerState(
        player_id=0,
        cities=[0, 1, 2, 3],
        knights_played=2,
        dev_cards=[
            DevCardType.KNIGHT.value
        ],
    )

    # 8 building VP; the third Knight should award
    # Largest Army and bring the player to 10.
    assert player.victory_points == 8

    agent = WinningPostRollAgent()

    result = run_turn(
        board,
        [player],
        [PlayerInventory()],
        [agent],
        player_id=0,
        roll=2,
    )

    assert player.has_largest_army
    assert player.victory_points == 10
    assert agent.action_calls == 0
    assert (
        result.action.action_type
        == ActionType.PASS
    )


def test_trade_proposal_ignores_recipient_hidden_resource_composition():
    from catanlab.board import build_random_board
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState
    from catanlab.strategies import StrategyType
    from catanlab.turns import AdaptiveStrategyAgent

    board = build_random_board(seed=321)

    players = [
        PlayerState(
            player_id=0,
            settlements=[0],
            roads=[(0, 1)],
        ),
        PlayerState(
            player_id=1,
            settlements=[10],
        ),
    ]

    agent = AdaptiveStrategyAgent(
        StrategyType.FIVE_RESOURCE
    )

    inventories_a = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    inventories_b = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    # Give the proposer the same exact private hand
    # in both worlds.
    for inventories in (
        inventories_a,
        inventories_b,
    ):
        inventories[0].add(
            Resource.WOOD,
            2,
        )
        inventories[0].add(
            Resource.BRICK,
            2,
        )
        inventories[0].add(
            Resource.SHEEP,
            1,
        )

    # Opponent has the same public hand size in both
    # worlds, but completely different hidden
    # resource identities.
    inventories_a[1].add(
        Resource.ORE,
        4,
    )

    inventories_b[1].add(
        Resource.WHEAT,
        4,
    )

    offer_a = agent.propose_player_trade(
        board,
        players,
        players[0],
        inventories_a,
    )

    offer_b = agent.propose_player_trade(
        board,
        players,
        players[0],
        inventories_b,
    )

    assert offer_a == offer_b


def test_trade_counteroffer_ignores_other_hidden_resource_composition():
    from catanlab.board import build_random_board
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState
    from catanlab.strategies import StrategyType
    from catanlab.trading import TradeOffer
    from catanlab.turns import AdaptiveStrategyAgent

    board = build_random_board(seed=654)

    players = [
        PlayerState(
            player_id=0,
            settlements=[0],
        ),
        PlayerState(
            player_id=1,
            settlements=[10],
        ),
    ]

    agent = AdaptiveStrategyAgent(
        StrategyType.FIVE_RESOURCE
    )

    inventories_a = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    inventories_b = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    # Counter-proposer's own exact hand is identical.
    for inventories in (
        inventories_a,
        inventories_b,
    ):
        inventories[1].add(
            Resource.WOOD,
            2,
        )
        inventories[1].add(
            Resource.BRICK,
            2,
        )
        inventories[1].add(
            Resource.SHEEP,
            1,
        )

    # Original proposer has equal public hand size,
    # but different hidden composition.
    inventories_a[0].add(
        Resource.ORE,
        4,
    )

    inventories_b[0].add(
        Resource.WHEAT,
        4,
    )

    offer = TradeOffer(
        proposer_id=0,
        recipient_id=1,
        give=((Resource.ORE, 1),),
        receive=((Resource.WOOD, 1),),
    )

    counter_a = agent.counter_player_trade(
        board,
        players,
        players[1],
        inventories_a,
        offer,
    )

    counter_b = agent.counter_player_trade(
        board,
        players,
        players[1],
        inventories_b,
        offer,
    )

    assert counter_a == counter_b


def test_dev_card_agent_can_receive_bank():
    from catanlab.board import build_random_board
    from catanlab.devcard_policy import (
        DevCardDecision,
    )
    from catanlab.economy import ResourceBank
    from catanlab.turns import TurnAgent

    class BankAwareAgent(TurnAgent):
        def __init__(self):
            self.received_bank = None

        def choose_dev_card_play(
            self,
            board,
            players,
            player,
            inventories,
            phase,
            bank=None,
        ):
            self.received_bank = bank

            return DevCardDecision(
                card=None,
                utility=0.0,
            )

        def choose_action(
            self,
            board,
            players,
            player,
            inventory,
            dev_deck=None,
            bank=None,
        ):
            return TurnAction(
                action_type=ActionType.PASS
            )

    board = build_random_board(seed=901)
    players = [
        PlayerState(player_id=0)
    ]
    inventories = [
        PlayerInventory()
    ]
    agent = BankAwareAgent()
    bank = ResourceBank()

    run_turn(
        board,
        players,
        inventories,
        [agent],
        player_id=0,
        roll=2,
        bank=bank,
    )

    assert agent.received_bank is bank


def test_dev_card_agent_can_receive_dev_deck():
    from catanlab.board import build_random_board
    from catanlab.devcard_policy import (
        DevCardDecision,
    )
    from catanlab.devcards import (
        build_dev_card_deck,
    )
    from catanlab.turns import TurnAgent

    class DeckAwareAgent(TurnAgent):
        def __init__(self):
            self.received_deck = None

        def choose_dev_card_play(
            self,
            board,
            players,
            player,
            inventories,
            phase,
            dev_deck=None,
        ):
            self.received_deck = dev_deck

            return DevCardDecision(
                card=None,
                utility=0.0,
            )

        def choose_action(
            self,
            board,
            players,
            player,
            inventory,
            dev_deck=None,
            bank=None,
        ):
            return TurnAction(
                action_type=ActionType.PASS
            )

    board = build_random_board(seed=902)
    players = [
        PlayerState(player_id=0)
    ]
    inventories = [
        PlayerInventory()
    ]
    agent = DeckAwareAgent()
    deck = build_dev_card_deck(seed=902)

    run_turn(
        board,
        players,
        inventories,
        [agent],
        player_id=0,
        roll=2,
        dev_deck=deck,
    )

    assert agent.received_deck is deck


def test_year_of_plenty_uses_explicit_resource_pair():
    from catanlab.board import build_random_board
    from catanlab.devcard_policy import (
        DevCardDecision,
    )
    from catanlab.devcards import DevCardType
    from catanlab.economy import ResourceBank
    from catanlab.resources import Resource
    from catanlab.turns import TurnAgent

    class ExplicitPlentyAgent(TurnAgent):
        def __init__(self):
            self.used = False

        def choose_dev_card_play(
            self,
            board,
            players,
            player,
            inventories,
            phase,
        ):
            if self.used:
                return DevCardDecision(
                    card=None,
                    utility=0.0,
                )

            self.used = True

            return DevCardDecision(
                card=(
                    DevCardType.YEAR_OF_PLENTY
                ),
                utility=10.0,
                resources=(
                    Resource.ORE,
                    Resource.ORE,
                ),
            )

        def choose_action(
            self,
            board,
            players,
            player,
            inventory,
            dev_deck=None,
            bank=None,
        ):
            return TurnAction(
                action_type=ActionType.PASS
            )

    board = build_random_board(seed=903)

    player = PlayerState(
        player_id=0,
        dev_cards=[
            DevCardType.YEAR_OF_PLENTY.value
        ],
    )

    inventory = PlayerInventory()
    bank = ResourceBank()

    ore_before = inventory.count(
        Resource.ORE
    )

    run_turn(
        board,
        [player],
        [inventory],
        [ExplicitPlentyAgent()],
        player_id=0,
        roll=2,
        bank=bank,
    )

    assert (
        inventory.count(Resource.ORE)
        == ore_before + 2
    )

    assert (
        DevCardType.YEAR_OF_PLENTY.value
        not in player.dev_cards
    )

    assert (
        DevCardType.YEAR_OF_PLENTY.value
        in player.played_dev_cards
    )
