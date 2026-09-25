from catanlab.board import build_random_board
from catanlab.devcards import (
    build_dev_card_deck,
)
from catanlab.economy import (
    PlayerInventory,
    ResourceBank,
)
from catanlab.rl_teacher import (
    DAggerAgent,
    RecordingSearchAgent,
)
from catanlab.simulation import PlayerState
from catanlab.strategies import StrategyType


def make_fixture():
    board = build_random_board(
        seed=123
    )

    players = [
        PlayerState(player_id=i)
        for i in range(4)
    ]

    inventories = [
        PlayerInventory()
        for _ in range(4)
    ]

    bank = ResourceBank()

    deck = build_dev_card_deck(
        seed=456
    )

    return (
        board,
        players,
        inventories,
        bank,
        deck,
    )


def test_recording_search_agent_records_example():
    (
        board,
        players,
        inventories,
        bank,
        deck,
    ) = make_fixture()

    agent = RecordingSearchAgent(
        StrategyType.FIVE_RESOURCE
    )

    assert agent.examples == []

    action = agent.choose_action(
        board,
        players,
        players[0],
        inventories[0],
        dev_deck=deck,
        bank=bank,
        inventories=inventories,
    )

    assert action is not None
    assert len(agent.examples) == 1

    example = agent.examples[0]

    assert len(example.observation) == 1138
    assert len(example.legal_mask) == 202

    assert example.legal_mask[
        example.action_id
    ]

    assert example.player_id == 0


def test_recorded_example_is_snapshot():
    (
        board,
        players,
        inventories,
        bank,
        deck,
    ) = make_fixture()

    agent = RecordingSearchAgent(
        StrategyType.FIVE_RESOURCE
    )

    agent.choose_action(
        board,
        players,
        players[0],
        inventories[0],
        dev_deck=deck,
        bank=bank,
        inventories=inventories,
    )

    example = agent.examples[0]

    before = example.observation

    # Mutating the live state must not mutate the
    # already-recorded training example.
    from catanlab.resources import Resource

    inventories[0].add(
        Resource.ORE,
        4,
    )

    assert example.observation == before


def test_teacher_label_is_always_mask_legal():
    (
        board,
        players,
        inventories,
        bank,
        deck,
    ) = make_fixture()

    agent = RecordingSearchAgent(
        StrategyType.FIVE_RESOURCE
    )

    for _ in range(5):
        agent.choose_action(
            board,
            players,
            players[0],
            inventories[0],
            dev_deck=deck,
            bank=bank,
            inventories=inventories,
        )

    assert len(agent.examples) == 5

    assert all(
        example.legal_mask[
            example.action_id
        ]
        for example in agent.examples
    )


def test_dagger_agent_records_teacher_label():
    from catanlab.rl_model import (
        CatanActorCritic,
    )

    (
        board,
        players,
        inventories,
        bank,
        deck,
    ) = make_fixture()

    agent = DAggerAgent(
        StrategyType.FIVE_RESOURCE,
        model=CatanActorCritic(),
        deterministic=True,
        seed=1,
    )

    action = agent.choose_action(
        board,
        players,
        players[0],
        inventories[0],
        dev_deck=deck,
        bank=bank,
        inventories=inventories,
    )

    assert action is not None
    assert len(agent.examples) == 1

    example = agent.examples[0]

    assert len(example.observation) == 1138
    assert len(example.legal_mask) == 202

    assert example.legal_mask[
        example.action_id
    ]

    assert example.legal_mask[
        example.learner_action_id
    ]

    assert example.player_id == 0


def test_dagger_returned_action_is_learner_action():
    from catanlab.action_space import (
        turn_action_id,
    )
    from catanlab.rl_interface import (
        build_rl_decision_context,
    )
    from catanlab.rl_model import (
        CatanActorCritic,
    )

    (
        board,
        players,
        inventories,
        bank,
        deck,
    ) = make_fixture()

    agent = DAggerAgent(
        StrategyType.FIVE_RESOURCE,
        model=CatanActorCritic(),
        deterministic=True,
        seed=2,
    )

    context = build_rl_decision_context(
        board,
        players,
        inventories,
        0,
        bank,
        deck,
    )

    returned = agent.choose_action(
        board,
        players,
        players[0],
        inventories[0],
        dev_deck=deck,
        bank=bank,
        inventories=inventories,
    )

    returned_id = turn_action_id(
        returned,
        context.vocabulary,
    )

    assert returned_id == (
        agent.examples[
            0
        ].learner_action_id
    )


def test_teacher_v2_example_supports_flat_masked_decision():
    from catanlab.rl_teacher import (
        TeacherDecisionKind,
        TeacherV2Example,
    )

    example = TeacherV2Example(
        decision_kind=(
            TeacherDecisionKind.ORDINARY_ACTION
        ),
        observation=(1.0, 2.0),
        player_id=0,
        label=17,
        legal_mask=(
            True,
            False,
            True,
        ),
    )

    assert (
        example.decision_kind
        == TeacherDecisionKind.ORDINARY_ACTION
    )
    assert example.label == 17
    assert example.has_legal_mask


def test_teacher_v2_example_supports_structured_decision():
    from catanlab.resources import Resource
    from catanlab.rl_teacher import (
        TeacherDecisionKind,
        TeacherV2Example,
    )

    example = TeacherV2Example(
        decision_kind=(
            TeacherDecisionKind.YEAR_OF_PLENTY
        ),
        observation=(0.0,),
        player_id=2,
        label=(
            Resource.ORE,
            Resource.WHEAT,
        ),
    )

    assert example.label == (
        Resource.ORE,
        Resource.WHEAT,
    )
    assert example.legal_mask is None
    assert not example.has_legal_mask


def test_teacher_v2_decision_kinds_are_distinct():
    from catanlab.rl_teacher import (
        TeacherDecisionKind,
    )

    values = [
        kind.value
        for kind in TeacherDecisionKind
    ]

    assert len(values) == len(set(values))

    assert (
        TeacherDecisionKind.ROBBER_TILE
        != TeacherDecisionKind.ROBBER_VICTIM
    )

    assert (
        TeacherDecisionKind.TRADE_PROPOSAL
        != TeacherDecisionKind.TRADE_RESPONSE
    )


def test_recording_search_agent_records_robber_tile_v2_example():
    from catanlab.rl_teacher import (
        TeacherDecisionKind,
    )

    (
        board,
        players,
        inventories,
        bank,
        deck,
    ) = make_fixture()

    agent = RecordingSearchAgent(
        StrategyType.FIVE_RESOURCE,
        search_robber_decisions=True,
    )

    old_robber_tile = board.robber_tile_id

    chosen = agent.choose_robber_tile(
        board,
        players,
        inventories,
        players[0],
        bank=bank,
        dev_deck=deck,
    )

    assert chosen is not None
    assert chosen != old_robber_tile

    assert len(agent.v2_examples) == 1

    example = agent.v2_examples[0]

    assert (
        example.decision_kind
        == TeacherDecisionKind.ROBBER_TILE
    )

    assert len(example.observation) == 1138

    assert example.legal_mask is not None
    assert example.legal_mask[
        example.label
    ]


def test_recording_search_agent_robber_tile_label_decodes_to_choice():
    from catanlab.rl_special_actions import (
        robber_tile_decision_input,
    )

    (
        board,
        players,
        inventories,
        bank,
        deck,
    ) = make_fixture()

    agent = RecordingSearchAgent(
        StrategyType.FIVE_RESOURCE,
        search_robber_decisions=True,
    )

    decision_input = (
        robber_tile_decision_input(
            board
        )
    )

    chosen = agent.choose_robber_tile(
        board,
        players,
        inventories,
        players[0],
        bank=bank,
        dev_deck=deck,
    )

    example = agent.v2_examples[0]

    assert decision_input.decode(
        example.label
    ) == chosen


def test_recording_search_agent_does_not_record_disabled_robber_search():
    (
        board,
        players,
        inventories,
        bank,
        deck,
    ) = make_fixture()

    agent = RecordingSearchAgent(
        StrategyType.FIVE_RESOURCE,
        search_robber_decisions=False,
    )

    agent.choose_robber_tile(
        board,
        players,
        inventories,
        players[0],
        bank=bank,
        dev_deck=deck,
    )

    assert agent.v2_examples == []


def test_recording_search_agent_records_robber_victim_v2_example():
    from catanlab.board import (
        Board,
        Tile,
        Vertex,
    )
    from catanlab.economy import PlayerInventory
    from catanlab.graph import HexCoord
    from catanlab.resources import Resource
    from catanlab.rl_special_actions import (
        robber_victim_decision_input,
    )
    from catanlab.rl_teacher import (
        TeacherDecisionKind,
    )
    from catanlab.simulation import PlayerState

    (
        _,
        _,
        _,
        bank,
        deck,
    ) = make_fixture()

    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=HexCoord(0, 0),
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
                adjacent_tiles=[0],
            ),
        ],
        edges=[],
        robber_tile_id=0,
    )

    players = [
        PlayerState(player_id=0),
        PlayerState(
            player_id=1,
            settlements=[0],
        ),
        PlayerState(
            player_id=2,
            cities=[1],
        ),
    ]

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
        PlayerInventory(),
    ]

    inventories[1].add(
        Resource.WOOD,
        6,
    )

    inventories[2].add(
        Resource.ORE,
        1,
    )

    agent = RecordingSearchAgent(
        StrategyType.FIVE_RESOURCE,
        search_robber_decisions=True,
    )

    decision_input = (
        robber_victim_decision_input(
            board,
            players,
            inventories,
            players[0],
        )
    )

    chosen = agent.choose_robber_victim(
        board,
        players,
        inventories,
        players[0],
        bank=bank,
        dev_deck=deck,
    )

    assert chosen == 2

    assert len(agent.v2_examples) == 1

    example = agent.v2_examples[0]

    assert (
        example.decision_kind
        == TeacherDecisionKind.ROBBER_VICTIM
    )

    from catanlab.observation import (
        game_observation,
    )
    from catanlab.observation_encoder import (
        encode_game_observation,
    )

    expected_observation = (
        encode_game_observation(
            game_observation(
                board,
                players,
                inventories,
                players[0].player_id,
                bank,
                deck,
            )
        )
    )

    assert (
        example.observation
        == expected_observation
    )

    assert example.legal_mask is not None
    assert example.legal_mask[
        example.label
    ]

    assert decision_input.decode(
        example.label
    ) == chosen


def test_recording_search_agent_records_monopoly_resource_v2_example():
    from catanlab.board import (
        Board,
        Tile,
        Vertex,
    )
    from catanlab.devcard_policy import (
        DevCardPhase,
    )
    from catanlab.devcards import (
        DevCardType,
    )
    from catanlab.economy import (
        PlayerInventory,
    )
    from catanlab.graph import HexCoord
    from catanlab.resources import Resource
    from catanlab.rl_special_actions import (
        monopoly_resource_decision_input,
    )
    from catanlab.rl_teacher import (
        TeacherDecisionKind,
    )
    from catanlab.simulation import PlayerState

    (
        _,
        _,
        _,
        bank,
        deck,
    ) = make_fixture()

    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=HexCoord(0, 0),
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
                adjacent_tiles=[0],
            ),
        ],
        edges=[],
    )

    player = PlayerState(
        player_id=0,
        settlements=[0],
        dev_cards=[
            DevCardType.MONOPOLY.value
        ],
    )

    opponent = PlayerState(
        player_id=1,
        settlements=[1],
    )

    players = [
        player,
        opponent,
    ]

    own_inventory = PlayerInventory()

    own_inventory.add(
        Resource.WHEAT,
        2,
    )

    own_inventory.add(
        Resource.ORE,
        1,
    )

    opponent_inventory = PlayerInventory()

    opponent_inventory.add(
        Resource.WOOD,
        2,
    )

    inventories = [
        own_inventory,
        opponent_inventory,
    ]

    agent = RecordingSearchAgent(
        StrategyType.FIVE_RESOURCE,
        search_depth=2,
        use_transposition_cache=False,
        search_maritime_trades=False,
        search_year_of_plenty=False,
        search_road_building=False,
        search_monopoly=True,
    )

    decision = agent.choose_dev_card_play(
        board,
        players,
        player,
        inventories,
        DevCardPhase.POST_ROLL,
        dev_deck=deck,
        bank=bank,
    )

    assert (
        decision.card
        == DevCardType.MONOPOLY
    )

    assert decision.resource is None

    assert len(agent.v2_examples) == 1

    example = agent.v2_examples[0]

    assert (
        example.decision_kind
        == TeacherDecisionKind.MONOPOLY_RESOURCE
    )

    from catanlab.observation import (
        game_observation,
    )
    from catanlab.observation_encoder import (
        encode_game_observation,
    )

    expected_observation = (
        encode_game_observation(
            game_observation(
                board,
                players,
                inventories,
                player.player_id,
                bank,
                deck,
            )
        )
    )

    assert (
        example.observation
        == expected_observation
    )

    assert example.legal_mask == (
        True,
        True,
        True,
        True,
        True,
    )

    decision_input = (
        monopoly_resource_decision_input()
    )

    assert decision_input.decode(
        example.label
    ) == Resource.ORE

    assert not example.has_candidate_features
    assert example.candidate_features is None

    # Recording must not consume Search-v2's pending
    # execution choice.
    assert (
        agent._pending_monopoly_resource
        == Resource.ORE
    )

    resource = agent.choose_monopoly_resource(
        board,
        players,
        inventories,
        player,
        suggested_resource=(
            decision.resource
        ),
    )

    assert resource == Resource.ORE

    assert (
        agent._pending_monopoly_resource
        is None
    )


def test_recording_search_agent_records_year_of_plenty_v2_example():
    from catanlab.devcard_policy import (
        DevCardPhase,
    )
    from catanlab.devcards import (
        DevCardType,
    )
    from catanlab.economy import (
        PlayerInventory,
    )
    from catanlab.resources import Resource
    from catanlab.rl_special_actions import (
        year_of_plenty_decision_input,
    )
    from catanlab.rl_teacher import (
        TeacherDecisionKind,
    )

    (
        board,
        players,
        inventories,
        bank,
        deck,
    ) = make_fixture()

    player = players[0]

    player.settlements.append(
        board.vertices[0].id
    )

    player.dev_cards.append(
        DevCardType.YEAR_OF_PLENTY.value
    )

    inventory = PlayerInventory()

    inventory.add(
        Resource.WHEAT,
        2,
    )

    inventory.add(
        Resource.ORE,
        1,
    )

    inventories[0] = inventory

    agent = RecordingSearchAgent(
        StrategyType.FIVE_RESOURCE,
        search_depth=2,
        use_transposition_cache=False,
        search_maritime_trades=False,
        search_year_of_plenty=True,
        search_road_building=False,
        search_monopoly=False,
    )

    decision = agent.choose_dev_card_play(
        board,
        players,
        player,
        inventories,
        DevCardPhase.POST_ROLL,
        dev_deck=deck,
        bank=bank,
    )

    assert (
        decision.card
        == DevCardType.YEAR_OF_PLENTY
    )

    assert decision.resources is None

    assert len(agent.v2_examples) == 1

    example = agent.v2_examples[0]

    assert (
        example.decision_kind
        == TeacherDecisionKind.YEAR_OF_PLENTY
    )

    decision_input = (
        year_of_plenty_decision_input(
            bank
        )
    )

    assert decision_input.decode(
        example.label
    ) == (
        Resource.ORE,
        Resource.ORE,
    )

    assert example.legal_mask[
        example.label
    ]

    # Recording must not consume Search-v2's pending
    # execution choice.
    assert (
        agent._pending_year_of_plenty_resources
        == (
            Resource.ORE,
            Resource.ORE,
        )
    )

    resources = (
        agent.choose_year_of_plenty_resources(
            board,
            players,
            inventories,
            player,
            bank=bank,
            suggested_resources=(
                decision.resources
            ),
        )
    )

    assert resources == (
        Resource.ORE,
        Resource.ORE,
    )

    assert (
        agent._pending_year_of_plenty_resources
        is None
    )


def test_recording_search_agent_records_road_building_v2_example():
    from catanlab.board import (
        Board,
        Edge,
        Vertex,
    )
    from catanlab.devcard_policy import (
        DevCardPhase,
    )
    from catanlab.devcards import (
        DevCardType,
    )
    from catanlab.economy import (
        PlayerInventory,
    )
    from catanlab.resources import Resource
    from catanlab.rl_special_actions import (
        road_building_decision_input,
    )
    from catanlab.rl_teacher import (
        TeacherDecisionKind,
    )
    from catanlab.simulation import (
        PlayerState,
    )

    (
        _,
        _,
        _,
        bank,
        deck,
    ) = make_fixture()

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
        dev_cards=[
            DevCardType.ROAD_BUILDING.value
        ],
    )

    players = [player]

    inventory = PlayerInventory()

    inventory.add(
        Resource.WOOD,
        1,
    )
    inventory.add(
        Resource.BRICK,
        1,
    )
    inventory.add(
        Resource.SHEEP,
        1,
    )
    inventory.add(
        Resource.WHEAT,
        1,
    )

    inventories = [inventory]

    agent = RecordingSearchAgent(
        StrategyType.FIVE_RESOURCE,
        search_depth=2,
        use_transposition_cache=False,
        search_maritime_trades=False,
        search_year_of_plenty=False,
        search_road_building=True,
        search_monopoly=False,
    )

    decision_input = (
        road_building_decision_input(
            board,
            players,
            player,
        )
    )

    decision = agent.choose_dev_card_play(
        board,
        players,
        player,
        inventories,
        DevCardPhase.POST_ROLL,
        dev_deck=deck,
        bank=bank,
    )

    assert (
        decision.card
        == DevCardType.ROAD_BUILDING
    )

    assert decision.road_edges is None

    assert len(agent.v2_examples) == 1

    example = agent.v2_examples[0]

    assert (
        example.decision_kind
        == TeacherDecisionKind.ROAD_BUILDING
    )

    assert (
        decision_input.decode(
            example.label
        )
        == (
            (0, 1),
            (1, 2),
        )
    )

    assert example.legal_mask[
        example.label
    ]

    assert example.has_candidate_features

    assert len(
        example.candidate_features
    ) == decision_input.action_dim

    assert (
        example.candidate_features[
            example.label
        ]
        == (
            0.0,
            1.0 / 54.0,
            1.0 / 54.0,
            2.0 / 54.0,
            1.0,
        )
    )

    # Recording is observational: the simulator still
    # owns consumption of the pending Search-v2 choice.
    assert (
        agent._pending_road_building_edges
        == (
            (0, 1),
            (1, 2),
        )
    )

    roads = (
        agent.choose_road_building_edges(
            board,
            players,
            inventories,
            player,
            suggested_edges=(
                decision.road_edges
            ),
        )
    )

    assert roads == (
        (0, 1),
        (1, 2),
    )

    assert (
        agent._pending_road_building_edges
        is None
    )


def test_recording_search_agent_records_discard_v2_example():
    from catanlab.board import (
        Board,
        Vertex,
    )
    from catanlab.economy import (
        PlayerInventory,
    )
    from catanlab.resources import Resource
    from catanlab.rl_special_actions import (
        discard_decision_input,
    )
    from catanlab.rl_teacher import (
        TeacherDecisionKind,
    )
    from catanlab.simulation import (
        PlayerState,
    )

    (
        _,
        _,
        _,
        bank,
        deck,
    ) = make_fixture()

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
                adjacent_tiles=[],
            ),
        ],
        edges=[],
    )

    player = PlayerState(
        player_id=0,
        settlements=[0],
    )

    inventory = PlayerInventory()

    inventory.add(Resource.ORE, 2)
    inventory.add(Resource.WHEAT, 2)
    inventory.add(Resource.WOOD, 2)
    inventory.add(Resource.BRICK, 2)

    players = [player]
    inventories = [inventory]

    agent = RecordingSearchAgent(
        StrategyType.FIVE_RESOURCE,
        search_discard_decisions=True,
    )

    decision_input = discard_decision_input(
        inventory,
        4,
    )

    discarded = (
        agent.choose_discards_with_context(
            board,
            players,
            inventories,
            player,
            inventory,
            4,
            bank=bank,
            dev_deck=deck,
        )
    )

    assert discarded.count(
        Resource.WOOD
    ) == 2

    assert discarded.count(
        Resource.BRICK
    ) == 2

    assert Resource.ORE not in discarded
    assert Resource.WHEAT not in discarded

    assert len(agent.v2_examples) == 1

    example = agent.v2_examples[0]

    assert (
        example.decision_kind
        == TeacherDecisionKind.DISCARD
    )

    assert (
        decision_input.decode(
            example.label
        )
        == (
            2,
            2,
            0,
            0,
            0,
        )
    )

    assert example.legal_mask[
        example.label
    ]

    assert example.has_candidate_features

    assert len(
        example.candidate_features
    ) == decision_input.action_dim

    assert (
        example.candidate_features[
            example.label
        ]
        == (
            2.0,
            2.0,
            0.0,
            0.0,
            0.0,
        )
    )

    # Recording must not alter the returned discard choice
    # or mutate the hand. Actual card removal remains the
    # simulator's responsibility.
    assert inventory.count(
        Resource.WOOD
    ) == 2

    assert inventory.count(
        Resource.BRICK
    ) == 2

    assert inventory.count(
        Resource.WHEAT
    ) == 2

    assert inventory.count(
        Resource.ORE
    ) == 2


def test_recording_search_agent_records_trade_proposal_v2_example():
    from catanlab.board import Board, Vertex
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.rl_special_actions import (
        trade_proposal_decision_input,
    )
    from catanlab.rl_teacher import (
        TeacherDecisionKind,
    )
    from catanlab.simulation import PlayerState
    from catanlab.trading import TradeOffer

    (
        _,
        _,
        _,
        bank,
        deck,
    ) = make_fixture()

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
                adjacent_tiles=[],
            ),
        ],
        edges=[],
    )

    players = [
        PlayerState(
            player_id=0,
            settlements=[0],
        ),
        PlayerState(player_id=1),
    ]

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    inventories[0].add(Resource.WHEAT, 2)
    inventories[0].add(Resource.ORE, 2)
    inventories[0].add(Resource.WOOD, 1)

    agent = RecordingSearchAgent(
        StrategyType.FIVE_RESOURCE,
        search_domestic_trades=True,
    )

    decision_input = (
        trade_proposal_decision_input(
            players,
            players[0],
            inventories[0],
        )
    )

    offer = agent.propose_player_trade(
        board,
        players,
        players[0],
        inventories,
        bank=bank,
        dev_deck=deck,
    )

    expected = TradeOffer(
        proposer_id=0,
        recipient_id=1,
        give=((Resource.WOOD, 1),),
        receive=((Resource.ORE, 1),),
    )

    assert offer == expected

    assert len(agent.v2_examples) == 1

    example = agent.v2_examples[0]

    assert (
        example.decision_kind
        == TeacherDecisionKind.TRADE_PROPOSAL
    )

    assert (
        decision_input.decode(
            example.label
        )
        == expected
    )

    assert example.legal_mask[
        example.label
    ]

    assert example.has_candidate_features

    assert len(
        example.candidate_features
    ) == decision_input.action_dim

    assert (
        example.candidate_features[
            example.label
        ][0]
        == 0.0
    )


def test_recording_search_agent_records_no_trade_proposal():
    from catanlab.board import Board
    from catanlab.economy import PlayerInventory
    from catanlab.rl_teacher import (
        TeacherDecisionKind,
    )
    from catanlab.simulation import PlayerState

    (
        _,
        _,
        _,
        bank,
        deck,
    ) = make_fixture()

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

    agent = RecordingSearchAgent(
        StrategyType.FIVE_RESOURCE,
        search_domestic_trades=True,
    )

    offer = agent.propose_player_trade(
        board,
        players,
        players[0],
        inventories,
        bank=bank,
        dev_deck=deck,
    )

    assert offer is None

    assert len(agent.v2_examples) == 1

    example = agent.v2_examples[0]

    assert (
        example.decision_kind
        == TeacherDecisionKind.TRADE_PROPOSAL
    )

    # Proposal vocab begins with explicit NO_PROPOSAL.
    assert example.label == 0
    assert example.legal_mask[0]


def test_recording_search_agent_records_trade_accept():
    from catanlab.board import Board, Vertex
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.rl_teacher import (
        TeacherDecisionKind,
    )
    from catanlab.simulation import PlayerState
    from catanlab.trading import TradeOffer

    (
        _,
        _,
        _,
        bank,
        deck,
    ) = make_fixture()

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
        PlayerState(player_id=0),
        PlayerState(
            player_id=1,
            settlements=[0],
        ),
    ]

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    inventories[0].add(Resource.ORE, 1)

    inventories[1].add(Resource.WHEAT, 2)
    inventories[1].add(Resource.ORE, 2)
    inventories[1].add(Resource.WOOD, 1)

    offer = TradeOffer(
        proposer_id=0,
        recipient_id=1,
        give=((Resource.ORE, 1),),
        receive=((Resource.WOOD, 1),),
    )

    agent = RecordingSearchAgent(
        StrategyType.FIVE_RESOURCE,
        search_domestic_trades=True,
    )

    accepted = agent.evaluate_player_trade(
        board,
        players,
        players[1],
        inventories,
        offer,
        bank=bank,
        dev_deck=deck,
    )

    assert accepted

    assert len(agent.v2_examples) == 1

    example = agent.v2_examples[0]

    assert (
        example.decision_kind
        == TeacherDecisionKind.TRADE_RESPONSE
    )

    # False = reject, True = accept.
    assert example.label == 1
    assert example.legal_mask[1]


def test_recording_search_agent_records_trade_reject():
    from catanlab.board import Board, Vertex
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.rl_teacher import (
        TeacherDecisionKind,
    )
    from catanlab.simulation import PlayerState
    from catanlab.trading import TradeOffer

    (
        _,
        _,
        _,
        bank,
        deck,
    ) = make_fixture()

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
        PlayerState(player_id=0),
        PlayerState(
            player_id=1,
            settlements=[0],
        ),
    ]

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    inventories[0].add(Resource.WOOD, 1)

    inventories[1].add(Resource.WHEAT, 2)
    inventories[1].add(Resource.ORE, 2)
    inventories[1].add(Resource.WOOD, 1)

    offer = TradeOffer(
        proposer_id=0,
        recipient_id=1,
        give=((Resource.WOOD, 1),),
        receive=((Resource.ORE, 1),),
    )

    agent = RecordingSearchAgent(
        StrategyType.FIVE_RESOURCE,
        search_domestic_trades=True,
    )

    accepted = agent.evaluate_player_trade(
        board,
        players,
        players[1],
        inventories,
        offer,
        bank=bank,
        dev_deck=deck,
    )

    assert not accepted

    assert len(agent.v2_examples) == 1

    example = agent.v2_examples[0]

    assert (
        example.decision_kind
        == TeacherDecisionKind.TRADE_RESPONSE
    )

    assert example.label == 0
    assert example.legal_mask[0]


def test_recording_search_agent_records_trade_counter():
    from catanlab.board import Board, Vertex
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.rl_special_actions import (
        trade_counter_decision_input,
    )
    from catanlab.rl_teacher import (
        TeacherDecisionKind,
    )
    from catanlab.simulation import PlayerState
    from catanlab.trading import TradeOffer

    (
        _,
        _,
        _,
        bank,
        deck,
    ) = make_fixture()

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
        PlayerState(player_id=0),
        PlayerState(
            player_id=1,
            settlements=[0],
        ),
    ]

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    inventories[1].add(Resource.WHEAT, 2)
    inventories[1].add(Resource.ORE, 2)
    inventories[1].add(Resource.WOOD, 1)

    incoming = TradeOffer(
        proposer_id=0,
        recipient_id=1,
        give=((Resource.SHEEP, 1),),
        receive=((Resource.ORE, 1),),
    )

    attempted = set()

    decision_input = (
        trade_counter_decision_input(
            players,
            players[1],
            inventories[1],
            incoming,
            attempted_offers=attempted,
        )
    )

    agent = RecordingSearchAgent(
        StrategyType.FIVE_RESOURCE,
        search_domestic_trades=True,
    )

    counter = agent.counter_player_trade(
        board,
        players,
        players[1],
        inventories,
        incoming,
        attempted_offers=attempted,
        bank=bank,
        dev_deck=deck,
    )

    expected = TradeOffer(
        proposer_id=1,
        recipient_id=0,
        give=((Resource.WOOD, 1),),
        receive=((Resource.ORE, 1),),
    )

    assert counter == expected

    assert len(agent.v2_examples) == 1

    example = agent.v2_examples[0]

    assert (
        example.decision_kind
        == TeacherDecisionKind.TRADE_COUNTER
    )

    assert (
        decision_input.decode(
            example.label
        )
        == expected
    )

    assert example.legal_mask[
        example.label
    ]


def test_recording_search_agent_records_no_trade_counter():
    from catanlab.board import Board
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.rl_teacher import (
        TeacherDecisionKind,
    )
    from catanlab.simulation import PlayerState
    from catanlab.trading import TradeOffer

    (
        _,
        _,
        _,
        bank,
        deck,
    ) = make_fixture()

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
        1,
    )

    incoming = TradeOffer(
        proposer_id=0,
        recipient_id=1,
        give=((Resource.WOOD, 1),),
        receive=((Resource.ORE, 1),),
    )

    agent = RecordingSearchAgent(
        StrategyType.FIVE_RESOURCE,
        search_domestic_trades=True,
    )

    # Exclude every structurally possible counter by marking
    # all candidate 1-for-1 offers as already attempted.
    from catanlab.rl_special_actions import (
        trade_counter_decision_input,
    )

    initial_input = (
        trade_counter_decision_input(
            players,
            players[1],
            inventories[1],
            incoming,
        )
    )

    attempted = {
        candidate
        for candidate in initial_input.vocabulary
        if candidate is not None
    }

    counter = agent.counter_player_trade(
        board,
        players,
        players[1],
        inventories,
        incoming,
        attempted_offers=attempted,
        bank=bank,
        dev_deck=deck,
    )

    assert counter is None

    assert len(agent.v2_examples) == 1

    example = agent.v2_examples[0]

    assert (
        example.decision_kind
        == TeacherDecisionKind.TRADE_COUNTER
    )

    assert example.label == 0
    assert example.legal_mask[0]


def test_teacher_v2_example_supports_dynamic_candidate_features():
    from catanlab.rl_teacher import (
        TeacherDecisionKind,
        TeacherV2Example,
    )

    example = TeacherV2Example(
        decision_kind=(
            TeacherDecisionKind.DISCARD
        ),
        observation=(0.0, 1.0),
        player_id=0,
        label=1,
        legal_mask=(
            True,
            True,
        ),
        candidate_features=(
            (
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
            (
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
            ),
        ),
    )

    assert example.has_legal_mask
    assert example.has_candidate_features

    assert len(
        example.candidate_features
    ) == 2

    assert (
        example.candidate_features[
            example.label
        ]
        == (
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
        )
    )


def test_teacher_v2_fixed_example_needs_no_candidate_features():
    from catanlab.rl_teacher import (
        TeacherDecisionKind,
        TeacherV2Example,
    )

    example = TeacherV2Example(
        decision_kind=(
            TeacherDecisionKind
            .MONOPOLY_RESOURCE
        ),
        observation=(0.0, 1.0),
        player_id=0,
        label=4,
        legal_mask=(
            True,
            True,
            True,
            True,
            True,
        ),
    )

    assert example.has_legal_mask
    assert not example.has_candidate_features
    assert example.candidate_features is None
