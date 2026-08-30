from catanlab.board import build_random_board
from catanlab.devcards import build_dev_card_deck
from catanlab.economy import (
    PlayerInventory,
    ResourceBank,
)
from catanlab.resources import Resource
from catanlab.search_agent import (
    OneStepLookaheadAgent,
)
from catanlab.simulation import PlayerState
from catanlab.strategies import StrategyType
from catanlab.turns import ActionType


def make_state():
    board = build_random_board(seed=10)

    players = [
        PlayerState(player_id=i)
        for i in range(4)
    ]

    inventories = [
        PlayerInventory()
        for _ in range(4)
    ]

    deck = build_dev_card_deck(seed=20)
    bank = ResourceBank()

    return (
        board,
        players,
        inventories,
        deck,
        bank,
    )


def test_lookahead_agent_passes_with_no_actions():
    (
        board,
        players,
        inventories,
        deck,
        bank,
    ) = make_state()

    agent = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE
    )

    action = agent.choose_action(
        board,
        players,
        players[0],
        inventories[0],
        dev_deck=deck,
        bank=bank,
    )

    assert action.action_type == ActionType.PASS


def test_lookahead_agent_prefers_city_over_pass():
    (
        board,
        players,
        inventories,
        deck,
        bank,
    ) = make_state()

    players[0].settlements.append(0)

    inventories[0].add(
        Resource.WHEAT,
        2,
    )

    inventories[0].add(
        Resource.ORE,
        3,
    )

    agent = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE
    )

    action = agent.choose_action(
        board,
        players,
        players[0],
        inventories[0],
        dev_deck=deck,
        bank=bank,
    )

    assert (
        action.action_type
        == ActionType.BUILD_CITY
    )

    assert action.vertex_id == 0


def test_lookahead_choice_does_not_mutate_live_state():
    (
        board,
        players,
        inventories,
        deck,
        bank,
    ) = make_state()

    players[0].settlements.append(0)

    inventories[0].add(
        Resource.WHEAT,
        2,
    )

    inventories[0].add(
        Resource.ORE,
        3,
    )

    before_settlements = list(
        players[0].settlements
    )

    before_cities = list(
        players[0].cities
    )

    before_wheat = (
        inventories[0].count(
            Resource.WHEAT
        )
    )

    before_ore = (
        inventories[0].count(
            Resource.ORE
        )
    )

    agent = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE
    )

    agent.choose_action(
        board,
        players,
        players[0],
        inventories[0],
        dev_deck=deck,
        bank=bank,
    )

    assert (
        players[0].settlements
        == before_settlements
    )

    assert players[0].cities == before_cities

    assert (
        inventories[0].count(
            Resource.WHEAT
        )
        == before_wheat
    )

    assert (
        inventories[0].count(
            Resource.ORE
        )
        == before_ore
    )


def test_evaluate_actions_returns_ranked_candidates():
    (
        board,
        players,
        inventories,
        deck,
        bank,
    ) = make_state()

    players[0].settlements.append(0)

    inventories[0].add(
        Resource.WHEAT,
        2,
    )

    inventories[0].add(
        Resource.ORE,
        3,
    )

    agent = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE
    )

    decision = agent.evaluate_actions(
        board,
        players,
        players[0],
        inventories[0],
        deck,
        bank,
    )

    assert decision.candidates

    assert (
        decision.action
        == decision.candidates[0].action
    )

    assert (
        decision.value
        == decision.candidates[0].value
    )

    values = [
        candidate.value
        for candidate in decision.candidates
    ]

    assert values == sorted(
        values,
        reverse=True,
    )


def test_choose_action_matches_search_decision():
    (
        board,
        players,
        inventories,
        deck,
        bank,
    ) = make_state()

    players[0].settlements.append(0)

    inventories[0].add(
        Resource.WHEAT,
        2,
    )

    inventories[0].add(
        Resource.ORE,
        3,
    )

    agent = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE
    )

    decision = agent.evaluate_actions(
        board,
        players,
        players[0],
        inventories[0],
        deck,
        bank,
    )

    action = agent.choose_action(
        board,
        players,
        players[0],
        inventories[0],
        dev_deck=deck,
        bank=bank,
    )

    assert action == decision.action


def test_search_depth_must_be_positive():
    import pytest

    with pytest.raises(
        ValueError,
        match="search_depth must be at least 1",
    ):
        OneStepLookaheadAgent(
            StrategyType.FIVE_RESOURCE,
            search_depth=0,
        )


def test_depth_one_has_no_continuations():
    (
        board,
        players,
        inventories,
        deck,
        bank,
    ) = make_state()

    players[0].settlements.append(0)

    inventories[0].add(
        Resource.WHEAT,
        2,
    )
    inventories[0].add(
        Resource.ORE,
        3,
    )

    agent = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE,
        search_depth=1,
    )

    decision = agent.evaluate_actions(
        board,
        players,
        players[0],
        inventories[0],
        deck,
        bank,
    )

    assert all(
        candidate.continuation == ()
        for candidate in decision.candidates
    )


def test_depth_two_records_best_continuation():
    (
        board,
        players,
        inventories,
        deck,
        bank,
    ) = make_state()

    players[0].settlements.append(0)

    inventories[0].add(
        Resource.WHEAT,
        2,
    )
    inventories[0].add(
        Resource.ORE,
        3,
    )

    agent = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE,
        search_depth=2,
    )

    decision = agent.evaluate_actions(
        board,
        players,
        players[0],
        inventories[0],
        deck,
        bank,
    )

    assert decision.principal_variation
    assert (
        decision.principal_variation[0]
        == decision.action
    )

    assert (
        len(decision.principal_variation)
        <= 2
    )

    pass_candidate = next(
        candidate
        for candidate in decision.candidates
        if (
            candidate.action.action_type
            == ActionType.PASS
        )
    )

    # PASS terminates the search line rather than
    # allowing impossible PASS -> action sequences.
    assert pass_candidate.continuation == ()


def test_depth_two_choice_does_not_mutate_live_state():
    (
        board,
        players,
        inventories,
        deck,
        bank,
    ) = make_state()

    players[0].settlements.append(0)

    inventories[0].add(
        Resource.WHEAT,
        2,
    )
    inventories[0].add(
        Resource.ORE,
        3,
    )

    before_settlements = list(
        players[0].settlements
    )
    before_cities = list(
        players[0].cities
    )
    before_deck = list(
        deck.cards
    )
    before_bank = dict(
        bank.resources
    )

    agent = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE,
        search_depth=2,
    )

    agent.evaluate_actions(
        board,
        players,
        players[0],
        inventories[0],
        deck,
        bank,
    )

    assert (
        players[0].settlements
        == before_settlements
    )
    assert players[0].cities == before_cities
    assert deck.cards == before_deck
    assert dict(bank.resources) == before_bank


def test_buy_dev_search_value_does_not_depend_on_hidden_top_card():
    from catanlab.devcards import DevCardType
    from catanlab.resources import Resource
    from catanlab.turns import ActionType

    (
        board,
        players,
        inventories,
        deck,
        bank,
    ) = make_state()

    inventories[0].add(
        Resource.SHEEP,
        1,
    )
    inventories[0].add(
        Resource.WHEAT,
        1,
    )
    inventories[0].add(
        Resource.ORE,
        1,
    )

    agent = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE,
        search_depth=1,
    )

    deck.cards[-1] = (
        DevCardType.VICTORY_POINT
    )

    decision_vp = agent.evaluate_actions(
        board,
        players,
        players[0],
        inventories[0],
        deck,
        bank,
    )

    buy_vp = next(
        candidate
        for candidate
        in decision_vp.candidates
        if (
            candidate.action.action_type
            == ActionType.BUY_DEV_CARD
        )
    )

    deck.cards[-1] = (
        DevCardType.KNIGHT
    )

    decision_knight = agent.evaluate_actions(
        board,
        players,
        players[0],
        inventories[0],
        deck,
        bank,
    )

    buy_knight = next(
        candidate
        for candidate
        in decision_knight.candidates
        if (
            candidate.action.action_type
            == ActionType.BUY_DEV_CARD
        )
    )

    assert (
        buy_vp.value
        == buy_knight.value
    )


def test_public_dev_history_changes_buy_dev_expected_value():
    from catanlab.devcards import DevCardType
    from catanlab.resources import Resource
    from catanlab.turns import ActionType

    (
        board,
        players,
        inventories,
        deck,
        bank,
    ) = make_state()

    inventories[0].add(
        Resource.SHEEP,
        1,
    )
    inventories[0].add(
        Resource.WHEAT,
        1,
    )
    inventories[0].add(
        Resource.ORE,
        1,
    )

    agent = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE,
        search_depth=1,
    )

    before = agent.evaluate_actions(
        board,
        players,
        players[0],
        inventories[0],
        deck,
        bank,
    )

    before_buy = next(
        candidate
        for candidate
        in before.candidates
        if (
            candidate.action.action_type
            == ActionType.BUY_DEV_CARD
        )
    )

    # Publicly revealing a Knight changes the
    # information-safe belief distribution.
    players[1].played_dev_cards.append(
        DevCardType.KNIGHT.value
    )

    after = agent.evaluate_actions(
        board,
        players,
        players[0],
        inventories[0],
        deck,
        bank,
    )

    after_buy = next(
        candidate
        for candidate
        in after.candidates
        if (
            candidate.action.action_type
            == ActionType.BUY_DEV_CARD
        )
    )

    assert (
        before_buy.value
        != after_buy.value
    )


def test_search_cache_key_ignores_hidden_deck_order():
    from catanlab.devcards import DevCardType

    (
        board,
        players,
        inventories,
        deck,
        bank,
    ) = make_state()

    agent = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE,
        search_depth=2,
    )

    state_a = agent._make_search_state(
        board,
        players,
        players[0],
        inventories[0],
        deck,
        bank,
    )

    state_b = state_a.clone()

    state_a.dev_deck.cards[-1] = (
        DevCardType.KNIGHT
    )

    state_b.dev_deck.cards[-1] = (
        DevCardType.VICTORY_POINT
    )

    assert agent._state_key(
        state_a,
        0,
    ) == agent._state_key(
        state_b,
        0,
    )


def test_search_cache_key_ignores_opponent_hidden_dev_identity():
    from catanlab.devcards import DevCardType

    (
        board,
        players,
        inventories,
        deck,
        bank,
    ) = make_state()

    agent = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE,
        search_depth=2,
    )

    state_a = agent._make_search_state(
        board,
        players,
        players[0],
        inventories[0],
        deck,
        bank,
    )

    state_b = state_a.clone()

    state_a.players[1].dev_cards = [
        DevCardType.KNIGHT.value,
    ]

    state_b.players[1].dev_cards = [
        DevCardType.VICTORY_POINT.value,
    ]

    assert agent._state_key(
        state_a,
        0,
    ) == agent._state_key(
        state_b,
        0,
    )


def test_transposition_cache_preserves_search_result():
    from catanlab.resources import Resource

    (
        board,
        players,
        inventories,
        deck,
        bank,
    ) = make_state()

    # Give the player enough flexibility for multiple
    # search branches and possible transpositions.
    for resource in Resource:
        if resource.value == "desert":
            continue

        inventories[0].add(
            resource,
            3,
        )

    cached = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE,
        search_depth=2,
        use_transposition_cache=True,
    )

    uncached = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE,
        search_depth=2,
        use_transposition_cache=False,
    )

    cached_decision = cached.evaluate_actions(
        board,
        players,
        players[0],
        inventories[0],
        deck,
        bank,
    )

    uncached_decision = uncached.evaluate_actions(
        board,
        players,
        players[0],
        inventories[0],
        deck,
        bank,
    )

    assert (
        cached_decision.action
        == uncached_decision.action
    )

    assert (
        cached_decision.value
        == uncached_decision.value
    )

    assert (
        cached_decision.principal_variation
        == uncached_decision.principal_variation
    )

    assert len(
        cached_decision.candidates
    ) == len(
        uncached_decision.candidates
    )

    for cached_candidate, uncached_candidate in zip(
        cached_decision.candidates,
        uncached_decision.candidates,
    ):
        assert (
            cached_candidate.action
            == uncached_candidate.action
        )

        assert (
            cached_candidate.value
            == uncached_candidate.value
        )


def test_depth_two_search_uses_maritime_trade_to_enable_city():
    from catanlab.economy import PlayerInventory
    from catanlab.ports import best_maritime_ratio
    from catanlab.resources import Resource
    from catanlab.turns import ActionType

    (
        board,
        players,
        inventories,
        deck,
        bank,
    ) = make_state()

    player = players[0]

    # Give the player one existing settlement so a city
    # upgrade is a legal candidate.
    player.settlements.append(
        board.vertices[0].id
    )

    inventory = PlayerInventory()

    # City costs 2 wheat + 3 ore.
    #
    # Start exactly one ore short, while holding enough
    # surplus wood for one maritime trade.
    inventory.add(
        Resource.WHEAT,
        2,
    )

    inventory.add(
        Resource.ORE,
        2,
    )

    ratio = best_maritime_ratio(
        board,
        player,
        Resource.WOOD,
    )

    inventory.add(
        Resource.WOOD,
        ratio,
    )

    agent = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE,
        search_depth=2,
        use_transposition_cache=False,
        search_maritime_trades=True,
    )

    decision = agent.evaluate_actions(
        board,
        players,
        player,
        inventory,
        deck,
        bank,
    )

    assert (
        decision.action.action_type
        == ActionType.MARITIME_TRADE
    )

    assert (
        decision.action.give_resource
        == Resource.WOOD
    )

    assert (
        decision.action.receive_resource
        == Resource.ORE
    )

    assert len(
        decision.principal_variation
    ) >= 2

    assert (
        decision.principal_variation[0].action_type
        == ActionType.MARITIME_TRADE
    )

    assert (
        decision.principal_variation[1].action_type
        == ActionType.BUILD_CITY
    )


def test_year_of_plenty_search_enables_city():
    from catanlab.devcard_policy import DevCardPhase
    from catanlab.devcards import DevCardType
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.search import (
        apply_search_year_of_plenty,
    )
    from catanlab.turns import ActionType

    (
        board,
        players,
        inventories,
        deck,
        bank,
    ) = make_state()

    player = players[0]

    # A city upgrade must have an existing settlement.
    player.settlements.append(
        board.vertices[0].id
    )

    # Make Year of Plenty playable. Do not put it in
    # new_dev_cards; this represents a card bought on
    # an earlier turn.
    player.dev_cards.append(
        DevCardType.YEAR_OF_PLENTY.value
    )

    inventory = PlayerInventory()

    # City costs 2 wheat + 3 ore.
    #
    # Start with the complete wheat requirement but
    # exactly two ore short. The economically decisive
    # YOP choice should therefore be ORE + ORE.
    inventory.add(
        Resource.WHEAT,
        2,
    )

    inventory.add(
        Resource.ORE,
        1,
    )

    inventories[0] = inventory

    agent = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE,
        search_depth=2,
        use_transposition_cache=False,

        # Keep this test focused specifically on YOP.
        search_maritime_trades=False,

        search_year_of_plenty=True,
    )

    dev_decision = agent.choose_dev_card_play(
        board,
        players,
        player,
        inventories,
        DevCardPhase.POST_ROLL,
        dev_deck=deck,
        bank=bank,
    )

    assert (
        dev_decision.card
        == DevCardType.YEAR_OF_PLENTY
    )

    # Search decides PLAY vs HOLD, while the
    # specialized hook owns the selected arguments.
    assert dev_decision.resources is None

    resources = (
        agent.choose_year_of_plenty_resources(
            board,
            players,
            inventories,
            player,
            bank=bank,
            suggested_resources=(
                dev_decision.resources
            ),
        )
    )

    assert resources == (
        Resource.ORE,
        Resource.ORE,
    )

    # Pending choices are single-use.
    assert (
        agent._pending_year_of_plenty_resources
        is None
    )

    # Reconstruct the state the YOP search evaluated.
    state = agent._make_search_state(
        board,
        players,
        player,
        inventory,
        deck,
        bank,
    )

    after_plenty = (
        apply_search_year_of_plenty(
            state,
            player.player_id,
            *resources,
        )
    )

    # The selected pair should make the city immediately
    # affordable.
    assert (
        after_plenty.inventories[
            player.player_id
        ].count(Resource.WHEAT)
        == 2
    )

    assert (
        after_plenty.inventories[
            player.player_id
        ].count(Resource.ORE)
        == 3
    )

    ordinary_decision = agent.evaluate_actions(
        after_plenty.board,
        after_plenty.players,
        after_plenty.players[
            player.player_id
        ],
        after_plenty.inventories[
            player.player_id
        ],
        after_plenty.dev_deck,
        after_plenty.bank,
    )

    assert (
        ordinary_decision.action.action_type
        == ActionType.BUILD_CITY
    )


def test_year_of_plenty_search_holds_when_resources_add_no_value():
    from catanlab.devcard_policy import DevCardPhase
    from catanlab.devcards import DevCardType
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource

    (
        board,
        players,
        inventories,
        deck,
        bank,
    ) = make_state()

    player = players[0]

    player.dev_cards.append(
        DevCardType.YEAR_OF_PLENTY.value
    )

    inventory = PlayerInventory()

    # At depth two, this is comfortably enough of every
    # resource to make two additional YOP resources
    # irrelevant to ordinary-action affordability.
    for resource in (
        Resource.WOOD,
        Resource.BRICK,
        Resource.SHEEP,
        Resource.WHEAT,
        Resource.ORE,
    ):
        inventory.add(
            resource,
            10,
        )

    inventories[0] = inventory

    agent = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE,
        search_depth=2,
        use_transposition_cache=False,
        search_maritime_trades=False,
        search_year_of_plenty=True,
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

    assert decision.card is None
    assert decision.resources is None


def test_year_of_plenty_search_does_not_play_new_card():
    from catanlab.devcard_policy import DevCardPhase
    from catanlab.devcards import DevCardType
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource

    (
        board,
        players,
        inventories,
        deck,
        bank,
    ) = make_state()

    player = players[0]

    card = DevCardType.YEAR_OF_PLENTY.value

    player.dev_cards.append(card)
    player.new_dev_cards.append(card)

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

    agent = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE,
        search_depth=2,
        use_transposition_cache=False,
        search_maritime_trades=False,
        search_year_of_plenty=True,
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

    assert decision.card is None


def test_year_of_plenty_search_preserves_pre_roll_policy():
    from catanlab.devcard_policy import DevCardPhase
    from catanlab.devcards import DevCardType
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.turns import AdaptiveStrategyAgent

    (
        board,
        players,
        inventories,
        deck,
        bank,
    ) = make_state()

    player = players[0]

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

    search_agent = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE,
        search_depth=2,
        use_transposition_cache=False,
        search_maritime_trades=True,
        search_year_of_plenty=True,
    )

    baseline_agent = AdaptiveStrategyAgent(
        StrategyType.FIVE_RESOURCE
    )

    expected = (
        baseline_agent.choose_dev_card_play(
            board,
            players,
            player,
            inventories,
            DevCardPhase.PRE_ROLL,
        )
    )

    actual = (
        search_agent.choose_dev_card_play(
            board,
            players,
            player,
            inventories,
            DevCardPhase.PRE_ROLL,
            dev_deck=deck,
            bank=bank,
        )
    )

    assert actual == expected


def test_year_of_plenty_search_preserves_other_dev_card_choice():
    from catanlab.devcard_policy import DevCardPhase
    from catanlab.devcards import DevCardType

    (
        board,
        players,
        inventories,
        deck,
        bank,
    ) = make_state()

    player = players[0]

    player.dev_cards.extend(
        [
            DevCardType.KNIGHT.value,
            DevCardType.YEAR_OF_PLENTY.value,
        ]
    )

    player.knights_played = 2

    agent = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE,
        search_depth=2,
        use_transposition_cache=False,
        search_maritime_trades=True,
        search_year_of_plenty=True,
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
        == DevCardType.KNIGHT
    )


def test_road_building_search_enables_settlement():
    from catanlab.board import Board, Edge, Vertex
    from catanlab.devcard_policy import DevCardPhase
    from catanlab.devcards import DevCardType
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState
    from catanlab.turns import ActionType

    (
        _,
        _,
        _,
        deck,
        bank,
    ) = make_state()

    # Linear network:
    #
    # settlement
    #    0 ----- 1 ----- 2
    #
    # Two free roads reach vertex 2, which satisfies
    # the distance rule from the settlement at 0.
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

    # Already possess the complete settlement cost.
    # The only missing ingredient is road connectivity
    # to a legal new settlement vertex.
    inventory.add(Resource.WOOD, 1)
    inventory.add(Resource.BRICK, 1)
    inventory.add(Resource.SHEEP, 1)
    inventory.add(Resource.WHEAT, 1)

    inventories = [inventory]

    agent = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE,
        search_depth=2,
        use_transposition_cache=False,
        search_maritime_trades=False,
        search_year_of_plenty=False,
        search_road_building=True,
    )

    dev_decision = agent.choose_dev_card_play(
        board,
        players,
        player,
        inventories,
        DevCardPhase.POST_ROLL,
        dev_deck=deck,
        bank=bank,
    )

    assert (
        dev_decision.card
        == DevCardType.ROAD_BUILDING
    )

    assert dev_decision.road_edges is None

    road_edges = (
        agent.choose_road_building_edges(
            board,
            players,
            inventories,
            player,
            suggested_edges=(
                dev_decision.road_edges
            ),
        )
    )

    assert road_edges == (
        (0, 1),
        (1, 2),
    )

    assert (
        agent._pending_road_building_edges
        is None
    )

    from catanlab.search import (
        apply_search_road_building,
    )

    after_roads = (
        apply_search_road_building(
            agent._make_search_state(
                board,
                players,
                player,
                inventory,
                deck,
                bank,
            ),
            0,
            *road_edges,
        )
    )

    ordinary = agent.evaluate_actions(
        after_roads.board,
        after_roads.players,
        after_roads.players[0],
        after_roads.inventories[0],
        after_roads.dev_deck,
        after_roads.bank,
    )

    assert (
        ordinary.action.action_type
        == ActionType.BUILD_SETTLEMENT
    )

    assert ordinary.action.vertex_id == 2


def test_monopoly_search_prefers_resource_that_enables_city():
    from catanlab.board import Board, Tile, Vertex
    from catanlab.devcard_policy import DevCardPhase
    from catanlab.devcards import DevCardType
    from catanlab.economy import PlayerInventory
    from catanlab.graph import HexCoord
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState
    from catanlab.turns import ActionType

    (
        _,
        _,
        _,
        deck,
        bank,
    ) = make_state()

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

    # City requires 2 wheat + 3 ore.
    # We already have wheat and one ore.
    own_inventory.add(
        Resource.WHEAT,
        2,
    )
    own_inventory.add(
        Resource.ORE,
        1,
    )

    opponent_inventory = PlayerInventory()

    # Only the hand SIZE should matter to search.
    # Actual hidden composition deliberately disagrees
    # with the public production-based belief.
    opponent_inventory.add(
        Resource.WOOD,
        2,
    )

    inventories = [
        own_inventory,
        opponent_inventory,
    ]

    agent = OneStepLookaheadAgent(
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

    resource = agent.choose_monopoly_resource(
        board,
        players,
        inventories,
        player,
        suggested_resource=decision.resource,
    )

    assert resource == Resource.ORE

    assert (
        agent._pending_monopoly_resource
        is None
    )

    from catanlab.search import (
        apply_search_monopoly_outcome,
    )

    after_monopoly = (
        apply_search_monopoly_outcome(
            agent._make_search_state(
                board,
                players,
                player,
                own_inventory,
                deck,
                bank,
            ),
            0,
            Resource.ORE,
            2,
        )
    )

    ordinary = agent.evaluate_actions(
        after_monopoly.board,
        after_monopoly.players,
        after_monopoly.players[0],
        after_monopoly.inventories[0],
        after_monopoly.dev_deck,
        after_monopoly.bank,
    )

    assert (
        ordinary.action.action_type
        == ActionType.BUILD_CITY
    )


def test_monopoly_search_ignores_opponent_hidden_composition():
    from catanlab.board import Board, Tile, Vertex
    from catanlab.devcard_policy import DevCardPhase
    from catanlab.devcards import DevCardType
    from catanlab.economy import PlayerInventory
    from catanlab.graph import HexCoord
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState

    (
        _,
        _,
        _,
        deck,
        bank,
    ) = make_state()

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

    def make_world(hidden_resource):
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

        own = PlayerInventory()
        own.add(Resource.WHEAT, 2)
        own.add(Resource.ORE, 1)

        hidden = PlayerInventory()
        hidden.add(hidden_resource, 2)

        return (
            [player, opponent],
            [own, hidden],
        )

    players_a, inventories_a = (
        make_world(Resource.WOOD)
    )

    players_b, inventories_b = (
        make_world(Resource.ORE)
    )

    agent_a = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE,
        search_depth=2,
        use_transposition_cache=False,
        search_maritime_trades=False,
        search_year_of_plenty=False,
        search_road_building=False,
        search_monopoly=True,
    )

    agent_b = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE,
        search_depth=2,
        use_transposition_cache=False,
        search_maritime_trades=False,
        search_year_of_plenty=False,
        search_road_building=False,
        search_monopoly=True,
    )

    decision_a = agent_a.choose_dev_card_play(
        board,
        players_a,
        players_a[0],
        inventories_a,
        DevCardPhase.POST_ROLL,
        dev_deck=deck,
        bank=bank,
    )

    decision_b = agent_b.choose_dev_card_play(
        board,
        players_b,
        players_b[0],
        inventories_b,
        DevCardPhase.POST_ROLL,
        dev_deck=deck,
        bank=bank,
    )

    assert decision_a == decision_b


def test_search_robber_decisions_disabled_preserves_baseline():
    from catanlab.board import (
        Board,
        Tile,
    )
    from catanlab.economy import PlayerInventory
    from catanlab.graph import HexCoord
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState
    from catanlab.strategies import StrategyType
    from catanlab.search_agent import OneStepLookaheadAgent

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

    player = PlayerState(player_id=0)
    inventories = [PlayerInventory()]

    agent = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE,
        search_robber_decisions=False,
    )

    assert (
        agent.choose_robber_tile(
            board,
            [player],
            inventories,
            player,
        )
        == 1
    )


def test_search_robber_prefers_high_threat_opponent():
    from catanlab.board import (
        Board,
        Tile,
        Vertex,
    )
    from catanlab.economy import PlayerInventory
    from catanlab.graph import HexCoord
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState
    from catanlab.strategies import StrategyType
    from catanlab.search_agent import OneStepLookaheadAgent

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
            Tile(
                id=2,
                coord=HexCoord(2, 0),
                resource=Resource.WHEAT,
                number=6,
            ),
        ],
        vertices=[
            Vertex(
                id=0,
                position=(1.0, 0.0),
                adjacent_tiles=[1],
            ),
            Vertex(
                id=1,
                position=(2.0, 0.0),
                adjacent_tiles=[2],
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
            settlements=[1],
            cities=[],
        ),
    ]

    # Give player 2 a much larger public threat level.
    players[2].settlements.extend(
        [10, 11, 12, 13, 14, 15, 16]
    )

    # The extra settlement IDs above are only intended
    # to affect public VP, but must not be dereferenced
    # by the robber evaluator. Restore the actual board
    # structure list and set public score through cities
    # would require additional vertices, so instead use
    # the simpler city target distinction below.
    players[2].settlements = [1]
    players[2].cities = [1]

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
        PlayerInventory(),
    ]

    inventories[1].add(Resource.WOOD)
    inventories[2].add(Resource.ORE)

    agent = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE,
        search_robber_decisions=True,
    )

    assert (
        agent.choose_robber_tile(
            board,
            players,
            inventories,
            players[0],
        )
        == 2
    )


def test_search_robber_victim_uses_public_threat():
    from catanlab.board import (
        Board,
        Tile,
        Vertex,
    )
    from catanlab.economy import PlayerInventory
    from catanlab.graph import HexCoord
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState
    from catanlab.strategies import StrategyType
    from catanlab.search_agent import OneStepLookaheadAgent

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

    inventories[1].add(Resource.WOOD, 6)
    inventories[2].add(Resource.ORE, 1)

    agent = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE,
        search_robber_decisions=True,
    )

    assert (
        agent.choose_robber_victim(
            board,
            players,
            inventories,
            players[0],
        )
        == 2
    )


def test_search_discard_decisions_disabled_uses_adaptive_policy():
    from catanlab.board import Board
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.search_agent import OneStepLookaheadAgent
    from catanlab.simulation import PlayerState
    from catanlab.strategies import StrategyType

    board = Board(
        tiles=[],
        vertices=[],
        edges=[],
    )

    player = PlayerState(
        player_id=0,
    )

    inventory = PlayerInventory()
    inventory.add(
        Resource.WOOD,
        4,
    )
    inventory.add(
        Resource.ORE,
        4,
    )

    agent = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE,
        search_discard_decisions=False,
    )

    expected = agent.choose_discards(
        player,
        inventory,
        4,
    )

    actual = (
        agent.choose_discards_with_context(
            board,
            [player],
            [inventory],
            player,
            inventory,
            4,
        )
    )

    assert actual == expected


def test_search_discard_preserves_city_ready_resources():
    from catanlab.board import (
        Board,
        Vertex,
    )
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.search_agent import OneStepLookaheadAgent
    from catanlab.simulation import PlayerState
    from catanlab.strategies import StrategyType

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

    # Eight cards -> four must be discarded.
    #
    # ORE + WHEAT are the valuable city-progress
    # resources, while WOOD + BRICK are expendable in
    # this isolated board state.
    inventory.add(
        Resource.ORE,
        2,
    )
    inventory.add(
        Resource.WHEAT,
        2,
    )
    inventory.add(
        Resource.WOOD,
        2,
    )
    inventory.add(
        Resource.BRICK,
        2,
    )

    agent = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE,
        search_discard_decisions=True,
    )

    discarded = (
        agent.choose_discards_with_context(
            board,
            [player],
            [inventory],
            player,
            inventory,
            4,
        )
    )

    assert len(discarded) == 4

    assert Resource.ORE not in discarded
    assert Resource.WHEAT not in discarded

    assert discarded.count(
        Resource.WOOD
    ) == 2

    assert discarded.count(
        Resource.BRICK
    ) == 2


def test_search_discard_returns_only_cards_actually_held():
    from catanlab.board import Board
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.search_agent import OneStepLookaheadAgent
    from catanlab.simulation import PlayerState
    from catanlab.strategies import StrategyType

    board = Board(
        tiles=[],
        vertices=[],
        edges=[],
    )

    player = PlayerState(
        player_id=0,
    )

    inventory = PlayerInventory()
    inventory.add(
        Resource.SHEEP,
        5,
    )
    inventory.add(
        Resource.WHEAT,
        3,
    )

    agent = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE,
        search_discard_decisions=True,
    )

    discarded = (
        agent.choose_discards_with_context(
            board,
            [player],
            [inventory],
            player,
            inventory,
            4,
        )
    )

    assert len(discarded) == 4

    assert set(discarded) <= {
        Resource.SHEEP,
        Resource.WHEAT,
    }

    assert (
        discarded.count(Resource.SHEEP)
        <= 5
    )

    assert (
        discarded.count(Resource.WHEAT)
        <= 3
    )


def test_search_dev_card_pending_choice_is_single_use():
    from catanlab.board import Board
    from catanlab.economy import PlayerInventory
    from catanlab.resources import Resource
    from catanlab.search_agent import OneStepLookaheadAgent
    from catanlab.simulation import PlayerState
    from catanlab.strategies import StrategyType

    board = Board(
        tiles=[],
        vertices=[],
        edges=[],
    )

    player = PlayerState(
        player_id=0,
    )

    inventories = [
        PlayerInventory(),
    ]

    agent = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE,
    )

    agent._pending_monopoly_resource = (
        Resource.ORE
    )

    first = agent.choose_monopoly_resource(
        board,
        [player],
        inventories,
        player,
        suggested_resource=Resource.WOOD,
    )

    assert first == Resource.ORE

    assert (
        agent._pending_monopoly_resource
        is None
    )

    second = agent.choose_monopoly_resource(
        board,
        [player],
        inventories,
        player,
        suggested_resource=Resource.WOOD,
    )

    assert second == Resource.WOOD
