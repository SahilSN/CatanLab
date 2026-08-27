from catanlab.board import build_random_board
from catanlab.devcards import build_dev_card_deck
from catanlab.economy import (
    PlayerInventory,
    ResourceBank,
)
from catanlab.resources import Resource
from catanlab.search import (
    SearchState,
    apply_search_action,
    clone_search_state,
    enumerate_search_actions,
)
from catanlab.simulation import PlayerState


def make_search_state() -> SearchState:
    board = build_random_board(seed=123)

    players = [
        PlayerState(player_id=i)
        for i in range(4)
    ]

    inventories = [
        PlayerInventory()
        for _ in range(4)
    ]

    inventories[0].add(
        Resource.WOOD,
        3,
    )

    players[0].settlements.append(0)
    players[0].roads.append((0, 1))

    return SearchState(
        board=board,
        players=players,
        inventories=inventories,
        dev_deck=build_dev_card_deck(
            seed=456
        ),
        bank=ResourceBank(),
    )


def test_clone_search_state_is_distinct():
    original = make_search_state()
    cloned = clone_search_state(original)

    assert cloned is not original
    assert cloned.board is not original.board
    assert cloned.players is not original.players
    assert (
        cloned.inventories
        is not original.inventories
    )
    assert (
        cloned.dev_deck
        is not original.dev_deck
    )
    assert cloned.bank is not original.bank


def test_clone_player_mutation_is_isolated():
    original = make_search_state()
    cloned = original.clone()

    cloned.players[0].roads.append(
        (1, 2)
    )

    cloned.players[0].settlements.append(
        3
    )

    assert original.players[0].roads == [
        (0, 1)
    ]

    assert original.players[0].settlements == [
        0
    ]


def test_clone_inventory_mutation_is_isolated():
    original = make_search_state()
    cloned = original.clone()

    cloned.inventories[0].remove(
        Resource.WOOD,
        2,
    )

    assert (
        cloned.inventories[0].count(
            Resource.WOOD
        )
        == 1
    )

    assert (
        original.inventories[0].count(
            Resource.WOOD
        )
        == 3
    )


def test_clone_robber_mutation_is_isolated():
    original = make_search_state()
    cloned = original.clone()

    original_robber = (
        original.board.robber_tile_id
    )

    alternate = next(
        tile.id
        for tile in cloned.board.tiles
        if tile.id != original_robber
    )

    cloned.board.robber_tile_id = alternate

    assert (
        original.board.robber_tile_id
        == original_robber
    )


def test_clone_dev_deck_mutation_is_isolated():
    original = make_search_state()
    cloned = original.clone()

    original_count = len(
        original.dev_deck.cards
    )

    cloned.dev_deck.cards.pop()

    assert (
        len(cloned.dev_deck.cards)
        == original_count - 1
    )

    assert (
        len(original.dev_deck.cards)
        == original_count
    )


def test_clone_bank_mutation_is_isolated():
    original = make_search_state()
    cloned = original.clone()

    original_wood = original.bank.count(
        Resource.WOOD
    )

    cloned.bank.remove(
        Resource.WOOD,
        2,
    )

    assert (
        cloned.bank.count(Resource.WOOD)
        == original_wood - 2
    )

    assert (
        original.bank.count(Resource.WOOD)
        == original_wood
    )


def test_enumerate_search_actions_always_includes_pass():
    from catanlab.search import (
        enumerate_search_actions,
    )
    from catanlab.turns import ActionType

    state = make_search_state()

    actions = enumerate_search_actions(
        state,
        player_id=0,
    )

    assert any(
        action.action_type
        == ActionType.PASS
        for action in actions
    )


def test_enumerate_search_actions_includes_dev_purchase():
    from catanlab.economy import BuildType
    from catanlab.search import (
        enumerate_search_actions,
    )
    from catanlab.turns import ActionType

    state = make_search_state()

    cost = {
        Resource.SHEEP: 1,
        Resource.WHEAT: 1,
        Resource.ORE: 1,
    }

    for resource, amount in cost.items():
        state.inventories[0].add(
            resource,
            amount,
        )

    assert state.inventories[0].can_afford(
        BuildType.DEV_CARD
    )

    actions = enumerate_search_actions(
        state,
        player_id=0,
    )

    assert any(
        action.action_type
        == ActionType.BUY_DEV_CARD
        for action in actions
    )


def test_apply_search_action_does_not_mutate_original():
    from catanlab.search import (
        apply_search_action,
    )
    from catanlab.turns import (
        ActionType,
        TurnAction,
    )

    state = make_search_state()

    state.inventories[0].add(
        Resource.WHEAT,
        2,
    )
    state.inventories[0].add(
        Resource.ORE,
        3,
    )

    original_settlements = list(
        state.players[0].settlements
    )
    original_cities = list(
        state.players[0].cities
    )

    next_state = apply_search_action(
        state,
        player_id=0,
        action=TurnAction(
            action_type=(
                ActionType.BUILD_CITY
            ),
            vertex_id=0,
        ),
    )

    assert state.players[0].settlements == (
        original_settlements
    )

    assert state.players[0].cities == (
        original_cities
    )

    assert 0 not in (
        state.players[0].cities
    )

    assert 0 in (
        next_state.players[0].cities
    )


def test_apply_search_action_spends_resources_on_clone():
    from catanlab.search import (
        apply_search_action,
    )
    from catanlab.turns import (
        ActionType,
        TurnAction,
    )

    state = make_search_state()

    state.inventories[0].add(
        Resource.WHEAT,
        2,
    )
    state.inventories[0].add(
        Resource.ORE,
        3,
    )

    before_wheat = (
        state.inventories[0].count(
            Resource.WHEAT
        )
    )

    before_ore = (
        state.inventories[0].count(
            Resource.ORE
        )
    )

    next_state = apply_search_action(
        state,
        player_id=0,
        action=TurnAction(
            action_type=(
                ActionType.BUILD_CITY
            ),
            vertex_id=0,
        ),
    )

    assert (
        state.inventories[0].count(
            Resource.WHEAT
        )
        == before_wheat
    )

    assert (
        state.inventories[0].count(
            Resource.ORE
        )
        == before_ore
    )

    assert (
        next_state.inventories[0].count(
            Resource.WHEAT
        )
        == before_wheat - 2
    )

    assert (
        next_state.inventories[0].count(
            Resource.ORE
        )
        == before_ore - 3
    )


def test_evaluate_search_state_rewards_city_upgrade():
    from catanlab.search import (
        apply_search_action,
        evaluate_search_state,
    )
    from catanlab.turns import (
        ActionType,
        TurnAction,
    )

    state = make_search_state()

    state.inventories[0].add(
        Resource.WHEAT,
        2,
    )
    state.inventories[0].add(
        Resource.ORE,
        3,
    )

    before = evaluate_search_state(
        state,
        player_id=0,
    )

    next_state = apply_search_action(
        state,
        player_id=0,
        action=TurnAction(
            action_type=(
                ActionType.BUILD_CITY
            ),
            vertex_id=0,
        ),
    )

    after = evaluate_search_state(
        next_state,
        player_id=0,
    )

    assert after > before


def test_search_dev_purchase_hides_card_identity():
    from catanlab.devcards import DevCardType
    from catanlab.search import apply_search_action
    from catanlab.turns import (
        ActionType,
        TurnAction,
    )

    state = make_search_state()

    state.inventories[0].add(
        Resource.SHEEP,
        1,
    )
    state.inventories[0].add(
        Resource.WHEAT,
        1,
    )
    state.inventories[0].add(
        Resource.ORE,
        1,
    )

    # Deliberately put a VP card on top. Search must
    # still not learn that this is what would be drawn.
    state.dev_deck.cards[-1] = (
        DevCardType.VICTORY_POINT
    )

    original_count = len(
        state.dev_deck.cards
    )

    next_state = apply_search_action(
        state,
        player_id=0,
        action=TurnAction(
            action_type=(
                ActionType.BUY_DEV_CARD
            )
        ),
    )

    assert (
        next_state.players[0].dev_cards[-1]
        == "unknown_dev_card"
    )

    assert (
        next_state.players[0].new_dev_cards[-1]
        == "unknown_dev_card"
    )

    assert (
        "victory_point"
        not in next_state.players[0].dev_cards
    )

    assert (
        len(next_state.dev_deck.cards)
        == original_count - 1
    )

    assert (
        len(state.dev_deck.cards)
        == original_count
    )


def test_unknown_dev_purchase_value_does_not_depend_on_hidden_identity():
    from catanlab.devcards import DevCardType
    from catanlab.search import (
        apply_search_action,
        evaluate_search_state,
    )
    from catanlab.turns import (
        ActionType,
        TurnAction,
    )

    vp_state = make_search_state()
    knight_state = make_search_state()

    for state in (
        vp_state,
        knight_state,
    ):
        state.inventories[0].add(
            Resource.SHEEP,
            1,
        )
        state.inventories[0].add(
            Resource.WHEAT,
            1,
        )
        state.inventories[0].add(
            Resource.ORE,
            1,
        )

    vp_state.dev_deck.cards[-1] = (
        DevCardType.VICTORY_POINT
    )

    knight_state.dev_deck.cards[-1] = (
        DevCardType.KNIGHT
    )

    action = TurnAction(
        action_type=(
            ActionType.BUY_DEV_CARD
        )
    )

    vp_after = apply_search_action(
        vp_state,
        player_id=0,
        action=action,
    )

    knight_after = apply_search_action(
        knight_state,
        player_id=0,
        action=action,
    )

    assert (
        evaluate_search_state(
            vp_after,
            player_id=0,
        )
        == evaluate_search_state(
            knight_after,
            player_id=0,
        )
    )


def test_dev_card_belief_starts_with_standard_distribution():
    from catanlab.devcards import DevCardType
    from catanlab.search import (
        build_dev_card_belief,
    )

    players = [
        PlayerState(player_id=i)
        for i in range(4)
    ]

    belief = build_dev_card_belief(
        players,
        player_id=0,
    )

    assert belief.total == 25

    assert belief.counts[
        DevCardType.KNIGHT
    ] == 14

    assert belief.counts[
        DevCardType.VICTORY_POINT
    ] == 5

    assert belief.counts[
        DevCardType.ROAD_BUILDING
    ] == 2

    assert belief.counts[
        DevCardType.YEAR_OF_PLENTY
    ] == 2

    assert belief.counts[
        DevCardType.MONOPOLY
    ] == 2


def test_dev_card_belief_subtracts_own_known_cards():
    from catanlab.devcards import DevCardType
    from catanlab.search import (
        build_dev_card_belief,
    )

    players = [
        PlayerState(
            player_id=0,
            dev_cards=[
                DevCardType.KNIGHT.value,
                DevCardType.VICTORY_POINT.value,
            ],
        ),
        PlayerState(player_id=1),
        PlayerState(player_id=2),
        PlayerState(player_id=3),
    ]

    belief = build_dev_card_belief(
        players,
        player_id=0,
    )

    assert belief.total == 23

    assert belief.counts[
        DevCardType.KNIGHT
    ] == 13

    assert belief.counts[
        DevCardType.VICTORY_POINT
    ] == 4


def test_dev_card_belief_subtracts_public_played_cards():
    from catanlab.devcards import DevCardType
    from catanlab.search import (
        build_dev_card_belief,
    )

    players = [
        PlayerState(
            player_id=0,
            played_dev_cards=[
                DevCardType.KNIGHT.value,
            ],
        ),
        PlayerState(
            player_id=1,
            played_dev_cards=[
                DevCardType.MONOPOLY.value,
                DevCardType.ROAD_BUILDING.value,
            ],
        ),
        PlayerState(player_id=2),
        PlayerState(player_id=3),
    ]

    belief = build_dev_card_belief(
        players,
        player_id=0,
    )

    assert belief.total == 22

    assert belief.counts[
        DevCardType.KNIGHT
    ] == 13

    assert belief.counts[
        DevCardType.MONOPOLY
    ] == 1

    assert belief.counts[
        DevCardType.ROAD_BUILDING
    ] == 1


def test_dev_card_belief_does_not_peek_at_opponent_hidden_cards():
    from catanlab.devcards import DevCardType
    from catanlab.search import (
        build_dev_card_belief,
    )

    players = [
        PlayerState(player_id=0),
        PlayerState(
            player_id=1,
            dev_cards=[
                DevCardType.KNIGHT.value,
                DevCardType.VICTORY_POINT.value,
                DevCardType.MONOPOLY.value,
            ],
        ),
        PlayerState(player_id=2),
        PlayerState(player_id=3),
    ]

    belief = build_dev_card_belief(
        players,
        player_id=0,
    )

    # Opponent identities are hidden, so they must
    # remain in the acting player's unknown pool.
    assert belief.total == 25

    assert belief.counts[
        DevCardType.KNIGHT
    ] == 14

    assert belief.counts[
        DevCardType.VICTORY_POINT
    ] == 5

    assert belief.counts[
        DevCardType.MONOPOLY
    ] == 2


def test_dev_card_belief_probabilities_sum_to_one():
    from catanlab.devcards import DevCardType
    from catanlab.search import (
        build_dev_card_belief,
    )

    players = [
        PlayerState(player_id=i)
        for i in range(4)
    ]

    belief = build_dev_card_belief(
        players,
        player_id=0,
    )

    total_probability = sum(
        belief.probability(card)
        for card in DevCardType
    )

    assert abs(
        total_probability - 1.0
    ) < 1e-12


def test_search_dev_outcome_uses_belief_branch_not_hidden_top_card():
    from catanlab.devcards import DevCardType
    from catanlab.search import (
        apply_search_dev_card_outcome,
    )

    state = make_search_state()

    state.inventories[0].add(
        Resource.SHEEP,
        1,
    )
    state.inventories[0].add(
        Resource.WHEAT,
        1,
    )
    state.inventories[0].add(
        Resource.ORE,
        1,
    )

    # The real hidden top card is deliberately VP.
    state.dev_deck.cards[-1] = (
        DevCardType.VICTORY_POINT
    )

    next_state = apply_search_dev_card_outcome(
        state,
        player_id=0,
        card=DevCardType.KNIGHT,
    )

    # The hypothetical branch is Knight because the
    # search supplied Knight. It did not peek at VP.
    assert (
        next_state.players[0].dev_cards[-1]
        == DevCardType.KNIGHT.value
    )

    assert (
        next_state.players[0].new_dev_cards[-1]
        == DevCardType.KNIGHT.value
    )

    # Original live/search-root state remains untouched.
    assert (
        DevCardType.KNIGHT.value
        not in state.players[0].dev_cards
    )


def test_search_dev_outcome_consumes_one_deck_slot():
    from catanlab.devcards import DevCardType
    from catanlab.search import (
        apply_search_dev_card_outcome,
    )

    state = make_search_state()

    state.inventories[0].add(
        Resource.SHEEP,
        1,
    )
    state.inventories[0].add(
        Resource.WHEAT,
        1,
    )
    state.inventories[0].add(
        Resource.ORE,
        1,
    )

    before = len(state.dev_deck.cards)

    next_state = apply_search_dev_card_outcome(
        state,
        player_id=0,
        card=DevCardType.MONOPOLY,
    )

    assert len(
        next_state.dev_deck.cards
    ) == before - 1

    assert len(
        state.dev_deck.cards
    ) == before


def test_fast_search_clone_matches_deep_clone_state():
    original = make_search_state()

    fast = original.fast_clone_for_ordinary_search()
    deep = original.clone()

    assert fast.players == deep.players
    assert fast.inventories == deep.inventories
    assert fast.dev_deck == deep.dev_deck
    assert fast.bank == deep.bank

    # Board is intentionally shared only by the fast
    # search clone.
    assert fast.board is original.board
    assert deep.board is not original.board


def test_fast_search_clone_mutable_state_is_independent():
    from catanlab.devcards import DevCardType
    from catanlab.resources import Resource

    original = make_search_state()
    cloned = (
        original.fast_clone_for_ordinary_search()
    )

    assert cloned.players is not original.players
    assert (
        cloned.inventories
        is not original.inventories
    )
    assert (
        cloned.dev_deck
        is not original.dev_deck
    )
    assert cloned.bank is not original.bank

    cloned.players[0].settlements.append(
        999
    )
    cloned.players[0].roads.append(
        (998, 999)
    )
    cloned.players[0].dev_cards.append(
        DevCardType.KNIGHT.value
    )

    cloned.inventories[0].add(
        Resource.WOOD,
        1,
    )

    cloned.bank.remove(
        Resource.BRICK,
        1,
    )

    cloned.dev_deck.cards.pop()

    assert (
        999
        not in original.players[0].settlements
    )
    assert (
        (998, 999)
        not in original.players[0].roads
    )
    assert (
        DevCardType.KNIGHT.value
        not in original.players[0].dev_cards
    )

    assert (
        cloned.inventories[0].count(
            Resource.WOOD
        )
        != original.inventories[0].count(
            Resource.WOOD
        )
    )

    assert (
        cloned.bank.count(Resource.BRICK)
        != original.bank.count(Resource.BRICK)
    )

    assert (
        len(cloned.dev_deck.cards)
        == len(original.dev_deck.cards) - 1
    )


def test_fast_clone_player_nested_lists_are_independent():
    original = make_search_state()
    cloned = (
        original.fast_clone_for_ordinary_search()
    )

    for original_player, cloned_player in zip(
        original.players,
        cloned.players,
    ):
        assert (
            cloned_player.settlements
            is not original_player.settlements
        )
        assert (
            cloned_player.cities
            is not original_player.cities
        )
        assert (
            cloned_player.roads
            is not original_player.roads
        )
        assert (
            cloned_player.dev_cards
            is not original_player.dev_cards
        )
        assert (
            cloned_player.new_dev_cards
            is not original_player.new_dev_cards
        )
        assert (
            cloned_player.played_dev_cards
            is not original_player.played_dev_cards
        )


def test_search_maritime_trades_are_opt_in():
    from catanlab.ports import best_maritime_ratio
    from catanlab.resources import Resource
    from catanlab.turns import ActionType

    state = make_search_state()

    player = state.players[0]
    inventory = state.inventories[0]

    inventory.resources.clear()

    ratio = best_maritime_ratio(
        state.board,
        player,
        Resource.WOOD,
    )

    inventory.add(
        Resource.WOOD,
        ratio,
    )

    without_trades = enumerate_search_actions(
        state,
        0,
    )

    with_trades = enumerate_search_actions(
        state,
        0,
        include_maritime_trades=True,
    )

    assert all(
        action.action_type
        != ActionType.MARITIME_TRADE
        for action in without_trades
    )

    trades = [
        action
        for action in with_trades
        if (
            action.action_type
            == ActionType.MARITIME_TRADE
        )
    ]

    assert trades

    assert any(
        action.give_resource == Resource.WOOD
        and action.receive_resource == Resource.ORE
        for action in trades
    )


def test_search_maritime_trade_respects_bank_supply():
    from catanlab.ports import best_maritime_ratio
    from catanlab.resources import Resource
    from catanlab.turns import ActionType

    state = make_search_state()

    player = state.players[0]
    inventory = state.inventories[0]

    inventory.resources.clear()

    ratio = best_maritime_ratio(
        state.board,
        player,
        Resource.WOOD,
    )

    inventory.add(
        Resource.WOOD,
        ratio,
    )

    state.bank.resources[
        Resource.ORE
    ] = 0

    actions = enumerate_search_actions(
        state,
        0,
        include_maritime_trades=True,
    )

    assert not any(
        action.action_type
        == ActionType.MARITIME_TRADE
        and action.receive_resource
        == Resource.ORE
        for action in actions
    )


def test_apply_search_maritime_trade_isolated():
    from catanlab.ports import best_maritime_ratio
    from catanlab.resources import Resource
    from catanlab.turns import (
        ActionType,
        TurnAction,
    )

    state = make_search_state()

    player = state.players[0]
    inventory = state.inventories[0]

    inventory.resources.clear()

    ratio = best_maritime_ratio(
        state.board,
        player,
        Resource.WOOD,
    )

    inventory.add(
        Resource.WOOD,
        ratio,
    )

    original_wood = inventory.count(
        Resource.WOOD
    )
    original_ore = inventory.count(
        Resource.ORE
    )

    result = apply_search_action(
        state,
        0,
        TurnAction(
            action_type=(
                ActionType.MARITIME_TRADE
            ),
            give_resource=Resource.WOOD,
            receive_resource=Resource.ORE,
        ),
    )

    assert (
        result.inventories[0].count(
            Resource.WOOD
        )
        == original_wood - ratio
    )

    assert (
        result.inventories[0].count(
            Resource.ORE
        )
        == original_ore + 1
    )

    # Original search state remains untouched.
    assert (
        state.inventories[0].count(
            Resource.WOOD
        )
        == original_wood
    )

    assert (
        state.inventories[0].count(
            Resource.ORE
        )
        == original_ore
    )


def test_apply_search_year_of_plenty_isolated():
    from catanlab.devcards import DevCardType
    from catanlab.resources import Resource
    from catanlab.search import (
        apply_search_year_of_plenty,
    )

    state = make_search_state()

    player = state.players[0]
    inventory = state.inventories[0]

    player.dev_cards.append(
        DevCardType.YEAR_OF_PLENTY.value
    )

    wood_before = inventory.count(
        Resource.WOOD
    )
    ore_before = inventory.count(
        Resource.ORE
    )

    result = apply_search_year_of_plenty(
        state,
        0,
        Resource.WOOD,
        Resource.ORE,
    )

    assert (
        result.inventories[0].count(
            Resource.WOOD
        )
        == wood_before + 1
    )

    assert (
        result.inventories[0].count(
            Resource.ORE
        )
        == ore_before + 1
    )

    assert (
        state.inventories[0].count(
            Resource.WOOD
        )
        == wood_before
    )

    assert (
        state.inventories[0].count(
            Resource.ORE
        )
        == ore_before
    )

    assert (
        DevCardType.YEAR_OF_PLENTY.value
        not in result.players[0].dev_cards
    )

    assert (
        DevCardType.YEAR_OF_PLENTY.value
        in state.players[0].dev_cards
    )


def test_apply_search_road_building_places_free_roads_isolated():
    from catanlab.devcards import DevCardType
    from catanlab.search import (
        apply_search_road_building,
    )
    from catanlab.turns import legal_road_edges

    state = make_search_state()

    player = state.players[0]

    player.dev_cards.append(
        DevCardType.ROAD_BUILDING.value
    )

    first_edges = legal_road_edges(
        state.board,
        state.players,
        player,
    )

    assert first_edges

    first = first_edges[0]

    # Temporarily extend the live player's network only
    # to discover a legal second edge, then undo it.
    player.roads.append(first)

    try:
        second_edges = legal_road_edges(
            state.board,
            state.players,
            player,
        )
    finally:
        player.roads.pop()

    second = (
        second_edges[0]
        if second_edges
        else None
    )

    original_roads = list(
        state.players[0].roads
    )

    result = apply_search_road_building(
        state,
        0,
        first,
        second,
    )

    assert first in result.players[0].roads

    if second is not None:
        assert second in result.players[0].roads

    # Original search state must remain untouched.
    assert (
        state.players[0].roads
        == original_roads
    )

    assert (
        DevCardType.ROAD_BUILDING.value
        not in result.players[0].dev_cards
    )

    assert (
        DevCardType.ROAD_BUILDING.value
        in state.players[0].dev_cards
    )

    assert (
        DevCardType.ROAD_BUILDING.value
        in result.players[0].played_dev_cards
    )


def test_monopoly_belief_ignores_hidden_resource_identity():
    from catanlab.board import Board, Tile, Vertex
    from catanlab.graph import HexCoord
    from catanlab.resources import Resource
    from catanlab.search import (
        build_monopoly_gain_belief,
    )
    from catanlab.simulation import PlayerState
    from catanlab.economy import PlayerInventory

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

    inventories_a = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    inventories_b = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    # Same public hand size: 5 cards.
    #
    # Hidden world A: all wood.
    inventories_a[1].add(
        Resource.WOOD,
        5,
    )

    # Hidden world B: all ore.
    inventories_b[1].add(
        Resource.ORE,
        5,
    )

    belief_a = build_monopoly_gain_belief(
        board,
        players,
        inventories_a,
        0,
    )

    belief_b = build_monopoly_gain_belief(
        board,
        players,
        inventories_b,
        0,
    )

    assert belief_a == belief_b

    # Public production is entirely ore, so the belief
    # should assign certainty to collecting all 5 ore.
    assert belief_a[Resource.ORE] == {
        5: 1.0
    }

    assert belief_a[Resource.WOOD] == {
        0: 1.0
    }


def test_apply_search_monopoly_outcome_isolated():
    from catanlab.devcards import DevCardType
    from catanlab.resources import Resource
    from catanlab.search import (
        apply_search_monopoly_outcome,
    )

    state = make_search_state()

    player = state.players[0]

    player.dev_cards.append(
        DevCardType.MONOPOLY.value
    )

    original_ore = (
        state.inventories[0].count(
            Resource.ORE
        )
    )

    result = apply_search_monopoly_outcome(
        state,
        0,
        Resource.ORE,
        3,
    )

    assert (
        result.inventories[0].count(
            Resource.ORE
        )
        == original_ore + 3
    )

    assert (
        state.inventories[0].count(
            Resource.ORE
        )
        == original_ore
    )

    assert (
        DevCardType.MONOPOLY.value
        not in result.players[0].dev_cards
    )

    assert (
        DevCardType.MONOPOLY.value
        in state.players[0].dev_cards
    )

    assert (
        DevCardType.MONOPOLY.value
        in result.players[0].played_dev_cards
    )
