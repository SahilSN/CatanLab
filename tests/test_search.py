from catanlab.board import build_random_board
from catanlab.devcards import build_dev_card_deck
from catanlab.economy import (
    PlayerInventory,
    ResourceBank,
)
from catanlab.resources import Resource
from catanlab.search import (
    SearchState,
    clone_search_state,
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
