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
