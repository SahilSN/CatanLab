from catanlab.action_space import (
    RLActionType,
    build_action_vocabulary,
    legal_action_mask,
    to_turn_action,
)
from catanlab.board import build_random_board
from catanlab.devcards import build_dev_card_deck
from catanlab.economy import (
    PlayerInventory,
    ResourceBank,
)
from catanlab.search import SearchState
from catanlab.simulation import PlayerState
from catanlab.turns import ActionType


def make_state():
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

    deck = build_dev_card_deck(
        seed=456
    )

    bank = ResourceBank()

    return SearchState(
        board=board,
        players=players,
        inventories=inventories,
        dev_deck=deck,
        bank=bank,
    )


def test_standard_action_vocabulary_is_deterministic():
    state = make_state()

    first = build_action_vocabulary(
        len(state.board.vertices),
        state.board.edges,
    )

    second = build_action_vocabulary(
        len(state.board.vertices),
        state.board.edges,
    )

    assert first == second


def test_standard_action_vocabulary_size():
    state = make_state()

    vocabulary = build_action_vocabulary(
        len(state.board.vertices),
        state.board.edges,
    )

    assert len(state.board.vertices) == 54
    assert len(state.board.edges) == 72

    assert len(vocabulary) == 202


def test_pass_is_always_present_and_convertible():
    state = make_state()

    vocabulary = build_action_vocabulary(
        len(state.board.vertices),
        state.board.edges,
    )

    assert (
        vocabulary[0].action_type
        == RLActionType.PASS
    )

    turn_action = to_turn_action(
        vocabulary[0]
    )

    assert (
        turn_action.action_type
        == ActionType.PASS
    )


def test_legal_mask_matches_vocabulary_length():
    state = make_state()

    vocabulary = build_action_vocabulary(
        len(state.board.vertices),
        state.board.edges,
    )

    mask = legal_action_mask(
        state,
        0,
        vocabulary,
    )

    assert len(mask) == len(vocabulary)


def test_pass_is_legal():
    state = make_state()

    vocabulary = build_action_vocabulary(
        len(state.board.vertices),
        state.board.edges,
    )

    mask = legal_action_mask(
        state,
        0,
        vocabulary,
    )

    assert mask[0] is True


def test_empty_inventory_cannot_buy_dev_card():
    state = make_state()

    vocabulary = build_action_vocabulary(
        len(state.board.vertices),
        state.board.edges,
    )

    mask = legal_action_mask(
        state,
        0,
        vocabulary,
    )

    buy_index = next(
        index
        for index, action
        in enumerate(vocabulary)
        if (
            action.action_type
            == RLActionType.BUY_DEV_CARD
        )
    )

    assert mask[buy_index] is False


def test_turn_action_mapping_round_trip():
    from catanlab.action_space import (
        turn_action_id,
    )

    state = make_state()

    vocabulary = build_action_vocabulary(
        len(state.board.vertices),
        state.board.edges,
    )

    for action_id, rl_action in enumerate(
        vocabulary
    ):
        turn_action = to_turn_action(
            rl_action
        )

        assert (
            turn_action_id(
                turn_action,
                vocabulary,
            )
            == action_id
        )


def test_turn_action_ids_are_unique():
    state = make_state()

    vocabulary = build_action_vocabulary(
        len(state.board.vertices),
        state.board.edges,
    )

    converted = [
        to_turn_action(action)
        for action in vocabulary
    ]

    assert len(converted) == len(
        set(
            repr(action)
            for action in converted
        )
    )
