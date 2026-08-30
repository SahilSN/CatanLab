from catanlab.board import build_random_board
from catanlab.devcards import (
    build_dev_card_deck,
)
from catanlab.economy import (
    PlayerInventory,
    ResourceBank,
)
from catanlab.rl_interface import (
    build_rl_decision_context,
)
from catanlab.simulation import PlayerState


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


def test_rl_policy_input_dimensions():
    (
        board,
        players,
        inventories,
        bank,
        deck,
    ) = make_fixture()

    context = build_rl_decision_context(
        board,
        players,
        inventories,
        0,
        bank,
        deck,
    )

    assert (
        context.policy_input.observation_dim
        == 1138
    )

    assert (
        context.policy_input.action_dim
        == 202
    )


def test_rl_policy_input_has_legal_action():
    (
        board,
        players,
        inventories,
        bank,
        deck,
    ) = make_fixture()

    context = build_rl_decision_context(
        board,
        players,
        inventories,
        0,
        bank,
        deck,
    )

    assert (
        len(
            context.policy_input.legal_action_ids
        )
        >= 1
    )


def test_rl_decision_context_is_deterministic():
    (
        board,
        players,
        inventories,
        bank,
        deck,
    ) = make_fixture()

    first = build_rl_decision_context(
        board,
        players,
        inventories,
        2,
        bank,
        deck,
    )

    second = build_rl_decision_context(
        board,
        players,
        inventories,
        2,
        bank,
        deck,
    )

    assert first == second
