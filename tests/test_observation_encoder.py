from catanlab.board import build_random_board
from catanlab.devcards import (
    DevCardType,
    build_dev_card_deck,
)
from catanlab.economy import (
    PlayerInventory,
    ResourceBank,
)
from catanlab.observation import (
    game_observation,
)
from catanlab.observation_encoder import (
    encode_game_observation,
)
from catanlab.resources import Resource
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


def encode(
    board,
    players,
    inventories,
    bank,
    deck,
    observer_id=0,
):
    observation = game_observation(
        board,
        players,
        inventories,
        observer_id,
        bank,
        deck,
    )

    return encode_game_observation(
        observation
    )


def test_encoder_is_deterministic():
    state = make_fixture()

    first = encode(*state)
    second = encode(*state)

    assert first == second


def test_encoder_has_fixed_length_for_all_observers():
    (
        board,
        players,
        inventories,
        bank,
        deck,
    ) = make_fixture()

    lengths = {
        len(
            encode(
                board,
                players,
                inventories,
                bank,
                deck,
                observer_id,
            )
        )
        for observer_id in range(4)
    }

    assert len(lengths) == 1


def test_encoder_changes_for_own_private_resources():
    (
        board,
        players,
        inventories,
        bank,
        deck,
    ) = make_fixture()

    before = encode(
        board,
        players,
        inventories,
        bank,
        deck,
        0,
    )

    inventories[0].add(
        Resource.ORE,
        3,
    )

    after = encode(
        board,
        players,
        inventories,
        bank,
        deck,
        0,
    )

    assert before != after


def test_encoder_ignores_opponent_hidden_resource_identity():
    (
        board,
        players,
        inventories,
        bank,
        deck,
    ) = make_fixture()

    inventories[1].add(
        Resource.WOOD,
        5,
    )

    wood_world = encode(
        board,
        players,
        inventories,
        bank,
        deck,
        0,
    )

    inventories[1].remove(
        Resource.WOOD,
        5,
    )

    inventories[1].add(
        Resource.ORE,
        5,
    )

    ore_world = encode(
        board,
        players,
        inventories,
        bank,
        deck,
        0,
    )

    assert wood_world == ore_world


def test_encoder_ignores_opponent_hidden_dev_identity():
    (
        board,
        players,
        inventories,
        bank,
        deck,
    ) = make_fixture()

    players[1].dev_cards = [
        DevCardType.MONOPOLY.value,
        DevCardType.KNIGHT.value,
    ]

    first = encode(
        board,
        players,
        inventories,
        bank,
        deck,
        0,
    )

    players[1].dev_cards = [
        DevCardType.VICTORY_POINT.value,
        DevCardType.ROAD_BUILDING.value,
    ]

    second = encode(
        board,
        players,
        inventories,
        bank,
        deck,
        0,
    )

    assert first == second


def test_encoder_exposes_own_hidden_victory_point():
    (
        board,
        players,
        inventories,
        bank,
        deck,
    ) = make_fixture()

    before = encode(
        board,
        players,
        inventories,
        bank,
        deck,
        0,
    )

    players[0].dev_cards.append(
        DevCardType.VICTORY_POINT.value
    )

    after = encode(
        board,
        players,
        inventories,
        bank,
        deck,
        0,
    )

    assert before != after


def test_encoder_ignores_hidden_deck_order():
    (
        board,
        players,
        inventories,
        bank,
        deck,
    ) = make_fixture()

    first = encode(
        board,
        players,
        inventories,
        bank,
        deck,
        0,
    )

    deck.cards.reverse()

    second = encode(
        board,
        players,
        inventories,
        bank,
        deck,
        0,
    )

    assert first == second
