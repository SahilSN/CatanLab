from catanlab.devcards import DevCardType
from catanlab.economy import PlayerInventory
from catanlab.observation import (
    PlayerObservation,
    PublicPlayerState,
    player_observation,
)
from catanlab.resources import Resource
from catanlab.simulation import PlayerState


def test_public_player_state_exposes_only_card_counts():
    players = [
        PlayerState(
            player_id=0,
        ),
        PlayerState(
            player_id=1,
            settlements=[4],
            cities=[8],
            dev_cards=[
                DevCardType.MONOPOLY.value,
                DevCardType.VICTORY_POINT.value,
            ],
        ),
    ]

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    inventories[1].add(
        Resource.ORE,
        3,
    )

    inventories[1].add(
        Resource.WHEAT,
        2,
    )

    observation = player_observation(
        players,
        inventories,
        player_id=0,
    )

    opponent = observation.opponent(
        1
    )

    assert opponent.resource_card_count == 5
    assert opponent.dev_card_count == 2

    # Hidden VP card is not reflected in public VP.
    assert (
        opponent.public_victory_points
        == 3
    )

    # There is intentionally no resource inventory
    # or development-card identity information.
    assert not hasattr(
        opponent,
        "inventory",
    )

    assert not hasattr(
        opponent,
        "dev_cards",
    )

    assert not hasattr(
        opponent,
        "new_dev_cards",
    )


def test_observer_keeps_own_private_information():
    players = [
        PlayerState(
            player_id=0,
            dev_cards=[
                DevCardType.MONOPOLY.value,
            ],
        ),
        PlayerState(
            player_id=1,
        ),
    ]

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    inventories[0].add(
        Resource.ORE,
        4,
    )

    observation = player_observation(
        players,
        inventories,
        player_id=0,
    )

    assert (
        observation.self_state
        is players[0]
    )

    assert (
        observation.self_inventory
        is inventories[0]
    )

    assert (
        observation.self_inventory.count(
            Resource.ORE
        )
        == 4
    )

    assert (
        observation.self_state.dev_cards
        == [
            DevCardType.MONOPOLY.value
        ]
    )


def test_observation_contains_all_other_players():
    players = [
        PlayerState(
            player_id=i
        )
        for i in range(4)
    ]

    inventories = [
        PlayerInventory()
        for _ in range(4)
    ]

    observation = player_observation(
        players,
        inventories,
        player_id=2,
    )

    assert {
        opponent.player_id
        for opponent
        in observation.opponents
    } == {
        0,
        1,
        3,
    }


def test_public_view_is_immutable():
    public = PublicPlayerState(
        player_id=1,
        settlements=(),
        cities=(),
        roads=(),
        resource_card_count=4,
        dev_card_count=2,
        played_dev_cards=(),
        knights_played=0,
        has_largest_army=False,
        has_longest_road=False,
        public_victory_points=0,
    )

    try:
        public.resource_card_count = 10
    except Exception:
        pass
    else:
        raise AssertionError(
            "PublicPlayerState should be immutable."
        )


def test_observation_rejects_mismatched_state_lengths():
    players = [
        PlayerState(
            player_id=0
        )
    ]

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    try:
        player_observation(
            players,
            inventories,
            player_id=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected mismatched player/inventory "
            "lists to be rejected."
        )


def test_played_dev_card_history_is_public():
    players = [
        PlayerState(
            player_id=0,
        ),
        PlayerState(
            player_id=1,
            dev_cards=[
                DevCardType.MONOPOLY.value,
                DevCardType.VICTORY_POINT.value,
            ],
            played_dev_cards=[
                DevCardType.KNIGHT.value,
                DevCardType.YEAR_OF_PLENTY.value,
            ],
        ),
    ]

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    observation = player_observation(
        players,
        inventories,
        player_id=0,
    )

    opponent = observation.opponent(1)

    assert opponent.dev_card_count == 2

    assert opponent.played_dev_cards == (
        DevCardType.KNIGHT.value,
        DevCardType.YEAR_OF_PLENTY.value,
    )

    # Unplayed identities remain hidden.
    assert not hasattr(
        opponent,
        "dev_cards",
    )


def _make_game_observation_fixture():
    from catanlab.board import (
        build_random_board,
    )
    from catanlab.devcards import (
        build_dev_card_deck,
    )
    from catanlab.economy import (
        ResourceBank,
    )

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

    dev_deck = build_dev_card_deck(
        seed=456
    )

    return (
        board,
        players,
        inventories,
        bank,
        dev_deck,
    )


def test_game_observation_hides_opponent_resource_identity():
    from catanlab.observation import (
        game_observation,
    )

    (
        board,
        players,
        inventories,
        bank,
        dev_deck,
    ) = _make_game_observation_fixture()

    inventories[1].add(
        Resource.WOOD,
        5,
    )

    observation_a = game_observation(
        board,
        players,
        inventories,
        0,
        bank,
        dev_deck,
    )

    inventories[1].remove(
        Resource.WOOD,
        5,
    )

    inventories[1].add(
        Resource.ORE,
        5,
    )

    observation_b = game_observation(
        board,
        players,
        inventories,
        0,
        bank,
        dev_deck,
    )

    assert observation_a == observation_b


def test_game_observation_hides_opponent_dev_card_identity():
    from catanlab.observation import (
        game_observation,
    )

    (
        board,
        players,
        inventories,
        bank,
        dev_deck,
    ) = _make_game_observation_fixture()

    players[1].dev_cards = [
        DevCardType.MONOPOLY.value,
        DevCardType.KNIGHT.value,
    ]

    observation_a = game_observation(
        board,
        players,
        inventories,
        0,
        bank,
        dev_deck,
    )

    players[1].dev_cards = [
        DevCardType.VICTORY_POINT.value,
        DevCardType.ROAD_BUILDING.value,
    ]

    observation_b = game_observation(
        board,
        players,
        inventories,
        0,
        bank,
        dev_deck,
    )

    assert observation_a == observation_b


def test_game_observation_preserves_own_private_information():
    from catanlab.observation import (
        game_observation,
    )

    (
        board,
        players,
        inventories,
        bank,
        dev_deck,
    ) = _make_game_observation_fixture()

    inventories[0].add(
        Resource.ORE,
        3,
    )

    players[0].dev_cards.append(
        DevCardType.MONOPOLY.value
    )

    observation = game_observation(
        board,
        players,
        inventories,
        0,
        bank,
        dev_deck,
    )

    assert (
        observation.self_state.resource_count(
            Resource.ORE
        )
        == 3
    )

    assert (
        observation.self_state.dev_cards
        == (
            DevCardType.MONOPOLY.value,
        )
    )


def test_game_observation_is_snapshot_not_live_reference():
    from catanlab.observation import (
        game_observation,
    )

    (
        board,
        players,
        inventories,
        bank,
        dev_deck,
    ) = _make_game_observation_fixture()

    inventories[0].add(
        Resource.WHEAT,
        2,
    )

    observation = game_observation(
        board,
        players,
        inventories,
        0,
        bank,
        dev_deck,
    )

    inventories[0].add(
        Resource.WHEAT,
        5,
    )

    players[0].settlements.append(
        7
    )

    assert (
        observation.self_state.resource_count(
            Resource.WHEAT
        )
        == 2
    )

    assert (
        observation.self_state.settlements
        == ()
    )


def test_game_observation_is_deterministic():
    from catanlab.observation import (
        game_observation,
    )

    (
        board,
        players,
        inventories,
        bank,
        dev_deck,
    ) = _make_game_observation_fixture()

    first = game_observation(
        board,
        players,
        inventories,
        2,
        bank,
        dev_deck,
    )

    second = game_observation(
        board,
        players,
        inventories,
        2,
        bank,
        dev_deck,
    )

    assert first == second


def test_game_observation_exposes_public_bank_and_deck_size():
    from catanlab.economy import (
        STANDARD_RESOURCE_SUPPLY,
    )
    from catanlab.observation import (
        game_observation,
    )

    (
        board,
        players,
        inventories,
        bank,
        dev_deck,
    ) = _make_game_observation_fixture()

    observation = game_observation(
        board,
        players,
        inventories,
        0,
        bank,
        dev_deck,
    )

    assert (
        observation.bank.count(
            Resource.ORE
        )
        == STANDARD_RESOURCE_SUPPLY
    )

    assert (
        observation.dev_deck_count
        == len(dev_deck.cards)
    )

    # The observation deliberately exposes no deck
    # card identities.
    assert not hasattr(
        observation,
        "dev_deck",
    )
