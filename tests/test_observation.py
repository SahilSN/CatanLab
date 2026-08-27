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
