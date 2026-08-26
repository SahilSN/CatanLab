from dataclasses import dataclass, field
from enum import Enum
import random


class DevCardType(str, Enum):
    KNIGHT = "knight"
    VICTORY_POINT = "victory_point"
    ROAD_BUILDING = "road_building"
    YEAR_OF_PLENTY = "year_of_plenty"
    MONOPOLY = "monopoly"


STANDARD_DEV_CARD_COUNTS = {
    DevCardType.KNIGHT: 14,
    DevCardType.VICTORY_POINT: 5,
    DevCardType.ROAD_BUILDING: 2,
    DevCardType.YEAR_OF_PLENTY: 2,
    DevCardType.MONOPOLY: 2,
}


@dataclass
class DevCardDeck:
    cards: list[DevCardType]


def build_dev_card_deck(
    seed: int | None = None,
) -> DevCardDeck:
    rng = random.Random(seed)

    cards: list[DevCardType] = []

    for card_type, count in (
        STANDARD_DEV_CARD_COUNTS.items()
    ):
        cards.extend(
            [card_type] * count
        )

    rng.shuffle(cards)

    return DevCardDeck(
        cards=cards
    )


def draw_dev_card(
    deck: DevCardDeck,
) -> DevCardType:
    if not deck.cards:
        raise ValueError(
            "Development card deck is empty."
        )

    return deck.cards.pop()


def buy_dev_card(
    player,
    inventory,
    deck: DevCardDeck,
):
    from catanlab.economy import BuildType

    inventory.spend(
        BuildType.DEV_CARD
    )

    card = draw_dev_card(
        deck
    )

    player.dev_cards.append(
        card.value
    )

    player.new_dev_cards.append(
        card.value
    )

    return card


def playable_dev_card_count(
    player,
    card: DevCardType,
) -> int:
    """
    Return the number of copies of a dev card that
    are currently playable.

    Newly purchased cards are owned immediately but
    cannot be played until the player's next turn.
    """

    owned = player.dev_cards.count(
        card.value
    )

    newly_bought = player.new_dev_cards.count(
        card.value
    )

    return max(
        0,
        owned - newly_bought,
    )


def has_playable_dev_card(
    player,
    card: DevCardType,
) -> bool:
    return (
        playable_dev_card_count(
            player,
            card,
        )
        > 0
    )


def play_knight(
    player,
) -> None:
    """
    Play one Knight development card.
    """

    knight = DevCardType.KNIGHT.value

    if not has_playable_dev_card(
        player,
        DevCardType.KNIGHT,
    ):
        raise ValueError(
            "Player does not have a playable Knight card."
        )

    player.dev_cards.remove(
        knight
    )

    player.knights_played += 1


def update_largest_army(
    players,
) -> int | None:
    """
    Update Largest Army ownership.

    A player needs at least 3 played Knights.

    The current holder keeps Largest Army on ties;
    another player must strictly exceed the holder.
    """

    current_holder = next(
        (
            player
            for player in players
            if player.has_largest_army
        ),
        None,
    )

    if current_holder is not None:
        best_challenger = max(
            players,
            key=lambda player: (
                player.knights_played
            ),
        )

        if (
            best_challenger.player_id
            != current_holder.player_id
            and best_challenger.knights_played
            > current_holder.knights_played
        ):
            current_holder.has_largest_army = False
            best_challenger.has_largest_army = True

            return best_challenger.player_id

        return current_holder.player_id

    eligible = [
        player
        for player in players
        if player.knights_played >= 3
    ]

    if not eligible:
        return None

    highest = max(
        player.knights_played
        for player in eligible
    )

    leaders = [
        player
        for player in eligible
        if player.knights_played == highest
    ]

    if len(leaders) != 1:
        return None

    winner = leaders[0]
    winner.has_largest_army = True

    return winner.player_id


def move_robber(
    board,
    tile_id: int,
) -> None:
    """
    Move the robber to another tile.
    """

    if tile_id < 0 or tile_id >= len(
        board.tiles
    ):
        raise ValueError(
            "Invalid robber tile."
        )

    if tile_id == board.robber_tile_id:
        raise ValueError(
            "Robber must move to a different tile."
        )

    board.robber_tile_id = tile_id


def play_knight_and_move_robber(
    player,
    board,
    tile_id: int,
) -> None:
    """
    Play a Knight and move the robber.
    """

    play_knight(
        player
    )

    move_robber(
        board,
        tile_id,
    )


def discard_for_seven(
    inventory,
    rng,
) -> list:
    """
    Discard half the player's hand, rounded down,
    if the player holds more than seven cards.

    Cards are selected randomly for now.
    Strategy-aware discard decisions will be
    added later.
    """

    if inventory.total() <= 7:
        return []

    discard_count = (
        inventory.total() // 2
    )

    discarded = []

    for _ in range(discard_count):
        available = [
            resource
            for resource, count
            in inventory.resources.items()
            for _ in range(count)
        ]

        resource = rng.choice(
            available
        )

        inventory.remove(
            resource
        )

        discarded.append(
            resource
        )

    return discarded


def players_adjacent_to_tile(
    board,
    players,
    tile_id: int,
    exclude_player_id: int | None = None,
) -> list[int]:
    """
    Return player IDs with a settlement or city
    touching the selected tile.
    """

    eligible = []

    for player in players:
        if (
            exclude_player_id is not None
            and player.player_id
            == exclude_player_id
        ):
            continue

        building_vertices = (
            player.settlements
            + player.cities
        )

        touches_tile = any(
            tile_id
            in board.vertices[
                vertex_id
            ].adjacent_tiles
            for vertex_id
            in building_vertices
        )

        if touches_tile:
            eligible.append(
                player.player_id
            )

    return eligible


def steal_random_resource(
    thief_id: int,
    victim_id: int,
    inventories,
    rng,
):
    """
    Transfer one randomly selected resource from
    the victim to the thief.

    Returns the stolen resource, or None if the
    victim has no resource cards.
    """

    thief = inventories[
        thief_id
    ]

    victim = inventories[
        victim_id
    ]

    available = [
        resource
        for resource, count
        in victim.resources.items()
        for _ in range(count)
    ]

    if not available:
        return None

    resource = rng.choice(
        available
    )

    victim.remove(
        resource
    )

    thief.add(
        resource
    )

    return resource


def rob_adjacent_player(
    board,
    players,
    inventories,
    thief_id: int,
    rng,
):
    """
    Steal one random card from a randomly selected
    eligible opponent adjacent to the robber.

    Opponents with empty hands are excluded.
    """

    if board.robber_tile_id is None:
        raise ValueError(
            "Robber is not placed on the board."
        )

    adjacent = players_adjacent_to_tile(
        board,
        players,
        board.robber_tile_id,
        exclude_player_id=thief_id,
    )

    eligible = [
        player_id
        for player_id in adjacent
        if inventories[
            player_id
        ].total() > 0
    ]

    if not eligible:
        return None

    victim_id = rng.choice(
        eligible
    )

    resource = steal_random_resource(
        thief_id,
        victim_id,
        inventories,
        rng,
    )

    return (
        victim_id,
        resource,
    )


def play_year_of_plenty(
    player,
    inventory,
    resource_a,
    resource_b,
) -> None:
    """
    Play Year of Plenty and gain any two resources.
    """

    card = DevCardType.YEAR_OF_PLENTY.value

    if not has_playable_dev_card(
        player,
        DevCardType.YEAR_OF_PLENTY,
    ):
        raise ValueError(
            "Player does not have a playable "
            "Year of Plenty card."
        )

    player.dev_cards.remove(
        card
    )

    inventory.add(
        resource_a
    )

    inventory.add(
        resource_b
    )


def play_monopoly(
    player,
    inventories,
    resource,
) -> int:
    """
    Play Monopoly and take all cards of one
    resource from every opponent.

    Returns the number of cards collected.
    """

    card = DevCardType.MONOPOLY.value

    if not has_playable_dev_card(
        player,
        DevCardType.MONOPOLY,
    ):
        raise ValueError(
            "Player does not have a playable Monopoly card."
        )

    player.dev_cards.remove(
        card
    )

    collected = 0

    for player_id, inventory in enumerate(
        inventories
    ):
        if player_id == player.player_id:
            continue

        amount = inventory.count(
            resource
        )

        if amount == 0:
            continue

        inventory.remove(
            resource,
            amount,
        )

        inventories[
            player.player_id
        ].add(
            resource,
            amount,
        )

        collected += amount

    return collected


def play_road_building(
    player,
    board,
    players,
    first_edge: tuple[int, int],
    second_edge: tuple[int, int],
) -> None:
    """
    Play Road Building and place two free roads.

    The first road may extend the player's
    network and make the second road legal.

    If either placement fails, the player's road
    state is restored and the card is not spent.
    """

    from catanlab.building import (
        build_road_free,
    )

    card = DevCardType.ROAD_BUILDING.value

    if not has_playable_dev_card(
        player,
        DevCardType.ROAD_BUILDING,
    ):
        raise ValueError(
            "Player does not have a playable "
            "Road Building card."
        )

    original_roads = list(
        player.roads
    )

    try:
        build_road_free(
            board,
            players,
            player,
            first_edge[0],
            first_edge[1],
        )

        build_road_free(
            board,
            players,
            player,
            second_edge[0],
            second_edge[1],
        )

    except ValueError:
        player.roads = original_roads
        raise

    player.dev_cards.remove(
        card
    )
