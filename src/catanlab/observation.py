from __future__ import annotations

from dataclasses import dataclass

from catanlab.economy import PlayerInventory
from catanlab.simulation import PlayerState


@dataclass(frozen=True)
class PublicPlayerState:
    """
    Information about a player that is publicly
    observable to every other player.

    Hidden resource identities and unplayed
    development-card identities are intentionally
    excluded.
    """

    player_id: int

    settlements: tuple[int, ...]
    cities: tuple[int, ...]
    roads: tuple[tuple[int, int], ...]

    resource_card_count: int
    dev_card_count: int

    played_dev_cards: tuple[str, ...]

    knights_played: int

    has_largest_army: bool
    has_longest_road: bool

    public_victory_points: int


@dataclass(frozen=True)
class PlayerObservation:
    """
    Information legally available to one player.

    The observing player receives their own complete
    private state and inventory, while opponents are
    represented only by public information.
    """

    self_state: PlayerState
    self_inventory: PlayerInventory

    opponents: tuple[
        PublicPlayerState,
        ...
    ]

    def opponent(
        self,
        player_id: int,
    ) -> PublicPlayerState:
        for opponent in self.opponents:
            if opponent.player_id == player_id:
                return opponent

        raise KeyError(
            f"Player {player_id} is not an opponent."
        )


def public_player_state(
    player: PlayerState,
    inventory: PlayerInventory,
) -> PublicPlayerState:
    """
    Construct the publicly observable state for one
    player.
    """

    return PublicPlayerState(
        player_id=player.player_id,
        settlements=tuple(
            player.settlements
        ),
        cities=tuple(
            player.cities
        ),
        roads=tuple(
            player.roads
        ),
        resource_card_count=(
            inventory.total()
        ),
        dev_card_count=len(
            player.dev_cards
        ),
        played_dev_cards=tuple(
            player.played_dev_cards
        ),
        knights_played=(
            player.knights_played
        ),
        has_largest_army=(
            player.has_largest_army
        ),
        has_longest_road=(
            player.has_longest_road
        ),
        public_victory_points=(
            player.public_victory_points
        ),
    )


def player_observation(
    players: list[PlayerState],
    inventories: list[PlayerInventory],
    player_id: int,
) -> PlayerObservation:
    """
    Build the legal-information view available to
    `player_id`.
    """

    if (
        player_id < 0
        or player_id >= len(players)
    ):
        raise ValueError(
            f"Invalid player_id: {player_id}"
        )

    if len(players) != len(inventories):
        raise ValueError(
            "players and inventories must have "
            "matching lengths."
        )

    self_state = players[
        player_id
    ]

    opponents = tuple(
        public_player_state(
            other,
            inventories[
                other.player_id
            ],
        )
        for other in players
        if (
            other.player_id
            != player_id
        )
    )

    return PlayerObservation(
        self_state=self_state,
        self_inventory=inventories[
            player_id
        ],
        opponents=opponents,
    )
