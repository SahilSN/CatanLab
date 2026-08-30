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


# ---------------------------------------------------------------------------
# Immutable full-game observation for learned agents
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrivatePlayerState:
    """
    Immutable snapshot of information privately known
    to the observing player.
    """

    player_id: int

    settlements: tuple[int, ...]
    cities: tuple[int, ...]
    roads: tuple[tuple[int, int], ...]

    resources: tuple[
        tuple[object, int],
        ...
    ]

    dev_cards: tuple[str, ...]
    new_dev_cards: tuple[str, ...]
    played_dev_cards: tuple[str, ...]

    knights_played: int

    has_largest_army: bool
    has_longest_road: bool

    public_victory_points: int
    victory_points: int

    def resource_count(
        self,
        resource,
    ) -> int:
        for stored_resource, count in (
            self.resources
        ):
            if stored_resource == resource:
                return count

        return 0


@dataclass(frozen=True)
class TileObservation:
    id: int
    resource: object
    number: int | None
    has_robber: bool


@dataclass(frozen=True)
class VertexObservation:
    id: int
    adjacent_tiles: tuple[int, ...]
    neighbors: tuple[int, ...]


@dataclass(frozen=True)
class EdgeObservation:
    vertex_a: int
    vertex_b: int


@dataclass(frozen=True)
class PortObservation:
    vertex_a: int
    vertex_b: int
    resource: object | None


@dataclass(frozen=True)
class BoardObservation:
    """
    Immutable public board configuration.

    Piece ownership remains in the per-player state,
    avoiding duplicate sources of truth.
    """

    tiles: tuple[
        TileObservation,
        ...
    ]

    vertices: tuple[
        VertexObservation,
        ...
    ]

    edges: tuple[
        EdgeObservation,
        ...
    ]

    ports: tuple[
        PortObservation,
        ...
    ]


@dataclass(frozen=True)
class BankObservation:
    """
    Public finite-bank resource counts.
    """

    resources: tuple[
        tuple[object, int],
        ...
    ]

    def count(
        self,
        resource,
    ) -> int:
        for stored_resource, count in (
            self.resources
        ):
            if stored_resource == resource:
                return count

        return 0


@dataclass(frozen=True)
class GameObservation:
    """
    Immutable legal-information snapshot suitable for
    learned agents.

    Opponent hidden resource identities, opponent
    unplayed development-card identities, and hidden
    development-deck order are deliberately excluded.
    """

    observer_id: int

    board: BoardObservation

    self_state: PrivatePlayerState

    opponents: tuple[
        PublicPlayerState,
        ...
    ]

    bank: BankObservation

    dev_deck_count: int

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


def game_observation(
    board,
    players,
    inventories,
    player_id: int,
    bank,
    dev_deck,
) -> GameObservation:
    """
    Build an immutable legal-information snapshot for
    one observing player.

    This function never exposes:

    - opponent resource identities
    - opponent unplayed development-card identities
    - development-deck order
    """
    from catanlab.economy import (
        PRODUCING_RESOURCES,
    )

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

    player = players[player_id]
    inventory = inventories[player_id]

    private = PrivatePlayerState(
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
        resources=tuple(
            (
                resource,
                inventory.count(resource),
            )
            for resource
            in PRODUCING_RESOURCES
        ),
        dev_cards=tuple(
            player.dev_cards
        ),
        new_dev_cards=tuple(
            player.new_dev_cards
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
        victory_points=(
            player.victory_points
        ),
    )

    opponents = tuple(
        public_player_state(
            other,
            inventories[
                other.player_id
            ],
        )
        for other in players
        if other.player_id != player_id
    )

    board_state = BoardObservation(
        tiles=tuple(
            TileObservation(
                id=tile.id,
                resource=tile.resource,
                number=tile.number,
                has_robber=(
                    tile.id
                    == board.robber_tile_id
                ),
            )
            for tile in board.tiles
        ),
        vertices=tuple(
            VertexObservation(
                id=vertex.id,
                adjacent_tiles=tuple(
                    vertex.adjacent_tiles
                ),
                neighbors=tuple(
                    vertex.neighbors
                ),
            )
            for vertex in board.vertices
        ),
        edges=tuple(
            EdgeObservation(
                vertex_a=edge.vertex_a,
                vertex_b=edge.vertex_b,
            )
            for edge in board.edges
        ),
        ports=tuple(
            PortObservation(
                vertex_a=port.vertex_a,
                vertex_b=port.vertex_b,
                resource=port.resource,
            )
            for port in board.ports
        ),
    )

    bank_state = BankObservation(
        resources=tuple(
            (
                resource,
                bank.count(resource),
            )
            for resource
            in PRODUCING_RESOURCES
        )
    )

    return GameObservation(
        observer_id=player_id,
        board=board_state,
        self_state=private,
        opponents=opponents,
        bank=bank_state,
        # Only the public number of remaining cards is
        # exposed. Card identities/order remain hidden.
        dev_deck_count=len(
            dev_deck.cards
        ),
    )
