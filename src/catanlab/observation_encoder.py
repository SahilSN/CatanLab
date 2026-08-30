from __future__ import annotations

from catanlab.devcards import DevCardType
from catanlab.observation import GameObservation
from catanlab.resources import Resource


PRODUCING_RESOURCES = (
    Resource.WOOD,
    Resource.BRICK,
    Resource.SHEEP,
    Resource.WHEAT,
    Resource.ORE,
)

TILE_RESOURCES = (
    Resource.WOOD,
    Resource.BRICK,
    Resource.SHEEP,
    Resource.WHEAT,
    Resource.ORE,
    Resource.DESERT,
)

DEV_CARD_TYPES = (
    DevCardType.KNIGHT,
    DevCardType.VICTORY_POINT,
    DevCardType.ROAD_BUILDING,
    DevCardType.YEAR_OF_PLENTY,
    DevCardType.MONOPOLY,
)


def _one_hot(
    value,
    choices,
) -> list[float]:
    return [
        1.0 if value == choice else 0.0
        for choice in choices
    ]


def _relative_opponents(
    observation: GameObservation,
):
    """
    Return opponents in clockwise relative-seat order.

    For observer i in an n-player game, order is:

        i+1, i+2, ..., wrapping around modulo n.

    The observation currently represents standard
    four-player Catan, but this helper derives n from the
    available player IDs rather than hard-coding 4.
    """
    player_ids = {
        observation.observer_id,
        *(
            opponent.player_id
            for opponent
            in observation.opponents
        ),
    }

    n_players = len(player_ids)

    opponent_by_id = {
        opponent.player_id: opponent
        for opponent in observation.opponents
    }

    ordered = []

    for offset in range(
        1,
        n_players,
    ):
        player_id = (
            observation.observer_id
            + offset
        ) % n_players

        ordered.append(
            opponent_by_id[player_id]
        )

    return tuple(ordered)


def encode_game_observation(
    observation: GameObservation,
) -> tuple[float, ...]:
    """
    Encode an information-safe GameObservation as a
    deterministic fixed-order numeric feature vector.

    No live simulator objects or hidden opponent state are
    consulted here. The encoder can only consume fields
    already present in GameObservation.
    """
    values: list[float] = []

    # ------------------------------------------------------------------
    # Board tiles
    # ------------------------------------------------------------------

    for tile in observation.board.tiles:
        values.extend(
            _one_hot(
                tile.resource,
                TILE_RESOURCES,
            )
        )

        # Standard Catan number tokens are 2..12 with no 7.
        # Desert/no-number is represented as 0.
        values.append(
            float(
                tile.number
                if tile.number is not None
                else 0
            )
        )

        values.append(
            float(tile.has_robber)
        )

    # ------------------------------------------------------------------
    # Board vertices
    #
    # Piece ownership is derived from player structure lists.
    #
    # Each vertex:
    #   empty
    #   self settlement
    #   self city
    #   relative opponent 1 settlement
    #   relative opponent 1 city
    #   ...
    # ------------------------------------------------------------------

    relative_opponents = (
        _relative_opponents(
            observation
        )
    )

    vertex_state = {}

    for vertex_id in (
        observation.self_state.settlements
    ):
        vertex_state[vertex_id] = (
            "self_settlement"
        )

    for vertex_id in (
        observation.self_state.cities
    ):
        vertex_state[vertex_id] = (
            "self_city"
        )

    for relative_index, opponent in enumerate(
        relative_opponents,
        start=1,
    ):
        for vertex_id in opponent.settlements:
            vertex_state[vertex_id] = (
                f"opponent_{relative_index}_settlement"
            )

        for vertex_id in opponent.cities:
            vertex_state[vertex_id] = (
                f"opponent_{relative_index}_city"
            )

    vertex_choices = ["empty"]

    vertex_choices.extend(
        [
            "self_settlement",
            "self_city",
        ]
    )

    for relative_index in range(
        1,
        len(relative_opponents) + 1,
    ):
        vertex_choices.extend(
            [
                (
                    f"opponent_{relative_index}"
                    "_settlement"
                ),
                (
                    f"opponent_{relative_index}"
                    "_city"
                ),
            ]
        )

    for vertex in observation.board.vertices:
        state = vertex_state.get(
            vertex.id,
            "empty",
        )

        values.extend(
            _one_hot(
                state,
                vertex_choices,
            )
        )

    # ------------------------------------------------------------------
    # Roads / edges
    #
    # Each edge:
    #   empty
    #   self
    #   relative opponent 1
    #   ...
    # ------------------------------------------------------------------

    road_owner = {}

    for edge in observation.self_state.roads:
        road_owner[
            tuple(sorted(edge))
        ] = 0

    for relative_index, opponent in enumerate(
        relative_opponents,
        start=1,
    ):
        for edge in opponent.roads:
            road_owner[
                tuple(sorted(edge))
            ] = relative_index

    road_choices = list(
        range(
            0,
            len(relative_opponents) + 1,
        )
    )

    # -1 means empty.
    road_choices = [
        -1,
        *road_choices,
    ]

    for edge in observation.board.edges:
        key = tuple(
            sorted(
                (
                    edge.vertex_a,
                    edge.vertex_b,
                )
            )
        )

        owner = road_owner.get(
            key,
            -1,
        )

        values.extend(
            _one_hot(
                owner,
                road_choices,
            )
        )

    # ------------------------------------------------------------------
    # Ports
    #
    # Endpoint IDs are public topology information.
    # Port type:
    #   generic 3:1 -> all resource one-hots zero
    #   resource 2:1 -> one-hot resource
    # ------------------------------------------------------------------

    for port in observation.board.ports:
        values.append(
            float(port.vertex_a)
        )

        values.append(
            float(port.vertex_b)
        )

        values.extend(
            _one_hot(
                port.resource,
                PRODUCING_RESOURCES,
            )
        )

        values.append(
            float(
                port.resource is None
            )
        )

    # ------------------------------------------------------------------
    # Self private information
    # ------------------------------------------------------------------

    for resource in PRODUCING_RESOURCES:
        values.append(
            float(
                observation.self_state.resource_count(
                    resource
                )
            )
        )

    values.append(
        float(
            observation.self_state.public_victory_points
        )
    )

    values.append(
        float(
            observation.self_state.victory_points
        )
    )

    for card_type in DEV_CARD_TYPES:
        values.append(
            float(
                observation.self_state.dev_cards.count(
                    card_type.value
                )
            )
        )

    for card_type in DEV_CARD_TYPES:
        values.append(
            float(
                observation.self_state.new_dev_cards.count(
                    card_type.value
                )
            )
        )

    values.append(
        float(
            observation.self_state.knights_played
        )
    )

    values.append(
        float(
            observation.self_state.has_largest_army
        )
    )

    values.append(
        float(
            observation.self_state.has_longest_road
        )
    )

    # ------------------------------------------------------------------
    # Opponent public information
    # ------------------------------------------------------------------

    for opponent in relative_opponents:
        values.append(
            float(
                opponent.public_victory_points
            )
        )

        values.append(
            float(
                opponent.resource_card_count
            )
        )

        values.append(
            float(
                opponent.dev_card_count
            )
        )

        values.append(
            float(
                opponent.knights_played
            )
        )

        values.append(
            float(
                opponent.has_largest_army
            )
        )

        values.append(
            float(
                opponent.has_longest_road
            )
        )

        values.append(
            float(
                len(opponent.settlements)
            )
        )

        values.append(
            float(
                len(opponent.cities)
            )
        )

        values.append(
            float(
                len(opponent.roads)
            )
        )

        for card_type in DEV_CARD_TYPES:
            values.append(
                float(
                    opponent.played_dev_cards.count(
                        card_type.value
                    )
                )
            )

    # ------------------------------------------------------------------
    # Shared public finite resources
    # ------------------------------------------------------------------

    for resource in PRODUCING_RESOURCES:
        values.append(
            float(
                observation.bank.count(
                    resource
                )
            )
        )

    values.append(
        float(
            observation.dev_deck_count
        )
    )

    return tuple(values)
