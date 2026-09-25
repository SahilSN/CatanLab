from __future__ import annotations

from dataclasses import dataclass

from catanlab.resources import Resource


MONOPOLY_RESOURCE_VOCABULARY = (
    Resource.WOOD,
    Resource.BRICK,
    Resource.SHEEP,
    Resource.WHEAT,
    Resource.ORE,
)


@dataclass(frozen=True)
class CategoricalDecisionInput:
    """
    Fixed-order categorical decision presented to a
    phase-specific learned-policy head.

    vocabulary[i] is the simulator-facing value represented
    by categorical action ID i.

    legal_mask[i] states whether that choice is currently
    legal.
    """

    vocabulary: tuple[object, ...]
    legal_mask: tuple[bool, ...]

    def __post_init__(self):
        if len(self.vocabulary) != len(
            self.legal_mask
        ):
            raise ValueError(
                "Categorical vocabulary and legal mask "
                "must have matching lengths."
            )

    @property
    def action_dim(self) -> int:
        return len(self.vocabulary)

    @property
    def legal_action_ids(
        self,
    ) -> tuple[int, ...]:
        return tuple(
            action_id
            for action_id, legal
            in enumerate(self.legal_mask)
            if legal
        )

    def encode(self, value) -> int:
        """
        Return the categorical ID for one vocabulary value.
        """
        try:
            return self.vocabulary.index(
                value
            )
        except ValueError as exc:
            raise ValueError(
                "Value is not represented in categorical "
                f"vocabulary: {value!r}"
            ) from exc

    def decode(self, action_id: int):
        """
        Return the simulator-facing value represented by ID.
        """
        if (
            action_id < 0
            or action_id >= len(self.vocabulary)
        ):
            raise ValueError(
                f"Invalid categorical action ID: "
                f"{action_id}"
            )

        return self.vocabulary[action_id]

    def is_legal_value(self, value) -> bool:
        action_id = self.encode(value)
        return self.legal_mask[action_id]


YEAR_OF_PLENTY_VOCABULARY = tuple(
    (
        resource_a,
        resource_b,
    )
    for index, resource_a
    in enumerate(
        MONOPOLY_RESOURCE_VOCABULARY
    )
    for resource_b
    in MONOPOLY_RESOURCE_VOCABULARY[
        index:
    ]
)


def year_of_plenty_decision_input(
    bank,
) -> CategoricalDecisionInput:
    """
    Build the fixed 15-way Year-of-Plenty decision.

    Resource pairs are canonical unordered pairs with
    repetition:

        WOOD/WOOD, WOOD/BRICK, ... , ORE/ORE

    Legality is determined only by public bank supply.
    """

    def pair_is_legal(pair) -> bool:
        resource_a, resource_b = pair

        if resource_a == resource_b:
            return bank.can_supply(
                resource_a,
                2,
            )

        return (
            bank.can_supply(
                resource_a,
                1,
            )
            and bank.can_supply(
                resource_b,
                1,
            )
        )

    return CategoricalDecisionInput(
        vocabulary=YEAR_OF_PLENTY_VOCABULARY,
        legal_mask=tuple(
            pair_is_legal(pair)
            for pair in YEAR_OF_PLENTY_VOCABULARY
        ),
    )


def robber_tile_decision_input(
    board,
) -> CategoricalDecisionInput:
    """
    Build the categorical robber-destination decision.

    Every actual board tile receives one category. The
    robber's current tile is the only masked destination.
    """
    vocabulary = tuple(
        tile.id
        for tile in board.tiles
    )

    legal_mask = tuple(
        tile_id != board.robber_tile_id
        for tile_id in vocabulary
    )

    return CategoricalDecisionInput(
        vocabulary=vocabulary,
        legal_mask=legal_mask,
    )


def robber_victim_decision_input(
    board,
    players,
    inventories,
    player,
) -> CategoricalDecisionInput:
    """
    Build the categorical robber-victim decision.

    Player IDs form the stable vocabulary. Only adjacent
    opponents with at least one public resource card are
    legal victims.
    """
    from catanlab.devcards import (
        players_adjacent_to_tile,
    )

    vocabulary = tuple(
        candidate.player_id
        for candidate in players
    )

    if board.robber_tile_id is None:
        eligible = set()
    else:
        eligible = set(
            players_adjacent_to_tile(
                board,
                players,
                board.robber_tile_id,
                exclude_player_id=(
                    player.player_id
                ),
            )
        )

        eligible = {
            victim_id
            for victim_id in eligible
            if inventories[
                victim_id
            ].total() > 0
        }

    legal_mask = tuple(
        player_id in eligible
        for player_id in vocabulary
    )

    return CategoricalDecisionInput(
        vocabulary=vocabulary,
        legal_mask=legal_mask,
    )


def monopoly_resource_decision_input(
) -> CategoricalDecisionInput:
    """
    Build the fixed five-way Monopoly resource decision.

    All producing resources are legal Monopoly targets.
    """
    return CategoricalDecisionInput(
        vocabulary=(
            MONOPOLY_RESOURCE_VOCABULARY
        ),
        legal_mask=tuple(
            True
            for _ in (
                MONOPOLY_RESOURCE_VOCABULARY
            )
        ),
    )


CATEGORICAL_TEACHER_DECISION_KINDS = (
    "robber_tile",
    "robber_victim",
    "monopoly_resource",
    "year_of_plenty",
    "road_building",
    "discard",
    "trade_proposal",
    "trade_response",
    "trade_counter",
)


def road_building_decision_input(
    board,
    players,
    player,
) -> CategoricalDecisionInput:
    """
    Enumerate all currently legal Road Building sequences.

    A sequence contains either one or two edges. Two-edge
    legality is sequential: the second edge is checked only
    after virtually placing the first.

    The enumeration deliberately mirrors Search-v2's Road
    Building candidate generation, including deduplication
    of the same final two-road set reached in both orders.
    """
    from copy import deepcopy

    from catanlab.turns import (
        legal_road_edges,
    )

    vocabulary = []
    seen_sequences = set()

    first_edges = legal_road_edges(
        board,
        players,
        player,
    )

    for first in first_edges:
        single = (first,)

        if single not in seen_sequences:
            seen_sequences.add(single)
            vocabulary.append(single)

        # Road placement does not mutate Board. Clone
        # players so probing the second edge cannot mutate
        # the real simulator state.
        probe_players = deepcopy(
            players
        )

        probe_player = probe_players[
            player.player_id
        ]

        probe_player.roads.append(
            first
        )

        second_edges = legal_road_edges(
            board,
            probe_players,
            probe_player,
        )

        for second in second_edges:
            # Search-v2 treats the same final two-road set
            # as one candidate even if both construction
            # orders are legal.
            final_key = tuple(
                sorted(
                    (
                        first,
                        second,
                    )
                )
            )

            if final_key in seen_sequences:
                continue

            seen_sequences.add(
                final_key
            )

            # Retain the first known-legal sequential order.
            vocabulary.append(
                (
                    first,
                    second,
                )
            )

    vocabulary = tuple(vocabulary)

    return CategoricalDecisionInput(
        vocabulary=vocabulary,
        legal_mask=tuple(
            True
            for _ in vocabulary
        ),
    )


DISCARD_RESOURCE_VOCABULARY = (
    Resource.WOOD,
    Resource.BRICK,
    Resource.SHEEP,
    Resource.WHEAT,
    Resource.ORE,
)


def discard_counts(
    discarded,
) -> tuple[int, int, int, int, int]:
    """
    Convert a discard resource sequence into the canonical
    five-resource count tuple:

        (WOOD, BRICK, SHEEP, WHEAT, ORE)
    """
    return tuple(
        discarded.count(resource)
        for resource in DISCARD_RESOURCE_VOCABULARY
    )


def discard_decision_input(
    inventory,
    count: int,
) -> CategoricalDecisionInput:
    """
    Enumerate every legal discard multiset for one hand.

    Each categorical value is:

        (wood, brick, sheep, wheat, ore)

    Components may not exceed the acting player's holdings,
    and every tuple sums exactly to `count`.
    """
    if count < 0:
        raise ValueError(
            "Discard count cannot be negative."
        )

    if count > inventory.total():
        raise ValueError(
            "Cannot discard more cards than are held."
        )

    held = tuple(
        inventory.count(resource)
        for resource in DISCARD_RESOURCE_VOCABULARY
    )

    vocabulary = []

    def enumerate_counts(
        index: int,
        remaining: int,
        prefix: tuple[int, ...],
    ) -> None:
        if index == len(held):
            if remaining == 0:
                vocabulary.append(prefix)
            return

        maximum = min(
            held[index],
            remaining,
        )

        for amount in range(
            maximum + 1
        ):
            enumerate_counts(
                index + 1,
                remaining - amount,
                (
                    *prefix,
                    amount,
                ),
            )

    enumerate_counts(
        0,
        count,
        (),
    )

    result = tuple(vocabulary)

    return CategoricalDecisionInput(
        vocabulary=result,
        legal_mask=tuple(
            True
            for _ in result
        ),
    )


TRADE_RESPONSE_VOCABULARY = (
    False,  # reject
    True,   # accept
)


def trade_proposal_decision_input(
    players,
    player,
    inventory,
    excluded_recipients=None,
) -> CategoricalDecisionInput:
    """
    Enumerate Search-v2's domestic-trade proposal space.

    The first class is None, meaning do not propose a trade.
    Remaining classes are every structurally possible
    Search-v2 1-for-1 offer that:

      * gives a resource the acting player actually holds,
      * requests a different producing resource,
      * targets a non-excluded opponent.

    Recipient hidden resource identities are deliberately
    irrelevant.
    """
    from catanlab.trading import (
        TradeOffer,
        validate_trade_terms,
    )

    if excluded_recipients is None:
        excluded_recipients = set()
    else:
        excluded_recipients = set(
            excluded_recipients
        )

    recipients = tuple(
        other.player_id
        for other in players
        if (
            other.player_id
            != player.player_id
            and other.player_id
            not in excluded_recipients
        )
    )

    vocabulary = [None]

    for give_resource in (
        MONOPOLY_RESOURCE_VOCABULARY
    ):
        if inventory.count(
            give_resource
        ) <= 0:
            continue

        for receive_resource in (
            MONOPOLY_RESOURCE_VOCABULARY
        ):
            if (
                receive_resource
                == give_resource
            ):
                continue

            for recipient_id in recipients:
                offer = TradeOffer(
                    proposer_id=(
                        player.player_id
                    ),
                    recipient_id=recipient_id,
                    give=(
                        (
                            give_resource,
                            1,
                        ),
                    ),
                    receive=(
                        (
                            receive_resource,
                            1,
                        ),
                    ),
                )

                if validate_trade_terms(
                    offer
                ):
                    vocabulary.append(
                        offer
                    )

    vocabulary = tuple(vocabulary)

    return CategoricalDecisionInput(
        vocabulary=vocabulary,
        legal_mask=tuple(
            True
            for _ in vocabulary
        ),
    )


def trade_response_decision_input(
    offer,
    inventories,
) -> CategoricalDecisionInput:
    """
    Encode REJECT / ACCEPT for an incoming trade.

    Reject is always legal. Accept is legal only when the
    complete trade is currently feasible.

    This is appropriate at response time because the
    recipient is a participant in the transaction and may
    verify their own hand; the engine performs the same
    feasibility check.
    """
    from catanlab.trading import (
        validate_trade_offer,
    )

    return CategoricalDecisionInput(
        vocabulary=(
            TRADE_RESPONSE_VOCABULARY
        ),
        legal_mask=(
            True,
            validate_trade_offer(
                offer,
                inventories,
            ),
        ),
    )


def trade_counter_decision_input(
    players,
    player,
    inventory,
    offer,
    attempted_offers=None,
) -> CategoricalDecisionInput:
    """
    Enumerate Search-v2's counteroffer space.

    The first class is None, meaning make no counteroffer.
    Remaining classes are legal 1-for-1 offers from the
    current recipient back to the original proposer.

    As in Search-v2 itself, the opponent's hidden hand is
    never inspected.
    """
    from catanlab.trading import (
        TradeOffer,
        validate_trade_terms,
    )

    if attempted_offers is None:
        attempted_offers = set()
    else:
        attempted_offers = set(
            attempted_offers
        )

    vocabulary = [None]

    # A counter is meaningful only for the player who
    # received the incoming offer.
    if (
        offer.recipient_id
        != player.player_id
    ):
        return CategoricalDecisionInput(
            vocabulary=tuple(vocabulary),
            legal_mask=(True,),
        )

    for give_resource in (
        MONOPOLY_RESOURCE_VOCABULARY
    ):
        if inventory.count(
            give_resource
        ) <= 0:
            continue

        for receive_resource in (
            MONOPOLY_RESOURCE_VOCABULARY
        ):
            if (
                receive_resource
                == give_resource
            ):
                continue

            candidate = TradeOffer(
                proposer_id=(
                    player.player_id
                ),
                recipient_id=(
                    offer.proposer_id
                ),
                give=(
                    (
                        give_resource,
                        1,
                    ),
                ),
                receive=(
                    (
                        receive_resource,
                        1,
                    ),
                ),
            )

            if candidate in attempted_offers:
                continue

            if not validate_trade_terms(
                candidate
            ):
                continue

            vocabulary.append(
                candidate
            )

    vocabulary = tuple(vocabulary)

    return CategoricalDecisionInput(
        vocabulary=vocabulary,
        legal_mask=tuple(
            True
            for _ in vocabulary
        ),
    )
