from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from catanlab.board import Board
from catanlab.devcards import DevCardDeck
from catanlab.economy import (
    PlayerInventory,
    ResourceBank,
)
from catanlab.simulation import PlayerState


@dataclass(frozen=True)
class DevCardBelief:
    """
    Information-safe belief over the identity of the
    next development card.

    Counts represent cards whose identities are still
    unknown to the acting player.
    """

    counts: dict["DevCardType", int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def probability(
        self,
        card: "DevCardType",
    ) -> float:
        if self.total == 0:
            return 0.0

        return (
            self.counts.get(card, 0)
            / self.total
        )


def build_dev_card_belief(
    players: list[PlayerState],
    player_id: int,
) -> DevCardBelief:
    """
    Build the acting player's belief over unknown
    development-card identities.

    Start from the standard deck composition, then
    subtract only identities legitimately known to
    the acting player:

    - their own unplayed development cards;
    - all publicly played action development cards.

    Opponents' hidden unplayed cards remain inside the
    unidentified pool.
    """
    from catanlab.devcards import (
        DevCardType,
        STANDARD_DEV_CARD_COUNTS,
    )

    counts = dict(
        STANDARD_DEV_CARD_COUNTS
    )

    acting_player = players[player_id]

    for card_value in acting_player.dev_cards:
        if card_value == "unknown_dev_card":
            continue

        card = DevCardType(card_value)

        counts[card] -= 1

    for player in players:
        for card_value in player.played_dev_cards:
            card = DevCardType(card_value)
            counts[card] -= 1

    if any(
        count < 0
        for count in counts.values()
    ):
        raise ValueError(
            "Observed development-card history "
            "is inconsistent with the standard deck."
        )

    return DevCardBelief(
        counts=counts
    )


@dataclass
class SearchState:
    """
    Mutable game state used for hypothetical search.

    Search code must operate on a cloned SearchState
    rather than mutating the live simulator state.
    """

    board: Board
    players: list[PlayerState]
    inventories: list[PlayerInventory]
    dev_deck: DevCardDeck
    bank: ResourceBank

    def clone(self) -> "SearchState":
        """
        Return a fully independent copy of this state.

        deepcopy is intentionally used for the first
        search implementation. The simulator state is
        composed of ordinary dataclasses, lists,
        Counters, enums, and primitive values.

        A specialized faster clone can be introduced
        later if profiling shows cloning is a search
        bottleneck.
        """
        return deepcopy(self)


def clone_search_state(
    state: SearchState,
) -> SearchState:
    """
    Convenience wrapper around SearchState.clone().
    """
    return state.clone()


def enumerate_search_actions(
    state: SearchState,
    player_id: int,
) -> list["TurnAction"]:
    """
    Enumerate deterministic ordinary actions that can
    be evaluated by one-step lookahead.

    This first search layer intentionally excludes
    domestic trades and development-card plays.
    """
    from catanlab.economy import BuildType
    from catanlab.turns import (
        ActionType,
        TurnAction,
        legal_city_vertices,
        legal_road_edges,
        legal_settlement_vertices,
    )

    player = state.players[player_id]
    inventory = state.inventories[player_id]

    actions: list[TurnAction] = []

    if inventory.can_afford(
        BuildType.CITY
    ):
        for vertex_id in legal_city_vertices(
            player
        ):
            actions.append(
                TurnAction(
                    action_type=(
                        ActionType.BUILD_CITY
                    ),
                    vertex_id=vertex_id,
                )
            )

    if inventory.can_afford(
        BuildType.SETTLEMENT
    ):
        for vertex_id in legal_settlement_vertices(
            state.board,
            state.players,
            player,
        ):
            actions.append(
                TurnAction(
                    action_type=(
                        ActionType.BUILD_SETTLEMENT
                    ),
                    vertex_id=vertex_id,
                )
            )

    if inventory.can_afford(
        BuildType.ROAD
    ):
        for edge in legal_road_edges(
            state.board,
            state.players,
            player,
        ):
            actions.append(
                TurnAction(
                    action_type=(
                        ActionType.BUILD_ROAD
                    ),
                    edge=edge,
                )
            )

    if (
        inventory.can_afford(
            BuildType.DEV_CARD
        )
        and state.dev_deck.cards
    ):
        actions.append(
            TurnAction(
                action_type=(
                    ActionType.BUY_DEV_CARD
                )
            )
        )

    actions.append(
        TurnAction(
            action_type=ActionType.PASS
        )
    )

    return actions


def apply_search_action(
    state: SearchState,
    player_id: int,
    action: "TurnAction",
) -> SearchState:
    """
    Apply one hypothetical ordinary action to an
    independent clone and return the resulting state.

    Development-card purchases intentionally hide the
    drawn card's identity from search. Search may know
    that a card was purchased, but not which hidden
    card is next in the real deck.
    """
    from catanlab.economy import BuildType
    from catanlab.turns import (
        ActionType,
        execute_action,
    )

    cloned = state.clone()

    player = cloned.players[player_id]
    inventory = cloned.inventories[player_id]

    if (
        action.action_type
        == ActionType.BUY_DEV_CARD
    ):
        if not cloned.dev_deck.cards:
            raise ValueError(
                "Development card deck is empty."
            )

        inventory.spend(
            BuildType.DEV_CARD,
            bank=cloned.bank,
        )

        # Consume one unknown card without exposing
        # its identity to the search evaluator.
        cloned.dev_deck.cards.pop()

        player.dev_cards.append(
            "unknown_dev_card"
        )

        player.new_dev_cards.append(
            "unknown_dev_card"
        )

        return cloned

    execute_action(
        cloned.board,
        cloned.players,
        player,
        inventory,
        action,
        dev_deck=cloned.dev_deck,
        bank=cloned.bank,
    )

    return cloned


def apply_search_dev_card_outcome(
    state: SearchState,
    player_id: int,
    card: "DevCardType",
) -> SearchState:
    """
    Apply one hypothetical development-card purchase
    outcome without observing the real hidden deck.

    The supplied card identity represents one branch
    of the search belief distribution.

    One physical deck slot is consumed, but its actual
    hidden identity is never inspected.
    """
    from catanlab.economy import BuildType

    cloned = state.clone()

    if not cloned.dev_deck.cards:
        raise ValueError(
            "Development card deck is empty."
        )

    player = cloned.players[player_id]
    inventory = cloned.inventories[player_id]

    inventory.spend(
        BuildType.DEV_CARD,
        bank=cloned.bank,
    )

    # Intentionally discard the return value. Search
    # must never observe the actual hidden top card.
    cloned.dev_deck.cards.pop()

    player.dev_cards.append(
        card.value
    )

    player.new_dev_cards.append(
        card.value
    )

    return cloned


def evaluate_search_state(
    state: SearchState,
    player_id: int,
) -> float:
    """
    Evaluate one player's position for shallow search.

    The evaluator rewards direct victory progress,
    productive infrastructure, build readiness, road
    development, and future settlement access.

    It intentionally does not reward raw hand size:
    spending resources on useful infrastructure should
    not automatically make a position look worse.
    """
    from catanlab.economy import BUILD_COSTS, BuildType
    from catanlab.longest_road import longest_road_length
    from catanlab.scoring import score_vertex
    from catanlab.turns import legal_settlement_vertices

    player = state.players[player_id]
    inventory = state.inventories[player_id]

    # Direct victory progress dominates the score.
    value = 12.0 * player.victory_points

    # Reward actual expected production. Cities receive
    # twice the production contribution of settlements.
    production = 0.0

    for vertex_id in player.settlements:
        production += score_vertex(
            state.board,
            state.board.vertices[vertex_id],
        ).production_score

    for vertex_id in player.cities:
        production += 2.0 * score_vertex(
            state.board,
            state.board.vertices[vertex_id],
        ).production_score

    value += 0.45 * production

    # Reward progress toward future builds instead of
    # simply rewarding cards for remaining unspent.
    for build_type, weight in (
        (BuildType.CITY, 2.0),
        (BuildType.SETTLEMENT, 1.8),
        (BuildType.DEV_CARD, 1.2),
        (BuildType.ROAD, 0.8),
    ):
        cost = BUILD_COSTS[build_type]
        total = sum(cost.values())

        satisfied = sum(
            min(
                inventory.count(resource),
                required,
            )
            for resource, required in cost.items()
        )

        if total:
            value += weight * satisfied / total

    # Roads matter both for Longest Road and because they
    # unlock future settlement locations.
    road_length = longest_road_length(
        player,
        state.players,
    )

    value += 0.65 * road_length

    future_sites = legal_settlement_vertices(
        state.board,
        state.players,
        player,
    )

    if future_sites:
        site_scores = [
            score_vertex(
                state.board,
                state.board.vertices[vertex_id],
            ).composite_score
            for vertex_id in future_sites
        ]

        # Best reachable site matters most, while having
        # several options provides a smaller flexibility
        # bonus.
        value += 0.35 * max(site_scores)
        value += 0.15 * len(future_sites)

    # Known non-VP development cards retain modest
    # option value. Search-only unknown cards are
    # valued by their expected hidden-VP contribution.
    #
    # Standard deck: 5 VP cards out of 25 total.
    # With victory points weighted at 12 above:
    #     (5 / 25) * 12 = 2.4
    #
    # This deliberately assigns no extra value yet to
    # Knight / Road Building / YOP / Monopoly effects.
    unknown_dev_cards = player.dev_cards.count(
        "unknown_dev_card"
    )

    known_non_vp_dev_cards = sum(
        card not in {
            "victory_point",
            "unknown_dev_card",
        }
        for card in player.dev_cards
    )

    value += 2.4 * unknown_dev_cards
    value += 0.75 * known_non_vp_dev_cards

    if player.has_longest_road:
        value += 2.0

    if player.has_largest_army:
        value += 2.0

    return value
