from dataclasses import dataclass
from enum import Enum

from catanlab.devcards import (
    DevCardType,
    has_playable_dev_card,
)
from catanlab.economy import BuildType
from catanlab.longest_road import longest_road_length
from catanlab.resources import Resource


class DevCardPhase(str, Enum):
    PRE_ROLL = "pre_roll"
    POST_ROLL = "post_roll"


@dataclass(frozen=True)
class DevCardDecision:
    card: DevCardType | None
    utility: float
    resource: Resource | None = None
    resources: (
        tuple[Resource, Resource] | None
    ) = None
    road_edges: (
        tuple[tuple[int, int], ...] | None
    ) = None


def knight_utility(
    player,
    players,
    strategy=None,
) -> float:
    if not has_playable_dev_card(
        player,
        DevCardType.KNIGHT,
    ):
        return float("-inf")

    utility = 1.0

    from catanlab.strategies import (
        StrategyType,
    )

    # OWS-style strategies naturally value
    # development cards and Largest Army more than
    # road-focused strategies.
    if strategy == StrategyType.FULL_OWS:
        utility += 1.5

    elif strategy == StrategyType.HYBRID_OWS:
        utility += 1.0

    if not player.has_largest_army:
        holder = next(
            (
                other
                for other in players
                if other.has_largest_army
            ),
            None,
        )

        target = (
            3
            if holder is None
            else holder.knights_played + 1
        )

        needed = target - player.knights_played

        if needed == 1:
            utility += 4.0

            if player.victory_points >= 8:
                utility += 6.0

        elif needed == 2:
            utility += 2.0

        elif (
            needed == 3
            and strategy in (
                StrategyType.FULL_OWS,
                StrategyType.HYBRID_OWS,
            )
        ):
            # OWS strategies should be willing to
            # begin the Largest Army race rather
            # than waiting until progress already
            # exists.
            utility += 1.0

    return utility


def monopoly_utility(
    player,
    players,
    inventories,
    phase: DevCardPhase,
    board=None,
) -> tuple[float, Resource | None]:
    """
    Estimate the value of playing Monopoly using
    public information only.

    Opponent resource identities are hidden. The
    heuristic therefore estimates likely holdings
    from each opponent's visible production profile
    and publicly known resource-card count.
    """

    if not has_playable_dev_card(
        player,
        DevCardType.MONOPOLY,
    ):
        return (
            float("-inf"),
            None,
        )

    resources = (
        Resource.WOOD,
        Resource.BRICK,
        Resource.SHEEP,
        Resource.WHEAT,
        Resource.ORE,
    )

    estimated_totals = {
        resource: 0.0
        for resource in resources
    }

    if board is not None:
        from catanlab.scoring import (
            resource_production,
        )

        for other in players:
            if (
                other.player_id
                == player.player_id
            ):
                continue

            # Opponent hand size is public, but its
            # resource composition is not.
            hand_size = inventories[
                other.player_id
            ].total()

            if hand_size <= 0:
                continue

            production = {
                resource: 0.0
                for resource in resources
            }

            for vertex_id in other.settlements:
                vertex_production = (
                    resource_production(
                        board,
                        board.vertices[
                            vertex_id
                        ],
                    )
                )

                for resource in resources:
                    production[
                        resource
                    ] += vertex_production[
                        resource
                    ]

            for vertex_id in other.cities:
                vertex_production = (
                    resource_production(
                        board,
                        board.vertices[
                            vertex_id
                        ],
                    )
                )

                # Cities produce twice as much as
                # settlements.
                for resource in resources:
                    production[
                        resource
                    ] += (
                        2
                        * vertex_production[
                            resource
                        ]
                    )

            total_production = sum(
                production.values()
            )

            if total_production <= 0:
                continue

            # Estimate how the opponent's public hand
            # might be distributed according to their
            # visible production profile.
            for resource in resources:
                estimated_totals[
                    resource
                ] += (
                    hand_size
                    * production[
                        resource
                    ]
                    / total_production
                )

    best_resource = max(
        resources,
        key=lambda resource: (
            estimated_totals[
                resource
            ],
            resource.value,
        ),
    )

    best_total = estimated_totals[
        best_resource
    ]

    # Without meaningful public evidence, hold the
    # card rather than guessing blindly.
    if best_total <= 0:
        return (
            0.0,
            None,
        )

    utility = float(
        best_total
    )

    # Before rolling, estimated Monopoly gains become
    # part of the player's hand and may increase
    # discard exposure if a seven is rolled.
    if phase == DevCardPhase.PRE_ROLL:
        own_inventory = inventories[
            player.player_id
        ]

        projected_hand = (
            own_inventory.total()
            + best_total
        )

        if projected_hand > 7:
            excess = (
                projected_hand
                - 7
            )

            utility -= (
                2.0
                + 0.5 * excess
            )

    return (
        utility,
        best_resource,
    )


def year_of_plenty_utility(
    player,
    inventory,
    phase: DevCardPhase,
) -> float:
    if not has_playable_dev_card(
        player,
        DevCardType.YEAR_OF_PLENTY,
    ):
        return float("-inf")

    utility = 1.0

    for build_type in (
        BuildType.CITY,
        BuildType.SETTLEMENT,
        BuildType.ROAD,
    ):
        if inventory.can_afford(
            build_type
        ):
            continue

        utility += 1.0

    if player.victory_points >= 8:
        utility += 2.0

    if (
        phase == DevCardPhase.PRE_ROLL
        and inventory.total() + 2 > 7
    ):
        excess = (
            inventory.total()
            + 2
            - 7
        )

        utility -= (
            2.0
            + 0.5 * excess
        )

    return utility


def road_building_utility(
    player,
    players,
) -> float:
    if not has_playable_dev_card(
        player,
        DevCardType.ROAD_BUILDING,
    ):
        return float("-inf")

    utility = 1.0

    current_length = longest_road_length(
        player,
        players,
    )

    if not player.has_longest_road:
        if current_length >= 3:
            utility += 2.0

        if current_length == 4:
            utility += 4.0

            if player.victory_points >= 8:
                utility += 5.0

    return utility


def choose_dev_card_play(
    player,
    players,
    inventories,
    phase: DevCardPhase = DevCardPhase.POST_ROLL,
    strategy=None,
    board=None,
) -> DevCardDecision:
    """
    Choose the highest-value development card
    currently worth playing.

    Returning card=None means hold all cards.
    """

    knight = knight_utility(
        player,
        players,
        strategy=strategy,
    )

    monopoly, monopoly_resource = (
        monopoly_utility(
            player,
            players,
            inventories,
            phase,
            board=board,
        )
    )

    plenty = year_of_plenty_utility(
        player,
        inventories[
            player.player_id
        ],
        phase,
    )

    road_building = (
        road_building_utility(
            player,
            players,
        )
    )

    decisions = [
        DevCardDecision(
            card=DevCardType.KNIGHT,
            utility=knight,
        ),
        DevCardDecision(
            card=DevCardType.MONOPOLY,
            utility=monopoly,
            resource=monopoly_resource,
        ),
        DevCardDecision(
            card=DevCardType.YEAR_OF_PLENTY,
            utility=plenty,
        ),
        DevCardDecision(
            card=DevCardType.ROAD_BUILDING,
            utility=road_building,
        ),
    ]

    best = max(
        decisions,
        key=lambda decision: (
            decision.utility,
            decision.card.value,
        ),
    )

    # Don't burn a card just because it exists.
    if best.utility <= 1.0:
        return DevCardDecision(
            card=None,
            utility=0.0,
        )

    return best
