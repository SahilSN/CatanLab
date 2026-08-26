from dataclasses import dataclass

from catanlab.economy import BuildType
from catanlab.longest_road import longest_road_length
from catanlab.simulation import PlayerState
from catanlab.strategies import StrategyType
from catanlab.turns import ActionType


@dataclass(frozen=True)
class ActionUtilities:
    build_city: float
    build_settlement: float
    build_road: float
    buy_dev_card: float
    pass_turn: float

    def as_dict(
        self,
    ) -> dict[ActionType, float]:
        return {
            ActionType.BUILD_CITY:
                self.build_city,
            ActionType.BUILD_SETTLEMENT:
                self.build_settlement,
            ActionType.BUILD_ROAD:
                self.build_road,
            ActionType.BUY_DEV_CARD:
                self.buy_dev_card,
            ActionType.PASS:
                self.pass_turn,
        }


BASE_ACTION_WEIGHTS = {
    StrategyType.FULL_OWS: {
        ActionType.BUILD_CITY: 6.0,
        ActionType.BUY_DEV_CARD: 5.5,
        ActionType.BUILD_SETTLEMENT: 1.5,
        ActionType.BUILD_ROAD: 0.5,
        ActionType.PASS: 0.0,
    },

    StrategyType.HYBRID_OWS: {
        ActionType.BUILD_CITY: 5.5,
        ActionType.BUY_DEV_CARD: 4.5,
        ActionType.BUILD_SETTLEMENT: 3.0,
        ActionType.BUILD_ROAD: 2.0,
        ActionType.PASS: 0.0,
    },

    StrategyType.ROAD_BUILDING: {
        ActionType.BUILD_CITY: 1.5,
        ActionType.BUY_DEV_CARD: 0.5,
        ActionType.BUILD_SETTLEMENT: 5.5,
        ActionType.BUILD_ROAD: 5.0,
        ActionType.PASS: 0.0,
    },

    StrategyType.ROADS_AND_CITIES: {
        ActionType.BUILD_CITY: 5.0,
        ActionType.BUY_DEV_CARD: 1.5,
        ActionType.BUILD_SETTLEMENT: 4.0,
        ActionType.BUILD_ROAD: 3.5,
        ActionType.PASS: 0.0,
    },

    StrategyType.FIVE_RESOURCE: {
        ActionType.BUILD_CITY: 4.0,
        ActionType.BUY_DEV_CARD: 2.5,
        ActionType.BUILD_SETTLEMENT: 4.0,
        ActionType.BUILD_ROAD: 3.0,
        ActionType.PASS: 0.0,
    },

    StrategyType.PORT: {
        ActionType.BUILD_CITY: 3.0,
        ActionType.BUY_DEV_CARD: 2.0,
        ActionType.BUILD_SETTLEMENT: 4.5,
        ActionType.BUILD_ROAD: 4.0,
        ActionType.PASS: 0.0,
    },
}


def largest_army_target(
    player: PlayerState,
    players: list[PlayerState],
) -> int:
    """
    Return how many additional played Knights the
    player needs to take Largest Army.

    Returns 0 if the player already holds it.
    """

    if player.has_largest_army:
        return 0

    holder = next(
        (
            other
            for other in players
            if other.has_largest_army
        ),
        None,
    )

    if holder is None:
        target = 3
    else:
        target = (
            holder.knights_played
            + 1
        )

    return max(
        0,
        target - player.knights_played,
    )


def longest_road_target(
    player: PlayerState,
    players: list[PlayerState],
) -> int:
    """
    Return how many additional road segments are
    approximately needed to take Longest Road.

    This is a heuristic based on current longest
    road length, not a guarantee of future topology.
    """

    if player.has_longest_road:
        return 0

    own_length = longest_road_length(
        player,
        players,
    )

    holder = next(
        (
            other
            for other in players
            if other.has_longest_road
        ),
        None,
    )

    if holder is None:
        target_length = 5
    else:
        target_length = (
            longest_road_length(
                holder,
                players,
            )
            + 1
        )

    return max(
        0,
        target_length - own_length,
    )


def score_actions(
    strategy: StrategyType,
    player: PlayerState,
    players: list[PlayerState],
) -> ActionUtilities:
    """
    Compute strategy-aware action utilities from
    the current player state.

    These values are heuristics. They will be tuned
    later through simulation rather than treated as
    fixed truths.
    """

    base = BASE_ACTION_WEIGHTS[
        strategy
    ]

    city = base[
        ActionType.BUILD_CITY
    ]

    settlement = base[
        ActionType.BUILD_SETTLEMENT
    ]

    road = base[
        ActionType.BUILD_ROAD
    ]

    dev = base[
        ActionType.BUY_DEV_CARD
    ]

    pass_turn = base[
        ActionType.PASS
    ]

    vp = player.victory_points

    # Cities become increasingly valuable near the
    # end of the game because each upgrade adds 1 VP.
    if vp >= 7:
        city += 2.0

    if vp >= 9:
        city += 4.0

    army_needed = largest_army_target(
        player,
        players,
    )

    # Dev-card pursuit becomes more attractive when
    # Largest Army is close, especially if the award
    # would create a direct winning path.
    if army_needed == 1:
        dev += 3.0

        if vp >= 8:
            dev += 5.0

    elif army_needed == 2:
        dev += 1.5

    road_needed = longest_road_target(
        player,
        players,
    )

    if road_needed == 1:
        road += 3.0

        if vp >= 8:
            road += 5.0

    elif road_needed == 2:
        road += 1.5

    # The road-building strategy should transition
    # from acquiring Longest Road to converting its
    # network into direct victory points.
    #
    # It remains reactive: if another player is close
    # to overtaking the road, road utility rises again.
    if (
        strategy == StrategyType.ROAD_BUILDING
        and player.has_longest_road
    ):
        own_road_length = longest_road_length(
            player,
            players,
        )

        best_opponent_length = max(
            (
                longest_road_length(
                    other,
                    players,
                )
                for other in players
                if (
                    other.player_id
                    != player.player_id
                )
            ),
            default=0,
        )

        road_lead = (
            own_road_length
            - best_opponent_length
        )

        if road_lead <= 1:
            # Longest Road is under immediate
            # pressure, so defend it.
            road += 2.0
            settlement += 0.5

        else:
            # With a comfortable lead, another road
            # is usually less useful than converting
            # the network into settlements/cities.
            road -= 5.5
            settlement += 2.0
            city += 2.5

        if vp >= 7:
            # Late-game pivot toward direct VP.
            city += 2.0
            settlement += 1.0
            road -= 1.0

    # Expansion becomes less attractive near the
    # finish unless it contributes directly to a
    # victory path.
    if vp >= 9:
        settlement -= 1.0

    return ActionUtilities(
        build_city=city,
        build_settlement=settlement,
        build_road=road,
        buy_dev_card=dev,
        pass_turn=pass_turn,
    )
