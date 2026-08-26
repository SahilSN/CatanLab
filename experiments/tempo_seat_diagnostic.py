import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

from catanlab.game import run_game
from catanlab.strategies import StrategyType
from catanlab.turns import ActionType


STRATEGIES = [
    StrategyType.FULL_OWS,
    StrategyType.HYBRID_OWS,
    StrategyType.ROAD_BUILDING,
    StrategyType.ROADS_AND_CITIES,
    StrategyType.FIVE_RESOURCE,
    StrategyType.PORT,
]

NUM_SEEDS = 100
MAX_TURNS = 500

RESULTS_DIR = Path(
    "results/tempo_seat_diagnostic"
)

RAW_CSV = RESULTS_DIR / "tempo.csv"
SUMMARY_CSV = RESULTS_DIR / "tempo_summary.csv"


def strategy_name(
    strategy: StrategyType,
) -> str:
    return strategy.value


def first_action_turn(
    result,
    player_id: int,
    action_type: ActionType,
):
    """
    Return both:
      - global individual-turn number
      - that player's own turn number

    for the first matching normal action.
    """

    own_turn = 0

    for global_index, turn in enumerate(
        result.turn_history,
        start=1,
    ):
        if turn.player_id != player_id:
            continue

        own_turn += 1

        if any(
            action.action_type
            == action_type
            for action in turn.actions
        ):
            return (
                global_index,
                own_turn,
            )

    return (
        None,
        None,
    )


def first_award_turn(
    result,
    player_id: int,
    award: str,
):
    for snapshot in (
        result.award_history
    ):
        if award == "largest_army":
            holder = (
                snapshot.largest_army_holder
            )

        elif award == "longest_road":
            holder = (
                snapshot.longest_road_holder
            )

        else:
            raise ValueError(
                f"Unknown award: {award}"
            )

        if holder == player_id:
            return snapshot.turn_number

    return None


def own_turn_number(
    global_turn: int | None,
    player_id: int,
) -> int | None:
    """
    Convert a global individual-player turn number
    to the number of turns that player has taken.

    Turn 1 = P1's first turn,
    Turn 2 = P2's first turn, etc.
    """

    if global_turn is None:
        return None

    first_turn = (
        player_id + 1
    )

    if global_turn < first_turn:
        return None

    return (
        (global_turn - first_turn)
        // 4
        + 1
    )


def optional_mean(
    values,
):
    present = [
        value
        for value in values
        if value is not None
    ]

    if not present:
        return None

    return mean(
        present
    )


def fmt(
    value,
) -> str:
    if value is None:
        return "-"

    return f"{value:.2f}"


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    total_games = (
        len(STRATEGIES)
        * NUM_SEEDS
    )

    completed = 0

    for strategy in STRATEGIES:
        name = strategy_name(
            strategy
        )

        print()
        print(
            f"Tempo control: {name}"
        )

        for seed in range(
            NUM_SEEDS
        ):
            result = run_game(
                [
                    strategy,
                    strategy,
                    strategy,
                    strategy,
                ],
                seed=seed,
                max_turns=MAX_TURNS,
            )

            completed += 1

            for player_id in range(4):
                road_global, road_own = (
                    first_action_turn(
                        result,
                        player_id,
                        ActionType.BUILD_ROAD,
                    )
                )

                settlement_global, settlement_own = (
                    first_action_turn(
                        result,
                        player_id,
                        ActionType.BUILD_SETTLEMENT,
                    )
                )

                city_global, city_own = (
                    first_action_turn(
                        result,
                        player_id,
                        ActionType.BUILD_CITY,
                    )
                )

                dev_global, dev_own = (
                    first_action_turn(
                        result,
                        player_id,
                        ActionType.BUY_DEV_CARD,
                    )
                )

                lr_global = first_award_turn(
                    result,
                    player_id,
                    "longest_road",
                )

                la_global = first_award_turn(
                    result,
                    player_id,
                    "largest_army",
                )

                rows.append(
                    {
                        "strategy":
                            name,
                        "seed":
                            seed,
                        "seat":
                            player_id + 1,
                        "winner":
                            int(
                                result.winner_id
                                == player_id
                            ),
                        "game_turns":
                            result.turns_played,

                        "first_road_global":
                            road_global,
                        "first_road_own":
                            road_own,

                        "first_settlement_global":
                            settlement_global,
                        "first_settlement_own":
                            settlement_own,

                        "first_city_global":
                            city_global,
                        "first_city_own":
                            city_own,

                        "first_dev_global":
                            dev_global,
                        "first_dev_own":
                            dev_own,

                        "first_lr_global":
                            lr_global,
                        "first_lr_own":
                            own_turn_number(
                                lr_global,
                                player_id,
                            ),

                        "first_la_global":
                            la_global,
                        "first_la_own":
                            own_turn_number(
                                la_global,
                                player_id,
                            ),
                    }
                )

            if (
                (seed + 1) % 20
                == 0
            ):
                print(
                    f"  {seed + 1:3d}/"
                    f"{NUM_SEEDS}"
                )

    print()
    print(
        f"Completed {completed}/"
        f"{total_games} games."
    )

    with RAW_CSV.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    summary_rows = []

    metrics = (
        "first_road_own",
        "first_settlement_own",
        "first_city_own",
        "first_dev_own",
        "first_lr_own",
        "first_la_own",
    )

    for strategy in STRATEGIES:
        name = strategy_name(
            strategy
        )

        for seat in range(
            1,
            5,
        ):
            subset = [
                row
                for row in rows
                if (
                    row["strategy"] == name
                    and row["seat"] == seat
                )
            ]

            summary = {
                "strategy":
                    name,
                "seat":
                    seat,
                "games":
                    len(subset),
                "win_rate":
                    mean(
                        row["winner"]
                        for row in subset
                    ),
            }

            for metric in metrics:
                summary[
                    metric
                ] = optional_mean(
                    row[metric]
                    for row in subset
                )

                reached = sum(
                    row[metric]
                    is not None
                    for row in subset
                )

                summary[
                    metric + "_rate"
                ] = (
                    reached
                    / len(subset)
                )

            summary_rows.append(
                summary
            )

    with SUMMARY_CSV.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    print()
    print("=" * 112)
    print(
        "EARLY-GAME TEMPO BY STRATEGY AND SEAT"
    )
    print("=" * 112)

    header = (
        f"{'Strategy':20} "
        f"{'Seat':>4} "
        f"{'Win%':>7} "
        f"{'Road':>7} "
        f"{'Sett':>7} "
        f"{'City':>7} "
        f"{'Dev':>7} "
        f"{'LR':>7} "
        f"{'LA':>7}"
    )

    print(header)
    print(
        "-" * len(header)
    )

    for row in summary_rows:
        print(
            f"{row['strategy']:20} "
            f"P{row['seat']:<3d} "
            f"{100 * row['win_rate']:6.1f}% "
            f"{fmt(row['first_road_own']):>7} "
            f"{fmt(row['first_settlement_own']):>7} "
            f"{fmt(row['first_city_own']):>7} "
            f"{fmt(row['first_dev_own']):>7} "
            f"{fmt(row['first_lr_own']):>7} "
            f"{fmt(row['first_la_own']):>7}"
        )

    print()
    print(
        "Values are average PLAYER turns to "
        "first milestone among games where that "
        "milestone was reached."
    )

    print(
        f"Raw tempo data: {RAW_CSV}"
    )
    print(
        f"Summary:        {SUMMARY_CSV}"
    )


if __name__ == "__main__":
    main()
