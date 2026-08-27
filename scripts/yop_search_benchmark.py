from __future__ import annotations

import argparse
import math
import statistics
import time
from pathlib import Path

import pandas as pd

from catanlab.game import run_game
from catanlab.search_agent import OneStepLookaheadAgent
from catanlab.strategies import StrategyType
from catanlab.turns import AdaptiveStrategyAgent


def mean_ci95(values):
    if not values:
        return 0.0, 0.0, 0.0

    mean = statistics.mean(values)

    if len(values) == 1:
        return mean, mean, mean

    stdev = statistics.stdev(values)
    half = 1.96 * stdev / math.sqrt(len(values))

    return mean, mean - half, mean + half


def make_lineup(
    target_seat: int,
    search_yop: bool,
):
    opponents = [
        StrategyType.HYBRID_OWS,
        StrategyType.FULL_OWS,
        StrategyType.PORT,
    ]

    strategies = []
    agents = []
    opponent_index = 0

    for seat in range(4):
        if seat == target_seat:
            strategy = StrategyType.FIVE_RESOURCE

            strategies.append(strategy)

            agents.append(
                OneStepLookaheadAgent(
                    strategy,
                    search_depth=2,
                    use_transposition_cache=False,
                    search_maritime_trades=True,
                    search_year_of_plenty=search_yop,
                )
            )

        else:
            strategy = opponents[
                opponent_index
            ]
            opponent_index += 1

            strategies.append(strategy)

            agents.append(
                AdaptiveStrategyAgent(
                    strategy
                )
            )

    return strategies, agents


def player_row(
    result,
    target_seat,
    variant,
    seed,
    runtime,
):
    player = result.players[target_seat]

    return {
        "variant": variant,
        "seed": seed,
        "seat": target_seat,
        "won": int(
            result.winner_id == target_seat
        ),
        "final_vp": player.victory_points,
        "roads": len(player.roads),
        "settlements": len(
            player.settlements
        ),
        "cities": len(player.cities),
        "dev_cards": len(
            player.dev_cards
        ),
        "has_longest_road": int(
            player.has_longest_road
        ),
        "has_largest_army": int(
            player.has_largest_army
        ),
        "runtime_seconds": runtime,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repetitions",
        type=int,
        default=25,
    )

    args = parser.parse_args()

    rows = []

    for repetition in range(
        args.repetitions
    ):
        for seat in range(4):
            seed = (
                repetition * 4
                + seat
            )

            for (
                variant,
                search_yop,
            ) in (
                ("yop_off", False),
                ("yop_on", True),
            ):
                (
                    strategies,
                    agents,
                ) = make_lineup(
                    target_seat=seat,
                    search_yop=search_yop,
                )

                start = (
                    time.perf_counter()
                )

                result = run_game(
                    seed=seed,
                    strategies=strategies,
                    turn_agents=agents,
                )

                runtime = (
                    time.perf_counter()
                    - start
                )

                rows.append(
                    player_row(
                        result,
                        seat,
                        variant,
                        seed,
                        runtime,
                    )
                )

            completed = (
                repetition * 4
                + seat
                + 1
            )

            total = (
                args.repetitions * 4
            )

            print(
                f"[{completed}/{total}] "
                f"rep={repetition + 1}/"
                f"{args.repetitions} "
                f"seat={seat}",
                flush=True,
            )

    df = pd.DataFrame(rows)

    summary = (
        df.groupby("variant")
        .agg(
            games=("won", "size"),
            win_rate=("won", "mean"),
            avg_vp=("final_vp", "mean"),
            avg_roads=("roads", "mean"),
            avg_settlements=(
                "settlements",
                "mean",
            ),
            avg_cities=(
                "cities",
                "mean",
            ),
            avg_dev_cards=(
                "dev_cards",
                "mean",
            ),
            longest_road_rate=(
                "has_longest_road",
                "mean",
            ),
            largest_army_rate=(
                "has_largest_army",
                "mean",
            ),
            avg_runtime_seconds=(
                "runtime_seconds",
                "mean",
            ),
        )
        .reset_index()
    )

    print()
    print(
        "=== YEAR OF PLENTY "
        "SEARCH SUMMARY ==="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    off = (
        df[
            df["variant"] == "yop_off"
        ]
        .sort_values(
            ["seed", "seat"]
        )
        .reset_index(drop=True)
    )

    on = (
        df[
            df["variant"] == "yop_on"
        ]
        .sort_values(
            ["seed", "seat"]
        )
        .reset_index(drop=True)
    )

    assert list(off["seed"]) == list(
        on["seed"]
    )

    assert list(off["seat"]) == list(
        on["seat"]
    )

    print()
    print(
        "=== PAIRED CHANGES "
        "(YOP ON - OFF) ==="
    )

    metrics = [
        "won",
        "final_vp",
        "roads",
        "settlements",
        "cities",
        "dev_cards",
        "has_longest_road",
        "has_largest_army",
        "runtime_seconds",
    ]

    for metric in metrics:
        diffs = (
            on[metric]
            - off[metric]
        ).tolist()

        mean, lo, hi = (
            mean_ci95(diffs)
        )

        print(
            f"{metric:22s} "
            f"{mean:+.4f} "
            f"95% CI "
            f"[{lo:+.4f}, "
            f"{hi:+.4f}]"
        )

    paired = off[
        [
            "seed",
            "seat",
        ]
    ].copy()

    for metric in metrics:
        paired[
            f"{metric}_off"
        ] = off[metric]

        paired[
            f"{metric}_on"
        ] = on[metric]

        paired[
            f"{metric}_diff"
        ] = (
            on[metric]
            - off[metric]
        )

    out_dir = Path(
        "results/yop_hold_play_search"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        out_dir
        / "player_games.csv",
        index=False,
    )

    summary.to_csv(
        out_dir
        / "summary.csv",
        index=False,
    )

    paired.to_csv(
        out_dir
        / "paired_games.csv",
        index=False,
    )

    print()
    print("Saved:")
    print(
        f"  {out_dir}/"
        "player_games.csv"
    )
    print(
        f"  {out_dir}/"
        "summary.csv"
    )
    print(
        f"  {out_dir}/"
        "paired_games.csv"
    )


if __name__ == "__main__":
    main()
