from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from catanlab.game import run_game
from catanlab.search_agent import (
    OneStepLookaheadAgent,
)
from catanlab.strategies import StrategyType
from catanlab.turns import AdaptiveStrategyAgent


@dataclass
class ResultRow:
    depth: int
    repetition: int
    seat: int
    seed: int
    won: int
    final_vp: int
    turns_played: int
    roads: int
    settlements: int
    cities: int
    dev_cards: int
    has_longest_road: int
    has_largest_army: int
    runtime_seconds: float


def make_strategies(
    target_seat: int,
) -> list[StrategyType]:
    opponents = [
        StrategyType.HYBRID_OWS,
        StrategyType.FULL_OWS,
        StrategyType.PORT,
    ]

    strategies = []
    opponent_index = 0

    for seat in range(4):
        if seat == target_seat:
            strategies.append(
                StrategyType.FIVE_RESOURCE
            )
        else:
            strategies.append(
                opponents[opponent_index]
            )
            opponent_index += 1

    return strategies


def make_agents(
    strategies: list[StrategyType],
    target_seat: int,
    depth: int,
):
    agents = []

    for seat, strategy in enumerate(
        strategies
    ):
        if seat == target_seat:
            agents.append(
                OneStepLookaheadAgent(
                    strategy,
                    search_depth=depth,
                )
            )
        else:
            agents.append(
                AdaptiveStrategyAgent(
                    strategy
                )
            )

    return agents


def run_one(
    depth: int,
    repetition: int,
    seat: int,
    seed: int,
) -> ResultRow:
    strategies = make_strategies(
        seat
    )

    agents = make_agents(
        strategies,
        seat,
        depth,
    )

    start = time.perf_counter()

    result = run_game(
        strategies,
        seed=seed,
        max_turns=2000,
        validate_conservation=True,
        turn_agents=agents,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    player = result.players[seat]

    return ResultRow(
        depth=depth,
        repetition=repetition,
        seat=seat,
        seed=seed,
        won=int(
            result.winner_id
            == player.player_id
        ),
        final_vp=player.victory_points,
        turns_played=result.turns_played,
        roads=len(player.roads),
        settlements=len(
            player.settlements
        ),
        cities=len(player.cities),
        dev_cards=len(
            player.dev_cards
        ),
        has_longest_road=int(
            player.has_longest_road
        ),
        has_largest_army=int(
            player.has_largest_army
        ),
        runtime_seconds=elapsed,
    )


def paired_ci(series):
    mean = series.mean()
    n = len(series)

    if n <= 1:
        return mean, mean, mean

    sd = series.std(ddof=1)
    se = sd / math.sqrt(n)

    return (
        mean,
        mean - 1.96 * se,
        mean + 1.96 * se,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repetitions",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--base-seed",
        type=int,
        default=20260827,
    )

    parser.add_argument(
        "--depth-a",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--depth-b",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    if args.depth_a < 1 or args.depth_b < 1:
        raise ValueError(
            "Search depths must be at least 1."
        )

    if args.depth_a == args.depth_b:
        raise ValueError(
            "Search depths must be different."
        )

    rows = []

    for repetition in range(
        args.repetitions
    ):
        for seat in range(4):
            seed = (
                args.base_seed
                + repetition * 4
                + seat
            )

            for depth in (
                args.depth_a,
                args.depth_b,
            ):
                rows.append(
                    run_one(
                        depth,
                        repetition,
                        seat,
                        seed,
                    )
                )

            print(
                f"completed "
                f"rep={repetition:02d} "
                f"seat={seat}"
            )

    df = pd.DataFrame(
        row.__dict__
        for row in rows
    )

    out_dir = Path(
        "results/search_depth"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        out_dir / "player_games.csv",
        index=False,
    )

    summary = (
        df.groupby("depth")
        .agg(
            games=("won", "size"),
            win_rate=("won", "mean"),
            avg_vp=("final_vp", "mean"),
            avg_roads=("roads", "mean"),
            avg_settlements=(
                "settlements",
                "mean",
            ),
            avg_cities=("cities", "mean"),
            avg_dev_cards=(
                "dev_cards",
                "mean",
            ),
            longest_road_rate=(
                "has_longest_road",
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
    print("=== DEPTH SUMMARY ===")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    depth1 = df[
        df["depth"] == args.depth_a
    ]

    depth2 = df[
        df["depth"] == args.depth_b
    ]

    merged = depth1.merge(
        depth2,
        on=[
            "repetition",
            "seat",
            "seed",
        ],
        suffixes=(
            "_d1",
            "_d2",
        ),
        validate="one_to_one",
    )

    print()
    print(
        "=== PAIRED CHANGES "
        f"(DEPTH {args.depth_b} - "
        f"DEPTH {args.depth_a}) ==="
    )

    for metric in [
        "won",
        "final_vp",
        "roads",
        "settlements",
        "cities",
        "dev_cards",
        "has_longest_road",
        "runtime_seconds",
    ]:
        diff = (
            merged[f"{metric}_d2"]
            - merged[f"{metric}_d1"]
        )

        mean, lo, hi = paired_ci(
            diff
        )

        print(
            f"{metric:20s} "
            f"{mean:+8.4f} "
            f"95% CI "
            f"[{lo:+.4f}, {hi:+.4f}]"
        )

    summary.to_csv(
        out_dir / "summary.csv",
        index=False,
    )

    print()
    print("Saved:")
    print(
        f"  {out_dir / 'player_games.csv'}"
    )
    print(
        f"  {out_dir / 'summary.csv'}"
    )


if __name__ == "__main__":
    main()
