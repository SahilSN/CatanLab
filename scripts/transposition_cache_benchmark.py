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
    variant: str
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
    """
    Use the same representative opponent composition
    for every paired cache comparison.
    """
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
    use_cache: bool,
):
    agents = []

    for seat, strategy in enumerate(
        strategies
    ):
        if seat == target_seat:
            agents.append(
                OneStepLookaheadAgent(
                    strategy,
                    search_depth=2,
                    use_transposition_cache=use_cache,
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
    variant: str,
    repetition: int,
    seat: int,
    seed: int,
    use_cache: bool,
) -> ResultRow:
    strategies = make_strategies(
        seat
    )

    agents = make_agents(
        strategies,
        seat,
        use_cache,
    )

    start = time.perf_counter()

    result = run_game(
        strategies,
        seed=seed,
        max_turns=2000,
        validate_conservation=True,
        turn_agents=agents,
    )

    runtime = (
        time.perf_counter()
        - start
    )

    player = result.players[seat]

    return ResultRow(
        variant=variant,
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
        runtime_seconds=runtime,
    )


def paired_ci(
    series,
):
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
        default=10,
    )

    parser.add_argument(
        "--base-seed",
        type=int,
        default=20260827,
    )

    args = parser.parse_args()

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

            uncached = run_one(
                "uncached",
                repetition,
                seat,
                seed,
                use_cache=False,
            )

            cached = run_one(
                "cached",
                repetition,
                seat,
                seed,
                use_cache=True,
            )

            # Caching is an implementation optimization,
            # not a policy change. Paired games must be
            # behaviorally identical.
            fields_that_must_match = [
                "won",
                "final_vp",
                "turns_played",
                "roads",
                "settlements",
                "cities",
                "dev_cards",
                "has_longest_road",
                "has_largest_army",
            ]

            for field in fields_that_must_match:
                uncached_value = getattr(
                    uncached,
                    field,
                )
                cached_value = getattr(
                    cached,
                    field,
                )

                if (
                    uncached_value
                    != cached_value
                ):
                    raise AssertionError(
                        "Cache changed game behavior: "
                        f"rep={repetition} "
                        f"seat={seat} "
                        f"seed={seed} "
                        f"field={field} "
                        f"uncached={uncached_value} "
                        f"cached={cached_value}"
                    )

            rows.extend(
                [
                    uncached,
                    cached,
                ]
            )

        print(
            f"completed "
            f"{repetition + 1}/"
            f"{args.repetitions} repetitions"
        )

    df = pd.DataFrame(
        row.__dict__
        for row in rows
    )

    out_dir = Path(
        "results/transposition_cache"
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
        df.groupby("variant")
        .agg(
            games=(
                "runtime_seconds",
                "size",
            ),
            avg_runtime_seconds=(
                "runtime_seconds",
                "mean",
            ),
            median_runtime_seconds=(
                "runtime_seconds",
                "median",
            ),
        )
        .reset_index()
    )

    uncached = df[
        df["variant"] == "uncached"
    ]

    cached = df[
        df["variant"] == "cached"
    ]

    paired = uncached.merge(
        cached,
        on=[
            "repetition",
            "seat",
            "seed",
        ],
        suffixes=(
            "_uncached",
            "_cached",
        ),
        validate="one_to_one",
    )

    runtime_diff = (
        paired[
            "runtime_seconds_cached"
        ]
        - paired[
            "runtime_seconds_uncached"
        ]
    )

    mean_diff, lo, hi = paired_ci(
        runtime_diff
    )

    speedups = (
        paired[
            "runtime_seconds_uncached"
        ]
        / paired[
            "runtime_seconds_cached"
        ]
    )

    savings = (
        1.0
        - (
            paired[
                "runtime_seconds_cached"
            ]
            / paired[
                "runtime_seconds_uncached"
            ]
        )
    )

    print()
    print("=== CACHE RUNTIME SUMMARY ===")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=== PAIRED RUNTIME CHANGE ===")

    print(
        "cached - uncached: "
        f"{mean_diff:+.4f} s "
        f"95% CI "
        f"[{lo:+.4f}, {hi:+.4f}]"
    )

    print(
        "mean paired speedup: "
        f"{speedups.mean():.3f}x"
    )

    print(
        "median paired speedup: "
        f"{speedups.median():.3f}x"
    )

    print(
        "mean paired runtime saving: "
        f"{100 * savings.mean():.2f}%"
    )

    print()
    print(
        "Behavioral equivalence: "
        f"PASS ({len(paired)} paired games)"
    )

    summary.to_csv(
        out_dir / "summary.csv",
        index=False,
    )

    paired.to_csv(
        out_dir / "paired_games.csv",
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
    print(
        f"  {out_dir / 'paired_games.csv'}"
    )


if __name__ == "__main__":
    main()
