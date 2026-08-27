from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from catanlab.game import run_game
from catanlab.search_agent import OneStepLookaheadAgent
from catanlab.strategies import StrategyType
from catanlab.turns import AdaptiveStrategyAgent


MATCHUPS = [
    (
        "hybrid_full_port",
        [
            StrategyType.HYBRID_OWS,
            StrategyType.FULL_OWS,
            StrategyType.PORT,
        ],
    ),
    (
        "hybrid_road_roadscity",
        [
            StrategyType.HYBRID_OWS,
            StrategyType.ROAD_BUILDING,
            StrategyType.ROADS_AND_CITIES,
        ],
    ),
    (
        "full_road_port",
        [
            StrategyType.FULL_OWS,
            StrategyType.ROAD_BUILDING,
            StrategyType.PORT,
        ],
    ),
    (
        "roadscity_port_hybrid",
        [
            StrategyType.ROADS_AND_CITIES,
            StrategyType.PORT,
            StrategyType.HYBRID_OWS,
        ],
    ),
    (
        "road_roadscity_port",
        [
            StrategyType.ROAD_BUILDING,
            StrategyType.ROADS_AND_CITIES,
            StrategyType.PORT,
        ],
    ),
]


@dataclass
class ResultRow:
    matchup: str
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


def make_strategies(
    target_seat: int,
    opponents: list[StrategyType],
) -> list[StrategyType]:
    result = []
    opponent_index = 0

    for seat in range(4):
        if seat == target_seat:
            result.append(
                StrategyType.FIVE_RESOURCE
            )
        else:
            result.append(
                opponents[opponent_index]
            )
            opponent_index += 1

    return result


def make_agents(
    strategies: list[StrategyType],
    target_seat: int,
    use_lookahead: bool,
    search_depth: int,
):
    agents = []

    for seat, strategy in enumerate(strategies):
        if (
            seat == target_seat
            and use_lookahead
        ):
            agents.append(
                OneStepLookaheadAgent(
                    strategy,
                    search_depth=search_depth,
                )
            )
        else:
            agents.append(
                AdaptiveStrategyAgent(strategy)
            )

    return agents


def run_one(
    matchup_name: str,
    opponents: list[StrategyType],
    variant: str,
    repetition: int,
    seat: int,
    seed: int,
    use_lookahead: bool,
    search_depth: int,
) -> ResultRow:
    strategies = make_strategies(
        seat,
        opponents,
    )

    agents = make_agents(
        strategies,
        seat,
        use_lookahead,
        search_depth,
    )

    result = run_game(
        strategies,
        seed=seed,
        max_turns=2000,
        validate_conservation=True,
        turn_agents=agents,
    )

    player = result.players[seat]

    return ResultRow(
        matchup=matchup_name,
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
        settlements=len(player.settlements),
        cities=len(player.cities),
        dev_cards=len(player.dev_cards),
        has_longest_road=int(
            player.has_longest_road
        ),
        has_largest_army=int(
            player.has_largest_army
        ),
    )


def paired_ci(series):
    mean = series.mean()
    n = len(series)

    if n <= 1:
        return mean, mean, mean

    sd = series.std(ddof=1)
    se = sd / math.sqrt(n)

    lo = mean - 1.96 * se
    hi = mean + 1.96 * se

    return mean, lo, hi


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repetitions",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--base-seed",
        type=int,
        default=20260827,
    )

    parser.add_argument(
        "--search-depth",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    if args.search_depth < 1:
        raise ValueError(
            "search_depth must be at least 1"
        )

    rows = []

    for matchup_index, (
        matchup_name,
        opponents,
    ) in enumerate(MATCHUPS):
        print()
        print(
            f"=== MATCHUP {matchup_index + 1}/"
            f"{len(MATCHUPS)}: {matchup_name} ==="
        )

        for repetition in range(
            args.repetitions
        ):
            for seat in range(4):
                seed = (
                    args.base_seed
                    + matchup_index * 1_000_000
                    + repetition * 4
                    + seat
                )

                rows.append(
                    run_one(
                        matchup_name,
                        opponents,
                        "baseline",
                        repetition,
                        seat,
                        seed,
                        use_lookahead=False,
                        search_depth=(
                            args.search_depth
                        ),
                    )
                )

                rows.append(
                    run_one(
                        matchup_name,
                        opponents,
                        "lookahead",
                        repetition,
                        seat,
                        seed,
                        use_lookahead=True,
                        search_depth=(
                            args.search_depth
                        ),
                    )
                )

            if (
                repetition + 1
            ) % 10 == 0:
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
        "results"
    ) / (
        "lookahead_matchups_depth"
        f"{args.search_depth}"
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
        df.groupby(
            [
                "matchup",
                "variant",
            ]
        )
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
            largest_army_rate=(
                "has_largest_army",
                "mean",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        out_dir / "summary.csv",
        index=False,
    )

    paired_rows = []

    metrics = [
        "won",
        "final_vp",
        "roads",
        "settlements",
        "cities",
        "dev_cards",
        "has_longest_road",
        "has_largest_army",
    ]

    for matchup_name, _ in MATCHUPS:
        subset = df[
            df["matchup"]
            == matchup_name
        ]

        baseline = subset[
            subset["variant"]
            == "baseline"
        ].sort_values(
            [
                "repetition",
                "seat",
            ]
        )

        lookahead = subset[
            subset["variant"]
            == "lookahead"
        ].sort_values(
            [
                "repetition",
                "seat",
            ]
        )

        merged = baseline.merge(
            lookahead,
            on=[
                "matchup",
                "repetition",
                "seat",
                "seed",
            ],
            suffixes=(
                "_baseline",
                "_lookahead",
            ),
            validate="one_to_one",
        )

        for metric in metrics:
            diff = (
                merged[
                    f"{metric}_lookahead"
                ]
                - merged[
                    f"{metric}_baseline"
                ]
            )

            mean, lo, hi = paired_ci(
                diff
            )

            paired_rows.append(
                {
                    "matchup": matchup_name,
                    "metric": metric,
                    "pairs": len(diff),
                    "mean_diff": mean,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "changed": int(
                        (diff != 0).sum()
                    ),
                }
            )

    paired = pd.DataFrame(
        paired_rows
    )

    paired.to_csv(
        out_dir / "paired_differences.csv",
        index=False,
    )

    print()
    print("=== SUMMARY ===")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("=== PAIRED WIN DIFFERENCES ===")

    win_rows = paired[
        paired["metric"] == "won"
    ]

    for row in win_rows.itertuples():
        print(
            f"{row.matchup:28s} "
            f"{row.mean_diff:+.4f} "
            f"95% CI "
            f"[{row.ci95_lo:+.4f}, "
            f"{row.ci95_hi:+.4f}]"
        )

    print()
    print("=== PAIRED VP DIFFERENCES ===")

    vp_rows = paired[
        paired["metric"]
        == "final_vp"
    ]

    for row in vp_rows.itertuples():
        print(
            f"{row.matchup:28s} "
            f"{row.mean_diff:+.4f} "
            f"95% CI "
            f"[{row.ci95_lo:+.4f}, "
            f"{row.ci95_hi:+.4f}]"
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
        f"  {out_dir / 'paired_differences.csv'}"
    )


if __name__ == "__main__":
    main()
