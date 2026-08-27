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


@dataclass
class PilotRow:
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
) -> list[StrategyType]:
    opponents = [
        StrategyType.HYBRID_OWS,
        StrategyType.FULL_OWS,
        StrategyType.PORT,
    ]

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
):
    agents = []

    for seat, strategy in enumerate(strategies):
        if (
            seat == target_seat
            and use_lookahead
        ):
            agents.append(
                OneStepLookaheadAgent(strategy)
            )
        else:
            agents.append(
                AdaptiveStrategyAgent(strategy)
            )

    return agents


def run_variant(
    variant: str,
    repetition: int,
    seat: int,
    seed: int,
    use_lookahead: bool,
) -> PilotRow:
    strategies = make_strategies(seat)
    agents = make_agents(
        strategies,
        seat,
        use_lookahead,
    )

    result = run_game(
        strategies,
        seed=seed,
        max_turns=2000,
        validate_conservation=True,
        turn_agents=agents,
    )

    player = result.players[seat]

    return PilotRow(
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repetitions",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=20260826,
    )
    args = parser.parse_args()

    rows = []

    repetitions = args.repetitions
    base_seed = args.base_seed

    for repetition in range(repetitions):
        for seat in range(4):
            seed = (
                base_seed
                + repetition * 4
                + seat
            )

            rows.append(
                run_variant(
                    "baseline",
                    repetition,
                    seat,
                    seed,
                    use_lookahead=False,
                )
            )

            rows.append(
                run_variant(
                    "lookahead",
                    repetition,
                    seat,
                    seed,
                    use_lookahead=True,
                )
            )

            print(
                f"completed rep={repetition:02d} "
                f"seat={seat}"
            )

    df = pd.DataFrame(
        row.__dict__
        for row in rows
    )

    out_dir = Path(
        "results/lookahead_pilot"
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

    print()
    print("=== PILOT SUMMARY ===")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    baseline = df[
        df["variant"] == "baseline"
    ].sort_values(
        ["repetition", "seat"]
    )

    lookahead = df[
        df["variant"] == "lookahead"
    ].sort_values(
        ["repetition", "seat"]
    )

    merged = baseline.merge(
        lookahead,
        on=[
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

    print()
    print("=== PAIRED CHANGES ===")

    for metric in [
        "won",
        "final_vp",
        "roads",
        "settlements",
        "cities",
        "dev_cards",
        "has_longest_road",
        "has_largest_army",
    ]:
        diff = (
            merged[
                f"{metric}_lookahead"
            ]
            - merged[
                f"{metric}_baseline"
            ]
        )

        mean = diff.mean()
        n = len(diff)

        if n > 1:
            sd = diff.std(ddof=1)
            se = sd / math.sqrt(n)
            lo = mean - 1.96 * se
            hi = mean + 1.96 * se
        else:
            lo = mean
            hi = mean

        print(
            f"{metric:20s} "
            f"{mean:+8.4f} "
            f"95% CI [{lo:+.4f}, {hi:+.4f}] "
            f"changed={(diff != 0).sum():4d}"
        )

    print()
    print("Saved:")
    print(f"  {out_dir / 'player_games.csv'}")
    print(f"  {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
