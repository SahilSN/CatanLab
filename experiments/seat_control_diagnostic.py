import csv
from collections import Counter
from pathlib import Path
from statistics import mean

from catanlab.game import run_game
from catanlab.strategies import StrategyType


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
    "results/seat_control_diagnostic"
)

RAW_CSV = RESULTS_DIR / "games.csv"


def strategy_name(strategy):
    return strategy.value


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for strategy in STRATEGIES:
        name = strategy_name(
            strategy
        )

        print()
        print(
            f"Running identical-agent control: "
            f"{name}"
        )

        wins = Counter()
        turns = []
        unfinished = 0

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

            turns.append(
                result.turns_played
            )

            if result.winner_id is None:
                unfinished += 1
            else:
                wins[
                    result.winner_id
                ] += 1

            rows.append(
                {
                    "strategy": name,
                    "seed": seed,
                    "winner_seat": (
                        ""
                        if result.winner_id
                        is None
                        else result.winner_id + 1
                    ),
                    "turns":
                        result.turns_played,
                    "finished":
                        int(
                            result.winner_id
                            is not None
                        ),
                }
            )

        print(
            f"  avg turns: "
            f"{mean(turns):.1f}"
        )

        print(
            f"  unfinished: "
            f"{unfinished}"
        )

        for seat in range(4):
            print(
                f"  P{seat + 1}: "
                f"{wins[seat]}/{NUM_SEEDS} "
                f"({100 * wins[seat] / NUM_SEEDS:.1f}%)"
            )

    with RAW_CSV.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "strategy",
                "seed",
                "winner_seat",
                "turns",
                "finished",
            ],
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print()
    print(
        f"Saved raw results to {RAW_CSV}"
    )


if __name__ == "__main__":
    main()
