import csv
import itertools
from collections import Counter, defaultdict
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

NUM_SEED_BLOCKS = 5
MAX_TURNS = 500

RESULTS_DIR = Path(
    "results/exhaustive_strategy_diagnostic"
)

RAW_CSV = RESULTS_DIR / "games.csv"
SUMMARY_CSV = RESULTS_DIR / "strategy_summary.csv"
SEAT_CSV = RESULTS_DIR / "seat_summary.csv"


def strategy_name(
    strategy: StrategyType,
) -> str:
    return strategy.value


def rotate(
    lineup: tuple[StrategyType, ...],
    amount: int,
) -> tuple[StrategyType, ...]:
    return (
        lineup[amount:]
        + lineup[:amount]
    )


def dev_vp_count(
    player,
) -> int:
    return sum(
        card == "victory_point"
        for card in player.dev_cards
    )


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    lineups = list(
        itertools.combinations(
            STRATEGIES,
            4,
        )
    )

    assert len(lineups) == 15

    total_games = (
        NUM_SEED_BLOCKS
        * len(lineups)
        * 4
    )

    rows = []

    completed = 0

    for seed_block in range(
        NUM_SEED_BLOCKS
    ):
        # All lineups and all four seat rotations
        # within a block use the same game seed.
        #
        # This gives us paired comparisons across
        # seat assignments while multiple blocks
        # expose strategies to different boards and
        # development-card orders.
        game_seed = seed_block

        for lineup_index, lineup in enumerate(
            lineups
        ):
            lineup_label = "|".join(
                strategy_name(
                    strategy
                )
                for strategy in lineup
            )

            for rotation in range(4):
                seated = rotate(
                    lineup,
                    rotation,
                )

                result = run_game(
                    list(seated),
                    seed=game_seed,
                    max_turns=MAX_TURNS,
                )

                completed += 1

                if result.winner_id is None:
                    winner_strategy = None
                else:
                    winner_strategy = (
                        seated[
                            result.winner_id
                        ]
                    )

                print(
                    f"[{completed:03d}/{total_games}] "
                    f"seed={seed_block} "
                    f"lineup={lineup_index + 1:02d}/15 "
                    f"rotation={rotation} "
                    f"winner="
                    f"{'None' if winner_strategy is None else strategy_name(winner_strategy)} "
                    f"turns={result.turns_played}"
                )

                for seat, player in enumerate(
                    result.players
                ):
                    strategy = seated[
                        seat
                    ]

                    rows.append(
                        {
                            "seed_block":
                                seed_block,
                            "game_seed":
                                game_seed,
                            "lineup_index":
                                lineup_index,
                            "lineup":
                                lineup_label,
                            "rotation":
                                rotation,
                            "seat":
                                seat + 1,
                            "player_id":
                                player.player_id,
                            "strategy":
                                strategy_name(
                                    strategy
                                ),
                            "winner":
                                int(
                                    result.winner_id
                                    == player.player_id
                                ),
                            "game_finished":
                                int(
                                    result.winner_id
                                    is not None
                                ),
                            "turns":
                                result.turns_played,
                            "victory_points":
                                player.victory_points,
                            "roads":
                                len(
                                    player.roads
                                ),
                            "settlements":
                                len(
                                    player.settlements
                                ),
                            "cities":
                                len(
                                    player.cities
                                ),
                            "knights":
                                player.knights_played,
                            "largest_army":
                                int(
                                    player.has_largest_army
                                ),
                            "longest_road":
                                int(
                                    player.has_longest_road
                                ),
                            "dev_vp":
                                dev_vp_count(
                                    player
                                ),
                        }
                    )

    fieldnames = list(
        rows[0].keys()
    )

    with RAW_CSV.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    # ------------------------------------------------
    # Strategy-level summary.
    # ------------------------------------------------

    strategy_rows = []

    for strategy in STRATEGIES:
        name = strategy_name(
            strategy
        )

        subset = [
            row
            for row in rows
            if row["strategy"] == name
        ]

        wins = sum(
            row["winner"]
            for row in subset
        )

        games = len(
            subset
        )

        finished_games = sum(
            row["game_finished"]
            for row in subset
        )

        strategy_rows.append(
            {
                "strategy":
                    name,
                "games":
                    games,
                "wins":
                    wins,
                "win_rate":
                    wins / games,
                "avg_vp":
                    mean(
                        row["victory_points"]
                        for row in subset
                    ),
                "avg_roads":
                    mean(
                        row["roads"]
                        for row in subset
                    ),
                "avg_settlements":
                    mean(
                        row["settlements"]
                        for row in subset
                    ),
                "avg_cities":
                    mean(
                        row["cities"]
                        for row in subset
                    ),
                "avg_knights":
                    mean(
                        row["knights"]
                        for row in subset
                    ),
                "largest_army_rate":
                    mean(
                        row["largest_army"]
                        for row in subset
                    ),
                "longest_road_rate":
                    mean(
                        row["longest_road"]
                        for row in subset
                    ),
                "avg_dev_vp":
                    mean(
                        row["dev_vp"]
                        for row in subset
                    ),
                "finished_game_rate":
                    finished_games / games,
            }
        )

    with SUMMARY_CSV.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                strategy_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            strategy_rows
        )

    # ------------------------------------------------
    # Strategy x seat summary.
    # ------------------------------------------------

    seat_rows = []

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

            games = len(
                subset
            )

            wins = sum(
                row["winner"]
                for row in subset
            )

            seat_rows.append(
                {
                    "strategy":
                        name,
                    "seat":
                        seat,
                    "games":
                        games,
                    "wins":
                        wins,
                    "win_rate":
                        wins / games,
                    "avg_vp":
                        mean(
                            row[
                                "victory_points"
                            ]
                            for row in subset
                        ),
                }
            )

    with SEAT_CSV.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                seat_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            seat_rows
        )

    # ------------------------------------------------
    # Console summary.
    # ------------------------------------------------

    unique_games = (
        NUM_SEED_BLOCKS
        * len(lineups)
        * 4
    )

    unfinished = sum(
        1
        for seed_block in range(
            NUM_SEED_BLOCKS
        )
        for lineup_index in range(
            len(lineups)
        )
        for rotation in range(4)
        if not any(
            row["winner"]
            for row in rows
            if (
                row["seed_block"]
                == seed_block
                and row["lineup_index"]
                == lineup_index
                and row["rotation"]
                == rotation
            )
        )
    )

    # The calculation above treats a finished game
    # with winner in seat 1 correctly because winner
    # is stored as an integer flag on that player's
    # row. At least one row is 1 for every finished
    # game.

    game_turns = {}

    for row in rows:
        key = (
            row["seed_block"],
            row["lineup_index"],
            row["rotation"],
        )

        game_turns[
            key
        ] = row["turns"]

    print()
    print("=" * 96)
    print(
        "EXHAUSTIVE STRATEGY DIAGNOSTIC"
    )
    print("=" * 96)

    print(
        f"Strategies:       "
        f"{len(STRATEGIES)}"
    )
    print(
        f"Lineups:          "
        f"{len(lineups)}"
    )
    print(
        f"Seed blocks:      "
        f"{NUM_SEED_BLOCKS}"
    )
    print(
        f"Total games:      "
        f"{unique_games}"
    )
    print(
        f"Unfinished games: "
        f"{unfinished}"
    )
    print(
        f"Average turns:    "
        f"{mean(game_turns.values()):.1f}"
    )

    print()

    header = (
        f"{'Strategy':20} "
        f"{'Games':>6} "
        f"{'Wins':>6} "
        f"{'Win%':>7} "
        f"{'VP':>6} "
        f"{'Road':>6} "
        f"{'Sett':>6} "
        f"{'City':>6} "
        f"{'Knight':>7} "
        f"{'LA%':>7} "
        f"{'LR%':>7} "
        f"{'DevVP':>7}"
    )

    print(header)
    print("-" * len(header))

    for row in strategy_rows:
        print(
            f"{row['strategy']:20} "
            f"{row['games']:6d} "
            f"{row['wins']:6d} "
            f"{100 * row['win_rate']:6.1f}% "
            f"{row['avg_vp']:6.2f} "
            f"{row['avg_roads']:6.2f} "
            f"{row['avg_settlements']:6.2f} "
            f"{row['avg_cities']:6.2f} "
            f"{row['avg_knights']:7.2f} "
            f"{100 * row['largest_army_rate']:6.1f}% "
            f"{100 * row['longest_road_rate']:6.1f}% "
            f"{row['avg_dev_vp']:7.2f}"
        )

    print()
    print("Seat-specific win rates:")

    for strategy in STRATEGIES:
        name = strategy_name(
            strategy
        )

        subset = [
            row
            for row in seat_rows
            if row["strategy"] == name
        ]

        seat_text = " | ".join(
            (
                f"P{row['seat']}: "
                f"{row['wins']}/{row['games']} "
                f"({100 * row['win_rate']:.1f}%)"
            )
            for row in subset
        )

        print(
            f"{name:20} {seat_text}"
        )

    print()
    print(
        f"Raw games:      {RAW_CSV}"
    )
    print(
        f"Strategy stats: {SUMMARY_CSV}"
    )
    print(
        f"Seat stats:     {SEAT_CSV}"
    )


if __name__ == "__main__":
    main()
