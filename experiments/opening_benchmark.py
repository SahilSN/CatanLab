from collections import defaultdict
from itertools import permutations

from catanlab.board import build_random_board
from catanlab.scoring import score_opening_pair
from catanlab.simulation import (
    BalancedAgent,
    ProductionAgent,
    RandomAgent,
    run_opening_draft,
)


NUM_BOARDS = 250


def make_agent(
    name: str,
    seed: int,
):
    if name == "balanced":
        return BalancedAgent()

    if name == "production":
        return ProductionAgent()

    if name == "random":
        return RandomAgent(
            seed=seed
        )

    raise ValueError(
        f"Unknown strategy: {name}"
    )


def main() -> None:
    strategies = [
        "balanced",
        "production",
        "random",
        "balanced",
    ]

    # Unique seat assignments for the four
    # strategy instances.
    seat_orders = sorted(
        set(
            permutations(
                strategies
            )
        )
    )

    stats = defaultdict(
        lambda: {
            "count": 0,
            "production": 0.0,
            "diversity": 0.0,
            "composite": 0.0,
            "all_five": 0,
        }
    )

    for order_index, order in enumerate(
        seat_orders
    ):
        for board_seed in range(
            NUM_BOARDS
        ):
            board = build_random_board(
                seed=board_seed
            )

            agents = [
                make_agent(
                    strategy,
                    seed=(
                        board_seed * 1000
                        + order_index * 10
                        + seat
                    ),
                )
                for seat, strategy
                in enumerate(order)
            ]

            result = run_opening_draft(
                board,
                agents,
            )

            for seat, player in enumerate(
                result.players
            ):
                a, b = player.settlements

                score = score_opening_pair(
                    board,
                    board.vertices[a],
                    board.vertices[b],
                )

                strategy = order[
                    seat
                ]

                key = (
                    strategy,
                    seat,
                )

                row = stats[
                    key
                ]

                row["count"] += 1

                row["production"] += (
                    score.production_score
                )

                row["diversity"] += (
                    score.unique_resources
                )

                row["composite"] += (
                    score.composite_score
                )

                if (
                    score.unique_resources
                    == 5
                ):
                    row["all_five"] += 1

    print()
    print(
        "Strategy      "
        "Seat   "
        "Prod    "
        "Div     "
        "Score    "
        "All5%"
    )

    print("-" * 58)

    for (
        strategy,
        seat,
    ), row in sorted(
        stats.items()
    ):
        count = row[
            "count"
        ]

        print(
            f"{strategy:<13}"
            f"{seat + 1:<7}"
            f"{row['production'] / count:<8.2f}"
            f"{row['diversity'] / count:<8.2f}"
            f"{row['composite'] / count:<9.2f}"
            f"{100 * row['all_five'] / count:>5.1f}"
        )


if __name__ == "__main__":
    main()
