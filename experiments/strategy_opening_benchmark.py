from collections import defaultdict
from itertools import combinations, permutations

from catanlab.board import build_random_board
from catanlab.scoring import (
    resource_production,
    score_opening_pair,
)
from catanlab.simulation import (
    StrategyOpeningAgent,
    run_opening_draft,
)
from catanlab.strategies import (
    StrategyType,
)


NUM_BOARDS = 100


def main() -> None:
    strategies = list(
        StrategyType
    )

    stats = defaultdict(
        lambda: {
            "count": 0,
            "production": 0.0,
            "diversity": 0.0,
            "composite": 0.0,
            "wood": 0.0,
            "brick": 0.0,
            "sheep": 0.0,
            "wheat": 0.0,
            "ore": 0.0,
        }
    )

    lineup_index = 0

    total_orders = 360
    completed_orders = 0

    for lineup in combinations(
        strategies,
        4,
    ):
        for order in permutations(
            lineup
        ):
            for board_seed in range(
                NUM_BOARDS
            ):
                board = build_random_board(
                    seed=(
                        board_seed
                        + lineup_index * 10000
                    )
                )

                agents = [
                    StrategyOpeningAgent(
                        strategy
                    )
                    for strategy in order
                ]

                result = run_opening_draft(
                    board,
                    agents,
                )

                for seat, player in enumerate(
                    result.players
                ):
                    strategy = order[
                        seat
                    ]

                    a, b = player.settlements

                    pair = score_opening_pair(
                        board,
                        board.vertices[a],
                        board.vertices[b],
                    )

                    prod_a = resource_production(
                        board,
                        board.vertices[a],
                    )

                    prod_b = resource_production(
                        board,
                        board.vertices[b],
                    )

                    combined = {
                        resource: (
                            prod_a[resource]
                            + prod_b[resource]
                        )
                        for resource in prod_a
                    }

                    key = (
                        strategy.value,
                        seat,
                    )

                    row = stats[
                        key
                    ]

                    row["count"] += 1

                    row["production"] += (
                        pair.production_score
                    )

                    row["diversity"] += (
                        pair.unique_resources
                    )

                    row["composite"] += (
                        pair.composite_score
                    )

                    for resource, amount in (
                        combined.items()
                    ):
                        row[
                            resource.value
                        ] += amount

            lineup_index += 1

    print()
    print(
        "Strategy          "
        "Seat  "
        "Prod   "
        "Div    "
        "Wood   "
        "Brick  "
        "Sheep  "
        "Wheat  "
        "Ore"
    )

    print("-" * 78)

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
            f"{strategy:<18}"
            f"{seat + 1:<6}"
            f"{row['production'] / count:<7.2f}"
            f"{row['diversity'] / count:<7.2f}"
            f"{row['wood'] / count:<7.2f}"
            f"{row['brick'] / count:<7.2f}"
            f"{row['sheep'] / count:<7.2f}"
            f"{row['wheat'] / count:<7.2f}"
            f"{row['ore'] / count:<7.2f}"
        )


if __name__ == "__main__":
    main()
