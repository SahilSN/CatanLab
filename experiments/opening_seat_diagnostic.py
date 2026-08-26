import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from catanlab.game import run_game
from catanlab.ports import ports_at_vertex
from catanlab.resources import Resource
from catanlab.scoring import (
    port_synergy_score,
    resource_production,
    score_opening_pair,
    score_vertex,
    strategic_pair_score,
)
from catanlab.strategies import (
    STRATEGY_PROFILES,
    StrategyType,
)


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
    "results/opening_seat_diagnostic"
)

RAW_CSV = RESULTS_DIR / "openings.csv"
SUMMARY_CSV = RESULTS_DIR / "seat_summary.csv"


RESOURCES = (
    Resource.WOOD,
    Resource.BRICK,
    Resource.SHEEP,
    Resource.WHEAT,
    Resource.ORE,
)


def strategy_name(
    strategy: StrategyType,
) -> str:
    return strategy.value


def player_opening_vertices(
    placements,
    player_id: int,
) -> tuple[int, int]:
    vertices = [
        vertex_id
        for placed_player_id, vertex_id
        in placements
        if placed_player_id == player_id
    ]

    if len(vertices) != 2:
        raise RuntimeError(
            f"Expected exactly two opening "
            f"settlements for player {player_id}, "
            f"found {len(vertices)}."
        )

    return (
        vertices[0],
        vertices[1],
    )


def player_opening_roads(
    roads,
    player_id: int,
):
    player_roads = [
        edge
        for road_player_id, edge
        in roads
        if road_player_id == player_id
    ]

    if len(player_roads) != 2:
        raise RuntimeError(
            f"Expected exactly two opening roads "
            f"for player {player_id}, "
            f"found {len(player_roads)}."
        )

    return (
        player_roads[0],
        player_roads[1],
    )


def port_label(
    board,
    vertex_id: int,
) -> str:
    ports = ports_at_vertex(
        board,
        vertex_id,
    )

    if not ports:
        return "none"

    labels = []

    for port in ports:
        if port.resource is None:
            labels.append(
                "generic_3to1"
            )
        else:
            labels.append(
                f"{port.resource.value}_2to1"
            )

    return "|".join(
        sorted(
            set(labels)
        )
    )


def second_settlement_resources(
    board,
    vertex_id: int,
) -> Counter:
    """
    Reconstruct the starting resources granted
    from the second opening settlement.
    """

    resources = Counter()

    vertex = board.vertices[
        vertex_id
    ]

    for tile_id in (
        vertex.adjacent_tiles
    ):
        tile = board.tiles[
            tile_id
        ]

        if (
            tile.resource
            == Resource.DESERT
        ):
            continue

        resources[
            tile.resource
        ] += 1

    return resources


def opening_metrics(
    board,
    strategy,
    first_vertex_id,
    second_vertex_id,
):
    first_vertex = board.vertices[
        first_vertex_id
    ]

    second_vertex = board.vertices[
        second_vertex_id
    ]

    first_score = score_vertex(
        board,
        first_vertex,
    )

    second_score = score_vertex(
        board,
        second_vertex,
    )

    pair = score_opening_pair(
        board,
        first_vertex,
        second_vertex,
    )

    profile = STRATEGY_PROFILES[
        strategy
    ]

    strategic_score = (
        strategic_pair_score(
            board,
            first_vertex,
            second_vertex,
            profile.resource_weights,
            profile.diversity_weight,
        )
    )

    synergy = port_synergy_score(
        board,
        [
            first_vertex,
            second_vertex,
        ],
    )

    # Port specialization explicitly uses the
    # port/resource synergy term in its opening
    # evaluation.
    if strategy == StrategyType.PORT:
        effective_strategy_score = (
            strategic_score
            + synergy
        )
    else:
        effective_strategy_score = (
            strategic_score
        )

    first_prod = resource_production(
        board,
        first_vertex,
    )

    second_prod = resource_production(
        board,
        second_vertex,
    )

    combined_prod = {
        resource: (
            first_prod[resource]
            + second_prod[resource]
        )
        for resource in RESOURCES
    }

    return {
        "first_production":
            first_score.production_score,
        "second_production":
            second_score.production_score,
        "combined_production":
            pair.production_score,
        "combined_diversity":
            pair.unique_resources,
        "standard_pair_score":
            pair.composite_score,
        "strategic_pair_score":
            strategic_score,
        "port_synergy":
            synergy,
        "effective_strategy_score":
            effective_strategy_score,
        "production_wood":
            combined_prod[
                Resource.WOOD
            ],
        "production_brick":
            combined_prod[
                Resource.BRICK
            ],
        "production_sheep":
            combined_prod[
                Resource.SHEEP
            ],
        "production_wheat":
            combined_prod[
                Resource.WHEAT
            ],
        "production_ore":
            combined_prod[
                Resource.ORE
            ],
    }


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
            f"Opening control: {name}"
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

            # Reconstruct the exact board used by
            # run_game() from the preserved internal
            # board seed.
            from catanlab.board import (
                build_random_board,
            )

            if result.board_seed is None:
                raise RuntimeError(
                    "GameResult is missing board_seed."
                )

            board = build_random_board(
                seed=result.board_seed
            )

            for seat in range(4):
                (
                    first_vertex_id,
                    second_vertex_id,
                ) = player_opening_vertices(
                    result.opening_placements,
                    seat,
                )

                (
                    first_road,
                    second_road,
                ) = player_opening_roads(
                    result.opening_roads,
                    seat,
                )

                metrics = opening_metrics(
                    board,
                    strategy,
                    first_vertex_id,
                    second_vertex_id,
                )

                starting = (
                    second_settlement_resources(
                        board,
                        second_vertex_id,
                    )
                )

                row = {
                    "strategy":
                        name,
                    "seed":
                        seed,
                    "board_seed":
                        result.board_seed,
                    "dev_seed":
                        result.dev_seed,
                    "seat":
                        seat + 1,
                    "winner":
                        int(
                            result.winner_id
                            == seat
                        ),
                    "turns":
                        result.turns_played,
                    "first_vertex":
                        first_vertex_id,
                    "second_vertex":
                        second_vertex_id,
                    "first_road":
                        (
                            f"{first_road[0]}-"
                            f"{first_road[1]}"
                        ),
                    "second_road":
                        (
                            f"{second_road[0]}-"
                            f"{second_road[1]}"
                        ),
                    "first_port":
                        port_label(
                            board,
                            first_vertex_id,
                        ),
                    "second_port":
                        port_label(
                            board,
                            second_vertex_id,
                        ),
                    "has_any_port":
                        int(
                            port_label(
                                board,
                                first_vertex_id,
                            ) != "none"
                            or port_label(
                                board,
                                second_vertex_id,
                            ) != "none"
                        ),
                    "start_wood":
                        starting[
                            Resource.WOOD
                        ],
                    "start_brick":
                        starting[
                            Resource.BRICK
                        ],
                    "start_sheep":
                        starting[
                            Resource.SHEEP
                        ],
                    "start_wheat":
                        starting[
                            Resource.WHEAT
                        ],
                    "start_ore":
                        starting[
                            Resource.ORE
                        ],
                    "start_cards":
                        sum(
                            starting.values()
                        ),
                }

                row.update(
                    metrics
                )

                rows.append(
                    row
                )

            if (
                (seed + 1) % 20
                == 0
            ):
                print(
                    f"  {seed + 1:3d}/"
                    f"{NUM_SEEDS} seeds"
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

    # --------------------------------------------
    # Aggregate strategy x seat opening quality.
    # --------------------------------------------

    summary_rows = []

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

            summary_rows.append(
                {
                    "strategy":
                        name,
                    "seat":
                        seat,
                    "games":
                        len(subset),
                    "wins":
                        sum(
                            row["winner"]
                            for row in subset
                        ),
                    "win_rate":
                        mean(
                            row["winner"]
                            for row in subset
                        ),
                    "combined_production":
                        mean(
                            row[
                                "combined_production"
                            ]
                            for row in subset
                        ),
                    "combined_diversity":
                        mean(
                            row[
                                "combined_diversity"
                            ]
                            for row in subset
                        ),
                    "strategic_score":
                        mean(
                            row[
                                "effective_strategy_score"
                            ]
                            for row in subset
                        ),
                    "port_rate":
                        mean(
                            row["has_any_port"]
                            for row in subset
                        ),
                    "start_cards":
                        mean(
                            row["start_cards"]
                            for row in subset
                        ),
                    "wood":
                        mean(
                            row["production_wood"]
                            for row in subset
                        ),
                    "brick":
                        mean(
                            row["production_brick"]
                            for row in subset
                        ),
                    "sheep":
                        mean(
                            row["production_sheep"]
                            for row in subset
                        ),
                    "wheat":
                        mean(
                            row["production_wheat"]
                            for row in subset
                        ),
                    "ore":
                        mean(
                            row["production_ore"]
                            for row in subset
                        ),
                }
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
    print("=" * 116)
    print(
        "OPENING QUALITY BY STRATEGY AND SEAT"
    )
    print("=" * 116)

    header = (
        f"{'Strategy':20} "
        f"{'Seat':>4} "
        f"{'Win%':>7} "
        f"{'Prod':>7} "
        f"{'Div':>6} "
        f"{'Strat':>8} "
        f"{'Port%':>7} "
        f"{'Start':>7} "
        f"{'Wood':>6} "
        f"{'Brick':>6} "
        f"{'Sheep':>6} "
        f"{'Wheat':>6} "
        f"{'Ore':>6}"
    )

    print(header)
    print("-" * len(header))

    for row in summary_rows:
        print(
            f"{row['strategy']:20} "
            f"P{row['seat']:<3d} "
            f"{100 * row['win_rate']:6.1f}% "
            f"{row['combined_production']:7.2f} "
            f"{row['combined_diversity']:6.2f} "
            f"{row['strategic_score']:8.2f} "
            f"{100 * row['port_rate']:6.1f}% "
            f"{row['start_cards']:7.2f} "
            f"{row['wood']:6.2f} "
            f"{row['brick']:6.2f} "
            f"{row['sheep']:6.2f} "
            f"{row['wheat']:6.2f} "
            f"{row['ore']:6.2f}"
        )

    print()
    print(
        f"Raw openings: {RAW_CSV}"
    )
    print(
        f"Seat summary: {SUMMARY_CSV}"
    )


if __name__ == "__main__":
    main()
