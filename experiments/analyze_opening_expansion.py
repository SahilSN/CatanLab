import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

from catanlab.board import build_random_board
from catanlab.scoring import score_vertex


INPUT = Path(
    "results/opening_seat_diagnostic/openings.csv"
)

OUTPUT = Path(
    "results/opening_seat_diagnostic/"
    "expansion_summary.csv"
)


def parse_edge(
    value: str,
) -> tuple[int, int]:
    a, b = value.split("-")

    return (
        int(a),
        int(b),
    )


def blocked_by_opening(
    board,
    occupied: set[int],
) -> set[int]:
    blocked = set(
        occupied
    )

    for vertex_id in occupied:
        blocked.update(
            board.vertices[
                vertex_id
            ].neighbors
        )

    return blocked


def road_frontier(
    road: tuple[int, int],
    own_settlements: set[int],
) -> int:
    """
    Return the road endpoint pointing away from the
    settlement that originally anchored the road.
    """

    a, b = road

    if (
        a in own_settlements
        and b not in own_settlements
    ):
        return b

    if (
        b in own_settlements
        and a not in own_settlements
    ):
        return a

    # In the unlikely ambiguous case, preserve a
    # deterministic result.
    return max(
        a,
        b,
    )


def reachable_candidates(
    board,
    frontier: int,
    blocked: set[int],
) -> list[int]:
    """
    Find potential settlement vertices reachable by
    extending one additional road from an opening
    road's frontier.

    This is an expansion-potential heuristic rather
    than a complete future-game search.
    """

    candidates = []

    for neighbor in (
        board.vertices[
            frontier
        ].neighbors
    ):
        if neighbor in blocked:
            continue

        candidates.append(
            neighbor
        )

    return candidates


def main():
    with INPUT.open(
        newline="",
    ) as handle:
        rows = list(
            csv.DictReader(
                handle
            )
        )

    enriched = []

    for row in rows:
        board = build_random_board(
            seed=int(
                row["board_seed"]
            )
        )

        first_vertex = int(
            row["first_vertex"]
        )
        second_vertex = int(
            row["second_vertex"]
        )

        own_settlements = {
            first_vertex,
            second_vertex,
        }

        first_road = parse_edge(
            row["first_road"]
        )

        second_road = parse_edge(
            row["second_road"]
        )

        # Reconstruct all eight opening settlements
        # on this board/strategy/seed from the raw
        # diagnostic rows.
        same_game_rows = [
            other
            for other in rows
            if (
                other["strategy"]
                == row["strategy"]
                and other["seed"]
                == row["seed"]
            )
        ]

        occupied = set()

        for other in same_game_rows:
            occupied.add(
                int(
                    other[
                        "first_vertex"
                    ]
                )
            )
            occupied.add(
                int(
                    other[
                        "second_vertex"
                    ]
                )
            )

        blocked = blocked_by_opening(
            board,
            occupied,
        )

        frontiers = [
            road_frontier(
                first_road,
                own_settlements,
            ),
            road_frontier(
                second_road,
                own_settlements,
            ),
        ]

        candidates = set()

        for frontier in frontiers:
            candidates.update(
                reachable_candidates(
                    board,
                    frontier,
                    blocked,
                )
            )

        candidate_scores = [
            score_vertex(
                board,
                board.vertices[
                    vertex_id
                ],
            ).composite_score
            for vertex_id in candidates
        ]

        best_score = (
            max(candidate_scores)
            if candidate_scores
            else 0.0
        )

        avg_score = (
            mean(candidate_scores)
            if candidate_scores
            else 0.0
        )

        enriched.append(
            {
                **row,
                "expansion_sites":
                    len(candidates),
                "best_expansion_score":
                    best_score,
                "avg_expansion_score":
                    avg_score,
            }
        )

    grouped = defaultdict(
        list
    )

    for row in enriched:
        key = (
            row["strategy"],
            int(row["seat"]),
        )

        grouped[
            key
        ].append(
            row
        )

    summary = []

    for (
        strategy,
        seat,
    ), subset in sorted(
        grouped.items()
    ):
        winners = [
            row
            for row in subset
            if int(
                row["winner"]
            )
        ]

        losers = [
            row
            for row in subset
            if not int(
                row["winner"]
            )
        ]

        summary.append(
            {
                "strategy":
                    strategy,
                "seat":
                    seat,
                "games":
                    len(subset),
                "win_rate":
                    mean(
                        int(
                            row["winner"]
                        )
                        for row in subset
                    ),
                "avg_expansion_sites":
                    mean(
                        row[
                            "expansion_sites"
                        ]
                        for row in subset
                    ),
                "avg_best_expansion_score":
                    mean(
                        row[
                            "best_expansion_score"
                        ]
                        for row in subset
                    ),
                "avg_expansion_score":
                    mean(
                        row[
                            "avg_expansion_score"
                        ]
                        for row in subset
                    ),
                "winner_best_expansion":
                    (
                        mean(
                            row[
                                "best_expansion_score"
                            ]
                            for row in winners
                        )
                        if winners
                        else 0.0
                    ),
                "loser_best_expansion":
                    (
                        mean(
                            row[
                                "best_expansion_score"
                            ]
                            for row in losers
                        )
                        if losers
                        else 0.0
                    ),
            }
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                summary[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            summary
        )

    print()
    print("=" * 105)
    print(
        "OPENING EXPANSION POTENTIAL BY STRATEGY / SEAT"
    )
    print("=" * 105)

    header = (
        f"{'Strategy':20} "
        f"{'Seat':>4} "
        f"{'Win%':>7} "
        f"{'Sites':>7} "
        f"{'Best':>7} "
        f"{'Avg':>7} "
        f"{'WinBest':>8} "
        f"{'LoseBest':>9}"
    )

    print(header)
    print(
        "-" * len(header)
    )

    for row in summary:
        print(
            f"{row['strategy']:20} "
            f"P{row['seat']:<3} "
            f"{100 * row['win_rate']:6.1f}% "
            f"{row['avg_expansion_sites']:7.2f} "
            f"{row['avg_best_expansion_score']:7.2f} "
            f"{row['avg_expansion_score']:7.2f} "
            f"{row['winner_best_expansion']:8.2f} "
            f"{row['loser_best_expansion']:9.2f}"
        )

    print()
    print(
        f"Saved summary to {OUTPUT}"
    )


if __name__ == "__main__":
    main()
