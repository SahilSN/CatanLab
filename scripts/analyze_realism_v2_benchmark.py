from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


POLICIES = (
    "learned",
    "teacher",
    "heuristic",
)


def wilson_interval(
    successes,
    total,
    *,
    z=1.959963984540054,
):
    if total <= 0:
        raise ValueError(
            "Wilson interval requires total > 0."
        )

    p = successes / total
    z2 = z * z

    denominator = (
        1.0
        + z2 / total
    )

    center = (
        p
        + z2 / (2.0 * total)
    ) / denominator

    radius = (
        z
        * math.sqrt(
            (
                p * (1.0 - p) / total
                + z2 / (4.0 * total * total)
            )
        )
        / denominator
    )

    return (
        center - radius,
        center + radius,
    )


def mean_t_interval(
    values,
    *,
    z=1.959963984540054,
):
    """
    Normal-approximation 95% interval for a sample mean.

    At n=100 this is appropriate for the descriptive
    benchmark summary and requires no scipy dependency.
    """
    values = list(values)

    if not values:
        raise ValueError(
            "Cannot compute interval of empty sample."
        )

    mean = statistics.mean(
        values
    )

    if len(values) == 1:
        return (
            mean,
            mean,
        )

    stdev = statistics.stdev(
        values
    )

    radius = (
        z
        * stdev
        / math.sqrt(
            len(values)
        )
    )

    return (
        mean - radius,
        mean + radius,
    )


def percentile(
    sorted_values,
    q,
):
    if not sorted_values:
        raise ValueError(
            "Cannot compute percentile "
            "of empty sample."
        )

    if not (
        0.0
        <= q
        <= 1.0
    ):
        raise ValueError(
            "q must be in [0, 1]."
        )

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (
        q
        * (
            len(sorted_values)
            - 1
        )
    )

    lower = int(
        math.floor(position)
    )
    upper = int(
        math.ceil(position)
    )

    if lower == upper:
        return sorted_values[
            lower
        ]

    fraction = (
        position
        - lower
    )

    return (
        sorted_values[lower]
        * (1.0 - fraction)
        + sorted_values[upper]
        * fraction
    )


def paired_bootstrap_ci(
    differences,
    *,
    iterations=20_000,
    seed=0,
):
    """
    Paired percentile bootstrap CI for the mean
    difference.

    `differences` must already contain one value per
    paired game, e.g. learned VP - teacher VP.
    """
    differences = tuple(
        differences
    )

    if not differences:
        raise ValueError(
            "No paired differences."
        )

    rng = random.Random(
        seed
    )

    n = len(
        differences
    )

    bootstrap_means = []

    for _ in range(
        iterations
    ):
        sample_total = 0.0

        for _ in range(n):
            sample_total += (
                differences[
                    rng.randrange(n)
                ]
            )

        bootstrap_means.append(
            sample_total / n
        )

    bootstrap_means.sort()

    return (
        percentile(
            bootstrap_means,
            0.025,
        ),
        percentile(
            bootstrap_means,
            0.975,
        ),
    )


def load_records(
    path,
):
    records = []

    with path.open(
        newline="",
    ) as handle:
        reader = (
            csv.DictReader(
                handle
            )
        )

        required = {
            "policy",
            "game_index",
            "seed",
            "target_seat",
            "winner_id",
            "target_won",
            "target_vp",
            "turns",
            "illegal_decisions",
            "fallbacks",
        }

        missing = (
            required
            - set(
                reader.fieldnames
                or ()
            )
        )

        if missing:
            raise ValueError(
                "Benchmark CSV missing columns: "
                f"{sorted(missing)}"
            )

        for raw in reader:
            winner_raw = raw[
                "winner_id"
            ]

            winner_id = (
                None
                if winner_raw
                in ("", "None")
                else int(
                    winner_raw
                )
            )

            records.append(
                {
                    "policy": raw[
                        "policy"
                    ],
                    "game_index": int(
                        raw[
                            "game_index"
                        ]
                    ),
                    "seed": int(
                        raw["seed"]
                    ),
                    "target_seat": int(
                        raw[
                            "target_seat"
                        ]
                    ),
                    "winner_id": (
                        winner_id
                    ),
                    "target_won": int(
                        raw[
                            "target_won"
                        ]
                    ),
                    "target_vp": float(
                        raw[
                            "target_vp"
                        ]
                    ),
                    "turns": int(
                        raw[
                            "turns"
                        ]
                    ),
                    "illegal_decisions": int(
                        raw[
                            "illegal_decisions"
                        ]
                    ),
                    "fallbacks": int(
                        raw[
                            "fallbacks"
                        ]
                    ),
                }
            )

    return records


def validate_pairing(
    records,
):
    by_game = defaultdict(
        dict
    )

    for record in records:
        game_index = record[
            "game_index"
        ]
        policy = record[
            "policy"
        ]

        if policy in by_game[
            game_index
        ]:
            raise ValueError(
                "Duplicate policy/game pair: "
                f"game={game_index}, "
                f"policy={policy}"
            )

        by_game[
            game_index
        ][policy] = record

    expected = set(
        POLICIES
    )

    for (
        game_index,
        group,
    ) in by_game.items():
        actual = set(
            group
        )

        if actual != expected:
            raise ValueError(
                "Incomplete policy pairing: "
                f"game={game_index}, "
                f"policies="
                f"{sorted(actual)}"
            )

        seeds = {
            record["seed"]
            for record
            in group.values()
        }

        seats = {
            record[
                "target_seat"
            ]
            for record
            in group.values()
        }

        if len(seeds) != 1:
            raise ValueError(
                "Paired policies do not share seed: "
                f"game={game_index}, "
                f"seeds={sorted(seeds)}"
            )

        if len(seats) != 1:
            raise ValueError(
                "Paired policies do not share "
                "target seat: "
                f"game={game_index}, "
                f"seats={sorted(seats)}"
            )

    return by_game


def policy_summary(
    records,
):
    output = {}

    for policy in POLICIES:
        subset = [
            record
            for record
            in records
            if record[
                "policy"
            ] == policy
        ]

        if not subset:
            raise ValueError(
                f"No records for {policy}."
            )

        wins = sum(
            record[
                "target_won"
            ]
            for record
            in subset
        )

        vp = [
            record[
                "target_vp"
            ]
            for record
            in subset
        ]

        turns = [
            record[
                "turns"
            ]
            for record
            in subset
        ]

        win_ci = (
            wilson_interval(
                wins,
                len(subset),
            )
        )

        vp_ci = (
            mean_t_interval(
                vp
            )
        )

        seat_summary = {}

        for seat in range(4):
            seat_subset = [
                record
                for record
                in subset
                if record[
                    "target_seat"
                ] == seat
            ]

            seat_wins = sum(
                record[
                    "target_won"
                ]
                for record
                in seat_subset
            )

            seat_summary[
                str(seat)
            ] = {
                "games": len(
                    seat_subset
                ),
                "wins": (
                    seat_wins
                ),
                "win_rate": (
                    seat_wins
                    / len(
                        seat_subset
                    )
                ),
                "mean_vp": (
                    statistics.mean(
                        record[
                            "target_vp"
                        ]
                        for record
                        in seat_subset
                    )
                ),
            }

        output[
            policy
        ] = {
            "games": len(
                subset
            ),
            "wins": wins,
            "win_rate": (
                wins
                / len(subset)
            ),
            "win_rate_ci95": (
                list(win_ci)
            ),
            "mean_vp": (
                statistics.mean(
                    vp
                )
            ),
            "mean_vp_ci95": (
                list(vp_ci)
            ),
            "mean_turns": (
                statistics.mean(
                    turns
                )
            ),
            "games_with_winner": (
                sum(
                    record[
                        "winner_id"
                    ]
                    is not None
                    for record
                    in subset
                )
            ),
            "illegal_decisions": (
                sum(
                    record[
                        "illegal_decisions"
                    ]
                    for record
                    in subset
                )
            ),
            "fallbacks": (
                sum(
                    record[
                        "fallbacks"
                    ]
                    for record
                    in subset
                )
            ),
            "by_seat": (
                seat_summary
            ),
        }

    return output


def paired_summary(
    by_game,
    *,
    bootstrap_iterations,
    bootstrap_seed,
):
    learned = "learned"

    output = {}

    for comparator in (
        "teacher",
        "heuristic",
    ):
        vp_differences = []
        win_differences = []

        higher = 0
        tied = 0
        lower = 0

        for game_index in sorted(
            by_game
        ):
            learned_record = (
                by_game[
                    game_index
                ][learned]
            )

            other_record = (
                by_game[
                    game_index
                ][comparator]
            )

            vp_delta = (
                learned_record[
                    "target_vp"
                ]
                - other_record[
                    "target_vp"
                ]
            )

            win_delta = (
                learned_record[
                    "target_won"
                ]
                - other_record[
                    "target_won"
                ]
            )

            vp_differences.append(
                vp_delta
            )

            win_differences.append(
                win_delta
            )

            if vp_delta > 0:
                higher += 1
            elif vp_delta < 0:
                lower += 1
            else:
                tied += 1

        vp_ci = (
            paired_bootstrap_ci(
                vp_differences,
                iterations=(
                    bootstrap_iterations
                ),
                seed=(
                    bootstrap_seed
                ),
            )
        )

        win_ci = (
            paired_bootstrap_ci(
                win_differences,
                iterations=(
                    bootstrap_iterations
                ),
                seed=(
                    bootstrap_seed
                    + 1
                ),
            )
        )

        output[
            f"learned_minus_{comparator}"
        ] = {
            "paired_games": len(
                vp_differences
            ),
            "mean_vp_delta": (
                statistics.mean(
                    vp_differences
                )
            ),
            "mean_vp_delta_ci95": (
                list(vp_ci)
            ),
            "mean_win_delta": (
                statistics.mean(
                    win_differences
                )
            ),
            "mean_win_delta_ci95": (
                list(win_ci)
            ),
            "vp_higher": higher,
            "vp_tied": tied,
            "vp_lower": lower,
        }

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--games-csv",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=20_000,
    )

    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    if (
        args.bootstrap_iterations
        <= 0
    ):
        raise ValueError(
            "--bootstrap-iterations "
            "must be positive."
        )

    records = load_records(
        args.games_csv
    )

    by_game = validate_pairing(
        records
    )

    policies = policy_summary(
        records
    )

    paired = paired_summary(
        by_game,
        bootstrap_iterations=(
            args.bootstrap_iterations
        ),
        bootstrap_seed=(
            args.bootstrap_seed
        ),
    )

    payload = {
        "games_csv": str(
            args.games_csv
        ),
        "paired_games": len(
            by_game
        ),
        "bootstrap_iterations": (
            args.bootstrap_iterations
        ),
        "bootstrap_seed": (
            args.bootstrap_seed
        ),
        "policies": policies,
        "paired": paired,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print(
        "=== REALISM-V2 STATISTICAL ANALYSIS ==="
    )
    print()

    print(
        f"{'policy':10s} "
        f"{'win rate':>22s} "
        f"{'mean VP':>24s} "
        f"{'mean turns':>12s}"
    )

    print("-" * 74)

    for policy in POLICIES:
        stats = policies[
            policy
        ]

        win_lo, win_hi = (
            stats[
                "win_rate_ci95"
            ]
        )

        vp_lo, vp_hi = (
            stats[
                "mean_vp_ci95"
            ]
        )

        print(
            f"{policy:10s} "
            f"{stats['win_rate']:.3f} "
            f"[{win_lo:.3f}, "
            f"{win_hi:.3f}]   "
            f"{stats['mean_vp']:.3f} "
            f"[{vp_lo:.3f}, "
            f"{vp_hi:.3f}]   "
            f"{stats['mean_turns']:8.1f}"
        )

    print()
    print(
        "Paired learned comparisons:"
    )

    for (
        name,
        stats,
    ) in paired.items():
        vp_lo, vp_hi = (
            stats[
                "mean_vp_delta_ci95"
            ]
        )

        win_lo, win_hi = (
            stats[
                "mean_win_delta_ci95"
            ]
        )

        print()
        print(name)

        print(
            "  mean VP delta: "
            f"{stats['mean_vp_delta']:+.3f} "
            f"[{vp_lo:+.3f}, "
            f"{vp_hi:+.3f}]"
        )

        print(
            "  mean win delta: "
            f"{stats['mean_win_delta']:+.3f} "
            f"[{win_lo:+.3f}, "
            f"{win_hi:+.3f}]"
        )

        print(
            "  VP W/T/L: "
            f"{stats['vp_higher']}/"
            f"{stats['vp_tied']}/"
            f"{stats['vp_lower']}"
        )

    print()
    print(
        "Seat breakdown:"
    )

    for policy in POLICIES:
        print(
            f"  {policy}:"
        )

        for seat in range(4):
            stats = (
                policies[
                    policy
                ][
                    "by_seat"
                ][
                    str(seat)
                ]
            )

            print(
                f"    seat {seat}: "
                f"n={stats['games']}, "
                f"win="
                f"{stats['win_rate']:.3f}, "
                f"VP="
                f"{stats['mean_vp']:.3f}"
            )

    print()
    print(
        f"analysis: {args.output}"
    )


if __name__ == "__main__":
    main()
