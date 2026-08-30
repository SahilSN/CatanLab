from __future__ import annotations

import argparse
from pathlib import Path

from catanlab.rl_teacher_generation import (
    TeacherV2GenerationConfig,
    generate_teacher_v2_dataset,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic realism-v2 "
            "Search-teacher supervision."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed-count",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--search-depth",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    config = TeacherV2GenerationConfig(
        seed_start=args.seed_start,
        seed_count=args.seed_count,
        max_turns=args.max_turns,
        search_depth=args.search_depth,
    )

    result = generate_teacher_v2_dataset(
        args.output_dir,
        config,
    )

    metadata = result[
        "metadata"
    ]

    print(
        "Generated realism-v2 teacher dataset"
    )

    print(
        f"  games: "
        f"{metadata['games_completed']}/"
        f"{metadata['games_attempted']}"
    )

    print(
        f"  games with winner: "
        f"{metadata['games_with_winner']}"
    )

    print(
        f"  examples: "
        f"{metadata['examples_total']}"
    )

    print(
        "  counts:"
    )

    for decision_kind, count in (
        metadata[
            "counts_by_decision_kind"
        ].items()
    ):
        print(
            f"    {decision_kind}: {count}"
        )

    print(
        f"  dataset: "
        f"{result['dataset_path']}"
    )

    print(
        f"  metadata: "
        f"{result['metadata_path']}"
    )


if __name__ == "__main__":
    main()
