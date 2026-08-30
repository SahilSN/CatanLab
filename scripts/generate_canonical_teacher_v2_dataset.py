from __future__ import annotations

import argparse
from pathlib import Path

from catanlab.rl_teacher_generation import (
    canonical_teacher_v2_config,
    generate_teacher_v2_dataset,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a frozen canonical realism-v2 "
            "teacher dataset split."
        )
    )

    parser.add_argument(
        "split",
        choices=(
            "train",
            "validation",
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    config = canonical_teacher_v2_config(
        args.split
    )

    result = generate_teacher_v2_dataset(
        args.output_dir,
        config,
    )

    metadata = result["metadata"]

    print(
        "Generated canonical realism-v2 "
        "teacher dataset"
    )

    print(
        f"  protocol: "
        f"{metadata['generation_protocol']}"
    )

    print(
        f"  split: "
        f"{metadata['split']}"
    )

    print(
        f"  seeds: "
        f"{metadata['seed_start']}.."
        f"{metadata['seeds'][-1]}"
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

    print("  counts:")

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
