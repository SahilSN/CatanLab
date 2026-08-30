from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from catanlab.action_space import (
    build_action_vocabulary,
)
from catanlab.board import build_random_board


def summarize_split(
    name,
    path,
    vocabulary,
):
    data = np.load(path)

    action_ids = data["action_ids"]

    counts = Counter(
        int(action_id)
        for action_id in action_ids
    )

    by_type = defaultdict(list)

    for action_id, action in enumerate(
        vocabulary
    ):
        by_type[
            action.action_type.value
        ].append(
            counts[action_id]
        )

    print()
    print(
        f"=== {name.upper()} ==="
    )

    print(
        f"examples: {len(action_ids)}"
    )

    print(
        f"action IDs observed: "
        f"{sum(count > 0 for count in counts.values())}"
        f"/{len(vocabulary)}"
    )

    print()
    print(
        "support by action type:"
    )

    for action_type, values in (
        by_type.items()
    ):
        array = np.asarray(
            values,
            dtype=np.int64,
        )

        nonzero = array[
            array > 0
        ]

        print()
        print(
            action_type
        )

        print(
            f"  actions:       "
            f"{len(array)}"
        )

        print(
            f"  observed:      "
            f"{len(nonzero)}"
        )

        print(
            f"  unseen:        "
            f"{int((array == 0).sum())}"
        )

        print(
            f"  total labels:  "
            f"{int(array.sum())}"
        )

        if len(nonzero):
            print(
                f"  min nonzero:   "
                f"{int(nonzero.min())}"
            )

            print(
                f"  median:        "
                f"{float(np.median(nonzero)):.1f}"
            )

            print(
                f"  mean nonzero:  "
                f"{float(nonzero.mean()):.1f}"
            )

            print(
                f"  max:           "
                f"{int(nonzero.max())}"
            )

            print(
                f"  <= 5 examples: "
                f"{int((nonzero <= 5).sum())}"
            )

            print(
                f"  <=10 examples: "
                f"{int((nonzero <= 10).sum())}"
            )

    print()
    print(
        "lowest-support observed action IDs:"
    )

    observed = [
        (
            counts[action_id],
            action_id,
            vocabulary[
                action_id
            ],
        )
        for action_id in range(
            len(vocabulary)
        )
        if counts[action_id] > 0
    ]

    observed.sort()

    for count, action_id, action in (
        observed[:30]
    ):
        print(
            f"id={action_id:3d} "
            f"count={count:3d} "
            f"{action}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    board = build_random_board(
        seed=0
    )

    vocabulary = build_action_vocabulary(
        len(board.vertices),
        board.edges,
    )

    for name in (
        "train",
        "val",
        "test",
    ):
        summarize_split(
            name,
            args.data_dir
            / f"{name}.npz",
            vocabulary,
        )


if __name__ == "__main__":
    main()
