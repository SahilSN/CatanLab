from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


KEYS = (
    "observations",
    "legal_masks",
    "action_ids",
    "learner_action_ids",
    "player_ids",
    "game_ids",
)


def save_rows(
    source,
    mask,
    path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        path,
        **{
            key: source[key][mask]
            for key in KEYS
        },
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.90,
    )

    args = parser.parse_args()

    data = np.load(
        args.input
    )

    game_ids = data[
        "game_ids"
    ]

    unique_games = np.unique(
        game_ids
    )

    if len(unique_games) < 10:
        raise ValueError(
            "Need at least 10 DAgger games."
        )

    split_index = int(
        len(unique_games)
        * args.train_fraction
    )

    train_games = unique_games[
        :split_index
    ]

    val_games = unique_games[
        split_index:
    ]

    train_mask = np.isin(
        game_ids,
        train_games,
    )

    val_mask = np.isin(
        game_ids,
        val_games,
    )

    if np.any(
        train_mask & val_mask
    ):
        raise RuntimeError(
            "DAgger train/validation overlap."
        )

    if not np.all(
        train_mask | val_mask
    ):
        raise RuntimeError(
            "Unassigned DAgger examples."
        )

    save_rows(
        data,
        train_mask,
        args.output_dir
        / "train.npz",
    )

    save_rows(
        data,
        val_mask,
        args.output_dir
        / "val.npz",
    )

    print(
        "=== DAGGER SPLIT ==="
    )

    print(
        f"train: "
        f"{int(train_mask.sum())} examples, "
        f"{len(train_games)} games"
    )

    print(
        f"val:   "
        f"{int(val_mask.sum())} examples, "
        f"{len(val_games)} games"
    )


if __name__ == "__main__":
    main()
