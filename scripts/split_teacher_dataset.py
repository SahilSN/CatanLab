from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ARRAY_KEYS = (
    "observations",
    "legal_masks",
    "action_ids",
    "player_ids",
    "game_ids",
)


def save_split(
    source,
    mask,
    output,
):
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output,
        **{
            key: source[key][mask]
            for key in ARRAY_KEYS
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

    args = parser.parse_args()

    data = np.load(
        args.input
    )

    game_ids = data["game_ids"]

    unique_games = np.unique(
        game_ids
    )

    num_games = len(
        unique_games
    )

    if num_games < 10:
        raise ValueError(
            "Need at least 10 games for an "
            "80/10/10 split."
        )

    train_end = int(
        num_games * 0.80
    )

    val_end = int(
        num_games * 0.90
    )

    train_games = set(
        unique_games[
            :train_end
        ]
    )

    val_games = set(
        unique_games[
            train_end:val_end
        ]
    )

    test_games = set(
        unique_games[
            val_end:
        ]
    )

    train_mask = np.isin(
        game_ids,
        list(train_games),
    )

    val_mask = np.isin(
        game_ids,
        list(val_games),
    )

    test_mask = np.isin(
        game_ids,
        list(test_games),
    )

    if np.any(
        train_mask & val_mask
    ):
        raise RuntimeError(
            "Train/validation overlap."
        )

    if np.any(
        train_mask & test_mask
    ):
        raise RuntimeError(
            "Train/test overlap."
        )

    if np.any(
        val_mask & test_mask
    ):
        raise RuntimeError(
            "Validation/test overlap."
        )

    if not np.all(
        train_mask
        | val_mask
        | test_mask
    ):
        raise RuntimeError(
            "Some rows were not assigned."
        )

    save_split(
        data,
        train_mask,
        args.output_dir / "train.npz",
    )

    save_split(
        data,
        val_mask,
        args.output_dir / "val.npz",
    )

    save_split(
        data,
        test_mask,
        args.output_dir / "test.npz",
    )

    print("=== TEACHER DATA SPLIT ===")

    for name, mask in (
        ("train", train_mask),
        ("val", val_mask),
        ("test", test_mask),
    ):
        ids = np.unique(
            game_ids[mask]
        )

        print(
            f"{name:5s}: "
            f"{mask.sum():5d} examples, "
            f"{len(ids):3d} games"
        )


if __name__ == "__main__":
    main()
