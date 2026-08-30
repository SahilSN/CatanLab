from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np


TRAIN_KEYS = (
    "observations",
    "legal_masks",
    "action_ids",
    "player_ids",
    "game_ids",
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--dagger-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    base = np.load(
        args.base_dir / "train.npz"
    )

    dagger = np.load(
        args.dagger_dir
        / "train.npz"
    )

    dagger_val = (
        args.dagger_dir
        / "val.npz"
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_game_ids = base[
        "game_ids"
    ]

    dagger_game_ids = dagger[
        "game_ids"
    ]

    # Shift DAgger game IDs so they do not collide
    # with the original training-game IDs.
    dagger_offset = (
        int(base_game_ids.max())
        + 1
    )

    shifted_dagger_game_ids = (
        dagger_game_ids
        + dagger_offset
    )

    merged = {}

    for key in TRAIN_KEYS:
        if key == "game_ids":
            dagger_values = (
                shifted_dagger_game_ids
            )
        else:
            dagger_values = dagger[
                key
            ]

        merged[key] = np.concatenate(
            [
                base[key],
                dagger_values,
            ],
            axis=0,
        )

    output_train = (
        args.output_dir
        / "train.npz"
    )

    np.savez_compressed(
        output_train,
        **merged,
    )

    # Keep validation and test exactly unchanged.
    shutil.copy2(
        args.base_dir / "val.npz",
        args.output_dir / "val.npz",
    )

    shutil.copy2(
        args.base_dir / "test.npz",
        args.output_dir / "test.npz",
    )

    shutil.copy2(
        dagger_val,
        args.output_dir
        / "dagger_val.npz",
    )

    print(
        "=== DAGGER TRAIN MERGE ==="
    )
    print(
        f"base training rows: "
        f"{len(base['action_ids'])}"
    )
    print(
        f"DAgger rows: "
        f"{len(dagger['action_ids'])}"
    )
    print(
        f"merged training rows: "
        f"{len(merged['action_ids'])}"
    )
    print(
        f"validation unchanged: "
        f"{len(np.load(args.output_dir / 'val.npz')['action_ids'])}"
    )
    print(
        f"test unchanged: "
        f"{len(np.load(args.output_dir / 'test.npz')['action_ids'])}"
    )
    print(
        f"output: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
