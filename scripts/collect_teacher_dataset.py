from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from catanlab.action_space import (
    RLActionType,
    build_action_vocabulary,
)
from catanlab.board import build_random_board
from catanlab.game import run_game
from catanlab.rl_teacher import (
    RecordingSearchAgent,
)
from catanlab.strategies import StrategyType
from catanlab.turns import (
    AdaptiveStrategyAgent,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--games",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/rl_teacher/"
            "teacher_dataset_100.npz"
        ),
    )

    args = parser.parse_args()

    all_examples = []
    all_game_ids = []

    wins = 0

    for game_index in range(
        args.games
    ):
        target_seat = game_index % 4

        target_strategy = (
            StrategyType.FIVE_RESOURCE
        )

        opponent_strategies = [
            StrategyType.HYBRID_OWS,
            StrategyType.FULL_OWS,
            StrategyType.PORT,
        ]

        strategies = []
        agents = []

        teacher = None
        opponent_index = 0

        for seat in range(4):
            if seat == target_seat:
                strategies.append(
                    target_strategy
                )

                teacher = (
                    RecordingSearchAgent(
                        target_strategy
                    )
                )

                agents.append(
                    teacher
                )
            else:
                strategy = (
                    opponent_strategies[
                        opponent_index
                    ]
                )

                opponent_index += 1

                strategies.append(
                    strategy
                )

                agents.append(
                    AdaptiveStrategyAgent(
                        strategy
                    )
                )

        result = run_game(
            strategies=strategies,
            seed=game_index,
            max_turns=2000,
            validate_conservation=True,
            turn_agents=agents,
        )

        if result.winner_id == target_seat:
            wins += 1

        all_examples.extend(
            teacher.examples
        )

        all_game_ids.extend(
            [game_index]
            * len(teacher.examples)
        )

        print(
            f"[{game_index + 1}/{args.games}] "
            f"seat={target_seat} "
            f"winner={result.winner_id} "
            f"turns={result.turns_played} "
            f"examples={len(teacher.examples)} "
            f"total={len(all_examples)}",
            flush=True,
        )

    if not all_examples:
        raise RuntimeError(
            "No teacher examples collected."
        )

    observations = np.asarray(
        [
            example.observation
            for example in all_examples
        ],
        dtype=np.float32,
    )

    legal_masks = np.asarray(
        [
            example.legal_mask
            for example in all_examples
        ],
        dtype=np.bool_,
    )

    action_ids = np.asarray(
        [
            example.action_id
            for example in all_examples
        ],
        dtype=np.int64,
    )

    player_ids = np.asarray(
        [
            example.player_id
            for example in all_examples
        ],
        dtype=np.int8,
    )

    game_ids = np.asarray(
        all_game_ids,
        dtype=np.int32,
    )

    if observations.shape[1] != 1138:
        raise RuntimeError(
            "Unexpected observation dimension: "
            f"{observations.shape}"
        )

    if legal_masks.shape[1] != 202:
        raise RuntimeError(
            "Unexpected action-mask dimension: "
            f"{legal_masks.shape}"
        )

    row_indices = np.arange(
        len(action_ids)
    )

    if not np.all(
        legal_masks[
            row_indices,
            action_ids,
        ]
    ):
        raise RuntimeError(
            "Dataset contains illegal teacher labels."
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        args.output,
        observations=observations,
        legal_masks=legal_masks,
        action_ids=action_ids,
        player_ids=player_ids,
        game_ids=game_ids,
    )

    board = build_random_board(
        seed=0
    )

    vocabulary = build_action_vocabulary(
        len(board.vertices),
        board.edges,
    )

    type_counts = Counter(
        vocabulary[action_id].action_type
        for action_id in action_ids
    )

    print()
    print("=== DATASET SUMMARY ===")
    print(
        f"games: {args.games}"
    )
    print(
        f"teacher wins: {wins}"
    )
    print(
        f"teacher win rate: "
        f"{wins / args.games:.3f}"
    )
    print(
        f"examples: {len(action_ids)}"
    )
    print(
        f"observations shape: "
        f"{observations.shape}"
    )
    print(
        f"legal masks shape: "
        f"{legal_masks.shape}"
    )
    print(
        f"output: {args.output}"
    )

    print()
    print("action types:")

    for action_type in RLActionType:
        count = type_counts[
            action_type
        ]

        fraction = (
            count / len(action_ids)
        )

        print(
            f"{action_type.value:20s} "
            f"{count:6d} "
            f"{fraction:8.3%}"
        )


if __name__ == "__main__":
    main()
