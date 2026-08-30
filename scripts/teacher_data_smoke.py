from __future__ import annotations

import argparse
from collections import Counter

from catanlab.action_space import (
    RLActionType,
    build_action_vocabulary,
)
from catanlab.board import (
    build_random_board,
)
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
        default=20,
    )

    args = parser.parse_args()

    all_examples = []

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

        print(
            f"[{game_index + 1}/{args.games}] "
            f"seat={target_seat} "
            f"winner={result.winner_id} "
            f"turns={result.turns_played} "
            f"examples={len(teacher.examples)}",
            flush=True,
        )

    board = build_random_board(
        seed=0
    )

    vocabulary = build_action_vocabulary(
        len(board.vertices),
        board.edges,
    )

    type_counts = Counter(
        vocabulary[
            example.action_id
        ].action_type
        for example in all_examples
    )

    legal_counts = [
        sum(example.legal_mask)
        for example in all_examples
    ]

    invalid_labels = sum(
        not example.legal_mask[
            example.action_id
        ]
        for example in all_examples
    )

    print()
    print("=== TEACHER DATA SMOKE ===")
    print(
        f"games: {args.games}"
    )
    print(
        f"teacher wins: {wins}"
    )
    print(
        f"examples: {len(all_examples)}"
    )
    print(
        f"invalid labels: {invalid_labels}"
    )

    if legal_counts:
        print(
            "legal actions / decision: "
            f"min={min(legal_counts)} "
            f"mean="
            f"{sum(legal_counts) / len(legal_counts):.2f} "
            f"max={max(legal_counts)}"
        )

    print()
    print("teacher action types:")

    for action_type in RLActionType:
        count = type_counts[
            action_type
        ]

        fraction = (
            count / len(all_examples)
            if all_examples
            else 0.0
        )

        print(
            f"{action_type.value:20s} "
            f"{count:6d} "
            f"{fraction:8.3%}"
        )


if __name__ == "__main__":
    main()
