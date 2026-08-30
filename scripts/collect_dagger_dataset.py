from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from catanlab.action_space import (
    RLActionType,
    build_action_vocabulary,
)
from catanlab.board import build_random_board
from catanlab.game import run_game
from catanlab.rl_agent import (
    NeuralPolicyAgent,
)
from catanlab.rl_model import (
    CatanActorCritic,
    FactorizedCatanActorCritic,
)
from catanlab.rl_teacher import (
    DAggerAgent,
)
from catanlab.strategies import StrategyType
from catanlab.turns import (
    AdaptiveStrategyAgent,
)


def load_model(path: Path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    model_class = checkpoint.get(
        "model_class"
    )

    if model_class is None:
        if "type_head.weight" in state_dict:
            model_class = "factorized"
        else:
            model_class = "flat"

    if model_class == "factorized":
        model = FactorizedCatanActorCritic(
            observation_dim=checkpoint[
                "observation_dim"
            ],
            action_dim=checkpoint[
                "action_dim"
            ],
            hidden_dim=checkpoint[
                "hidden_dim"
            ],
        )
    elif model_class == "flat":
        model = CatanActorCritic(
            observation_dim=checkpoint[
                "observation_dim"
            ],
            action_dim=checkpoint[
                "action_dim"
            ],
            hidden_dim=checkpoint[
                "hidden_dim"
            ],
        )
    else:
        raise ValueError(
            "Unknown model_class: "
            f"{model_class!r}"
        )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--games",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--seed-offset",
        type=int,
        default=100000,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    model = load_model(
        args.model
    )

    all_examples = []
    all_game_ids = []

    wins = 0

    for game_index in range(
        args.games
    ):
        target_seat = (
            game_index % 4
        )

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

        dagger_agent = None
        opponent_index = 0

        for seat in range(4):
            if seat == target_seat:
                strategies.append(
                    target_strategy
                )

                dagger_agent = DAggerAgent(
                    target_strategy,
                    model=model,
                    deterministic=True,
                    seed=(
                        args.seed_offset
                        + game_index
                    ),
                )

                agents.append(
                    dagger_agent
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

        game_seed = (
            args.seed_offset
            + game_index
        )

        result = run_game(
            strategies=strategies,
            seed=game_seed,
            max_turns=2000,
            validate_conservation=True,
            turn_agents=agents,
        )

        if (
            result.winner_id
            == target_seat
        ):
            wins += 1

        all_examples.extend(
            dagger_agent.examples
        )

        all_game_ids.extend(
            [game_index]
            * len(
                dagger_agent.examples
            )
        )

        agreements = sum(
            example.agrees
            for example in (
                dagger_agent.examples
            )
        )

        count = len(
            dagger_agent.examples
        )

        agreement_rate = (
            agreements / count
            if count
            else 0.0
        )

        print(
            f"[{game_index + 1}/{args.games}] "
            f"seat={target_seat} "
            f"winner={result.winner_id} "
            f"turns={result.turns_played} "
            f"examples={count} "
            f"agreement={agreement_rate:.3f} "
            f"total={len(all_examples)}",
            flush=True,
        )

    if not all_examples:
        raise RuntimeError(
            "No DAgger examples collected."
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

    # Search-teacher labels.
    action_ids = np.asarray(
        [
            example.action_id
            for example in all_examples
        ],
        dtype=np.int64,
    )

    learner_action_ids = np.asarray(
        [
            example.learner_action_id
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

    rows = np.arange(
        len(action_ids)
    )

    if observations.shape[1] != 1138:
        raise RuntimeError(
            "Unexpected observation shape: "
            f"{observations.shape}"
        )

    if legal_masks.shape[1] != 202:
        raise RuntimeError(
            "Unexpected mask shape: "
            f"{legal_masks.shape}"
        )

    if not np.all(
        legal_masks[
            rows,
            action_ids,
        ]
    ):
        raise RuntimeError(
            "Illegal search-teacher labels found."
        )

    if not np.all(
        legal_masks[
            rows,
            learner_action_ids,
        ]
    ):
        raise RuntimeError(
            "Illegal learner actions found."
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
        learner_action_ids=(
            learner_action_ids
        ),
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

    teacher_types = Counter(
        vocabulary[
            int(action_id)
        ].action_type
        for action_id in action_ids
    )

    learner_types = Counter(
        vocabulary[
            int(action_id)
        ].action_type
        for action_id in learner_action_ids
    )

    agreements = int(
        (
            action_ids
            == learner_action_ids
        ).sum()
    )

    print()
    print("=== DAGGER DATASET SUMMARY ===")
    print(
        f"games: {args.games}"
    )
    print(
        f"learner wins: {wins}"
    )
    print(
        f"learner win rate: "
        f"{wins / args.games:.3f}"
    )
    print(
        f"examples: "
        f"{len(action_ids)}"
    )
    print(
        f"teacher/learner exact agreement: "
        f"{agreements / len(action_ids):.3f}"
    )
    print(
        f"output: {args.output}"
    )

    print()
    print(
        "teacher vs learner action types:"
    )

    for action_type in RLActionType:
        teacher_count = teacher_types[
            action_type
        ]

        learner_count = learner_types[
            action_type
        ]

        print(
            f"{action_type.value:20s} "
            f"teacher={teacher_count:6d} "
            f"{teacher_count / len(action_ids):8.3%} "
            f"learner={learner_count:6d} "
            f"{learner_count / len(action_ids):8.3%}"
        )


if __name__ == "__main__":
    main()
