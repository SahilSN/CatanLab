from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from catanlab.game import run_game
from catanlab.rl_realism_v2_checkpoint import (
    load_realism_v2_checkpoint,
)
from catanlab.rl_realism_v2_dagger import (
    RealismV2DAggerAgent,
)
from catanlab.rl_teacher import (
    TeacherDecisionKind,
    TeacherV2Example,
)
from catanlab.rl_teacher_dataset import (
    save_teacher_v2_jsonl,
)
from catanlab.strategies import StrategyType
from catanlab.turns import AdaptiveStrategyAgent


DAGGER_PROTOCOL = "realism-v2-dagger-ordinary-v1"

TARGET_STRATEGY = (
    StrategyType.FIVE_RESOURCE
)

OPPONENT_STRATEGIES = (
    StrategyType.HYBRID_OWS,
    StrategyType.FULL_OWS,
    StrategyType.PORT,
)


def build_game_agents(
    *,
    model,
    target_seat,
    seed,
):
    target = RealismV2DAggerAgent(
        TARGET_STRATEGY,
        model,
        deterministic=True,
        seed=seed + 10_000_000,
    )

    strategies = []
    agents = []

    opponent_index = 0

    for seat in range(4):
        if seat == target_seat:
            strategies.append(
                TARGET_STRATEGY
            )
            agents.append(
                target
            )
            continue

        strategy = (
            OPPONENT_STRATEGIES[
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

    return (
        strategies,
        agents,
        target,
    )


def dagger_to_teacher_v2(
    example,
):
    return TeacherV2Example(
        decision_kind=(
            TeacherDecisionKind
            .ORDINARY_ACTION
        ),
        observation=(
            example.observation
        ),
        player_id=(
            example.player_id
        ),
        label=(
            example.action_id
        ),
        legal_mask=(
            example.legal_mask
        ),
        candidate_features=None,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate ordinary-action Search-v2 "
            "DAgger labels on realism-v2 learner "
            "trajectories."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        required=True,
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

    args = parser.parse_args()

    if args.seed_start < 0:
        raise ValueError(
            "--seed-start cannot be negative."
        )

    if args.seed_count <= 0:
        raise ValueError(
            "--seed-count must be positive."
        )

    if args.max_turns <= 0:
        raise ValueError(
            "--max-turns must be positive."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_path = (
        args.output_dir
        / "train.jsonl"
    )

    metadata_path = (
        args.output_dir
        / "metadata.json"
    )

    if dataset_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite "
            f"{dataset_path}"
        )

    if metadata_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite "
            f"{metadata_path}"
        )

    model = load_realism_v2_checkpoint(
        args.model,
        device="cpu",
    )

    examples = []

    games = []

    total_agreements = 0
    total_disagreements = 0

    decision_counts = Counter()

    games_with_winner = 0

    for game_index in range(
        args.seed_count
    ):
        seed = (
            args.seed_start
            + game_index
        )

        target_seat = (
            game_index % 4
        )

        (
            strategies,
            agents,
            target,
        ) = build_game_agents(
            model=model,
            target_seat=target_seat,
            seed=seed,
        )

        result = run_game(
            strategies=strategies,
            seed=seed,
            max_turns=args.max_turns,
            validate_conservation=True,
            turn_agents=agents,
        )

        if result.winner_id is not None:
            games_with_winner += 1

        if target.illegal_decisions:
            raise RuntimeError(
                "Learner made illegal special "
                f"decisions at seed {seed}."
            )

        if target.fallbacks:
            raise RuntimeError(
                "Learner used fallback at "
                f"seed {seed}."
            )

        ordinary_count = (
            target.decision_counts[
                "ordinary_action"
            ]
        )

        if (
            len(target.dagger_examples)
            != ordinary_count
        ):
            raise RuntimeError(
                "DAgger example count mismatch "
                f"at seed {seed}: "
                f"examples="
                f"{len(target.dagger_examples)}, "
                f"ordinary={ordinary_count}"
            )

        game_examples = [
            dagger_to_teacher_v2(
                example
            )
            for example in (
                target.dagger_examples
            )
        ]

        examples.extend(
            game_examples
        )

        agreements = (
            target.dagger_agreements
        )

        disagreements = (
            target.dagger_disagreements
        )

        total_agreements += (
            agreements
        )

        total_disagreements += (
            disagreements
        )

        decision_counts.update(
            target.decision_counts
        )

        games.append(
            {
                "game_index": game_index,
                "seed": seed,
                "target_seat": (
                    target_seat
                ),
                "winner_id": (
                    result.winner_id
                ),
                "target_vp": (
                    result.players[
                        target_seat
                    ].victory_points
                ),
                "turns_played": (
                    result.turns_played
                ),
                "ordinary_examples": (
                    len(game_examples)
                ),
                "agreements": (
                    agreements
                ),
                "disagreements": (
                    disagreements
                ),
                "agreement_rate": (
                    target
                    .dagger_agreement_rate
                ),
            }
        )

        print(
            f"[{game_index + 1}/"
            f"{args.seed_count}] "
            f"seed={seed} "
            f"seat={target_seat} "
            f"winner="
            f"{result.winner_id} "
            f"vp="
            f"{result.players[target_seat].victory_points} "
            f"dagger="
            f"{len(game_examples)} "
            f"agree={agreements} "
            f"disagree={disagreements} "
            f"rate="
            f"{target.dagger_agreement_rate:.3f}",
            flush=True,
        )

    if not examples:
        raise RuntimeError(
            "DAgger generation produced no "
            "ordinary examples."
        )

    save_teacher_v2_jsonl(
        dataset_path,
        examples,
    )

    total_examples = len(
        examples
    )

    agreement_rate = (
        total_agreements
        / total_examples
    )

    observation_dims = sorted(
        {
            len(example.observation)
            for example in examples
        }
    )

    action_dims = sorted(
        {
            len(example.legal_mask)
            for example in examples
        }
    )

    metadata = {
        "generation_protocol": (
            DAGGER_PROTOCOL
        ),
        "source_checkpoint": str(
            args.model
        ),
        "decision_kind": (
            TeacherDecisionKind
            .ORDINARY_ACTION
            .value
        ),
        "seed_start": (
            args.seed_start
        ),
        "seed_count": (
            args.seed_count
        ),
        "max_turns": (
            args.max_turns
        ),
        "games_completed": (
            args.seed_count
        ),
        "games_with_winner": (
            games_with_winner
        ),
        "examples_total": (
            total_examples
        ),
        "teacher_learner_agreements": (
            total_agreements
        ),
        "teacher_learner_disagreements": (
            total_disagreements
        ),
        "teacher_learner_agreement_rate": (
            agreement_rate
        ),
        "observation_dims": (
            observation_dims
        ),
        "action_dims": (
            action_dims
        ),
        "counts_by_decision_kind": {
            TeacherDecisionKind
            .ORDINARY_ACTION
            .value: total_examples
        },
        "learned_decision_counts": {
            key: decision_counts[key]
            for key in sorted(
                decision_counts
            )
        },
        "teacher": {
            "search_depth": 2,
            "use_transposition_cache": False,
            "search_maritime_trades": True,
            "search_year_of_plenty": True,
            "search_road_building": True,
            "search_monopoly": True,
            "search_robber_decisions": True,
            "search_discard_decisions": True,
            "search_domestic_trades": True,
        },
        "games": games,
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print(
        "=== REALISM-V2 DAGGER DATASET ==="
    )

    print(
        f"games: "
        f"{args.seed_count}"
    )

    print(
        f"games with winner: "
        f"{games_with_winner}"
    )

    print(
        f"ordinary examples: "
        f"{total_examples}"
    )

    print(
        "teacher/learner agreement: "
        f"{agreement_rate:.4f}"
    )

    print(
        f"observation dims: "
        f"{observation_dims}"
    )

    print(
        f"action dims: "
        f"{action_dims}"
    )

    print(
        f"dataset: {dataset_path}"
    )

    print(
        f"metadata: {metadata_path}"
    )


if __name__ == "__main__":
    main()
