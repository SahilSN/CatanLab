from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from catanlab.game import run_game
from catanlab.rl_realism_v2_checkpoint import (
    load_realism_v2_checkpoint,
)
from catanlab.rl_realism_v2_dagger import (
    RealismV2DAggerAgent,
)
from catanlab.strategies import StrategyType
from catanlab.turns import AdaptiveStrategyAgent


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
        default=5,
    )

    parser.add_argument(
        "--seed-offset",
        type=int,
        default=1_410_000,
    )

    args = parser.parse_args()

    if args.games <= 0:
        raise ValueError(
            "--games must be positive."
        )

    model = load_realism_v2_checkpoint(
        args.model,
        device="cpu",
    )

    total_examples = 0
    total_agreements = 0
    total_disagreements = 0

    decision_counts = Counter()

    for game_index in range(
        args.games
    ):
        seed = (
            args.seed_offset
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
            max_turns=2000,
            validate_conservation=True,
            turn_agents=agents,
        )

        examples = len(
            target.dagger_examples
        )

        ordinary_decisions = (
            target.decision_counts[
                "ordinary_action"
            ]
        )

        if (
            examples
            != ordinary_decisions
        ):
            raise RuntimeError(
                "DAgger example count does not "
                "match learner ordinary decisions: "
                f"examples={examples}, "
                f"ordinary={ordinary_decisions}"
            )

        if target.illegal_decisions:
            raise RuntimeError(
                "Learned policy made illegal "
                "special decisions."
            )

        if target.fallbacks:
            raise RuntimeError(
                "Learned policy used fallbacks."
            )

        agreements = (
            target.dagger_agreements
        )

        disagreements = (
            target.dagger_disagreements
        )

        total_examples += examples

        total_agreements += (
            agreements
        )

        total_disagreements += (
            disagreements
        )

        decision_counts.update(
            target.decision_counts
        )

        print(
            f"[{game_index + 1}/"
            f"{args.games}] "
            f"seed={seed} "
            f"seat={target_seat} "
            f"winner="
            f"{result.winner_id} "
            f"vp="
            f"{result.players[target_seat].victory_points} "
            f"turns="
            f"{result.turns_played} "
            f"dagger={examples} "
            f"agree={agreements} "
            f"disagree={disagreements} "
            f"rate="
            f"{target.dagger_agreement_rate:.3f}",
            flush=True,
        )

    if total_examples == 0:
        raise RuntimeError(
            "DAgger smoke produced no examples."
        )

    agreement_rate = (
        total_agreements
        / total_examples
    )

    print()
    print(
        "=== REALISM-V2 DAGGER SMOKE ==="
    )

    print(
        f"games: {args.games}"
    )

    print(
        f"ordinary examples: "
        f"{total_examples}"
    )

    print(
        f"teacher/learner agreements: "
        f"{total_agreements}"
    )

    print(
        f"teacher/learner disagreements: "
        f"{total_disagreements}"
    )

    print(
        f"agreement rate: "
        f"{agreement_rate:.4f}"
    )

    print()
    print(
        "learned decision counts:"
    )

    for key in sorted(
        decision_counts
    ):
        print(
            f"  {key:20s} "
            f"{decision_counts[key]}"
        )

    print()
    print(
        "illegal decisions: 0"
    )

    print(
        "fallbacks: 0"
    )


if __name__ == "__main__":
    main()
