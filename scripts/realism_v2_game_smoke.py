from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from catanlab.game import run_game
from catanlab.rl_realism_v2_agent import (
    RealismV2PolicyAgent,
)
from catanlab.rl_realism_v2_checkpoint import (
    load_realism_v2_checkpoint,
)
from catanlab.strategies import (
    StrategyType,
)
from catanlab.turns import (
    AdaptiveStrategyAgent,
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
        default=1_300_000,
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

    opponent_strategies = (
        StrategyType.HYBRID_OWS,
        StrategyType.FULL_OWS,
        StrategyType.PORT,
    )

    decision_totals = Counter()

    wins = 0
    vp_total = 0.0
    turn_total = 0
    illegal_total = 0
    fallback_total = 0

    for game_index in range(
        args.games
    ):
        target_seat = (
            game_index % 4
        )

        target_strategy = (
            StrategyType.FIVE_RESOURCE
        )

        strategies = []
        agents = []

        opponent_index = 0
        target_agent = None

        for seat in range(4):
            if seat == target_seat:
                target_agent = (
                    RealismV2PolicyAgent(
                        target_strategy,
                        model=model,
                        deterministic=True,
                        seed=(
                            args.seed_offset
                            + game_index
                            + 1_000_000
                        ),
                    )
                )

                strategies.append(
                    target_strategy
                )

                agents.append(
                    target_agent
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
            seed=(
                args.seed_offset
                + game_index
            ),
            max_turns=2000,
            validate_conservation=True,
            turn_agents=agents,
        )

        if target_agent is None:
            raise RuntimeError(
                "Target agent was not created."
            )

        target_vp = (
            result.players[
                target_seat
            ].victory_points
        )

        won = (
            result.winner_id
            == target_seat
        )

        wins += int(won)
        vp_total += target_vp
        turn_total += result.turns_played

        illegal_total += (
            target_agent.illegal_decisions
        )

        fallback_total += (
            target_agent.fallbacks
        )

        decision_totals.update(
            target_agent.decision_counts
        )

        print(
            f"[{game_index + 1}/"
            f"{args.games}] "
            f"seed="
            f"{args.seed_offset + game_index} "
            f"seat={target_seat} "
            f"winner={result.winner_id} "
            f"vp={target_vp} "
            f"turns={result.turns_played}",
            flush=True,
        )

        observed = {
            key: value
            for key, value
            in target_agent
            .decision_counts
            .items()
            if value
        }

        print(
            f"  decisions={observed}",
            flush=True,
        )

    print()
    print(
        "=== REALISM-V2 GAME SMOKE ==="
    )
    print(
        f"games: {args.games}"
    )
    print(
        f"wins: {wins}"
    )
    print(
        f"win rate: "
        f"{wins / args.games:.3f}"
    )
    print(
        f"mean VP: "
        f"{vp_total / args.games:.3f}"
    )
    print(
        f"mean turns: "
        f"{turn_total / args.games:.1f}"
    )

    print()
    print("decision counts:")

    for kind in sorted(
        decision_totals
    ):
        print(
            f"  {kind:20s} "
            f"{decision_totals[kind]}"
        )

    print()
    print(
        f"illegal decisions: "
        f"{illegal_total}"
    )

    print(
        f"fallbacks: "
        f"{fallback_total}"
    )

    if illegal_total != 0:
        raise RuntimeError(
            "Realism-v2 smoke encountered "
            "illegal learned decisions."
        )

    if fallback_total != 0:
        raise RuntimeError(
            "Realism-v2 smoke encountered "
            "heuristic fallbacks."
        )


if __name__ == "__main__":
    main()
