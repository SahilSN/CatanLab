from __future__ import annotations

import argparse

from catanlab.game import run_game
from catanlab.rl_agent import (
    RandomMaskedAgent,
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

    completed = 0
    wins = 0
    total_actions = 0
    max_turns_seen = 0

    for game_index in range(
        args.games
    ):
        target_seat = (
            game_index % 4
        )

        game_seed = game_index

        strategies = [
            StrategyType.FIVE_RESOURCE,
            StrategyType.HYBRID_OWS,
            StrategyType.FULL_OWS,
            StrategyType.PORT,
        ]

        # Rotate FIVE_RESOURCE with the random agent.
        target_strategy = (
            StrategyType.FIVE_RESOURCE
        )

        opponent_strategies = [
            StrategyType.HYBRID_OWS,
            StrategyType.FULL_OWS,
            StrategyType.PORT,
        ]

        seat_strategies = []
        agents = []

        opponent_index = 0
        target_agent = None

        for seat in range(4):
            if seat == target_seat:
                seat_strategies.append(
                    target_strategy
                )

                target_agent = (
                    RandomMaskedAgent(
                        target_strategy,
                        seed=(
                            100_000
                            + game_index
                        ),
                    )
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

                seat_strategies.append(
                    strategy
                )

                agents.append(
                    AdaptiveStrategyAgent(
                        strategy
                    )
                )

        result = run_game(
            strategies=seat_strategies,
            seed=game_seed,
            max_turns=2000,
            validate_conservation=True,
            turn_agents=agents,
        )

        completed += 1

        if result.winner_id == target_seat:
            wins += 1

        total_actions += (
            target_agent.actions_selected
        )

        max_turns_seen = max(
            max_turns_seen,
            result.turns_played,
        )

        print(
            f"[{completed}/{args.games}] "
            f"seat={target_seat} "
            f"winner={result.winner_id} "
            f"turns={result.turns_played} "
            f"rl_actions="
            f"{target_agent.actions_selected}",
            flush=True,
        )

    print()
    print("=== RANDOM MASKED SMOKE ===")
    print(
        f"games: {completed}"
    )
    print(
        f"wins: {wins}"
    )
    print(
        f"total RL ordinary actions: "
        f"{total_actions}"
    )
    print(
        f"max turns seen: "
        f"{max_turns_seen}"
    )


if __name__ == "__main__":
    main()
