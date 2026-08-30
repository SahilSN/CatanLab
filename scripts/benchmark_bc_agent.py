from __future__ import annotations

import argparse
from pathlib import Path

import torch

from catanlab.game import run_game
from catanlab.rl_agent import NeuralPolicyAgent
from catanlab.rl_model import (
    CatanActorCritic,
    FactorizedCatanActorCritic,
)
from catanlab.strategies import StrategyType
from catanlab.turns import AdaptiveStrategyAgent


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
            "Unknown checkpoint model_class: "
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
        default=100,
    )

    parser.add_argument(
        "--stochastic",
        action="store_true",
    )

    args = parser.parse_args()

    model = load_model(
        args.model
    )

    wins = 0
    completed = 0
    total_actions = 0
    total_turns = 0

    vp_total = 0.0

    seat_games = [0] * 4
    seat_wins = [0] * 4

    opponent_strategies = [
        StrategyType.HYBRID_OWS,
        StrategyType.FULL_OWS,
        StrategyType.PORT,
    ]

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

        target_agent = None
        opponent_index = 0

        for seat in range(4):
            if seat == target_seat:
                strategies.append(
                    target_strategy
                )

                target_agent = NeuralPolicyAgent(
                    target_strategy,
                    model=model,
                    deterministic=(
                        not args.stochastic
                    ),
                    seed=(
                        100_000
                        + game_index
                    ),
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
            seed=game_index,
            max_turns=2000,
            validate_conservation=True,
            turn_agents=agents,
        )

        completed += 1

        seat_games[
            target_seat
        ] += 1

        total_actions += (
            target_agent.actions_selected
        )

        total_turns += (
            result.turns_played
        )

        # run_game's result should expose final players.
        # Adapt this line if your result field has a
        # slightly different name.
        target_vp = (
            result.players[
                target_seat
            ].victory_points
        )

        vp_total += target_vp

        if (
            result.winner_id
            == target_seat
        ):
            wins += 1

            seat_wins[
                target_seat
            ] += 1

        print(
            f"[{completed}/{args.games}] "
            f"seat={target_seat} "
            f"winner={result.winner_id} "
            f"vp={target_vp} "
            f"turns={result.turns_played} "
            f"nn_actions="
            f"{target_agent.actions_selected}",
            flush=True,
        )

    print()
    print("=== BC POLICY BENCHMARK ===")
    print(
        f"games: {completed}"
    )
    print(
        f"wins: {wins}"
    )
    print(
        f"win rate: "
        f"{wins / completed:.3f}"
    )
    print(
        f"mean VP: "
        f"{vp_total / completed:.3f}"
    )
    print(
        f"mean turns: "
        f"{total_turns / completed:.1f}"
    )
    print(
        f"ordinary actions: "
        f"{total_actions}"
    )

    print()
    print("seat results:")

    for seat in range(4):
        rate = (
            seat_wins[seat]
            / seat_games[seat]
            if seat_games[seat]
            else 0.0
        )

        print(
            f"seat {seat}: "
            f"{seat_wins[seat]}/"
            f"{seat_games[seat]} "
            f"({rate:.3f})"
        )


if __name__ == "__main__":
    main()
