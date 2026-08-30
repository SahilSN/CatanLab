from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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
            "Unknown model_class: "
            f"{model_class!r}"
        )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


def run_one(
    model,
    game_index,
    seed_offset,
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

    opponent_index = 0
    target_agent = None

    for seat in range(4):
        if seat == target_seat:
            strategies.append(
                target_strategy
            )

            target_agent = NeuralPolicyAgent(
                target_strategy,
                model=model,
                deterministic=True,
                seed=(
                    seed_offset
                    + game_index
                    + 1000000
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
        seed=(
            seed_offset
            + game_index
        ),
        max_turns=2000,
        validate_conservation=True,
        turn_agents=agents,
    )

    won = int(
        result.winner_id
        == target_seat
    )

    vp = float(
        result.players[
            target_seat
        ].victory_points
    )

    return won, vp


def bootstrap_ci(
    values,
    rng,
    repetitions=10000,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    n = len(values)

    samples = rng.integers(
        0,
        n,
        size=(
            repetitions,
            n,
        ),
    )

    means = values[
        samples
    ].mean(
        axis=1
    )

    return (
        float(
            np.quantile(
                means,
                0.025,
            )
        ),
        float(
            np.quantile(
                means,
                0.975,
            )
        ),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-a",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model-b",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--games",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    model_a = load_model(
        args.model_a
    )

    model_b = load_model(
        args.model_b
    )

    wins_a = []
    wins_b = []

    vp_a = []
    vp_b = []

    b_only_wins = 0
    a_only_wins = 0

    for game_index in range(
        args.games
    ):
        a_win, a_vp = run_one(
            model_a,
            game_index,
            args.seed_offset,
        )

        b_win, b_vp = run_one(
            model_b,
            game_index,
            args.seed_offset,
        )

        wins_a.append(
            a_win
        )

        wins_b.append(
            b_win
        )

        vp_a.append(
            a_vp
        )

        vp_b.append(
            b_vp
        )

        if a_win and not b_win:
            a_only_wins += 1

        if b_win and not a_win:
            b_only_wins += 1

        print(
            f"[{game_index + 1}/{args.games}] "
            f"A: win={a_win} vp={a_vp:.1f} | "
            f"B: win={b_win} vp={b_vp:.1f}",
            flush=True,
        )

    wins_a = np.asarray(
        wins_a,
        dtype=np.float64,
    )

    wins_b = np.asarray(
        wins_b,
        dtype=np.float64,
    )

    vp_a = np.asarray(
        vp_a,
        dtype=np.float64,
    )

    vp_b = np.asarray(
        vp_b,
        dtype=np.float64,
    )

    win_diff = (
        wins_b - wins_a
    )

    vp_diff = (
        vp_b - vp_a
    )

    rng = np.random.default_rng(
        args.bootstrap_seed
    )

    win_ci = bootstrap_ci(
        win_diff,
        rng,
    )

    vp_ci = bootstrap_ci(
        vp_diff,
        rng,
    )

    print()
    print(
        "=== PAIRED MODEL COMPARISON ==="
    )

    print(
        f"games: {args.games}"
    )

    print()
    print(
        f"A win rate: "
        f"{wins_a.mean():.4f}"
    )

    print(
        f"B win rate: "
        f"{wins_b.mean():.4f}"
    )

    print(
        f"B - A win-rate difference: "
        f"{win_diff.mean():+.4f}"
    )

    print(
        "95% paired bootstrap CI: "
        f"[{win_ci[0]:+.4f}, "
        f"{win_ci[1]:+.4f}]"
    )

    print()
    print(
        f"A mean VP: "
        f"{vp_a.mean():.4f}"
    )

    print(
        f"B mean VP: "
        f"{vp_b.mean():.4f}"
    )

    print(
        f"B - A mean VP: "
        f"{vp_diff.mean():+.4f}"
    )

    print(
        "95% paired bootstrap CI: "
        f"[{vp_ci[0]:+.4f}, "
        f"{vp_ci[1]:+.4f}]"
    )

    print()
    print(
        f"A-only wins: "
        f"{a_only_wins}"
    )

    print(
        f"B-only wins: "
        f"{b_only_wins}"
    )


if __name__ == "__main__":
    main()
