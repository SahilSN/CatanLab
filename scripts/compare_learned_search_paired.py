from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from catanlab.game import run_game
from catanlab.rl_agent import NeuralPolicyAgent
from catanlab.rl_model import (
    CatanActorCritic,
    FactorizedCatanActorCritic,
)
from catanlab.search_agent import OneStepLookaheadAgent
from catanlab.strategies import StrategyType
from catanlab.turns import AdaptiveStrategyAgent


def load_model(path: Path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    state_dict = checkpoint["model_state_dict"]
    model_class = checkpoint.get("model_class")

    if model_class is None:
        if "type_head.weight" in state_dict:
            model_class = "factorized"
        else:
            model_class = "flat"

    if model_class == "factorized":
        model = FactorizedCatanActorCritic(
            observation_dim=checkpoint["observation_dim"],
            action_dim=checkpoint["action_dim"],
            hidden_dim=checkpoint["hidden_dim"],
        )
    elif model_class == "flat":
        model = CatanActorCritic(
            observation_dim=checkpoint["observation_dim"],
            action_dim=checkpoint["action_dim"],
            hidden_dim=checkpoint["hidden_dim"],
        )
    else:
        raise ValueError(
            f"Unknown model_class: {model_class!r}"
        )

    model.load_state_dict(state_dict)
    model.eval()

    return model


def make_opponents(
    target_seat: int,
    target_agent,
):
    target_strategy = StrategyType.FIVE_RESOURCE

    opponent_strategies = [
        StrategyType.HYBRID_OWS,
        StrategyType.FULL_OWS,
        StrategyType.PORT,
    ]

    strategies = []
    agents = []

    opponent_index = 0

    for seat in range(4):
        if seat == target_seat:
            strategies.append(target_strategy)
            agents.append(target_agent)
        else:
            strategy = opponent_strategies[
                opponent_index
            ]
            opponent_index += 1

            strategies.append(strategy)
            agents.append(
                AdaptiveStrategyAgent(strategy)
            )

    return strategies, agents


def run_search_game(
    game_index: int,
    seed_offset: int,
):
    target_seat = game_index % 4
    target_strategy = StrategyType.FIVE_RESOURCE

    search_agent = OneStepLookaheadAgent(
        target_strategy,
        search_depth=2,
        use_transposition_cache=False,
        search_maritime_trades=True,
        search_year_of_plenty=False,
        search_road_building=True,
        search_monopoly=False,
    )

    strategies, agents = make_opponents(
        target_seat,
        search_agent,
    )

    start = time.perf_counter()

    result = run_game(
        strategies=strategies,
        seed=seed_offset + game_index,
        max_turns=2000,
        validate_conservation=True,
        turn_agents=agents,
    )

    elapsed = time.perf_counter() - start

    won = int(
        result.winner_id == target_seat
    )

    vp = float(
        result.players[
            target_seat
        ].victory_points
    )

    return won, vp, elapsed


def run_learned_game(
    model,
    game_index: int,
    seed_offset: int,
):
    target_seat = game_index % 4
    target_strategy = StrategyType.FIVE_RESOURCE

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

    strategies, agents = make_opponents(
        target_seat,
        target_agent,
    )

    start = time.perf_counter()

    result = run_game(
        strategies=strategies,
        seed=seed_offset + game_index,
        max_turns=2000,
        validate_conservation=True,
        turn_agents=agents,
    )

    elapsed = time.perf_counter() - start

    won = int(
        result.winner_id == target_seat
    )

    vp = float(
        result.players[
            target_seat
        ].victory_points
    )

    return won, vp, elapsed


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
        "--model",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--games",
        type=int,
        default=400,
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

    if args.games <= 0:
        raise ValueError(
            "--games must be positive"
        )

    model = load_model(args.model)

    search_wins = []
    learned_wins = []

    search_vp = []
    learned_vp = []

    search_times = []
    learned_times = []

    search_only_wins = 0
    learned_only_wins = 0

    for game_index in range(args.games):
        sw, svp, stime = run_search_game(
            game_index,
            args.seed_offset,
        )

        lw, lvp, ltime = run_learned_game(
            model,
            game_index,
            args.seed_offset,
        )

        search_wins.append(sw)
        learned_wins.append(lw)

        search_vp.append(svp)
        learned_vp.append(lvp)

        search_times.append(stime)
        learned_times.append(ltime)

        if sw and not lw:
            search_only_wins += 1

        if lw and not sw:
            learned_only_wins += 1

        completed = game_index + 1

        if (
            completed % 25 == 0
            or completed == args.games
        ):
            print(
                f"[{completed:>4}/{args.games}] "
                f"search_win="
                f"{np.mean(search_wins):.3f} "
                f"learned_win="
                f"{np.mean(learned_wins):.3f}",
                flush=True,
            )

    search_wins_array = np.asarray(
        search_wins,
        dtype=np.float64,
    )

    learned_wins_array = np.asarray(
        learned_wins,
        dtype=np.float64,
    )

    search_vp_array = np.asarray(
        search_vp,
        dtype=np.float64,
    )

    learned_vp_array = np.asarray(
        learned_vp,
        dtype=np.float64,
    )

    win_differences = (
        learned_wins_array
        - search_wins_array
    )

    vp_differences = (
        learned_vp_array
        - search_vp_array
    )

    rng = np.random.default_rng(
        args.bootstrap_seed
    )

    win_ci = bootstrap_ci(
        win_differences,
        rng,
    )

    vp_ci = bootstrap_ci(
        vp_differences,
        rng,
    )

    mean_search_time = float(
        np.mean(search_times)
    )

    mean_learned_time = float(
        np.mean(learned_times)
    )

    total_search_time = float(
        np.sum(search_times)
    )

    total_learned_time = float(
        np.sum(learned_times)
    )

    print()
    print(
        "=== PAIRED LEARNED VS SEARCH "
        "COMPARISON ==="
    )
    print(f"games: {args.games}")
    print()

    print(
        "A: depth-2 search "
        "(cache off, maritime on, "
        "road building on)"
    )
    print(
        f"B: learned model "
        f"({args.model})"
    )
    print()

    print(
        f"A win rate: "
        f"{search_wins_array.mean():.4f}"
    )

    print(
        f"B win rate: "
        f"{learned_wins_array.mean():.4f}"
    )

    print(
        "B - A win-rate difference: "
        f"{win_differences.mean():+.4f}"
    )

    print(
        "95% paired bootstrap CI: "
        f"[{win_ci[0]:+.4f}, "
        f"{win_ci[1]:+.4f}]"
    )

    print()

    print(
        f"A mean VP: "
        f"{search_vp_array.mean():.4f}"
    )

    print(
        f"B mean VP: "
        f"{learned_vp_array.mean():.4f}"
    )

    print(
        "B - A mean VP: "
        f"{vp_differences.mean():+.4f}"
    )

    print(
        "95% paired bootstrap CI: "
        f"[{vp_ci[0]:+.4f}, "
        f"{vp_ci[1]:+.4f}]"
    )

    print()

    print(
        f"A-only wins: "
        f"{search_only_wins}"
    )

    print(
        f"B-only wins: "
        f"{learned_only_wins}"
    )

    print()

    print(
        "=== FULL-GAME WALL CLOCK ==="
    )

    print(
        f"A total seconds: "
        f"{total_search_time:.3f}"
    )

    print(
        f"B total seconds: "
        f"{total_learned_time:.3f}"
    )

    print(
        f"A mean seconds/game: "
        f"{mean_search_time:.4f}"
    )

    print(
        f"B mean seconds/game: "
        f"{mean_learned_time:.4f}"
    )

    if mean_learned_time > 0:
        print(
            "A/B game-time ratio: "
            f"{mean_search_time / mean_learned_time:.2f}x"
        )

    print()
    print(
        "Note: timing measures complete games, "
        "not isolated target-agent decision latency."
    )


if __name__ == "__main__":
    main()
