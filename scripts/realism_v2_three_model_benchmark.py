from __future__ import annotations

import argparse
from pathlib import Path
import statistics

import torch

from catanlab.game import run_game
from catanlab.rl_realism_v2_agent import (
    RealismV2PolicyAgent,
)
from catanlab.rl_realism_v2_checkpoint import (
    load_realism_v2_checkpoint,
)
from catanlab.strategies import StrategyType
from catanlab.turns import AdaptiveStrategyAgent


def make_agents(
    model,
    target_seat: int,
    seed: int,
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
            strategies.append(
                target_strategy
            )

            agents.append(
                RealismV2PolicyAgent(
                    target_strategy,
                    model=model,
                    deterministic=True,
                    seed=seed + 1_000_000,
                )
            )
        else:
            strategy = opponent_strategies[
                opponent_index
            ]
            opponent_index += 1

            strategies.append(strategy)
            agents.append(
                AdaptiveStrategyAgent(
                    strategy
                )
            )

    return strategies, agents


def run_model_game(
    model,
    *,
    game_index: int,
    seed_offset: int,
):
    seed = seed_offset + game_index
    target_seat = game_index % 4

    strategies, agents = make_agents(
        model,
        target_seat,
        seed,
    )

    result = run_game(
        strategies=strategies,
        seed=seed,
        max_turns=2000,
        validate_conservation=True,
        turn_agents=agents,
    )

    vp = float(
        result.players[
            target_seat
        ].victory_points
    )

    won = int(
        result.winner_id == target_seat
    )

    return {
        "seed": seed,
        "target_seat": target_seat,
        "won": won,
        "vp": vp,
    }


def summarize(rows):
    return {
        "wins": sum(
            row["won"]
            for row in rows
        ),
        "win_rate": statistics.mean(
            row["won"]
            for row in rows
        ),
        "mean_vp": statistics.mean(
            row["vp"]
            for row in rows
        ),
    }


def paired_summary(
    left,
    right,
):
    vp_deltas = [
        a["vp"] - b["vp"]
        for a, b in zip(
            left,
            right,
            strict=True,
        )
    ]

    win_deltas = [
        a["won"] - b["won"]
        for a, b in zip(
            left,
            right,
            strict=True,
        )
    ]

    vp_wins = sum(
        delta > 0
        for delta in vp_deltas
    )
    vp_ties = sum(
        delta == 0
        for delta in vp_deltas
    )
    vp_losses = sum(
        delta < 0
        for delta in vp_deltas
    )

    return {
        "mean_vp_delta": (
            statistics.mean(
                vp_deltas
            )
        ),
        "mean_win_delta": (
            statistics.mean(
                win_deltas
            )
        ),
        "vp_wtl": (
            vp_wins,
            vp_ties,
            vp_losses,
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--original",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--control",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dagger",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    if args.games <= 0:
        raise ValueError(
            "--games must be positive."
        )

    paths = {
        "original": args.original,
        "control": args.control,
        "dagger": args.dagger,
    }

    models = {}

    for name, path in paths.items():
        loaded = (
            load_realism_v2_checkpoint(
                path,
                device="cpu",
            )
        )

        # Support either a loader returning the model
        # directly or a small checkpoint wrapper.
        model = getattr(
            loaded,
            "model",
            loaded,
        )

        model.eval()
        models[name] = model

    rows = {
        name: []
        for name in models
    }

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

        print(
            f"\n[{game_index + 1}/"
            f"{args.games}] "
            f"seed={seed} "
            f"seat={target_seat}",
            flush=True,
        )

        for name in (
            "original",
            "control",
            "dagger",
        ):
            row = run_model_game(
                models[name],
                game_index=game_index,
                seed_offset=(
                    args.seed_offset
                ),
            )

            rows[name].append(row)

            print(
                f"  {name:8s} "
                f"win={row['won']} "
                f"vp={row['vp']:.0f}",
                flush=True,
            )

    print(
        "\n=== THREE-MODEL "
        "REALISM-V2 BENCHMARK ==="
    )

    for name in (
        "original",
        "control",
        "dagger",
    ):
        summary = summarize(
            rows[name]
        )

        print(
            f"{name:8s} "
            f"wins={summary['wins']:3d} "
            f"rate="
            f"{summary['win_rate']:.3f} "
            f"mean_vp="
            f"{summary['mean_vp']:.3f}"
        )

    comparisons = (
        (
            "control_minus_original",
            "control",
            "original",
        ),
        (
            "dagger_minus_original",
            "dagger",
            "original",
        ),
        (
            "dagger_minus_control",
            "dagger",
            "control",
        ),
    )

    print(
        "\n=== PAIRED DELTAS ==="
    )

    for label, left, right in comparisons:
        summary = paired_summary(
            rows[left],
            rows[right],
        )

        w, t, l = (
            summary["vp_wtl"]
        )

        print(label)
        print(
            "  mean VP delta: "
            f"{summary['mean_vp_delta']:+.3f}"
        )
        print(
            "  mean win delta: "
            f"{summary['mean_win_delta']:+.3f}"
        )
        print(
            "  VP W/T/L: "
            f"{w}/{t}/{l}"
        )


if __name__ == "__main__":
    main()
