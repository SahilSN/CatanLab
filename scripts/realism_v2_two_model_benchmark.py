from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from catanlab.game import run_game
from catanlab.rl_realism_v2_agent import RealismV2PolicyAgent
from catanlab.rl_realism_v2_checkpoint import (
    load_realism_v2_checkpoint,
)
from catanlab.strategies import StrategyType
from catanlab.turns import AdaptiveStrategyAgent


OPPONENT_STRATEGIES = (
    StrategyType.HYBRID_OWS,
    StrategyType.FULL_OWS,
    StrategyType.PORT,
)


def make_agents(model, target_seat, seed):
    strategies = []
    agents = []

    opponent_index = 0

    for seat in range(4):
        if seat == target_seat:
            strategy = StrategyType.FIVE_RESOURCE

            strategies.append(strategy)
            agents.append(
                RealismV2PolicyAgent(
                    strategy,
                    model=model,
                    deterministic=True,
                    seed=seed + 1_000_000,
                )
            )
        else:
            strategy = OPPONENT_STRATEGIES[
                opponent_index
            ]
            opponent_index += 1

            strategies.append(strategy)
            agents.append(
                AdaptiveStrategyAgent(strategy)
            )

    return strategies, agents


def run_one(model, game_index, seed_offset):
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
        "vp": vp,
        "won": won,
    }


def bootstrap_mean_ci(
    values,
    *,
    repetitions,
    rng,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    indices = rng.integers(
        0,
        len(values),
        size=(
            repetitions,
            len(values),
        ),
    )

    means = values[
        indices
    ].mean(axis=1)

    low, high = np.quantile(
        means,
        [0.025, 0.975],
    )

    return (
        float(low),
        float(high),
        means,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--r1",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--continuation",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--games",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--seed-offset",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=100_000,
    )

    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    if args.games <= 0:
        raise ValueError(
            "--games must be positive."
        )

    r1 = load_realism_v2_checkpoint(
        args.r1,
        device="cpu",
    )
    continuation = load_realism_v2_checkpoint(
        args.continuation,
        device="cpu",
    )

    r1.eval()
    continuation.eval()

    rows = []

    for game_index in range(args.games):
        seed = (
            args.seed_offset
            + game_index
        )
        seat = game_index % 4

        print(
            f"\n[{game_index + 1}/{args.games}] "
            f"seed={seed} seat={seat}",
            flush=True,
        )

        r1_row = run_one(
            r1,
            game_index,
            args.seed_offset,
        )

        cont_row = run_one(
            continuation,
            game_index,
            args.seed_offset,
        )

        rows.append(
            {
                "r1": r1_row,
                "continuation": cont_row,
            }
        )

        print(
            f"  r1           "
            f"win={r1_row['won']} "
            f"vp={r1_row['vp']:.0f}",
            flush=True,
        )

        print(
            f"  continuation "
            f"win={cont_row['won']} "
            f"vp={cont_row['vp']:.0f}",
            flush=True,
        )

    r1_vp = np.asarray(
        [
            row["r1"]["vp"]
            for row in rows
        ],
        dtype=np.float64,
    )

    cont_vp = np.asarray(
        [
            row["continuation"]["vp"]
            for row in rows
        ],
        dtype=np.float64,
    )

    r1_win = np.asarray(
        [
            row["r1"]["won"]
            for row in rows
        ],
        dtype=np.float64,
    )

    cont_win = np.asarray(
        [
            row["continuation"]["won"]
            for row in rows
        ],
        dtype=np.float64,
    )

    vp_delta = cont_vp - r1_vp
    win_delta = cont_win - r1_win

    rng = np.random.default_rng(
        args.bootstrap_seed
    )

    vp_low, vp_high, vp_boot = (
        bootstrap_mean_ci(
            vp_delta,
            repetitions=(
                args.bootstrap_repetitions
            ),
            rng=rng,
        )
    )

    win_low, win_high, win_boot = (
        bootstrap_mean_ci(
            win_delta,
            repetitions=(
                args.bootstrap_repetitions
            ),
            rng=rng,
        )
    )

    vp_p = min(
        1.0,
        2.0
        * min(
            float(
                np.mean(
                    vp_boot <= 0
                )
            ),
            float(
                np.mean(
                    vp_boot >= 0
                )
            ),
        ),
    )

    win_p = min(
        1.0,
        2.0
        * min(
            float(
                np.mean(
                    win_boot <= 0
                )
            ),
            float(
                np.mean(
                    win_boot >= 0
                )
            ),
        ),
    )

    vp_wins = int(
        np.sum(vp_delta > 0)
    )
    vp_ties = int(
        np.sum(vp_delta == 0)
    )
    vp_losses = int(
        np.sum(vp_delta < 0)
    )

    print()
    print(
        "=== FINAL CHECKPOINT-SELECTION "
        "ENGINEERING BENCHMARK ==="
    )

    print(
        f"r1           "
        f"wins={int(r1_win.sum()):3d} "
        f"rate={r1_win.mean():.3f} "
        f"mean_vp={r1_vp.mean():.3f}"
    )

    print(
        f"continuation "
        f"wins={int(cont_win.sum()):3d} "
        f"rate={cont_win.mean():.3f} "
        f"mean_vp={cont_vp.mean():.3f}"
    )

    print()
    print(
        "=== PAIRED CONTINUATION - R1 ==="
    )

    print(
        f"VP delta: "
        f"{vp_delta.mean():+.3f} "
        f"95% CI "
        f"[{vp_low:+.3f}, {vp_high:+.3f}] "
        f"bootstrap-p={vp_p:.4f}"
    )

    print(
        f"win delta: "
        f"{win_delta.mean():+.3f} "
        f"95% CI "
        f"[{win_low:+.3f}, {win_high:+.3f}] "
        f"bootstrap-p={win_p:.4f}"
    )

    print(
        f"VP W/T/L: "
        f"{vp_wins}/{vp_ties}/{vp_losses}"
    )


if __name__ == "__main__":
    main()
