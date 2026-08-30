from __future__ import annotations

import argparse
import csv
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


VARIANT_ORDER = (
    "adaptive",
    "search",
    "bc_dagger",
    "ppo_bckl",
)


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


def make_target_agent(
    variant,
    strategy,
    model,
    agent_seed,
):
    if variant == "adaptive":
        return AdaptiveStrategyAgent(strategy)

    if variant == "search":
        return OneStepLookaheadAgent(
            strategy,
            search_depth=2,
            use_transposition_cache=False,
            search_maritime_trades=True,
            search_year_of_plenty=False,
            search_road_building=True,
            search_monopoly=False,
        )

    if variant in {
        "bc_dagger",
        "ppo_bckl",
    }:
        if model is None:
            raise ValueError(
                f"Model required for {variant}"
            )

        return NeuralPolicyAgent(
            strategy,
            model=model,
            deterministic=True,
            seed=agent_seed,
        )

    raise ValueError(
        f"Unknown variant: {variant!r}"
    )


def make_lineup(
    target_seat,
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
            strategies.append(
                target_strategy
            )
            agents.append(
                target_agent
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


def run_variant(
    variant,
    model,
    game_index,
    seed_offset,
):
    target_seat = game_index % 4
    game_seed = seed_offset + game_index

    target_strategy = (
        StrategyType.FIVE_RESOURCE
    )

    target_agent = make_target_agent(
        variant=variant,
        strategy=target_strategy,
        model=model,
        agent_seed=(
            game_seed
            + 1_000_000
        ),
    )

    strategies, agents = make_lineup(
        target_seat=target_seat,
        target_agent=target_agent,
    )

    start = time.perf_counter()

    result = run_game(
        strategies=strategies,
        seed=game_seed,
        max_turns=2000,
        validate_conservation=True,
        turn_agents=agents,
    )

    runtime = (
        time.perf_counter()
        - start
    )

    player = result.players[
        target_seat
    ]

    return {
        "variant": variant,
        "game_index": game_index,
        "seed": game_seed,
        "seat": target_seat,
        "won": int(
            result.winner_id
            == target_seat
        ),
        "final_vp": float(
            player.victory_points
        ),
        "roads": len(
            player.roads
        ),
        "settlements": len(
            player.settlements
        ),
        "cities": len(
            player.cities
        ),
        "dev_cards": len(
            player.dev_cards
        ),
        "has_longest_road": int(
            player.has_longest_road
        ),
        "has_largest_army": int(
            player.has_largest_army
        ),
        "turns": int(
            result.turns_played
        ),
        "runtime_seconds": runtime,
    }


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


def rows_for_variant(
    rows,
    variant,
):
    return [
        row
        for row in rows
        if row["variant"] == variant
    ]


def print_summary(rows):
    print()
    print(
        "=== CANONICAL AGENT BENCHMARK ==="
    )

    print(
        "variant       games   "
        "win_rate   mean_vp   "
        "mean_turns   sec/game"
    )

    print("-" * 70)

    for variant in VARIANT_ORDER:
        variant_rows = rows_for_variant(
            rows,
            variant,
        )

        wins = np.asarray(
            [
                row["won"]
                for row in variant_rows
            ],
            dtype=np.float64,
        )

        vp = np.asarray(
            [
                row["final_vp"]
                for row in variant_rows
            ],
            dtype=np.float64,
        )

        turns = np.asarray(
            [
                row["turns"]
                for row in variant_rows
            ],
            dtype=np.float64,
        )

        runtime = np.asarray(
            [
                row["runtime_seconds"]
                for row in variant_rows
            ],
            dtype=np.float64,
        )

        print(
            f"{variant:<13}"
            f"{len(variant_rows):>6}   "
            f"{wins.mean():>8.4f}   "
            f"{vp.mean():>7.4f}   "
            f"{turns.mean():>10.2f}   "
            f"{runtime.mean():>8.4f}"
        )


def print_seat_summary(rows):
    print()
    print("=== WIN RATE BY SEAT ===")

    print(
        "variant         "
        "seat0   seat1   seat2   seat3"
    )

    print("-" * 52)

    for variant in VARIANT_ORDER:
        values = []

        for seat in range(4):
            seat_rows = [
                row
                for row in rows
                if (
                    row["variant"]
                    == variant
                    and row["seat"]
                    == seat
                )
            ]

            rate = np.mean(
                [
                    row["won"]
                    for row in seat_rows
                ]
            )

            values.append(rate)

        print(
            f"{variant:<15}"
            + " ".join(
                f"{value:>7.4f}"
                for value in values
            )
        )


def paired_difference(
    rows,
    variant_a,
    variant_b,
    bootstrap_seed,
):
    a_rows = rows_for_variant(
        rows,
        variant_a,
    )

    b_rows = rows_for_variant(
        rows,
        variant_b,
    )

    a_rows = sorted(
        a_rows,
        key=lambda row: (
            row["game_index"],
            row["seat"],
        ),
    )

    b_rows = sorted(
        b_rows,
        key=lambda row: (
            row["game_index"],
            row["seat"],
        ),
    )

    if len(a_rows) != len(b_rows):
        raise ValueError(
            "Paired variants have "
            "different game counts"
        )

    for a_row, b_row in zip(
        a_rows,
        b_rows,
    ):
        if (
            a_row["seed"]
            != b_row["seed"]
            or a_row["seat"]
            != b_row["seat"]
        ):
            raise ValueError(
                "Pairing mismatch"
            )

    a_win = np.asarray(
        [
            row["won"]
            for row in a_rows
        ],
        dtype=np.float64,
    )

    b_win = np.asarray(
        [
            row["won"]
            for row in b_rows
        ],
        dtype=np.float64,
    )

    a_vp = np.asarray(
        [
            row["final_vp"]
            for row in a_rows
        ],
        dtype=np.float64,
    )

    b_vp = np.asarray(
        [
            row["final_vp"]
            for row in b_rows
        ],
        dtype=np.float64,
    )

    win_difference = (
        b_win - a_win
    )

    vp_difference = (
        b_vp - a_vp
    )

    rng = np.random.default_rng(
        bootstrap_seed
    )

    win_ci = bootstrap_ci(
        win_difference,
        rng,
    )

    vp_ci = bootstrap_ci(
        vp_difference,
        rng,
    )

    return {
        "a_win": float(
            a_win.mean()
        ),
        "b_win": float(
            b_win.mean()
        ),
        "win_difference": float(
            win_difference.mean()
        ),
        "win_ci": win_ci,
        "a_vp": float(
            a_vp.mean()
        ),
        "b_vp": float(
            b_vp.mean()
        ),
        "vp_difference": float(
            vp_difference.mean()
        ),
        "vp_ci": vp_ci,
    }


def print_pairwise(
    rows,
    bootstrap_seed,
):
    comparisons = [
        (
            "adaptive",
            "search",
        ),
        (
            "adaptive",
            "bc_dagger",
        ),
        (
            "adaptive",
            "ppo_bckl",
        ),
        (
            "bc_dagger",
            "ppo_bckl",
        ),
        (
            "search",
            "ppo_bckl",
        ),
    ]

    print()
    print(
        "=== PAIRED DIFFERENCES (B - A) ==="
    )

    for index, (
        variant_a,
        variant_b,
    ) in enumerate(comparisons):
        result = paired_difference(
            rows,
            variant_a,
            variant_b,
            bootstrap_seed + index,
        )

        print()
        print(
            f"{variant_a} -> {variant_b}"
        )

        print(
            "  win: "
            f"{result['a_win']:.4f} "
            "-> "
            f"{result['b_win']:.4f} "
            f"diff="
            f"{result['win_difference']:+.4f} "
            "95% CI "
            f"[{result['win_ci'][0]:+.4f}, "
            f"{result['win_ci'][1]:+.4f}]"
        )

        print(
            "  VP:  "
            f"{result['a_vp']:.4f} "
            "-> "
            f"{result['b_vp']:.4f} "
            f"diff="
            f"{result['vp_difference']:+.4f} "
            "95% CI "
            f"[{result['vp_ci'][0]:+.4f}, "
            f"{result['vp_ci'][1]:+.4f}]"
        )


def write_csv(
    path,
    rows,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "variant",
        "game_index",
        "seed",
        "seat",
        "won",
        "final_vp",
        "roads",
        "settlements",
        "cities",
        "dev_cards",
        "has_longest_road",
        "has_largest_army",
        "turns",
        "runtime_seconds",
    ]

    with path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bc-model",
        type=Path,
        default=Path(
            "results/rl_baselines/"
            "bc_dagger_v1.pt"
        ),
    )

    parser.add_argument(
        "--ppo-model",
        type=Path,
        default=Path(
            "results/rl_baselines/"
            "ppo_bckl_v1.pt"
        ),
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

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    if args.games <= 0:
        raise ValueError(
            "--games must be positive"
        )

    bc_model = load_model(
        args.bc_model
    )

    ppo_model = load_model(
        args.ppo_model
    )

    models = {
        "adaptive": None,
        "search": None,
        "bc_dagger": bc_model,
        "ppo_bckl": ppo_model,
    }

    rows = []

    for game_index in range(
        args.games
    ):
        for variant in VARIANT_ORDER:
            row = run_variant(
                variant=variant,
                model=models[variant],
                game_index=game_index,
                seed_offset=args.seed_offset,
            )

            rows.append(row)

        completed = game_index + 1

        if (
            completed % 25 == 0
            or completed == args.games
        ):
            print(
                f"[{completed:>4}/"
                f"{args.games}]",
                flush=True,
            )

    print_summary(rows)

    print_seat_summary(rows)

    print_pairwise(
        rows,
        args.bootstrap_seed,
    )

    if args.output_csv is not None:
        write_csv(
            args.output_csv,
            rows,
        )

        print()
        print(
            "saved CSV: "
            f"{args.output_csv}"
        )


if __name__ == "__main__":
    main()
