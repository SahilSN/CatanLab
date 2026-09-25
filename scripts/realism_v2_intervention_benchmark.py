from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from catanlab.game import run_game
from catanlab.rl_realism_v2_checkpoint import (
    load_realism_v2_checkpoint,
)
from catanlab.rl_realism_v2_hybrid import (
    RealismV2HybridAgent,
)
from catanlab.strategies import StrategyType
from catanlab.turns import AdaptiveStrategyAgent


ARMS = {
    "learned": frozenset(),
    "teacher_ordinary": frozenset(
        {"ordinary"}
    ),
    "teacher_trades": frozenset(
        {"trades"}
    ),
    "teacher_robber": frozenset(
        {"robber"}
    ),
    "teacher_discard": frozenset(
        {"discard"}
    ),
    "teacher_dev_cards": frozenset(
        {"dev_cards"}
    ),
}


TARGET_STRATEGY = (
    StrategyType.FIVE_RESOURCE
)

OPPONENT_STRATEGIES = (
    StrategyType.HYBRID_OWS,
    StrategyType.FULL_OWS,
    StrategyType.PORT,
)


def build_agents(
    arm,
    *,
    model,
    target_seat,
    seed,
):
    target = RealismV2HybridAgent(
        TARGET_STRATEGY,
        model,
        interventions=ARMS[arm],
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


def summarize(
    records,
    *,
    arms,
):
    summary = {}

    for arm in arms:
        subset = [
            record
            for record in records
            if record["arm"] == arm
        ]

        games = len(subset)

        wins = sum(
            record["target_won"]
            for record in subset
        )

        summary[arm] = {
            "games": games,
            "wins": wins,
            "win_rate": (
                wins / games
            ),
            "mean_vp": (
                sum(
                    record["target_vp"]
                    for record in subset
                )
                / games
            ),
            "mean_turns": (
                sum(
                    record["turns"]
                    for record in subset
                )
                / games
            ),
        }

    baseline = {
        record["game_index"]: record
        for record in records
        if record["arm"] == "learned"
    }

    paired = {}

    for arm in arms:
        if arm == "learned":
            continue

        vp_deltas = []
        win_deltas = []

        higher = 0
        tied = 0
        lower = 0

        for record in records:
            if record["arm"] != arm:
                continue

            reference = baseline[
                record["game_index"]
            ]

            vp_delta = (
                record["target_vp"]
                - reference["target_vp"]
            )

            win_delta = (
                record["target_won"]
                - reference["target_won"]
            )

            vp_deltas.append(
                vp_delta
            )
            win_deltas.append(
                win_delta
            )

            if vp_delta > 0:
                higher += 1
            elif vp_delta < 0:
                lower += 1
            else:
                tied += 1

        paired[arm] = {
            "paired_games": len(
                vp_deltas
            ),
            "mean_vp_gain_vs_learned": (
                sum(vp_deltas)
                / len(vp_deltas)
            ),
            "mean_win_gain_vs_learned": (
                sum(win_deltas)
                / len(win_deltas)
            ),
            "vp_higher": higher,
            "vp_tied": tied,
            "vp_lower": lower,
        }

    return (
        summary,
        paired,
    )


def write_csv(
    path,
    records,
):
    fields = (
        "arm",
        "game_index",
        "seed",
        "target_seat",
        "winner_id",
        "target_won",
        "target_vp",
        "turns",
        "illegal_decisions",
        "fallbacks",
        "teacher_takeovers",
    )

    with path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    key: record[key]
                    for key in fields
                }
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Paired realism-v2 family intervention "
            "benchmark."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--games",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--arms",
        nargs="+",
        choices=tuple(ARMS),
        default=None,
        help=(
            "Subset of intervention arms to run. "
            "Default: all arms."
        ),
    )

    parser.add_argument(
        "--seed-offset",
        type=int,
        default=1_401_000,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    if args.games <= 0:
        raise ValueError(
            "--games must be positive."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = load_realism_v2_checkpoint(
        args.model,
        device="cpu",
    )

    selected_arms = (
        tuple(args.arms)
        if args.arms is not None
        else tuple(ARMS)
    )

    if "learned" not in selected_arms:
        raise ValueError(
            "--arms must include 'learned' "
            "so paired gains have a baseline."
        )

    records = []

    takeover_totals = {
        arm: Counter()
        for arm in selected_arms
    }

    total_runs = (
        args.games
        * len(selected_arms)
    )

    completed = 0

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

        for arm in selected_arms:
            (
                strategies,
                agents,
                target,
            ) = build_agents(
                arm,
                model=model,
                target_seat=(
                    target_seat
                ),
                seed=seed,
            )

            result = run_game(
                strategies=strategies,
                seed=seed,
                max_turns=2000,
                validate_conservation=True,
                turn_agents=agents,
            )

            target_vp = (
                result.players[
                    target_seat
                ].victory_points
            )

            target_won = int(
                result.winner_id
                == target_seat
            )

            if (
                target.illegal_decisions
                != 0
            ):
                raise RuntimeError(
                    f"{arm}: learned policy "
                    "made illegal decisions."
                )

            if target.fallbacks != 0:
                raise RuntimeError(
                    f"{arm}: learned policy "
                    "used fallbacks."
                )

            takeovers = dict(
                target.teacher_takeovers
            )

            takeover_totals[
                arm
            ].update(
                takeovers
            )

            records.append(
                {
                    "arm": arm,
                    "game_index": (
                        game_index
                    ),
                    "seed": seed,
                    "target_seat": (
                        target_seat
                    ),
                    "winner_id": (
                        result.winner_id
                    ),
                    "target_won": (
                        target_won
                    ),
                    "target_vp": (
                        target_vp
                    ),
                    "turns": (
                        result.turns_played
                    ),
                    "illegal_decisions": (
                        target
                        .illegal_decisions
                    ),
                    "fallbacks": (
                        target.fallbacks
                    ),
                    "teacher_takeovers": (
                        json.dumps(
                            takeovers,
                            sort_keys=True,
                        )
                    ),
                }
            )

            completed += 1

            print(
                f"[{completed}/{total_runs}] "
                f"game={game_index + 1}/"
                f"{args.games} "
                f"seed={seed} "
                f"seat={target_seat} "
                f"arm={arm:18s} "
                f"winner="
                f"{result.winner_id} "
                f"vp={target_vp} "
                f"turns="
                f"{result.turns_played} "
                f"takeovers="
                f"{takeovers}",
                flush=True,
            )

    summary, paired = summarize(
        records,
        arms=selected_arms,
    )

    csv_path = (
        args.output_dir
        / "games.csv"
    )

    json_path = (
        args.output_dir
        / "summary.json"
    )

    write_csv(
        csv_path,
        records,
    )

    payload = {
        "games_per_arm": (
            args.games
        ),
        "seed_offset": (
            args.seed_offset
        ),
        "arms": {
            arm: sorted(
                ARMS[arm]
            )
            for arm in selected_arms
        },
        "summary": summary,
        "paired_vs_learned": (
            paired
        ),
        "teacher_takeovers": {
            arm: dict(
                takeover_totals[
                    arm
                ]
            )
            for arm in selected_arms
        },
    }

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print(
        "=== REALISM-V2 FAMILY "
        "INTERVENTIONS ==="
    )

    print(
        f"{'arm':20s} "
        f"{'wins':>6s} "
        f"{'rate':>8s} "
        f"{'mean_vp':>9s} "
        f"{'gain_vp':>9s}"
    )

    print("-" * 58)

    for arm in selected_arms:
        stats = summary[
            arm
        ]

        if arm == "learned":
            gain = 0.0
        else:
            gain = (
                paired[
                    arm
                ][
                    "mean_vp_gain_vs_learned"
                ]
            )

        print(
            f"{arm:20s} "
            f"{stats['wins']:6d} "
            f"{stats['win_rate']:8.3f} "
            f"{stats['mean_vp']:9.3f} "
            f"{gain:+9.3f}"
        )

    print()
    print(
        "paired gains vs learned:"
    )

    for arm, stats in (
        paired.items()
    ):
        print(
            f"  {arm}: "
            f"VP="
            f"{stats['mean_vp_gain_vs_learned']:+.3f}, "
            f"win="
            f"{stats['mean_win_gain_vs_learned']:+.3f}, "
            f"W/T/L="
            f"{stats['vp_higher']}/"
            f"{stats['vp_tied']}/"
            f"{stats['vp_lower']}"
        )

    print()
    print(
        "teacher takeover totals:"
    )

    for arm in selected_arms:
        print(
            f"  {arm:20s} "
            f"{dict(takeover_totals[arm])}"
        )

    print()
    print(
        f"games CSV: {csv_path}"
    )
    print(
        f"summary:   {json_path}"
    )


if __name__ == "__main__":
    main()
