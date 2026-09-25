from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from catanlab.game import run_game
from catanlab.rl_realism_v2_agent import (
    RealismV2PolicyAgent,
)
from catanlab.rl_realism_v2_checkpoint import (
    load_realism_v2_checkpoint,
)
from catanlab.rl_teacher import (
    RecordingSearchAgent,
    TeacherDecisionKind,
)
from catanlab.strategies import (
    StrategyType,
)
from catanlab.turns import (
    AdaptiveStrategyAgent,
)


POLICIES = (
    "learned",
    "teacher",
    "heuristic",
)

TARGET_STRATEGY = (
    StrategyType.FIVE_RESOURCE
)

OPPONENT_STRATEGIES = (
    StrategyType.HYBRID_OWS,
    StrategyType.FULL_OWS,
    StrategyType.PORT,
)


def make_target_agent(
    policy_name,
    *,
    model,
    seed,
):
    if policy_name == "learned":
        return RealismV2PolicyAgent(
            TARGET_STRATEGY,
            model=model,
            deterministic=True,
            seed=seed,
        )

    if policy_name == "teacher":
        # Exact canonical realism-v2 Search teacher
        # configuration.
        return RecordingSearchAgent(
            TARGET_STRATEGY,
            search_depth=2,
            use_transposition_cache=False,
            search_maritime_trades=True,
            search_year_of_plenty=True,
            search_road_building=True,
            search_monopoly=True,
            search_robber_decisions=True,
            search_discard_decisions=True,
            search_domestic_trades=True,
        )

    if policy_name == "heuristic":
        return AdaptiveStrategyAgent(
            TARGET_STRATEGY
        )

    raise ValueError(
        f"Unknown policy: {policy_name!r}"
    )


def build_game_agents(
    policy_name,
    *,
    target_seat,
    model,
    agent_seed,
):
    strategies = []
    agents = []

    target_agent = None
    opponent_index = 0

    for seat in range(4):
        if seat == target_seat:
            target_agent = make_target_agent(
                policy_name,
                model=model,
                seed=agent_seed,
            )

            strategies.append(
                TARGET_STRATEGY
            )
            agents.append(
                target_agent
            )

        else:
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

    if target_agent is None:
        raise RuntimeError(
            "Target agent was not created."
        )

    return (
        strategies,
        agents,
        target_agent,
    )


def summarize_records(
    records,
):
    by_policy = {}

    for policy_name in POLICIES:
        subset = [
            record
            for record in records
            if (
                record["policy"]
                == policy_name
            )
        ]

        if not subset:
            continue

        games = len(subset)

        wins = sum(
            record["target_won"]
            for record in subset
        )

        completed = sum(
            record["winner_id"] is not None
            for record in subset
        )

        mean_vp = (
            sum(
                record["target_vp"]
                for record in subset
            )
            / games
        )

        mean_turns = (
            sum(
                record["turns"]
                for record in subset
            )
            / games
        )

        seat_games = [0] * 4
        seat_wins = [0] * 4

        for record in subset:
            seat = record[
                "target_seat"
            ]

            seat_games[seat] += 1
            seat_wins[seat] += (
                record["target_won"]
            )

        seat_win_rates = []

        for seat in range(4):
            if seat_games[seat]:
                rate = (
                    seat_wins[seat]
                    / seat_games[seat]
                )
            else:
                rate = None

            seat_win_rates.append(
                rate
            )

        by_policy[policy_name] = {
            "games": games,
            "games_with_winner": (
                completed
            ),
            "wins": wins,
            "win_rate": wins / games,
            "mean_vp": mean_vp,
            "mean_turns": mean_turns,
            "seat_games": seat_games,
            "seat_wins": seat_wins,
            "seat_win_rates": (
                seat_win_rates
            ),
        }

    return by_policy


def paired_deltas(
    records,
):
    indexed = {
        (
            record["policy"],
            record["game_index"],
        ): record
        for record in records
    }

    output = {}

    for comparator in (
        "teacher",
        "heuristic",
    ):
        vp_deltas = []
        win_deltas = []

        for game_index in sorted(
            {
                record["game_index"]
                for record in records
            }
        ):
            learned = indexed.get(
                (
                    "learned",
                    game_index,
                )
            )

            other = indexed.get(
                (
                    comparator,
                    game_index,
                )
            )

            if (
                learned is None
                or other is None
            ):
                continue

            vp_deltas.append(
                learned["target_vp"]
                - other["target_vp"]
            )

            win_deltas.append(
                learned["target_won"]
                - other["target_won"]
            )

        if vp_deltas:
            output[
                f"learned_minus_{comparator}"
            ] = {
                "paired_games": (
                    len(vp_deltas)
                ),
                "mean_vp_delta": (
                    sum(vp_deltas)
                    / len(vp_deltas)
                ),
                "mean_win_delta": (
                    sum(win_deltas)
                    / len(win_deltas)
                ),
            }

    return output


def write_csv(
    path,
    records,
):
    fieldnames = (
        "policy",
        "game_index",
        "seed",
        "target_seat",
        "winner_id",
        "target_won",
        "target_vp",
        "turns",
        "illegal_decisions",
        "fallbacks",
    )

    with path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    key: record[key]
                    for key in fieldnames
                }
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Paired game-level evaluation of the "
            "realism-v2 learned policy, canonical "
            "Search-v2 teacher, and heuristic baseline."
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
        default=20,
    )

    parser.add_argument(
        "--seed-offset",
        type=int,
        default=1_300_000,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/realism_v2_game_benchmark"
        ),
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

    model = (
        load_realism_v2_checkpoint(
            args.model,
            device="cpu",
        )
    )

    records = []

    learned_decisions = Counter()

    total_runs = (
        args.games
        * len(POLICIES)
    )

    completed_runs = 0

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

        for policy_name in POLICIES:
            (
                strategies,
                agents,
                target_agent,
            ) = build_game_agents(
                policy_name,
                target_seat=target_seat,
                model=model,
                agent_seed=(
                    seed
                    + 10_000_000
                ),
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

            illegal_decisions = 0
            fallbacks = 0

            if policy_name == "learned":
                illegal_decisions = (
                    target_agent
                    .illegal_decisions
                )

                fallbacks = (
                    target_agent
                    .fallbacks
                )

                learned_decisions.update(
                    target_agent
                    .decision_counts
                )

                if illegal_decisions:
                    raise RuntimeError(
                        "Learned policy made an "
                        "illegal decision."
                    )

                if fallbacks:
                    raise RuntimeError(
                        "Learned policy used a "
                        "heuristic fallback."
                    )

            record = {
                "policy": policy_name,
                "game_index": game_index,
                "seed": seed,
                "target_seat": target_seat,
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
                    illegal_decisions
                ),
                "fallbacks": (
                    fallbacks
                ),
            }

            records.append(
                record
            )

            completed_runs += 1

            print(
                f"[{completed_runs}/"
                f"{total_runs}] "
                f"game={game_index + 1}/"
                f"{args.games} "
                f"seed={seed} "
                f"seat={target_seat} "
                f"policy={policy_name:9s} "
                f"winner="
                f"{result.winner_id} "
                f"vp={target_vp} "
                f"turns="
                f"{result.turns_played}",
                flush=True,
            )

    summary = (
        summarize_records(
            records
        )
    )

    paired = paired_deltas(
        records
    )

    csv_path = (
        args.output_dir
        / "games.csv"
    )

    summary_path = (
        args.output_dir
        / "summary.json"
    )

    write_csv(
        csv_path,
        records,
    )

    payload = {
        "model": str(
            args.model
        ),
        "games_per_policy": (
            args.games
        ),
        "seed_offset": (
            args.seed_offset
        ),
        "policies": summary,
        "paired": paired,
        "learned_decision_counts": {
            kind.value: (
                learned_decisions[
                    kind.value
                ]
            )
            for kind in (
                TeacherDecisionKind
            )
        },
    }

    summary_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print(
        "=== REALISM-V2 BENCHMARK ==="
    )

    print(
        f"{'policy':10s} "
        f"{'wins':>6s} "
        f"{'rate':>8s} "
        f"{'mean_vp':>9s} "
        f"{'turns':>9s}"
    )

    print("-" * 48)

    for policy_name in POLICIES:
        values = summary[
            policy_name
        ]

        print(
            f"{policy_name:10s} "
            f"{values['wins']:6d} "
            f"{values['win_rate']:8.3f} "
            f"{values['mean_vp']:9.3f} "
            f"{values['mean_turns']:9.1f}"
        )

    print()
    print("paired learned deltas:")

    for name, values in (
        paired.items()
    ):
        print(
            f"  {name}: "
            f"VP="
            f"{values['mean_vp_delta']:+.3f}, "
            f"win="
            f"{values['mean_win_delta']:+.3f}"
        )

    print()
    print(
        "learned decision counts:"
    )

    for kind in TeacherDecisionKind:
        print(
            f"  {kind.value:20s} "
            f"{learned_decisions[kind.value]}"
        )

    print()
    print(
        f"games CSV: {csv_path}"
    )
    print(
        f"summary:   {summary_path}"
    )


if __name__ == "__main__":
    main()
