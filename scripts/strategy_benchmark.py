from __future__ import annotations

import argparse
import csv
import itertools
from collections import defaultdict
from pathlib import Path
from statistics import mean

from catanlab.game import run_game
from catanlab.strategies import StrategyType
from catanlab.turns import ActionType


STRATEGIES = list(StrategyType)


def rotated(values, amount):
    amount %= len(values)

    return (
        values[amount:]
        + values[:amount]
    )


def safe_mean(values):
    if not values:
        return None

    return mean(values)


def fmt(value):
    if value is None:
        return ""

    if isinstance(value, float):
        return f"{value:.6f}"

    return value


def first_action_turn(
    game,
    player_id,
    action_type,
):
    """
    Return the 1-based game turn on which this
    player first performed action_type.
    """

    for turn_number, turn in enumerate(
        game.turn_history,
        start=1,
    ):
        if turn.player_id != player_id:
            continue

        if any(
            action.action_type == action_type
            for action in turn.actions
        ):
            return turn_number

    return None


def player_game_row(
    *,
    game_id,
    lineup_id,
    rotation,
    repetition,
    seed,
    strategies,
    game,
    player_id,
):
    strategy = strategies[player_id]
    player = game.players[player_id]

    action_counts = {
        action_type: 0
        for action_type in ActionType
    }

    domestic_trades = 0
    trade_offers = 0
    trade_sequences = 0
    discards = 0

    for turn in game.turn_history:
        if turn.player_id == player_id:
            for action in turn.actions:
                action_counts[
                    action.action_type
                ] += 1

            domestic_trades += len(
                turn.player_trades
            )

            trade_offers += (
                turn.trade_offer_count
            )

            trade_sequences += (
                turn.trade_sequence_count
            )

        # A roll of seven can make players other
        # than the active player discard.
        discards += len(
            turn.discards.get(
                player_id,
                [],
            )
        )

    non_pass_actions = sum(
        count
        for action_type, count
        in action_counts.items()
        if action_type != ActionType.PASS
    )

    total_actions = sum(
        action_counts.values()
    )

    return {
        "game_id": game_id,
        "lineup_id": lineup_id,
        "rotation": rotation,
        "repetition": repetition,
        "seed": seed,
        "seat": player_id,
        "strategy": strategy.value,

        "won": int(
            game.winner_id == player_id
        ),
        "game_completed": int(
            game.winner_id is not None
        ),
        "turns_played": game.turns_played,

        "final_vp": player.victory_points,
        "final_public_vp":
            player.public_victory_points,

        "final_roads": len(
            player.roads
        ),
        "final_settlements": len(
            player.settlements
        ),
        "final_cities": len(
            player.cities
        ),
        "final_dev_cards": len(
            player.dev_cards
        ),
        "hidden_vp_cards": sum(
            card == "victory_point"
            for card in player.dev_cards
        ),
        "knights_played":
            player.knights_played,

        "has_longest_road": int(
            player.has_longest_road
        ),
        "has_largest_army": int(
            player.has_largest_army
        ),

        "build_road_actions":
            action_counts[
                ActionType.BUILD_ROAD
            ],
        "build_settlement_actions":
            action_counts[
                ActionType.BUILD_SETTLEMENT
            ],
        "build_city_actions":
            action_counts[
                ActionType.BUILD_CITY
            ],
        "buy_dev_card_actions":
            action_counts[
                ActionType.BUY_DEV_CARD
            ],
        "maritime_trade_actions":
            action_counts[
                ActionType.MARITIME_TRADE
            ],
        "pass_actions":
            action_counts[
                ActionType.PASS
            ],

        "non_pass_actions":
            non_pass_actions,
        "total_actions":
            total_actions,

        "domestic_trades":
            domestic_trades,
        "trade_offers":
            trade_offers,
        "trade_sequences":
            trade_sequences,
        "cards_discarded":
            discards,

        "first_road_turn":
            first_action_turn(
                game,
                player_id,
                ActionType.BUILD_ROAD,
            ),
        "first_expansion_turn":
            first_action_turn(
                game,
                player_id,
                ActionType.BUILD_SETTLEMENT,
            ),
        "first_city_turn":
            first_action_turn(
                game,
                player_id,
                ActionType.BUILD_CITY,
            ),
        "first_dev_card_turn":
            first_action_turn(
                game,
                player_id,
                ActionType.BUY_DEV_CARD,
            ),
        "first_maritime_trade_turn":
            first_action_turn(
                game,
                player_id,
                ActionType.MARITIME_TRADE,
            ),
    }


def strategy_summary(rows):
    groups = defaultdict(list)

    for row in rows:
        groups[
            row["strategy"]
        ].append(row)

    summaries = []

    for strategy in STRATEGIES:
        strategy_rows = groups[
            strategy.value
        ]

        if not strategy_rows:
            continue

        games = len(
            strategy_rows
        )

        completed_games = sum(
            row["game_completed"]
            for row in strategy_rows
        )

        wins = sum(
            row["won"]
            for row in strategy_rows
        )

        def avg(field):
            values = [
                row[field]
                for row in strategy_rows
                if row[field] is not None
            ]

            return safe_mean(
                values
            )

        total_non_pass = sum(
            row["non_pass_actions"]
            for row in strategy_rows
        )

        def action_share(field):
            if total_non_pass == 0:
                return 0.0

            return (
                sum(
                    row[field]
                    for row in strategy_rows
                )
                / total_non_pass
            )

        summaries.append(
            {
                "strategy":
                    strategy.value,
                "games":
                    games,
                "completed_games":
                    completed_games,
                "wins":
                    wins,
                "win_rate":
                    wins / games,

                "avg_final_vp":
                    avg("final_vp"),
                "avg_public_vp":
                    avg("final_public_vp"),
                "avg_game_turns":
                    avg("turns_played"),

                "avg_final_roads":
                    avg("final_roads"),
                "avg_final_settlements":
                    avg("final_settlements"),
                "avg_final_cities":
                    avg("final_cities"),
                "avg_dev_cards_bought":
                    avg("buy_dev_card_actions"),
                "avg_knights_played":
                    avg("knights_played"),

                "longest_road_end_rate":
                    avg("has_longest_road"),
                "largest_army_end_rate":
                    avg("has_largest_army"),

                "road_action_share":
                    action_share(
                        "build_road_actions"
                    ),
                "settlement_action_share":
                    action_share(
                        "build_settlement_actions"
                    ),
                "city_action_share":
                    action_share(
                        "build_city_actions"
                    ),
                "dev_action_share":
                    action_share(
                        "buy_dev_card_actions"
                    ),
                "maritime_action_share":
                    action_share(
                        "maritime_trade_actions"
                    ),

                "avg_domestic_trades":
                    avg("domestic_trades"),
                "avg_trade_offers":
                    avg("trade_offers"),
                "avg_trade_sequences":
                    avg("trade_sequences"),
                "avg_maritime_trades":
                    avg("maritime_trade_actions"),

                "avg_first_road_turn":
                    avg("first_road_turn"),
                "avg_first_expansion_turn":
                    avg("first_expansion_turn"),
                "avg_first_city_turn":
                    avg("first_city_turn"),
                "avg_first_dev_card_turn":
                    avg("first_dev_card_turn"),
                "avg_first_maritime_trade_turn":
                    avg(
                        "first_maritime_trade_turn"
                    ),
            }
        )

    return summaries


def write_csv(
    path,
    rows,
):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    key: fmt(value)
                    for key, value
                    in row.items()
                }
            )


def print_summary(
    summaries,
):
    print()
    print(
        "Strategy behavioral summary"
    )
    print(
        "=" * 110
    )

    header = (
        f"{'strategy':18}"
        f"{'games':>7}"
        f"{'win%':>9}"
        f"{'VP':>8}"
        f"{'roads':>8}"
        f"{'sett':>8}"
        f"{'city':>8}"
        f"{'dev':>8}"
        f"{'LR%':>8}"
        f"{'LA%':>8}"
        f"{'mar':>8}"
    )

    print(
        header
    )
    print(
        "-" * len(header)
    )

    for row in summaries:
        print(
            f"{row['strategy']:18}"
            f"{row['games']:>7}"
            f"{100 * row['win_rate']:>8.2f}%"
            f"{row['avg_final_vp']:>8.2f}"
            f"{row['avg_final_roads']:>8.2f}"
            f"{row['avg_final_settlements']:>8.2f}"
            f"{row['avg_final_cities']:>8.2f}"
            f"{row['avg_dev_cards_bought']:>8.2f}"
            f"{100 * row['longest_road_end_rate']:>7.2f}%"
            f"{100 * row['largest_army_end_rate']:>7.2f}%"
            f"{row['avg_maritime_trades']:>8.2f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark behavioral differences "
            "between CatanLab strategies."
        )
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help=(
            "Games per lineup/seat rotation. "
            "5 = 300 total games."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260826,
        help="Base deterministic seed.",
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--validate-conservation",
        action="store_true",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/strategy_benchmark"
        ),
    )

    args = parser.parse_args()

    if args.repetitions <= 0:
        raise ValueError(
            "--repetitions must be positive."
        )

    lineups = list(
        itertools.combinations(
            STRATEGIES,
            4,
        )
    )

    total_games = (
        len(lineups)
        * 4
        * args.repetitions
    )

    print(
        f"Strategies: {len(STRATEGIES)}"
    )
    print(
        f"Four-strategy lineups: {len(lineups)}"
    )
    print(
        f"Seat rotations per lineup: 4"
    )
    print(
        f"Repetitions: {args.repetitions}"
    )
    print(
        f"Total games: {total_games}"
    )
    print()

    rows = []

    game_id = 0

    for lineup_index, lineup_tuple in enumerate(
        lineups
    ):
        lineup = list(
            lineup_tuple
        )

        lineup_id = (
            f"lineup_{lineup_index:02d}"
        )

        for rotation in range(4):
            seated_strategies = rotated(
                lineup,
                rotation,
            )

            for repetition in range(
                args.repetitions
            ):
                seed = (
                    args.seed
                    + game_id
                )

                game_id += 1

                print(
                    f"[{game_id:4d}/{total_games}] "
                    f"{lineup_id} "
                    f"rot={rotation} "
                    f"rep={repetition} "
                    f"seed={seed}",
                    flush=True,
                )

                game = run_game(
                    seated_strategies,
                    seed=seed,
                    max_turns=args.max_turns,
                    validate_conservation=(
                        args.validate_conservation
                    ),
                )

                for player_id in range(4):
                    rows.append(
                        player_game_row(
                            game_id=game_id,
                            lineup_id=lineup_id,
                            rotation=rotation,
                            repetition=repetition,
                            seed=seed,
                            strategies=(
                                seated_strategies
                            ),
                            game=game,
                            player_id=player_id,
                        )
                    )

    summaries = strategy_summary(
        rows
    )

    write_csv(
        args.output_dir
        / "player_games.csv",
        rows,
    )

    write_csv(
        args.output_dir
        / "strategy_summary.csv",
        summaries,
    )

    print_summary(
        summaries
    )

    print()
    print(
        "Saved:"
    )
    print(
        "  "
        + str(
            args.output_dir
            / "player_games.csv"
        )
    )
    print(
        "  "
        + str(
            args.output_dir
            / "strategy_summary.csv"
        )
    )


if __name__ == "__main__":
    main()
