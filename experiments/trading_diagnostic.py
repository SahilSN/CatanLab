from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from catanlab.game import run_game
from catanlab.strategies import StrategyType
from catanlab.trading import bundle_size


STRATEGIES = [
    StrategyType.FULL_OWS,
    StrategyType.HYBRID_OWS,
    StrategyType.ROAD_BUILDING,
    StrategyType.ROADS_AND_CITIES,
    StrategyType.FIVE_RESOURCE,
    StrategyType.PORT,
]

NUM_SEEDS = 100
MAX_TURNS = 500


def strategy_name(strategy):
    return strategy.value


def main():
    rows = []

    for strategy in STRATEGIES:
        name = strategy_name(
            strategy
        )

        print()
        print(
            f"Trading diagnostic: {name}"
        )

        for seed in range(
            NUM_SEEDS
        ):
            result = run_game(
                [
                    strategy,
                    strategy,
                    strategy,
                    strategy,
                ],
                seed=seed,
                max_turns=MAX_TURNS,
            )

            offers = sum(
                turn.trade_offer_count
                for turn in result.turn_history
            )

            sequences = sum(
                turn.trade_sequence_count
                for turn in result.turn_history
            )

            accepted = sum(
                len(
                    turn.player_trades
                )
                for turn in result.turn_history
            )

            give_cards = []
            receive_cards = []

            mixed_count = 0
            one_for_one = 0
            unequal_count = 0

            partner_pairs = Counter()

            for turn in result.turn_history:
                for trade in (
                    turn.player_trades
                ):
                    give_n = bundle_size(
                        trade.give
                    )

                    receive_n = bundle_size(
                        trade.receive
                    )

                    give_cards.append(
                        give_n
                    )

                    receive_cards.append(
                        receive_n
                    )

                    if (
                        len(trade.give) > 1
                        or len(trade.receive) > 1
                    ):
                        mixed_count += 1

                    if (
                        give_n == 1
                        and receive_n == 1
                    ):
                        one_for_one += 1

                    if (
                        give_n != receive_n
                    ):
                        unequal_count += 1

                    partner_pairs[
                        (
                            trade.proposer_id,
                            trade.recipient_id,
                        )
                    ] += 1

            rows.append(
                {
                    "strategy":
                        name,
                    "seed":
                        seed,
                    "turns":
                        result.turns_played,
                    "offers":
                        offers,
                    "sequences":
                        sequences,
                    "accepted":
                        accepted,
                    "acceptance_rate":
                        (
                            accepted / offers
                            if offers
                            else 0.0
                        ),
                    "avg_give_cards":
                        (
                            mean(give_cards)
                            if give_cards
                            else 0.0
                        ),
                    "avg_receive_cards":
                        (
                            mean(receive_cards)
                            if receive_cards
                            else 0.0
                        ),
                    "mixed_trades":
                        mixed_count,
                    "one_for_one":
                        one_for_one,
                    "unequal_trades":
                        unequal_count,
                    "partner_pairs":
                        dict(
                            partner_pairs
                        ),
                }
            )

            if (
                (seed + 1) % 20
                == 0
            ):
                print(
                    f"  {seed + 1}/"
                    f"{NUM_SEEDS}"
                )

    print()
    print("=" * 120)
    print(
        "DOMESTIC TRADING DIAGNOSTIC"
    )
    print("=" * 120)

    header = (
        f"{'Strategy':20} "
        f"{'Offers':>8} "
        f"{'Seq':>7} "
        f"{'Trades':>8} "
        f"{'Accept%':>8} "
        f"{'Give':>7} "
        f"{'Recv':>7} "
        f"{'Mixed%':>8} "
        f"{'1:1%':>7} "
        f"{'Unequal%':>9}"
    )

    print(header)
    print(
        "-" * len(header)
    )

    for strategy in STRATEGIES:
        name = strategy_name(
            strategy
        )

        subset = [
            row
            for row in rows
            if row["strategy"] == name
        ]

        total_offers = sum(
            row["offers"]
            for row in subset
        )

        total_sequences = sum(
            row["sequences"]
            for row in subset
        )

        total_trades = sum(
            row["accepted"]
            for row in subset
        )

        total_mixed = sum(
            row["mixed_trades"]
            for row in subset
        )

        total_one_for_one = sum(
            row["one_for_one"]
            for row in subset
        )

        total_unequal = sum(
            row["unequal_trades"]
            for row in subset
        )

        all_give = []
        all_recv = []

        for row in subset:
            if row["accepted"] > 0:
                all_give.extend(
                    [
                        row["avg_give_cards"]
                    ]
                    * row["accepted"]
                )

                all_recv.extend(
                    [
                        row["avg_receive_cards"]
                    ]
                    * row["accepted"]
                )

        acceptance_rate = (
            total_trades
            / total_offers
            if total_offers
            else 0.0
        )

        mixed_rate = (
            total_mixed
            / total_trades
            if total_trades
            else 0.0
        )

        one_for_one_rate = (
            total_one_for_one
            / total_trades
            if total_trades
            else 0.0
        )

        unequal_rate = (
            total_unequal
            / total_trades
            if total_trades
            else 0.0
        )

        print(
            f"{name:20} "
            f"{total_offers / NUM_SEEDS:8.2f} "
            f"{total_sequences / NUM_SEEDS:7.2f} "
            f"{total_trades / NUM_SEEDS:8.2f} "
            f"{100 * acceptance_rate:7.1f}% "
            f"{mean(all_give) if all_give else 0.0:7.2f} "
            f"{mean(all_recv) if all_recv else 0.0:7.2f} "
            f"{100 * mixed_rate:7.1f}% "
            f"{100 * one_for_one_rate:6.1f}% "
            f"{100 * unequal_rate:8.1f}%"
        )


if __name__ == "__main__":
    main()
