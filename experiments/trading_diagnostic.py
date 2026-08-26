from collections import Counter
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
        name = strategy_name(strategy)

        print()
        print(
            f"Trading diagnostic: {name}"
        )

        for seed in range(NUM_SEEDS):
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

            offers = 0
            sequences = 0
            accepted = 0

            give_cards = []
            receive_cards = []

            mixed_count = 0
            one_for_one = 0
            unequal_count = 0

            turns_with_negotiation = 0
            turns_with_trade = 0
            multi_sequence_turns = 0

            sequence_distribution = Counter()

            for turn in result.turn_history:
                offers += (
                    turn.trade_offer_count
                )

                sequences += (
                    turn.trade_sequence_count
                )

                turn_trades = len(
                    turn.player_trades
                )

                accepted += turn_trades

                sequence_distribution[
                    turn.trade_sequence_count
                ] += 1

                if (
                    turn.trade_sequence_count
                    > 0
                ):
                    turns_with_negotiation += 1

                if (
                    turn.trade_sequence_count
                    > 1
                ):
                    multi_sequence_turns += 1

                if turn_trades > 0:
                    turns_with_trade += 1

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

                    if give_n != receive_n:
                        unequal_count += 1

            turns = len(
                result.turn_history
            )

            rows.append(
                {
                    "strategy": name,
                    "seed": seed,
                    "turns": turns,
                    "offers": offers,
                    "sequences": sequences,
                    "accepted": accepted,
                    "turns_with_negotiation":
                        turns_with_negotiation,
                    "turns_with_trade":
                        turns_with_trade,
                    "multi_sequence_turns":
                        multi_sequence_turns,
                    "sequence_distribution":
                        sequence_distribution,
                    "give_cards":
                        give_cards,
                    "receive_cards":
                        receive_cards,
                    "mixed_trades":
                        mixed_count,
                    "one_for_one":
                        one_for_one,
                    "unequal_trades":
                        unequal_count,
                }
            )

            if (seed + 1) % 20 == 0:
                print(
                    f"  {seed + 1}/"
                    f"{NUM_SEEDS}"
                )

    print()
    print("=" * 154)
    print(
        "DOMESTIC TRADING DIAGNOSTIC"
    )
    print("=" * 154)

    header = (
        f"{'Strategy':20} "
        f"{'Turns':>7} "
        f"{'Offers':>8} "
        f"{'Seq':>7} "
        f"{'Trades':>8} "
        f"{'Accept%':>8} "
        f"{'Seq/T':>7} "
        f"{'Trade/T':>8} "
        f"{'NegTurn%':>9} "
        f"{'TradeTurn%':>11} "
        f"{'MultiSeq%':>10} "
        f"{'Give':>6} "
        f"{'Recv':>6} "
        f"{'Mixed%':>8} "
        f"{'1:1%':>7}"
    )

    print(header)
    print("-" * len(header))

    for strategy in STRATEGIES:
        name = strategy_name(strategy)

        subset = [
            row
            for row in rows
            if row["strategy"] == name
        ]

        total_turns = sum(
            row["turns"]
            for row in subset
        )

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

        total_negotiating_turns = sum(
            row["turns_with_negotiation"]
            for row in subset
        )

        total_trade_turns = sum(
            row["turns_with_trade"]
            for row in subset
        )

        total_multi_sequence_turns = sum(
            row["multi_sequence_turns"]
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

        all_give = [
            value
            for row in subset
            for value in row["give_cards"]
        ]

        all_recv = [
            value
            for row in subset
            for value in row[
                "receive_cards"
            ]
        ]

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

        multi_sequence_rate = (
            total_multi_sequence_turns
            / total_negotiating_turns
            if total_negotiating_turns
            else 0.0
        )

        print(
            f"{name:20} "
            f"{total_turns / NUM_SEEDS:7.1f} "
            f"{total_offers / NUM_SEEDS:8.2f} "
            f"{total_sequences / NUM_SEEDS:7.2f} "
            f"{total_trades / NUM_SEEDS:8.2f} "
            f"{100 * acceptance_rate:7.1f}% "
            f"{total_sequences / total_turns:7.3f} "
            f"{total_trades / total_turns:8.3f} "
            f"{100 * total_negotiating_turns / total_turns:8.1f}% "
            f"{100 * total_trade_turns / total_turns:10.1f}% "
            f"{100 * multi_sequence_rate:9.1f}% "
            f"{mean(all_give) if all_give else 0.0:6.2f} "
            f"{mean(all_recv) if all_recv else 0.0:6.2f} "
            f"{100 * mixed_rate:7.1f}% "
            f"{100 * one_for_one_rate:6.1f}%"
        )

    print()
    print("=" * 80)
    print(
        "NEGOTIATION SEQUENCES PER TURN"
    )
    print("=" * 80)

    print(
        f"{'Strategy':20} "
        f"{'0 seq':>10} "
        f"{'1 seq':>10} "
        f"{'2 seq':>10} "
        f"{'3 seq':>10}"
    )

    print("-" * 65)

    for strategy in STRATEGIES:
        name = strategy_name(strategy)

        subset = [
            row
            for row in rows
            if row["strategy"] == name
        ]

        dist = Counter()

        for row in subset:
            dist.update(
                row[
                    "sequence_distribution"
                ]
            )

        total = sum(
            dist.values()
        )

        def pct(n):
            if total == 0:
                return 0.0
            return (
                100.0
                * dist[n]
                / total
            )

        print(
            f"{name:20} "
            f"{pct(0):9.1f}% "
            f"{pct(1):9.1f}% "
            f"{pct(2):9.1f}% "
            f"{pct(3):9.1f}%"
        )


if __name__ == "__main__":
    main()
