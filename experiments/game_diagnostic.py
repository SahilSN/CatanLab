from collections import Counter, defaultdict
from statistics import mean

from catanlab.game import run_game
from catanlab.strategies import StrategyType


STRATEGIES = [
    StrategyType.FULL_OWS,
    StrategyType.ROAD_BUILDING,
    StrategyType.FIVE_RESOURCE,
    StrategyType.HYBRID_OWS,
]


def strategy_name(strategy):
    return strategy.value


def main():
    num_games = 20

    wins = Counter()
    turns = []

    stats = defaultdict(
        lambda: {
            "games": 0,
            "vp": [],
            "roads": [],
            "settlements": [],
            "cities": [],
            "knights": [],
            "largest_army": 0,
            "longest_road": 0,
            "dev_vp": [],
        }
    )

    for seed in range(num_games):
        result = run_game(
            STRATEGIES,
            seed=seed,
            max_turns=500,
        )

        turns.append(
            result.turns_played
        )

        if result.winner_id is not None:
            winner_strategy = STRATEGIES[
                result.winner_id
            ]
            wins[
                strategy_name(
                    winner_strategy
                )
            ] += 1

        print(
            f"Game {seed + 1:02d}/{num_games} | "
            f"seed={seed} | "
            f"winner="
            f"{result.winner_id} | "
            f"turns="
            f"{result.turns_played}"
        )

        for player in result.players:
            strategy = STRATEGIES[
                player.player_id
            ]
            name = strategy_name(
                strategy
            )

            entry = stats[name]

            entry["games"] += 1

            entry["vp"].append(
                player.victory_points
            )

            entry["roads"].append(
                len(player.roads)
            )

            entry[
                "settlements"
            ].append(
                len(
                    player.settlements
                )
            )

            entry["cities"].append(
                len(player.cities)
            )

            entry["knights"].append(
                player.knights_played
            )

            entry[
                "largest_army"
            ] += int(
                player.has_largest_army
            )

            entry[
                "longest_road"
            ] += int(
                player.has_longest_road
            )

            entry["dev_vp"].append(
                sum(
                    card
                    == "victory_point"
                    for card
                    in player.dev_cards
                )
            )

    print()
    print("=" * 80)
    print("20-GAME DIAGNOSTIC SUMMARY")
    print("=" * 80)

    print()
    print(
        f"Average game length: "
        f"{mean(turns):.1f} turns"
    )

    print()

    header = (
        f"{'Strategy':20} "
        f"{'Wins':>5} "
        f"{'AvgVP':>7} "
        f"{'Roads':>7} "
        f"{'Sett':>7} "
        f"{'Cities':>7} "
        f"{'Knights':>8} "
        f"{'LA':>4} "
        f"{'LR':>4} "
        f"{'DevVP':>7}"
    )

    print(header)
    print("-" * len(header))

    for strategy in STRATEGIES:
        name = strategy_name(
            strategy
        )
        entry = stats[name]

        print(
            f"{name:20} "
            f"{wins[name]:5d} "
            f"{mean(entry['vp']):7.2f} "
            f"{mean(entry['roads']):7.2f} "
            f"{mean(entry['settlements']):7.2f} "
            f"{mean(entry['cities']):7.2f} "
            f"{mean(entry['knights']):8.2f} "
            f"{entry['largest_army']:4d} "
            f"{entry['longest_road']:4d} "
            f"{mean(entry['dev_vp']):7.2f}"
        )


if __name__ == "__main__":
    main()
