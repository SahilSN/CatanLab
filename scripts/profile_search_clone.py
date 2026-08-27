from __future__ import annotations

import argparse
import time

from catanlab.game import (
    _setup_game_with_bank,
)
from catanlab.search import (
    disable_clone_profile,
    get_clone_profile,
    reset_clone_profile,
)
from catanlab.search_agent import (
    OneStepLookaheadAgent,
)
from catanlab.strategies import StrategyType


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    strategies = [
        StrategyType.FIVE_RESOURCE,
        StrategyType.HYBRID_OWS,
        StrategyType.FULL_OWS,
        StrategyType.PORT,
    ]

    (
        board,
        draft,
        inventories,
        agents,
        dev_deck,
        bank,
    ) = _setup_game_with_bank(
        strategies,
        board_seed=12345,
        dev_seed=67890,
    )

    player = draft.players[0]

    search_agent = OneStepLookaheadAgent(
        StrategyType.FIVE_RESOURCE,
        search_depth=args.depth,
        use_transposition_cache=False,
    )

    reset_clone_profile(
        enabled=True
    )

    start = time.perf_counter()

    for _ in range(args.iterations):
        search_agent.evaluate_actions(
            board,
            draft.players,
            player,
            inventories[0],
            dev_deck,
            bank,
        )

    total_seconds = (
        time.perf_counter()
        - start
    )

    clone_calls, clone_seconds = (
        get_clone_profile()
    )

    disable_clone_profile()

    print(
        "=== SEARCH CLONE PROFILE ==="
    )

    print(
        f"depth: {args.depth}"
    )

    print(
        f"iterations: {args.iterations}"
    )

    print(
        f"total search time: "
        f"{total_seconds:.6f} s"
    )

    print(
        f"clone calls: {clone_calls}"
    )

    print(
        f"clone time: "
        f"{clone_seconds:.6f} s"
    )

    if clone_calls:
        print(
            f"avg clone time: "
            f"{1e6 * clone_seconds / clone_calls:.2f} us"
        )

    if total_seconds > 0:
        print(
            f"clone share of search time: "
            f"{100 * clone_seconds / total_seconds:.2f}%"
        )


if __name__ == "__main__":
    main()
