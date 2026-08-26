from dataclasses import dataclass, field
import random

from catanlab.board import Board, build_random_board
from catanlab.devcards import (
    DevCardDeck,
    build_dev_card_deck,
    update_largest_army,
)
from catanlab.economy import (
    PlayerInventory,
    ResourceBank,
    validate_resource_conservation,
)
from catanlab.longest_road import (
    update_longest_road,
)
from catanlab.simulation import (
    DraftResult,
    PlayerState,
    StrategyOpeningAgent,
    grant_second_settlement_resources,
    run_opening_draft,
)
from catanlab.strategies import StrategyType
from catanlab.turns import (
    AdaptiveStrategyAgent,
    TurnResult,
    run_turn,
)


VICTORY_POINTS_TO_WIN = 10


@dataclass(frozen=True)
class AwardSnapshot:
    turn_number: int
    player_id: int
    largest_army_holder: int | None
    longest_road_holder: int | None
    knights_played: tuple[int, ...]
    road_lengths: tuple[int, ...]


@dataclass
class GameResult:
    winner_id: int | None
    turns_played: int
    players: list[PlayerState]
    inventories: list[PlayerInventory]
    turn_history: list[TurnResult] = field(
        default_factory=list
    )
    award_history: list[AwardSnapshot] = field(
        default_factory=list
    )
    opening_placements: tuple[
        tuple[int, int],
        ...,
    ] = ()
    opening_roads: tuple[
        tuple[
            int,
            tuple[int, int],
        ],
        ...,
    ] = ()
    board_seed: int | None = None
    dev_seed: int | None = None
    bank: ResourceBank | None = None


def _setup_game_with_bank(
    strategies: list[StrategyType],
    board_seed: int | None = None,
    dev_seed: int | None = None,
):
    """
    Create a four-player game and complete setup.
    """

    if len(strategies) != 4:
        raise ValueError(
            "A Catan game requires exactly "
            "four strategies."
        )

    board = build_random_board(
        seed=board_seed
    )

    opening_agents = [
        StrategyOpeningAgent(
            strategy
        )
        for strategy in strategies
    ]

    draft = run_opening_draft(
        board,
        opening_agents,
    )

    inventories = [
        PlayerInventory()
        for _ in draft.players
    ]

    # The finite bank exists before starting
    # resources are distributed so conservation
    # holds from the beginning of the game.
    bank = ResourceBank()

    grant_second_settlement_resources(
        board,
        draft.players,
        inventories,
        bank=bank,
    )

    turn_agents = [
        AdaptiveStrategyAgent(
            strategy
        )
        for strategy in strategies
    ]

    dev_deck = build_dev_card_deck(
        seed=dev_seed
    )

    return (
        board,
        draft,
        inventories,
        turn_agents,
        dev_deck,
        bank,
    )



def setup_game(
    strategies: list[StrategyType],
    board_seed: int | None = None,
    dev_seed: int | None = None,
):
    """
    Public setup API.

    Preserves the original five-value return shape.
    """

    (
        board,
        draft,
        inventories,
        turn_agents,
        dev_deck,
        bank,
    ) = _setup_game_with_bank(
        strategies,
        board_seed=board_seed,
        dev_seed=dev_seed,
    )

    return (
        board,
        draft,
        inventories,
        turn_agents,
        dev_deck,
    )

def winner(
    players: list[PlayerState],
) -> int | None:
    """
    Return the first player at or above the
    victory-point threshold.
    """

    for player in players:
        if (
            player.victory_points
            >= VICTORY_POINTS_TO_WIN
        ):
            return player.player_id

    return None


def run_game(
    strategies: list[StrategyType],
    seed: int | None = None,
    max_turns: int = 2000,
    validate_conservation: bool = False,
) -> GameResult:
    """
    Run one complete four-player Catan game.

    `turns_played` counts individual player turns,
    not full four-player rounds.
    """

    rng = random.Random(
        seed
    )

    board_seed = rng.randrange(
        2**31
    )

    dev_seed = rng.randrange(
        2**31
    )

    (
        board,
        draft,
        inventories,
        agents,
        dev_deck,
        bank,
    ) = _setup_game_with_bank(
        strategies,
        board_seed=board_seed,
        dev_seed=dev_seed,
    )

    players = draft.players

    # One finite resource bank persists for the
    # entire game.

    history = []
    award_history = []

    for turn_index in range(
        max_turns
    ):
        player_id = (
            turn_index
            % len(players)
        )

        roll = (
            rng.randint(1, 6)
            + rng.randint(1, 6)
        )

        result = run_turn(
            board,
            players,
            inventories,
            agents,
            player_id=player_id,
            roll=roll,
            dev_deck=dev_deck,
            rng=rng,
            bank=bank,
        )

        history.append(
            result
        )

        if validate_conservation:
            validate_resource_conservation(
                bank,
                inventories,
            )

        update_largest_army(
            players
        )

        update_longest_road(
            players
        )

        from catanlab.longest_road import (
            longest_road_length,
        )

        largest_army_holder = next(
            (
                p.player_id
                for p in players
                if p.has_largest_army
            ),
            None,
        )

        longest_road_holder = next(
            (
                p.player_id
                for p in players
                if p.has_longest_road
            ),
            None,
        )

        award_history.append(
            AwardSnapshot(
                turn_number=turn_index + 1,
                player_id=player_id,
                largest_army_holder=(
                    largest_army_holder
                ),
                longest_road_holder=(
                    longest_road_holder
                ),
                knights_played=tuple(
                    p.knights_played
                    for p in players
                ),
                road_lengths=tuple(
                    longest_road_length(
                        p,
                        players,
                    )
                    for p in players
                ),
            )
        )

        # Standard Catan victory is checked on the
        # active player's turn.
        active_player = players[
            player_id
        ]

        if (
            active_player.victory_points
            >= VICTORY_POINTS_TO_WIN
        ):
            return GameResult(
                winner_id=player_id,
                turns_played=(
                    turn_index + 1
                ),
                players=players,
                inventories=inventories,
                turn_history=history,
                award_history=award_history,
                opening_placements=tuple(
                    draft.placement_order
                ),
                opening_roads=tuple(
                    draft.road_order
                ),
                board_seed=board_seed,
                dev_seed=dev_seed,
                bank=bank,
            )

    return GameResult(
        winner_id=None,
        turns_played=max_turns,
        players=players,
        inventories=inventories,
        turn_history=history,
        award_history=award_history,
        opening_placements=tuple(
            draft.placement_order
        ),
        opening_roads=tuple(
            draft.road_order
        ),
        board_seed=board_seed,
        dev_seed=dev_seed,
    )
