import random

from dataclasses import dataclass
from enum import Enum

from catanlab.board import Board
from catanlab.building import (
    build_city,
    build_road,
    build_settlement,
    can_build_connected_settlement,
    can_build_road,
)
from catanlab.devcards import (
    DevCardDeck,
    buy_dev_card,
)
from catanlab.economy import (
    BuildType,
    PlayerInventory,
    ResourceBank,
    produce_for_roll,
)
from catanlab.resources import Resource
from catanlab.simulation import PlayerState
from catanlab.trading import (
    TradeOffer,
    execute_player_trade,
    validate_trade_offer,
)


class ActionType(str, Enum):
    BUILD_CITY = "build_city"
    BUILD_SETTLEMENT = "build_settlement"
    BUILD_ROAD = "build_road"
    BUY_DEV_CARD = "buy_dev_card"
    MARITIME_TRADE = "maritime_trade"
    PASS = "pass"


@dataclass(frozen=True)
class TurnAction:
    action_type: ActionType
    vertex_id: int | None = None
    edge: tuple[int, int] | None = None
    give_resource: Resource | None = None
    receive_resource: Resource | None = None


@dataclass(frozen=True)
class TurnResult:
    player_id: int
    roll: int
    action: TurnAction
    actions: tuple[
        TurnAction,
        ...,
    ]
    discards: dict[
        int,
        list,
    ]
    player_trades: tuple[
        TradeOffer,
        ...,
    ] = ()
    trade_offer_count: int = 0
    trade_sequence_count: int = 0


def legal_city_vertices(
    player: PlayerState,
) -> list[int]:
    from catanlab.building import (
        MAX_CITIES,
    )

    if len(player.cities) >= MAX_CITIES:
        return []

    return list(
        player.settlements
    )


def legal_settlement_vertices(
    board: Board,
    players: list[PlayerState],
    player: PlayerState,
) -> list[int]:
    return [
        vertex.id
        for vertex in board.vertices
        if can_build_connected_settlement(
            board,
            players,
            player,
            vertex.id,
        )
    ]


def legal_road_edges(
    board: Board,
    players: list[PlayerState],
    player: PlayerState,
) -> list[tuple[int, int]]:
    legal: list[
        tuple[int, int]
    ] = []

    for edge in board.edges:
        candidate = tuple(
            sorted(
                (
                    edge.vertex_a,
                    edge.vertex_b,
                )
            )
        )

        if can_build_road(
            board,
            players,
            player,
            candidate[0],
            candidate[1],
        ):
            legal.append(
                candidate
            )

    return legal


class TurnAgent:
    def choose_action(
        self,
        board: Board,
        players: list[PlayerState],
        player: PlayerState,
        inventory: PlayerInventory,
        dev_deck: DevCardDeck | None = None,
    ) -> TurnAction:
        raise NotImplementedError


    def choose_discards(
        self,
        player,
        inventory: PlayerInventory,
        count: int,
    ) -> list[Resource]:
        """
        Default deterministic discard policy for
        simple agents.
        """

        discarded = []

        for resource in (
            Resource.WOOD,
            Resource.BRICK,
            Resource.SHEEP,
            Resource.WHEAT,
            Resource.ORE,
        ):
            take = min(
                count - len(discarded),
                inventory.count(
                    resource
                ),
            )

            discarded.extend(
                [resource] * take
            )

            if len(discarded) >= count:
                break

        return discarded

    def propose_player_trade(
        self,
        board,
        players,
        player,
        inventories,
        excluded_recipients=None,
        agents=None,
    ):
        """
        Return a TradeOffer or None.

        Generic agents do not initiate domestic
        trades unless they override this method.
        """
        return None

    def evaluate_player_trade(
        self,
        board,
        players,
        player,
        inventories,
        offer,
    ) -> bool:
        """
        Return whether this player accepts a
        domestic trade offer.

        Generic agents reject by default.
        """
        return False

    def counter_player_trade(
        self,
        board,
        players,
        player,
        inventories,
        offer,
        attempted_offers=None,
    ):
        """
        Return a counteroffer or None.

        Generic agents do not negotiate.
        """
        return None

    def _resource_trade_value(
        self,
        resource: Resource,
        inventory: PlayerInventory,
    ) -> float:
        """
        Estimate the current strategic value of one
        resource card.

        Strategy preference supplies the baseline,
        while current build deficits add urgency.
        """

        from catanlab.economy import (
            BUILD_COSTS,
        )

        value = self.profile.resource_weights.get(
            resource,
            0.0,
        )

        preferred_builds = (
            BuildType.CITY,
            BuildType.SETTLEMENT,
            BuildType.DEV_CARD,
            BuildType.ROAD,
        )

        for build_type in preferred_builds:
            required = BUILD_COSTS[
                build_type
            ].get(
                resource,
                0,
            )

            if (
                required > 0
                and inventory.count(
                    resource
                ) < required
            ):
                value += 0.5

        return value

    def choose_dev_card_play(
        self,
        board,
        players,
        player,
        inventories,
        phase,
    ):
        from catanlab.devcard_policy import (
            DevCardDecision,
        )

        return DevCardDecision(
            card=None,
            utility=0.0,
        )


class GreedyBuildAgent(TurnAgent):
    """
    Simple baseline build priority:

    city
    settlement
    road
    pass
    """

    def choose_action(
        self,
        board: Board,
        players: list[PlayerState],
        player: PlayerState,
        inventory: PlayerInventory,
        dev_deck: DevCardDeck | None = None,
    ) -> TurnAction:
        if inventory.can_afford(
            BuildType.CITY
        ):
            cities = legal_city_vertices(
                player
            )

            if cities:
                return TurnAction(
                    action_type=(
                        ActionType.BUILD_CITY
                    ),
                    vertex_id=cities[0],
                )

        if inventory.can_afford(
            BuildType.SETTLEMENT
        ):
            settlements = (
                legal_settlement_vertices(
                    board,
                    players,
                    player,
                )
            )

            if settlements:
                return TurnAction(
                    action_type=(
                        ActionType.BUILD_SETTLEMENT
                    ),
                    vertex_id=settlements[0],
                )

        if inventory.can_afford(
            BuildType.ROAD
        ):
            roads = legal_road_edges(
                board,
                players,
                player,
            )

            if roads:
                return TurnAction(
                    action_type=(
                        ActionType.BUILD_ROAD
                    ),
                    edge=roads[0],
                )

        return TurnAction(
            action_type=ActionType.PASS
        )


def execute_action(
    board: Board,
    players: list[PlayerState],
    player: PlayerState,
    inventory: PlayerInventory,
    action: TurnAction,
    dev_deck: DevCardDeck | None = None,
    bank: ResourceBank | None = None,
) -> None:
    if (
        action.action_type
        == ActionType.PASS
    ):
        return

    if (
        action.action_type
        == ActionType.MARITIME_TRADE
    ):
        from catanlab.ports import (
            maritime_trade,
        )

        if (
            action.give_resource is None
            or action.receive_resource is None
        ):
            raise ValueError(
                "Maritime trade requires "
                "give_resource and receive_resource."
            )

        maritime_trade(
            board,
            player,
            inventory,
            give=action.give_resource,
            receive=action.receive_resource,
            bank=bank,
        )

        return

    if (
        action.action_type
        == ActionType.BUILD_CITY
    ):
        if action.vertex_id is None:
            raise ValueError(
                "City action requires vertex_id."
            )

        build_city(
            player,
            inventory,
            action.vertex_id,
            bank=bank,
        )

        return

    if (
        action.action_type
        == ActionType.BUILD_SETTLEMENT
    ):
        if action.vertex_id is None:
            raise ValueError(
                "Settlement action requires "
                "vertex_id."
            )

        if not can_build_connected_settlement(
            board,
            players,
            player,
            action.vertex_id,
        ):
            raise ValueError(
                "Settlement placement is not legal."
            )

        build_settlement(
            board,
            players,
            player,
            inventory,
            action.vertex_id,
            bank=bank,
        )

        return

    if (
        action.action_type
        == ActionType.BUILD_ROAD
    ):
        if action.edge is None:
            raise ValueError(
                "Road action requires an edge."
            )

        a, b = action.edge

        build_road(
            board,
            players,
            player,
            inventory,
            a,
            b,
            bank=bank,
        )

        return

    if (
        action.action_type
        == ActionType.BUY_DEV_CARD
    ):
        if dev_deck is None:
            raise ValueError(
                "Development-card action requires a deck."
            )

        buy_dev_card(
            player,
            inventory,
            dev_deck,
            bank=bank,
        )

        return

    raise ValueError(
        f"Unknown action: "
        f"{action.action_type}"
    )


def _year_of_plenty_resources(
    inventory: PlayerInventory,
    bank: ResourceBank | None = None,
) -> tuple[Resource, Resource] | None:
    """
    Pick two useful resources for Year of Plenty.

    Prefer deficits toward city, settlement,
    development card, then road.

    When a finite bank is supplied, only cards
    actually available from the bank are eligible.
    """

    from catanlab.economy import (
        BUILD_COSTS,
    )
    from catanlab.resources import (
        Resource,
    )

    priorities = (
        BuildType.CITY,
        BuildType.SETTLEMENT,
        BuildType.DEV_CARD,
        BuildType.ROAD,
    )

    resources = (
        Resource.WOOD,
        Resource.BRICK,
        Resource.SHEEP,
        Resource.WHEAT,
        Resource.ORE,
    )

    # Build an ordered preference list. Repeating a
    # resource represents wanting multiple copies.
    preferred: list[Resource] = []

    for build_type in priorities:
        cost = BUILD_COSTS[
            build_type
        ]

        missing: list[Resource] = []

        for resource, required in (
            cost.items()
        ):
            deficit = max(
                0,
                required
                - inventory.count(
                    resource
                ),
            )

            missing.extend(
                [resource] * deficit
            )

        if missing:
            preferred.extend(
                missing
            )

    # Fall back to all resource types so YOP can
    # still be played when no immediate build
    # deficit exists.
    preferred.extend(
        resources
    )

    chosen: list[Resource] = []
    reserved = {}

    for resource in preferred:
        already_reserved = reserved.get(
            resource,
            0,
        )

        if bank is not None:
            if not bank.can_supply(
                resource,
                already_reserved + 1,
            ):
                continue

        chosen.append(
            resource
        )

        reserved[resource] = (
            already_reserved + 1
        )

        if len(chosen) == 2:
            return (
                chosen[0],
                chosen[1],
            )

    # With a finite bank, fewer than two resource
    # cards may exist in total. Do not burn the
    # development card if its effect cannot be
    # completed.
    return None

def _road_building_edges(
    board: Board,
    players: list[PlayerState],
    player: PlayerState,
):
    """
    Choose up to two legal free roads.

    Prefer a two-road sequence that maximizes the
    resulting Longest Road length. If only one road
    can legally be placed, return the best single
    road instead of making Road Building unusable.
    """

    from catanlab.longest_road import (
        longest_road_length,
    )

    first_edges = legal_road_edges(
        board,
        players,
        player,
    )

    if not first_edges:
        return None

    best_pair = None
    best_pair_length = -1

    best_single = None
    best_single_length = -1

    for first in first_edges:
        player.roads.append(
            first
        )

        try:
            single_length = (
                longest_road_length(
                    player,
                    players,
                )
            )

            if (
                single_length
                > best_single_length
                or (
                    single_length
                    == best_single_length
                    and (
                        best_single is None
                        or first < best_single
                    )
                )
            ):
                best_single_length = (
                    single_length
                )
                best_single = first

            second_edges = legal_road_edges(
                board,
                players,
                player,
            )

            for second in second_edges:
                player.roads.append(
                    second
                )

                try:
                    length = (
                        longest_road_length(
                            player,
                            players,
                        )
                    )
                finally:
                    player.roads.pop()

                pair = (
                    first,
                    second,
                )

                if (
                    length > best_pair_length
                    or (
                        length
                        == best_pair_length
                        and (
                            best_pair is None
                            or pair < best_pair
                        )
                    )
                ):
                    best_pair_length = length
                    best_pair = pair

        finally:
            player.roads.pop()

    if best_pair is not None:
        return best_pair

    if best_single is not None:
        return (
            best_single,
        )

    return None


def _knight_target_tile(
    board: Board,
    players: list[PlayerState],
    inventories: list[PlayerInventory],
    player: PlayerState,
):
    """
    Choose a strategic robber destination.

    The robber primarily aims to suppress valuable
    opponent production while avoiding the active
    player's own production.

    Additional preference is given to:
    - stronger opponents,
    - tiles with a stealable adjacent opponent,
    - cities over settlements.

    The same logic is used for both rolling a 7 and
    playing a Knight.
    """

    from catanlab.dice import (
        production_weight,
    )

    candidates = [
        tile
        for tile in board.tiles
        if tile.id != board.robber_tile_id
    ]

    if not candidates:
        return None

    def buildings_on_tile(
        other_player,
        tile_id,
    ):
        settlement_count = sum(
            1
            for vertex_id
            in other_player.settlements
            if tile_id
            in board.vertices[
                vertex_id
            ].adjacent_tiles
        )

        city_count = sum(
            1
            for vertex_id
            in other_player.cities
            if tile_id
            in board.vertices[
                vertex_id
            ].adjacent_tiles
        )

        return (
            settlement_count,
            city_count,
        )

    def score(tile):
        number_weight = (
            production_weight(
                tile.number
            )
            if tile.number is not None
            else 0.0
        )

        opponent_block_score = 0.0
        self_block_score = 0.0

        steal_score = 0.0
        strongest_victim_vp = 0

        for other in players:
            (
                settlement_count,
                city_count,
            ) = buildings_on_tile(
                other,
                tile.id,
            )

            # Cities lose twice as much production
            # from a blocked tile as settlements.
            production_units = (
                settlement_count
                + 2 * city_count
            )

            if production_units == 0:
                continue

            blocked_value = (
                production_units
                * number_weight
            )

            if (
                other.player_id
                == player.player_id
            ):
                self_block_score += (
                    blocked_value
                )
                continue

            # Players closer to winning are more
            # important robber targets.
            threat_multiplier = (
                1.0
                + 0.15
                * other.public_victory_points
            )

            opponent_block_score += (
                blocked_value
                * threat_multiplier
            )

            # Hand size should not scale the robber
            # value linearly: only one random card can
            # actually be stolen. Having at least one
            # card creates the steal opportunity.
            if (
                inventories[
                    other.player_id
                ].total()
                > 0
            ):
                victim_vp = (
                    other.public_victory_points
                )

                strongest_victim_vp = max(
                    strongest_victim_vp,
                    victim_vp,
                )

                steal_score = max(
                    steal_score,
                    1.0
                    + 0.10
                    * min(
                        inventories[
                            other.player_id
                        ].total(),
                        7,
                    )
                    + 0.15
                    * victim_vp,
                )

        # Blocking your own production should be
        # meaningfully worse than an equally valuable
        # amount of opponent disruption.
        total_score = (
            opponent_block_score
            - 1.50
            * self_block_score
            + 0.75
            * steal_score
        )

        return (
            total_score,
            opponent_block_score,
            -self_block_score,
            steal_score,
            strongest_victim_vp,
            number_weight,
            -tile.id,
        )

    return max(
        candidates,
        key=score,
    ).id




def _choose_robber_victim(
    board: Board,
    players: list[PlayerState],
    inventories: list[PlayerInventory],
    thief: PlayerState,
):
    """
    Choose which adjacent opponent to rob.

    Prefer:
    - opponents closer to victory,
    - opponents with larger hands.

    Hand size matters only moderately because exactly
    one random resource will be stolen.
    """

    from catanlab.devcards import (
        players_adjacent_to_tile,
    )

    if board.robber_tile_id is None:
        return None

    adjacent = players_adjacent_to_tile(
        board,
        players,
        board.robber_tile_id,
        exclude_player_id=(
            thief.player_id
        ),
    )

    eligible = [
        victim_id
        for victim_id in adjacent
        if inventories[
            victim_id
        ].total() > 0
    ]

    if not eligible:
        return None

    def victim_score(victim_id):
        victim = players[
            victim_id
        ]

        hand_size = inventories[
            victim_id
        ].total()

        return (
            victim.public_victory_points,
            min(
                hand_size,
                7,
            ),
            -victim_id,
        )

    return max(
        eligible,
        key=victim_score,
    )

def _execute_dev_card_decision(
    board: Board,
    players: list[PlayerState],
    inventories: list[PlayerInventory],
    player: PlayerState,
    decision,
    rng: random.Random,
    bank: ResourceBank | None = None,
) -> bool:
    """
    Execute one action development-card decision.

    Returns True if a card was actually played.
    """

    from catanlab.devcards import (
        DevCardType,
        play_knight_and_move_robber,
        play_monopoly,
        play_road_building,
        play_year_of_plenty,
        rob_adjacent_player,
        update_largest_army,
    )

    if decision.card is None:
        return False

    if decision.card == DevCardType.MONOPOLY:
        if decision.resource is None:
            return False

        play_monopoly(
            player,
            inventories,
            decision.resource,
        )

        return True

    if (
        decision.card
        == DevCardType.YEAR_OF_PLENTY
    ):
        resources = (
            _year_of_plenty_resources(
                inventories[
                    player.player_id
                ],
                bank=bank,
            )
        )

        if resources is None:
            return False

        resource_a, resource_b = (
            resources
        )

        play_year_of_plenty(
            player,
            inventories[
                player.player_id
            ],
            resource_a,
            resource_b,
            bank=bank,
        )

        return True

    if (
        decision.card
        == DevCardType.ROAD_BUILDING
    ):
        edges = _road_building_edges(
            board,
            players,
            player,
        )

        if edges is None:
            return False

        play_road_building(
            player,
            board,
            players,
            edges[0],
            (
                edges[1]
                if len(edges) > 1
                else None
            ),
        )

        return True

    if decision.card == DevCardType.KNIGHT:
        tile_id = _knight_target_tile(
            board,
            players,
            inventories,
            player,
        )

        if tile_id is None:
            return False

        play_knight_and_move_robber(
            player,
            board,
            tile_id,
        )

        update_largest_army(
            players
        )

        victim_id = (
            _choose_robber_victim(
                board,
                players,
                inventories,
                player,
            )
        )

        rob_adjacent_player(
            board,
            players,
            inventories,
            thief_id=player.player_id,
            rng=rng,
            victim_id=victim_id,
        )

        return True

    return False


def _choose_normal_action(
    agent: TurnAgent,
    board: Board,
    players: list[PlayerState],
    player: PlayerState,
    inventory: PlayerInventory,
    dev_deck: DevCardDeck | None,
    bank: ResourceBank | None = None,
) -> TurnAction:
    """
    Call an agent's normal-action policy.

    Newer agents may accept dev_deck so they can
    avoid trying to buy from an empty deck.

    Older/custom agents remain supported.
    """

    import inspect

    parameters = inspect.signature(
        agent.choose_action
    ).parameters

    kwargs = {}

    if "dev_deck" in parameters:
        kwargs[
            "dev_deck"
        ] = dev_deck

    if "bank" in parameters:
        kwargs[
            "bank"
        ] = bank

    return agent.choose_action(
        board,
        players,
        player,
        inventory,
        **kwargs,
    )


MAX_TRADE_OFFERS_PER_SEQUENCE = 4
MAX_TRADE_OFFERS_PER_TURN = 12
MAX_TRADE_SEQUENCES_PER_TURN = 3


def _trade_state_signature(
    player,
    inventory,
):
    """
    Capture the parts of the active player's state
    that matter for whether reopening negotiation
    makes sense.

    A failed negotiation should not immediately be
    repeated unless this state changes.
    """

    resources = (
        Resource.WOOD,
        Resource.BRICK,
        Resource.SHEEP,
        Resource.WHEAT,
        Resource.ORE,
    )

    return (
        tuple(
            inventory.count(
                resource
            )
            for resource in resources
        ),
        tuple(
            sorted(
                player.settlements
            )
        ),
        tuple(
            sorted(
                player.cities
            )
        ),
        tuple(
            sorted(
                player.roads
            )
        ),
        len(
            player.dev_cards
        ),
    )


def _run_trade_sequence(
    board,
    players,
    inventories,
    agents,
    initial_offer,
    remaining_offer_budget,
):
    """
    Run one bounded negotiation between two players.

    Returns:
        accepted_trade,
        offers_made
    """

    attempted_offers = set()

    offer = initial_offer

    offers_made = 0

    sequence_limit = min(
        MAX_TRADE_OFFERS_PER_SEQUENCE,
        remaining_offer_budget,
    )

    for _ in range(
        sequence_limit
    ):
        if offer is None:
            break

        if offer in attempted_offers:
            break

        if not validate_trade_offer(
            offer,
            inventories,
        ):
            break

        attempted_offers.add(
            offer
        )

        offers_made += 1

        recipient_id = (
            offer.recipient_id
        )

        recipient = players[
            recipient_id
        ]

        recipient_agent = agents[
            recipient_id
        ]

        if recipient_agent.evaluate_player_trade(
            board,
            players,
            recipient,
            inventories,
            offer,
        ):
            execute_player_trade(
                offer,
                inventories,
            )

            return (
                offer,
                offers_made,
            )

        counteroffer = (
            recipient_agent.counter_player_trade(
                board,
                players,
                recipient,
                inventories,
                offer,
                attempted_offers=(
                    attempted_offers
                ),
            )
        )

        if counteroffer is None:
            break

        # A counteroffer must remain within the same
        # two-player negotiation. Only the recipient
        # of the current offer may counter, and the
        # counter must go back to the current proposer.
        if (
            counteroffer.proposer_id
            != offer.recipient_id
            or counteroffer.recipient_id
            != offer.proposer_id
        ):
            break

        offer = counteroffer

    return (
        None,
        offers_made,
    )


def _run_one_domestic_trade_sequence(
    board,
    players,
    inventories,
    agents,
    player,
    agent,
    remaining_offer_budget,
):
    """
    Give the active player one opportunity to start
    a domestic negotiation sequence.

    The sequence may contain up to four offers, but
    is also constrained by the remaining whole-turn
    offer budget.

    Returns:
        accepted_trade,
        offers_made
    """

    if remaining_offer_budget <= 0:
        return (
            None,
            0,
        )

    try:
        initial_offer = (
            agent.propose_player_trade(
                board,
                players,
                player,
                inventories,
                agents=agents,
            )
        )
    except TypeError:
        # Backward compatibility for simple custom
        # test/baseline agents implementing the older
        # proposal signature.
        initial_offer = (
            agent.propose_player_trade(
                board,
                players,
                player,
                inventories,
            )
        )

    if initial_offer is None:
        return (
            None,
            0,
        )

    if (
        initial_offer.proposer_id
        != player.player_id
    ):
        return (
            None,
            0,
        )

    return _run_trade_sequence(
        board,
        players,
        inventories,
        agents,
        initial_offer,
        remaining_offer_budget,
    )


def run_turn(
    board: Board,
    players: list[PlayerState],
    inventories: list[PlayerInventory],
    agents: list[TurnAgent],
    player_id: int,
    roll: int,
    dev_deck: DevCardDeck | None = None,
    rng: random.Random | None = None,
    bank: ResourceBank | None = None,
) -> TurnResult:
    """
    Execute one simplified Catan turn.

    Action development cards may be played either
    before or after the dice roll, but at most one
    action dev card may be played during the turn.
    """

    from catanlab.devcard_policy import (
        DevCardPhase,
    )

    if rng is None:
        rng = random.Random()

    player = players[
        player_id
    ]

    inventory = inventories[
        player_id
    ]

    agent = agents[
        player_id
    ]

    # Cards bought during this player's previous
    # turn become playable at the beginning of the
    # player's new turn.
    player.new_dev_cards.clear()

    played_action_dev_card = False

    # Turn-result bookkeeping must exist before any
    # possible victory exit. A player may already have
    # a winning score at the start of their turn, or
    # may reach it with a pre-roll development card.
    discards: dict[int, list] = {}

    player_trades: list[
        TradeOffer
    ] = []

    trade_offer_count = 0
    trade_sequence_count = 0

    actions: list[TurnAction] = []

    def active_player_has_won() -> bool:
        """
        Recompute public awards and determine whether
        the active player has reached the standard
        10-point victory threshold.

        Victory in Catan is recognized on the active
        player's own turn, so only this player is
        checked here.
        """

        from catanlab.devcards import (
            update_largest_army,
        )
        from catanlab.longest_road import (
            update_longest_road,
        )

        update_largest_army(
            players
        )

        update_longest_road(
            players
        )

        return (
            player.victory_points
            >= 10
        )

    def try_action_dev_card(
        phase,
    ) -> bool:
        """
        Give the active player one development-card
        opportunity.

        Once an action development card has been
        played this turn, later opportunities are
        automatically suppressed.
        """

        nonlocal played_action_dev_card

        if played_action_dev_card:
            return False

        decision = agent.choose_dev_card_play(
            board,
            players,
            player,
            inventories,
            phase,
        )

        if not _execute_dev_card_decision(
            board,
            players,
            inventories,
            player,
            decision,
            rng,
            bank=bank,
        ):
            return False

        played_action_dev_card = True

        # Knight can change Largest Army, and Road
        # Building can change Longest Road.
        active_player_has_won()

        return True

    # ------------------------------------------------
    # PRE-ROLL development-card opportunity.
    # ------------------------------------------------

    if active_player_has_won():
        actions.append(
            TurnAction(
                action_type=ActionType.PASS
            )
        )

        return TurnResult(
            player_id=player_id,
            roll=roll,
            action=actions[-1],
            actions=tuple(
                actions
            ),
            discards=discards,
            player_trades=tuple(
                player_trades
            ),
            trade_offer_count=(
                trade_offer_count
            ),
            trade_sequence_count=(
                trade_sequence_count
            ),
        )

    try_action_dev_card(
        DevCardPhase.PRE_ROLL
    )

    if active_player_has_won():
        actions.append(
            TurnAction(
                action_type=ActionType.PASS
            )
        )

        return TurnResult(
            player_id=player_id,
            roll=roll,
            action=actions[-1],
            actions=tuple(
                actions
            ),
            discards=discards,
            player_trades=tuple(
                player_trades
            ),
            trade_offer_count=(
                trade_offer_count
            ),
            trade_sequence_count=(
                trade_sequence_count
            ),
        )

    # ------------------------------------------------
    # Dice roll.
    # ------------------------------------------------

    if roll == 7:
        for pid, other_inventory in enumerate(
            inventories
        ):
            # Standard Catan rule:
            # only players holding more than 7
            # resource cards discard, and they
            # discard half rounded down.
            if other_inventory.total() <= 7:
                continue

            discard_count = (
                other_inventory.total()
                // 2
            )

            discarded = agents[
                pid
            ].choose_discards(
                players[
                    pid
                ],
                other_inventory,
                discard_count,
            )

            # Defensive validation: an agent must
            # return exactly the required number of
            # resource cards and cannot discard cards
            # it does not actually hold.
            if (
                len(discarded)
                != discard_count
            ):
                raise ValueError(
                    "Agent returned an invalid "
                    "number of discard cards: "
                    f"player={pid}, "
                    f"required={discard_count}, "
                    f"returned={len(discarded)}"
                )

            requested = {}

            for resource in discarded:
                requested[
                    resource
                ] = (
                    requested.get(
                        resource,
                        0,
                    )
                    + 1
                )

            for resource, amount in (
                requested.items()
            ):
                if (
                    other_inventory.count(
                        resource
                    )
                    < amount
                ):
                    raise ValueError(
                        "Agent attempted to discard "
                        "resources it does not hold: "
                        f"player={pid}, "
                        f"resource={resource}, "
                        f"requested={amount}, "
                        "held="
                        f"{other_inventory.count(resource)}"
                    )

            # Validate the full choice before
            # mutating the inventory, then remove
            # the selected cards.
            for resource in discarded:
                other_inventory.remove(
                    resource,
                    1,
                )

                if bank is not None:
                    bank.add(
                        resource,
                        1,
                    )

            discards[
                pid
            ] = discarded

        # After all required discards, the active
        # player must move the robber and may steal
        # one resource from an adjacent opponent.
        from catanlab.devcards import (
            move_robber,
            rob_adjacent_player,
        )

        robber_target = _knight_target_tile(
            board,
            players,
            inventories,
            player,
        )

        if robber_target is not None:
            move_robber(
                board,
                robber_target,
            )

            victim_id = (
                _choose_robber_victim(
                    board,
                    players,
                    inventories,
                    player,
                )
            )

            rob_adjacent_player(
                board,
                players,
                inventories,
                thief_id=player.player_id,
                rng=rng,
                victim_id=victim_id,
            )

    else:
        produce_for_roll(
            board,
            players,
            inventories,
            roll,
            bank=bank,
        )

    # ------------------------------------------------
    # POST-ROLL development-card opportunity.
    #
    # Only available if an action dev card was not
    # already played before rolling.
    # ------------------------------------------------

    try_action_dev_card(
        DevCardPhase.POST_ROLL
    )

    if active_player_has_won():
        actions.append(
            TurnAction(
                action_type=ActionType.PASS
            )
        )

        return TurnResult(
            player_id=player_id,
            roll=roll,
            action=actions[-1],
            actions=tuple(actions),
            discards=discards,
            player_trades=tuple(
                player_trades
            ),
            trade_offer_count=(
                trade_offer_count
            ),
            trade_sequence_count=(
                trade_sequence_count
            ),
        )

    # ------------------------------------------------
    # Normal build / purchase action.
    # ------------------------------------------------

    last_failed_trade_state = None

    # Safety cap against buggy agents that never pass.
    max_normal_actions = 50

    for _ in range(
        max_normal_actions
    ):
        # --------------------------------------------
        # Domestic negotiation opportunity.
        #
        # A new negotiation may occur after each
        # previous build/trade because the player's
        # resource needs and bargaining position may
        # have changed.
        #
        # The total number of offers across all such
        # sequences remains capped for the whole turn.
        # --------------------------------------------

        current_trade_state = (
            _trade_state_signature(
                player,
                inventory,
            )
        )

        may_negotiate = (
            trade_offer_count
            < MAX_TRADE_OFFERS_PER_TURN
            and trade_sequence_count
            < MAX_TRADE_SEQUENCES_PER_TURN
            and (
                last_failed_trade_state
                is None
                or current_trade_state
                != last_failed_trade_state
            )
        )

        if may_negotiate:
            remaining_trade_budget = (
                MAX_TRADE_OFFERS_PER_TURN
                - trade_offer_count
            )

            accepted_trade, offers_made = (
                _run_one_domestic_trade_sequence(
                    board,
                    players,
                    inventories,
                    agents,
                    player,
                    agent,
                    remaining_trade_budget,
                )
            )

            trade_offer_count += (
                offers_made
            )

            if offers_made > 0:
                trade_sequence_count += 1

            if accepted_trade is not None:
                player_trades.append(
                    accepted_trade
                )

                # The hand changed, so future
                # negotiation may again make sense.
                last_failed_trade_state = None

                # Development cards may be played
                # between other actions during the
                # player's turn.
                try_action_dev_card(
                    DevCardPhase.POST_ROLL
                )

                # A development card played after an
                # accepted trade may immediately create
                # a winning score through Largest Army
                # or Longest Road.
                if active_player_has_won():
                    if not actions:
                        actions.append(
                            TurnAction(
                                action_type=(
                                    ActionType.PASS
                                )
                            )
                        )
                    break

            elif offers_made > 0:
                # Do not immediately reopen bargaining
                # from the exact same state.
                last_failed_trade_state = (
                    current_trade_state
                )

        action = _choose_normal_action(
            agent,
            board,
            players,
            player,
            inventory,
            dev_deck,
            bank=bank,
        )

        actions.append(
            action
        )

        if (
            action.action_type
            == ActionType.PASS
        ):
            break

        execute_action(
            board,
            players,
            player,
            inventory,
            action,
            dev_deck=dev_deck,
            bank=bank,
        )

        # Building can change VP directly and roads
        # or settlements can alter Longest Road.
        if active_player_has_won():
            break

        # A legal action development card may be
        # played between normal turn actions.
        try_action_dev_card(
            DevCardPhase.POST_ROLL
        )

        if active_player_has_won():
            break

    else:
        raise RuntimeError(
            "Turn exceeded maximum normal-action "
            "count without passing."
        )

    return TurnResult(
        player_id=player_id,
        roll=roll,
        action=actions[-1],
        actions=tuple(
            actions
        ),
        discards=discards,
        player_trades=tuple(
            player_trades
        ),
        trade_offer_count=(
            trade_offer_count
        ),
        trade_sequence_count=(
            trade_sequence_count
        ),
    )


class AdaptiveStrategyAgent(TurnAgent):
    """
    State-aware gameplay agent driven by one of
    CatanLab's strategy profiles.

    High-level action choice comes from dynamic
    action utilities.

    Candidate selection within an action type uses
    board-specific strategic scoring.
    """

    def __init__(
        self,
        strategy,
    ) -> None:
        from catanlab.strategies import (
            STRATEGY_PROFILES,
        )

        self.strategy = strategy
        self.profile = STRATEGY_PROFILES[
            strategy
        ]

    def choose_dev_card_play(
        self,
        board,
        players,
        player,
        inventories,
        phase,
    ):
        from catanlab.devcard_policy import (
            choose_dev_card_play,
        )

        return choose_dev_card_play(
            player,
            players,
            inventories,
            phase=phase,
            strategy=self.strategy,
            board=board,
        )

    def _build_resource_progress(
        self,
        inventory: PlayerInventory,
        build_type: BuildType,
    ) -> float:
        """
        Return the fraction of a build's resource
        cost currently satisfied.

        Each required card counts independently.

        Example:
            settlement requires 4 cards total.
            Holding wood + brick gives 0.50.
        """

        from catanlab.economy import (
            BUILD_COSTS,
        )

        cost = BUILD_COSTS[
            build_type
        ]

        total_required = sum(
            cost.values()
        )

        if total_required <= 0:
            return 1.0

        satisfied = sum(
            min(
                inventory.count(
                    resource
                ),
                required,
            )
            for resource, required
            in cost.items()
        )

        return (
            satisfied
            / total_required
        )

    def _missing_build_cards(
        self,
        inventory: PlayerInventory,
        build_type: BuildType,
    ) -> int:
        """
        Number of resource cards still missing for
        the given build.
        """

        from catanlab.economy import (
            BUILD_COSTS,
        )

        cost = BUILD_COSTS[
            build_type
        ]

        return sum(
            max(
                0,
                required
                - inventory.count(
                    resource
                ),
            )
            for resource, required
            in cost.items()
        )

    def _discard_resource_value(
        self,
        resource: Resource,
        inventory: PlayerInventory,
    ) -> float:
        """
        Strategic value of retaining one card of this
        resource during a discard.

        Higher means we would rather keep it.
        """

        from catanlab.economy import (
            BUILD_COSTS,
        )

        value = (
            self.profile.resource_weights.get(
                resource,
                0.0,
            )
        )

        for build_type in (
            BuildType.CITY,
            BuildType.SETTLEMENT,
            BuildType.DEV_CARD,
            BuildType.ROAD,
        ):
            required = BUILD_COSTS[
                build_type
            ].get(
                resource,
                0,
            )

            if required <= 0:
                continue

            held = inventory.count(
                resource
            )

            # Preserve cards that are still part of a
            # currently incomplete build requirement.
            if held <= required:
                value += 0.75

        return value

    def choose_discards(
        self,
        player,
        inventory: PlayerInventory,
        count: int,
    ) -> list[Resource]:
        """
        Choose which resource cards to discard after
        rolling a 7.

        Re-evaluate the strategic value of the
        remaining hand after every discarded card.
        """

        if count <= 0:
            return []

        from catanlab.economy import (
            PlayerInventory,
        )

        remaining_inventory = (
            PlayerInventory()
        )

        for resource in (
            Resource.WOOD,
            Resource.BRICK,
            Resource.SHEEP,
            Resource.WHEAT,
            Resource.ORE,
        ):
            amount = inventory.count(
                resource
            )

            if amount:
                remaining_inventory.add(
                    resource,
                    amount,
                )

        discarded = []

        for _ in range(count):
            candidates = [
                resource
                for resource in (
                    Resource.WOOD,
                    Resource.BRICK,
                    Resource.SHEEP,
                    Resource.WHEAT,
                    Resource.ORE,
                )
                if (
                    remaining_inventory.count(
                        resource
                    )
                    > 0
                )
            ]

            if not candidates:
                break

            def discard_key(resource):
                return (
                    self._discard_resource_value(
                        resource,
                        remaining_inventory,
                    ),

                    # Among equally valuable cards,
                    # prefer shedding from the largest
                    # remaining pile.
                    -remaining_inventory.count(
                        resource
                    ),

                    resource.value,
                )

            resource = min(
                candidates,
                key=discard_key,
            )

            discarded.append(
                resource
            )

            remaining_inventory.remove(
                resource,
                1,
            )

        return discarded

    def propose_player_trade(
        self,
        board,
        players,
        player,
        inventories,
        excluded_recipients=None,
        agents=None,
    ):
        """
        Generate a strategic initial domestic trade
        proposal.

        Unlike the earlier version, initial offers
        may contain multiple cards and multiple
        resource types on either side.

        Search is bounded to keep negotiation fast.
        """

        from catanlab.action_scoring import (
            score_actions,
        )
        from catanlab.economy import (
            BUILD_COSTS,
        )
        from catanlab.trading import (
            TradeOffer,
            bundle_size,
            generate_trade_bundles,
            generate_trade_request_bundles,
            validate_trade_terms,
        )

        if excluded_recipients is None:
            excluded_recipients = set()

        inventory = inventories[
            player.player_id
        ]

        utilities = score_actions(
            self.strategy,
            player,
            players,
        ).as_dict()

        build_to_action = {
            BuildType.CITY:
                ActionType.BUILD_CITY,
            BuildType.SETTLEMENT:
                ActionType.BUILD_SETTLEMENT,
            BuildType.DEV_CARD:
                ActionType.BUY_DEV_CARD,
            BuildType.ROAD:
                ActionType.BUILD_ROAD,
        }

        legal_builds = []

        if legal_city_vertices(
            player
        ):
            legal_builds.append(
                BuildType.CITY
            )

        if legal_settlement_vertices(
            board,
            players,
            player,
        ):
            legal_builds.append(
                BuildType.SETTLEMENT
            )

        if legal_road_edges(
            board,
            players,
            player,
        ):
            legal_builds.append(
                BuildType.ROAD
            )

        legal_builds.append(
            BuildType.DEV_CARD
        )

        legal_builds = [
            build_type
            for build_type in legal_builds
            if utilities[
                build_to_action[
                    build_type
                ]
            ] > utilities[
                ActionType.PASS
            ]
        ]

        legal_builds.sort(
            key=lambda build_type: (
                -utilities[
                    build_to_action[
                        build_type
                    ]
                ],
                build_type.value,
            )
        )

        # Generate bundles the active player could
        # actually offer.
        give_bundles = (
            generate_trade_bundles(
                inventory,
                max_cards=4,
                max_types=3,
            )
        )

        candidates = []

        for build_type in legal_builds:
            cost = BUILD_COSTS[
                build_type
            ]

            # The proposal should help fill an actual
            # deficit toward a strategically useful
            # build.
            missing_cards = (
                self._missing_build_cards(
                    inventory,
                    build_type,
                )
            )

            # Domestic trading is primarily a
            # near-term completion tool. Do not
            # negotiate for a build we can already
            # afford, or one that is still several
            # cards away.
            if missing_cards == 0:
                continue

            if missing_cards > 1:
                continue

            deficits = {
                resource:
                    required
                    - inventory.count(
                        resource
                    )
                for resource, required
                in cost.items()
                if (
                    inventory.count(
                        resource
                    )
                    < required
                )
            }

            if not deficits:
                continue

            for other in players:
                if (
                    other.player_id
                    == player.player_id
                    or other.player_id
                    in excluded_recipients
                ):
                    continue

                # Requests are generated without
                # inspecting the recipient's hidden hand.
                # The recipient may later reject an offer
                # they cannot actually satisfy.
                receive_bundles = (
                    generate_trade_request_bundles(
                        max_cards=4,
                        max_types=3,
                    )
                )

                # Only consider bundles that contribute
                # resources actually missing for this
                # target build.
                useful_receive = []

                for bundle in receive_bundles:
                    useful = False
                    excess = False

                    for resource, amount in bundle:
                        deficit = deficits.get(
                            resource,
                            0,
                        )

                        if deficit > 0:
                            useful = True

                        # Avoid initially requesting
                        # obviously irrelevant cards.
                        if deficit == 0:
                            excess = True

                    if (
                        useful
                        and not excess
                    ):
                        useful_receive.append(
                            bundle
                        )

                if not useful_receive:
                    continue

                # Prefer bundles that close more of the
                # target deficit.
                useful_receive.sort(
                    key=lambda bundle: (
                        -sum(
                            min(
                                amount,
                                deficits.get(
                                    resource,
                                    0,
                                ),
                            )
                            for resource, amount
                            in bundle
                        ),
                        -self._trade_bundle_value(
                            bundle,
                            inventory,
                        ),
                        bundle_size(
                            bundle
                        ),
                    )
                )

                # Keep proposal search compact.
                useful_receive = (
                    useful_receive[
                        :10
                    ]
                )

                # Prefer giving away low-value cards.
                sorted_give = sorted(
                    give_bundles,
                    key=lambda bundle: (
                        self._trade_bundle_value(
                            bundle,
                            inventory,
                        ),
                        bundle_size(
                            bundle
                        ),
                    ),
                )[:12]

                for give in sorted_give:
                    give_value = (
                        self._trade_bundle_value(
                            give,
                            inventory,
                        )
                    )

                    for receive in useful_receive:
                        receive_value = (
                            self._trade_bundle_value(
                                receive,
                                inventory,
                            )
                        )

                        # Do not propose deals that are
                        # clearly bad for ourselves.
                        if (
                            receive_value
                            < give_value
                        ):
                            continue

                        offer = TradeOffer(
                            proposer_id=(
                                player.player_id
                            ),
                            recipient_id=(
                                other.player_id
                            ),
                            give=give,
                            receive=receive,
                        )

                        if not validate_trade_terms(
                            offer
                        ):
                            continue

                        # --------------------------------
                        # Concrete build-progress test.
                        #
                        # Domestic trading should happen
                        # because it materially advances a
                        # plan, not merely because the
                        # incoming cards have a higher
                        # abstract valuation.
                        # --------------------------------

                        before_progress = (
                            self._build_resource_progress(
                                inventory,
                                build_type,
                            )
                        )

                        from catanlab.economy import (
                            PlayerInventory,
                        )

                        simulated = PlayerInventory()

                        for resource in (
                            Resource.WOOD,
                            Resource.BRICK,
                            Resource.SHEEP,
                            Resource.WHEAT,
                            Resource.ORE,
                        ):
                            amount = inventory.count(
                                resource
                            )

                            if amount:
                                simulated.add(
                                    resource,
                                    amount,
                                )

                        for resource, amount in offer.give:
                            simulated.remove(
                                resource,
                                amount,
                            )

                        for resource, amount in offer.receive:
                            simulated.add(
                                resource,
                                amount,
                            )

                        after_progress = (
                            self._build_resource_progress(
                                simulated,
                                build_type,
                            )
                        )

                        progress_gain = (
                            after_progress
                            - before_progress
                        )

                        # Require at least one quarter of
                        # a build's card cost worth of
                        # concrete progress.
                        #
                        # For settlements, for example,
                        # this means gaining at least one
                        # genuinely useful card.
                        if progress_gain < 0.20:
                            continue

                        # Screen for negotiation
                        # plausibility, NOT immediate
                        # acceptance.
                        #
                        # Calling evaluate_player_trade()
                        # here would guarantee that every
                        # proposal surviving this filter is
                        # accepted when the actual
                        # negotiation evaluates the same
                        # unchanged offer.
                        # The proposer cannot inspect
                        # the recipient's hidden hand to
                        # predict their exact valuation.
                        #
                        # Use only the publicly visible card
                        # quantities as a rough bargaining
                        # heuristic.
                        receive_cards = bundle_size(
                            offer.receive
                        )

                        if receive_cards > 0:
                            recipient_ratio = (
                                bundle_size(
                                    offer.give
                                )
                                / receive_cards
                            )
                        else:
                            recipient_ratio = 1.0

                        # A player at 9 visible VP is already
                        # rejected by adaptive recipients, so
                        # avoid starting a pointless
                        # negotiation.
                        if (
                            player.public_victory_points
                            >= 9
                        ):
                            continue

                        deficit_progress = sum(
                            min(
                                amount,
                                deficits.get(
                                    resource,
                                    0,
                                ),
                            )
                            for resource, amount
                            in receive
                        )

                        self_gain = (
                            receive_value
                            - give_value
                        )

                        simplicity = -(
                            bundle_size(
                                give
                            )
                            + bundle_size(
                                receive
                            )
                        )

                        completes_build = (
                            simulated.can_afford(
                                build_type
                            )
                        )

                        # A realistic opening offer
                        # should be plausible without being
                        # deliberately over-generous.
                        #
                        # The recipient accepts around 1.0+
                        # value; an opening near 0.92 leaves
                        # room for genuine bargaining.
                        bargaining_fit = -abs(
                            recipient_ratio
                            - 0.92
                        )

                        candidates.append(
                            (
                                utilities[
                                    build_to_action[
                                        build_type
                                    ]
                                ],
                                int(
                                    completes_build
                                ),
                                progress_gain,
                                deficit_progress,
                                bargaining_fit,
                                self_gain,
                                simplicity,
                                -other.player_id,
                                offer,
                            )
                        )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                item[3],
                item[4],
                item[5],
                item[6],
                item[7],
                repr(
                    item[8]
                ),
            ),
            reverse=True,
        )

        return candidates[
            0
        ][8]

    def _trade_bundle_value(
        self,
        bundle,
        inventory: PlayerInventory,
    ) -> float:
        """
        Estimate the strategic value of an entire
        resource bundle to this agent.
        """

        return sum(
            self._resource_trade_value(
                resource,
                inventory,
            )
            * amount
            for resource, amount in bundle
        )

    def evaluate_player_trade(
        self,
        board,
        players,
        player,
        inventories,
        offer,
    ) -> bool:
        """
        Decide whether this player accepts a domestic
        trade.

        Acceptance requires a real strategic benefit,
        not merely a nonnegative abstract resource
        exchange.
        """

        from catanlab.action_scoring import (
            score_actions,
        )
        from catanlab.economy import (
            PlayerInventory,
        )
        from catanlab.trading import (
            validate_trade_offer,
        )

        if (
            offer.recipient_id
            != player.player_id
        ):
            return False

        if not validate_trade_offer(
            offer,
            inventories,
        ):
            return False

        proposer = players[
            offer.proposer_id
        ]

        # Do not voluntarily help an opponent who is
        # already one visible point from victory.
        if proposer.public_victory_points >= 9:
            return False

        inventory = inventories[
            player.player_id
        ]

        incoming_value = (
            self._trade_bundle_value(
                offer.give,
                inventory,
            )
        )

        outgoing_value = (
            self._trade_bundle_value(
                offer.receive,
                inventory,
            )
        )

        if outgoing_value <= 0:
            value_ratio = (
                2.0
                if incoming_value > 0
                else 1.0
            )
        else:
            value_ratio = (
                incoming_value
                / outgoing_value
            )

        # Very poor terms are not worth considering,
        # regardless of short-term build progress.
        if value_ratio < 0.90:
            return False

        # Simulate the recipient's post-trade hand.
        simulated = PlayerInventory()

        for resource in (
            Resource.WOOD,
            Resource.BRICK,
            Resource.SHEEP,
            Resource.WHEAT,
            Resource.ORE,
        ):
            amount = inventory.count(
                resource
            )

            if amount:
                simulated.add(
                    resource,
                    amount,
                )

        # Recipient receives offer.give and gives
        # offer.receive.
        for resource, amount in offer.receive:
            simulated.remove(
                resource,
                amount,
            )

        for resource, amount in offer.give:
            simulated.add(
                resource,
                amount,
            )

        utilities = score_actions(
            self.strategy,
            player,
            players,
        ).as_dict()

        build_to_action = {
            BuildType.CITY:
                ActionType.BUILD_CITY,
            BuildType.SETTLEMENT:
                ActionType.BUILD_SETTLEMENT,
            BuildType.DEV_CARD:
                ActionType.BUY_DEV_CARD,
            BuildType.ROAD:
                ActionType.BUILD_ROAD,
        }

        legal_builds = []

        if legal_city_vertices(
            player
        ):
            legal_builds.append(
                BuildType.CITY
            )

        if legal_settlement_vertices(
            board,
            players,
            player,
        ):
            legal_builds.append(
                BuildType.SETTLEMENT
            )

        if legal_road_edges(
            board,
            players,
            player,
        ):
            legal_builds.append(
                BuildType.ROAD
            )

        legal_builds.append(
            BuildType.DEV_CARD
        )

        best_progress_gain = 0.0
        newly_affordable = False

        for build_type in legal_builds:
            action_type = (
                build_to_action[
                    build_type
                ]
            )

            if (
                utilities[action_type]
                <= utilities[
                    ActionType.PASS
                ]
            ):
                continue

            before = (
                self._build_resource_progress(
                    inventory,
                    build_type,
                )
            )

            after = (
                self._build_resource_progress(
                    simulated,
                    build_type,
                )
            )

            best_progress_gain = max(
                best_progress_gain,
                after - before,
            )

            if (
                not inventory.can_afford(
                    build_type
                )
                and simulated.can_afford(
                    build_type
                )
            ):
                newly_affordable = True

        # Strongly favorable terms are enough on
        # their own.
        if value_ratio >= 1.15:
            return True

        # Otherwise the trade must materially improve
        # something the recipient actually wants to
        # build.
        if (
            newly_affordable
            and value_ratio >= 0.95
        ):
            return True

        if (
            best_progress_gain >= 0.20
            and value_ratio >= 1.00
        ):
            return True

        return False

    def _bundle_distance(
        self,
        first,
        second,
    ) -> int:
        """
        Simple card-count distance between bundles.
        """

        resources = (
            Resource.WOOD,
            Resource.BRICK,
            Resource.SHEEP,
            Resource.WHEAT,
            Resource.ORE,
        )

        first_map = dict(
            first
        )

        second_map = dict(
            second
        )

        return sum(
            abs(
                first_map.get(
                    resource,
                    0,
                )
                - second_map.get(
                    resource,
                    0,
                )
            )
            for resource in resources
        )

    def counter_player_trade(
        self,
        board,
        players,
        player,
        inventories,
        offer,
        attempted_offers=None,
    ):
        """
        Generate a bounded strategic counteroffer.

        Counteroffers may change quantities and may
        contain multiple resource types.
        """

        from catanlab.trading import (
            TradeOffer,
            bundle_size,
            generate_trade_bundles,
            generate_trade_request_bundles,
            validate_trade_terms,
        )

        if (
            offer.recipient_id
            != player.player_id
        ):
            return None

        if attempted_offers is None:
            attempted_offers = set()

        inventory = inventories[
            player.player_id
        ]

        # A counteroffer reverses perspective:
        #
        # current player now proposes what they give
        # and what they want from the previous
        # proposer.
        baseline_give = (
            offer.receive
        )

        baseline_receive = (
            offer.give
        )

        outgoing_bundles = (
            generate_trade_bundles(
                inventory,
                max_cards=4,
                max_types=3,
            )
        )

        incoming_bundles = (
            generate_trade_request_bundles(
                max_cards=4,
                max_types=3,
            )
        )

        # Keep the search compact. Good outgoing
        # bundles are inexpensive to this player.
        outgoing_bundles.sort(
            key=lambda bundle: (
                self._trade_bundle_value(
                    bundle,
                    inventory,
                ),
                bundle_size(
                    bundle
                ),
                self._bundle_distance(
                    bundle,
                    baseline_give,
                ),
            )
        )

        # Good incoming bundles are valuable to this
        # player.
        incoming_bundles.sort(
            key=lambda bundle: (
                -self._trade_bundle_value(
                    bundle,
                    inventory,
                ),
                bundle_size(
                    bundle
                ),
                self._bundle_distance(
                    bundle,
                    baseline_receive,
                ),
            )
        )

        outgoing_bundles = (
            outgoing_bundles[
                :12
            ]
        )

        incoming_bundles = (
            incoming_bundles[
                :12
            ]
        )

        candidates = []

        for give in outgoing_bundles:
            outgoing_value = (
                self._trade_bundle_value(
                    give,
                    inventory,
                )
            )

            for receive in incoming_bundles:
                incoming_value = (
                    self._trade_bundle_value(
                        receive,
                        inventory,
                    )
                )

                # A counteroffer should not make this
                # player strictly worse off according
                # to its own current valuation.
                # Counteroffers should improve this
                # player's position enough to justify
                # continuing the negotiation. Merely
                # breaking even creates almost-certain
                # deal closure between similar agents.
                if outgoing_value > 0:
                    counter_ratio = (
                        incoming_value
                        / outgoing_value
                    )
                else:
                    counter_ratio = (
                        2.0
                        if incoming_value > 0
                        else 1.0
                    )

                if counter_ratio < 1.05:
                    continue

                candidate = TradeOffer(
                    proposer_id=(
                        player.player_id
                    ),
                    recipient_id=(
                        offer.proposer_id
                    ),
                    give=give,
                    receive=receive,
                )

                if (
                    candidate
                    in attempted_offers
                ):
                    continue

                if not validate_trade_terms(
                    candidate
                ):
                    continue

                distance = (
                    self._bundle_distance(
                        give,
                        baseline_give,
                    )
                    + self._bundle_distance(
                        receive,
                        baseline_receive,
                    )
                )

                gain = (
                    incoming_value
                    - outgoing_value
                )

                total_cards = (
                    bundle_size(
                        give
                    )
                    + bundle_size(
                        receive
                    )
                )

                candidates.append(
                    (
                        gain,
                        -distance,
                        -total_cards,
                        candidate,
                    )
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                repr(
                    item[3]
                ),
            ),
            reverse=True,
        )

        return candidates[
            0
        ][3]

    def _best_maritime_trade(
        self,
        board: Board,
        player: PlayerState,
        inventory: PlayerInventory,
        players: list[PlayerState] | None = None,
        utilities: dict[ActionType, float] | None = None,
        dev_deck: DevCardDeck | None = None,
        bank: ResourceBank | None = None,
    ) -> tuple[
        TurnAction,
        BuildType,
    ] | None:
        """
        Find a maritime trade that immediately
        completes a useful build.

        The returned BuildType lets the caller score
        the trade according to the action it enables.
        """

        from catanlab.economy import (
            BUILD_COSTS,
        )
        from catanlab.ports import (
            best_maritime_ratio,
        )

        if players is None:
            players = [player]

        build_to_action = {
            BuildType.CITY:
                ActionType.BUILD_CITY,
            BuildType.SETTLEMENT:
                ActionType.BUILD_SETTLEMENT,
            BuildType.DEV_CARD:
                ActionType.BUY_DEV_CARD,
            BuildType.ROAD:
                ActionType.BUILD_ROAD,
        }

        build_priority = [
            BuildType.CITY,
            BuildType.SETTLEMENT,
            BuildType.DEV_CARD,
            BuildType.ROAD,
        ]

        # Only trade toward builds that are actually
        # legal in the current game state.
        legal_builds = []

        if legal_city_vertices(
            player
        ):
            legal_builds.append(
                BuildType.CITY
            )

        if legal_settlement_vertices(
            board,
            players,
            player,
        ):
            legal_builds.append(
                BuildType.SETTLEMENT
            )

        if (
            dev_deck is None
            or len(dev_deck.cards) > 0
        ):
            legal_builds.append(
                BuildType.DEV_CARD
            )

        if legal_road_edges(
            board,
            players,
            player,
        ):
            legal_builds.append(
                BuildType.ROAD
            )

        if utilities is not None:
            pass_utility = utilities[
                ActionType.PASS
            ]

            # Do not spend resources to enable an
            # action that the strategy values less
            # than simply saving the cards.
            legal_builds = [
                build_type
                for build_type in legal_builds
                if utilities[
                    build_to_action[
                        build_type
                    ]
                ] > pass_utility
            ]

            priority_index = {
                build_type: index
                for index, build_type
                in enumerate(
                    build_priority
                )
            }

            legal_builds.sort(
                key=lambda build_type: (
                    -utilities[
                        build_to_action[
                            build_type
                        ]
                    ],
                    priority_index[
                        build_type
                    ],
                )
            )
        else:
            legal_builds.sort(
                key=build_priority.index
            )

        resources = (
            Resource.WOOD,
            Resource.BRICK,
            Resource.SHEEP,
            Resource.WHEAT,
            Resource.ORE,
        )

        for build_type in legal_builds:
            cost = BUILD_COSTS[
                build_type
            ]

            deficits = {
                resource: max(
                    0,
                    required
                    - inventory.count(
                        resource
                    ),
                )
                for resource, required
                in cost.items()
            }

            total_missing = sum(
                deficits.values()
            )

            if total_missing == 0:
                continue

            # Choose one currently missing resource.
            # The surrounding multi-action turn loop
            # will reevaluate after each trade, so a
            # build that is several cards short can be
            # approached over multiple useful trades.
            available_receives = [
                resource
                for resource, deficit
                in deficits.items()
                if (
                    deficit > 0
                    and (
                        bank is None
                        or bank.can_supply(
                            resource,
                            1,
                        )
                    )
                )
            ]

            if not available_receives:
                continue

            receive = max(
                available_receives,
                key=lambda resource: (
                    deficits[resource],
                    resource.value,
                ),
            )

            for give in resources:
                if give == receive:
                    continue

                ratio = best_maritime_ratio(
                    board,
                    player,
                    give,
                )

                # Never trade away cards required by
                # the target build. The give resource
                # must be genuine surplus.
                required_after_trade = (
                    cost.get(
                        give,
                        0,
                    )
                )

                surplus = (
                    inventory.count(
                        give
                    )
                    - required_after_trade
                )

                if surplus < ratio:
                    continue

                return (
                    TurnAction(
                        action_type=(
                            ActionType.MARITIME_TRADE
                        ),
                        give_resource=give,
                        receive_resource=receive,
                    ),
                    build_type,
                )

        return None

    def _best_city_vertex(
        self,
        board: Board,
        player: PlayerState,
    ) -> int | None:
        from catanlab.scoring import (
            score_vertex,
        )

        vertices = legal_city_vertices(
            player
        )

        if not vertices:
            return None

        return max(
            vertices,
            key=lambda vertex_id: (
                score_vertex(
                    board,
                    board.vertices[
                        vertex_id
                    ],
                ).production_score,
                -vertex_id,
            ),
        )

    def _best_settlement_vertex(
        self,
        board: Board,
        players: list[PlayerState],
        player: PlayerState,
    ) -> int | None:
        from catanlab.scoring import (
            port_synergy_score,
            strategic_vertex_score,
        )
        from catanlab.strategies import (
            StrategyType,
        )

        legal = legal_settlement_vertices(
            board,
            players,
            player,
        )

        if not legal:
            return None

        best_vertex = None
        best_score = None

        for vertex_id in legal:
            vertex = board.vertices[
                vertex_id
            ]

            score = strategic_vertex_score(
                board,
                vertex,
                self.profile.resource_weights,
                self.profile.diversity_weight,
            )

            if (
                self.strategy
                == StrategyType.PORT
            ):
                score += port_synergy_score(
                    board,
                    [
                        vertex,
                    ],
                )

            if (
                best_score is None
                or score > best_score
                or (
                    score == best_score
                    and vertex_id < best_vertex
                )
            ):
                best_score = score
                best_vertex = vertex_id

        return best_vertex

    def _best_road_edge(
        self,
        board: Board,
        players: list[PlayerState],
        player: PlayerState,
    ) -> tuple[int, int] | None:
        from catanlab.longest_road import (
            longest_road_length,
        )

        legal = legal_road_edges(
            board,
            players,
            player,
        )

        if not legal:
            return None

        best_edge = None
        best_length = None

        for edge in legal:
            player.roads.append(
                edge
            )

            length = longest_road_length(
                player,
                players,
            )

            player.roads.pop()

            if (
                best_length is None
                or length > best_length
                or (
                    length == best_length
                    and edge < best_edge
                )
            ):
                best_length = length
                best_edge = edge

        return best_edge

    def choose_action(
        self,
        board: Board,
        players: list[PlayerState],
        player: PlayerState,
        inventory: PlayerInventory,
        dev_deck: DevCardDeck | None = None,
        bank: ResourceBank | None = None,
    ) -> TurnAction:
        from catanlab.action_scoring import (
            score_actions,
        )

        utilities = score_actions(
            self.strategy,
            player,
            players,
        ).as_dict()

        candidates: list[
            tuple[
                float,
                TurnAction,
            ]
        ] = []

        if inventory.can_afford(
            BuildType.CITY
        ):
            vertex_id = (
                self._best_city_vertex(
                    board,
                    player,
                )
            )

            if vertex_id is not None:
                candidates.append(
                    (
                        utilities[
                            ActionType.BUILD_CITY
                        ],
                        TurnAction(
                            action_type=(
                                ActionType.BUILD_CITY
                            ),
                            vertex_id=vertex_id,
                        ),
                    )
                )

        if inventory.can_afford(
            BuildType.SETTLEMENT
        ):
            vertex_id = (
                self._best_settlement_vertex(
                    board,
                    players,
                    player,
                )
            )

            if vertex_id is not None:
                candidates.append(
                    (
                        utilities[
                            ActionType.BUILD_SETTLEMENT
                        ],
                        TurnAction(
                            action_type=(
                                ActionType.BUILD_SETTLEMENT
                            ),
                            vertex_id=vertex_id,
                        ),
                    )
                )

        if inventory.can_afford(
            BuildType.ROAD
        ):
            edge = self._best_road_edge(
                board,
                players,
                player,
            )

            if edge is not None:
                candidates.append(
                    (
                        utilities[
                            ActionType.BUILD_ROAD
                        ],
                        TurnAction(
                            action_type=(
                                ActionType.BUILD_ROAD
                            ),
                            edge=edge,
                        ),
                    )
                )

        if (
            inventory.can_afford(
                BuildType.DEV_CARD
            )
            and (
                dev_deck is None
                or len(dev_deck.cards) > 0
            )
        ):
            candidates.append(
                (
                    utilities[
                        ActionType.BUY_DEV_CARD
                    ],
                    TurnAction(
                        action_type=(
                            ActionType.BUY_DEV_CARD
                        )
                    ),
                )
            )

        trade_result = (
            self._best_maritime_trade(
                board,
                player,
                inventory,
                players=players,
                utilities=utilities,
                dev_deck=dev_deck,
                bank=bank,
            )
        )

        if trade_result is not None:
            (
                trade_action,
                target_build,
            ) = trade_result

            target_action = {
                BuildType.CITY:
                    ActionType.BUILD_CITY,
                BuildType.SETTLEMENT:
                    ActionType.BUILD_SETTLEMENT,
                BuildType.DEV_CARD:
                    ActionType.BUY_DEV_CARD,
                BuildType.ROAD:
                    ActionType.BUILD_ROAD,
            }[
                target_build
            ]

            candidates.append(
                (
                    utilities[
                        target_action
                    ],
                    trade_action,
                )
            )

        candidates.append(
            (
                utilities[
                    ActionType.PASS
                ],
                TurnAction(
                    action_type=(
                        ActionType.PASS
                    )
                ),
            )
        )

        # The utility score is the primary decision.
        # Action name provides deterministic
        # tie-breaking.
        _, action = max(
            candidates,
            key=lambda item: (
                item[0],
                item[1].action_type.value,
            ),
        )

        return action
