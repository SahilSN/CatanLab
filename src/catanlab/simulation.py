from dataclasses import dataclass, field

from catanlab.board import Board
from catanlab.scoring import (
    rank_vertices,
    score_opening_pair,
)


@dataclass
class PlayerState:
    player_id: int
    settlements: list[int] = field(
        default_factory=list
    )
    cities: list[int] = field(
        default_factory=list
    )
    roads: list[tuple[int, int]] = field(
        default_factory=list
    )
    dev_cards: list[str] = field(
        default_factory=list
    )

    # Cards bought during the player's current turn.
    # They are owned immediately, but action cards
    # cannot be played until the player's next turn.
    new_dev_cards: list[str] = field(
        default_factory=list
    )

    # Action development cards that have been
    # publicly played during the game.
    #
    # Victory-point cards never enter this list
    # because they remain hidden until game end.
    played_dev_cards: list[str] = field(
        default_factory=list
    )

    knights_played: int = 0
    has_largest_army: bool = False
    has_longest_road: bool = False

    @property
    def public_victory_points(self) -> int:
        """
        Victory points visible to other players.

        Hidden victory-point development cards are
        intentionally excluded.
        """

        largest_army_vp = (
            2
            if self.has_largest_army
            else 0
        )

        longest_road_vp = (
            2
            if self.has_longest_road
            else 0
        )

        return (
            len(self.settlements)
            + 2 * len(self.cities)
            + largest_army_vp
            + longest_road_vp
        )

    @property
    def victory_points(self) -> int:
        """
        Player's true victory-point total.

        Includes hidden victory-point development
        cards and is therefore suitable for win
        detection and the player's own reasoning.
        """

        dev_vp = sum(
            card == "victory_point"
            for card in self.dev_cards
        )

        return (
            self.public_victory_points
            + dev_vp
        )


@dataclass
class DraftResult:
    players: list[PlayerState]
    placement_order: list[
        tuple[int, int]
    ]
    road_order: list[
        tuple[
            int,
            tuple[int, int],
        ]
    ] = field(
        default_factory=list
    )


def blocked_vertices(
    board: Board,
    occupied: set[int],
) -> set[int]:
    """
    Return all vertices unavailable for settlement
    placement under the Catan distance rule.
    """

    blocked = set(occupied)

    for vertex_id in occupied:
        blocked.update(
            board.vertices[
                vertex_id
            ].neighbors
        )

    return blocked


def legal_vertices(
    board: Board,
    occupied: set[int],
) -> list[int]:
    """Return all currently legal settlement vertices."""

    blocked = blocked_vertices(
        board,
        occupied,
    )

    return [
        vertex.id
        for vertex in board.vertices
        if vertex.id not in blocked
    ]


class Agent:
    def choose_vertex(
        self,
        board: Board,
        legal: list[int],
        player: PlayerState,
    ) -> int:
        raise NotImplementedError


class ProductionAgent(Agent):
    """
    Chooses the highest-production legal vertex.
    """

    def choose_vertex(
        self,
        board: Board,
        legal: list[int],
        player: PlayerState,
    ) -> int:
        ranked = rank_vertices(
            board
        )

        legal_set = set(
            legal
        )

        for result in ranked:
            if result.vertex_id in legal_set:
                return result.vertex_id

        raise RuntimeError(
            "No legal settlement vertex available."
        )


class BalancedAgent(Agent):
    """
    Uses individual composite score for the
    first settlement and opening-pair score
    for the second.
    """

    def choose_vertex(
        self,
        board: Board,
        legal: list[int],
        player: PlayerState,
    ) -> int:
        if not player.settlements:
            ranked = rank_vertices(
                board
            )

            legal_set = set(
                legal
            )

            for result in ranked:
                if result.vertex_id in legal_set:
                    return result.vertex_id

            raise RuntimeError(
                "No legal settlement vertex available."
            )

        first_vertex = board.vertices[
            player.settlements[0]
        ]

        best_vertex = None
        best_score = None

        for vertex_id in legal:
            candidate = board.vertices[
                vertex_id
            ]

            pair = score_opening_pair(
                board,
                first_vertex,
                candidate,
            )

            score = (
                pair.composite_score
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

        if best_vertex is None:
            raise RuntimeError(
                "No legal settlement vertex available."
            )

        return best_vertex


def run_opening_draft(
    board: Board,
    agents: list[Agent],
) -> DraftResult:
    """
    Run the standard 4-player snake setup.

    Each settlement is immediately paired with one
    setup road connected to that settlement.
    """

    if len(agents) != 4:
        raise ValueError(
            "Opening draft requires exactly "
            "four agents."
        )

    players = [
        PlayerState(
            player_id=i
        )
        for i in range(4)
    ]

    order = [
        0,
        1,
        2,
        3,
        3,
        2,
        1,
        0,
    ]

    placement_order = []
    road_order = []

    occupied_vertices: set[int] = set()
    occupied_roads: set[
        tuple[int, int]
    ] = set()

    for player_id in order:
        player = players[
            player_id
        ]

        legal = legal_vertices(
            board,
            occupied_vertices,
        )

        vertex_id = agents[
            player_id
        ].choose_vertex(
            board,
            legal,
            player,
        )

        if vertex_id not in legal:
            raise ValueError(
                "Agent selected illegal "
                "opening vertex."
            )

        player.settlements.append(
            vertex_id
        )

        occupied_vertices.add(
            vertex_id
        )

        placement_order.append(
            (
                player_id,
                vertex_id,
            )
        )

        road = choose_setup_road(
            board,
            vertex_id,
            occupied_roads,
            agent=agents[
                player_id
            ],
            occupied_vertices=(
                occupied_vertices
            ),
        )

        player.roads.append(
            road
        )

        occupied_roads.add(
            road
        )

        road_order.append(
            (
                player_id,
                road,
            )
        )

    return DraftResult(
        players=players,
        placement_order=placement_order,
        road_order=road_order,
    )


class StrategyOpeningAgent(Agent):
    """
    Opening-draft agent driven by one of the
    CatanLab strategy profiles.

    First placement:
        score the individual vertex.

    Second placement:
        score the combined opening pair.

    Port strategies additionally evaluate port /
    resource-production synergy.
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

    def choose_vertex(
        self,
        board: Board,
        legal: list[int],
        player: PlayerState,
    ) -> int:
        from catanlab.scoring import (
            port_synergy_score,
            score_opening_pair,
            score_vertex,
            strategic_pair_score,
            strategic_vertex_score,
        )
        from catanlab.strategies import (
            StrategyType,
        )

        if not legal:
            raise RuntimeError(
                "No legal settlement vertex available."
            )

        best_vertex = None
        best_key = None

        for vertex_id in legal:
            vertex = board.vertices[
                vertex_id
            ]

            if not player.settlements:
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

                standard = score_vertex(
                    board,
                    vertex,
                )

                key = (
                    score,
                    standard.composite_score,
                    standard.production_score,
                    -vertex_id,
                )

            else:
                first = board.vertices[
                    player.settlements[0]
                ]

                score = strategic_pair_score(
                    board,
                    first,
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
                            first,
                            vertex,
                        ],
                    )

                standard = score_opening_pair(
                    board,
                    first,
                    vertex,
                )

                key = (
                    score,
                    standard.composite_score,
                    standard.production_score,
                    -vertex_id,
                )

            if (
                best_key is None
                or key > best_key
            ):
                best_key = key
                best_vertex = vertex_id

        if best_vertex is None:
            raise RuntimeError(
                "No legal settlement vertex available."
            )

        return best_vertex


def choose_setup_road(
    board: Board,
    settlement_vertex: int,
    occupied_roads: set[
        tuple[int, int]
    ],
    agent=None,
    occupied_vertices: set[int] | None = None,
) -> tuple[int, int]:
    """
    Choose a legal setup road touching the newly
    placed settlement.

    When a StrategyOpeningAgent is supplied, prefer
    roads pointing toward strong future settlement
    opportunities for that strategy.

    The original generic expansion heuristic remains
    as a secondary criterion and fallback.
    """

    from catanlab.scoring import (
        port_synergy_score,
        strategic_vertex_score,
    )
    from catanlab.strategies import (
        StrategyType,
    )

    if occupied_vertices is None:
        occupied_vertices = {
            settlement_vertex
        }

    blocked = blocked_vertices(
        board,
        occupied_vertices,
    )

    candidates = []

    for edge in board.edges:
        endpoints = {
            edge.vertex_a,
            edge.vertex_b,
        }

        if settlement_vertex not in endpoints:
            continue

        canonical = tuple(
            sorted(
                (
                    edge.vertex_a,
                    edge.vertex_b,
                )
            )
        )

        if canonical in occupied_roads:
            continue

        far_vertex = (
            edge.vertex_b
            if edge.vertex_a
            == settlement_vertex
            else edge.vertex_a
        )

        expansion = len(
            board.vertices[
                far_vertex
            ].neighbors
        )

        future_scores = []

        for future_vertex_id in (
            board.vertices[
                far_vertex
            ].neighbors
        ):
            if (
                future_vertex_id
                in blocked
            ):
                continue

            future_vertex = board.vertices[
                future_vertex_id
            ]

            is_strategy_agent = (
                agent is not None
                and hasattr(
                    agent,
                    "profile",
                )
                and hasattr(
                    agent,
                    "strategy",
                )
            )

            if not is_strategy_agent:
                # Generic agents such as BalancedAgent
                # keep the original topology-based
                # setup-road behavior.
                future_score = 1.0

            else:
                future_score = (
                    strategic_vertex_score(
                        board,
                        future_vertex,
                        agent.profile.resource_weights,
                        agent.profile.diversity_weight,
                    )
                )

                if (
                    agent.strategy
                    == StrategyType.PORT
                ):
                    future_score += (
                        port_synergy_score(
                            board,
                            [
                                future_vertex,
                            ],
                        )
                    )

            future_scores.append(
                future_score
            )

        best_future_score = (
            max(future_scores)
            if future_scores
            else 0.0
        )

        future_site_count = len(
            future_scores
        )

        candidates.append(
            (
                best_future_score,
                future_site_count,
                expansion,
                canonical,
            )
        )

    if not candidates:
        raise RuntimeError(
            "No legal setup road available."
        )

    candidates.sort(
        key=lambda item: (
            -item[0],
            -item[1],
            -item[2],
            item[3],
        )
    )

    return candidates[
        0
    ][3]


def grant_second_settlement_resources(
    board: Board,
    players: list[PlayerState],
    inventories,
    bank=None,
) -> None:
    """
    Grant each player starting resources from
    their second settlement only.

    Each adjacent non-desert tile contributes one
    resource card.

    If a finite bank is supplied, each granted card
    is removed from that bank.
    """

    from catanlab.resources import Resource

    if len(players) != len(inventories):
        raise ValueError(
            "players and inventories must have "
            "matching lengths."
        )

    for player, inventory in zip(
        players,
        inventories,
    ):
        if len(player.settlements) < 2:
            raise ValueError(
                "Player does not have two "
                "starting settlements."
            )

        second_vertex_id = (
            player.settlements[1]
        )

        vertex = board.vertices[
            second_vertex_id
        ]

        for tile_id in vertex.adjacent_tiles:
            tile = board.tiles[
                tile_id
            ]

            if (
                tile.resource
                == Resource.DESERT
            ):
                continue

            if bank is not None:
                if not bank.can_supply(
                    tile.resource,
                    1,
                ):
                    raise ValueError(
                        "Bank cannot supply "
                        "starting resource "
                        f"{tile.resource.value}"
                    )

                bank.remove(
                    tile.resource,
                    1,
                )

            inventory.add(
                tile.resource
            )
