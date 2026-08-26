from dataclasses import dataclass

from catanlab.board import Board, Vertex
from catanlab.dice import production_weight
from catanlab.resources import Resource


DIVERSITY_WEIGHT = 1.5


@dataclass(frozen=True)
class SettlementScore:
    vertex_id: int
    production_score: int
    production_probability: float
    resources: tuple[Resource, ...]
    numbers: tuple[int, ...]
    unique_resources: int
    composite_score: float


def score_vertex(
    board: Board,
    vertex: Vertex,
) -> SettlementScore:
    """
    Score one settlement vertex using expected
    production and resource diversity.
    """

    production_score = 0
    resources: list[Resource] = []
    numbers: list[int] = []

    for tile_id in vertex.adjacent_tiles:
        tile = board.tiles[tile_id]

        if tile.resource == Resource.DESERT:
            continue

        if tile.number is None:
            continue

        production_score += production_weight(
            tile.number
        )

        resources.append(
            tile.resource
        )

        numbers.append(
            tile.number
        )

    unique_resources = len(
        set(resources)
    )

    composite_score = (
        production_score
        + DIVERSITY_WEIGHT * unique_resources
    )

    return SettlementScore(
        vertex_id=vertex.id,
        production_score=production_score,
        production_probability=production_score / 36,
        resources=tuple(resources),
        numbers=tuple(numbers),
        unique_resources=unique_resources,
        composite_score=composite_score,
    )


def rank_vertices(
    board: Board,
) -> list[SettlementScore]:
    """
    Rank all settlement vertices using the
    composite production + diversity score.
    """

    scores = [
        score_vertex(
            board,
            vertex,
        )
        for vertex in board.vertices
    ]

    return sorted(
        scores,
        key=lambda result: (
            -result.composite_score,
            -result.production_score,
            result.vertex_id,
        ),
    )


@dataclass(frozen=True)
class OpeningPairScore:
    vertex_a: int
    vertex_b: int
    production_score: int
    unique_resources: int
    resources: tuple[Resource, ...]
    composite_score: float


def score_opening_pair(
    board: Board,
    vertex_a: Vertex,
    vertex_b: Vertex,
) -> OpeningPairScore:
    """
    Score a pair of starting settlements.

    The two vertices must satisfy the Catan
    distance rule and therefore cannot be
    directly adjacent.
    """

    if vertex_b.id in vertex_a.neighbors:
        raise ValueError(
            "Opening settlements cannot be adjacent."
        )

    score_a = score_vertex(
        board,
        vertex_a,
    )

    score_b = score_vertex(
        board,
        vertex_b,
    )

    resources = (
        score_a.resources
        + score_b.resources
    )

    unique_resources = len(
        set(resources)
    )

    production_score = (
        score_a.production_score
        + score_b.production_score
    )

    composite_score = (
        production_score
        + DIVERSITY_WEIGHT * unique_resources
    )

    return OpeningPairScore(
        vertex_a=vertex_a.id,
        vertex_b=vertex_b.id,
        production_score=production_score,
        unique_resources=unique_resources,
        resources=resources,
        composite_score=composite_score,
    )


def rank_opening_pairs(
    board: Board,
) -> list[OpeningPairScore]:
    """
    Rank every legal pair of starting
    settlement vertices.
    """

    pairs: list[OpeningPairScore] = []

    for i, vertex_a in enumerate(
        board.vertices
    ):
        for vertex_b in board.vertices[
            i + 1:
        ]:
            if (
                vertex_b.id
                in vertex_a.neighbors
            ):
                continue

            pairs.append(
                score_opening_pair(
                    board,
                    vertex_a,
                    vertex_b,
                )
            )

    return sorted(
        pairs,
        key=lambda result: (
            -result.composite_score,
            -result.production_score,
            result.vertex_a,
            result.vertex_b,
        ),
    )



def resource_production(
    board: Board,
    vertex: Vertex,
) -> dict[Resource, int]:
    """
    Return the production weight contributed by
    each resource at a settlement vertex.
    """

    production = {
        Resource.WOOD: 0,
        Resource.BRICK: 0,
        Resource.SHEEP: 0,
        Resource.WHEAT: 0,
        Resource.ORE: 0,
    }

    for tile_id in vertex.adjacent_tiles:
        tile = board.tiles[
            tile_id
        ]

        if tile.resource == Resource.DESERT:
            continue

        if tile.number is None:
            continue

        production[
            tile.resource
        ] += production_weight(
            tile.number
        )

    return production


def strategic_vertex_score(
    board: Board,
    vertex: Vertex,
    resource_weights: dict[
        Resource,
        float,
    ],
    diversity_weight: float = 0.0,
) -> float:
    """
    Score a vertex according to a strategy's
    resource preferences.
    """

    production = resource_production(
        board,
        vertex,
    )

    weighted_production = sum(
        production[resource]
        * resource_weights.get(
            resource,
            0.0,
        )
        for resource in production
    )

    unique_resources = sum(
        amount > 0
        for amount in production.values()
    )

    return (
        weighted_production
        + diversity_weight
        * unique_resources
    )


def strategic_pair_score(
    board: Board,
    vertex_a: Vertex,
    vertex_b: Vertex,
    resource_weights: dict[
        Resource,
        float,
    ],
    diversity_weight: float = 0.0,
) -> float:
    """
    Score a pair of opening settlements according
    to a strategy's resource preferences.
    """

    production_a = resource_production(
        board,
        vertex_a,
    )

    production_b = resource_production(
        board,
        vertex_b,
    )

    combined = {
        resource: (
            production_a[resource]
            + production_b[resource]
        )
        for resource in production_a
    }

    weighted_production = sum(
        combined[resource]
        * resource_weights.get(
            resource,
            0.0,
        )
        for resource in combined
    )

    unique_resources = sum(
        amount > 0
        for amount in combined.values()
    )

    return (
        weighted_production
        + diversity_weight
        * unique_resources
    )


def port_synergy_score(
    board: Board,
    vertices: list[Vertex],
) -> float:
    """
    Score how well an opening's resource production
    works with the ports it controls.

    Specific 2:1 ports are rewarded in proportion
    to production of their matching resource.

    Generic 3:1 ports receive a smaller bonus based
    on total production.
    """

    from catanlab.ports import (
        ports_at_vertex,
    )

    combined = {
        Resource.WOOD: 0,
        Resource.BRICK: 0,
        Resource.SHEEP: 0,
        Resource.WHEAT: 0,
        Resource.ORE: 0,
    }

    for vertex in vertices:
        production = resource_production(
            board,
            vertex,
        )

        for resource, amount in production.items():
            combined[
                resource
            ] += amount

    ports = []

    for vertex in vertices:
        for port in ports_at_vertex(
            board,
            vertex.id,
        ):
            if port not in ports:
                ports.append(
                    port
                )

    if not ports:
        return 0.0

    total_production = sum(
        combined.values()
    )

    bonus = 0.0

    for port in ports:
        if port.resource is None:
            # Generic ports improve 4:1 -> 3:1.
            # Keep this useful but noticeably weaker
            # than strong 2:1 specialization.
            bonus += (
                total_production
                * 0.10
            )

        else:
            # Matching 2:1 ports become increasingly
            # valuable as production of that resource
            # increases.
            bonus += (
                combined[
                    port.resource
                ]
                * 0.50
            )

    return bonus
