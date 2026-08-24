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
