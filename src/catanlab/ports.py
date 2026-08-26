import math
import random

from catanlab.board import (
    Board,
    Port,
)
from catanlab.resources import Resource
from catanlab.simulation import PlayerState


STANDARD_PORT_RESOURCES = [
    None,
    None,
    None,
    None,
    Resource.WOOD,
    Resource.BRICK,
    Resource.SHEEP,
    Resource.WHEAT,
    Resource.ORE,
]


def coastal_edges(
    board: Board,
) -> list[tuple[int, int]]:
    """
    Return every edge on the outside coast.

    A coastal edge borders exactly one hex.
    An interior edge borders two.
    """

    coast = []

    for edge in board.edges:
        a = board.vertices[
            edge.vertex_a
        ]

        b = board.vertices[
            edge.vertex_b
        ]

        shared_tiles = set(
            a.adjacent_tiles
        ) & set(
            b.adjacent_tiles
        )

        if len(shared_tiles) == 1:
            coast.append(
                tuple(
                    sorted(
                        (
                            edge.vertex_a,
                            edge.vertex_b,
                        )
                    )
                )
            )

    return coast


def _edge_angle(
    board: Board,
    edge: tuple[int, int],
) -> float:
    """
    Return the polar angle of a coastal edge's
    midpoint around the center of the board.
    """

    a = board.vertices[
        edge[0]
    ].position

    b = board.vertices[
        edge[1]
    ].position

    x = (
        a[0] + b[0]
    ) / 2

    y = (
        a[1] + b[1]
    ) / 2

    return math.atan2(
        y,
        x,
    )


def standard_port_edges(
    board: Board,
) -> list[tuple[int, int]]:
    """
    Select nine well-spaced coastal edges for
    standard port positions.

    The standard board has 30 coastal edges.
    """

    coast = sorted(
        coastal_edges(board),
        key=lambda edge: _edge_angle(
            board,
            edge,
        ),
    )

    if len(coast) != 30:
        raise ValueError(
            "Standard Catan board should have "
            "30 coastal edges."
        )

    # Nine positions distributed around the
    # coastline with gaps between port edges.
    indices = [
        0,
        3,
        7,
        10,
        13,
        17,
        20,
        23,
        27,
    ]

    selected = [
        coast[index]
        for index in indices
    ]

    # A valid port layout should never place two
    # ports sharing the same coastal vertex.
    used_vertices: set[int] = set()

    for edge in selected:
        if (
            edge[0] in used_vertices
            or edge[1] in used_vertices
        ):
            raise ValueError(
                "Generated port edges overlap."
            )

        used_vertices.update(
            edge
        )

    return selected


def assign_standard_ports(
    board: Board,
    seed: int | None = None,
) -> None:
    """
    Assign the nine standard Catan ports.

    Port positions are fixed around the coastline,
    while port types are shuffled reproducibly.
    """

    rng = random.Random(
        seed
    )

    resources = (
        STANDARD_PORT_RESOURCES.copy()
    )

    rng.shuffle(
        resources
    )

    edges = standard_port_edges(
        board
    )

    board.ports = [
        Port(
            vertex_a=edge[0],
            vertex_b=edge[1],
            resource=resource,
        )
        for edge, resource
        in zip(
            edges,
            resources,
        )
    ]


def player_ports(
    board: Board,
    player: PlayerState,
) -> list[Port]:
    """
    Return every port controlled by a player.

    A settlement or city on either endpoint
    controls the port.
    """

    buildings = set(
        player.settlements
        + player.cities
    )

    return [
        port
        for port in board.ports
        if (
            port.vertex_a in buildings
            or port.vertex_b in buildings
        )
    ]


def best_maritime_ratio(
    board: Board,
    player: PlayerState,
    resource: Resource,
) -> int:
    """
    Return the best bank/port trade ratio available
    when giving the selected resource.

    Default: 4:1
    Generic port: 3:1
    Matching resource port: 2:1
    """

    ratio = 4

    for port in player_ports(
        board,
        player,
    ):
        if port.resource is None:
            ratio = min(
                ratio,
                3,
            )

        elif port.resource == resource:
            ratio = min(
                ratio,
                2,
            )

    return ratio


def maritime_trade(
    board: Board,
    player: PlayerState,
    inventory,
    give: Resource,
    receive: Resource,
) -> int:
    """
    Trade resources with the bank using the best
    maritime ratio available to the player.

    Returns the number of cards spent.
    """

    if give == Resource.DESERT:
        raise ValueError(
            "Cannot trade desert."
        )

    if receive == Resource.DESERT:
        raise ValueError(
            "Cannot receive desert."
        )

    if give == receive:
        raise ValueError(
            "Trade resources must be different."
        )

    ratio = best_maritime_ratio(
        board,
        player,
        give,
    )

    if inventory.count(
        give
    ) < ratio:
        raise ValueError(
            f"Need {ratio} {give.value} "
            f"for this maritime trade."
        )

    inventory.remove(
        give,
        ratio,
    )

    inventory.add(
        receive
    )

    return ratio


def ports_at_vertex(
    board: Board,
    vertex_id: int,
) -> list[Port]:
    """
    Return ports accessible from a settlement or
    city placed at the selected vertex.
    """

    return [
        port
        for port in board.ports
        if vertex_id in (
            port.vertex_a,
            port.vertex_b,
        )
    ]
